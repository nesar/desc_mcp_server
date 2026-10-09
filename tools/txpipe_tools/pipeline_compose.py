"""Compose and validate ceci pipeline/config YAML for TXPipe without importing it.

Presets start from TXPipe's own example pipelines (examples/mock_shear,
metadetect_source_only, metadetect | metacal | lensfit) loaded as the base,
pruned to the stages a target product needs (dependency closure over the
input/output tags in the stage index) and then overridden with the user's
spec (inputs, z-bin edges, nside, 2pt binning, threads, output/log dirs).

Validation mirrors what `ceci --dry-run` checks - every stage input tag is an
overall input or another stage's output, no duplicate outputs, no cycles -
and produces the per-stage command lines the way ceci's
``PipelineStage.generate_command`` + ``LocalSite.command`` would, so nothing
here needs a TXPipe environment.

The generated pipeline always uses ``site: {name: local, max_threads: N}``
and ``launcher: {name: mini}``: inside a facility job ceci still runs as the
local site on one node (the dispatch engine owns the allocation).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from ..common import require_clone
from .stage_index import (decorate_local, find_stage, get_index, load_pipeline_yaml,
                          output_filename, stage_command)

PRESETS = ("mock_shear", "source_only", "3x2pt_real", "3x2pt_fourier", "custom")
CATALOG_TYPES = ("metadetect", "metacal", "lensfit", "mock")

# base example directory per catalog type (full 3x2pt pipelines)
EXAMPLE_FOR_CATALOG = {"metadetect": "metadetect", "metacal": "metacal",
                       "lensfit": "lensfit", "mock": "mock_shear"}
SOURCE_SELECTOR = {"metadetect": "TXSourceSelectorMetadetect", "metacal": "TXSourceSelectorMetacal",
                   "lensfit": "TXSourceSelectorLensfit", "mock": "TXSourceSelectorSimple"}
SHEAR_CATALOG_FILE = {"metadetect": "data/example/inputs/metadetect_shear_catalog.hdf5",
                      "metacal": "data/example/inputs/shear_catalog.hdf5",
                      "lensfit": "data/example/inputs/lensfit_shear_catalog.hdf5"}

# target stage instances per preset (the closure adds everything they need)
# each target is a list of alternatives: the first one present in the base
# example is used (metacal plots with TXTwoPointPlotsTheory, metadetect with
# TXTwoPointPlots); a target absent from the example is appended as a plain
# stage entry when `add_if_missing` lists it.
PRESET_TARGETS = {
    "3x2pt_real": [["TXTwoPoint"], ["TXBlinding"], ["TXTwoPointTheoryReal"],
                   ["TXTwoPointPlots", "TXTwoPointPlotsTheory"],
                   ["TXPhotozPlotSource"], ["TXPhotozPlotLens"]],
    "3x2pt_fourier": [["TXTwoPointFourier"], ["TXTwoPointTheoryFourier"],
                      ["TXPhotozPlotSource"], ["TXPhotozPlotLens"]],
}
ADD_IF_MISSING = {"3x2pt_real": ["TXTwoPointTheoryReal"],
                  "3x2pt_fourier": ["TXTwoPointTheoryFourier"]}
COVARIANCE_STAGE = {"3x2pt_real": "TXRealGaussianCovariance", "3x2pt_fourier": "TXFourierGaussianCovariance"}

NONE_STRINGS = ("none", "null")


# --------------------------------------------------------------------------
# example loading
# --------------------------------------------------------------------------

def example_dir(name: str) -> Path:
    clone = require_clone("txpipe")
    d = clone / "examples" / name
    if not (d / "pipeline.yml").is_file():
        raise ValueError(f"TXPipe example '{name}' not found under {clone / 'examples'}")
    return d


def load_example(name: str) -> tuple[dict, dict]:
    """(pipeline dict, config dict) of examples/<name>/{pipeline,config}.yml."""
    d = example_dir(name)
    pipeline = load_pipeline_yaml(d / "pipeline.yml")
    cfg_path = d / "config.yml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    config = {k: v for k, v in (config or {}).items() if not str(k).startswith("_")}
    return pipeline, config


def _is_none(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in NONE_STRINGS)


def _entry(stage) -> dict:
    return {"name": stage} if isinstance(stage, str) else dict(stage)


def _aliased(entry: dict, tag: str) -> str:
    return (entry.get("aliases") or {}).get(tag, tag)


def closure(stages: list[dict], targets: list[str], overall_inputs: dict) -> tuple[list[dict], list[str]]:
    """Subset of `stages` (original order) needed to produce the target instances."""
    by_name = {e["name"]: e for e in stages}
    producers: dict[str, str] = {}
    for e in stages:
        rec = find_stage(e.get("classname", e["name"]))
        if rec:
            for tag, _ in rec["outputs"]:
                producers.setdefault(_aliased(e, tag), e["name"])
    needed: set[str] = set()
    warnings: list[str] = []
    todo = [t for t in targets if t in by_name]
    for t in targets:
        if t not in by_name:
            warnings.append(f"target stage {t} is not in the base example; skipped")
    while todo:
        name = todo.pop()
        if name in needed:
            continue
        needed.add(name)
        e = by_name[name]
        rec = find_stage(e.get("classname", name))
        if rec is None:
            warnings.append(f"{name}: unknown stage class, dependencies not traced")
            continue
        for tag, _ in rec["inputs"]:
            atag = _aliased(e, tag)
            if atag in overall_inputs:
                continue
            prod = producers.get(atag)
            if prod is None:
                warnings.append(f"{name}: input '{atag}' has no producer in the base example and is not an overall input")
            elif prod not in needed:
                todo.append(prod)
    return [e for e in stages if e["name"] in needed], warnings


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------

def _abs_clone(value: str, clone: Path) -> str:
    """Resolve an example-relative path (data/..., ./data/..., submodules/...) against the clone."""
    if not isinstance(value, str) or _is_none(value):
        return value
    v = value[2:] if value.startswith("./") else value
    if v.startswith(("data/", "submodules/", "examples/")) and not os.path.isabs(value):
        return str(clone / v)
    return value


def compose(spec: dict) -> dict:
    """Build {"pipeline": dict, "config": dict, "report": dict} from a high-level spec.

    spec keys: preset, catalog_type, stages (custom), inputs, source_zbin_edges,
    lens_zbin_edges, nside, min_sep, max_sep, nbins (arcmin, real space),
    ell_min, ell_max, n_ell (Fourier), threads, output_dir, log_dir, config_path,
    resume, blinding ('null'|'muir'), include_covariance, allow_mpi,
    config_overrides {stage: {option: value}}.
    """
    clone = require_clone("txpipe")
    preset = spec.get("preset", "3x2pt_real")
    catalog_type = spec.get("catalog_type", "metadetect")
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
    if catalog_type not in CATALOG_TYPES:
        raise ValueError(f"catalog_type must be one of {CATALOG_TYPES}, got {catalog_type!r}")
    report: dict = {"preset": preset, "catalog_type": catalog_type, "warnings": [], "notes": []}

    # --- base example -----------------------------------------------------
    if preset == "mock_shear":
        if catalog_type != "mock":
            report["notes"].append("mock_shear preset implies catalog_type='mock'")
            catalog_type = "mock"
        base = "mock_shear"
    elif preset == "source_only":
        if catalog_type == "mock":
            raise ValueError("source_only needs a metadetect/metacal/lensfit catalog; use preset='mock_shear' for the mock.")
        base = "metadetect_source_only"
    elif preset in ("3x2pt_real", "3x2pt_fourier"):
        if catalog_type == "mock":
            raise ValueError(f"{preset} needs a metadetect/metacal/lensfit catalog (the mock example has no 2pt stages).")
        base = EXAMPLE_FOR_CATALOG[catalog_type]
    else:  # custom
        base = EXAMPLE_FOR_CATALOG[catalog_type]
    report["base_example"] = f"examples/{base}"
    pipeline, config = load_example(base)
    stages = [_entry(s) for s in pipeline.get("stages", [])]
    overall_inputs: dict = dict(pipeline.get("inputs") or {})

    # --- source-only selector swap for other catalog types -------------------
    if preset == "source_only" and catalog_type != "metadetect":
        sel = SOURCE_SELECTOR[catalog_type]
        for e in stages:
            if e["name"] == "TXSourceSelectorMetadetect":
                e["name"] = sel
        _, other_cfg = load_example(EXAMPLE_FOR_CATALOG[catalog_type])
        if sel in other_cfg:
            config[sel] = other_cfg[sel]
        config.pop("TXSourceSelectorMetadetect", None)
        overall_inputs["shear_catalog"] = SHEAR_CATALOG_FILE[catalog_type]
        report["notes"].append(f"source selector swapped to {sel}; its config section copied from examples/{EXAMPLE_FOR_CATALOG[catalog_type]}")

    # --- stage selection -----------------------------------------------------
    blinding = spec.get("blinding", "null")
    if preset == "custom":
        user_stages = spec.get("stages") or []
        if not user_stages:
            raise ValueError("preset='custom' needs a non-empty 'stages' list.")
        by_name = {e["name"]: e for e in stages}
        chosen = []
        for s in user_stages:
            e = _entry(s)
            template = by_name.get(e["name"])
            if template:
                merged = dict(template)
                merged.update(e)
                e = merged
            chosen.append(e)
        stages = chosen
    elif preset in PRESET_TARGETS:
        if blinding == "null":
            # TXNullBlinding has the same tags as TXBlinding; swap before closure
            for e in stages:
                if e["name"] == "TXBlinding":
                    e["name"] = "TXNullBlinding"
        present = {e["name"] for e in stages}
        for extra in ADD_IF_MISSING[preset]:
            if extra not in present:
                stages.append({"name": extra})
                present.add(extra)
        if spec.get("include_covariance"):
            # the examples comment the covariance stages out; add them explicitly
            for extra in [COVARIANCE_STAGE[preset]] + (["TXTwoPointPlotsFourier"] if preset == "3x2pt_fourier" else []):
                if extra not in present:
                    stages.append({"name": extra, "threads_per_process": 2})
                    present.add(extra)
        targets = []
        for alternatives in PRESET_TARGETS[preset]:
            alternatives = ["TXNullBlinding" if a == "TXBlinding" and blinding == "null" else a for a in alternatives]
            pick = next((a for a in alternatives if a in present), None)
            if pick:
                targets.append(pick)
            else:
                report["warnings"].append(f"none of {alternatives} is in examples/{base}; skipped")
        if spec.get("include_covariance"):
            targets.append(COVARIANCE_STAGE[preset])
            if preset == "3x2pt_fourier":
                targets.append("TXTwoPointPlotsFourier")
        stages, warns = closure(stages, targets, overall_inputs)
        report["warnings"].extend(warns)
        report["targets"] = targets
    # mock_shear / source_only: the whole example
    if blinding == "null":
        for e in stages:
            if e["name"] == "TXBlinding":
                e["name"] = "TXNullBlinding"
        config.pop("TXBlinding", None)
    if blinding == "null" and preset in ("3x2pt_real", "source_only"):
        report["notes"].append("blinding='null' -> TXNullBlinding (simulations); use blinding='muir' for real data")

    # --- threads / MPI -----------------------------------------------------------
    threads = int(spec.get("threads", 2))
    mpi_stripped = []
    for e in stages:
        if e.get("nprocess", 1) > 1 and not spec.get("allow_mpi"):
            mpi_stripped.append(e["name"])
            e.pop("nprocess", None)
        if e.get("threads_per_process", 1) > threads:
            e["threads_per_process"] = threads
        e.pop("nodes", None)
    if mpi_stripped:
        report["notes"].append(f"nprocess>1 removed from {mpi_stripped} (needs mpirun + MPI h5py; pass allow_mpi=true to keep)")

    # --- inputs ----------------------------------------------------------------
    user_inputs = spec.get("inputs") or {}
    inputs_out: dict = {}
    for tag, val in overall_inputs.items():
        inputs_out[tag] = _abs_clone(val, clone) if isinstance(val, str) else val
    for tag, val in user_inputs.items():
        inputs_out[tag] = val
    # prune inputs no stage consumes (keeps the YAML honest after pruning)
    consumed: set[str] = set()
    produced: set[str] = set()
    unknown: list[str] = []
    for e in stages:
        rec = find_stage(e.get("classname", e["name"]))
        if rec is None:
            unknown.append(e["name"])
            continue
        consumed.update(_aliased(e, t) for t, _ in rec["inputs"])
        produced.update(_aliased(e, t) for t, _ in rec["outputs"])
    if not unknown:
        inputs_out = {t: v for t, v in inputs_out.items() if t in consumed or t in user_inputs}
    report["unknown_stages"] = unknown

    # --- config: prune to used sections, apply overrides -----------------------------
    instance_names = [e["name"] for e in stages]
    new_cfg: dict = {"global": dict(config.get("global") or {})}
    for name in instance_names:
        if name in config and isinstance(config[name], dict):
            new_cfg[name] = copy.deepcopy(config[name])
            new_cfg[name].pop("aliases", None)

    def stages_with_option(opt: str, also: str | None = None) -> list[str]:
        out = []
        for e in stages:
            rec = find_stage(e.get("classname", e["name"]))
            if rec and opt in rec["config_options"] and (also is None or also in rec["config_options"]):
                out.append(e["name"])
        return out

    def set_opt(names: list[str], key: str, value) -> None:
        for n in names:
            new_cfg.setdefault(n, {})[key] = value

    if spec.get("source_zbin_edges"):
        set_opt(stages_with_option("source_zbin_edges"), "source_zbin_edges", [float(x) for x in spec["source_zbin_edges"]])
    if spec.get("lens_zbin_edges"):
        set_opt(stages_with_option("lens_zbin_edges"), "lens_zbin_edges", [float(x) for x in spec["lens_zbin_edges"]])
    if spec.get("nside"):
        nside = int(spec["nside"])
        new_cfg["global"]["nside"] = nside
        for n, sec in new_cfg.items():
            if n != "global" and isinstance(sec, dict) and "nside" in sec:
                sec["nside"] = nside
        report["notes"].append(f"nside={nside} applied to global and every stage section that set one (the example used 64/128/256 tiers)")
    sep_stages = stages_with_option("min_sep", also="max_sep")
    for key in ("min_sep", "max_sep", "nbins"):
        if spec.get(key) is not None:
            set_opt(sep_stages, key, spec[key])
    if sep_stages and any(spec.get(k) is not None for k in ("min_sep", "max_sep", "nbins")):
        set_opt([n for n in sep_stages if "sep_units" in (find_stage(n) or {}).get("config_options", {})], "sep_units", "arcmin")
    ell_stages = stages_with_option("ell_min", also="ell_max")
    for key in ("ell_min", "ell_max", "n_ell"):
        if spec.get(key) is not None:
            set_opt(ell_stages, key, int(spec[key]))
            for n in ell_stages:
                new_cfg[n].pop("bandwidth", None)  # example-config leftover, not a TXTwoPointFourier option
    for stage_name, opts in (spec.get("config_overrides") or {}).items():
        new_cfg.setdefault(stage_name, {}).update(opts)

    # paths inside the config (example-relative) -> absolute clone paths; caches -> output_dir
    out_dir = str(Path(spec.get("output_dir") or "txpipe_outputs").expanduser())
    log_dir = str(Path(spec.get("log_dir") or (Path(out_dir).parent / (Path(out_dir).name + "_logs"))).expanduser())
    for n, sec in new_cfg.items():
        if not isinstance(sec, dict):
            continue
        for k, v in list(sec.items()):
            if isinstance(v, str):
                sec[k] = _abs_clone(v, clone)
        if n != "global":
            rec = find_stage(n) or (find_stage(next((e.get("classname", "") for e in stages if e["name"] == n), "")) or {})
            for cache_key in ("cache_dir", "patch_dir"):
                if cache_key in (rec.get("config_options") or {}):
                    sec[cache_key] = str(Path(out_dir) / "cache" / n)

    # --- pipeline dict ---------------------------------------------------------
    modules = str(pipeline.get("modules", "txpipe")).split()
    if "txpipe.extensions" not in modules and any(
            (find_stage(e.get("classname", e["name"])) or {}).get("in_extensions") for e in stages):
        modules.append("txpipe.extensions")
    python_paths = [_abs_clone(p, clone) for p in (pipeline.get("python_paths") or [])]
    config_path = spec.get("config_path") or "config.yml"
    new_pipeline = {
        "stages": stages,
        "modules": " ".join(modules),
        "python_paths": python_paths,
        "output_dir": out_dir,
        "launcher": {"name": "mini", "interval": 1.0},
        "site": {"name": "local", "max_threads": threads},
        "config": config_path,
        "inputs": inputs_out,
        "resume": bool(spec.get("resume", True)),
        "log_dir": log_dir,
        "pipeline_log": str(Path(log_dir) / "pipeline_log.txt"),
    }
    report.update({"n_stages": len(stages), "stages": [e["name"] for e in stages],
                   "output_dir": out_dir, "log_dir": log_dir, "threads": threads,
                   "modules": modules, "clone_commit": get_index()["metadata"]["clone_commit"]})
    return {"pipeline": new_pipeline, "config": new_cfg, "report": report}


# --------------------------------------------------------------------------
# validation (index-based mirror of ceci --dry-run)
# --------------------------------------------------------------------------

def validate(pipeline: dict, base_dir: Path | None = None) -> dict:
    """DAG + input-file + command-line report for a pipeline dict."""
    idx = get_index()
    stages = [_entry(s) for s in pipeline.get("stages", [])]
    overall_raw = pipeline.get("inputs") or {}
    output_dir = pipeline.get("output_dir") or "."
    config_path = pipeline.get("config") or "config.yml"
    base_dir = Path(base_dir) if base_dir else Path.cwd()

    def resolve(p: str) -> Path:
        return Path(p).expanduser() if os.path.isabs(p) else base_dir / p

    report: dict = {"ok": True, "n_stages": len(stages), "unknown_stages": [], "duplicate_instances": [],
                    "unresolved_inputs": [], "duplicate_outputs": [], "cycle": False,
                    "overall_inputs": {}, "missing_input_files": [], "none_inputs": [],
                    "stages": [], "order": [], "commands": [], "warnings": []}
    # overall inputs
    paths: dict[str, str] = {}
    for tag, val in overall_raw.items():
        if _is_none(val):
            paths[tag] = "None"
            report["none_inputs"].append(tag)
            report["overall_inputs"][tag] = {"path": None, "exists": None}
        elif isinstance(val, dict):
            paths[tag] = f"registry:{val}"
            report["overall_inputs"][tag] = {"path": val, "exists": None, "note": "data-registry lookup (facility only)"}
        else:
            paths[tag] = str(val)
            exists = resolve(str(val)).exists()
            report["overall_inputs"][tag] = {"path": str(val), "exists": exists}
            if not exists:
                report["missing_input_files"].append({"tag": tag, "path": str(val)})
    # stages, producers
    seen: set[str] = set()
    producers: dict[str, tuple[str, str, str]] = {}  # aliased tag -> (instance, file type, original tag)
    resolved: list[tuple[dict, dict]] = []
    for e in stages:
        name = e["name"]
        if name in seen:
            report["duplicate_instances"].append(name)
        seen.add(name)
        rec = find_stage(e.get("classname", name))
        if rec is None:
            report["unknown_stages"].append(e.get("classname", name))
            report["stages"].append({"instance": name, "classname": e.get("classname", name), "known": False})
            continue
        resolved.append((e, rec))
        for tag, ftype in rec["outputs"]:
            atag = _aliased(e, tag)
            if atag in producers:
                report["duplicate_outputs"].append({"tag": atag, "stages": [producers[atag][0], name]})
            if atag in paths:
                report["warnings"].append(f"tag {atag} is produced by {name} but also listed as an overall input (ceci refuses this)")
            producers[atag] = (name, ftype, tag)
    # edges
    deps: dict[str, set[str]] = {}
    for e, rec in resolved:
        name = e["name"]
        deps[name] = set()
        inputs_info = {}
        for tag, ftype in rec["inputs"]:
            atag = _aliased(e, tag)
            if atag in paths:
                inputs_info[atag] = "overall input"
            elif atag in producers:
                inputs_info[atag] = f"from {producers[atag][0]}"
                deps[name].add(producers[atag][0])
            else:
                inputs_info[atag] = "UNRESOLVED"
                report["unresolved_inputs"].append({"stage": name, "tag": atag, "file_type": ftype})
        report["stages"].append({
            "instance": name, "classname": rec["name"], "known": True, "group": rec["group"],
            "inputs": inputs_info, "outputs": [_aliased(e, t) for t, _ in rec["outputs"]],
            "threads_per_process": e.get("threads_per_process", 1), "nprocess": e.get("nprocess", 1),
        })
    # topological order (Kahn), ceci-style
    order: list[str] = []
    remaining = {n: set(d) for n, d in deps.items()}
    while remaining:
        ready = sorted(n for n, d in remaining.items() if not d)
        if not ready:
            report["cycle"] = True
            break
        for n in ready:
            order.append(n)
            del remaining[n]
        for d in remaining.values():
            d.difference_update(ready)
    report["order"] = order
    # commands
    threads_default = (pipeline.get("site") or {}).get("max_threads", 1)
    python_paths = pipeline.get("python_paths") or []
    by_name = {e["name"]: (e, rec) for e, rec in resolved}
    for name in order:
        e, rec = by_name[name]
        ins, outs = {}, {}
        for tag, ftype in rec["inputs"]:
            atag = _aliased(e, tag)
            if atag in paths:
                ins[atag] = paths[atag]
            elif atag in producers:
                ins[atag] = str(Path(output_dir) / output_filename(atag, producers[atag][1], idx["suffixes"]))
            else:
                ins[atag] = "MISSING"
        for tag, ftype in rec["outputs"]:
            atag = _aliased(e, tag)
            outs[atag] = str(Path(output_dir) / output_filename(atag, ftype, idx["suffixes"]))
        core = stage_command(rec, ins, config_path, outs, e.get("aliases"), name,
                             module=rec["module"].split(".")[0] if not rec.get("external") else "ceci")
        if rec.get("external"):
            core = core.replace(f"python3 -m ceci {rec['name']}", f"python3 -m ceci {rec['module']}.{rec['name']}")
        tpp = min(int(e.get("threads_per_process", 1)), int(threads_default or 1))
        report["commands"].append({"stage": name, "command": decorate_local(core, tpp, int(e.get("nprocess", 1)), python_paths)})
    report["ok"] = not (report["unknown_stages"] or report["unresolved_inputs"] or report["duplicate_outputs"]
                        or report["cycle"] or report["duplicate_instances"])
    report["site"] = pipeline.get("site")
    report["launcher"] = pipeline.get("launcher")
    report["output_dir"] = output_dir
    report["log_dir"] = pipeline.get("log_dir")
    report["config"] = config_path
    site_name = (pipeline.get("site") or {}).get("name") if isinstance(pipeline.get("site"), dict) else None
    if site_name and site_name != "local":
        report["warnings"].append(f"site.name={site_name}: this server runs pipelines as site local (one node); override with site.name=local")
    return report


def write_yaml(path: Path, data: dict, header: str = "") -> None:
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120)
    path.write_text((f"# {header}\n" if header else "") + text, encoding="utf-8")
