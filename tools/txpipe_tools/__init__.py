"""TXPipe tools: compose ceci pipelines from TXPipe's source metadata and run
them OUTSIDE this server (a local TXPipe env named by DESC_TXPIPE_ENV, or a
facility job through the dispatch engine under a client-supplied env_setup).

TXPipe is never imported here (notes/DESIGN.md D1/D2): its firecrown 1.7 pin
and compiled MPI/NaMaster/TreeCorr stack cannot coexist with firecrown 1.16.
Stage metadata comes from an AST index of the read-only clone
(stage_index.py); pipelines are derived from TXPipe's own examples
(pipeline_compose.py); execution is `ceci pipeline.yml` as a subprocess in
the user's TXPipe environment (tools/inner/txpipe_run.py).
"""

from __future__ import annotations

import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, validate_call

from ..common import (ArtifactResult, clone_dir, param_slug, require_clone, resolve_outdir,
                      write_json)
from . import pipeline_compose as pc
from .stage_index import (GROUPS, find_stage, get_index, load_pipeline_yaml, output_filename,
                          stage_command)

__all__ = ["txpipe_list_stages", "txpipe_describe_stage", "txpipe_list_examples",
           "txpipe_generate_pipeline", "txpipe_validate_pipeline", "txpipe_run_pipeline",
           "txpipe_run_status", "txpipe_fetch_example_data"]

EXAMPLE_DATA = {
    "example": {"url": "https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz",
                "version": "v10 (same bytes as example-v10.tar.gz; TXPipe CI pins EXAMPLE_DATA_FILE_VERSION=v10)",
                "expected_mb": 347, "extracts_to": "data/example/inputs",
                "used_by": "metadetect, metadetect_source_only, metacal, lensfit, mock_shear, redmagic examples"},
    "cmb": {"url": "https://portal.nersc.gov/cfs/lsst/txpipe/data/example-cmb-lensing.tar.gz",
            "version": "current", "expected_mb": 189, "extracts_to": "data/example-cmb-lensing",
            "used_by": "examples/cmb/quaia.yml (Quaia x Planck lensing)"},
}

CONVENTIONS = {
    "execution": "compose + dispatch: TXPipe runs as `ceci pipeline.yml` in a SEPARATE process/env; "
                 "this server only reads TXPipe's source (AST) and its example YAMLs",
    "site": "always site.name=local + launcher mini (one node); inside a facility job the dispatch engine "
            "owns the allocation, ceci never calls srun/sbatch",
    "environments": "local: DESC_TXPIPE_ENV = shell snippet activating a TXPipe env (e.g. "
                    "'source ~/TXPipe/conda/bin/activate; cd ~/TXPipe'); remote: env_setup argument per call "
                    "(the user's TXPipe env + checkout on the facility); neither is ever defaulted by the server",
    "paths": "example inputs are resolved to absolute paths under the TXPipe clone; facility inputs are "
             "the paths the USER supplies in `inputs` (the server never assumes where data lives)",
    "units": "min_sep/max_sep in arcmin (sep_units=arcmin), ell integer multipoles, nside healpix, z-bin edges in z",
    "outputs": "{output_dir}/{tag}.{suffix}; stages write inprogress_{tag}.{suffix} and rename on success "
               "(that is how resume=true detects completed stages)",
    "products": "twopoint_data_real(.raw).sacc, twopoint_gamma_x.sacc, twopoint_data_fourier.sacc, "
                "twopoint_theory_*.sacc, summary_statistics_*.sacc (data + covariance, only with include_covariance), "
                "PNG plots, tracer_metadata_yml.yml; HDF5 catalogs/maps stay where the pipeline ran",
    "next": "inspect the sacc with sacc_inspect, then prepare_sacc_for_firecrown / attach a covariance "
            "(skill txpipe-sacc-to-likelihood)",
}

CAVEATS = [
    "TXPipe is NOT importable in this server (by design); txpipe_run_pipeline needs DESC_TXPIPE_ENV (local) or env_setup (facility).",
    "The 1 deg^2 example data (347 MB, txpipe_fetch_example_data) is 'too small to check numerical results' - it tests that the pipeline runs (minutes); 20 deg^2 cosmoDC2 takes hours and 13.6 GB.",
    "twopoint_data_*.sacc files have NO covariance; set include_covariance=true (slow: TJPCov/NaMaster) or attach a Gaussian covariance with the sacc tools before any likelihood.",
    "Changing config does not invalidate existing outputs: delete downstream files or run with resume=false.",
    "nprocess>1 (MPI) is stripped from generated pipelines unless allow_mpi=true; it needs mpirun + MPI-enabled h5py.",
    "TXConvergenceMaps and HOSFSB need TXPipe's git submodules (WLMassMap, pyfsb); the metadetect example includes them, the presets here do not.",
    "RAIL stages (FlowCreator, GridSelection, NZDirInformer, BPZlite*) are indexed from a built-in signature table, not from the clone; RAIL must be in the TXPipe env.",
    "ceci's real --dry-run only runs if DESC_TXPIPE_ENV is set; the index-based validation is the always-on check.",
]

DISPATCH_KERNELS = {
    "txpipe_run": {
        "function": "envkernel.run_in_env",
        "inner": "txpipe_run",
        "params": {"pipeline": "<pipeline.yml as dict>", "config": "<config.yml as dict>",
                   "max_threads": 32, "resume": True, "server_clone_dir": "<server TXPipe clone path>",
                   "stage_outputs": {"<instance>": ["<tag>.<suffix>"]}, "timeout_s": 3000,
                   "txpipe_dir": "<optional existing TXPipe checkout on the facility>"},
        "env_setup_required": True,
        "suitable_envs": ["user TXPipe env"],
        "duration_hint_s": 3600,
        "returns": "per-stage status manifest (complete/inprogress/failed), produced sacc/PNG/yml files with "
                   "sizes, small sacc files base64-inline (<2 MB) and yml/txt inline, log tails; HDF5 outputs "
                   "stay in the job directory (paths reported)",
        "note": "if `txpipe` is not importable under env_setup the inner script clones "
                "https://github.com/LSSTDESC/TXPipe (depth 1) into the job dir; inputs must be facility paths",
    },
}


def _base_dir_for(pipeline_path: Path) -> Path:
    clone = clone_dir("txpipe")
    if clone and str(pipeline_path).startswith(str(clone.resolve())):
        return clone
    return pipeline_path.parent


def _load_config_for(pipeline: dict, pipeline_path: Path) -> tuple[dict, Path | None]:
    cfg = pipeline.get("config")
    if not cfg:
        return {}, None
    p = Path(cfg).expanduser()
    if not p.is_absolute():
        p = _base_dir_for(pipeline_path) / p
    if not p.is_file():
        return {}, p
    import yaml
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {k: v for k, v in data.items() if not str(k).startswith("_")}, p


def _expected_outputs(pipeline: dict) -> dict[str, list[str]]:
    suffixes = get_index()["suffixes"]
    out: dict[str, list[str]] = {}
    for e in pipeline.get("stages", []):
        e = pc._entry(e)
        rec = find_stage(e.get("classname", e["name"]))
        if rec:
            out[e["name"]] = [output_filename(pc._aliased(e, t), ft, suffixes) for t, ft in rec["outputs"]]
    return out


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

@validate_call
def txpipe_list_stages(
    group: Annotated[str, Field(description="Filter by purpose group: ingest, photo-z, selection, calibration, lss-weights, maps, two-point, covariance, blinding, null-tests, plots, extensions; '' = all.")] = "",
    include_extensions: Annotated[bool, Field(description="Include txpipe.extensions stages (cluster counts, CMB lensing, IA self-calibration, HOS); they need 'txpipe.extensions' in the pipeline's modules.")] = True,
    include_abstract: Annotated[bool, Field(description="Include base classes that are not meant to run (e.g. the TXPipe PipelineStage base).")] = False,
) -> ArtifactResult:
    """List every TXPipe pipeline stage (AST index of the clone; no TXPipe import) with its purpose group.

    Each entry: YAML/CLI name, class, module, one-line doc, group, number of
    inputs/outputs, MPI (`parallel`) / dask flags, and which example
    pipelines use it. Groups follow the TXPipe docs: ingest (mocks, survey
    ingestion), photo-z (stacks, RAIL glue), selection (source/lens
    tomography), calibration (shear calibration, catalog splitting),
    lss-weights, maps (maps, masks, randoms, jackknife, metadata), two-point
    (TreeCorr real space, NaMaster Fourier, theory), covariance, blinding,
    null-tests (gamma_t, Rowe/tau, PSF), plots, extensions. Call
    txpipe_describe_stage for inputs/outputs/config of one stage and
    txpipe_generate_pipeline to compose a runnable pipeline.
    """
    idx = get_index()
    if group and group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; choose from {GROUPS}")
    rows = []
    for name, r in sorted(idx["stages"].items(), key=lambda kv: (GROUPS.index(kv[1]["group"]), kv[0])):
        if group and r["group"] != group:
            continue
        if not include_extensions and r["in_extensions"]:
            continue
        if not include_abstract and r["abstract"]:
            continue
        rows.append({"name": name, "class": r["class"], "module": r["module"], "group": r["group"],
                     "doc": r["doc"], "n_inputs": len(r["inputs"]), "n_outputs": len(r["outputs"]),
                     "n_config_options": len(r["config_options"]), "parallel": r["parallel"],
                     "dask_parallel": r["dask_parallel"], "extension": r["in_extensions"],
                     "n_examples_using": len(r["used_in_examples"])})
    groups = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    meta = idx["metadata"]
    return ArtifactResult(
        status="success", files=[],
        message=f"{len(rows)} TXPipe stages" + (f" in group '{group}'" if group else "") +
                f" (clone {meta['clone_commit'][:9]}, {meta['n_examples']} example pipelines indexed).",
        metadata={"stages": rows, "groups": groups, "clone_commit": meta["clone_commit"],
                  "clone_dir": meta["clone_dir"], "index_note": meta["note"]},
    )


@validate_call
def txpipe_describe_stage(
    name: Annotated[str, Field(min_length=1, description="Stage name as used in pipeline YAML (e.g. 'TXTwoPoint'); class names are accepted too (e.g. 'TXSourceSelectorBase').")],
) -> ArtifactResult:
    """Describe one TXPipe stage: inputs/outputs (tag + file type), config options with dtype/default/required/help, CLI template, and the examples that use it.

    `config_options` entries come from ceci StageParameter declarations or
    bare values (bare type = required); inherited options are resolved
    through the class hierarchy. `cli_template` is the single-stage command
    ceci would generate (`python -m txpipe <Stage> --<input>=... --config=config.yml --<output>=...`).
    For stage output products use sacc_inspect on the .sacc files it writes.
    """
    rec = find_stage(name)
    if rec is None:
        idx = get_index()
        close = [n for n in idx["stages"] if name.lower() in n.lower()][:10]
        raise ValueError(f"No TXPipe stage named {name!r}." + (f" Close matches: {close}" if close else "") +
                         " Use txpipe_list_stages.")
    ins = {t: "<path>" for t, _ in rec["inputs"]}
    outs = {t: f"<out>/{output_filename(t, ft)}" for t, ft in rec["outputs"]}
    cli = stage_command(rec, ins, "config.yml", outs)
    yaml_snippet = {"stages": [{"name": rec["name"]}],
                    "inputs": {t: "<path>" for t, _ in rec["inputs"]}}
    required = [k for k, v in rec["config_options"].items() if v["required"]]
    return ArtifactResult(
        status="success", files=[],
        message=f"{rec['name']} ({rec['group']}): {rec['doc'] or 'no docstring'}. "
                f"{len(rec['inputs'])} inputs, {len(rec['outputs'])} outputs, "
                f"{len(rec['config_options'])} config options ({len(required)} required).",
        metadata={**{k: v for k, v in rec.items() if k not in ("declares_name",)},
                  "required_options": required, "cli_template": cli, "yaml_snippet": yaml_snippet,
                  "output_filenames": {t: output_filename(t, ft) for t, ft in rec["outputs"]},
                  "source": f"{get_index()['metadata']['clone_dir']}/{rec['file']}" if not rec.get("external") else rec["module"]},
    )


@validate_call
def txpipe_list_examples() -> ArtifactResult:
    """List TXPipe's example pipelines (examples/**/*.yml with a stages list): stage count, site, inputs and whether they exist locally, output dir.

    The laptop-sized ones are metadetect (61 stages, the CI test),
    metadetect_source_only (28), metacal (52), lensfit (43), mock_shear (6,
    seconds) - all on the 1 deg^2 example data (txpipe_fetch_example_data).
    The survey/DC2 pipelines reference NERSC paths or the DESC data
    registry and are listed for reference only. Presets in
    txpipe_generate_pipeline are derived from these files.
    """
    idx = get_index()
    rows = []
    for ex in idx["examples"]:
        rows.append({k: v for k, v in ex.items() if k != "stages"} | {"stage_names": [s["name"] for s in ex.get("stages", [])]})
    local = [r["path"] for r in rows if r.get("site") == "local"]
    return ArtifactResult(
        status="success", files=[],
        message=f"{len(rows)} example pipelines in {idx['metadata']['clone_dir']}/examples; "
                f"{len(local)} declare site local. Example data present locally: "
                f"{(require_clone('txpipe') / 'data/example/inputs').is_dir()}.",
        metadata={"examples": rows, "example_data": EXAMPLE_DATA, "clone_commit": idx["metadata"]["clone_commit"]},
    )


# --------------------------------------------------------------------------
# compose / validate
# --------------------------------------------------------------------------

@validate_call
def txpipe_generate_pipeline(
    output_dir: Annotated[str, Field(min_length=1, description="Directory for the generated pipeline.yml, config.yml and dependency_report.json (a subdirectory per preset+spec is created).")],
    preset: Annotated[Literal["mock_shear", "source_only", "3x2pt_real", "3x2pt_fourier", "custom"], Field(description="mock_shear: 6-stage smoke test (seconds); source_only: shear-only real-space xi+/- (28 stages); 3x2pt_real: TreeCorr xi+/-, gamma_t, w(theta) with photo-z + maps + randoms (27 stages); 3x2pt_fourier: NaMaster C_ell; custom: explicit `stages`.")] = "3x2pt_real",
    catalog_type: Annotated[Literal["metadetect", "metacal", "lensfit", "mock"], Field(description="Shear catalog flavour -> source selector and base example (metadetect is the CI-tested path).")] = "metadetect",
    stages: Annotated[list[str | dict], Field(description="custom preset only: stage names or {name, classname, aliases, threads_per_process} entries, in any order (ceci orders by tags).")] = [],
    inputs: Annotated[dict[str, str], Field(description="Overall inputs tag -> path, overriding the example's (e.g. shear_catalog, photometry_catalog, star_catalog, exposures, flow, fiducial_cosmology). Facility paths for remote runs.")] = {},
    source_zbin_edges: Annotated[list[float], Field(description="Source tomographic bin edges in z (example: [0.5, 0.7, 0.9, 1.1, 2.0]).")] = [],
    lens_zbin_edges: Annotated[list[float], Field(description="Lens bin edges in z (example: [0.0, 0.2, 0.4]).")] = [],
    nside: Annotated[int | None, Field(ge=8, le=8192, description="Healpix nside for all map stages (example tiers 64/128/256; a single value is applied everywhere).")] = None,
    min_sep: Annotated[float | None, Field(gt=0, description="Real-space minimum separation [arcmin] (example 2.5; TXPipe default 0.5).")] = None,
    max_sep: Annotated[float | None, Field(gt=0, description="Real-space maximum separation [arcmin] (example 60; default 300).")] = None,
    nbins: Annotated[int | None, Field(ge=1, le=100, description="Number of log-spaced separation bins (example 10).")] = None,
    ell_min: Annotated[int | None, Field(ge=2, description="Fourier: minimum ell (example 30, default 100).")] = None,
    ell_max: Annotated[int | None, Field(ge=3, description="Fourier: maximum ell (example 100, default 1500; <= 3 nside).")] = None,
    n_ell: Annotated[int | None, Field(ge=1, le=200, description="Fourier: number of ell bins (default 20, log spacing).")] = None,
    threads: Annotated[int, Field(ge=1, le=256, description="site.max_threads for ceci's local site; per-stage threads_per_process are capped to it.")] = 2,
    pipeline_output_dir: Annotated[str, Field(description="Where the pipeline writes its products (default <run dir>/outputs). On a facility the inner script re-roots this to the job directory.")] = "",
    log_dir: Annotated[str, Field(description="Per-stage logs <log_dir>/<Stage>.out (default <run dir>/logs).")] = "",
    resume: Annotated[bool, Field(description="ceci resume: skip stages whose final outputs already exist.")] = True,
    blinding: Annotated[Literal["null", "muir"], Field(description="'null' = TXNullBlinding (copy; for simulations), 'muir' = TXBlinding (Muir et al. parameter-shift blinding, deletes the unblinded sacc).")] = "null",
    include_covariance: Annotated[bool, Field(description="Add the Gaussian covariance stage (TXRealGaussianCovariance / TXFourierGaussianCovariance -> summary_statistics_*.sacc). Slow; the examples leave it out.")] = False,
    allow_mpi: Annotated[bool, Field(description="Keep nprocess>1 entries from the example (needs mpirun + MPI h5py).")] = False,
    config_overrides: Annotated[dict[str, dict], Field(description="Per-stage config overrides {StageInstance: {option: value}} applied last (see txpipe_describe_stage for option names).")] = {},
) -> ArtifactResult:
    """Compose a runnable TXPipe pipeline.yml + config.yml from a preset (derived from TXPipe's examples) and your overrides, with a dependency report.

    The base example (mock_shear | metadetect_source_only | metadetect/metacal/
    lensfit) is loaded, pruned to the stages the preset's products need
    (dependency closure over input/output tags), then overridden. Always
    site local + mini launcher. Example-relative paths become absolute
    paths under the TXPipe clone, so the missing-input report tells you
    which example files you still need (txpipe_fetch_example_data) or
    which `inputs` to point at your own catalogs. Then call
    txpipe_validate_pipeline and txpipe_run_pipeline. Returns pipeline.yml,
    config.yml, dependency_report.json; metadata holds the stage list,
    the validation summary and the dry-run command for the main 2pt stage.
    """
    if preset == "mock_shear":
        catalog_type = "mock"
    spec = {"preset": preset, "catalog_type": catalog_type, "stages": stages, "inputs": inputs,
            "source_zbin_edges": source_zbin_edges, "lens_zbin_edges": lens_zbin_edges, "nside": nside,
            "min_sep": min_sep, "max_sep": max_sep, "nbins": nbins, "ell_min": ell_min, "ell_max": ell_max,
            "n_ell": n_ell, "threads": threads, "resume": resume, "blinding": blinding,
            "include_covariance": include_covariance, "allow_mpi": allow_mpi,
            "config_overrides": config_overrides}
    outdir = resolve_outdir(output_dir)
    slug = param_slug({k: str(v) for k, v in spec.items()})
    run_dir = outdir / f"txpipe_{preset}_{catalog_type}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    spec["output_dir"] = pipeline_output_dir or str(run_dir / "outputs")
    spec["log_dir"] = log_dir or str(run_dir / "logs")
    spec["config_path"] = str(run_dir / "config.yml")
    composed = pc.compose(spec)
    pipeline, config, report = composed["pipeline"], composed["config"], composed["report"]
    validation = pc.validate(pipeline, base_dir=run_dir)
    header = (f"generated by desc-mcp-server txpipe_generate_pipeline preset={preset} catalog_type={catalog_type} "
              f"base={report['base_example']} TXPipe@{report['clone_commit'][:9]}")
    pc.write_yaml(run_dir / "pipeline.yml", pipeline, header)
    pc.write_yaml(run_dir / "config.yml", config, header)
    dep = {"spec": spec, "compose": report, "validation": validation,
           "expected_outputs": _expected_outputs(pipeline)}
    write_json(run_dir / "dependency_report.json", dep)
    main_stage = next((c for c in validation["commands"] if c["stage"] in
                       ("TXTwoPoint", "TXTwoPointFourier", "TXShearCalibration")), None)
    missing = validation["missing_input_files"]
    msg = (f"{preset}/{catalog_type}: {report['n_stages']} stages from {report['base_example']}; "
           f"DAG {'OK' if validation['ok'] else 'has problems'}; "
           f"{len(missing)} input file(s) missing locally" +
           (f" ({', '.join(m['tag'] for m in missing[:5])}{'...' if len(missing) > 5 else ''})" if missing else "") +
           ". Next: txpipe_validate_pipeline, then txpipe_run_pipeline.")
    return ArtifactResult(
        status="success",
        files=[str(run_dir / "pipeline.yml"), str(run_dir / "config.yml"), str(run_dir / "dependency_report.json")],
        message=msg,
        metadata={"run_dir": str(run_dir), "pipeline_yml": str(run_dir / "pipeline.yml"),
                  "stages": report["stages"], "order": validation["order"], "compose": report,
                  "validation_ok": validation["ok"], "unresolved_inputs": validation["unresolved_inputs"],
                  "unknown_stages": validation["unknown_stages"], "missing_input_files": missing,
                  "overall_inputs": validation["overall_inputs"], "main_stage_command": main_stage,
                  "pipeline_output_dir": spec["output_dir"], "log_dir": spec["log_dir"],
                  "expected_outputs": dep["expected_outputs"]},
    )


@validate_call
def txpipe_validate_pipeline(
    pipeline_yml: Annotated[str, Field(min_length=1, description="Path to a ceci pipeline YAML (generated here or one of TXPipe's examples).")],
    run_ceci_dry_run: Annotated[bool, Field(description="Also run the real `ceci --dry-run` if DESC_TXPIPE_ENV is set (needs the TXPipe env; ~10-60 s).")] = True,
) -> ArtifactResult:
    """Validate a pipeline against the stage index (DAG, inputs, unknown stages) and list the per-stage dry-run commands; optionally run ceci's own --dry-run.

    Index-based checks (always, milliseconds, no TXPipe needed): every stage
    class is known; every input tag is an overall input or another stage's
    output; no duplicate outputs/instances; no cycles; which overall input
    files are missing locally (facility paths show as missing - that is
    expected for remote runs); the topological order; one command line per
    stage mirroring ceci's generate_command + local-site decoration. If
    DESC_TXPIPE_ENV is set, `ceci --dry-run` runs in that env as the
    authoritative check and its output is included.
    """
    path = Path(pipeline_yml).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"pipeline file not found: {path}")
    pipeline = load_pipeline_yaml(path)
    if not pipeline.get("stages"):
        raise ValueError(f"{path} has no 'stages' list - not a ceci pipeline file.")
    base_dir = _base_dir_for(path)
    report = pc.validate(pipeline, base_dir=base_dir)
    config, cfg_path = _load_config_for(pipeline, path)
    report["config_file"] = str(cfg_path) if cfg_path else None
    report["config_file_exists"] = bool(cfg_path and cfg_path.is_file())
    unknown_cfg = [k for k in config if k != "global" and k not in {pc._entry(s)["name"] for s in pipeline["stages"]}]
    if unknown_cfg:
        report["warnings"].append(f"config sections for stages not in the pipeline (ignored by ceci): {unknown_cfg[:10]}")
    ceci = None
    env = os.environ.get("DESC_TXPIPE_ENV")
    if run_ceci_dry_run and env:
        import subprocess
        cmd = f"{env.strip().rstrip(';')}; ceci --dry-run {path}"
        try:
            proc = subprocess.run(["bash", "-lc", cmd], cwd=str(base_dir), capture_output=True, text=True, timeout=300)
            tail = lambda s: "\n".join((s or "").splitlines()[-80:])
            ceci = {"returncode": proc.returncode, "stdout_tail": tail(proc.stdout), "stderr_tail": tail(proc.stderr),
                    "ok": proc.returncode == 0}
        except Exception as exc:  # noqa: BLE001
            ceci = {"ok": False, "error": str(exc)}
    problems = []
    if report["unknown_stages"]:
        problems.append(f"unknown stages {report['unknown_stages']}")
    if report["unresolved_inputs"]:
        problems.append(f"{len(report['unresolved_inputs'])} unresolved input tags")
    if report["duplicate_outputs"]:
        problems.append("duplicate outputs")
    if report["cycle"]:
        problems.append("dependency cycle")
    msg = (f"{report['n_stages']} stages, DAG {'OK' if report['ok'] else 'FAILED: ' + '; '.join(problems)}; "
           f"{len(report['missing_input_files'])} overall input file(s) missing locally; "
           f"{len(report['commands'])} dry-run commands." +
           (f" ceci --dry-run {'passed' if ceci.get('ok') else 'FAILED'} under DESC_TXPIPE_ENV." if ceci else
            " (set DESC_TXPIPE_ENV to also run ceci --dry-run)"))
    return ArtifactResult(status="success", files=[], message=msg,
                          metadata={"pipeline_yml": str(path), "base_dir": str(base_dir), **report,
                                    "ceci_dry_run": ceci})


# --------------------------------------------------------------------------
# run / status
# --------------------------------------------------------------------------

_REFUSAL = (
    "txpipe_run_pipeline cannot run: no TXPipe environment is available. TXPipe is never imported by this "
    "server, so choose ONE of:\n"
    "  (a) LOCAL: set DESC_TXPIPE_ENV in the server's environment to a shell snippet that activates your "
    "TXPipe env and enters its checkout, e.g. 'source /path/TXPipe/conda/bin/activate; cd /path/TXPipe' "
    "(built with TXPipe/bin/install.sh); or pass that snippet as env_setup with dispatch set to local;\n"
    "  (b) FACILITY: set_dispatch('perlmutter'|'polaris') (or the client-side dispatch pack) and pass "
    "env_setup naming YOUR TXPipe environment on the facility, e.g. 'module load python; conda activate "
    "/path/to/txpipe/conda; cd /path/to/TXPipe' (built once with bin/perlmutter-install.sh). The server "
    "ships no default facility environment; inputs must then be facility paths."
)


@validate_call
def txpipe_run_pipeline(
    output_dir: Annotated[str, Field(min_length=1, description="Server-side directory for the run manifest and any products returned in-band (sacc/yml/txt).")],
    pipeline_yml: Annotated[str, Field(min_length=1, description="pipeline.yml from txpipe_generate_pipeline (its config: path must resolve).")],
    env_setup: Annotated[str | None, Field(description="Shell snippet activating the TXPipe environment where the pipeline runs (facility: required; local: overrides DESC_TXPIPE_ENV). Never defaulted by the server.")] = None,
    max_threads: Annotated[int, Field(ge=1, le=512, description="site.max_threads for ceci inside the job (node cores for a facility run).")] = 4,
    walltime_s: Annotated[int, Field(ge=120, le=86400, description="Job walltime / local timeout in seconds (example pipeline: ~600-1800; 20 deg^2: hours).")] = 1800,
    nodes: Annotated[int, Field(ge=1, le=1, description="Facility nodes (first release: 1; ceci runs as site local on that node).")] = 1,
    resume: Annotated[bool, Field(description="Continue a previous run in the same output dir (skips stages with final outputs).")] = True,
    txpipe_dir: Annotated[str, Field(description="Optional existing TXPipe checkout on the execution host (used if `txpipe` is not importable; otherwise the inner script clones TXPipe into the job dir).")] = "",
) -> ArtifactResult:
    """Run a composed TXPipe pipeline with ceci in a TXPipe environment: locally under DESC_TXPIPE_ENV, or as a facility job under the client-supplied env_setup (HEAVY).

    Local mode (dispatch local): `ceci pipeline.yml` as a subprocess under
    DESC_TXPIPE_ENV (or env_setup) with KMP_DUPLICATE_LIB_OK=TRUE and
    HDF5_USE_FILE_LOCKING=FALSE; products stay in the pipeline's output_dir.
    Remote mode (set_dispatch / dispatch pack): the env-kernel `txpipe_run`
    writes the YAMLs into the job directory, ensures a TXPipe checkout
    (clones if `txpipe` is not importable), runs ceci with site local and
    max_threads, and returns a per-stage manifest, the produced
    sacc/PNG/yml list, small sacc files base64 (decoded here into
    output_dir) and log tails. Large HDF5 catalogs/maps stay on the facility
    (paths in the manifest). Refuses up front when neither environment
    option is available. Expect minutes for the 1 deg^2 example, hours for
    20 deg^2. Then: sacc_inspect on twopoint_data_real.sacc, txpipe_run_status
    on the manifest, and the txpipe-sacc-to-likelihood skill.
    """
    path = Path(pipeline_yml).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"pipeline file not found: {path}")
    pipeline = load_pipeline_yaml(path)
    config, cfg_path = _load_config_for(pipeline, path)
    if cfg_path is not None and not cfg_path.is_file():
        raise ValueError(f"config file referenced by the pipeline not found: {cfg_path}")
    report = pc.validate(pipeline, base_dir=_base_dir_for(path))
    if not report["ok"]:
        raise ValueError("pipeline fails the index-based validation; fix it first (txpipe_validate_pipeline): "
                         f"unknown={report['unknown_stages']} unresolved={report['unresolved_inputs'][:5]} "
                         f"cycle={report['cycle']} duplicate_outputs={report['duplicate_outputs']}")

    from mcp_server.dispatch import remote_site, run_env_kernel  # lazy: server-only
    from ..envkernel import run_in_env

    site = remote_site()
    local_env = env_setup or os.environ.get("DESC_TXPIPE_ENV")
    if site and not env_setup:
        raise RuntimeError(f"dispatch is set to {site} but env_setup was not given.\n" + _REFUSAL)
    if not site and not local_env:
        raise RuntimeError(_REFUSAL)

    outdir = resolve_outdir(output_dir)
    slug = param_slug({"pipeline": str(path), "threads": max_threads, "resume": resume, "site": site or "local"})
    clone = clone_dir("txpipe")
    params = {"pipeline": pipeline, "config": config, "max_threads": max_threads, "resume": resume,
              "server_clone_dir": str(clone) if clone else None, "stage_outputs": _expected_outputs(pipeline),
              "timeout_s": max(60, walltime_s - 60), "inline_max_bytes": 2_000_000,
              "txpipe_dir": txpipe_dir or (str(clone) if (not site and clone) else None)}
    t0 = time.time()
    if site:
        res = run_env_kernel(env_setup, "txpipe_run", params, duration=walltime_s, nodes=nodes)
        engine = res.get("result") or {}
        inner = engine.get("result") if isinstance(engine, dict) else None
        host = res.get("host", site)
        env_check = engine.get("env_check") if isinstance(engine, dict) else None
        ran_on = f"{site} ({host})"
    else:
        params["output_dir"] = pipeline.get("output_dir")
        params["log_dir"] = pipeline.get("log_dir")
        job_dir = outdir / f"txpipe_job_{slug}"
        job_dir.mkdir(parents=True, exist_ok=True)
        params["job_dir"] = str(job_dir)
        res = run_in_env(local_env, "txpipe_run", params, timeout_s=walltime_s, job_dir=str(job_dir))
        if isinstance(res, str):
            raise RuntimeError(f"local TXPipe run failed under DESC_TXPIPE_ENV/env_setup: {res}")
        inner = res["result"]
        env_check = res.get("env_check")
        ran_on = "local"
    if not isinstance(inner, dict):
        raise RuntimeError(f"txpipe_run kernel returned no manifest: {inner!r}")

    # decode in-band products, write the manifest
    products_dir = outdir / f"txpipe_products_{slug}"
    files: list[str] = []
    for rel, blob in (inner.get("inline_files") or {}).items():
        target = products_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if blob.get("encoding") == "base64":
            import base64
            target.write_bytes(base64.b64decode(blob["data"]))
        else:
            target.write_text(blob.get("data", ""), encoding="utf-8")
        files.append(str(target))
    manifest = {k: v for k, v in inner.items() if k != "inline_files"}
    manifest.update({"ran_on": ran_on, "env_check": env_check, "pipeline_yml": str(path),
                     "wall_seconds": round(time.time() - t0, 1), "products_returned": files,
                     "dispatch_site": site or "local"})
    manifest_path = outdir / f"txpipe_run_manifest_{slug}.json"
    write_json(manifest_path, manifest)
    files.insert(0, str(manifest_path))
    counts = inner.get("stage_status_counts", {})
    saccs = [p["name"] for p in inner.get("produced", []) if p["kind"] == "sacc"]
    return ArtifactResult(
        status="success", files=files,
        message=f"TXPipe run {inner.get('status', '?')} on {ran_on} in {inner.get('ceci_seconds', '?')} s: "
                f"stages {counts}; sacc products: {saccs or 'none'}; "
                f"{len(files) - 1} file(s) returned in-band. Outputs at {inner.get('output_dir')} "
                f"(logs {inner.get('log_dir')}). Next: sacc_inspect / txpipe_run_status.",
        metadata={"manifest": str(manifest_path), "status": inner.get("status"), "ran_on": ran_on,
                  "stage_status_counts": counts, "produced": inner.get("produced", []),
                  "failed_stages": [n for n, s in (inner.get("stages") or {}).items()
                                    if s["status"] == "failed_or_incomplete"],
                  "stdout_tail": inner.get("stdout_tail"), "stderr_tail": inner.get("stderr_tail"),
                  "env_check": env_check, "txpipe_source": inner.get("txpipe_source"),
                  "output_dir": inner.get("output_dir"), "log_dir": inner.get("log_dir")},
    )


txpipe_run_pipeline.weight = "heavy"


@validate_call
def txpipe_run_status(
    run_output_dir: Annotated[str, Field(description="The pipeline's output_dir to scan (final vs inprogress_* files).")] = "",
    log_dir: Annotated[str, Field(description="The pipeline's log_dir (<Stage>.out per stage); tails and error markers are reported.")] = "",
    pipeline_yml: Annotated[str, Field(description="Optional pipeline.yml so stages can be classified by their expected outputs.")] = "",
    manifest_json: Annotated[str, Field(description="Alternative: a run manifest written by txpipe_run_pipeline (summarised instead of scanning).")] = "",
) -> ArtifactResult:
    """Classify a TXPipe run's stages (complete / inprogress / failed / not started) from an output dir + logs, or from a saved run manifest.

    Without pipeline_yml the output dir is scanned generically (final
    products vs inprogress_* leftovers) and each log is checked for
    Traceback/Error markers. For facility runs pass the manifest JSON
    returned by txpipe_run_pipeline (the output dir is not on this machine).
    """
    if manifest_json:
        m = json.loads(Path(manifest_json).expanduser().read_text(encoding="utf-8"))
        stages = m.get("stages") or {}
        counts = m.get("stage_status_counts") or {}
        return ArtifactResult(
            status="success", files=[],
            message=f"Manifest {Path(manifest_json).name}: run {m.get('status')} on {m.get('ran_on')}; stages {counts}; "
                    f"{len(m.get('produced', []))} products at {m.get('output_dir')}.",
            metadata={"status": m.get("status"), "stages": stages, "counts": counts, "produced": m.get("produced"),
                      "failed_stages": [n for n, s in stages.items() if s["status"] == "failed_or_incomplete"],
                      "output_dir": m.get("output_dir"), "log_dir": m.get("log_dir"),
                      "stdout_tail": m.get("stdout_tail"), "stderr_tail": m.get("stderr_tail")},
        )
    if not run_output_dir and not log_dir:
        raise ValueError("give run_output_dir and/or log_dir, or manifest_json.")
    out = Path(run_output_dir).expanduser() if run_output_dir else None
    logs = Path(log_dir).expanduser() if log_dir else None
    final, inprog = [], []
    if out and out.is_dir():
        for p in sorted(out.iterdir()):
            if p.is_file():
                (inprog if p.name.startswith("inprogress_") else final).append(
                    {"name": p.name, "bytes": p.stat().st_size})
    log_info = {}
    if logs and logs.is_dir():
        for p in sorted(logs.glob("*.out")):
            text = p.read_text(encoding="utf-8", errors="replace")
            log_info[p.stem] = {"path": str(p), "lines": text.count("\n"),
                                "error_marker": any(m in text for m in ("Traceback", "Error", "error:")),
                                "tail": "\n".join(text.splitlines()[-10:])}
    stages = None
    if pipeline_yml:
        from ..inner.txpipe_run import classify_stages
        pipeline = load_pipeline_yaml(Path(pipeline_yml).expanduser())
        stages = classify_stages(pipeline, str(out or pipeline.get("output_dir")),
                                 str(logs or pipeline.get("log_dir")), _expected_outputs(pipeline))
    counts = {}
    for s in (stages or {}).values():
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    return ArtifactResult(
        status="success", files=[],
        message=(f"{len(final)} final and {len(inprog)} in-progress files in {out}; " if out else "") +
                (f"{len(log_info)} stage logs, {sum(1 for v in log_info.values() if v['error_marker'])} with error markers; " if logs else "") +
                (f"stages {counts}." if stages else "pass pipeline_yml for per-stage classification."),
        metadata={"final_files": final, "inprogress_files": inprog, "logs": log_info, "stages": stages,
                  "counts": counts, "sacc_files": [f["name"] for f in final if f["name"].endswith(".sacc")]},
    )


@validate_call
def txpipe_fetch_example_data(
    dest: Annotated[str, Field(min_length=1, description="Directory to extract into; the tarball unpacks to <dest>/data/example/inputs (the layout TXPipe's examples expect when run from <dest>). Use the TXPipe clone dir only if it is writable and you accept files there.")],
    which: Annotated[Literal["example", "cmb"], Field(description="'example': 1 deg^2 catalogs for the metadetect/metacal/lensfit/mock examples (347 MB, v10); 'cmb': Quaia x Planck example (189 MB).")] = "example",
    force: Annotated[bool, Field(description="Re-download even if the extracted inputs already exist.")] = False,
) -> ArtifactResult:
    """Download and extract TXPipe's example data tarball (portal.nersc.gov; 347 MB for 'example') with a size check; for LOCAL use only.

    The hosted deployment does not carry this data (DESIGN.md section 8).
    After extraction, point txpipe_generate_pipeline `inputs` at
    <dest>/data/example/inputs/... (or run the pipeline from <dest> so the
    examples' relative paths resolve). The example is 1 deg^2 of simulated
    sky - enough to exercise every stage, not to check numbers.
    """
    info = EXAMPLE_DATA[which]
    dest_dir = resolve_outdir(dest)
    target = dest_dir / info["extracts_to"]
    if target.exists() and not force:
        n = sum(1 for _ in target.rglob("*") if _.is_file())
        return ArtifactResult(status="success", files=[str(target)],
                              message=f"{target} already exists with {n} files; skipped download (force=true to redo).",
                              metadata={"skipped": True, **info, "n_files": n})
    tarball = dest_dir / Path(info["url"]).name
    t0 = time.time()
    req = urllib.request.Request(info["url"], headers={"User-Agent": "desc-mcp-server"})
    with urllib.request.urlopen(req, timeout=60) as resp, tarball.open("wb") as fh:
        expected = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
    size_mb = got / 1e6
    if expected and got != expected:
        raise RuntimeError(f"download truncated: {got} of {expected} bytes for {info['url']}")
    if size_mb < 0.5 * info["expected_mb"]:
        raise RuntimeError(f"downloaded only {size_mb:.0f} MB, expected ~{info['expected_mb']} MB - the portal may have moved {info['url']}")
    with tarfile.open(tarball) as tf:
        try:
            tf.extractall(dest_dir, filter="data")
        except TypeError:  # python < 3.12
            tf.extractall(dest_dir)
    tarball.unlink(missing_ok=True)
    n = sum(1 for _ in target.rglob("*") if _.is_file()) if target.exists() else 0
    return ArtifactResult(
        status="success", files=[str(target)],
        message=f"Fetched {which} data ({size_mb:.0f} MB, {info['version']}) in {time.time() - t0:.0f} s; "
                f"{n} files under {target}.",
        metadata={**info, "downloaded_mb": round(size_mb, 1), "n_files": n, "dest": str(dest_dir)},
    )
