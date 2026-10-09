"""AST-based index of every TXPipe pipeline stage (no TXPipe import).

TXPipe cannot live in this server's Python process (its firecrown 1.7 pin and
compiled MPI/NaMaster/TreeCorr stack cannot coexist with firecrown 1.16, see
notes/DESIGN.md D1/D2). Every ceci stage, however, declares what we need as
plain class attributes (`name`, `inputs`, `outputs`, `config_options`,
`parallel`, `dask_parallel`), so this module reads the clone's source files
as text with :mod:`ast` and resolves the small set of expression forms TXPipe
actually uses:

- ``inputs = [("tag", FileType), ...]`` and ``Base.inputs + [...]``
- ``config_options = {"k": StageParameter(dtype, default, required=, msg=), ...}``,
  bare values (``"nside": 512``) and bare types (``"nside": int`` = required),
  ``{**Base.config_options, ...}``, ``SHARED | {...}``, ``Base.config_options.copy()``
  followed by ``config_options.update({...})`` / ``config_options["k"] = ...``
- inheritance: a subclass that does not redeclare an attribute inherits it from
  its first stage base (resolved within the index, module-level dicts included)

The clone is read-only; nothing here writes into it. The index is cached per
process (common.get_cached) and records the clone commit in its metadata.
"""

from __future__ import annotations

import ast
import re
import subprocess
import warnings
from pathlib import Path
from typing import Any

import yaml

from ..common import clone_dir, get_cached, require_clone

# ceci file-type roots and their suffixes (external/ceci/ceci/file_types.py)
_CECI_SUFFIXES = {
    "DataFile": None, "HDFFile": "hdf5", "FitsFile": "fits", "TextFile": "txt",
    "YamlFile": "yml", "Directory": "", "FileCollection": "", "PNGFile": "png",
    "PickleFile": "pkl", "ParquetFile": "pq",
}

# RAIL stages used by the example pipelines (rail is not part of the clone;
# signatures from rail.creation.engines.flowEngine, rail.creation.degraders.
# grid_selection, rail.estimation.algos.nz_dir / bpz_lite). Only inputs,
# outputs and file suffixes are needed for DAG checks and dry-run commands.
RAIL_STAGES: dict[str, dict] = {
    "FlowCreator": {"module": "rail.creation.engines.flowEngine",
                    "doc": "Draw a mock photometric sample from a pzflow normalizing flow (RAIL).",
                    "inputs": [("model", "ModelHandle")], "outputs": [("output", "PqHandle")]},
    "GridSelection": {"module": "rail.creation.degraders.grid_selection",
                      "doc": "Degrade a sample to a spectroscopic-like selection on a colour-magnitude grid (RAIL).",
                      "inputs": [("input", "PqHandle")], "outputs": [("output", "PqHandle")]},
    "NZDirInformer": {"module": "rail.estimation.algos.nz_dir",
                      "doc": "Train the direct-calibration (DIR) n(z) model from a spectroscopic sample (RAIL).",
                      "inputs": [("input", "TableHandle")], "outputs": [("model", "ModelHandle")]},
    "BPZliteInformer": {"module": "rail.estimation.algos.bpz_lite",
                        "doc": "Prepare the BPZ-lite photo-z prior/model (RAIL).",
                        "inputs": [("input", "TableHandle")], "outputs": [("model", "ModelHandle")]},
    "BPZliteEstimator": {"module": "rail.estimation.algos.bpz_lite",
                         "doc": "Estimate per-object photo-z PDFs with BPZ-lite (RAIL).",
                         "inputs": [("model", "ModelHandle"), ("input", "TableHandle")],
                         "outputs": [("output", "QPHandle")]},
}
_RAIL_SUFFIXES = {"ModelHandle": "pkl", "PqHandle": "pq", "TableHandle": "hdf5",
                  "QPHandle": "hdf5", "Hdf5Handle": "hdf5"}

# Purpose groups (survey section 2 / docs/make-stages.py), by module path
# first and by name pattern second.
GROUPS = ["ingest", "photo-z", "selection", "calibration", "lss-weights", "maps",
          "two-point", "covariance", "blinding", "null-tests", "plots", "extensions", "other"]
_MODULE_GROUPS = [
    ("extensions/", "extensions"),
    ("ingest/", "ingest"), ("simulation.py", "ingest"), ("exposure_info.py", "ingest"),
    ("magnification.py", "ingest"),
    ("photoz_stack.py", "photo-z"), ("rail/", "photo-z"),
    ("source_selection/", "selection"), ("lens_selector.py", "selection"),
    ("calibrate.py", "calibration"), ("shear_calibration/", "calibration"),
    ("lssweights.py", "lss-weights"),
    ("maps.py", "maps"), ("auxiliary_maps.py", "maps"), ("masks.py", "maps"),
    ("noise_maps.py", "maps"), ("convergence.py", "maps"), ("random_cats.py", "maps"),
    ("jackknife.py", "maps"), ("metadata.py", "maps"), ("randoms/", "maps"), ("mapping/", "maps"),
    ("twopoint.py", "two-point"), ("twopoint_fourier.py", "two-point"), ("theory.py", "two-point"),
    ("delta_sigma.py", "two-point"),
    ("covariance.py", "covariance"), ("covariance_nmt.py", "covariance"),
    ("blinding.py", "blinding"),
    ("twopoint_null_tests.py", "null-tests"), ("psf_diagnostics.py", "null-tests"),
    ("spatial_diagnostics.py", "null-tests"), ("map_correlations.py", "null-tests"),
    ("diagnostics.py", "plots"), ("map_plots.py", "plots"), ("twopoint_plots.py", "plots"),
]
_NAME_GROUPS = [
    (re.compile(r"(Splitter|CutCatalog|CutShearCatalog)"), "calibration"),
    (re.compile(r"^(TXGammaT|TXRowe|TXTau|TXPSFDiag|TXPSFMoment|TXGalaxyStar|TXBrighterFatter|TXFocalPlane|TXApertureMass)"), "null-tests"),
    (re.compile(r"Covariance"), "covariance"),
    (re.compile(r"Blinding"), "blinding"),
    (re.compile(r"Plot"), "plots"),
]


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------

def _first_line(doc: str | None) -> str:
    lines = [ln.strip() for ln in (doc or "").splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _name_of(node: ast.AST) -> str | None:
    """Dotted name of a Name/Attribute node, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _literal(node: ast.AST) -> tuple[bool, Any]:
    """(ok, value) for a literal node; non-literals come back as source text."""
    try:
        return True, ast.literal_eval(node)
    except Exception:
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = _name_of(node)
            if name in ("int", "float", "str", "bool", "list", "dict"):
                return True, {"__type__": name}
            if name in ("np.inf", "numpy.inf", "inf"):
                return True, float("inf")
        return False, ast.unparse(node)


def _parse_stage_parameter(call: ast.Call) -> dict:
    """StageParameter(dtype=None, default=None, fmt="%s", required=None, msg=...)."""
    pos = ["dtype", "default", "fmt", "required", "msg"]
    kw: dict[str, Any] = {}
    for i, arg in enumerate(call.args):
        if i < len(pos):
            kw[pos[i]] = arg
    for k in call.keywords:
        if k.arg:
            kw[k.arg] = k.value
    dtype = None
    if "dtype" in kw:
        dtype = _name_of(kw["dtype"])
        if dtype is None:
            dtype = ast.unparse(kw["dtype"])
        if dtype == "None":
            dtype = None
    default = None
    if "default" in kw:
        ok, default = _literal(kw["default"])
    required = None
    if "required" in kw:
        ok, required = _literal(kw["required"])
        if not ok:
            required = None
    if required is None:
        required = default is None
    help_text = ""
    if "msg" in kw:
        ok, help_text = _literal(kw["msg"])
        help_text = help_text if isinstance(help_text, str) else str(help_text)
    return {"dtype": dtype, "default": default, "required": bool(required),
            "help": help_text, "form": "StageParameter"}


def _parse_option_value(node: ast.AST) -> dict:
    """One config_options value: StageParameter(...), bare value, or bare type."""
    if isinstance(node, ast.Call) and _name_of(node.func) in ("StageParameter", "ceci.StageParameter", "config.StageParameter"):
        return _parse_stage_parameter(node)
    ok, val = _literal(node)
    if ok and isinstance(val, dict) and "__type__" in val:
        return {"dtype": val["__type__"], "default": None, "required": True, "help": "", "form": "bare-type"}
    if ok:
        dtype = None if val is None else type(val).__name__
        return {"dtype": dtype, "default": val, "required": False, "help": "", "form": "bare-value"}
    return {"dtype": None, "default": val, "required": False, "help": "", "form": "expression"}


class _Resolver:
    """Resolve class/module-level expressions for inputs/outputs/config dicts."""

    def __init__(self, classes: dict[str, dict], module_vars: dict[str, dict[str, ast.AST]]):
        self.classes = classes          # class name -> raw record
        self.module_vars = module_vars  # module -> {var name: node}
        self._global_vars: dict[str, tuple[str, ast.AST]] = {}
        for mod, d in module_vars.items():
            for k, v in d.items():
                self._global_vars.setdefault(k, (mod, v))
        self._resolving: set[tuple[str, str]] = set()

    # -- variable / attribute lookup ------------------------------------
    def _var(self, name: str, module: str) -> tuple[str, ast.AST] | None:
        d = self.module_vars.get(module, {})
        if name in d:
            return module, d[name]
        if name in self._global_vars:
            return self._global_vars[name]
        return None

    def class_attr(self, cls_name: str, attr: str):
        """Resolved attribute of a (possibly base) class, with inheritance."""
        rec = self.classes.get(cls_name)
        if rec is None:
            return None
        key = (cls_name, attr)
        if key in self._resolving:
            return None
        self._resolving.add(key)
        try:
            if attr in rec["raw"]:
                return self.resolve_attr(attr, rec["raw"][attr], rec["module"], rec)
            for base in rec["bases"]:
                val = self.class_attr(base.split(".")[-1], attr)
                if val is not None:
                    return val
            return None
        finally:
            self._resolving.discard(key)

    # -- expression resolution --------------------------------------------
    def resolve_attr(self, attr: str, nodes: list[ast.AST], module: str, rec: dict):
        if attr in ("inputs", "outputs"):
            return self.io_list(nodes[0], module, rec["class"])
        if attr == "config_options":
            opts = self.config_dict(nodes[0], module)
            for extra in nodes[1:]:
                self._apply_mutation(opts, extra, module)
            return opts
        ok, val = _literal(nodes[0])
        return val

    def _tag_value(self, node: ast.AST, module: str, cls_name: str | None):
        """A tag is normally a string literal; TXTwoPointFourier uses a class
        attribute (``output_main = "twopoint_data_fourier"``) by name."""
        ok, val = _literal(node)
        if ok:
            return val
        if isinstance(node, ast.Name):
            if cls_name:
                val = self.class_attr(cls_name, node.id)
                if isinstance(val, str):
                    return val
            found = self._var(node.id, module)
            if found:
                ok, val = _literal(found[1])
                if ok:
                    return val
        return ast.unparse(node)

    def io_list(self, node: ast.AST, module: str, cls_name: str | None = None) -> list[tuple[str, str]] | None:
        if isinstance(node, (ast.List, ast.Tuple)):
            out = []
            for el in node.elts:
                if isinstance(el, (ast.Tuple, ast.List)) and len(el.elts) == 2:
                    tag = self._tag_value(el.elts[0], module, cls_name)
                    ftype = _name_of(el.elts[1]) or ast.unparse(el.elts[1])
                    out.append((str(tag), ftype.split(".")[-1]))
                elif isinstance(el, ast.Starred):
                    inner = self.io_list(el.value, module, cls_name)
                    out.extend(inner or [])
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.io_list(node.left, module, cls_name) or []
            right = self.io_list(node.right, module, cls_name) or []
            return left + right
        if isinstance(node, ast.Attribute) and node.attr in ("inputs", "outputs"):
            owner = _name_of(node.value)
            if owner:
                return self.class_attr(owner.split(".")[-1], node.attr) or []
        if isinstance(node, ast.Name):
            found = self._var(node.id, module)
            if found:
                return self.io_list(found[1], found[0])
        if isinstance(node, ast.Call) and _name_of(node.func) == "list":
            return self.io_list(node.args[0], module, cls_name) if node.args else []
        return None

    def config_dict(self, node: ast.AST, module: str) -> dict:
        if isinstance(node, ast.Dict):
            out: dict = {}
            for k, v in zip(node.keys, node.values):
                if k is None:  # **expr
                    out.update(self.config_dict(v, module) or {})
                else:
                    ok, key = _literal(k)
                    out[str(key)] = _parse_option_value(v)
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            out = dict(self.config_dict(node.left, module) or {})
            out.update(self.config_dict(node.right, module) or {})
            return out
        if isinstance(node, ast.Call):
            fname = _name_of(node.func) or ""
            if fname.endswith(".copy") and isinstance(node.func, ast.Attribute):
                return dict(self.config_dict(node.func.value, module) or {})
            if fname == "dict":
                out = {}
                for a in node.args:
                    out.update(self.config_dict(a, module) or {})
                for k in node.keywords:
                    if k.arg is None:
                        out.update(self.config_dict(k.value, module) or {})
                    else:
                        out[k.arg] = _parse_option_value(k.value)
                return out
        if isinstance(node, ast.Attribute) and node.attr == "config_options":
            owner = _name_of(node.value)
            if owner:
                return dict(self.class_attr(owner.split(".")[-1], "config_options") or {})
        if isinstance(node, ast.Name):
            found = self._var(node.id, module)
            if found:
                return self.config_dict(found[1], found[0])
        return {}

    def _apply_mutation(self, opts: dict, node: ast.AST, module: str) -> None:
        # config_options.update({...}) / config_options["k"] = value
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if _name_of(call.func) == "config_options.update" and call.args:
                opts.update(self.config_dict(call.args[0], module) or {})
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            tgt = node.targets[0]
            if _name_of(tgt.value) == "config_options":
                ok, key = _literal(tgt.slice)
                opts[str(key)] = _parse_option_value(node.value)


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

_STAGE_ROOTS = {"PipelineStage", "PipelineStageBase", "ceci.PipelineStage", "RailStage"}


def _scan_module(path: Path, pkg_root: Path) -> tuple[str, dict[str, dict], dict[str, ast.AST], list[str]]:
    """Parse one file: (module name, {class: raw record}, module vars, errors)."""
    rel = path.relative_to(pkg_root.parent).with_suffix("")
    module = ".".join(rel.parts)
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    errors: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # upstream SyntaxWarnings (invalid escapes) are not ours
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return module, {}, {}, [f"{path.name}: {exc}"]
    module_vars: dict[str, ast.AST] = {}
    classes: dict[str, dict] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            module_vars[node.targets[0].id] = node.value
        elif isinstance(node, ast.ClassDef):
            raw: dict[str, list[ast.AST]] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    tgt = stmt.targets[0]
                    if isinstance(tgt, ast.Name):
                        raw.setdefault(tgt.id, []).append(stmt.value)
                    elif isinstance(tgt, ast.Subscript) and _name_of(tgt.value) == "config_options":
                        raw.setdefault("config_options", []).append(stmt)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    raw.setdefault(stmt.target.id, []).append(stmt.value)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                        and _name_of(stmt.value.func) == "config_options.update":
                    raw.setdefault("config_options", []).append(stmt)
            # the first element of config_options must be the base expression;
            # a subscript assignment / update without a base means "inherit then mutate"
            if "config_options" in raw and not isinstance(raw["config_options"][0], ast.expr):
                raw["config_options"].insert(0, ast.parse("dict()", mode="eval").body)
            classes[node.name] = {
                "class": node.name, "module": module, "file": str(path.relative_to(pkg_root.parent)),
                "bases": [(_name_of(b) or ast.unparse(b)) for b in node.bases],
                "doc": _first_line(ast.get_docstring(node)),
                "raw": raw, "lineno": node.lineno,
            }
    return module, classes, module_vars, errors


def _file_type_suffixes(pkg_root: Path) -> dict[str, str | None]:
    """FileType class -> suffix, from txpipe/data_types.py + ceci roots."""
    suffixes: dict[str, str | None] = dict(_CECI_SUFFIXES)
    suffixes.update(_RAIL_SUFFIXES)
    bases: dict[str, str] = {}
    own: dict[str, str | None] = {}
    dt = pkg_root / "data_types.py"
    if dt.is_file():
        tree = ast.parse(dt.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases[node.name] = (_name_of(node.bases[0]) or "").split(".")[-1] if node.bases else ""
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and _name_of(stmt.targets[0]) == "suffix":
                        ok, val = _literal(stmt.value)
                        own[node.name] = val

    def resolve(name: str, depth=0):
        if name in suffixes:
            return suffixes[name]
        if name in own:
            return own[name]
        if name in bases and depth < 10:
            return resolve(bases[name], depth + 1)
        return None

    for name in bases:
        suffixes[name] = resolve(name)
    return suffixes


def _group_for(record: dict) -> str:
    name = record["name"] or record["class"]
    file = record["file"].replace("\\", "/")
    if "/extensions/" in file:
        return "extensions"
    for pat, grp in _NAME_GROUPS:
        if pat.search(name):
            return grp
    for frag, grp in _MODULE_GROUPS:
        if frag in file:
            return grp
    return "other"


def _git_head(clone: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(clone), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


_JINJA = re.compile(r"\{\{[^}]*\}\}")
_JINJA_BLOCK = re.compile(r"^\s*\{%.*?%\}\s*$", re.MULTILINE)


def load_pipeline_yaml(path: Path) -> dict:
    """Load a ceci pipeline YAML, tolerating Jinja placeholders (dp1 examples)."""
    text = path.read_text(encoding="utf-8")
    text = _JINJA_BLOCK.sub("", _JINJA.sub("TEMPLATE", text))
    data = yaml.safe_load(text) or {}
    return data if isinstance(data, dict) else {}


def _scan_examples(clone: Path) -> list[dict]:
    """Every examples/**/*.yml with a `stages:` list (pipeline files)."""
    out = []
    ex_root = clone / "examples"
    if not ex_root.is_dir():
        return out
    for path in sorted(ex_root.rglob("*.yml")):
        try:
            data = load_pipeline_yaml(path)
        except Exception as exc:  # noqa: BLE001
            out.append({"path": str(path.relative_to(clone)), "error": str(exc)})
            continue
        stages = data.get("stages")
        if not isinstance(stages, list):
            continue
        entries = []
        for st in stages:
            if isinstance(st, dict) and "name" in st:
                entries.append({"name": st["name"], "classname": st.get("classname", st["name"]),
                                "aliases": st.get("aliases") or {},
                                "nprocess": st.get("nprocess", 1),
                                "threads_per_process": st.get("threads_per_process", 1)})
        inputs = data.get("inputs") or {}
        inputs_info = {}
        for tag, val in (inputs.items() if isinstance(inputs, dict) else []):
            if isinstance(val, str) and val.lower() != "none":
                p = Path(val) if Path(val).is_absolute() else clone / val
                inputs_info[tag] = {"path": val, "exists_locally": p.exists()}
            elif isinstance(val, dict):
                inputs_info[tag] = {"path": val, "exists_locally": False, "note": "data-registry lookup"}
            else:
                inputs_info[tag] = {"path": None, "exists_locally": False, "note": "explicit none (optional input)"}
        site = data.get("site") or {}
        out.append({
            "path": str(path.relative_to(clone)), "example": path.parent.relative_to(ex_root).parts[0],
            "n_stages": len(entries), "stages": entries,
            "site": site.get("name") if isinstance(site, dict) else site,
            "max_threads": site.get("max_threads") if isinstance(site, dict) else None,
            "launcher": (data.get("launcher") or {}).get("name") if isinstance(data.get("launcher"), dict) else data.get("launcher"),
            "modules": str(data.get("modules", "")).split(),
            "python_paths": data.get("python_paths") or [],
            "output_dir": data.get("output_dir"), "log_dir": data.get("log_dir"),
            "config": data.get("config"), "inputs": inputs_info,
            "n_inputs_missing_locally": sum(1 for v in inputs_info.values()
                                            if v["path"] is not None and not v["exists_locally"]),
            "jinja_templated": bool(_JINJA.search(path.read_text(encoding="utf-8"))),
        })
    return out


def build_index(clone: Path | None = None) -> dict:
    """Scan the clone and return {"stages": {name: record}, "metadata": {...}, ...}."""
    clone = clone or require_clone("txpipe")
    pkg_root = clone / "txpipe"
    if not pkg_root.is_dir():
        raise RuntimeError(f"{clone} has no txpipe/ package directory.")
    classes: dict[str, dict] = {}
    module_vars: dict[str, dict[str, ast.AST]] = {}
    errors: list[str] = []
    n_files = 0
    for path in sorted(pkg_root.rglob("*.py")):
        if any(part in ("test", "tests", "__pycache__", "ui") for part in path.relative_to(pkg_root).parts):
            continue
        n_files += 1
        module, cls, mv, errs = _scan_module(path, pkg_root)
        errors.extend(errs)
        module_vars[module] = mv
        for cname, rec in cls.items():
            # later definitions with the same class name (rare) keep the first
            classes.setdefault(cname, rec)

    # stage-ness: any ancestor is a ceci PipelineStage
    def is_stage(cname: str, depth=0) -> bool:
        rec = classes.get(cname)
        if rec is None or depth > 15:
            return False
        for base in rec["bases"]:
            short = base.split(".")[-1]
            if base in _STAGE_ROOTS or short in _STAGE_ROOTS:
                return True
            if is_stage(short, depth + 1):
                return True
        return False

    resolver = _Resolver(classes, module_vars)
    suffixes = _file_type_suffixes(pkg_root)
    examples = _scan_examples(clone)
    usage: dict[str, list[str]] = {}
    for ex in examples:
        for st in ex.get("stages", []):
            usage.setdefault(st["classname"], []).append(ex["path"])

    stages: dict[str, dict] = {}
    by_class: dict[str, str] = {}
    for cname, rec in classes.items():
        if not is_stage(cname):
            continue
        name = resolver.class_attr(cname, "name")
        declares_name = "name" in rec["raw"]
        if not isinstance(name, str):
            name = cname
        inputs = resolver.class_attr(cname, "inputs") or []
        outputs = resolver.class_attr(cname, "outputs") or []
        config = resolver.class_attr(cname, "config_options") or {}
        parallel = resolver.class_attr(cname, "parallel")
        dask = resolver.class_attr(cname, "dask_parallel")
        stage_bases = [b.split(".")[-1] for b in rec["bases"] if is_stage(b.split(".")[-1])]
        record = {
            "name": name, "class": cname, "module": rec["module"], "file": rec["file"],
            "doc": rec["doc"], "bases": stage_bases,
            "declares_name": declares_name,
            "abstract": (not declares_name) or name == "BaseStageDoNotRunDirectly",
            "inputs": [[t, ft] for t, ft in inputs],
            "outputs": [[t, ft] for t, ft in outputs],
            "config_options": config,
            "parallel": True if parallel is None else bool(parallel),
            "dask_parallel": bool(dask) if dask is not None else False,
            "in_extensions": "/extensions/" in rec["file"].replace("\\", "/"),
            "used_in_examples": sorted(set(usage.get(name, []) + usage.get(cname, []))),
        }
        record["group"] = _group_for(record)
        # a class that declares its own name wins the YAML name slot
        prev = stages.get(name)
        if prev is None or (declares_name and not prev["declares_name"]):
            stages[name] = record
        by_class[cname] = name
    for rname, r in RAIL_STAGES.items():
        if rname not in stages:
            stages[rname] = {
                "name": rname, "class": rname, "module": r["module"], "file": "(RAIL, not in clone)",
                "doc": r["doc"], "bases": ["RailStage"], "declares_name": True, "abstract": False,
                "inputs": [list(x) for x in r["inputs"]], "outputs": [list(x) for x in r["outputs"]],
                "config_options": {}, "parallel": False, "dask_parallel": False,
                "in_extensions": False, "used_in_examples": sorted(set(usage.get(rname, []))),
                "group": "photo-z", "external": True,
            }
    metadata = {
        "clone_dir": str(clone), "clone_commit": _git_head(clone),
        "n_files_scanned": n_files, "n_classes": len(classes), "n_stages": len(stages),
        "n_examples": len(examples), "parse_errors": errors,
        "groups": {g: sum(1 for s in stages.values() if s["group"] == g) for g in GROUPS},
        "file_type_suffixes": suffixes,
        "note": "AST index of the read-only TXPipe clone; TXPipe is never imported by this server.",
    }
    return {"stages": stages, "by_class": by_class, "examples": examples,
            "suffixes": suffixes, "metadata": metadata}


def get_index() -> dict:
    """Process-cached index (rebuild only in a new process)."""
    clone = clone_dir("txpipe")
    key = f"txpipe_stage_index:{clone}"
    return get_cached(key, lambda: build_index(require_clone("txpipe")))


def find_stage(name: str) -> dict | None:
    """Stage record by YAML/CLI name or by class name (case-sensitive first)."""
    idx = get_index()
    rec = idx["stages"].get(name)
    if rec is None and name in idx["by_class"]:
        rec = idx["stages"].get(idx["by_class"][name])
    if rec is None:
        low = name.lower()
        for n, r in idx["stages"].items():
            if n.lower() == low or r["class"].lower() == low:
                return r
    return rec


def output_filename(tag: str, file_type: str, suffixes: dict | None = None) -> str:
    """ceci FileType.make_name: '{tag}.{suffix}' (or bare tag for directories)."""
    suffixes = suffixes or get_index()["suffixes"]
    suffix = suffixes.get(file_type)
    if suffix is None:
        suffix = "hdf5" if file_type.endswith(("Catalog", "File", "Maps")) else "dat"
    return f"{tag}.{suffix}" if suffix else tag


def stage_command(stage: dict, inputs: dict[str, str], config_path: str, outputs: dict[str, str],
                  aliases: dict | None = None, instance_name: str | None = None,
                  module: str = "txpipe") -> str:
    """Mirror of ceci.PipelineStage.generate_command (no TXPipe import).

    inputs/outputs map ALIASED tags to paths; the flags use the class's own
    tag names, exactly as ceci does.
    """
    aliases = aliases or {}
    flags = [stage["name"]]
    for tag, _ in stage["inputs"]:
        flags.append(f"--{tag}={inputs[aliases.get(tag, tag)]}")
    if instance_name and instance_name != stage["name"]:
        flags.append(f"--name={instance_name}")
    flags.append(f"--config={config_path}")
    for tag, _ in stage["outputs"]:
        flags.append(f"--{tag}={outputs[aliases.get(tag, tag)]}")
    return f"python3 -m {module} " + " ".join(flags)


def decorate_local(cmd: str, threads: int = 1, nprocess: int = 1,
                   python_paths: list[str] | None = None, mpi_command: str = "mpirun -n") -> str:
    """ceci LocalSite.command (non-container branch)."""
    parts = [f"OMP_NUM_THREADS={threads}", f"NUMBA_NUM_THREADS={threads}"]
    if python_paths:
        parts.append("PYTHONPATH=" + ":".join(python_paths) + ":$PYTHONPATH")
    if nprocess > 1:
        parts.append(f"{mpi_command} {nprocess}")
    parts.append(cmd)
    if nprocess > 1:
        parts.append("--mpi")
    return " ".join(parts)
