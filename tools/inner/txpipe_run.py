"""Inner script: run a composed TXPipe/ceci pipeline under a TXPipe environment.

Runs where the user's TXPipe environment lives: on a facility compute node
under the client-supplied ``env_setup`` (via tools/envkernel.py), or on this
machine under ``DESC_TXPIPE_ENV``. Stdlib only - the environment guarantees
nothing beyond TXPipe's own stack. JSON in, JSON out.

params (all JSON-safe):
    pipeline: dict           the pipeline YAML (as a dict) composed by the server
    config: dict             the per-stage config YAML (as a dict)
    max_threads: int         site.max_threads for ceci's local site (node cores)
    resume: bool             ceci resume flag (True continues a previous run)
    output_dir: str|None     where outputs go; default <job cwd>/outputs
    log_dir: str|None        default <job cwd>/logs
    txpipe_dir: str|None     an existing TXPipe checkout to use if `txpipe` is
                             not importable; else one is cloned into the job dir
    txpipe_git: str          clone URL (default https://github.com/LSSTDESC/TXPipe)
    server_clone_dir: str|None  path prefix of the server's TXPipe clone, so
                             example-relative paths the composer made absolute
                             (data/fiducial_cosmology.yml, bpz columns, ...)
                             are re-rooted onto the checkout used here
    inline_max_bytes: int    sacc files smaller than this come back base64 (default 2 MB)
    timeout_s: int           subprocess timeout for ceci (default 3000)

returns: {"status": "success"|"failed", "returncode", "command", "stages": {...},
          "produced": [...], "inline_files": {...}, "stdout_tail", "stderr_tail", ...}
"""

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

IN_PROGRESS = "inprogress_"
SMALL_TEXT_MAX = 200_000


def _dump_yaml(data) -> str:
    """Minimal YAML writer (PyYAML is present in any ceci env, but be safe)."""
    try:
        import yaml
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120)
    except Exception:  # noqa: BLE001
        return json.dumps(data, indent=1)  # JSON is valid YAML


def _rewrite_paths(obj, old_prefix, new_prefix):
    """Replace a path prefix in every string of a nested structure."""
    if not old_prefix:
        return obj
    if isinstance(obj, str):
        if obj.startswith(old_prefix):
            return new_prefix + obj[len(old_prefix):]
        return obj
    if isinstance(obj, list):
        return [_rewrite_paths(x, old_prefix, new_prefix) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_paths(v, old_prefix, new_prefix) for k, v in obj.items()}
    return obj


def _tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


def ensure_txpipe(job_dir, txpipe_dir, git_url, log):
    """Return (checkout dir or None, how). TXPipe is run from a checkout
    (there is no pip package); the pack cannot carry its 127 files."""
    if importlib.util.find_spec("txpipe") is not None:
        spec = importlib.util.find_spec("txpipe")
        where = os.path.dirname(os.path.dirname(spec.origin)) if spec.origin else None
        log.append(f"txpipe importable from {spec.origin}")
        return where, "importable"
    # env_setup conventionally ends with `cd <checkout>`, so the cwd itself is a candidate
    candidates = [txpipe_dir, os.getcwd(), os.path.join(job_dir, "TXPipe"), os.path.join(os.getcwd(), "TXPipe")]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "txpipe", "__init__.py")):
            log.append(f"using existing TXPipe checkout {c}")
            return os.path.abspath(c), "existing checkout"
    dest = os.path.join(job_dir, "TXPipe")
    log.append(f"cloning {git_url} -> {dest}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", git_url, dest], check=True,
                       capture_output=True, text=True, timeout=900)
        try:
            subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
                           cwd=dest, capture_output=True, text=True, timeout=900)
        except Exception as exc:  # noqa: BLE001 - submodules are optional (WLMassMap, pyfsb)
            log.append(f"submodule init skipped: {exc}")
        return dest, "cloned"
    except Exception as exc:  # noqa: BLE001
        log.append(f"git clone failed: {exc}")
        return None, "unavailable"


def classify_stages(pipeline, output_dir, log_dir, stage_outputs):
    """Per-stage status from the output dir + logs (final vs inprogress_*)."""
    stages = {}
    for entry in pipeline.get("stages", []):
        name = entry["name"] if isinstance(entry, dict) else entry
        outs = stage_outputs.get(name, [])
        final = [f for f in outs if os.path.exists(os.path.join(output_dir, f))]
        inprog = [f for f in outs if os.path.exists(os.path.join(output_dir, IN_PROGRESS + f))]
        log_path = os.path.join(log_dir, name + ".out")
        log_tail = ""
        if os.path.isfile(log_path):
            try:
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    log_tail = _tail(fh.read(), 15)
            except Exception:  # noqa: BLE001
                log_tail = "(unreadable)"
        if outs and len(final) == len(outs):
            status = "complete"
        elif inprog:
            status = "inprogress"
        elif os.path.isfile(log_path):
            status = "failed_or_incomplete"
        else:
            status = "not_started"
        stages[name] = {"status": status, "expected_outputs": outs, "final": final,
                        "inprogress": inprog, "log": log_path if os.path.isfile(log_path) else None,
                        "log_tail": log_tail}
    return stages


def collect_products(output_dir, inline_max_bytes):
    produced, inline = [], {}
    if not os.path.isdir(output_dir):
        return produced, inline
    for root, _dirs, files in os.walk(output_dir):
        if os.path.basename(root) == "cache" or "/cache/" in root.replace("\\", "/") + "/":
            continue
        for f in sorted(files):
            if f.startswith(IN_PROGRESS):
                continue
            path = os.path.join(root, f)
            ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            if ext not in ("sacc", "png", "yml", "yaml", "txt", "json"):
                continue
            size = os.path.getsize(path)
            rel = os.path.relpath(path, output_dir)
            produced.append({"path": path, "name": rel, "bytes": size, "kind": ext})
            try:
                if ext == "sacc" and size <= inline_max_bytes:
                    with open(path, "rb") as fh:
                        inline[rel] = {"encoding": "base64", "bytes": size,
                                       "data": base64.b64encode(fh.read()).decode("ascii")}
                elif ext in ("yml", "yaml", "txt", "json") and size <= SMALL_TEXT_MAX:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        inline[rel] = {"encoding": "text", "bytes": size, "data": fh.read()}
            except Exception:  # noqa: BLE001
                pass
    return produced, inline


def main(params: dict) -> dict:
    t0 = time.time()
    log = []
    # The job dir is NOT the cwd: env_setup usually ends with `cd <TXPipe checkout>`.
    # The pack root (the directory holding tools/) is the engine's job directory on
    # a facility; the local wrapper always passes job_dir explicitly.
    pack_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    job_dir = os.path.abspath(params.get("job_dir") or pack_root)
    os.makedirs(job_dir, exist_ok=True)
    pipeline = dict(params["pipeline"])
    config = dict(params.get("config") or {})
    max_threads = int(params.get("max_threads") or os.cpu_count() or 1)
    output_dir = os.path.abspath(params.get("output_dir") or os.path.join(job_dir, "outputs"))
    log_dir = os.path.abspath(params.get("log_dir") or os.path.join(job_dir, "logs"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkout, how = ensure_txpipe(job_dir, params.get("txpipe_dir"), params.get("txpipe_git")
                                  or "https://github.com/LSSTDESC/TXPipe", log)
    result = {"status": "failed", "job_dir": job_dir, "output_dir": output_dir, "log_dir": log_dir,
              "txpipe_checkout": checkout, "txpipe_source": how, "log": log,
              "python": sys.executable, "host": os.uname().nodename if hasattr(os, "uname") else ""}
    if checkout is None:
        result["error"] = ("TXPipe is neither importable in this environment nor cloneable here. "
                           "Point env_setup at a TXPipe env (cd into its checkout) or pass txpipe_dir.")
        return result

    # re-root example-relative paths the composer resolved against the server's clone
    server_clone = params.get("server_clone_dir")
    if server_clone and os.path.abspath(server_clone) != os.path.abspath(checkout):
        pipeline = _rewrite_paths(pipeline, server_clone.rstrip("/"), checkout.rstrip("/"))
        config = _rewrite_paths(config, server_clone.rstrip("/"), checkout.rstrip("/"))
        log.append(f"re-rooted {server_clone} -> {checkout}")
    # caches under the output dir so nothing lands in the checkout
    for name, sec in config.items():
        if isinstance(sec, dict):
            for key in ("cache_dir", "patch_dir"):
                if key in sec:
                    sec[key] = os.path.join(output_dir, "cache", str(name))
    pipeline["output_dir"] = output_dir
    pipeline["log_dir"] = log_dir
    pipeline["pipeline_log"] = os.path.join(log_dir, "pipeline_log.txt")
    pipeline["site"] = {"name": "local", "max_threads": max_threads}
    pipeline["launcher"] = {"name": "mini", "interval": 1.0}
    pipeline["resume"] = bool(params.get("resume", True))
    pipeline["python_paths"] = [p if os.path.isabs(p) else os.path.join(checkout, p)
                                for p in (pipeline.get("python_paths") or [])]
    config_path = os.path.join(job_dir, "config.yml")
    pipeline["config"] = config_path
    pipeline_path = os.path.join(job_dir, "pipeline.yml")
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(_dump_yaml(config))
    with open(pipeline_path, "w", encoding="utf-8") as fh:
        fh.write(_dump_yaml(pipeline))
    result["pipeline_yml"] = pipeline_path
    result["config_yml"] = config_path

    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # keep the checkout pristine (read-only clone locally)
    env["PYTHONPATH"] = checkout + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", str(max_threads))
    ceci = shutil.which("ceci")
    cmd = ([ceci] if ceci else [sys.executable, "-m", "ceci"]) + [pipeline_path]
    result["command"] = " ".join(cmd)
    timeout_s = int(params.get("timeout_s") or 3000)
    t_run = time.time()
    try:
        proc = subprocess.run(cmd, cwd=job_dir, env=env, capture_output=True, text=True, timeout=timeout_s)
        result["returncode"] = proc.returncode
        result["stdout_tail"] = _tail(proc.stdout, 60)
        result["stderr_tail"] = _tail(proc.stderr, 60)
        result["status"] = "success" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        result["returncode"] = None
        result["stdout_tail"] = _tail(exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout, 60)
        result["stderr_tail"] = _tail(exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr, 60)
        result["status"] = "timeout"
        result["error"] = f"ceci exceeded {timeout_s}s; rerun with resume=true to continue"
    result["ceci_seconds"] = round(time.time() - t_run, 1)

    stage_outputs = params.get("stage_outputs") or {}
    result["stages"] = classify_stages(pipeline, output_dir, log_dir, stage_outputs)
    counts = {}
    for s in result["stages"].values():
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    result["stage_status_counts"] = counts
    produced, inline = collect_products(output_dir, int(params.get("inline_max_bytes") or 2_000_000))
    result["produced"] = produced
    result["inline_files"] = inline
    result["n_inline_files"] = len(inline)
    result["elapsed_seconds"] = round(time.time() - t0, 1)
    return result
