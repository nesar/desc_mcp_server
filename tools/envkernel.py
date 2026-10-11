"""Generic "env-kernel": run an inner script under a facility-resident
Python environment chosen by the CLIENT.

Why this exists. Most of the DESC stack (firecrown, numcosmo, TXPipe's
compiled dependencies) is conda-forge only, so the hep-genesis engine's
venv + pip bootstrap cannot build it on a compute node. What it CAN do is run
this kernel under the facility's module python (``pip_deps=None``), and this
kernel then launches the real computation as a subprocess under an
environment the user names in ``env_setup`` - a shell snippet such as
``source /global/common/software/lsst/common/miniconda/setup_current_python.sh``.

Design rules (hep-genesis dispatch contract R1/R5):
- stdlib only; this module must import under any Python >= 3.9
- JSON-safe in and out
- a failure returns a string starting with "Error" (the engine's convention)
- no facility identity here: ``env_setup`` is an ARGUMENT, and the candidate
  defaults below are public collaboration paths, not user settings

Inner scripts live in ``tools/inner/`` (its ``__init__`` is empty so importing
one never drags pydantic/matplotlib into the facility environment). Each
defines ``main(params: dict) -> dict``. Locally (no dispatch) the wrappers call
``main`` in-process; remotely this kernel runs it under ``env_setup``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Directory that contains the ``tools`` package: <job dir> on a facility
# (the pack is staged as <job dir>/tools), the server checkout locally.
PACK_ROOT = str(Path(__file__).resolve().parents[1])

# Public, documented environment activations an agent may offer the user.
# They are CANDIDATES: the client picks (or supplies its own); nothing here
# is a default identity or project.
ENV_SETUP_CANDIDATES = {
    "perlmutter": [
        {
            "name": "desc-cosmology",
            "env_setup": "source $CFS/lsst/groups/MCP/setup-cosmology.sh",
            "provides": "firecrown 1.14.3, augur, pyccl, sacc, tjpcov, cosmosis, namaster, mpi4py (DESC MCP working group env)",
            "suitable_for": ["firecrown", "augur", "ccl", "sacc"],
            "reference": "https://github.com/LSSTDESC/desc-cosmology-env",
            "status": "verified 2026-10-09: firecrown chain kernel imports and runs",
        },
        {
            "name": "desc-python",
            "env_setup": "source /global/common/software/lsst/common/miniconda/setup_current_python.sh",
            "provides": "firecrown, pyccl 3.3.6, sacc 2.4, tjpcov 0.5.1, numcosmo 0.27 (DESC-maintained)",
            "suitable_for": ["firecrown", "augur", "ccl", "sacc"],
            "reference": "https://github.com/LSSTDESC/desc-python",
            "status": "BROKEN 2026-10-09: its firecrown (0.0.0) imports a missing 'crow' module; "
                      "try desc-cosmology first",
        },
        {
            "name": "user TXPipe env",
            "env_setup": "module load python; conda activate <your TXPipe env>; cd <your TXPipe checkout>",
            "provides": "TXPipe + ceci + NaMaster/TreeCorr/RAIL stack (built once with TXPipe/bin/perlmutter-install.sh)",
            "suitable_for": ["txpipe"],
            "reference": "https://github.com/LSSTDESC/TXPipe/blob/master/bin/perlmutter-install.sh",
        },
    ],
    "polaris": [
        {
            "name": "user-built desc-cosmology env",
            "env_setup": "source <your conda prefix>/bin/activate <your env>",
            "provides": "build once from external/desc-cosmology-env/slac/env-nobuild-*.yml (or the desc-python hpc lock) in your own ALCF area; no DESC-maintained stack exists on ALCF",
            "suitable_for": ["firecrown", "augur", "ccl", "sacc"],
            "reference": "https://github.com/LSSTDESC/desc-cosmology-env",
        },
    ],
}

_DRIVER = """\
import json, os, sys, traceback
# the pack root (the directory holding tools/): the job dir on a facility,
# the server checkout locally
sys.path.insert(0, {pack_root!r})
sys.path.insert(0, {job_dir!r})
out = {{"ok": False}}
try:
    import importlib
    mod = importlib.import_module("tools.inner." + {inner!r})
    with open({params_path!r}, encoding="utf-8") as fh:
        params = json.load(fh)
    out["result"] = mod.main(params)
    out["ok"] = True
except Exception:
    out["error"] = traceback.format_exc()
env_check = {{"python": sys.version.split()[0], "executable": sys.executable}}
for name in ("pyccl", "sacc", "firecrown", "augur", "tjpcov", "ceci", "txpipe", "pymaster", "smokescreen"):
    try:
        m = importlib.import_module(name)
        env_check[name] = getattr(m, "__version__", "present")
    except Exception:
        env_check[name] = None
out["env_check"] = env_check
with open({result_path!r}, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
"""


def _write_driver(inner: str, params: dict, job_dir: str, tag: str = "") -> tuple[str, str, str]:
    """Write params + driver for ``tools.inner.<inner>`` into job_dir;
    returns (params_path, driver_path, result_path). ``tag`` keeps several
    runs of the same inner script apart (background chains)."""
    stem = f"inner_{inner}" + (f"_{tag}" if tag else "")
    params_path = os.path.join(job_dir, f"{stem}_params.json")
    result_path = os.path.join(job_dir, f"{stem}_result.json")
    driver_path = os.path.join(job_dir, f"{stem}_driver.py")
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh)
    with open(driver_path, "w", encoding="utf-8") as fh:
        fh.write(_DRIVER.format(job_dir=job_dir, pack_root=PACK_ROOT, inner=inner,
                                params_path=params_path, result_path=result_path))
    return params_path, driver_path, result_path


def start_in_background(env_setup: str | None, inner: str, params: dict,
                        job_dir: str, tag: str = "") -> dict:
    """Start ``tools.inner.<inner>.main(params)`` as a DETACHED subprocess and
    return at once: {"pid", "driver_path", "result_path", "log_path",
    "started_at"}. Under ``env_setup`` when given, else this interpreter.
    The result file appears when the run ends ({"ok": ..}); stdout/stderr
    go to the log file. Local long runs only - on a facility the job IS the
    background."""
    import sys
    import time

    if not inner.replace("_", "").isalnum():
        raise ValueError(f"invalid inner script name {inner!r}.")
    job_dir = str(Path(job_dir).resolve())
    os.makedirs(job_dir, exist_ok=True)
    _, driver_path, result_path = _write_driver(inner, params, job_dir, tag)
    stem = f"inner_{inner}" + (f"_{tag}" if tag else "")
    log_path = os.path.join(job_dir, f"{stem}_driver.log")
    if env_setup and env_setup.strip():
        argv = ["bash", "-lc", f"{env_setup.strip().rstrip(';')}; exec python {driver_path!s}"]
    else:
        argv = [sys.executable, driver_path]
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child
    try:
        proc = subprocess.Popen(argv, cwd=job_dir, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        log.close()
    info = {"pid": proc.pid, "driver_path": driver_path, "result_path": result_path, "log_path": log_path,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "inner": inner, "job_dir": job_dir}
    with open(os.path.join(job_dir, f"{stem}_background.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    return info


def run_in_env(env_setup: str, inner: str, params: dict,
               timeout_s: int = 3000, job_dir: str | None = None) -> dict | str:
    """Run ``tools.inner.<inner>.main(params)`` under ``env_setup``.

    Returns {"result": ..., "env_check": {...}, "stdout_tail": ..., "stderr_tail": ...}
    or a string starting with "Error" (engine convention) on failure.
    """
    if not env_setup or not env_setup.strip():
        return ("Error: env_setup is required - a shell snippet that activates a "
                "facility Python environment containing the DESC stack (e.g. the "
                "desc-python activation on NERSC). See ENV_SETUP_CANDIDATES.")
    if not inner.replace("_", "").isalnum():
        return f"Error: invalid inner script name {inner!r}."
    job_dir = str(Path(job_dir or os.getcwd()).resolve())
    _, driver_path, result_path = _write_driver(inner, params, job_dir)
    cmd = f"{env_setup.strip().rstrip(';')}; python {driver_path!s}"
    try:
        proc = subprocess.run(["bash", "-lc", cmd], cwd=job_dir, capture_output=True,
                              text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"Error: inner script {inner} exceeded {timeout_s}s under env_setup={env_setup!r}."
    tail = lambda s: "\n".join((s or "").splitlines()[-40:])
    if not os.path.exists(result_path):
        return ("Error: inner script wrote no result (environment activation or import "
                f"failed?). exit={proc.returncode}\nSTDOUT:\n{tail(proc.stdout)}\n"
                f"STDERR:\n{tail(proc.stderr)}")
    with open(result_path, encoding="utf-8") as fh:
        out = json.load(fh)
    if not out.get("ok"):
        return (f"Error: inner script {inner} failed:\n{out.get('error', '')}\n"
                f"env_check={out.get('env_check')}\nSTDERR:\n{tail(proc.stderr)}")
    return {"result": out["result"], "env_check": out.get("env_check", {}),
            "stdout_tail": tail(proc.stdout), "stderr_tail": tail(proc.stderr)}


if __name__ == "__main__":  # tiny self-test: python -m tools.envkernel
    print(json.dumps(run_in_env("true", "noop", {"x": 1}, job_dir=os.getcwd()), indent=1))
