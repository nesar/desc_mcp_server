"""Inner script: report the environment a kernel runs in (python, executable,
DESC package versions) - the smoke test for lock-built environments."""
import platform
import sys


def main(params: dict) -> dict:
    import importlib

    versions = {}
    for name in ("pyccl", "sacc", "firecrown", "augur", "tjpcov", "pymaster", "cobaya", "getdist", "numpy"):
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "present")
        except Exception as exc:  # noqa: BLE001 - absence is the finding
            versions[name] = f"missing ({type(exc).__name__})"
    return {"echo": params, "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "host": platform.node(),
            "env_check": {"python": sys.version.split()[0], "executable": sys.executable, **versions}}
