"""Validate the desc-mcp Python environment: import every DESC package and
run one tiny computation per family. Run from OUTSIDE the clone directories:

    python tests/smoke_env.py

Exit code 0 only if every check passes.
"""

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHECKS = []


def check(name, fn):
    t = time.time()
    try:
        detail = fn()
        CHECKS.append((name, "OK", f"{detail} ({time.time() - t:.1f}s)"))
    except Exception as exc:  # noqa: BLE001
        CHECKS.append((name, "FAIL", f"{type(exc).__name__}: {str(exc)[:120]}"))


def versions():
    out = []
    for m in ("numpy", "pyccl", "camb", "sacc", "firecrown", "augur", "tjpcov",
              "ceci", "cobaya", "mcp", "hep_genesis"):
        try:
            mod = importlib.import_module(m)
            out.append(f"{m} {getattr(mod, '__version__', 'present')}")
        except Exception:
            out.append(f"{m} MISSING")
    return "; ".join(out)


def ccl_pk():
    import numpy as np
    import pyccl as ccl
    from tools.cosmology import CosmologyParams
    c = CosmologyParams().build()
    p = ccl.nonlin_matter_power(c, np.array([1.0 * 0.67]), 1.0) * 0.67**3
    assert 200 < p[0] < 1000, p  # ~400 (Mpc/h)^3 for vanilla LCDM
    return f"P_nl(k=1 h/Mpc, z=0) = {p[0]:.0f} (Mpc/h)^3"


def sacc_io():
    import numpy as np
    import sacc
    s = sacc.Sacc()
    z = np.linspace(0, 2, 50)
    s.add_tracer("NZ", "src0", z, np.exp(-((z - 0.8) / 0.3) ** 2), quantity="galaxy_shear")
    s.add_ell_cl("galaxy_shear_cl_ee", "src0", "src0", np.arange(10, 100, 10), np.ones(9) * 1e-8)
    return f"sacc {sacc.__version__}: {len(s.tracers)} tracer, {len(s.data)} points"


def firecrown_loglike():
    from firecrown.modeling_tools import ModelingTools
    from firecrown.ccl_factory import CCLFactory
    from firecrown.updatable import get_default_params_map
    tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True,
                                                 creation_mode="pure_ccl_mode"))
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()
    cosmo = tools.get_ccl_cosmology()
    tools.reset()
    return f"firecrown pure-CCL ModelingTools -> Omega_m={cosmo['Omega_m']:.3f}"


def augur_import():
    import augur
    from augur.analyze import Analyze  # noqa: F401
    from augur.generate import generate  # noqa: F401
    import importlib.metadata as md
    return f"augur {md.version('augur')} (generate, Analyze importable)"


def ceci_yaml():
    import ceci  # noqa: F401
    from tools.common import clone_dir
    tx = clone_dir("txpipe")
    assert tx is not None, "TXPipe clone not found (DESC_TXPIPE_DIR)"
    yml = tx / "examples" / "metadetect" / "pipeline.yml"
    assert yml.exists()
    return f"ceci importable; TXPipe clone at {tx}"


def envkernel():
    from tools.envkernel import run_in_env
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = run_in_env("true", "noop", {"ping": 1}, job_dir=d)
    assert isinstance(out, dict), out
    return f"env-kernel ran under {out['env_check']['executable']}"


def tjpcov_import():
    import importlib.metadata as md
    from tjpcov.covariance_calculator import CovarianceCalculator  # noqa: F401
    from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky, RealGaussianFsky  # noqa: F401
    return f"tjpcov {md.version('tjpcov')} (CovarianceCalculator, Gaussian f_sky classes importable)"


def namaster_small():
    import numpy as np
    import healpy as hp
    import pymaster as nmt
    nside = 16
    mask = np.ones(hp.nside2npix(nside))
    m = np.random.default_rng(0).normal(size=mask.size)
    f = nmt.NmtField(mask, [m], lmax=3 * nside - 1)
    b = nmt.NmtBin.from_nside_linear(nside, 8)
    cl = nmt.compute_full_master(f, f, b)
    assert cl.shape[0] == 1 and np.all(np.isfinite(cl))
    return f"pymaster {nmt.__version__}: full-sky spin-0 bandpowers at nside {nside} ({cl.shape[1]} bins)"


def smokescreen_import():
    import importlib.metadata as md
    from smokescreen import ConcealDataVector  # noqa: F401
    from smokescreen.encryption import decrypt_file, encrypt_file  # noqa: F401
    from smokescreen.param_shifts import draw_flat_or_deterministic_param_shifts  # noqa: F401
    return f"smokescreen {md.version('smokescreen')} (ConcealDataVector, encryption importable)"


if __name__ == "__main__":
    check("versions", versions)
    check("ccl P(k)", ccl_pk)
    check("sacc write", sacc_io)
    check("firecrown ModelingTools", firecrown_loglike)
    check("augur import", augur_import)
    check("ceci + TXPipe clone", ceci_yaml)
    check("env-kernel", envkernel)
    check("tjpcov import", tjpcov_import)
    check("namaster bandpowers", namaster_small)
    check("smokescreen import", smokescreen_import)
    fails = 0
    for name, status, detail in CHECKS:
        print(f"{status:5s} {name:26s} {detail}")
        fails += status == "FAIL"
    print(f"\n{len(CHECKS)} checks: {len(CHECKS) - fails} OK, {fails} FAIL")
    sys.exit(1 if fails else 0)
