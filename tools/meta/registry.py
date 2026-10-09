"""Registry of the DESC packages this server wraps and the tool families built on them.

Single source of truth for list_desc_packages / describe_desc_tool_family.
Versions are detected at call time from the running environment; the
`clone_version` entries are what the read-only reference clones were at when
the server was built (see notes/survey_*.md).

status values:
- "active"   : wrapped by tools in this server
- "compose"  : the server reads the package's source/configs and composes
               work for it, but never imports it (TXPipe)
- "indirect" : used through another package (TJPCov via augur/TXPipe)
"""

PACKAGES: dict[str, dict] = {
    "pyccl": {
        "role": "Core Cosmology Library: distances, growth, P(k), angular C_ell, "
                "correlation functions, halo model, baryons, mu-Sigma MG, emulators",
        "families": ["ccl"],
        "import_name": "pyccl",
        "clone_version": "3.3.6",
        "units": "CCL is h-free (Mpc, 1/Mpc, Mpc^3, Msun) and takes scale factor a; "
                 "this server converts to z and h/Mpc at the tool boundary",
        "citation": "Chisari et al. 2019, ApJS 242, 2 (arXiv:1812.05995)",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/LSSTDESC/CCL",
        "examples": "https://github.com/LSSTDESC/CCLX (notebooks)",
        "status": "active",
    },
    "sacc": {
        "role": "Save All Correlations and Covariances: the data-vector file format "
                "(tracers with n(z), two-point data, covariance) every DESC code exchanges",
        "families": ["sacc"],
        "import_name": "sacc",
        "clone_version": "2.4",
        "citation": "github.com/LSSTDESC/sacc",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/LSSTDESC/sacc",
        "status": "active",
    },
    "firecrown": {
        "role": "DESC likelihood framework: Gaussian 3x2pt / SN / cluster likelihoods with "
                "systematics, built on pyccl; connectors to Cobaya, CosmoSIS, NumCosmo",
        "families": ["firecrown"],
        "import_name": "firecrown",
        "clone_version": "1.16.0+74",
        "parameter_names": "cosmology: Omega_c Omega_b h n_s sigma8|A_s Omega_k Neff m_nu w0 wa T_CMB; "
                           "per-bin: {tracer}_delta_z {tracer}_sigma_z {tracer}_mult_bias {tracer}_bias "
                           "{tracer}_alphaz {tracer}_alphag {tracer}_z_piv {tracer}_mag_bias; "
                           "global IA: ia_bias alphaz z_piv (TATT: ia_a_1 ia_a_2 ia_a_d ...)",
        "citation": "github.com/LSSTDESC/firecrown (Vitenti, Zuntz et al.)",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/LSSTDESC/firecrown",
        "note": "conda-forge only (not on PyPI); core import needs numcosmo and crow",
        "status": "active",
    },
    "augur": {
        "role": "Fisher forecasting for LSST (SRD Y1/Y10 3x2pt): synthetic sacc data "
                "vectors with Gaussian/SRD/TJPCov covariance, numerical derivatives, "
                "Fisher matrices, priors, Fisher bias, contour plots",
        "families": ["augur"],
        "import_name": "augur",
        "clone_version": "1.2.4",
        "citation": "github.com/LSSTDESC/augur",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/LSSTDESC/augur",
        "note": "requires firecrown >= 1.14 (factory API); installed from the clone",
        "status": "active",
    },
    "tjpcov": {
        "role": "Covariance calculator (Gaussian f_sky, NaMaster, SSC) used by augur "
                "(cov_type: tjpcov) and TXPipe's covariance stages",
        "families": ["augur", "txpipe"],
        "import_name": "tjpcov",
        "clone_version": "0.5.1",
        "citation": "github.com/LSSTDESC/TJPCov",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/LSSTDESC/TJPCov",
        "status": "indirect",
    },
    "ceci": {
        "role": "Pipeline framework TXPipe runs on: stage classes with declared "
                "inputs/outputs/config, pipeline + config YAML, local/NERSC sites",
        "families": ["txpipe"],
        "import_name": "ceci",
        "clone_version": "2.5.1",
        "citation": "github.com/LSSTDESC/ceci",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/LSSTDESC/ceci",
        "status": "active",
    },
    "txpipe": {
        "role": "The DESC 3x2pt measurement pipeline: catalog ingest, tomographic "
                "selection, photo-z stacks, maps/masks/randoms, TreeCorr/NaMaster "
                "two-point measurements, covariances, null tests, blinding -> sacc files",
        "families": ["txpipe"],
        "import_name": "txpipe",
        "clone_version": "efad606 (2026-10-06)",
        "citation": "Prat, Zuntz et al. (TXPipe framework paper); github.com/LSSTDESC/TXPipe",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/LSSTDESC/TXPipe",
        "note": "NOT imported by this server (pins firecrown 1.7; compiled MPI/NaMaster/"
                "TreeCorr stack). Tools read its source for stage metadata, compose "
                "ceci pipelines, and run them in a separate TXPipe env (local, optional) "
                "or on a facility via the dispatch engine.",
        "status": "compose",
    },
}

FAMILIES: dict[str, dict] = {
    "meta": {"module": "tools.meta", "prefix": "",
             "summary": "discovery, skills, parameter-name conversion"},
    "ccl": {"module": "tools.ccl", "prefix": "ccl_",
            "summary": "pyccl theory predictions: background, P(k), C_ell, xi, halo model, SRD n(z)"},
    "sacc": {"module": "tools.sacc_tools", "prefix": "sacc_",
             "summary": "inspect/convert/prepare sacc data vectors; attach covariances"},
    "firecrown": {"module": "tools.firecrown_tools", "prefix": "firecrown_",
                  "summary": "build likelihoods from sacc + systematics; loglike, theory vectors, scans, chains"},
    "augur": {"module": "tools.augur_tools", "prefix": "augur_",
              "summary": "LSST Y1/Y10 Fisher forecasts: config, synthetic data, Fisher, contours"},
    "txpipe": {"module": "tools.txpipe_tools", "prefix": "txpipe_",
               "summary": "stage catalog, pipeline composition/validation, run locally or on a facility, inspect outputs"},
    "dispatch": {"module": "mcp_server.dispatch", "prefix": "",
                 "summary": "HPC execution: set_dispatch, get_dispatch, auth_status, export_dispatch_pack"},
}


def installed_version(import_name: str) -> str | None:
    import importlib
    import importlib.metadata as md
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return None
    v = getattr(mod, "__version__", None)
    if v and v != "?":
        return str(v)
    for dist in (import_name, f"lsstdesc-{import_name}", "lsstdesc-" + import_name.replace("_", "-")):
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
    return "present"
