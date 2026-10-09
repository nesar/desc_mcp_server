"""Inner script: evaluate a firecrown TwoPointExperiment likelihood at one or
more parameter points (env-kernel; see tools/envkernel.py).

Runs identically in-process (local) and under a facility environment
(remote). Only numpy + firecrown/sacc are imported, inside functions, and
only APIs that exist in firecrown 1.14-1.16 are used (TwoPointExperiment,
CCLFactory, ModelingTools, ParamsMap, get_default_params_map).

params (JSON):
  experiment_yaml : str   path of a TwoPointExperiment YAML (built by
                          firecrown_build_likelihood)
  points          : list[dict]  parameter overrides per point (cosmology +
                          nuisance, firecrown names); missing names take the
                          firecrown defaults
  return_vectors  : bool  include theory/data/sigma per statistic for the
                          FIRST point (default True)
  per_statistic   : bool  include per-statistic chi2 breakdown per point

returns: {"required": {name: default}, "n_data": int, "n_statistics": int,
          "statistics": [{data_type, tracer1, tracer2, n, x_name, x}],
          "results": [{"loglike", "chi2", "unused_keys", "error",
                       "per_statistic": [...], "theory", "data", "sigma"}]}
"""

import math

# module-level cache: experiment yaml path -> (likelihood, tools, sacc)
_CACHE: dict = {}


def materialize_experiment(params: dict) -> str:
    """Return the experiment YAML path for this call.

    Locally the wrapper passes ``experiment_yaml`` (a path). Remotely the
    wrapper ships ``experiment_yaml_text`` plus either ``sacc_remote_path``
    (a facility path) or ``sacc_b64``/``sacc_name`` (the file inline); both
    are written into the job CWD and the YAML's sacc path is rewritten.
    """
    if params.get("experiment_yaml") and not params.get("experiment_yaml_text"):
        return str(params["experiment_yaml"])
    import base64
    import os
    import re

    text = params["experiment_yaml_text"]
    if params.get("sacc_remote_path"):
        sacc_path = os.path.abspath(params["sacc_remote_path"])
    else:
        sacc_path = os.path.abspath(params.get("sacc_name") or "data.hdf5")
        with open(sacc_path, "wb") as fh:
            fh.write(base64.b64decode(params["sacc_b64"]))
    text = re.sub(r"(sacc_data_file:\s*).*", lambda m: m.group(1) + sacc_path, text, count=1)
    yaml_path = os.path.abspath(params.get("experiment_name") or "experiment.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    _CACHE.pop(yaml_path, None)
    return yaml_path


def load_experiment(experiment_yaml: str):
    """Build (likelihood, tools, sacc_data) for an experiment YAML (cached)."""
    key = str(experiment_yaml)
    if key in _CACHE:
        return _CACHE[key]
    import warnings

    from firecrown.likelihood.factories import TwoPointExperiment, load_sacc_data
    from firecrown.modeling_tools import ModelingTools

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = TwoPointExperiment.load_from_yaml(key)
        like = exp.make_likelihood()
        tools = ModelingTools(ccl_factory=exp.ccl_factory)
        sacc_data = load_sacc_data(exp.data_source.sacc_data_file)
    _CACHE[key] = (like, tools, sacc_data)
    return _CACHE[key]


def required_defaults(like, tools) -> dict:
    """Every sampler parameter the likelihood + tools need, with defaults."""
    from firecrown.updatable import get_default_params_map

    pmap = get_default_params_map(tools, like)
    out = {}
    for k, v in dict(pmap).items():
        if isinstance(v, (list, tuple)):
            # firecrown 1.16 reports m_nu as [] for massless neutrinos; the
            # tool boundary uses the summed mass in eV (0.0 = massless)
            out[k] = 0.0 if len(v) == 0 else [float(x) for x in v]
        else:
            out[k] = float(v)
    return out


def make_params_map(defaults: dict, overrides: dict):
    """ParamsMap = defaults with overrides applied via item assignment
    (ParamsMap.update refuses keys that already exist); ints cast to float."""
    from firecrown.updatable import ParamsMap

    pmap = ParamsMap(dict(defaults))
    for k, v in (overrides or {}).items():
        if isinstance(v, (list, tuple)):
            pmap[k] = [float(x) for x in v]
        else:
            pmap[k] = float(v)
    return pmap


def statistic_layout(like, sacc_data) -> list:
    """Per-statistic bookkeeping: data type, tracers, sacc indices and the
    ell/theta abscissa (read from the sacc tags so it works for both spaces)."""
    import numpy as np

    out = []
    for gs in like.statistics:
        st = getattr(gs, "statistic", gs)
        idx = np.asarray(st.sacc_indices, dtype=int)
        tr = st.sacc_tracers
        name1 = getattr(tr, "name1", None) or (tr[0] if isinstance(tr, (tuple, list)) else str(tr))
        name2 = getattr(tr, "name2", None) or (tr[1] if isinstance(tr, (tuple, list)) else str(tr))
        dtype = str(st.sacc_data_type)
        x_name = "theta_arcmin" if "_xi" in dtype else "ell"
        tag = "theta" if x_name == "theta_arcmin" else "ell"
        x = []
        for i in idx:
            try:
                x.append(float(sacc_data.data[int(i)].get_tag(tag)))
            except Exception:  # noqa: BLE001 - tag missing: fall back to position
                x.append(float(len(x)))
        out.append({"data_type": dtype, "tracer1": str(name1), "tracer2": str(name2),
                    "n": int(len(idx)), "indices": idx.tolist(), "x_name": x_name, "x": x})
    return out


def evaluate_point(like, tools, pmap, layout, want_vectors: bool, per_statistic: bool) -> dict:
    """One full update -> prepare -> compute -> reset cycle."""
    import numpy as np

    res: dict = {"loglike": None, "chi2": None, "unused_keys": [], "error": None}
    try:
        tools.update(pmap)
        tools.prepare()
        like.update(pmap)
        loglike = float(like.compute_loglike(tools))
        chi2 = float(like.compute_chisq(tools))
        res["loglike"] = loglike if math.isfinite(loglike) else None
        res["chi2"] = chi2 if math.isfinite(chi2) else None
        if not math.isfinite(loglike):
            res["error"] = "non-finite log-likelihood (CCL integration failure?)"
        try:
            res["unused_keys"] = sorted(str(k) for k in pmap.get_unused_keys())
        except Exception:  # noqa: BLE001 - older firecrown
            res["unused_keys"] = []
        if want_vectors or per_statistic:
            theory = np.asarray(like.get_theory_vector(), dtype=float)
            data = np.asarray(like.get_data_vector(), dtype=float)
            cov = np.asarray(like.get_cov(), dtype=float)
            sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
            resid = data - theory
            inv_cov = np.asarray(like.inv_cov, dtype=float) if hasattr(like, "inv_cov") \
                else np.linalg.inv(cov)
            share_vec = resid * (inv_cov @ resid)   # sums to chi2
            if want_vectors:
                res["theory"] = theory.tolist()
                res["data"] = data.tolist()
                res["sigma"] = sigma.tolist()
            if per_statistic:
                pos = 0
                rows = []
                for st in layout:
                    n = st["n"]
                    sl = slice(pos, pos + n)
                    r = resid[sl]
                    block = cov[sl, sl]
                    try:
                        chi2_block = float(r @ np.linalg.solve(block, r))
                    except np.linalg.LinAlgError:
                        chi2_block = float("nan")
                    rows.append({"data_type": st["data_type"], "tracer1": st["tracer1"],
                                 "tracer2": st["tracer2"], "n": n,
                                 "chi2_block": chi2_block if math.isfinite(chi2_block) else None,
                                 "chi2_share": float(share_vec[sl].sum())})
                    pos += n
                res["per_statistic"] = rows
    except Exception as exc:  # noqa: BLE001 - report, never crash a scan
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            like.reset()
        except Exception:  # noqa: BLE001
            pass
        try:
            tools.reset()
        except Exception:  # noqa: BLE001
            pass
    return res


def main(params: dict) -> dict:
    experiment_yaml = materialize_experiment(params)
    points = params.get("points") or [params.get("params") or {}]
    want_vectors = bool(params.get("return_vectors", True))
    per_statistic = bool(params.get("per_statistic", True))

    like, tools, sacc_data = load_experiment(experiment_yaml)
    defaults = required_defaults(like, tools)
    layout = statistic_layout(like, sacc_data)
    n_data = int(sum(st["n"] for st in layout))

    results = []
    for i, overrides in enumerate(points):
        unknown = sorted(k for k in (overrides or {}) if k not in defaults)
        pmap = make_params_map(defaults, overrides)
        res = evaluate_point(like, tools, pmap, layout, want_vectors and i == 0, per_statistic)
        res["unknown_keys"] = unknown  # not required by this likelihood
        results.append(res)

    import firecrown

    return {"required": defaults, "n_data": n_data, "n_statistics": len(layout),
            "statistics": [{k: v for k, v in st.items() if k != "indices"} for st in layout],
            "results": results, "firecrown_version": getattr(firecrown, "__version__", "?")}
