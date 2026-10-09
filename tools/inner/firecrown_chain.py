"""Inner script: Cobaya MCMC over a firecrown TwoPointExperiment in PURE_CCL
mode (no Cobaya theory block; CCL computes everything) - env-kernel.

params (JSON):
  experiment_yaml : str   TwoPointExperiment YAML (ccl_factory.creation_mode
                          must be pure_ccl_mode)
  fixed           : dict  parameter values held fixed (firecrown names);
                          anything not in `priors` and not fixed takes the
                          firecrown default
  priors          : dict  {name: {"min": lo, "max": hi, "ref": x, "proposal": w}}
                          uniform priors for the sampled parameters
  max_samples     : int   cobaya mcmc max_samples
  rminus1_stop    : float Gelman-Rubin stopping criterion
  work_dir        : str   where the chain files are written (job CWD remotely)
  chain_prefix    : str   file prefix inside work_dir
  seed            : int | None
  resume          : bool  continue from the Cobaya checkpoint of a previous run
                          with the same prefix (walltime-bounded chains); the
                          sampler settings must match that run

returns: {"chain_files": [...], "n_samples", "acceptance_rate", "summary":
          {name: {"mean", "std", "lower68", "upper68"}}, "sampled": [...],
          "converged": bool|None, "Rminus1": float|None}
"""

import os

from .firecrown_loglike import load_experiment, materialize_experiment, required_defaults

_FACTORY_SRC = '''"""Firecrown likelihood factory written by desc-mcp-server (firecrown_run_chain)."""
from firecrown.likelihood.factories import TwoPointExperiment
from firecrown.modeling_tools import ModelingTools


def build_likelihood(build_parameters):
    exp = TwoPointExperiment.load_from_yaml(build_parameters.get_string("likelihood_config"))
    return exp.make_likelihood(), ModelingTools(ccl_factory=exp.ccl_factory)
'''


def _weighted_summary(names, values, weights) -> dict:
    import numpy as np

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = {}
    for j, name in enumerate(names):
        x = np.asarray(values[:, j], dtype=float)
        mean = float((w * x).sum())
        std = float(np.sqrt(max((w * (x - mean) ** 2).sum(), 0.0)))
        order = np.argsort(x)
        cdf = np.cumsum(w[order])
        lo = float(x[order][np.searchsorted(cdf, 0.16)]) if len(x) > 2 else mean
        hi = float(x[order][min(np.searchsorted(cdf, 0.84), len(x) - 1)]) if len(x) > 2 else mean
        out[name] = {"mean": mean, "std": std, "lower68": lo, "upper68": hi}
    return out


def main(params: dict) -> dict:
    import warnings

    import numpy as np

    experiment_yaml = os.path.abspath(materialize_experiment(params))
    fixed = dict(params.get("fixed") or {})
    priors = dict(params.get("priors") or {})
    max_samples = int(params.get("max_samples", 200))
    rminus1_stop = float(params.get("rminus1_stop", 0.05))
    work_dir = os.path.abspath(params.get("work_dir") or os.getcwd())
    prefix = str(params.get("chain_prefix") or "chain")
    seed = params.get("seed")
    resume = bool(params.get("resume", False))
    os.makedirs(work_dir, exist_ok=True)

    # cheap check that the experiment loads and learn the required names
    like, tools, _ = load_experiment(experiment_yaml)
    defaults = required_defaults(like, tools)
    unknown = sorted(k for k in list(fixed) + list(priors) if k not in defaults)
    if unknown:
        raise ValueError(f"parameters not used by this likelihood: {unknown}; "
                         f"required names: {sorted(defaults)}")
    if not priors:
        raise ValueError("priors is empty: give at least one sampled parameter "
                         "as {name: {min, max}}.")

    factory_path = os.path.join(work_dir, f"{prefix}_factory.py")
    with open(factory_path, "w", encoding="utf-8") as fh:
        fh.write(_FACTORY_SRC)

    cobaya_params: dict = {}
    for name, default in defaults.items():
        if name in priors:
            spec = priors[name]
            lo, hi = float(spec["min"]), float(spec["max"])
            ref = float(spec.get("ref", fixed.get(name, default)))
            ref = min(max(ref, lo), hi)
            prop = float(spec.get("proposal", (hi - lo) / 20.0))
            cobaya_params[name] = {"prior": {"min": lo, "max": hi}, "ref": ref,
                                   "proposal": prop, "latex": name.replace("_", r"\_")}
        elif name in fixed:
            cobaya_params[name] = float(fixed[name])
        else:
            cobaya_params[name] = float(default) if not isinstance(default, list) else default

    from firecrown.connector.cobaya.likelihood import LikelihoodConnector

    info = {
        "likelihood": {"firecrown": {"external": LikelihoodConnector,
                                     "firecrownIni": factory_path,
                                     "build_parameters": {"likelihood_config": experiment_yaml}}},
        "params": cobaya_params,
        "sampler": {"mcmc": {"max_samples": max_samples, "Rminus1_stop": rminus1_stop,
                             "burn_in": 0, "max_tries": 10 * max(1, len(priors)) * 100}},
        "output": os.path.join(work_dir, prefix),
    }
    # resume continues from <prefix>.checkpoint (Cobaya refuses mismatched
    # settings); force restarts over any previous files of this prefix.
    info["resume" if resume else "force"] = True
    if seed is not None:
        info["sampler"]["mcmc"]["seed"] = int(seed)

    from cobaya.run import run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _updated, sampler = run(info, no_mpi=True)

    sample = sampler.products()["sample"]
    sampled = list(priors)
    # all weighted samples (cobaya already removed burn_in=0)
    data = sample.data if hasattr(sample, "data") else sample
    weights = np.asarray(data["weight"], dtype=float)
    values = np.column_stack([np.asarray(data[n], dtype=float) for n in sampled])
    summary = _weighted_summary(sampled, values, weights)
    try:  # getdist means/stds when available (same numbers, canonical tool)
        gd = sample.to_getdist()
        means = gd.getMeans()
        stds = np.sqrt(gd.getVars())
        for j, name in enumerate(gd.getParamNames().list()):
            if name in summary:
                summary[name]["getdist_mean"] = float(means[j])
                summary[name]["getdist_std"] = float(stds[j])
        used_getdist = True
    except Exception:  # noqa: BLE001 - getdist optional
        used_getdist = False

    n_samples = int(len(weights))
    try:
        acc = float(sampler.get_acceptance_rate())
    except Exception:  # noqa: BLE001
        acc = None
    rminus1 = None
    converged = None
    try:
        rminus1 = float(sampler.Rminus1_last) if hasattr(sampler, "Rminus1_last") else None
        converged = bool(sampler.converged) if hasattr(sampler, "converged") else None
    except Exception:  # noqa: BLE001
        pass

    chain_files = sorted(os.path.join(work_dir, f) for f in os.listdir(work_dir)
                         if f.startswith(prefix + ".") or f == os.path.basename(factory_path))
    loglike_col = None
    for col in ("minuslogpost", "minuslogprior", "chi2"):
        if col in data:
            loglike_col = col
            break
    best = None
    if "minuslogpost" in data:
        i = int(np.argmin(np.asarray(data["minuslogpost"])))
        best = {n: float(data[n].iloc[i]) if hasattr(data[n], "iloc") else float(data[n][i])
                for n in sampled}
        best["minuslogpost"] = float(np.asarray(data["minuslogpost"])[i])
    return {"chain_files": chain_files, "n_samples": n_samples, "n_weighted": float(weights.sum()),
            "acceptance_rate": acc, "summary": summary, "sampled": sampled,
            "fixed": {k: v for k, v in cobaya_params.items() if not isinstance(v, dict)},
            "converged": converged, "Rminus1": rminus1, "used_getdist": used_getdist,
            "best_fit": best, "loglike_column": loglike_col, "factory_file": factory_path}
