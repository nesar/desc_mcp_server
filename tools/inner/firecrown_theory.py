"""Inner script: theory data vector of a firecrown TwoPointExperiment, with an
optional noiseless / noisy realization written as a new sacc file
(env-kernel; see tools/envkernel.py).

params (JSON):
  experiment_yaml : str   TwoPointExperiment YAML
  params          : dict  parameter overrides (firecrown names)
  write_sacc      : bool  also write a sacc file whose data vector is the
                          theory (add_noise=False) or a draw from the
                          covariance (add_noise=True)
  add_noise       : bool
  seed            : int | None  numpy seed for the noisy draw
  sacc_output     : str   output path for the sacc (.hdf5 or .fits)
  strict          : bool  require the likelihood to cover every sacc point
                          (False keeps unmodelled points, e.g. scale cuts)

returns: {"n_data", "statistics": [{data_type, tracer1, tracer2, n, x_name,
          x, theory, data, sigma}], "loglike", "chi2", "sacc_file", ...}
"""

from .firecrown_loglike import (evaluate_point, load_experiment, make_params_map,
                                materialize_experiment, required_defaults, statistic_layout)


def main(params: dict) -> dict:
    import numpy as np

    experiment_yaml = materialize_experiment(params)
    overrides = params.get("params") or {}
    write_sacc = bool(params.get("write_sacc", False))
    add_noise = bool(params.get("add_noise", False))
    seed = params.get("seed")
    strict = bool(params.get("strict", False))

    like, tools, sacc_data = load_experiment(experiment_yaml)
    defaults = required_defaults(like, tools)
    layout = statistic_layout(like, sacc_data)
    pmap = make_params_map(defaults, overrides)

    out: dict = {"n_data": int(sum(st["n"] for st in layout)), "sacc_file": None,
                 "unknown_keys": sorted(k for k in overrides if k not in defaults)}
    if not write_sacc:
        res = evaluate_point(like, tools, pmap, layout, True, True)
    else:
        # make_realization needs the COMPUTED state, so run the cycle by hand
        res = {"loglike": None, "chi2": None, "error": None}
        try:
            tools.update(pmap)
            tools.prepare()
            like.update(pmap)
            res["loglike"] = float(like.compute_loglike(tools))
            res["chi2"] = float(like.compute_chisq(tools))
            theory = np.asarray(like.get_theory_vector(), dtype=float)
            data = np.asarray(like.get_data_vector(), dtype=float)
            cov = np.asarray(like.get_cov(), dtype=float)
            res["theory"], res["data"] = theory.tolist(), data.tolist()
            res["sigma"] = np.sqrt(np.clip(np.diag(cov), 0, None)).tolist()
            if seed is not None:
                np.random.seed(int(seed))
            new_sacc = like.make_realization(sacc_data, add_noise=add_noise, strict=strict)
            path = str(params.get("sacc_output") or "theory_realization.hdf5")
            if path.lower().endswith((".fits", ".fit")):
                new_sacc.save_fits(path, overwrite=True)
            else:
                new_sacc.save_hdf5(path, overwrite=True)
            out["sacc_file"] = path
            out["realization_n_points"] = int(len(new_sacc.mean))
        except Exception as exc:  # noqa: BLE001
            res["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                like.reset()
            finally:
                tools.reset()

    out.update({"loglike": res.get("loglike"), "chi2": res.get("chi2"),
                "error": res.get("error"), "unused_keys": res.get("unused_keys", [])})
    stats = []
    theory, data, sigma = res.get("theory"), res.get("data"), res.get("sigma")
    pos = 0
    for st in layout:
        n = st["n"]
        row = {k: v for k, v in st.items() if k != "indices"}
        if theory is not None:
            row["theory"] = theory[pos:pos + n]
            row["data"] = data[pos:pos + n]
            row["sigma"] = sigma[pos:pos + n]
        stats.append(row)
        pos += n
    out["statistics"] = stats
    return out
