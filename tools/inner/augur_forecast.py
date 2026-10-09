"""Inner env-kernel for the augur tools: synthetic data vector and Fisher matrix.

``main(params) -> dict`` runs augur 1.2.x (``generate`` and ``Analyze``) on a
forecast configuration and returns JSON-safe results. The SAME function is
called in-process by the wrappers when the server runs locally and through
``tools.envkernel.run_in_env`` under a client-chosen ``env_setup`` on a
facility node. Keep this module free of pydantic/matplotlib/mcp imports and
of imports from tools.common / tools.cosmology (only numpy + DESC packages).

params
------
config : dict            augur configuration (already Jinja-free, absolute paths)
config_path : str        alternative: path to a YAML config (rendered with augur.parse)
mode : "generate"|"fisher"   (default "fisher")
inline_files : {name: {"kind": "ascii"|"npy_b64", "data": ...}}
                         optional input files to materialise in the job cwd
                         (n(z) tables, SRD covariance) - used for remote runs;
                         config paths whose basename matches ``name`` are patched
relocate_outputs : bool  rewrite fiducial_sacc_path / fisher.output / fid_output
                         into the job cwd (remote runs; the server's paths do not
                         exist on the node)
max_cov_return : int     return the full covariance as nested lists only when
                         n_data <= this (default 600); the diagonal is always returned

returns (mode generate): versions, n_data, data_vector[{stat,tr1,tr2,ell,cl,sigma,in_likelihood}],
    nz{tracer: {z, nz, zmean}}, cov (optional), cov_diag, sacc_path, scale_cuts, timing
returns (mode fisher): all of the above (without cov) + var_pars, param_names (after
    transforms), fiducials, fisher, fisher_with_priors|None, priors{name: sigma},
    derivatives (n_par x n_data), theory_vector, bias|None, n_evaluations, timing
"""

import base64
import copy
import io
import os
import time


def _load_config(params):
    if params.get("config") is not None:
        return copy.deepcopy(params["config"])
    path = params.get("config_path")
    if not path:
        raise ValueError("params needs 'config' (dict) or 'config_path' (YAML path)")
    try:
        from augur.parser import parse  # Jinja2 + yaml
        return parse(path)
    except ImportError:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)


def _materialise_inline_files(config, inline_files, job_dir):
    """Write inline input files to job_dir and patch matching config paths."""
    import numpy as np

    if not inline_files:
        return {}
    in_dir = os.path.join(job_dir, "augur_inputs")
    os.makedirs(in_dir, exist_ok=True)
    written = {}
    for name, spec in inline_files.items():
        dest = os.path.join(in_dir, name)
        if spec["kind"] == "ascii":
            np.savetxt(dest, np.asarray(spec["data"], dtype=float))
        elif spec["kind"] == "npy_b64":
            with open(dest, "wb") as fh:
                fh.write(base64.b64decode(spec["data"]))
        else:
            raise ValueError(f"unknown inline file kind {spec['kind']!r}")
        written[name] = dest

    def patch(section, key):
        if section in config and isinstance(config[section], dict):
            blk = config[section]
            for k in list(blk.keys()):
                if k == key and isinstance(blk[k], str) and os.path.basename(blk[k]) in written:
                    blk[k] = written[os.path.basename(blk[k])]
                elif isinstance(blk[k], dict):
                    for kk, vv in list(blk[k].items()):
                        if kk == key and isinstance(vv, str) and os.path.basename(vv) in written:
                            blk[k][kk] = written[os.path.basename(vv)]

    patch("sources", "input_file")
    patch("lenses", "input_file")
    patch("cov_options", "SRD_cov_path")
    return written


def _relocate_outputs(config, job_dir):
    out_dir = os.path.join(job_dir, "augur_out")
    os.makedirs(out_dir, exist_ok=True)
    config["fiducial_sacc_path"] = os.path.join(
        out_dir, os.path.basename(str(config.get("fiducial_sacc_path", "fiducial.sacc"))))
    if "fisher" in config:
        f = config["fisher"]
        f["output"] = os.path.join(out_dir, os.path.basename(str(f.get("output", "fisher.dat"))))
        f["fid_output"] = os.path.join(out_dir, os.path.basename(str(f.get("fid_output", "fiducials.dat"))))


def _ensure_output_dirs(config):
    """augur uses np.savetxt / save_fits without creating directories."""
    paths = [config.get("fiducial_sacc_path")]
    if "fisher" in config:
        paths += [config["fisher"].get("output"), config["fisher"].get("fid_output")]
    for p in paths:
        if p:
            d = os.path.dirname(os.path.abspath(str(p)))
            os.makedirs(d, exist_ok=True)


def _versions():
    out = {}
    for name in ("augur", "firecrown", "pyccl", "sacc", "numdifftools", "tjpcov"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "present")
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def _likelihood_ells(lk):
    """{(tr1, tr2): [ells kept in the likelihood]} - scale cuts applied by firecrown filters."""
    kept = {}
    for st in getattr(lk, "statistics", []):
        s = getattr(st, "statistic", st)
        try:
            tr1 = s.source0.sacc_tracer
            tr2 = s.source1.sacc_tracer
            ells = getattr(s, "ells", None)
            if ells is not None:
                kept[(tr1, tr2)] = [float(e) for e in ells]
        except Exception:  # noqa: BLE001
            continue
    return kept


def _data_vector_rows(S, lk):
    import numpy as np

    cov = S.covariance.covmat if S.covariance is not None else None
    diag = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov is not None else None
    kept = _likelihood_ells(lk)
    rows = []
    for dtype in S.get_data_types():
        for tr1, tr2 in S.get_tracer_combinations(data_type=dtype):
            idx = S.indices(data_type=dtype, tracers=(tr1, tr2))
            ells, cls = S.get_ell_cl(dtype, tr1, tr2)
            kept_ells = kept.get((tr1, tr2), kept.get((tr2, tr1)))
            for j, (ell, cl) in enumerate(zip(ells, cls)):
                in_lk = True
                if kept_ells is not None:
                    in_lk = bool(np.any(np.isclose(kept_ells, ell, rtol=1e-6)))
                rows.append({"stat": dtype, "tr1": tr1, "tr2": tr2, "ell": float(ell),
                             "cl": float(cl),
                             "sigma": float(diag[idx[j]]) if diag is not None else None,
                             "in_likelihood": in_lk})
    return rows, cov


def _nz_tables(S):
    import numpy as np

    out = {}
    for name in S.tracers:
        tr = S.get_tracer(name)
        z = np.asarray(tr.z, dtype=float)
        nz = np.asarray(tr.nz, dtype=float)
        zmean = float(np.average(z, weights=nz)) if nz.sum() > 0 else None
        out[name] = {"z": z.tolist(), "nz": nz.tolist(), "zmean": zmean,
                     "quantity": getattr(tr, "quantity", None)}
    return out


def _scale_cuts(S, lk, rows):
    """Per tracer pair: the highest ell kept in the likelihood (lmax after kmax conversion)."""
    kept = _likelihood_ells(lk)
    out = {}
    for (tr1, tr2), ells in kept.items():
        out[f"{tr1}-{tr2}"] = {"n_ell_kept": len(ells),
                               "lmax_kept": max(ells) if ells else None}
    return out


def main(params: dict) -> dict:
    import numpy as np
    import contextlib

    t_start = time.time()
    mode = params.get("mode", "fisher")
    job_dir = os.path.abspath(params.get("job_dir") or os.getcwd())
    config = _load_config(params)
    inline = _materialise_inline_files(config, params.get("inline_files"), job_dir)
    if params.get("relocate_outputs"):
        _relocate_outputs(config, job_dir)
    _ensure_output_dirs(config)
    max_cov_return = int(params.get("max_cov_return", 600))

    from augur.generate import generate

    # augur pops keys from the config and mutates global pyccl accuracy
    # parameters: always hand it a deep copy and keep ours pristine.
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        res = generate(copy.deepcopy(config), return_all_outputs=True)
    if len(res) == 4:
        lk, S, tools, sys_params = res
    elif len(res) == 3:  # use_sacc path: (lk, tools, sys_params)
        lk, tools, sys_params = res
        import sacc as _sacc
        S = _sacc.Sacc.load_fits(config["fiducial_sacc_path"])
    else:
        raise RuntimeError(f"unexpected augur.generate return arity {len(res)}")
    t_generate = time.time() - t0

    rows, cov = _data_vector_rows(S, lk)
    n_data = len(rows)
    out = {
        "mode": mode,
        "versions": _versions(),
        "n_data": n_data,
        "n_data_likelihood": int(len(lk.get_data_vector())),
        "data_vector": rows,
        "nz": _nz_tables(S),
        "cov_diag": (np.diag(cov).tolist() if cov is not None else None),
        "sacc_path": os.path.abspath(str(config["fiducial_sacc_path"])),
        "scale_cuts": _scale_cuts(S, lk, rows),
        "inline_files_written": inline,
        "cov_type": config.get("cov_options", {}).get("cov_type"),
        "timing_s": {"generate": t_generate},
        "job_dir": job_dir,
    }
    if cov is not None and n_data <= max_cov_return:
        out["cov"] = np.asarray(cov, dtype=float).tolist()
    if mode == "generate":
        out["timing_s"]["total"] = time.time() - t_start
        return out

    # ---- Fisher ------------------------------------------------------------
    from augur.analyze import Analyze

    fcfg = config.get("fisher")
    if not fcfg:
        raise ValueError("config has no 'fisher' section; nothing to compute")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        an = Analyze(copy.deepcopy(config), likelihood=lk, tools=tools, req_params=sys_params)
        F = an.get_fisher_matrix(save_txt=True)
    t_fisher = time.time() - t0

    var_pars = list(an.var_pars)
    names = list(var_pars)
    fid = [float(v) for v in np.asarray(an.x).ravel()]
    if getattr(an, "transform_S8", False) and "sigma8" in names:
        i = names.index("sigma8")
        names[i] = "S8"
        fid[i] = float(an.get_S8())
    if getattr(an, "transform_Omega_m", False) and "Omega_c" in names:
        i = names.index("Omega_c")
        names[i] = "Omega_m"
        fid[i] = float(an.get_Om())

    priors = {}
    if an.gprior_pars is not None:
        priors = {str(p): float(s) for p, s in zip(an.gprior_pars, an.gpriors)}
    method = an.derivative_method
    n_par = len(var_pars)
    if "5pt" in str(method):
        n_evals = 4 * n_par + 1
    elif "derivkit" in str(method):
        n_evals = 27 * n_par
    else:
        n_evals = 2 * n_par + 1

    out.update({
        "var_pars": var_pars,
        "param_names": names,
        "fiducials": fid,
        "fisher": np.asarray(F, dtype=float).tolist(),
        "fisher_with_priors": (np.asarray(an.Fij_with_gprior, dtype=float).tolist()
                               if an.Fij_with_gprior is not None else None),
        "priors": priors,
        "derivatives": np.asarray(an.derivatives, dtype=float).tolist(),
        "theory_vector": [float(v) for v in np.asarray(lk.get_data_vector()).ravel()],
        "derivative_method": str(method),
        "step": float(an.step_size),
        "n_evaluations": n_evals,
        "transform_S8": bool(getattr(an, "transform_S8", False)),
        "transform_Omega_m": bool(getattr(an, "transform_Omega_m", False)),
        "augur_output": fcfg.get("output"),
        "augur_fid_output": fcfg.get("fid_output"),
        "bias": None,
    })
    out["timing_s"]["fisher"] = t_fisher

    want_bias = params.get("include_bias")
    if want_bias is None:
        want_bias = "fisher_bias" in fcfg
    if want_bias and "fisher_bias" in fcfg:
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            bi = an.get_fisher_bias(save_txt=True)
        bias_params = dict(fcfg["fisher_bias"].get("bias_params", {}) or {})
        out["bias"] = {"shift": [float(v) for v in np.asarray(bi).ravel()],
                       "bias_params": {k: float(v) for k, v in bias_params.items()},
                       "biased_dv_file": fcfg["fisher_bias"].get("biased_dv", "") or "",
                       "delta_theory": [float(v) for v in np.asarray(an.biased_cls).ravel()]}
        out["timing_s"]["bias"] = time.time() - t0
    out["timing_s"]["total"] = time.time() - t_start
    return out
