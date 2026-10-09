"""Inner script: compute a TJPCov covariance for a sacc file (env-kernel; see
tools/envkernel.py). Runs identically in-process (local) and under a facility
environment (remote: the NERSC desc-python / desc-cosmology stacks carry
tjpcov). Only numpy + sacc + pyccl + tjpcov are imported, inside functions.

params (JSON):
  tjpcov          : dict  the `tjpcov:` section of a TJPCov config WITHOUT
                          sacc_file / cosmo (added here): cov_type, fsky,
                          Ngal_<tracer>, sigma_e_<tracer>, bias_<tracer>, IA,
                          mask_file / mask_names / nside (NaMaster types) ...
  extra_sections  : dict  other top-level sections (ProjectedReal, HOD, SSC,
                          cNG, NaMaster, cache) passed through verbatim
  cosmo_kwargs    : dict  pyccl.Cosmology keyword arguments (plain values)
  sacc_path       : str   local path of the input sacc (local runs)
  sacc_b64 / sacc_name     the input sacc inline (remote runs)
  output_name     : str   name of the sacc written with the covariance
                          (.hdf5 or .fits by extension)
  work_dir        : str   where outputs go ("." remotely)
  save_terms      : bool  also write one sacc per covariance term
  inline_max_bytes: int   return output files base64 when smaller than this
                          (remote runs; 0 = never)

returns: {"output_file", "term_files": {term: path}, "n_data", "terms":
          {term: {diag_mean, diag_min, diag_max, frac_of_total_diag}},
          "covariance": {diag_min, diag_max, n_nonpositive_diag,
          condition_number, min_eigenvalue, positive_definite,
          max_abs_offdiag_corr}, "snr_total", "elapsed_s", "warnings",
          "inline_files": {name: b64}}
"""

import base64
import os
import time
import warnings


def _materialize_sacc(params: dict) -> str:
    if params.get("sacc_path") and not params.get("sacc_b64"):
        return str(params["sacc_path"])
    path = os.path.abspath(params.get("sacc_name") or "input.hdf5")
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(params["sacc_b64"]))
    return path


def _load_sacc(path: str):
    import sacc

    return sacc.Sacc.load(path)


def estimate_edges(values, weight):
    """Bandpower edges from dense (mode-coupled) windows: half-maximum extent of
    each column, tiled so neighbouring bins touch. (nv,) values, (nv, nbpw) weight."""
    import numpy as np

    values = np.asarray(values, float)
    weight = np.asarray(weight, float)
    lo, hi = [], []
    for k in range(weight.shape[1]):
        w = weight[:, k]
        m = w >= 0.5 * w.max()
        lo.append(float(values[m].min()))
        hi.append(float(values[m].max()))
    edges = [lo[0]]
    for k in range(1, len(lo)):
        edges.append(0.5 * (hi[k - 1] + lo[k]) + 0.5)
    edges.append(hi[-1] + 1.0)
    return [int(round(e)) for e in edges]


def with_tophat_windows(s, edges):
    """Copy of a harmonic sacc whose windows are replaced by top-hat bandpowers
    on `edges` (what TJPCov's f_sky binning expects); edges=None drops windows.
    Data order is untouched (tags are rewritten in place on the copy)."""
    import numpy as np
    import sacc

    s2 = s.copy()
    groups: dict = {}
    for i, d in enumerate(s2.data):
        groups.setdefault((d.data_type, tuple(d.tracers)), []).append(i)
    if edges is None:
        for d in s2.data:
            d.tags.pop("window", None)
            d.tags.pop("window_ind", None)
        return s2
    edges = np.asarray(edges, float)
    # TJPCov takes the LAST ell carrying weight as the exclusive upper edge of
    # the final bandpower, so the final top-hat extends one ell past its edge
    # (otherwise that bin loses its last multipole).
    values = np.arange(int(edges[0]), int(edges[-1]) + 1)
    for idx in groups.values():
        ells = np.array([float(s2.data[i].get_tag("ell")) for i in idx])
        weight = np.zeros((values.size, len(idx)))
        for k, ell in enumerate(ells):
            b = int(np.searchsorted(edges, ell, side="right") - 1)
            b = min(max(b, 0), len(edges) - 2)
            hi = edges[b + 1] + (1 if b == len(edges) - 2 else 0)
            inside = (values >= edges[b]) & (values < hi)
            weight[inside, k] = 1.0 / max(inside.sum(), 1)
        win = sacc.BandpowerWindow(values, weight)
        for k, i in enumerate(idx):
            s2.data[i].tags["window"] = win
            s2.data[i].tags["window_ind"] = k
    return s2


def _save_sacc(s, path: str) -> None:
    if path.lower().endswith((".fits", ".fit")):
        s.save_fits(path, overwrite=True)
    else:
        s.save_hdf5(path, overwrite=True)


def _real_space_covariance(config: dict, s):
    """RealGaussianFsky assembled block by block in the sacc's own order.

    TJPCov 0.5.1's CovarianceCalculator path for real space (a) assumes xi_+
    and xi_- are interleaved per theta inside a tracer pair and (b) never
    fills the xi_- auto block (its `auto` shortcut mis-indexes), so the matrix
    comes out singular for TXPipe/firecrown-ordered files. Its public
    per-block method is correct; this uses it directly.
    """
    import numpy as np
    from tjpcov.covariance_gaussian_fsky import RealGaussianFsky

    cov_obj = RealGaussianFsky(config)
    groups = []
    for dt in s.get_data_types():
        for combo in s.get_tracer_combinations(dt):
            groups.append((dt, list(combo), np.asarray(s.indices(dt, combo), dtype=int)))

    def pm(dt):
        return "minus" if "minus" in dt else "plus"

    n = len(s.data)
    cov = np.zeros((n, n))
    for gi, (dt1, c1, ix1) in enumerate(groups):
        for dt2, c2, ix2 in groups[gi:]:
            block = np.asarray(cov_obj.get_covariance_block(c1, c2, pm(dt1), pm(dt2)), dtype=float)
            if block.shape != (len(ix1), len(ix2)):
                raise ValueError(f"TJPCov real-space block {dt1}{c1} x {dt2}{c2} has shape {block.shape}, "
                                 f"expected {(len(ix1), len(ix2))}: every data type must share one theta grid")
            cov[np.ix_(ix1, ix2)] = block
            cov[np.ix_(ix2, ix1)] = block.T
    return {"gauss": cov}, cov


def cov_stats(cov) -> dict:
    """Diagnostics of a dense covariance (mirrors tools.sacc_tools._cov_info,
    re-implemented here because inner scripts may not import the server)."""
    import numpy as np

    cov = np.asarray(cov, dtype=float)
    d = np.diag(cov)
    info = {"shape": list(cov.shape), "diag_min": float(d.min()), "diag_max": float(d.max()),
            "n_nonpositive_diag": int((d <= 0).sum()), "condition_number": None,
            "min_eigenvalue": None, "positive_definite": None}
    if cov.shape[0] <= 3000:
        try:
            info["condition_number"] = float(np.linalg.cond(cov))
            w = np.linalg.eigvalsh(cov)
            info["min_eigenvalue"] = float(w.min())
            # Strict, like firecrown's Cholesky: a matrix spanning so many
            # decades that its smallest eigenvalue rounds to <= 0 (xi_- at small
            # theta with a low lmax, an empty bandpower) is unusable as it is.
            info["positive_definite"] = bool(w.min() > 0 and info["n_nonpositive_diag"] == 0)
            info["ill_conditioned"] = bool(not np.isfinite(info["condition_number"]) or info["condition_number"] > 1e15)
        except np.linalg.LinAlgError:
            pass
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(np.sqrt(np.abs(d)), np.sqrt(np.abs(d)))
    np.fill_diagonal(corr, 0.0)
    info["max_abs_offdiag_corr"] = float(np.nanmax(np.abs(corr))) if cov.shape[0] > 1 else 0.0
    return info


def main(params: dict) -> dict:
    import numpy as np
    import pyccl as ccl
    from tjpcov.covariance_calculator import CovarianceCalculator

    t0 = time.time()
    work_dir = os.path.abspath(params.get("work_dir") or ".")
    os.makedirs(work_dir, exist_ok=True)
    sacc_path = _materialize_sacc(params)
    s = _load_sacc(sacc_path)

    # TJPCov's f_sky types read the bandpower EDGES from the sacc windows as
    # "where the weight is nonzero" - wrong for dense NaMaster windows. For
    # those types the wrapper asks for top-hat windows (edges from the sacc
    # metadata, else estimated from the dense windows); NaMaster types keep
    # the real windows. The covariance is attached to the ORIGINAL file.
    mode = params.get("window_mode", "keep")
    edges = params.get("ell_edges")
    s_in = s
    if mode in ("tophat", "drop"):
        if mode == "tophat" and not edges:
            for d in s.data:
                if d.tags.get("window") is not None:
                    try:
                        idx = s.indices(d.data_type, tuple(d.tracers))
                        bpw = s.get_bandpower_windows(idx)
                        if bpw is not None:
                            edges = estimate_edges(bpw.values, bpw.weight)
                    except Exception:  # noqa: BLE001 - fall back to dropping
                        edges = None
                    break
        s_in = with_tophat_windows(s, edges if (mode == "tophat" and edges) else None)

    section = dict(params.get("tjpcov") or {})
    section["sacc_file"] = s_in                    # object: works for HDF5 and FITS
    section["cosmo"] = ccl.Cosmology(**dict(params.get("cosmo_kwargs") or {}))
    section["outdir"] = work_dir                   # NaMaster types cache blocks here
    section["use_mpi"] = False                     # one process (the engine job or the server)
    config = {"tjpcov": section}
    config.update(dict(params.get("extra_sections") or {}))
    cov_types = section.get("cov_type") or []
    cov_types = [cov_types] if isinstance(cov_types, str) else list(cov_types)

    # Mask-based Fourier types build NaMaster workspaces themselves and need an
    # NmtBin (binning_info) whose lmax the mask fields share (NaMaster 3 refuses
    # mismatched lmax): derive both from the bandpower edges.
    if any(t in ("FourierGaussianNmt", "FourierSSCHaloModel", "FouriercNGHaloModel") for t in cov_types):
        if not edges:
            raise ValueError("NaMaster covariance types need the bandpower edges: give a sacc with windows "
                             "(namaster_compute_cls) or binning metadata.")
        import pymaster as nmt

        e = [int(x) for x in edges]
        bins = nmt.NmtBin.from_edges(e[:-1], e[1:])
        section["binning_info"] = bins
        nm = config.setdefault("NaMaster", {})
        nm.setdefault("f", {})["lmax"] = int(bins.lmax)
        for k in ("w", "cw"):
            nm.setdefault(k, {})

    caught: list = []
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        try:
            if cov_types == ["RealGaussianFsky"]:
                terms, cov = _real_space_covariance(config, s_in)
            else:
                calc = CovarianceCalculator(config)
                terms = calc.get_covariance_terms()
                cov = calc.get_covariance()
        except TypeError as exc:
            if "NmtCovarianceWorkspace" in str(exc) or "NmtWorkspace" in str(exc):
                raise RuntimeError(
                    "TJPCov 0.5.1's NaMaster covariance (FourierGaussianNmt) calls the NaMaster 2 API "
                    "(NmtCovarianceWorkspace() without fields), which pymaster 3 removed - it cannot run "
                    "in an environment with pymaster >= 3. Use FourierGaussianFsky with f_sky,eff from the "
                    "sacc metadata, or run this config under an environment with NaMaster 2.x "
                    "(e.g. a facility stack) via env_setup.") from exc
            raise
        # sacc warns "Empty index selected" for every B-mode type absent from
        # the file while TJPCov scans its concise data types: benign.
        caught = sorted({str(w.message).strip() for w in wrec if "Empty index" not in str(w.message)})

    cov = np.asarray(cov, dtype=float)
    n = len(s.data)
    if cov.shape != (n, n):
        raise ValueError(f"TJPCov returned a {cov.shape} covariance for {n} data points")
    out_name = params.get("output_name") or "cls_cov.hdf5"
    out_path = os.path.join(work_dir, out_name)
    s_out = s.copy()
    s_out.add_covariance(cov, overwrite=True)
    s_out.metadata["tjpcov_cov_types"] = ",".join(section.get("cov_type", []) if isinstance(
        section.get("cov_type"), list) else [str(section.get("cov_type"))])
    _save_sacc(s_out, out_path)

    term_files = {}
    term_info = {}
    total_diag = np.diag(cov)
    for name, mat in terms.items():
        mat = np.asarray(mat, dtype=float)
        d = np.diag(mat)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = float(np.nanmean(d / total_diag)) if np.all(total_diag > 0) else None
        term_info[name] = {"diag_mean": float(d.mean()), "diag_min": float(d.min()),
                           "diag_max": float(d.max()), "frac_of_total_diag": frac}
        if params.get("save_terms", True) and len(terms) > 1:
            stem, ext = os.path.splitext(out_name)
            tpath = os.path.join(work_dir, f"{stem}_{name}{ext}")
            st = s.copy()
            st.add_covariance(mat, overwrite=True)
            _save_sacc(st, tpath)
            term_files[name] = tpath

    data = np.asarray(s.mean, dtype=float)
    try:
        snr = float(np.sqrt(data @ np.linalg.solve(cov, data)))
    except np.linalg.LinAlgError:
        snr = None

    inline = {}
    cap = int(params.get("inline_max_bytes") or 0)
    if cap:
        for p in [out_path] + list(term_files.values()):
            if os.path.getsize(p) <= cap:
                with open(p, "rb") as fh:
                    inline[os.path.basename(p)] = base64.b64encode(fh.read()).decode("ascii")

    return {"output_file": out_path, "term_files": term_files, "n_data": n,
            "terms": term_info, "covariance": cov_stats(cov), "snr_total": snr,
            "elapsed_s": round(time.time() - t0, 1), "warnings": caught,
            "inline_files": inline, "tjpcov_version": _version(),
            "window_mode": mode, "ell_edges_used": [int(e) for e in edges] if (mode == "tophat" and edges) else None}


def _version() -> str:
    try:
        import importlib.metadata as md

        return md.version("tjpcov")
    except Exception:  # noqa: BLE001
        return "unknown"
