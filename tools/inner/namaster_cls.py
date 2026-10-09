"""Inner script: pseudo-C_ell bandpowers from HEALPix maps with NaMaster
(env-kernel; see tools/envkernel.py). Only numpy + healpy + pymaster are
imported, inside functions, so it runs under any environment that carries
NaMaster (the NERSC desc-cosmology stack, a TXPipe env) and in-process here.

params (JSON):
  fields   : list of {name, spin (0|2), mask_path, map_paths ([T] or [Q, U]),
             hdf5_names (optional, TXPipe maps file: names under maps/),
             nside (needed for HDF5 maps), apodize_deg (0 = none), apotype
             (C1|C2|Smooth), purify_b, n_iter, noise: {n_gal_arcmin2, sigma_e}
             or null}
  pairs    : list of [name_a, name_b]
  binning  : {scheme: linear|log|edges, nlb, n_bins, ell_min, ell_max, edges}
  lmax     : int | null (default 3*nside - 1)
  lite     : bool  lite fields (no templates, less memory)
  work_dir : str   where the npz with windows goes
  npz_name : str | null  (null = no npz)
  return_windows : bool  also return bandpower windows inline (big)

returns: {"nside", "lmax", "ell_eff", "ell_edges", "n_bins",
          "fields": {name: {spin, fsky_mask, mean_w, mean_w2, fsky_eff,
                     noise_coupled}},
          "pairs": [{a, b, spins, components, cls: [[...] per component],
                     noise_coupled: [...] per component, noise_decoupled}],
          "npz": path | null, "elapsed_s", "n_workspaces"}
"""

import os
import time

ARCMIN2_PER_SR = (180.0 * 60.0 / 3.141592653589793) ** 2

COMPONENTS = {(0, 0): ["TT"], (0, 2): ["TE", "TB"], (2, 0): ["ET", "BT"],
              (2, 2): ["EE", "EB", "BE", "BB"]}


def read_map(path: str, hdf5_name=None, nside=None, field: int = 0):
    """HEALPix FITS (healpy) or a TXPipe-style HDF5 maps file (maps/<name>/{pixel,value})."""
    import numpy as np

    if str(path).lower().endswith((".h5", ".hdf5")):
        import h5py
        import healpy as hp

        if nside is None or hdf5_name is None:
            raise ValueError("HDF5 maps need nside and hdf5_names")
        with h5py.File(path, "r") as f:
            grp = f[f"maps/{hdf5_name}"]
            m = np.zeros(hp.nside2npix(int(nside)))
            m[grp["pixel"][:]] = grp["value"][:]
        return m
    import healpy as hp

    return np.asarray(hp.read_map(path, field=field, dtype=np.float64), dtype=float)


def make_bins(binning: dict, lmax: int, nside: int | None = None):
    import numpy as np
    import pymaster as nmt

    scheme = (binning or {}).get("scheme", "linear")
    ell_min = int((binning or {}).get("ell_min") or 2)
    # default ceiling 2*nside: above it pixelization and aliasing bias the
    # bandpowers (the 3*nside-1 limit is what the transforms allow, not what
    # is accurate); an explicit ell_max may go up to lmax
    nside = nside or (lmax + 1) // 3
    ell_max = int((binning or {}).get("ell_max") or min(lmax, 2 * nside))
    ell_max = min(ell_max, lmax)
    if scheme == "edges":
        edges = np.asarray(binning["edges"], dtype=int)
    elif scheme == "log":
        n_bins = int((binning or {}).get("n_bins") or 20)
        edges = np.unique(np.round(np.logspace(np.log10(ell_min), np.log10(ell_max + 1), n_bins + 1)).astype(int))
    else:
        nlb = int((binning or {}).get("nlb") or 20)
        edges = np.arange(ell_min, ell_max + 2, nlb)
        if edges[-1] <= ell_max:
            # leftover ells above the last full bin: a bin of their own when at
            # least half a bandpower wide, else folded into the last bin (a one-
            # or two-ell stub bin is useless and breaks TJPCov's edge detection)
            if ell_max + 1 - edges[-1] >= max(2, nlb // 2):
                edges = np.append(edges, ell_max + 1)
            else:
                edges[-1] = ell_max + 1
    edges = edges[edges <= lmax + 1]
    edges = np.unique(edges)
    if len(edges) >= 2 and np.any(np.diff(edges) < 2):
        raise ValueError(f"bandpowers narrower than 2 ells: edges {edges.tolist()}")
    if len(edges) < 2:
        raise ValueError(f"binning gives no bandpower below lmax={lmax}: {binning}")
    b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
    return b, edges


def main(params: dict) -> dict:
    import numpy as np
    import pymaster as nmt

    t0 = time.time()
    work_dir = os.path.abspath(params.get("work_dir") or ".")
    os.makedirs(work_dir, exist_ok=True)
    specs = {f["name"]: f for f in params["fields"]}
    lite = bool(params.get("lite", True))

    masks = {}
    fields = {}
    field_info = {}
    nside = None
    for name, f in specs.items():
        hdf5 = f.get("hdf5_names") or []
        mask = read_map(f["mask_path"], hdf5[0] if hdf5 else None, f.get("nside"))
        import healpy as hp

        this_nside = hp.npix2nside(mask.size)
        nside = nside or this_nside
        if this_nside != nside:
            raise ValueError(f"field {name}: nside {this_nside} differs from {nside}")
        apod = float(f.get("apodize_deg") or 0.0)
        if apod > 0:
            mask = nmt.mask_apodization(mask, apod, apotype=f.get("apotype") or "C1")
        maps = [read_map(p, hdf5[i + 1] if len(hdf5) > i + 1 else None, f.get("nside"))
                for i, p in enumerate(f["map_paths"])]
        spin = int(f.get("spin", 0))
        if (spin == 2 and len(maps) != 2) or (spin == 0 and len(maps) != 1):
            raise ValueError(f"field {name}: spin {spin} needs {'2 maps (Q, U)' if spin == 2 else '1 map'}")
        masks[name] = mask
        mean_w = float(mask.mean())
        mean_w2 = float((mask ** 2).mean())
        noise = f.get("noise") or None
        n_coupled = None
        if noise and noise.get("n_gal_arcmin2"):
            nbar = float(noise["n_gal_arcmin2"]) * ARCMIN2_PER_SR
            var = (float(noise.get("sigma_e") or 0.26) ** 2) if spin == 2 else 1.0
            n_coupled = mean_w2 * var / nbar
        field_info[name] = {"spin": spin, "fsky_mask": float((mask > 0).mean()), "mean_w": mean_w,
                            "mean_w2": mean_w2, "fsky_eff": (mean_w ** 2 / mean_w2) if mean_w2 > 0 else 0.0,
                            "noise_coupled": n_coupled, "apodize_deg": apod}
        fields[name] = (mask, maps, spin, f)

    lmax = int(params.get("lmax") or (3 * nside - 1))
    lmax = min(lmax, 3 * nside - 1)
    bins, edges = make_bins(params.get("binning") or {}, lmax, nside)
    ell_eff = bins.get_effective_ells()
    lmax = int(bins.lmax)  # NaMaster requires fields and bins to share lmax (= last edge - 1)

    nmt_fields = {}
    for name, (mask, maps, spin, f) in fields.items():
        kw = {"spin": spin, "lite": lite, "lmax": lmax}
        if f.get("purify_b") and spin == 2:
            kw["purify_b"] = True
        if f.get("n_iter") is not None:
            kw["n_iter"] = int(f["n_iter"])
        nmt_fields[name] = nmt.NmtField(mask, maps, **kw)

    workspaces: dict = {}
    windows: dict = {}
    out_pairs = []

    def mask_key(name):
        f = specs[name]
        return (os.path.abspath(f["mask_path"]), float(f.get("apodize_deg") or 0.0), f.get("apotype") or "C1")

    for a, b in params["pairs"]:
        sa, sb = field_info[a]["spin"], field_info[b]["spin"]
        if sa > sb:  # canonical order: lower spin first (TE/TB rather than ET/BT)
            a, b, sa, sb = b, a, sb, sa
        wkey = (mask_key(a), sa, mask_key(b), sb)
        if wkey not in workspaces:
            workspaces[wkey] = nmt.NmtWorkspace.from_fields(nmt_fields[a], nmt_fields[b], bins)
        w = workspaces[wkey]
        comps = COMPONENTS[(sa, sb)]
        cl_coupled = np.asarray(nmt.compute_coupled_cell(nmt_fields[a], nmt_fields[b]), dtype=float)
        n_cl = cl_coupled.shape[0]
        noise = np.zeros_like(cl_coupled)
        if a == b and field_info[a]["noise_coupled"]:
            nc = field_info[a]["noise_coupled"]
            if sa == 0:
                noise[0, :] = nc
            else:
                noise[0, :] = nc   # EE
                noise[3, :] = nc   # BB
        cl_dec = np.asarray(w.decouple_cell(cl_coupled - noise), dtype=float)
        noise_dec = np.asarray(w.decouple_cell(noise), dtype=float) if noise.any() else np.zeros_like(cl_dec)
        win = np.asarray(w.get_bandpower_windows(), dtype=float)   # (n_cl, n_bpw, n_cl, lmax+1)
        diag = np.stack([win[i, :, i, :] for i in range(n_cl)])     # (n_cl, n_bpw, lmax+1)
        windows[f"{a}__{b}"] = diag
        out_pairs.append({"a": a, "b": b, "spins": [sa, sb], "components": comps,
                          "cls": cl_dec.tolist(), "noise_decoupled": noise_dec.tolist(),
                          "noise_coupled": [float(noise[i, 0]) for i in range(n_cl)],
                          "n_bins": int(cl_dec.shape[1])})

    npz_path = None
    if params.get("npz_name"):
        npz_path = os.path.join(work_dir, params["npz_name"])
        payload = {"ell_eff": ell_eff, "ell_edges": np.asarray(edges, dtype=float), "lmax": lmax, "nside": nside}
        for key, arr in windows.items():
            payload[f"win_{key}"] = arr
        for p in out_pairs:
            payload[f"cl_{p['a']}__{p['b']}"] = np.asarray(p["cls"])
        np.savez_compressed(npz_path, **payload)
    out = {"nside": int(nside), "lmax": lmax, "ell_eff": [float(x) for x in ell_eff],
           "ell_edges": [int(x) for x in edges], "n_bins": int(len(ell_eff)), "fields": field_info,
           "pairs": out_pairs, "npz": npz_path, "elapsed_s": round(time.time() - t0, 1),
           "n_workspaces": len(workspaces), "pymaster_version": getattr(nmt, "__version__", "?")}
    if params.get("return_windows"):
        out["windows"] = {k: v.tolist() for k, v in windows.items()}
    return out
