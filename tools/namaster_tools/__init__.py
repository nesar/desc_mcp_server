"""NaMaster tools: pseudo-C_ell power spectra from masked HEALPix maps.

NaMaster (github.com/LSSTDESC/NaMaster, python package `pymaster`) is the
DESC mode-coupling (MASTER) estimator used by TXPipe's Fourier-space
measurements and TJPCov's mask-coupled covariances. This family covers the
measurement side: mask properties, a simulated test bed (Gaussian maps from
CCL theory), and bandpowers of spin-0/spin-2 fields written as a sacc file
with bandpower windows and coupled-noise metadata - ready for
tjpcov_compute_covariance (FourierGaussianNmt / Fsky) and firecrown.

Every tool writes NEW files into output_dir. Package name is
`namaster_tools` so it never shadows the library.
"""

import contextlib
import json
import sys
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, validate_call

from ..common import ArtifactResult, param_slug, read_csv, resolve_outdir, write_csv, write_json
from ..cosmology import CosmologyParams

__all__ = ["namaster_mask_properties", "namaster_simulate_maps", "namaster_compute_cls"]

CONVENTIONS = {
    "maps": "HEALPix RING maps in FITS (healpy) - one map for spin-0 fields (density contrast, CMB "
            "convergence), two (Q, U = gamma_1, gamma_2) for spin-2 shear; or a TXPipe-style HDF5 maps "
            "file (maps/<name>/{pixel,value}, needs nside and hdf5_names)",
    "masks": "weight maps (0-1); apodization with NaMaster's C1/C2/Smooth kernels in degrees; f_sky,eff "
             "= <w>^2/<w^2> is the number TJPCov/sacc Gaussian covariances want",
    "bandpowers": "NmtBin from edges (linear width nlb, log n_bins, or explicit edges); effective ells and "
                  "the bandpower windows go into the sacc (BandpowerWindow per data point)",
    "components": "spin0 x spin0: TT; spin0 x spin2: TE, TB; spin2 x spin2: EE, EB, BE, BB; written as "
                  "galaxy_density_cl, galaxy_shearDensity_cl_e/_b (shear tracer first), galaxy_shear_cl_ee/eb/be/bb",
    "noise": "white shot/shape noise is subtracted analytically for auto-spectra when n_gal_arcmin2 (+ sigma_e) "
             "is given: coupled N = <w^2> sigma_e^2 / n_bar (shear, per component) or <w^2> / n_bar (density), "
             "n_bar in sr^-1; the value is stored as tracer metadata n_ell_coupled for TJPCov",
    "tracers": "sacc tracers are NZ (quantity galaxy_shear / galaxy_density) when an n(z) CSV is given per "
               "field - that is what firecrown and TJPCov need - else Misc tracers with the field metadata",
}
CAVEATS = [
    "Bandpowers above ell ~ 2*nside are biased by pixelization and aliasing (at nside 32 the top bin near "
    "3*nside came out 90% high on a simulation): the default ell_max is 2*nside; raise nside, not ell_max.",
    "Mode coupling scales as lmax^3: nside 256 (lmax 767) takes seconds per workspace, nside 1024 minutes, "
    "nside 4096 hours - dispatch large measurements (env_setup with NaMaster, map paths on the facility).",
    "Workspaces are shared by fields with the same mask (path + apodization) and spins; different masks "
    "per field multiply the cost.",
    "Noise subtraction assumes uniform white noise under the mask; for real catalogs measure the noise "
    "(random rotations / split maps) and pass it as explicit noise, or leave n_gal unset and subtract later.",
    "B-mode purification (purify_b) needs an apodized mask; it increases the variance of EE slightly and is "
    "off by default.",
    "The sacc from namaster_compute_cls has NO covariance: next is tjpcov_generate_config (FourierGaussianNmt "
    "with the same masks, or FourierGaussianFsky with f_sky,eff) and tjpcov_compute_covariance.",
    "Remote runs return the bandpowers; the windows come back only when return_windows is set (large) - "
    "without them the sacc is written without windows.",
]

DISPATCH_KERNELS = {
    "namaster_cls": {
        "function": "envkernel.run_in_env", "inner": "namaster_cls",
        "params": {"fields": [{"name": "src0", "spin": 2, "mask_path": "<facility path>",
                               "map_paths": ["<Q>", "<U>"], "apodize_deg": 0.5, "apotype": "C1",
                               "noise": {"n_gal_arcmin2": 10.0, "sigma_e": 0.26}}],
                   "pairs": [["src0", "src0"]], "binning": {"scheme": "linear", "nlb": 20, "ell_min": 20},
                   "lmax": None, "lite": True, "work_dir": ".", "npz_name": "namaster_cls.npz",
                   "return_windows": False},
        "env_setup_required": True, "suitable_envs": ["desc-cosmology", "user TXPipe env"],
        "duration_hint_s": 3600,
        "returns": "effective ells, bandpowers per pair/component, coupled noise, field f_sky stats (+ windows on request)",
    },
}

Kind = Literal["density", "shear", "cmb_convergence"]
SPIN = {"density": 0, "shear": 2, "cmb_convergence": 0}
_SACC_TYPES = {
    ("density", "density"): {"TT": "galaxy_density_cl"},
    ("density", "shear"): {"TE": "galaxy_shearDensity_cl_e", "TB": "galaxy_shearDensity_cl_b"},
    ("shear", "shear"): {"EE": "galaxy_shear_cl_ee", "EB": "galaxy_shear_cl_eb", "BE": "galaxy_shear_cl_be",
                         "BB": "galaxy_shear_cl_bb"},
    ("cmb_convergence", "cmb_convergence"): {"TT": "cmb_convergence_cl"},
    ("cmb_convergence", "density"): {"TT": "cmbGalaxy_convergenceDensity_cl"},
    ("cmb_convergence", "shear"): {"TE": "cmbGalaxy_convergenceShear_cl_e", "TB": "cmbGalaxy_convergenceShear_cl_b"},
}
# sacc tracer order inside a data type: shear first for shearDensity, cmb first for cmbGalaxy
_SACC_ORDER = {"cmb_convergence": 0, "shear": 1, "density": 2}


class FieldSpec(BaseModel):
    """One masked map field."""

    model_config = {"extra": "forbid"}

    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", description="Tracer name (src0 / lens0 for firecrown).")]
    kind: Annotated[Kind, Field(description="density (spin 0), shear (spin 2, Q/U maps), cmb_convergence (spin 0).")]
    mask_path: Annotated[str, Field(min_length=1, description="HEALPix mask/weight map (FITS) or TXPipe HDF5 maps file.")]
    map_paths: Annotated[list[str], Field(min_length=1, max_length=2, description="[map] for spin 0, [Q, U] for shear.")]
    hdf5_names: Annotated[list[str] | None, Field(description="HDF5 maps files: names under maps/ for [mask, map(s)].")] = None
    nside: Annotated[int | None, Field(description="nside of HDF5 maps (FITS maps carry it).")] = None
    apodize_deg: Annotated[float, Field(ge=0.0, le=30.0, description="Apodization scale in degrees (0 = none).")] = 0.0
    apotype: Literal["C1", "C2", "Smooth"] = "C1"
    purify_b: bool = False
    n_iter: Annotated[int | None, Field(ge=0, le=10)] = None
    n_gal_arcmin2: Annotated[float | None, Field(gt=0.0, description="Number density for analytic white-noise subtraction (auto-spectra) and for the n_ell_coupled metadata.")] = None
    sigma_e: Annotated[float, Field(gt=0.0, le=1.0, description="Per-component shape noise (shear).")] = 0.26
    nz_csv: Annotated[str | None, Field(description="CSV with z and n(z) columns -> NZ tracer in the sacc (firecrown/TJPCov need it).")] = None
    nz_column: Annotated[str | None, Field(description="Column of nz_csv holding n(z) (default: 'nz', else the 2nd column).")] = None


class BinningSpec(BaseModel):
    model_config = {"extra": "forbid"}

    scheme: Literal["linear", "log", "edges"] = "linear"
    nlb: Annotated[int, Field(ge=1, le=500, description="Bandpower width (linear).")] = 20
    n_bins: Annotated[int, Field(ge=2, le=200, description="Number of bandpowers (log).")] = 20
    ell_min: Annotated[int, Field(ge=2, le=10000)] = 2
    ell_max: Annotated[int | None, Field(ge=10, le=20000, description="Default 2*nside (bandpowers above it are biased by pixelization/aliasing); the hard limit is 3*nside - 1.")] = None
    edges: Annotated[list[int] | None, Field(description="Explicit bandpower edges (scheme='edges').")] = None


class SimTracer(BaseModel):
    """A tracer to simulate: Gaussian n(z) or a CSV, plus its noise level."""

    model_config = {"extra": "forbid"}

    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
    kind: Literal["density", "shear"]
    z_mean: Annotated[float, Field(gt=0.0, le=5.0)] = 0.8
    z_sigma: Annotated[float, Field(gt=0.0, le=2.0)] = 0.2
    nz_csv: Annotated[str | None, Field(description="Use this n(z) CSV instead of the Gaussian (columns z, nz or nz_column).")] = None
    nz_column: str | None = None
    bias: Annotated[float, Field(gt=0.0, le=5.0, description="Linear galaxy bias (density).")] = 1.5
    n_gal_arcmin2: Annotated[float, Field(gt=0.0, description="Number density; sets the white noise.")] = 10.0
    sigma_e: Annotated[float, Field(gt=0.0, le=1.0)] = 0.26


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _read_nz(path: str, column: str | None) -> tuple[np.ndarray, np.ndarray]:
    _, cols = read_csv(path)
    zc = next((c for c in cols if c.lower() == "z"), None)
    if zc is None:
        raise ValueError(f"{path}: no 'z' column")
    if column:
        if column not in cols:
            raise ValueError(f"{path}: no column {column!r}; has {list(cols)}")
        nc = column
    else:
        nc = "nz" if "nz" in cols else next(c for c in cols if c != zc)
    return np.asarray(cols[zc], float), np.asarray(cols[nc], float)


def _run_inner(params: dict, env_setup: str | None, duration: int) -> tuple[dict, str]:
    from mcp_server.dispatch import remote_site, run_env_kernel  # lazy: server-only

    site = remote_site()
    if site:
        if not env_setup:
            raise ValueError(f"Dispatch is set to {site}: this call needs env_setup (a facility environment "
                             "with NaMaster, e.g. the NERSC desc-cosmology stack or your TXPipe env); map and "
                             "mask paths must be facility paths.")
        p = dict(params, work_dir=".")
        job = run_env_kernel(env_setup, "namaster_cls", p, duration=duration)
        return job["result"]["result"], job.get("host", site)
    import importlib

    mod = importlib.import_module("tools.inner.namaster_cls")
    with contextlib.redirect_stdout(sys.stderr):
        return mod.main(params), "local"


def _mollview(path: Path, m: np.ndarray, title: str, **kw) -> None:
    import healpy as hp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import rc_params

    with plt.rc_context(rc_params()):
        fig = plt.figure(figsize=(7.2, 4.2))
        hp.mollview(m, title=title, fig=fig.number, cmap=kw.get("cmap", "viridis"),
                    min=kw.get("vmin"), max=kw.get("vmax"), cbar=True)
        hp.graticule(dpar=30, dmer=60, alpha=0.3)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def _fsky_cap_mask(nside: int, f_sky: float) -> np.ndarray:
    """Polar cap of sky fraction f_sky (ring pixels with theta < theta_c)."""
    import healpy as hp

    theta_c = np.arccos(1.0 - 2.0 * f_sky)
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return (theta < theta_c).astype(float)


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
@validate_call
def namaster_mask_properties(
    output_dir: Annotated[str, Field(min_length=1)],
    mask_path: Annotated[str, Field(min_length=1, description="HEALPix mask/weight map (FITS) or TXPipe HDF5 maps file.")],
    hdf5_name: Annotated[str | None, Field(description="HDF5 maps file: the mask's name under maps/.")] = None,
    nside: Annotated[int | None, Field(description="nside of an HDF5 mask.")] = None,
    apodize_deg: Annotated[float, Field(ge=0.0, le=30.0, description="Write an apodized copy with this scale (degrees); 0 = none.")] = 0.0,
    apotype: Literal["C1", "C2", "Smooth"] = "C1",
    plot: bool = True,
) -> ArtifactResult:
    """Characterise a mask: nside, area, f_sky (raw and effective <w>^2/<w^2>), weight statistics; optionally write an apodized copy.

    f_sky,eff is the number to pass to sacc_attach_gaussian_covariance /
    tjpcov_generate_config(f_sky=...). Apodization (C1: cosine taper of the
    given width, C2: smoother, Smooth: Gaussian) reduces mode coupling
    leakage and is required for B-mode purification; it lowers the
    effective area. Writes <stem>_mask_stats.json, <stem>_apod<deg>.fits
    when apodize_deg > 0, and a Mollweide PNG.
    """
    import healpy as hp

    from ..inner.namaster_cls import read_map

    p = Path(mask_path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"mask not found: {p}")
    outdir = resolve_outdir(output_dir)
    m = read_map(str(p), hdf5_name, nside)
    ns = hp.npix2nside(m.size)
    stem = p.stem.replace(".fits", "")

    def stats(w):
        mean_w, mean_w2 = float(w.mean()), float((w ** 2).mean())
        return {"fsky_raw": float((w > 0).mean()), "mean_w": mean_w, "mean_w2": mean_w2,
                "fsky_eff": (mean_w ** 2 / mean_w2) if mean_w2 > 0 else 0.0,
                "area_deg2": float((w > 0).mean() * 41252.96), "w_min": float(w.min()), "w_max": float(w.max()),
                "n_pix_nonzero": int((w > 0).sum())}

    info = {"mask": str(p), "nside": int(ns), "npix": int(m.size), "pixel_area_arcmin2": float(hp.nside2pixarea(ns, degrees=True) * 3600),
            "raw": stats(m)}
    files = []
    if apodize_deg > 0:
        import pymaster as nmt

        ma = nmt.mask_apodization(m, apodize_deg, apotype=apotype)
        info["apodized"] = {"apodize_deg": apodize_deg, "apotype": apotype, **stats(ma)}
        apath = outdir / f"{stem}_apod{apodize_deg:g}{apotype}.fits"
        hp.write_map(str(apath), ma, overwrite=True, dtype=np.float64)
        files.append(str(apath))
        shown, label = ma, f"apodized {apodize_deg:g} deg {apotype}"
    else:
        shown, label = m, "raw"
    jpath = outdir / f"{stem}_mask_stats.json"
    write_json(jpath, info)
    files.append(str(jpath))
    if plot:
        ppath = outdir / f"{stem}_mask.png"
        _mollview(ppath, shown, f"{p.name} ({label})", vmin=0, vmax=1)
        files.append(str(ppath))
    r = info.get("apodized") or info["raw"]
    msg = (f"{p.name}: nside {ns}, f_sky raw {info['raw']['fsky_raw']:.4f}, "
           f"f_sky,eff {info['raw']['fsky_eff']:.4f} ({info['raw']['area_deg2']:.0f} deg^2)"
           + (f"; apodized {apodize_deg:g} deg {apotype}: f_sky,eff {r['fsky_eff']:.4f}" if apodize_deg > 0 else "") + ".")
    return ArtifactResult(status="success", files=files, message=msg, metadata=info)


@validate_call
def namaster_simulate_maps(
    output_dir: Annotated[str, Field(min_length=1)],
    tracers: Annotated[list[SimTracer], Field(min_length=1, description="Tracers to simulate, e.g. [{name:'src0', kind:'shear', z_mean:0.9, z_sigma:0.25, n_gal_arcmin2:8, sigma_e:0.26}, {name:'lens0', kind:'density', z_mean:0.5, z_sigma:0.1, bias:1.6, n_gal_arcmin2:4}].")],
    nside: Annotated[int, Field(ge=16, le=2048, description="HEALPix resolution (64-256 for quick tests; cost grows as nside^3 downstream).")] = 128,
    f_sky: Annotated[float, Field(gt=0.0, le=1.0, description="Sky fraction of the polar-cap mask (ignored with mask_path).")] = 0.25,
    mask_path: Annotated[str | None, Field(description="Use this HEALPix mask (same nside) instead of the polar cap.")] = None,
    apodize_deg: Annotated[float, Field(ge=0.0, le=30.0, description="Apodize the written mask (degrees; 0 = binary).")] = 0.0,
    cosmology: Annotated[CosmologyParams | None, Field(description="Input cosmology for the theory C_ell (None = vanilla LCDM).")] = None,
    add_noise: Annotated[bool, Field(description="Add white shot/shape noise from n_gal_arcmin2 (and sigma_e).")] = True,
    seed: Annotated[int, Field(ge=0)] = 0,
    lmax: Annotated[int | None, Field(description="Multipole cut of the realization (default 3*nside - 1).")] = None,
    name: Annotated[str, Field(min_length=1)] = "sim",
) -> ArtifactResult:
    """Simulate correlated Gaussian HEALPix maps (density contrast, shear Q/U) from CCL theory C_ell for a set of tracers, with a mask and white noise - a self-contained test bed for namaster_compute_cls.

    All tracer pairs are correlated through their CCL cross-spectra
    (healpy synalm); shear maps come from the E-mode alm (B = 0); density
    uses a constant linear bias. Noise: Gaussian per pixel with variance
    1/(n_bar Omega_pix) (density) or sigma_e^2/(n_bar Omega_pix) per
    component (shear). Writes <name>_<tracer>.fits (1 or 2 maps), the mask,
    nz_<tracer>.csv, theory_cls.csv (input C_ell per pair), a Mollweide
    PNG, and <name>_fields.json - the `fields_json` input of
    namaster_compute_cls, already carrying masks, maps, n(z) and noise.
    """
    import healpy as hp
    import pyccl as ccl

    outdir = resolve_outdir(output_dir)
    spec = cosmology or CosmologyParams()
    cosmo = spec.build()
    lmax = int(lmax or (3 * nside - 1))
    lmax = min(lmax, 3 * nside - 1)
    ell = np.arange(lmax + 1)
    rng = np.random.default_rng(seed)
    names = [t.name for t in tracers]
    if len(set(names)) != len(names):
        raise ValueError("tracer names must be unique")

    # CCL tracers + n(z) files
    z = np.linspace(0.005, 4.0, 400)
    ccl_tr, nz_files = {}, {}
    for t in tracers:
        if t.nz_csv:
            zz, nz = _read_nz(t.nz_csv, t.nz_column)
        else:
            zz = z
            nz = np.exp(-0.5 * ((z - t.z_mean) / t.z_sigma) ** 2)
        nz = nz / np.trapezoid(nz, zz)
        npath = outdir / f"{name}_nz_{t.name}.csv"
        write_csv(npath, {"z": zz, "nz": nz}, [f"label: n(z) {t.name}", "quantity: nz", f"tracer: {t.name}",
                                                f"kind: {t.kind}"])
        nz_files[t.name] = str(npath)
        if t.kind == "shear":
            ccl_tr[t.name] = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz))
        else:
            ccl_tr[t.name] = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zz, nz),
                                                    bias=(zz, np.full_like(zz, t.bias)))
    # theory C_ell for every pair, healpy "new" ordering (diagonal-major)
    cls = {}
    for i, a in enumerate(names):
        for b in names[i:]:
            c = ccl.angular_cl(cosmo, ccl_tr[a], ccl_tr[b], ell)
            c[:2] = 0.0
            cls[(a, b)] = cls[(b, a)] = c
    n = len(names)
    ordered = [cls[(names[i], names[i + k])] for k in range(n) for i in range(n - k)]
    np.random.seed(seed)
    alms = hp.synalm(ordered, lmax=lmax, new=True)
    alms = [alms] if n == 1 else list(alms)
    tcols = {"ell": ell.astype(float)}
    for i, a in enumerate(names):
        for b in names[i:]:
            tcols[f"{a}|{b}"] = cls[(a, b)]
    tpath = outdir / f"{name}_theory_cls.csv"
    write_csv(tpath, tcols, [f"label: input C_ell ({spec.label()})", "quantity: cl_theory",
                             "pairs: columns a|b (E-mode for shear, no IA, no magnification)"])

    # mask
    if mask_path:
        mask = np.asarray(hp.read_map(mask_path, dtype=np.float64), float)
        if hp.npix2nside(mask.size) != nside:
            raise ValueError(f"mask nside {hp.npix2nside(mask.size)} != {nside}")
    else:
        mask = _fsky_cap_mask(nside, f_sky)
    if apodize_deg > 0:
        import pymaster as nmt

        mask = nmt.mask_apodization(mask, apodize_deg, apotype="C1")
    mpath = outdir / f"{name}_mask.fits"
    hp.write_map(str(mpath), mask, overwrite=True, dtype=np.float64)
    pix_sr = hp.nside2pixarea(nside)
    arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2

    files = [str(tpath), str(mpath)] + list(nz_files.values())
    fields = []
    first_map = None
    for i, t in enumerate(tracers):
        nbar = t.n_gal_arcmin2 * arcmin2_per_sr
        if t.kind == "shear":
            q, u = hp.alm2map_spin([alms[i], np.zeros_like(alms[i])], nside, 2, lmax)
            if add_noise:
                sig = t.sigma_e / np.sqrt(nbar * pix_sr)
                q = q + rng.normal(0, sig, q.size)
                u = u + rng.normal(0, sig, u.size)
            paths = [outdir / f"{name}_{t.name}_Q.fits", outdir / f"{name}_{t.name}_U.fits"]
            hp.write_map(str(paths[0]), q * (mask > 0), overwrite=True, dtype=np.float64)
            hp.write_map(str(paths[1]), u * (mask > 0), overwrite=True, dtype=np.float64)
            first_map = first_map if first_map is not None else q * (mask > 0)
        else:
            d = hp.alm2map(alms[i], nside, lmax=lmax)
            if add_noise:
                d = d + rng.normal(0, 1.0 / np.sqrt(nbar * pix_sr), d.size)
            paths = [outdir / f"{name}_{t.name}.fits"]
            hp.write_map(str(paths[0]), d * (mask > 0), overwrite=True, dtype=np.float64)
            first_map = first_map if first_map is not None else d * (mask > 0)
        files += [str(p) for p in paths]
        fields.append({"name": t.name, "kind": t.kind, "mask_path": str(mpath), "map_paths": [str(p) for p in paths],
                       "n_gal_arcmin2": t.n_gal_arcmin2 if add_noise else None, "sigma_e": t.sigma_e,
                       "nz_csv": nz_files[t.name], "nz_column": "nz"})
    fjson = outdir / f"{name}_fields.json"
    fjson.write_text(json.dumps({"fields": fields, "nside": nside, "lmax": lmax, "seed": seed,
                                 "cosmology": spec.firecrown_params(), "theory_cls_csv": str(tpath),
                                 "f_sky": float((mask > 0).mean())}, indent=1), encoding="utf-8")
    files.append(str(fjson))
    ppath = outdir / f"{name}_maps.png"
    _mollview(ppath, np.where(mask > 0, first_map, hp.UNSEEN), f"{names[0]} ({tracers[0].kind}), nside {nside}")
    files.append(str(ppath))
    msg = (f"Simulated {n} tracer maps at nside {nside} (lmax {lmax}, f_sky {float((mask > 0).mean()):.3f}, "
           f"{'with' if add_noise else 'no'} noise, cosmology {spec.label()}). Next: namaster_compute_cls("
           f"fields_json='{fjson.name}').")
    return ArtifactResult(
        status="success", files=files, message=msg,
        metadata={"nside": nside, "lmax": lmax, "f_sky": float((mask > 0).mean()), "tracers": names,
                  "fields_json": str(fjson), "theory_cls_csv": str(tpath), "mask": str(mpath),
                  "cosmology": spec.label(), "seed": seed, "noise": add_noise},
    )


@validate_call
def namaster_compute_cls(
    output_dir: Annotated[str, Field(min_length=1)],
    fields: Annotated[list[FieldSpec] | None, Field(description="Field specs (masks, maps, noise, n(z)); or give fields_json.")] = None,
    fields_json: Annotated[str | None, Field(description="JSON with a 'fields' list (namaster_simulate_maps writes one).")] = None,
    pairs: Annotated[Literal["all", "auto"] | list[list[str]], Field(description="'all' = every pair i<=j, 'auto' = auto-spectra only, or explicit [[a, b], ...].")] = "all",
    binning: Annotated[BinningSpec | None, Field(description="Bandpowers; default linear, width 20 from ell 2 to 3*nside-1.")] = None,
    lmax: Annotated[int | None, Field(description="Multipole cut (default 3*nside - 1; lower = faster).")] = None,
    lite: Annotated[bool, Field(description="Lite NaMaster fields (less memory; no contaminant templates).")] = True,
    write_sacc: Annotated[bool, Field(description="Write a sacc file with tracers, bandpowers, windows and n_ell_coupled metadata.")] = True,
    include_b_modes: Annotated[bool, Field(description="Keep EB/BE/BB and TB components in the sacc (drop them before a likelihood with sacc_prepare_for_firecrown).")] = True,
    theory_csv: Annotated[str | None, Field(description="CSV of theory C_ell (columns ell, a|b) to bin with the windows and overlay in the plot (namaster_simulate_maps writes one).")] = None,
    plot: bool = True,
    return_windows: Annotated[bool, Field(description="Remote runs: also return the bandpower windows inline (large) so the sacc gets them.")] = False,
    env_setup: Annotated[str | None, Field(description="Facility environment with NaMaster when dispatch is remote (map/mask paths must be facility paths).")] = None,
    duration_s: Annotated[int, Field(ge=300, le=28800)] = 3600,
    name: Annotated[str, Field(min_length=1)] = "namaster",
) -> ArtifactResult:
    """Measure pseudo-C_ell bandpowers of masked spin-0/spin-2 maps with NaMaster and write them as a sacc file with bandpower windows and coupled-noise metadata.

    For each pair: NmtField(mask, maps), one NmtWorkspace per (mask, spin)
    combination, decoupled bandpowers with the white noise subtracted for
    auto-spectra when n_gal_arcmin2 is given. Writes <name>_<a>_<b>.csv
    (ell_eff + components), <name>_cls.npz (windows), <name>.sacc (NZ
    tracers when nz_csv is given, data types by kind, BandpowerWindow per
    point, tracer metadata n_ell_coupled and f_sky), a PNG of the
    bandpowers (with the binned theory when theory_csv is given) and a
    JSON summary. Weight: dispatchable (nside <= 256 seconds-minutes here;
    larger maps on a facility with env_setup). Next: tjpcov_generate_config
    on the sacc (FourierGaussianNmt with the same masks, or Fsky with
    f_sky,eff from metadata), then firecrown_build_likelihood.
    """
    if (fields is None) == (fields_json is None):
        raise ValueError("give exactly one of fields or fields_json")
    if fields_json:
        doc = json.loads(Path(fields_json).expanduser().read_text(encoding="utf-8"))
        raw = doc["fields"] if isinstance(doc, dict) else doc
        fields = [FieldSpec.model_validate({k: v for k, v in f.items() if k in FieldSpec.model_fields}) for f in raw]
    assert fields is not None
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise ValueError("field names must be unique")
    by_name = {f.name: f for f in fields}
    if pairs == "all":
        pair_list = [[a, b] for i, a in enumerate(names) for b in names[i:]]
    elif pairs == "auto":
        pair_list = [[a, a] for a in names]
    else:
        pair_list = [list(p) for p in pairs]
        for a, b in pair_list:
            if a not in by_name or b not in by_name:
                raise ValueError(f"pair {[a, b]} names an unknown field; fields: {names}")
    outdir = resolve_outdir(output_dir)
    from mcp_server.dispatch import remote_site

    remote = remote_site() is not None
    for f in fields:
        if not remote:
            for p in [f.mask_path] + f.map_paths:
                if not Path(p).expanduser().is_file():
                    raise ValueError(f"file not found: {p}")
        if f.kind == "shear" and len(f.map_paths) != 2:
            raise ValueError(f"field {f.name}: shear needs [Q, U] maps")
        if f.kind != "shear" and len(f.map_paths) != 1:
            raise ValueError(f"field {f.name}: spin-0 fields take one map")

    bspec = binning or BinningSpec()
    slug = param_slug({"fields": ",".join(names), "pairs": str(pair_list), "bin": bspec.model_dump_json(), "lmax": lmax})
    params = {
        "fields": [{"name": f.name, "spin": SPIN[f.kind], "mask_path": str(Path(f.mask_path).expanduser()),
                    "map_paths": [str(Path(p).expanduser()) for p in f.map_paths], "hdf5_names": f.hdf5_names,
                    "nside": f.nside, "apodize_deg": f.apodize_deg, "apotype": f.apotype, "purify_b": f.purify_b,
                    "n_iter": f.n_iter,
                    "noise": ({"n_gal_arcmin2": f.n_gal_arcmin2, "sigma_e": f.sigma_e} if f.n_gal_arcmin2 else None)}
                   for f in fields],
        "pairs": pair_list, "binning": bspec.model_dump(), "lmax": lmax, "lite": lite,
        "work_dir": str(outdir), "npz_name": f"{name}_cls_{slug}.npz", "return_windows": bool(return_windows and remote),
    }
    res, computed_on = _run_inner(params, env_setup, duration_s)

    ell_eff = np.asarray(res["ell_eff"], float)
    windows: dict[str, np.ndarray] = {}
    npz = res.get("npz")
    if npz and Path(npz).is_file():
        with np.load(npz) as z:
            for k in z.files:
                if k.startswith("win_"):
                    windows[k[4:]] = np.asarray(z[k])
    elif res.get("windows"):
        windows = {k: np.asarray(v) for k, v in res["windows"].items()}
    files = [npz] if npz and Path(npz).is_file() else []

    # theory, binned with the windows when available
    theory: dict[str, np.ndarray] = {}
    theory_binned: dict[tuple, np.ndarray] = {}
    if theory_csv:
        _, tcols = read_csv(theory_csv)
        tell = np.asarray(tcols["ell"], int)
        for k, v in tcols.items():
            if "|" in k:
                a, b = k.split("|")
                theory[f"{a}|{b}"] = theory[f"{b}|{a}"] = np.asarray(v, float)

    # per-pair CSV
    per_pair = []
    for p in res["pairs"]:
        a, b, comps = p["a"], p["b"], p["components"]
        cl = np.asarray(p["cls"], float)
        cols = {"ell_eff": ell_eff}
        for i, c in enumerate(comps):
            cols[c] = cl[i]
        nd = np.asarray(p["noise_decoupled"], float)
        if nd.any():
            for i, c in enumerate(comps):
                cols[f"noise_{c}"] = nd[i]
        key = f"{a}|{b}"
        if key in theory and windows.get(f"{a}__{b}") is not None:
            w = windows[f"{a}__{b}"][0]           # first component (TT/TE/EE) window: (n_bpw, lmax+1)
            th = np.zeros(w.shape[1])
            n = min(len(tell), w.shape[1])
            th[:n] = theory[key][:n]
            theory_binned[(a, b)] = w @ th
            cols["theory_binned"] = theory_binned[(a, b)]
        cpath = outdir / f"{name}_{a}_{b}.csv"
        write_csv(cpath, cols, [f"label: NaMaster bandpowers {a} x {b}", "quantity: cl_bandpowers",
                                f"components: {','.join(comps)}", f"spins: {p['spins']}",
                                f"noise_coupled: {p['noise_coupled']}", f"computed_on: {computed_on}"])
        files.append(str(cpath))
        per_pair.append({"a": a, "b": b, "components": comps, "csv": str(cpath),
                         "first_component_mean": float(np.mean(cl[0])),
                         "theory_binned": theory_binned.get((a, b), np.array([])).tolist() or None})

    # sacc
    sacc_path = None
    sacc_entries = []
    if write_sacc:
        import sacc

        s = sacc.Sacc()
        s.metadata["namaster_nside"] = int(res["nside"])
        s.metadata["binning/ell_max"] = int(res["lmax"])
        # bandpower edges as text: TJPCov's f_sky types need top-hat edges, and
        # dense NaMaster windows do not reveal them (tjpcov_compute_covariance reads this)
        s.metadata["binning/ell_edges"] = ",".join(str(int(e)) for e in res["ell_edges"])
        for f in fields:
            info = res["fields"][f.name]
            meta = {"n_ell_coupled": float(info["noise_coupled"]) if info["noise_coupled"] else 0.0,
                    "fsky_eff": float(info["fsky_eff"]), "fsky_mask": float(info["fsky_mask"]),
                    "mask_path": str(f.mask_path), "apodize_deg": float(f.apodize_deg)}
            if f.nz_csv:
                zz, nz = _read_nz(f.nz_csv, f.nz_column)
                quantity = {"shear": "galaxy_shear", "density": "galaxy_density",
                            "cmb_convergence": "cmb_convergence"}[f.kind]
                s.add_tracer("NZ", f.name, zz, nz, quantity=quantity, metadata=meta)
            else:
                s.add_tracer("Misc", f.name, metadata=dict(meta, kind=f.kind, spin=SPIN[f.kind]))
        lmax_eff = int(res["lmax"])
        values = np.arange(lmax_eff + 1)
        for p in res["pairs"]:
            a, b = p["a"], p["b"]
            ka, kb = by_name[a].kind, by_name[b].kind
            # sacc order inside the data type (shear first, cmb first); NaMaster order is lower spin first
            t1, t2 = (a, b) if _SACC_ORDER[ka] <= _SACC_ORDER[kb] else (b, a)
            key = tuple(sorted((ka, kb), key=lambda k: ("cmb_convergence", "density", "shear").index(k)))
            type_map = _SACC_TYPES.get(key)
            if type_map is None:
                continue
            cl = np.asarray(p["cls"], float)
            win = windows.get(f"{a}__{b}")
            for i, comp in enumerate(p["components"]):
                dtype = type_map.get(comp)
                if dtype is None:
                    continue
                if not include_b_modes and ("B" in comp):
                    continue
                window = sacc.BandpowerWindow(values, win[i].T) if win is not None else None
                s.add_ell_cl(dtype, t1, t2, ell_eff, cl[i], window=window)
                sacc_entries.append([dtype, t1, t2, comp])
        sacc_path = outdir / f"{name}_{slug}.sacc"
        s.save_fits(str(sacc_path), overwrite=True)
        files.append(str(sacc_path))

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..plotting import rc_params

        n_p = len(res["pairs"])
        ncol = min(3, n_p)
        nrow = int(np.ceil(n_p / ncol))
        with plt.rc_context(rc_params(11)):
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False, constrained_layout=True)
            for ax, p in zip(axes.ravel(), res["pairs"]):
                cl = np.asarray(p["cls"], float)
                ax.plot(ell_eff, cl[0], "o", ms=4, label=p["components"][0])
                if (p["a"], p["b"]) in theory_binned:
                    ax.plot(ell_eff, theory_binned[(p["a"], p["b"])], "-", color="k", lw=1.2, label="theory (binned)")
                ax.set_xscale("log")
                ax.set_yscale("symlog", linthresh=max(1e-12, 0.01 * np.nanmax(np.abs(cl[0]))))
                ax.set_title(f"{p['a']} x {p['b']}")
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(r"$C_\ell$")
                ax.legend(fontsize=8)
            for ax in axes.ravel()[n_p:]:
                ax.axis("off")
            ppath = outdir / f"{name}_cls_{slug}.png"
            fig.savefig(ppath)
            plt.close(fig)
        files.append(str(ppath))

    jpath = outdir / f"{name}_cls_{slug}.json"
    summary = {"nside": res["nside"], "lmax": res["lmax"], "n_bins": res["n_bins"], "ell_eff": res["ell_eff"],
               "ell_edges": res["ell_edges"], "fields": res["fields"], "pairs": per_pair, "sacc": str(sacc_path) if sacc_path else None,
               "sacc_entries": sacc_entries, "windows_in_sacc": bool(windows), "computed_on": computed_on,
               "elapsed_s": res.get("elapsed_s"), "n_workspaces": res.get("n_workspaces")}
    write_json(jpath, summary)
    files.append(str(jpath))
    fsky = {k: round(v["fsky_eff"], 4) for k, v in res["fields"].items()}
    msg = (f"{len(res['pairs'])} pair(s), {res['n_bins']} bandpowers (ell {res['ell_edges'][0]}-{res['ell_edges'][-1]}, "
           f"nside {res['nside']}) on {computed_on} in {res.get('elapsed_s', 0)} s; f_sky,eff {fsky}"
           + (f"; sacc {Path(sacc_path).name} with {len(sacc_entries)} data types"
              + ("" if windows else " (NO windows)") if sacc_path else "") + ".")
    return ArtifactResult(status="success", files=files, message=msg, metadata=summary)


namaster_compute_cls.weight = "dispatchable"
