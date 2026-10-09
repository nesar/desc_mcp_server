"""TJPCov tools: analysis-grade covariances for sacc data vectors.

TJPCov (github.com/LSSTDESC/TJPCov) is the DESC covariance calculator used by
TXPipe's covariance stages and by augur (cov_type: tjpcov). This family
exposes it directly: a validated config for a sacc file, the computation
(Gaussian f_sky / NaMaster mask-coupled Gaussian, halo-model super-sample
and connected non-Gaussian terms, real-space projections), and a comparison
tool for two covariances of the same data vector.

Every tool writes NEW files into output_dir; inputs are never edited.
Package name is `tjpcov_tools` so it never shadows the library.
"""

import contextlib
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import Field, validate_call

from ..common import ArtifactResult, param_slug, resolve_outdir, write_csv, write_json
from ..cosmology import CosmologyParams
from ..sacc_tools import _cov_dense, _cov_info, _load, _tracer_kind

__all__ = ["tjpcov_list_covariance_types", "tjpcov_generate_config",
           "tjpcov_compute_covariance", "tjpcov_compare_covariances"]

CONVENTIONS = {
    "config": "a TJPCov YAML: `tjpcov:` section (sacc_file, cosmo: set, cov_type list, fsky, Ngal_<tracer> "
              "[arcmin^-2], sigma_e_<tracer>, bias_<tracer>, IA, mask_file/mask_names/nside for NaMaster "
              "types) + `parameters:` (pyccl.Cosmology kwargs) + optional ProjectedReal/HOD/SSC/cNG sections; "
              "written by tjpcov_generate_config, runnable by TJPCov's own run_tjpcov.py as well",
    "tracers": "TJPCov tells shear from density by tracer.quantity (galaxy_shear/galaxy_density) or by the "
               "name containing src/source (shear) or lens (density); anything else is skipped - run "
               "sacc_prepare_for_firecrown first",
    "binning": "harmonic: bandpower edges come from the sacc windows (namaster_compute_cls / TXPipe write "
               "them); without windows TJPCov assumes LINEAR bandpowers around the listed ells. Real space: "
               "log-spaced theta bins inferred from the listed thetas [arcmin]; ProjectedReal.lmax sets the "
               "ell range of the Fourier covariance that is projected",
    "noise": "Ngal_<tracer> in galaxies per arcmin^2 (TJPCov converts to sr^-1); shear noise sigma_e^2/n, "
             "density noise 1/n; NaMaster types read the coupled noise from tracer metadata n_ell_coupled",
    "terms": "the output sacc carries the SUM of the requested terms; with save_terms one sacc per term "
             "(gauss, SSC, cNG) is written too",
}
CAVEATS = [
    "FourierGaussianFsky is the Knox formula (same physics as sacc_attach_gaussian_covariance, with TJPCov's "
    "window-based binning); SSC and cNG add the halo-model super-sample and connected terms (minutes, scale "
    "with the number of tracer pairs); FourierGaussianNmt needs the masks (HEALPix FITS, or TXPipe HDF5 + "
    "nside) and computes mode coupling with NaMaster (minutes to hours).",
    "A sacc without bandpower windows gets TJPCov's linear-binning fallback: fine for linearly binned "
    "bandpowers, approximate for log-spaced ells. namaster_compute_cls writes windows.",
    "cNG needs an HOD; tjpcov_generate_config writes TJPCov's example HOD when you give none - a stand-in, "
    "not a fitted model. Replace it for a publication covariance.",
    "Real-space covariances (RealGaussianFsky) project the Fourier Gaussian term with Wigner transforms; "
    "SSC/cNG are harmonic-only in TJPCov 0.5.",
    "Remote runs ship the sacc inline (<= 8 MB) unless sacc_remote_path is given; mask files for NaMaster "
    "types must already be on the facility (give facility paths in the config).",
]

COV_TYPES: dict[str, dict] = {
    "FourierGaussianFsky": {"space": "harmonic", "term": "gauss", "needs": ["fsky"],
                            "what": "disconnected Gaussian (Knox) with the f_sky approximation; seconds"},
    "FourierGaussianNmt": {"space": "harmonic", "term": "gauss", "needs": ["mask_file", "mask_names"],
                           "what": "disconnected Gaussian with NaMaster mode coupling from the masks; "
                                   "reads coupled noise from tracer metadata n_ell_coupled; minutes-hours. "
                                   "TJPCov 0.5.1 calls the NaMaster 2 API here: needs an environment with "
                                   "pymaster < 3 (not this server's; use env_setup on a facility)"},
    "FourierSSCHaloModelFsky": {"space": "harmonic", "term": "SSC", "needs": ["fsky"],
                                "what": "super-sample covariance, halo model (linear bias), f_sky sigma_B; minutes"},
    "FourierSSCHaloModel": {"space": "harmonic", "term": "SSC", "needs": ["mask_file", "mask_names"],
                            "what": "super-sample covariance with sigma_B from the mask power spectrum"},
    "FouriercNGHaloModelFsky": {"space": "harmonic", "term": "cNG", "needs": ["fsky", "HOD"],
                                "what": "connected non-Gaussian 1-halo trispectrum term with an HOD; minutes"},
    "FouriercNGHaloModel": {"space": "harmonic", "term": "cNG", "needs": ["mask_file", "mask_names", "HOD"],
                            "what": "connected non-Gaussian term with the mask area"},
    "RealGaussianFsky": {"space": "real", "term": "gauss", "needs": ["fsky", "ProjectedReal.lmax"],
                         "what": "Gaussian covariance for xi_+/xi_-/gamma_t/w(theta): Fourier Knox term "
                                 "projected with Wigner transforms; seconds-minutes"},
    "ClusterCountsGaussian": {"space": "clusters", "term": "gauss", "needs": ["cluster sections"],
                              "what": "cluster number-count Gaussian term (not configured by this server)"},
    "ClusterCountsSSC": {"space": "clusters", "term": "SSC", "needs": ["cluster sections"],
                         "what": "cluster number-count SSC (not configured by this server)"},
    "ClusterMass": {"space": "clusters", "term": "gauss", "needs": ["cluster sections"],
                    "what": "cluster mass covariance (not configured by this server)"},
}
SUPPORTED = [k for k, v in COV_TYPES.items() if v["space"] in ("harmonic", "real")]

# TJPCov's own example HOD (examples/full_3x2pt_cov_example.yml): a stand-in.
EXAMPLE_HOD = {"log10Mmin_0": 12.0, "log10Mmin_p": 0.0, "siglnM_0": 0.4, "siglnM_p": 0.0,
               "log10M0_0": 7.0, "log10M0_p": 0.0, "log10M1_0": 13.3, "log10M1_p": 0.0,
               "alpha_0": 1.0, "alpha_p": 0.0, "fc_0": 1.0, "fc_p": 0.0, "bg_0": 1.0, "bg_p": 0.0,
               "bmax_0": 1.0, "bmax_p": 0.0, "a_pivot": 1.0, "ns_independent": False}

_INLINE_SACC_MAX = 8 * 1024 * 1024

DISPATCH_KERNELS = {
    "tjpcov_cov": {
        "function": "envkernel.run_in_env", "inner": "tjpcov_cov",
        "params": {"tjpcov": {"cov_type": ["FourierGaussianFsky"], "fsky": 0.3, "Ngal_src0": 10.0,
                              "sigma_e_src0": 0.26, "bias_lens0": 1.5},
                   "extra_sections": {}, "cosmo_kwargs": {"Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67,
                                                           "n_s": 0.96, "sigma8": 0.81},
                   "sacc_b64": "<input sacc inline> or sacc_remote_path", "output_name": "cls_cov.hdf5",
                   "work_dir": ".", "save_terms": True, "inline_max_bytes": 8000000},
        "env_setup_required": True, "suitable_envs": ["desc-python", "desc-cosmology"],
        "duration_hint_s": 1800,
        "returns": "sacc with the covariance (inline when small), per-term diagnostics, condition number, S/N",
    },
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _tjpcov_kind(s, name: str) -> str | None:
    """What TJPCov itself will decide for a tracer (its rule, not ours)."""
    q = getattr(s.tracers[name], "quantity", None) or ""
    if q == "galaxy_shear" or "src" in name or "source" in name:
        return "shear"
    if q == "galaxy_density" or "lens" in name:
        return "density"
    if q == "cmb_convergence":
        return "cmb_convergence"
    return None


def _space(s) -> str:
    dts = s.get_data_types()
    if not dts:
        raise ValueError("the sacc file has no data points.")
    if all("_cl" in d for d in dts):
        return "harmonic"
    if all("_xi" in d for d in dts):
        return "real"
    return "mixed"


def _has_windows(s) -> bool:
    for d in s.data:
        w = d.tags.get("window")
        if w is not None:
            return True
    return False


def _per_tracer(value, names: list[str], label: str, needed: list[str]) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, (int, float)):
        return {n: float(value) for n in needed}
    out = {k: float(v) for k, v in dict(value).items()}
    unknown = [k for k in out if k not in names]
    if unknown:
        raise ValueError(f"{label}: unknown tracers {unknown}; file has {names}")
    return out


def _read_config(config_path: str) -> tuple[Path, dict]:
    import yaml

    p = Path(config_path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"TJPCov config not found: {p} (write one with tjpcov_generate_config)")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "tjpcov" not in doc or not isinstance(doc["tjpcov"], dict):
        raise ValueError(f"{p} has no `tjpcov:` section")
    return p, doc


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
@validate_call
def tjpcov_list_covariance_types() -> ArtifactResult:
    """List TJPCov's covariance types: what each term is, the space it applies to, and what the config needs for it.

    Pick types for tjpcov_generate_config(cov_types=[...]): one `gauss`
    term (FourierGaussianFsky or FourierGaussianNmt) plus optional SSC and
    cNG terms for harmonic data; RealGaussianFsky alone for real-space
    xi(theta) data. Fsky variants need f_sky; the others need HEALPix masks.
    Cluster types exist in TJPCov but are not configured by this server.
    """
    return ArtifactResult(
        status="success", files=[],
        message=f"{len(SUPPORTED)} covariance types usable here: {', '.join(SUPPORTED)}; "
                f"{len(COV_TYPES) - len(SUPPORTED)} cluster types listed for reference.",
        metadata={"types": COV_TYPES, "supported_here": SUPPORTED,
                  "recommended_sets": {
                      "quick_harmonic": ["FourierGaussianFsky"],
                      "forecast_harmonic": ["FourierGaussianFsky", "FourierSSCHaloModelFsky"],
                      "full_harmonic": ["FourierGaussianFsky", "FourierSSCHaloModelFsky", "FouriercNGHaloModelFsky"],
                      "measured_maps": ["FourierGaussianNmt", "FourierSSCHaloModel"],
                      "real_space": ["RealGaussianFsky"]},
                  "conventions": CONVENTIONS, "caveats": CAVEATS},
    )


@validate_call
def tjpcov_generate_config(
    output_dir: Annotated[str, Field(min_length=1)],
    sacc_path: Annotated[str, Field(min_length=1, description="sacc file whose data vector the covariance is for (its n(z) tracers and ell/theta binning are used; its values are not).")],
    cov_types: Annotated[list[str], Field(min_length=1, description=f"TJPCov covariance classes to sum, e.g. ['FourierGaussianFsky', 'FourierSSCHaloModelFsky']; see tjpcov_list_covariance_types. Usable: {SUPPORTED}.")],
    n_gal: Annotated[dict[str, float] | float, Field(description="Effective number density per tracer in galaxies/arcmin^2 ({tracer: n} or one value for all).")],
    f_sky: Annotated[float | None, Field(gt=0.0, le=1.0, description="Sky fraction for the *Fsky types (LSST Y1 ~0.3, DES Y1 ~0.03).")] = None,
    sigma_e: Annotated[dict[str, float] | float, Field(description="Per-component shape noise for shear tracers (dict or one value).")] = 0.26,
    galaxy_bias: Annotated[dict[str, float] | float, Field(description="Linear galaxy bias per density tracer (dict or one value).")] = 1.5,
    ia_amplitude: Annotated[float | None, Field(description="Constant NLA intrinsic-alignment amplitude for the shear signal C_ell (TJPCov key IA); None = no IA.")] = None,
    cosmology: Annotated[CosmologyParams | None, Field(description="Fiducial cosmology for the signal C_ell (None = vanilla LCDM, CAMB transfer function, halofit).")] = None,
    lmax: Annotated[int | None, Field(ge=10, le=20000, description="Real space only (RealGaussianFsky): multipole range of the Fourier covariance that is projected (ProjectedReal.lmax); default 3000.")] = None,
    mask_files: Annotated[dict[str, str] | None, Field(description="For non-Fsky (NaMaster) types: {tracer: path} of HEALPix masks (FITS, or TXPipe HDF5 with mask_nside). Facility paths when the covariance will run remotely.")] = None,
    mask_names: Annotated[dict[str, str] | None, Field(description="Mask name per tracer (tracers sharing a mask share its NaMaster workspace); default = tracer name.")] = None,
    mask_nside: Annotated[int | None, Field(description="nside of HDF5 masks (TXPipe maps file).")] = None,
    hod: Annotated[dict[str, float | bool] | None, Field(description="HOD parameters for the cNG term (pyccl HaloProfileHOD names: log10Mmin_0, siglnM_0, log10M0_0, log10M1_0, alpha_0, ...). None = TJPCov's example HOD (a stand-in).")] = None,
    name: Annotated[str, Field(min_length=1, description="Stem of the config file.")] = "tjpcov_config",
) -> ArtifactResult:
    """Write a validated TJPCov YAML config for a sacc file: covariance terms, noise and bias per tracer, f_sky or masks, cosmology.

    Checks before anything runs: the file is harmonic or real (not mixed),
    every tracer is one TJPCov can classify (shear/density/cmb_convergence;
    otherwise run sacc_prepare_for_firecrown), the requested types match the
    space and have what they need (f_sky, masks, HOD, lmax), noise is given
    for every tracer, bias for every density tracer. Reports whether the
    file carries bandpower windows (TJPCov's binning source) and the
    n_ell_coupled metadata NaMaster types need. The YAML stores absolute
    paths, `cosmo: set` with a `parameters:` block of pyccl kwargs, and is
    runnable by TJPCov's run_tjpcov.py too. Next: tjpcov_compute_covariance.
    """
    s = _load(sacc_path)
    spath = Path(sacc_path).expanduser().resolve()
    outdir = resolve_outdir(output_dir)
    warnings_: list[str] = []

    unknown = [t for t in cov_types if t not in COV_TYPES]
    if unknown:
        raise ValueError(f"unknown TJPCov covariance types {unknown}; known: {list(COV_TYPES)}")
    unsupported = [t for t in cov_types if t not in SUPPORTED]
    if unsupported:
        raise ValueError(f"{unsupported} are cluster covariances; this server configures only {SUPPORTED}")
    space = _space(s)
    if space == "mixed":
        raise ValueError("the sacc mixes harmonic and real-space data types; keep one family with "
                         "sacc_prepare_for_firecrown(keep_data_types=...) first.")
    wrong = [t for t in cov_types if COV_TYPES[t]["space"] != space]
    if wrong:
        raise ValueError(f"{wrong} do not apply to {space}-space data; see tjpcov_list_covariance_types.")
    terms = [COV_TYPES[t]["term"] for t in cov_types]
    if len(set(terms)) != len(terms):
        raise ValueError(f"each term may be requested once; you asked for {terms}")
    if "gauss" not in terms:
        warnings_.append("no Gaussian term requested: the covariance will contain only the "
                         "non-Gaussian terms you listed (usually you want a *Gaussian* type too).")

    names = list(s.tracers)
    kinds = {n: _tjpcov_kind(s, n) for n in names}
    bad = [n for n, k in kinds.items() if k is None]
    if bad:
        raise ValueError(f"TJPCov cannot classify tracers {bad} (needs quantity galaxy_shear/galaxy_density or "
                         "a name containing src/source/lens): run sacc_prepare_for_firecrown first.")
    ours = {n: _tracer_kind(s, n) for n in names}
    mismatch = [n for n in names if ours[n] and kinds[n] != ours[n] and kinds[n] != "cmb_convergence"]
    if mismatch:
        raise ValueError(f"tracer kind disagreement for {mismatch}: TJPCov would use {[kinds[n] for n in mismatch]} "
                         f"but the data types imply {[ours[n] for n in mismatch]}; fix names/quantities first.")
    for n in names:
        tr = s.tracers[n]
        if kinds[n] != "cmb_convergence" and (getattr(tr, "z", None) is None or not len(tr.z)):
            raise ValueError(f"tracer {n} has no n(z); TJPCov builds its CCL tracers from the sacc n(z).")

    shear = [n for n in names if kinds[n] == "shear"]
    density = [n for n in names if kinds[n] == "density"]
    galaxy = shear + density
    ngal = _per_tracer(n_gal, names, "n_gal", galaxy)
    sige = _per_tracer(sigma_e, names, "sigma_e", shear)
    bias = _per_tracer(galaxy_bias, names, "galaxy_bias", density)
    missing = [n for n in galaxy if n not in ngal]
    if missing:
        raise ValueError(f"n_gal missing for tracers {missing}")
    missing = [n for n in shear if n not in sige]
    if missing:
        raise ValueError(f"sigma_e missing for shear tracers {missing}")
    missing = [n for n in density if n not in bias]
    if missing:
        raise ValueError(f"galaxy_bias missing for density tracers {missing}")

    needs = {need for t in cov_types for need in COV_TYPES[t]["needs"]}
    section: dict = {"sacc_file": str(spath), "cosmo": "set", "cov_type": list(cov_types),
                     "outdir": str(outdir), "use_mpi": False}
    if "fsky" in needs:
        if f_sky is None:
            raise ValueError(f"f_sky is required by {[t for t in cov_types if 'fsky' in COV_TYPES[t]['needs']]}")
        section["fsky"] = float(f_sky)
    if "mask_file" in needs:
        if not mask_files:
            raise ValueError("mask_files ({tracer: path}) are required by the NaMaster covariance types "
                             f"{[t for t in cov_types if 'mask_file' in COV_TYPES[t]['needs']]}")
        missing = [n for n in names if n not in mask_files]
        if missing:
            raise ValueError(f"mask_files missing for tracers {missing}")
        section["mask_file"] = {n: str(mask_files[n]) for n in names}
        section["mask_names"] = {n: (mask_names or {}).get(n, n) for n in names}
        if mask_nside:
            section["nside"] = int(mask_nside)
        absent = [n for n in names if not Path(mask_files[n]).expanduser().is_file()]
        if absent:
            warnings_.append(f"mask files not found on this machine for {absent} (fine if they are facility "
                             "paths for a remote run).")
        no_noise = [n for n in galaxy if "n_ell_coupled" not in (getattr(s.tracers[n], "metadata", {}) or {})]
        if no_noise:
            warnings_.append(f"tracers {no_noise} carry no n_ell_coupled metadata: the NaMaster Gaussian "
                             "term will warn about missing coupled noise (namaster_compute_cls writes it).")
    for n in galaxy:
        section[f"Ngal_{n}"] = ngal[n]
    for n in shear:
        section[f"sigma_e_{n}"] = sige[n]
    for n in density:
        section[f"bias_{n}"] = bias[n]
    if ia_amplitude is not None:
        section["IA"] = float(ia_amplitude)

    spec = cosmology or CosmologyParams()
    if spec.matter_power_spectrum not in ("halofit", "linear", "camb_hmcode") or spec.baryons != "none" \
            or spec.mg_mu0 or spec.mg_sigma0:
        warnings_.append("TJPCov's cosmology is a plain pyccl.Cosmology: emulator P(k), baryon and MG "
                         "switches are not carried over (halofit used).")
    cosmo_kwargs = spec.ccl_kwargs()
    doc: dict = {"tjpcov": section, "parameters": cosmo_kwargs}
    if "mask_file" in needs:
        try:
            import importlib.metadata as _md

            if int(_md.version("pymaster").split(".")[0]) >= 3:
                warnings_.append("FourierGaussianNmt/SSC/cNG with masks: TJPCov 0.5.1 uses the NaMaster 2 API; "
                                 "this server's pymaster is 3.x, so run this config remotely (env_setup with "
                                 "NaMaster 2.x) or use the Fsky types locally.")
        except Exception:  # noqa: BLE001
            pass
    if "ProjectedReal.lmax" in needs:
        doc["ProjectedReal"] = {"lmax": int(lmax or 3000)}
        thetas = [float(d.get_tag("theta")) for d in s.data if d.get_tag("theta") is not None]
        if thetas and doc["ProjectedReal"]["lmax"] < 2000 and min(thetas) < 20:
            warnings_.append(f"lmax={doc['ProjectedReal']['lmax']} is low for theta down to {min(thetas):.1f} arcmin: "
                             "xi_- and gamma_t variances at small angles will be underestimated and the matrix "
                             "ill-conditioned; use lmax >= 3000 (slower).")
    elif lmax is not None:
        warnings_.append("lmax only matters for RealGaussianFsky; ignored for harmonic types.")
    if "HOD" in needs:
        doc["HOD"] = dict(hod) if hod else dict(EXAMPLE_HOD)
        if not hod:
            warnings_.append("cNG term uses TJPCov's example HOD (a stand-in): give hod={...} for a real one.")

    has_windows = _has_windows(s) if space == "harmonic" else None
    if space == "harmonic" and not has_windows:
        ells = sorted({float(d.get_tag("ell")) for d in s.data if d.get_tag("ell") is not None})
        if len(ells) > 2 and np.std(np.diff(ells)) > 1e-6 * np.mean(np.diff(ells)):
            warnings_.append("no bandpower windows and the ells are not linearly spaced: TJPCov's linear "
                             "fallback approximates the bandwidths (namaster_compute_cls writes windows).")

    import yaml

    slug = param_slug({"types": ",".join(cov_types), "fsky": f_sky, "cosmo": spec.slug(), "sacc": str(spath)})
    cpath = outdir / f"{name}_{slug}.yaml"
    cpath.write_text(
        "# TJPCov configuration written by desc-mcp-server tjpcov_generate_config\n"
        f"# input sacc: {spath}\n# cov_type: {cov_types}; cosmology: {spec.label()}\n"
        + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    msg = (f"Wrote {cpath.name}: {space}-space, {len(names)} tracers ({len(shear)} shear, {len(density)} density), "
           f"terms {terms}" + (f", f_sky={f_sky}" if f_sky is not None else "")
           + (", bandpower windows present" if has_windows else
              (", NO bandpower windows (linear fallback)" if space == "harmonic" else "")) + ".")
    if warnings_:
        msg += " Warnings: " + " | ".join(warnings_)
    return ArtifactResult(
        status="success", files=[str(cpath)], message=msg,
        metadata={"config_path": str(cpath), "sacc_path": str(spath), "space": space, "n_points": len(s.data),
                  "tracer_kinds": kinds, "cov_types": list(cov_types), "terms": terms, "f_sky": f_sky,
                  "n_gal_arcmin2": ngal, "sigma_e": sige, "galaxy_bias": bias, "ia_amplitude": ia_amplitude,
                  "cosmology": spec.label(), "has_windows": has_windows, "lmax": doc.get("ProjectedReal", {}).get("lmax"),
                  "warnings": warnings_, "next_step": "tjpcov_compute_covariance(config_path=...)"},
    )


@validate_call
def tjpcov_compute_covariance(
    output_dir: Annotated[str, Field(min_length=1)],
    config_path: Annotated[str, Field(min_length=1, description="TJPCov YAML from tjpcov_generate_config.")],
    output_name: Annotated[str | None, Field(description="Name of the sacc written with the covariance (.hdf5 or .fits); default <input stem>_tjpcov_<slug>.hdf5.")] = None,
    save_terms: Annotated[bool, Field(description="Also write one sacc per covariance term (gauss/SSC/cNG) when more than one is requested.")] = True,
    plot: Annotated[bool, Field(description="Correlation-matrix PNG and, with several terms, a per-term diagonal plot.")] = True,
    env_setup: Annotated[str | None, Field(description="Facility environment activation when dispatch is remote (desc-python / desc-cosmology carry tjpcov); ignored locally.")] = None,
    sacc_remote_path: Annotated[str | None, Field(description="Remote runs: facility path of the input sacc instead of shipping it inline (required above 8 MB).")] = None,
    duration_s: Annotated[int, Field(ge=300, le=28800, description="Walltime for a remote job (Gaussian: minutes; SSC+cNG or NaMaster: hours).")] = 1800,
) -> ArtifactResult:
    """Run TJPCov on a config and write the sacc file with the covariance attached (plus per-term files and diagnostics).

    Local (dispatch local) or on a facility as an env-kernel (set_dispatch +
    env_setup; the sacc travels inline unless sacc_remote_path). Writes
    <stem>_tjpcov_<slug>.hdf5 (input copy + total covariance, metadata
    tjpcov_cov_types), one sacc per term when save_terms, a diagnostics JSON
    (per-term diagonal fractions, condition number, minimum eigenvalue,
    positive-definiteness, total S/N = sqrt(d^T C^-1 d)), a correlation PNG
    and a per-term diagonal PNG. positive_definite must be true before the
    file goes into firecrown_build_likelihood. Weight: dispatchable (Gaussian
    f_sky seconds; SSC/cNG minutes; NaMaster types minutes to hours).
    """
    cpath, doc = _read_config(config_path)
    section = dict(doc["tjpcov"])
    sacc_in = Path(str(section.pop("sacc_file", ""))).expanduser()
    if not sacc_in.is_absolute():
        sacc_in = (cpath.parent / sacc_in).resolve()
    if not sacc_in.is_file() and not sacc_remote_path:
        raise ValueError(f"input sacc referenced by the config not found: {sacc_in}")
    section.pop("cosmo", None)
    section.pop("outdir", None)
    section.pop("use_mpi", None)
    cosmo_kwargs = dict(doc.get("parameters") or {})
    if not cosmo_kwargs:
        raise ValueError("config has no `parameters:` block (pyccl.Cosmology kwargs); regenerate it with "
                         "tjpcov_generate_config.")
    extra = {k: v for k, v in doc.items() if k not in ("tjpcov", "parameters")}
    cov_types = section.get("cov_type") or []
    cov_types = [cov_types] if isinstance(cov_types, str) else list(cov_types)
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"config": str(cpath), "types": ",".join(cov_types)})
    out_name = output_name or f"{sacc_in.stem.replace('.sacc', '')}_tjpcov_{slug}.hdf5"

    from mcp_server.dispatch import remote_site, run_env_kernel  # lazy: server-only

    # Windows: TJPCov's f_sky types derive the bandpower edges from "where the
    # window weight is nonzero" - wrong for dense NaMaster windows - so those
    # runs get top-hat windows (edges from the sacc metadata namaster_compute_cls
    # writes, else estimated); NaMaster types keep the real windows.
    window_mode, ell_edges = "keep", None
    if cov_types and all(COV_TYPES.get(t, {}).get("space") == "harmonic" for t in cov_types):
        window_mode = "tophat"
        if sacc_in.is_file():
            edges_meta = _load(str(sacc_in)).metadata.get("binning/ell_edges")
            if edges_meta:
                ell_edges = [int(float(x)) for x in str(edges_meta).split(",") if x.strip()]
    params = {"tjpcov": section, "extra_sections": extra, "cosmo_kwargs": cosmo_kwargs,
              "output_name": out_name, "save_terms": save_terms,
              "window_mode": window_mode, "ell_edges": ell_edges}
    site = remote_site()
    if site:
        if not env_setup:
            raise ValueError(f"Dispatch is set to {site}: this call needs env_setup (a facility environment "
                             "with tjpcov, e.g. the NERSC desc-python stack; see export_dispatch_pack).")
        import base64

        if sacc_remote_path:
            params["sacc_path"] = sacc_remote_path
        else:
            size = sacc_in.stat().st_size
            if size > _INLINE_SACC_MAX:
                raise ValueError(f"{sacc_in.name} is {size / 1e6:.1f} MB (> 8 MB): stage it on the facility and "
                                 "pass sacc_remote_path.")
            params["sacc_b64"] = base64.b64encode(sacc_in.read_bytes()).decode("ascii")
            params["sacc_name"] = sacc_in.name
        params.update({"work_dir": ".", "inline_max_bytes": _INLINE_SACC_MAX})
        job = run_env_kernel(env_setup, "tjpcov_cov", params, duration=duration_s)
        res = job["result"]["result"]
        computed_on = job.get("host", site)
        files = []
        for fname, b64 in (res.get("inline_files") or {}).items():
            p = outdir / fname
            p.write_bytes(base64.b64decode(b64))
            files.append(str(p))
        out_path = outdir / out_name
        if not out_path.is_file():
            raise RuntimeError(f"the remote job finished but the output sacc was not returned inline "
                               f"(remote path {res.get('output_file')}); fetch it from the job directory.")
    else:
        import importlib

        params.update({"sacc_path": str(sacc_in), "work_dir": str(outdir), "inline_max_bytes": 0})
        mod = importlib.import_module("tools.inner.tjpcov_cov")
        with contextlib.redirect_stdout(sys.stderr):
            res = mod.main(params)
        computed_on = "local"
        out_path = Path(res["output_file"])
        files = [str(out_path)] + [str(Path(p)) for p in res.get("term_files", {}).values()]

    cov_info = res["covariance"]
    jpath = outdir / f"{Path(out_name).stem}_diagnostics.json"
    write_json(jpath, {"config": str(cpath), "input_sacc": str(sacc_in), "output_sacc": str(out_path),
                       "cov_types": cov_types, "terms": res["terms"], "covariance": cov_info,
                       "snr_total": res.get("snr_total"), "computed_on": computed_on,
                       "elapsed_s": res.get("elapsed_s"), "tjpcov_version": res.get("tjpcov_version"),
                       "warnings": res.get("warnings", [])})
    files.append(str(jpath))

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..plotting import rc_params

        s_out = _load(str(out_path))
        cov = _cov_dense(s_out)
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(d, d)
        with plt.rc_context(rc_params()):
            fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            pos = 0
            for dt in s_out.get_data_types():
                for combo in s_out.get_tracer_combinations(dt):
                    pos += len(s_out.indices(dt, combo))
                ax.axhline(pos - 0.5, color="k", lw=0.5)
                ax.axvline(pos - 0.5, color="k", lw=0.5)
            ax.set_title(f"TJPCov correlation: {', '.join(cov_types)}")
            ax.set_xlabel("data index")
            ax.set_ylabel("data index")
            fig.colorbar(im, ax=ax, label="correlation")
            ppath = outdir / f"{Path(out_name).stem}_corr.png"
            fig.savefig(ppath)
            plt.close(fig)
        files.append(str(ppath))
        if len(res["terms"]) > 1 and res.get("term_files"):
            with plt.rc_context(rc_params()):
                fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
                idx = np.arange(cov.shape[0])
                ax.plot(idx, np.diag(cov), color="k", lw=1.5, label="total")
                for term, tfile in res["term_files"].items():
                    tp = Path(tfile) if Path(tfile).is_file() else outdir / Path(tfile).name
                    if tp.is_file():
                        ax.plot(idx, np.diag(_cov_dense(_load(str(tp)))), lw=1.2, label=term)
                ax.set_yscale("log")
                ax.set_xlabel("data index")
                ax.set_ylabel("variance")
                ax.legend()
                ax.set_title("Covariance diagonal by term")
                tpath = outdir / f"{Path(out_name).stem}_terms.png"
                fig.savefig(tpath)
                plt.close(fig)
            files.append(str(tpath))

    pd = cov_info.get("positive_definite")
    frac = {k: v.get("frac_of_total_diag") for k, v in res["terms"].items()}
    msg = (f"Wrote {out_path.name} ({res['n_data']}x{res['n_data']} covariance, terms {list(res['terms'])}) "
           f"on {computed_on} in {res.get('elapsed_s', 0)} s. "
           + (f"S/N {res['snr_total']:.1f}; " if res.get("snr_total") else "")
           + (f"condition number {cov_info['condition_number']:.3g}; " if cov_info.get("condition_number") else "")
           + ("positive definite." if pd else "NOT positive definite - check the inputs before any likelihood."
              if pd is False else "positive-definiteness not checked (large matrix)."))
    if len(frac) > 1:
        msg += " Mean diagonal fractions: " + ", ".join(f"{k} {v:.2f}" for k, v in frac.items() if v is not None) + "."
    if res.get("warnings"):
        msg += " TJPCov warnings: " + " | ".join(res["warnings"][:4])
    return ArtifactResult(
        status="success", files=files, message=msg,
        metadata={"config": str(cpath), "input_sacc": str(sacc_in), "output_sacc": str(out_path),
                  "cov_types": cov_types, "terms": res["terms"], "covariance": cov_info,
                  "snr_total": res.get("snr_total"), "computed_on": computed_on, "elapsed_s": res.get("elapsed_s"),
                  "warnings": res.get("warnings", []), "n_points": res["n_data"],
                  "next_step": "firecrown_build_likelihood(sacc_path=<output_sacc>) or tjpcov_compare_covariances"},
    )


tjpcov_compute_covariance.weight = "dispatchable"


@validate_call
def tjpcov_compare_covariances(
    output_dir: Annotated[str, Field(min_length=1)],
    sacc_a: Annotated[str, Field(min_length=1, description="First sacc with a covariance (e.g. sacc_attach_gaussian_covariance output).")],
    sacc_b: Annotated[str, Field(min_length=1, description="Second sacc with a covariance for the SAME data vector (e.g. TJPCov output).")],
    label_a: Annotated[str, Field(min_length=1)] = "A",
    label_b: Annotated[str, Field(min_length=1)] = "B",
    plot: bool = True,
) -> ArtifactResult:
    """Compare two covariances of the same data vector: per-point sigma ratio by data type, correlation differences, S/N of each.

    Both files must list the same data points in the same order (checked:
    data types, tracer pairs, ell/theta). Reports per data type the median,
    min and max of sigma_B/sigma_A, the largest absolute difference between
    the two correlation matrices, and the total S/N under each covariance.
    Typical use: Knox f_sky (sacc_attach_gaussian_covariance) vs TJPCov
    Gaussian + SSC to see where the super-sample term matters, or a TXPipe
    covariance vs a TJPCov recomputation. Writes <stem>_sigma_ratio.csv and
    a two-panel PNG (sigma ratio by index; correlation difference).
    """
    sa, sb = _load(sacc_a), _load(sacc_b)
    pa, pb = Path(sacc_a).expanduser().resolve(), Path(sacc_b).expanduser().resolve()
    outdir = resolve_outdir(output_dir)
    ca, cb = _cov_dense(sa), _cov_dense(sb)
    if ca is None or cb is None:
        raise ValueError("both files need a covariance.")
    if len(sa.data) != len(sb.data):
        raise ValueError(f"different data vectors: {len(sa.data)} vs {len(sb.data)} points.")

    def _layout(s):
        rows = []
        for d in s.data:
            tag = "theta" if "_xi" in d.data_type else "ell"
            rows.append((d.data_type, tuple(d.tracers), round(float(d.get_tag(tag) or -1), 6)))
        return rows

    la, lb = _layout(sa), _layout(sb)
    if la != lb:
        first = next(i for i in range(len(la)) if la[i] != lb[i])
        raise ValueError(f"data vectors differ at index {first}: {la[first]} vs {lb[first]} - the comparison needs "
                         "identical layouts (same data types, tracer pairs and binning, same order).")
    siga = np.sqrt(np.clip(np.diag(ca), 0, None))
    sigb = np.sqrt(np.clip(np.diag(cb), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = sigb / siga
        corr_a = ca / np.outer(siga, siga)
        corr_b = cb / np.outer(sigb, sigb)
    np.fill_diagonal(corr_a, 1.0)
    np.fill_diagonal(corr_b, 1.0)
    dcorr = corr_b - corr_a
    per_type = {}
    for dt in sa.get_data_types():
        idx = np.array([i for i, r in enumerate(la) if r[0] == dt])
        r = ratio[idx]
        r = r[np.isfinite(r)]
        if len(r):
            per_type[dt] = {"n": int(len(idx)), "median_ratio": float(np.median(r)),
                            "min_ratio": float(r.min()), "max_ratio": float(r.max())}
    data = np.asarray(sa.mean, float)

    def _snr(c):
        try:
            return float(np.sqrt(data @ np.linalg.solve(c, data)))
        except np.linalg.LinAlgError:
            return None

    snr_a, snr_b = _snr(ca), _snr(cb)
    stem = f"{pa.stem.replace('.sacc', '')}_vs_{pb.stem.replace('.sacc', '')}"
    cpath = outdir / f"{stem}_sigma_ratio.csv"
    write_csv(cpath, {"index": np.arange(len(la), dtype=float), "x": np.array([r[2] for r in la]),
                      "sigma_a": siga, "sigma_b": sigb, "ratio_b_over_a": ratio},
              [f"label: sigma {label_b}/{label_a}", "quantity: sigma_ratio", f"a: {pa}", f"b: {pb}",
               "x: ell (harmonic) or theta [arcmin] (real)"])
    files = [str(cpath)]
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..plotting import rc_params

        with plt.rc_context(rc_params()):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
            ax1.plot(np.arange(len(ratio)), ratio, lw=1.3)
            ax1.axhline(1.0, color="k", lw=0.8, ls="--")
            pos = 0
            for dt in sa.get_data_types():
                n = sum(1 for r in la if r[0] == dt)
                ax1.axvline(pos + n - 0.5, color="0.6", lw=0.6)
                ax1.text(pos + n / 2, ax1.get_ylim()[1], dt.replace("galaxy_", ""), ha="center", va="top",
                         fontsize=7, rotation=90)
                pos += n
            ax1.set_xlabel("data index")
            ax1.set_ylabel(f"sigma({label_b}) / sigma({label_a})")
            vmax = float(np.nanmax(np.abs(dcorr))) or 1.0
            im = ax2.imshow(dcorr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax2.set_title(f"corr({label_b}) - corr({label_a})")
            fig.colorbar(im, ax=ax2)
            ppath = outdir / f"{stem}_compare.png"
            fig.savefig(ppath)
            plt.close(fig)
        files.append(str(ppath))
    med = {k: round(v["median_ratio"], 3) for k, v in per_type.items()}
    msg = (f"{label_b} vs {label_a} over {len(la)} points: median sigma ratio by data type {med}; "
           f"max |corr difference| {float(np.nanmax(np.abs(dcorr))):.3f}; "
           f"S/N {label_a} {snr_a:.1f}, {label_b} {snr_b:.1f}." if snr_a and snr_b else
           f"{label_b} vs {label_a}: median sigma ratio {med}.")
    return ArtifactResult(
        status="success", files=files, message=msg,
        metadata={"a": str(pa), "b": str(pb), "n_points": len(la), "per_data_type": per_type,
                  "max_abs_corr_difference": float(np.nanmax(np.abs(dcorr))),
                  "snr_total": {label_a: snr_a, label_b: snr_b},
                  "covariance_a": _cov_info(sa), "covariance_b": _cov_info(sb)},
    )
