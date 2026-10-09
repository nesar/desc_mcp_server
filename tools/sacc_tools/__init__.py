"""sacc data-vector tools: inspect, export, prepare for firecrown/augur, and
attach a mode-counting Gaussian covariance.

sacc (github.com/LSSTDESC/sacc) is the data contract between DESC packages:
TXPipe writes it, firecrown and augur read it. This family never edits an
input file - every tool writes NEW files into output_dir.

Package name is `sacc_tools` (not `sacc`) so it never shadows the library.
"""

import csv
import re
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, validate_call

from ..common import (ArtifactResult, param_slug, resolve_outdir, write_csv,
                      write_json)
from ..cosmology import CosmologyParams

__all__ = ["sacc_inspect", "sacc_to_csv", "sacc_prepare_for_firecrown",
           "sacc_attach_gaussian_covariance"]

CONVENTIONS = {
    "formats": "HDF5 (sacc >= 2 default) and FITS are auto-detected on read; choose the output format per tool",
    "tracers": "NZ tracers carry (z, nz); firecrown/augur want names src{i} (shear) and lens{i} (density) "
               "with quantity galaxy_shear / galaxy_density; TXPipe writes source_{i}/lens_{i} with quantity 'generic'",
    "data_types": "sacc standard names: galaxy_shear_cl_ee, galaxy_shearDensity_cl_e, galaxy_density_cl (harmonic); "
                  "galaxy_shear_xi_plus/minus, galaxy_shearDensity_xi_t, galaxy_density_xi (real)",
    "theta": "arcmin (sacc convention for the 'theta' tag)", "ell": "integer multipole (tag 'ell')",
    "gamma_t_order": "galaxy_shearDensity tracers are ordered (shear, density), i.e. (src, lens)",
    "n_gal": "number density in galaxies per arcmin^2", "sigma_e": "per-component shape noise",
}
CAVEATS = [
    "sacc_attach_gaussian_covariance is the disconnected (Gaussian, mode-counting) covariance only: no "
    "super-sample or connected non-Gaussian terms, no mask coupling beyond f_sky. Fine for sanity checks "
    "and forecasts; use TJPCov (or augur cov_type tjpcov) for an analysis-grade covariance.",
    "Real-space (xi) files cannot get a covariance from this family: use TJPCov or augur.",
    "sacc_prepare_for_firecrown only renames/relabels; it does not change theta units, ell binning or windows.",
    "Old-format sacc files (pre-2.0, e.g. external/sacc/examples/example-txpipe-sacc1.sacc) do not load with sacc 2.4.",
]

ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2  # arcmin^2 in a steradian

_SOURCE_RE = re.compile(r"^(?:source|src|shear)[_-]?(\d+)$", re.IGNORECASE)
_LENS_RE = re.compile(r"^(?:lens|density|galaxy)[_-]?(\d+)$", re.IGNORECASE)

HARMONIC_SUPPORTED = ("galaxy_shear_cl_ee", "galaxy_shearDensity_cl_e", "galaxy_density_cl")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load(path_str: str):
    """sacc.Sacc.load (auto-detects HDF5/FITS) with an actionable error."""
    import sacc

    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"sacc file not found: {path}")
    try:
        return sacc.Sacc.load(str(path))
    except Exception as exc:  # noqa: BLE001 - try firecrown's loader, then explain
        try:
            from firecrown.likelihood.factories import load_sacc_data
            return load_sacc_data(str(path))
        except Exception:  # noqa: BLE001
            pass
        raise ValueError(
            f"Could not read {path} as a sacc 2.x file ({type(exc).__name__}: {exc}). "
            "Pre-2.0 sacc files (e.g. the 'example-txpipe-sacc1.sacc' shipped with the sacc "
            "repo) are not readable by sacc 2.4; re-write them with the version that made them.") from exc


def _x_tag(data_type: str) -> str:
    return "theta" if "_xi" in data_type else "ell"


def _tracer_kind(s, name: str) -> str | None:
    """'shear' | 'density' | None, from quantity, then data types, then name."""
    tr = s.tracers[name]
    q = getattr(tr, "quantity", None) or ""
    if q == "galaxy_shear":
        return "shear"
    if q == "galaxy_density":
        return "density"
    kinds = set()
    for dt in s.get_data_types():
        for combo in s.get_tracer_combinations(dt):
            if name not in combo:
                continue
            if dt.startswith("galaxy_shearDensity"):
                kinds.add("shear" if combo[0] == name else "density")
            elif dt.startswith("galaxy_shear"):
                kinds.add("shear")
            elif dt.startswith("galaxy_density"):
                kinds.add("density")
    if len(kinds) == 1:
        return kinds.pop()
    if _SOURCE_RE.match(name):
        return "shear"
    if _LENS_RE.match(name):
        return "density"
    return None


def _tracer_rows(s) -> list[dict]:
    rows = []
    for name, tr in s.tracers.items():
        row = {"name": name, "type": type(tr).__name__.replace("Tracer", "") or "?",
               "quantity": getattr(tr, "quantity", None), "inferred_kind": _tracer_kind(s, name),
               "n_z": None, "z_min": None, "z_max": None, "z_mean": None}
        z = getattr(tr, "z", None)
        nz = getattr(tr, "nz", None)
        if z is not None and nz is not None and len(z):
            z = np.asarray(z, dtype=float)
            nz = np.asarray(nz, dtype=float)
            norm = np.trapezoid(nz, z) if len(z) > 1 else nz.sum()
            row.update({"n_z": int(len(z)), "z_min": float(z.min()), "z_max": float(z.max()),
                        "z_mean": float(np.trapezoid(z * nz, z) / norm) if norm > 0 else None})
        meta = getattr(tr, "metadata", None)
        if meta:
            row["metadata"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in meta.items()}
        rows.append(row)
    return rows


def _datatype_rows(s) -> list[dict]:
    rows = []
    for dt in s.get_data_types():
        combos = s.get_tracer_combinations(dt)
        tag = _x_tag(dt)
        xs = []
        n = 0
        for combo in combos:
            pts = s.get_data_points(dt, combo)
            n += len(pts)
            for p in pts:
                try:
                    xs.append(float(p.get_tag(tag)))
                except Exception:  # noqa: BLE001
                    pass
        rows.append({"data_type": dt, "n_points": n, "n_pairs": len(combos),
                     "tracer_pairs": [list(c) for c in combos], "x": tag,
                     "x_min": float(min(xs)) if xs else None, "x_max": float(max(xs)) if xs else None,
                     "has_window": any("window" in p.tags for c in combos[:1] for p in s.get_data_points(dt, c)[:1])})
    return rows


def _cov_dense(s):
    cov = s.covariance
    if cov is None:
        return None
    if hasattr(cov, "dense"):
        return np.asarray(cov.dense, dtype=float)
    return np.asarray(getattr(cov, "covmat", cov), dtype=float)


def _cov_info(s) -> dict:
    cov = _cov_dense(s)
    if cov is None:
        return {"present": False}
    info = {"present": True, "type": type(s.covariance).__name__, "shape": list(cov.shape)}
    d = np.diag(cov)
    info["diag_min"] = float(d.min())
    info["diag_max"] = float(d.max())
    info["n_nonpositive_diag"] = int((d <= 0).sum())
    if cov.shape[0] <= 3000:
        try:
            info["condition_number"] = float(np.linalg.cond(cov))
            w = np.linalg.eigvalsh(cov)
            info["min_eigenvalue"] = float(w.min())
            info["positive_definite"] = bool(w.min() > 0)
        except np.linalg.LinAlgError:
            info["condition_number"] = None
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(np.sqrt(np.abs(d)), np.sqrt(np.abs(d)))
    np.fill_diagonal(corr, 0.0)
    info["max_abs_offdiag_corr"] = float(np.nanmax(np.abs(corr))) if cov.shape[0] > 1 else 0.0
    info["is_diagonal"] = bool(info["max_abs_offdiag_corr"] < 1e-6)
    return info


def _write_table(path: Path, columns: dict[str, list], header_lines: list[str]) -> None:
    """CSV with '# key: value' headers that also allows string columns."""
    n = len(next(iter(columns.values())))
    with path.open("w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        w = csv.writer(f)
        w.writerow(list(columns.keys()))
        for i in range(n):
            w.writerow([(f"{v[i]:.8g}" if isinstance(v[i], (float, np.floating)) else v[i])
                        for v in columns.values()])


def _save(s, path: Path) -> None:
    if path.suffix.lower() in (".fits", ".fit"):
        s.save_fits(str(path), overwrite=True)
    else:
        s.save_hdf5(str(path), overwrite=True)


def _out_name(stem: str, output_format: str, input_path: Path) -> str:
    if output_format == "same":
        ext = ".fits" if input_path.suffix.lower() in (".fits", ".fit") else ".hdf5"
    else:
        ext = ".fits" if output_format == "fits" else ".hdf5"
    return stem + ext


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
@validate_call
def sacc_inspect(
    output_dir: Annotated[str, Field(min_length=1)],
    sacc_path: Annotated[str, Field(min_length=1, description="Path of a sacc file (.hdf5/.sacc/.fits; format auto-detected).")],
    write_nz: Annotated[bool, Field(description="Write nz_<tracer>.csv (z, nz) for every NZ tracer.")] = True,
    plot_covariance: Annotated[bool, Field(description="Write a correlation-matrix PNG when a covariance is present.")] = True,
) -> ArtifactResult:
    """Inspect a sacc file: tracers, data types with tracer pairs and ell/theta ranges, and the covariance.

    The first thing to call on any sacc file (TXPipe output, firecrown/augur
    input, a CCL-generated data vector). Reports per tracer: type (NZ/Map/
    Misc), quantity (galaxy_shear / galaxy_density / generic), number of z
    samples, z range and mean z, and the kind inferred from the data types
    it appears in ('shear'/'density' - what firecrown will assume). Per data
    type: number of points, tracer pairs, and the ell or theta [arcmin]
    range. Covariance: presence, class, shape, condition number, minimum
    eigenvalue (positive_definite must be true for a likelihood), and
    whether it is diagonal.

    Files: tracers.csv (one row per tracer), nz_<tracer>.csv (z, nz),
    datavector.csv (index, data_type, tracer1, tracer2, x, x_name, value,
    sigma [NaN without covariance]) and optionally covariance_corr.png.
    metadata.firecrown_ready says whether tracer names/quantities already
    follow the src{i}/lens{i} convention; if not, call
    sacc_prepare_for_firecrown next. If covariance.present is false and the
    file is harmonic-space, sacc_attach_gaussian_covariance can add one.
    """
    s = _load(sacc_path)
    path = Path(sacc_path).expanduser().resolve()
    outdir = resolve_outdir(output_dir)
    stem = path.stem.replace(".sacc", "")
    files: list[str] = []

    tracers = _tracer_rows(s)
    dtypes = _datatype_rows(s)
    cov = _cov_info(s)
    space = ("harmonic" if all(r["x"] == "ell" for r in dtypes) else
             "real" if all(r["x"] == "theta" for r in dtypes) else "mixed") if dtypes else "empty"

    # tracers.csv
    tpath = outdir / f"{stem}_tracers.csv"
    _write_table(tpath, {
        "name": [r["name"] for r in tracers], "type": [r["type"] for r in tracers],
        "quantity": [r["quantity"] or "" for r in tracers],
        "inferred_kind": [r["inferred_kind"] or "" for r in tracers],
        "n_z": [r["n_z"] if r["n_z"] is not None else "" for r in tracers],
        "z_min": [r["z_min"] if r["z_min"] is not None else "" for r in tracers],
        "z_max": [r["z_max"] if r["z_max"] is not None else "" for r in tracers],
        "z_mean": [r["z_mean"] if r["z_mean"] is not None else "" for r in tracers],
    }, [f"label: tracers of {path.name}", "quantity: sacc_tracers", f"source: {path}"])
    files.append(str(tpath))

    if write_nz:
        for name, tr in s.tracers.items():
            z = getattr(tr, "z", None)
            nz = getattr(tr, "nz", None)
            if z is None or nz is None or not len(z):
                continue
            p = outdir / f"{stem}_nz_{name}.csv"
            write_csv(p, {"z": np.asarray(z, float), "nz": np.asarray(nz, float)},
                      [f"label: n(z) {name} ({path.name})", "quantity: nz",
                       f"tracer: {name}", f"tracer_quantity: {getattr(tr, 'quantity', None)}"])
            files.append(str(p))

    # datavector.csv in sacc index order
    cov_dense = _cov_dense(s)
    sigma = np.sqrt(np.clip(np.diag(cov_dense), 0, None)) if cov_dense is not None else None
    cols = {"index": [], "data_type": [], "tracer1": [], "tracer2": [], "x": [], "x_name": [],
            "value": [], "sigma": []}
    for i, d in enumerate(s.data):
        tag = _x_tag(d.data_type)
        try:
            x = float(d.get_tag(tag))
        except Exception:  # noqa: BLE001
            x = float("nan")
        cols["index"].append(i)
        cols["data_type"].append(d.data_type)
        cols["tracer1"].append(d.tracers[0] if len(d.tracers) > 0 else "")
        cols["tracer2"].append(d.tracers[1] if len(d.tracers) > 1 else "")
        cols["x"].append(x)
        cols["x_name"].append("theta_arcmin" if tag == "theta" else "ell")
        cols["value"].append(float(d.value))
        cols["sigma"].append(float(sigma[i]) if sigma is not None else float("nan"))
    dpath = outdir / f"{stem}_datavector.csv"
    _write_table(dpath, cols, [f"label: data vector of {path.name}", "quantity: datavector",
                               f"n_points: {len(s.data)}", f"space: {space}",
                               "units: theta in arcmin, ell integer; sigma = sqrt(diag cov) (NaN if no covariance)"])
    files.append(str(dpath))

    if plot_covariance and cov_dense is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..plotting import rc_params

        d = np.sqrt(np.clip(np.diag(cov_dense), 1e-300, None))
        corr = cov_dense / np.outer(d, d)
        with plt.rc_context(rc_params()):
            fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            # data-type boundaries
            pos = 0
            for r in dtypes:
                pos += r["n_points"]
                ax.axhline(pos - 0.5, color="k", lw=0.5)
                ax.axvline(pos - 0.5, color="k", lw=0.5)
            ax.set_title(f"Correlation matrix: {path.name}")
            ax.set_xlabel("data index")
            ax.set_ylabel("data index")
            fig.colorbar(im, ax=ax, label="correlation")
            ppath = outdir / f"{stem}_covariance_corr.png"
            fig.savefig(ppath)
            plt.close(fig)
        files.append(str(ppath))

    names = list(s.tracers)
    names_ready = bool(names) and all(
        (re.match(r"^src\d+$", n) and _tracer_kind(s, n) == "shear")
        or (re.match(r"^lens\d+$", n) and _tracer_kind(s, n) == "density") for n in names)
    quantities_ready = bool(names) and all(
        getattr(s.tracers[n], "quantity", None) in ("galaxy_shear", "galaxy_density") for n in names)
    ready = names_ready  # firecrown infers shear/density from src*/lens* names
    n_pts = len(s.data)
    readiness = ("Tracer names and quantities are firecrown/augur-ready." if names_ready and quantities_ready else
                 "Tracer names are firecrown-ready (src{i}/lens{i}); quantities are not set (augur needs them: "
                 "sacc_prepare_for_firecrown sets galaxy_shear/galaxy_density)." if names_ready else
                 "Tracer names are NOT firecrown-ready: run sacc_prepare_for_firecrown.")
    msg = (f"{path.name}: {len(names)} tracers ({', '.join(names)}), {len(dtypes)} data types, "
           f"{n_pts} points, {space} space, covariance {'present' if cov['present'] else 'ABSENT'}"
           + (f" (cond {cov.get('condition_number', float('nan')):.3g})" if cov.get("condition_number") else "")
           + ". " + readiness)
    return ArtifactResult(
        status="success", files=files, message=msg,
        metadata={"sacc_path": str(path), "n_points": n_pts, "space": space, "tracers": tracers,
                  "data_types": dtypes, "covariance": cov, "firecrown_ready": ready,
                  "tracer_names_ready": names_ready, "quantities_ready": quantities_ready,
                  "sacc_version": getattr(s, "version", None)},
    )


@validate_call
def sacc_to_csv(
    output_dir: Annotated[str, Field(min_length=1)],
    sacc_path: Annotated[str, Field(min_length=1)],
    data_types: Annotated[list[str] | None, Field(description="Subset of data types to export; None = all.")] = None,
) -> ArtifactResult:
    """Export each data type of a sacc file to a CSV (tracer1, tracer2, ell|theta_arcmin, value, sigma).

    One file per data type, rows in sacc order, sigma = sqrt(diag cov) (NaN
    without a covariance). Use it to plot or diff measurements outside the
    sacc API; sacc_inspect already writes a single all-in-one datavector.csv.
    """
    s = _load(sacc_path)
    path = Path(sacc_path).expanduser().resolve()
    outdir = resolve_outdir(output_dir)
    stem = path.stem.replace(".sacc", "")
    present = s.get_data_types()
    wanted = data_types or present
    missing = [d for d in wanted if d not in present]
    if missing:
        raise ValueError(f"data types {missing} not in file; present: {present}")
    cov = _cov_dense(s)
    sigma_all = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov is not None else None
    files = []
    counts = {}
    for dt in wanted:
        tag = _x_tag(dt)
        cols = {"tracer1": [], "tracer2": [], ("theta_arcmin" if tag == "theta" else "ell"): [],
                "value": [], "sigma": []}
        for combo in s.get_tracer_combinations(dt):
            idx = s.indices(dt, combo)
            for i in idx:
                d = s.data[i]
                cols["tracer1"].append(d.tracers[0])
                cols["tracer2"].append(d.tracers[1] if len(d.tracers) > 1 else "")
                cols[list(cols)[2]].append(float(d.get_tag(tag)))
                cols["value"].append(float(d.value))
                cols["sigma"].append(float(sigma_all[i]) if sigma_all is not None else float("nan"))
        p = outdir / f"{stem}_{dt}.csv"
        _write_table(p, cols, [f"label: {dt} from {path.name}", f"quantity: {dt}",
                               "units: theta arcmin / ell integer; sigma from diag(cov)"])
        files.append(str(p))
        counts[dt] = len(cols["value"])
    return ArtifactResult(status="success", files=files,
                          message=f"Exported {len(files)} data types from {path.name}: {counts}.",
                          metadata={"sacc_path": str(path), "counts": counts,
                                    "has_covariance": cov is not None})


@validate_call
def sacc_prepare_for_firecrown(
    output_dir: Annotated[str, Field(min_length=1)],
    sacc_path: Annotated[str, Field(min_length=1, description="Input sacc (never modified).")],
    tracer_map: Annotated[dict[str, str] | None, Field(description="Explicit old->new tracer names, e.g. {'source_0': 'src0'}. Default rule: source_i/src_i -> src{i}, lens_i -> lens{i}.")] = None,
    keep_data_types: Annotated[list[str] | None, Field(description="Data types to keep (others dropped, covariance sliced accordingly). None = keep all. E.g. ['galaxy_shear_cl_ee','galaxy_shearDensity_cl_e','galaxy_density_cl'] drops B-modes.")] = None,
    set_quantity: Annotated[bool, Field(description="Set tracer.quantity to galaxy_shear / galaxy_density from the inferred kind.")] = True,
    output_format: Literal["hdf5", "fits", "same"] = "hdf5",
    output_name: Annotated[str | None, Field(description="File name (extension added); default <stem>_firecrown.<ext>.")] = None,
) -> ArtifactResult:
    """Write a copy of a sacc file with firecrown/augur tracer conventions: src{i}/lens{i} names, galaxy_shear/galaxy_density quantities, requested data types only.

    TXPipe writes tracers source_0.. / lens_0.. with quantity 'generic' and
    extra data types (B-modes, cross checks); firecrown infers shear vs
    density from names (src*/lens*) and augur REQUIRES src{i}/lens{i} with
    quantities set. This tool renames by rule (or by your tracer_map),
    sets quantities, drops data types not in keep_data_types (the
    covariance, if any, is sliced consistently), and reports whether a
    covariance is present (if not and the file is harmonic, call
    sacc_attach_gaussian_covariance next; if real-space, use TJPCov/augur).
    The input is never edited; the output goes to output_dir in the
    requested format (hdf5 default, fits, or same as input).
    """
    s = _load(sacc_path)
    path = Path(sacc_path).expanduser().resolve()
    outdir = resolve_outdir(output_dir)
    s = s.copy()
    present = s.get_data_types()
    if keep_data_types:
        missing = [d for d in keep_data_types if d not in present]
        if missing:
            raise ValueError(f"keep_data_types {missing} not in file; present: {present}")

    # 1. rename
    rename: dict[str, str] = {}
    for name in list(s.tracers):
        if tracer_map and name in tracer_map:
            new = tracer_map[name]
        else:
            m = _SOURCE_RE.match(name)
            if m:
                new = f"src{int(m.group(1))}"
            else:
                m = _LENS_RE.match(name)
                new = f"lens{int(m.group(1))}" if m else name
        if new != name:
            rename[name] = new
    if len(set(rename.values())) != len(rename) or any(v in s.tracers and v not in rename for v in rename.values()):
        raise ValueError(f"tracer renaming is not injective or collides with existing names: {rename}")
    for old, new in rename.items():
        s.rename_tracer(old, new)

    # 2. quantities
    quantities = {}
    unresolved = []
    for name, tr in s.tracers.items():
        if set_quantity:
            kind = _tracer_kind(s, name)
            if kind == "shear":
                tr.quantity = "galaxy_shear"
            elif kind == "density":
                tr.quantity = "galaxy_density"
            else:
                unresolved.append(name)
        quantities[name] = getattr(tr, "quantity", None)

    # 3. data types
    dropped = []
    if keep_data_types:
        for dt in present:
            if dt not in keep_data_types:
                s.remove_selection(data_type=dt)
                dropped.append(dt)

    stem = output_name or f"{path.stem.replace('.sacc', '')}_firecrown"
    out = outdir / _out_name(stem, output_format, path)
    _save(s, out)

    cov = _cov_info(s)
    names = list(s.tracers)
    ok_names = all(re.match(r"^(src|lens)\d+$", n) for n in names)
    space = "harmonic" if all("_cl" in d for d in s.get_data_types()) else "real"
    next_step = ("covariance present: call firecrown_build_likelihood" if cov["present"] else
                 ("NO covariance: call sacc_attach_gaussian_covariance (harmonic file)" if space == "harmonic"
                  else "NO covariance and real-space data: compute one with TJPCov or augur (cov_type tjpcov); "
                       "this family cannot build real-space covariances"))
    msg = (f"Wrote {out.name}: renamed {len(rename)} tracers ({rename or 'none needed'}), "
           f"quantities {quantities}, dropped data types {dropped or 'none'}, {len(s.data)} points. {next_step}.")
    if unresolved:
        msg += f" Could not infer shear/density for {unresolved}: set quantity via tracer_map names src*/lens*."
    return ArtifactResult(
        status="success", files=[str(out)], message=msg,
        metadata={"input": str(path), "output": str(out), "renamed": rename, "quantities": quantities,
                  "dropped_data_types": dropped, "kept_data_types": s.get_data_types(),
                  "n_points": len(s.data), "covariance": cov, "space": space,
                  "firecrown_ready": ok_names and not unresolved, "next_step": next_step},
    )


def _delta_ell_per_point(s, ell_edges: list[float] | None) -> np.ndarray:
    """Effective bandwidth per data point: from bandpower windows when present,
    else from user ell_edges, else from the midpoints between neighbouring ells."""
    import sacc as _sacc

    n = len(s.data)
    dl = np.full(n, np.nan)
    # group points by (data_type, tracers) to infer edges from neighbours
    groups: dict[tuple, list[int]] = {}
    for i, d in enumerate(s.data):
        groups.setdefault((d.data_type, tuple(d.tracers)), []).append(i)
    edges = np.asarray(ell_edges, dtype=float) if ell_edges else None
    for idx in groups.values():
        ells = np.array([float(s.data[i].get_tag("ell")) for i in idx])
        done = np.zeros(len(idx), bool)
        for j, i in enumerate(idx):
            w = s.data[i].tags.get("window")
            if isinstance(w, _sacc.BandpowerWindow):
                # Effective bandwidth of THIS point's window column: the number
                # of modes a weighted bandpower averages over, N = (sum w(2l+1))^2 /
                # sum w^2(2l+1), expressed as a width at the point's ell. Exact
                # for top-hat windows; right for dense NaMaster (mode-coupled)
                # windows too, where "where the weight is nonzero" is not.
                vals = np.asarray(w.values, float)
                wt = np.asarray(w.weight, float)
                col = int(s.data[i].tags.get("window_ind", 0) or 0)
                wt = wt[:, min(col, wt.shape[1] - 1)] if wt.ndim > 1 else wt
                two_l_plus_1 = 2.0 * vals + 1.0
                den = float(np.sum(wt ** 2 * two_l_plus_1))
                if den > 0:
                    n_modes = float(np.sum(wt * two_l_plus_1)) ** 2 / den
                    dl[i] = max(n_modes / (2.0 * ells[j] + 1.0), 1.0)
                    done[j] = True
        if done.all():
            continue
        if edges is not None:
            for j, i in enumerate(idx):
                if done[j]:
                    continue
                k = np.searchsorted(edges, ells[j], side="right") - 1
                if 0 <= k < len(edges) - 1:
                    dl[i] = edges[k + 1] - edges[k]
                    done[j] = True
        if done.all():
            continue
        order = np.argsort(ells)
        se = ells[order]
        if len(se) == 1:
            width = np.array([max(1.0, 0.2 * se[0])])
        else:
            mids = 0.5 * (se[1:] + se[:-1])
            lo = np.concatenate([[max(se[0] - (mids[0] - se[0]), 0.0)], mids])
            hi = np.concatenate([mids, [se[-1] + (se[-1] - mids[-1])]])
            width = np.maximum(hi - lo, 1.0)
        for jj, j in enumerate(order):
            if not done[j]:
                dl[idx[j]] = width[jj]
    return dl


@validate_call
def sacc_attach_gaussian_covariance(
    output_dir: Annotated[str, Field(min_length=1)],
    sacc_path: Annotated[str, Field(min_length=1, description="Harmonic-space sacc (galaxy_shear_cl_ee / galaxy_shearDensity_cl_e / galaxy_density_cl) without a covariance.")],
    f_sky: Annotated[float, Field(gt=0.0, le=1.0, description="Sky fraction of the footprint (LSST Y1 ~0.3, DES Y1 ~0.03, TXPipe example data ~1e-3-1e-2).")],
    n_gal: Annotated[dict[str, float] | float, Field(description="Effective number density per tracer in galaxies/arcmin^2: a dict {tracer: n} or one value for all tracers.")],
    sigma_e: Annotated[float, Field(gt=0.0, le=1.0, description="Per-component shape noise for shear tracers (LSST SRD 0.26).")] = 0.26,
    galaxy_bias: Annotated[dict[str, float] | float, Field(description="Linear bias per density tracer (dict or single value) used in the signal C_ell.")] = 1.5,
    cosmology: Annotated[CosmologyParams | None, Field(description="Cosmology for the theory C_ell (None = vanilla LCDM, halofit).")] = None,
    ell_edges: Annotated[list[float] | None, Field(description="Bandpower edges if the file has no window functions and the ells are bin centres; default infers widths from neighbouring ells.")] = None,
    overwrite_existing: Annotated[bool, Field(description="Replace a covariance already in the file (default: refuse).")] = False,
    output_format: Literal["hdf5", "fits", "same"] = "hdf5",
    output_name: Annotated[str | None, Field(description="File name (extension added); default <stem>_gausscov.<ext>.")] = None,
) -> ArtifactResult:
    """Attach a mode-counting Gaussian covariance to a harmonic-space sacc file that lacks one (e.g. TXPipe twopoint_data_fourier).

    Cov[(ab,l),(cd,l)] = (C_ac C_bd + C_ad C_bc) / ((2l+1) delta_l f_sky),
    zero between different ell bins, where C_xx includes the noise
    sigma_e^2/n_bar (shear) or 1/n_bar (density), n_bar = n_gal [arcmin^-2]
    converted to sr^-1. Signal C_ell are computed with pyccl from the
    file's own n(z) tracers at `cosmology` (Limber, halofit), density
    tracers with the given linear bias, no IA/magnification. delta_l comes
    from bandpower windows when the file has them, else from ell_edges,
    else from the spacing of neighbouring ells. This is augur's
    `gaus_internal` prescription: disconnected Gaussian only, no
    super-sample covariance, no mask coupling. Good for sanity checks and
    forecasts, not for a publication covariance (use TJPCov).

    Refuses real-space files (xi): compute those with TJPCov or augur
    (cov_type tjpcov). Supported data types: galaxy_shear_cl_ee,
    galaxy_shearDensity_cl_e, galaxy_density_cl - drop B-modes first with
    sacc_prepare_for_firecrown(keep_data_types=...). Writes a NEW file and
    reports the total S/N = sqrt(d^T C^-1 d) and the covariance diagnostics.
    """
    s = _load(sacc_path)
    path = Path(sacc_path).expanduser().resolve()
    outdir = resolve_outdir(output_dir)
    present = s.get_data_types()
    if not present:
        raise ValueError("the sacc file has no data points.")
    real = [d for d in present if "_xi" in d]
    if real:
        raise ValueError(
            f"{path.name} holds real-space data ({real}); the mode-counting Gaussian covariance is "
            "defined for angular power spectra only. For xi(theta) use TJPCov (tjpcov.covariance_calculator) "
            "or augur_generate_forecast_config with cov_type 'tjpcov', which build real-space covariances.")
    unsupported = [d for d in present if d not in HARMONIC_SUPPORTED]
    if unsupported:
        raise ValueError(f"unsupported data types {unsupported}; supported: {list(HARMONIC_SUPPORTED)}. "
                         "Drop the others with sacc_prepare_for_firecrown(keep_data_types=[...]).")
    if s.covariance is not None and not overwrite_existing:
        raise ValueError(f"{path.name} already has a covariance ({type(s.covariance).__name__}); "
                         "pass overwrite_existing=True to replace it.")
    s = s.copy()
    names = list(s.tracers)
    kinds = {n: _tracer_kind(s, n) for n in names}
    unknown = [n for n, k in kinds.items() if k is None]
    if unknown:
        raise ValueError(f"cannot tell shear from density for tracers {unknown}; "
                         "run sacc_prepare_for_firecrown first (names src*/lens*, quantities set).")
    n_gal_map = {n: float(n_gal) for n in names} if isinstance(n_gal, (int, float)) else dict(n_gal)
    missing = [n for n in names if n not in n_gal_map]
    if missing:
        raise ValueError(f"n_gal missing for tracers {missing} (give a dict with every tracer or one number).")
    bias_map = ({n: float(galaxy_bias) for n in names} if isinstance(galaxy_bias, (int, float))
                else dict(galaxy_bias))
    for n in names:
        if kinds[n] == "density" and n not in bias_map:
            raise ValueError(f"galaxy_bias missing for density tracer {n}.")
        tr = s.tracers[n]
        if getattr(tr, "z", None) is None or not len(tr.z):
            raise ValueError(f"tracer {n} has no n(z); only NZ tracers are supported.")

    # theory C_ell on the union of ells
    import pyccl as ccl

    spec = cosmology or CosmologyParams()
    cosmo = spec.build()
    ccl_tracers = {}
    for n in names:
        tr = s.tracers[n]
        z = np.asarray(tr.z, float)
        nz = np.asarray(tr.nz, float)
        if kinds[n] == "shear":
            ccl_tracers[n] = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
        else:
            ccl_tracers[n] = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz),
                                                    bias=(z, np.full_like(z, bias_map[n])))
    ells_all = np.array([float(d.get_tag("ell")) for d in s.data])
    ell_grid = np.unique(ells_all)
    used_pairs = set()
    for d in s.data:
        a, b = d.tracers[0], d.tracers[1]
        used_pairs.add(tuple(sorted((a, b))))
    # every pair among tracers that co-occur (needed for the cross terms)
    involved = sorted({t for p in used_pairs for t in p})
    cl: dict[tuple, np.ndarray] = {}
    for i, a in enumerate(involved):
        for b in involved[i:]:
            cl[(a, b)] = ccl.angular_cl(cosmo, ccl_tracers[a], ccl_tracers[b], ell_grid)
            cl[(b, a)] = cl[(a, b)]
    noise = {n: (sigma_e ** 2 / (n_gal_map[n] * ARCMIN2_PER_SR) if kinds[n] == "shear"
                 else 1.0 / (n_gal_map[n] * ARCMIN2_PER_SR)) for n in names}

    def c_tot(a, b, k):
        v = float(cl[(a, b)][k])
        return v + (noise[a] if a == b else 0.0)

    dl = _delta_ell_per_point(s, ell_edges)
    n = len(s.data)
    cov = np.zeros((n, n))
    ell_index = {float(l): k for k, l in enumerate(ell_grid)}
    by_ell: dict[float, list[int]] = {}
    for i, l in enumerate(ells_all):
        by_ell.setdefault(float(l), []).append(i)
    for l, idx in by_ell.items():
        k = ell_index[l]
        for i in idx:
            a, b = s.data[i].tracers[0], s.data[i].tracers[1]
            for j in idx:
                if j < i:
                    continue
                c, d_ = s.data[j].tracers[0], s.data[j].tracers[1]
                nmodes = (2.0 * l + 1.0) * 0.5 * (dl[i] + dl[j]) * f_sky
                val = (c_tot(a, c, k) * c_tot(b, d_, k) + c_tot(a, d_, k) * c_tot(b, c, k)) / nmodes
                cov[i, j] = cov[j, i] = val
    s.add_covariance(cov, overwrite=True)

    stem = output_name or f"{path.stem.replace('.sacc', '')}_gausscov"
    out = outdir / _out_name(stem, output_format, path)
    _save(s, out)

    data = np.asarray(s.mean, float)
    info = _cov_info(s)
    try:
        snr = float(np.sqrt(data @ np.linalg.solve(cov, data)))
    except np.linalg.LinAlgError:
        snr = None
    slug = param_slug({"f_sky": f_sky, "sigma_e": sigma_e, "cosmo": spec.slug()})
    jpath = outdir / f"{stem}_{slug}.json"
    write_json(jpath, {"f_sky": f_sky, "sigma_e": sigma_e, "n_gal_arcmin2": n_gal_map,
                       "galaxy_bias": {k: v for k, v in bias_map.items() if kinds.get(k) == "density"},
                       "noise_sr": noise, "cosmology": spec.firecrown_params(),
                       "delta_ell_min": float(np.nanmin(dl)), "delta_ell_max": float(np.nanmax(dl)),
                       "covariance": info, "snr_total": snr})
    return ArtifactResult(
        status="success", files=[str(out), str(jpath)],
        message=(f"Wrote {out.name} with a {n}x{n} Gaussian covariance (f_sky={f_sky}, sigma_e={sigma_e}, "
                 f"delta_ell {np.nanmin(dl):.3g}-{np.nanmax(dl):.3g}); total S/N "
                 f"{snr:.1f}" if snr else f"Wrote {out.name} with a {n}x{n} Gaussian covariance")
                + (f"; condition number {info.get('condition_number', 0):.3g}." if info.get("condition_number") else "."),
        metadata={"input": str(path), "output": str(out), "f_sky": f_sky, "sigma_e": sigma_e,
                  "n_gal_arcmin2": n_gal_map, "tracer_kinds": kinds, "cosmology": spec.label(),
                  "delta_ell_source": ("window" if any(isinstance(d.tags.get("window"), object) and d.tags.get("window") is not None for d in s.data)
                                       else "ell_edges" if ell_edges else "neighbour spacing"),
                  "snr_total": snr, "covariance": info, "n_points": n,
                  "formula": "Cov = (C13 C24 + C14 C23)/((2l+1) dl f_sky); noise sigma_e^2/nbar (shear), 1/nbar (density)"},
    )
