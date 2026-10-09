"""Publication-style figures for every plotting tool on this server.

Design constraints:
- NO system LaTeX: hosted VMs have no TeX install. Math is rendered with
  matplotlib's built-in mathtext using the STIX fonts (Times-like, the
  look of journal figures), which ship with matplotlib itself.
- Nothing may overflow the canvas: axis labels are short physics labels
  derived from each file's `quantity`/column names (never the long curve
  labels), titles are wrapped, and the legend moves below the axes when it
  has many or long entries. Layout is constrained_layout + a tight bbox.
- Colors: a fixed-order, colorblind-validated categorical palette, always
  paired with distinct linestyles (identity is never color-alone). Curves
  that differ only by one numeric parameter use a single-hue sequential
  ramp with a colorbar instead.
"""

import re
from pathlib import Path

import numpy as np

# Okabe-Ito-derived, reordered so adjacent slots stay separable under
# deutan/protan CVD (validated: adjacent-pair CVD dE >= 11, normal >= 18).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#882255",
           "#B8860B", "#CC79A7", "#6B4C9A", "#E07B39"]
LINESTYLES = ["-", "--", "-.", ":", (0, (6, 1.5, 1, 1.5, 1, 1.5)),
              (0, (1, 1)), (0, (8, 2)), (0, (3, 1, 1, 1))]
INK = "#222222"
MUTED = "#6b6b6b"

# quantity -> (y-axis label, log-scale y?, draw a y=1 reference line?)
QUANTITY_AXES = {
    "power_spectrum": (r"$P(k)\ [h^{-3}\,\mathrm{Mpc}^{3}]$", True, False),
    "boost": (r"$B(k) = P_{\rm MG}/P_{\Lambda\rm CDM}$", False, True),
    "suppression": (r"$S(k) = P_{\rm hydro}/P_{\rm DMO}$", False, True),
    "ratio": (r"ratio", False, True),
    "composed": (r"product of input ratios", False, True),
    "hmf": (r"$\mathrm{d}n/\mathrm{d}\ln M\ [h^{3}\,\mathrm{Mpc}^{-3}]$", True, False),
    "multipoles": (r"$P_0(k)\ [h^{-3}\,\mathrm{Mpc}^{3}]$", False, False),
    "p1d": (r"$P_{\rm 1D}(k_\parallel)\ [\mathrm{Mpc}]$", True, False),
    # DESC / CCL family quantities
    "angular_cl": (r"$C_\ell$", True, False),
    "correlation_function": (r"$\xi(\theta)$", True, False),
    "correlation_3d": (r"$\xi(r)$", False, False),
    "nz": (r"$n(z)$", False, False),
    "background": (r"distance [Mpc]", False, False),
    "hmf_ccl": (r"$\mathrm{d}n/\mathrm{d}\log_{10}M\ [\mathrm{Mpc}^{-3}]$", True, False),
    "halo_model_pk": (r"$P(k)\ [h^{-3}\,\mathrm{Mpc}^{3}]$", True, False),
}
DIMENSIONLESS = {"boost", "suppression", "ratio", "composed"}

# first-column name -> x-axis label
X_AXES = {
    "k_h_per_Mpc": r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
    "ell": r"Multipole $\ell$",
    "k_par_1_per_Mpc": r"$k_\parallel\ [\mathrm{Mpc}^{-1}]$",
    "M200c_Msun_per_h": r"$M_{200c}\ [h^{-1}\,M_\odot]$",
    "M200m_Msun_per_h": r"$M_{200m}\ [h^{-1}\,M_\odot]$",
    "M500c_Msun_per_h": r"$M_{500c}\ [h^{-1}\,M_\odot]$",
    "Mfof_Msun_per_h": r"$M_{\rm FoF}\ [h^{-1}\,M_\odot]$",
    # DESC / CCL family x axes
    "k_per_Mpc": r"$k\ [\mathrm{Mpc}^{-1}]$",
    "z": r"$z$",
    "theta_arcmin": r"$\theta$ [arcmin]",
    "r_Mpc_over_h": r"$r\ [h^{-1}\,\mathrm{Mpc}]$",
    "r_Mpc": r"$r\ [\mathrm{Mpc}]$",
    "M_Msun": r"$M\ [M_\odot]$",
}

# y-column name -> label, used when `quantity` alone is not specific enough
Y_COLUMNS = {
    "Dl_muK2": (r"$\mathcal{D}_\ell\ [\mu\mathrm{K}^2]$", False),
    "Cl_kappa": (r"$C_\ell^{\kappa\kappa}$", True),
    # DESC / CCL family y columns
    "Pk_lin_Mpc3": (r"$P_{\rm lin}(k)\ [\mathrm{Mpc}^{3}]$", True),
    "Pk_nl_Mpc3": (r"$P_{\rm nl}(k)\ [\mathrm{Mpc}^{3}]$", True),
    "Pk_lin_Mpc3_over_h3": (r"$P_{\rm lin}(k)\ [h^{-3}\,\mathrm{Mpc}^{3}]$", True),
    "Pk_nl_Mpc3_over_h3": (r"$P_{\rm nl}(k)\ [h^{-3}\,\mathrm{Mpc}^{3}]$", True),
    "chi_Mpc": (r"$\chi(z)\ [\mathrm{Mpc}]$", False),
    "nz_total": (r"$n(z)$", False),
    "xi_r": (r"$\xi(r)$", False),
}

# Tool argument name -> mathtext symbol (scan colorbars and titles).
PARAM_SYMBOLS = {
    "z": r"$z$", "z_source": r"$z_s$", "Om": r"$\Omega_{\rm m}$",
    "Ob": r"$\Omega_{\rm b}$", "h": r"$h$", "ns": r"$n_s$", "n_s": r"$n_s$",
    "sigma8": r"$\sigma_8$", "sigma_8": r"$\sigma_8$", "As": r"$A_s$",
    "mnu": r"$\sum m_\nu$ [eV]", "w0": r"$w_0$", "w_0": r"$w_0$",
    "wa": r"$w_a$", "w_a": r"$w_a$",
    "minus_log10_fR0": r"$-\log_{10}|f_{R0}|$", "H0rc": r"$H_0 r_c$",
    "f_phi": r"$f_\phi$", "log10_M_c": r"$\log_{10} M_c$",
    "fb_a": r"$f_{b,a}$", "fb_pow": r"$f_{b,{\rm pow}}$",
    "ln10As": r"$\ln(10^{10}A_s)$", "omega_b": r"$\omega_b$",
    "omega_cdm": r"$\omega_c$", "tau": r"$\tau$", "b1": r"$b_1$",
}


def param_symbol(name: str) -> str:
    return PARAM_SYMBOLS.get(name, name)


# Readable replacements applied to auto-generated curve labels.
_LABEL_SUBS = [
    (r"-log10\|fR0\|=([\d.]+)", lambda m: rf"$|f_{{R0}}|=10^{{-{float(m.group(1)):g}}}$"),
    (r"\bfofr\b", "f(R)"),
    (r"\bndgp\b", "nDGP"),
    (r"\bcubic_galileon\b", "cubic Galileon"),
    (r"\bH0rc=([\d.]+)", lambda m: rf"$H_0 r_c={float(m.group(1)):g}$"),
    (r"\bf_phi=([\d.]+)", lambda m: rf"$f_\phi={float(m.group(1)):g}$"),
    (r"\bsigma_?8=", r"$\\sigma_8$="),
    (r"\bOm=", r"$\\Omega_{\\rm m}$="),
    (r"\bz_s~", r"$z_s\\approx$"),
    (r"\bbaryon suppression\b", "S(k)"),
    (r"\bboost\b", "B(k)"),
]


def rc_params(base_size: float = 12.0) -> dict:
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "axes.titlesize": base_size + 1,
        "legend.fontsize": base_size - 1.5,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "savefig.dpi": 200,
        "figure.dpi": 100,
        "axes.unicode_minus": True,
    }


def prettify_label(label: str) -> str:
    out = label
    for pattern, repl in _LABEL_SUBS:
        out = re.sub(pattern, repl, out)
    return out


def _wrap(text: str, width: int) -> str:
    """Wrap plain text without breaking inside $...$ math spans."""
    if len(text) <= width:
        return text
    tokens = re.findall(r"\$[^$]*\$|\S+", text)
    lines, current = [], ""
    for tok in tokens:
        candidate = f"{current} {tok}".strip()
        # math markup is much longer than what it renders to
        visible = len(re.sub(r"\\[a-zA-Z]+|[{}$^_\\]", "", candidate))
        if visible > width and current:
            lines.append(current)
            current = tok
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def shorten_labels(labels: list[str]) -> tuple[list[str], str]:
    """Strip the words every label shares; return (short labels, common text).

    Five curves labeled 'fofr boost (-log10|fR0|=6.0) z=0 x baryon
    suppression spk (fb_a=0.4, fb_pow=0.3, SO=200) z=0' differ only in a
    few key=value tokens — those go in the legend, the shared part becomes
    the legend title.
    """
    if len(labels) < 2:
        return labels, ""
    split = [re.findall(r"[^\s(),\[\]]+", lab) for lab in labels]
    common = set(split[0]).intersection(*map(set, split[1:]))

    def join(tokens):
        out = ""
        for t in tokens:
            sep = "" if not out else (", " if "=" in t and "=" in out.split()[-1] else " ")
            out += sep + t
        return out

    short = [join(t for t in toks if t not in common) or lab
             for toks, lab in zip(split, labels)]
    if len(set(short)) < len(short):        # stripping made labels collide
        return labels, ""
    seen, shared_tokens = set(), []
    for t in split[0]:
        if t in common and t not in seen and t not in ("x", "/"):
            seen.add(t)
            shared_tokens.append(t)
    return short, join(shared_tokens)


def _axis_spec(quantities, headers, columns):
    """Return (ylabel, logy, unity_line) for a set of curves."""
    qset = set(quantities)
    ycol = columns[0][1] if columns and len(columns[0]) > 1 else ""
    if ycol in Y_COLUMNS:
        label, logy = Y_COLUMNS[ycol]
        if headers[0].get("spectrum") == "PP":
            label = r"$C_\ell^{\phi\phi}$ (Capse native)"
        return label, logy, False
    symbols = {h.get("symbol") for h in headers}
    if len(symbols) == 1 and None not in symbols:
        q = quantities[0]
        logy = QUANTITY_AXES.get(q, ("", False, False))[1]
        return f"${symbols.pop()}$", logy, q in DIMENSIONLESS
    if qset <= DIMENSIONLESS and len(qset) > 1:
        return "dimensionless ratio", False, True
    q = quantities[0]
    if q in QUANTITY_AXES:
        return QUANTITY_AXES[q]
    if q.startswith("subgrid_"):
        return f"{q.removeprefix('subgrid_')} (emulator-native units)", False, False
    units = headers[0].get("units", "")
    ylabel = units.split(",")[-1].strip() if "," in units else (q or "value")
    return ylabel, True, False


def _x_label(xcol: str, header: dict) -> str:
    if xcol in X_AXES:
        return X_AXES[xcol]
    units = header.get("units", "")
    return units.split(",")[0].strip() if "," in units else xcol


def _uncertainty(cols: dict, y: np.ndarray):
    """Absolute 1-sigma band from a file's uncertainty column, if any."""
    if "gp_std" in cols and np.any(np.asarray(cols["gp_std"]) > 0):
        return np.asarray(cols["gp_std"], dtype=float)
    if "emulator_rel_std" in cols and np.any(np.asarray(cols["emulator_rel_std"]) > 0):
        return np.asarray(cols["emulator_rel_std"], dtype=float) * np.abs(y)
    return None


def plot_files(
    curve_files: list[str],
    output_path: Path,
    *,
    title: str | None = None,
    labels: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    logx: bool | None = None,
    logy: bool | None = None,
    ratio_panel: str = "auto",
    reference_index: int = 0,
    color_values: list[float] | None = None,
    colorbar_label: str | None = None,
    save_pdf: bool = False,
) -> dict:
    """Render curves from server CSVs. Returns plot metadata.

    ratio_panel: "auto" (on for >=2 curves of a non-dimensionless quantity),
    "on", or "off". color_values switches to a sequential colormap keyed
    on one numeric parameter (one value per file) with a colorbar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib import cm

    from .common import read_csv

    curves = []
    for path_str in curve_files:
        header, cols = read_csv(path_str)
        names = list(cols.keys())
        curves.append({
            "header": header, "names": names,
            "label": header.get("label", Path(path_str).stem),
            "quantity": header.get("quantity", "unknown"),
            "x": np.asarray(cols[names[0]], dtype=float),
            "y": np.asarray(cols[names[1]], dtype=float),
            "err": _uncertainty(cols, np.asarray(cols[names[1]], dtype=float)),
        })
    n = len(curves)

    quantities = [c["quantity"] for c in curves]
    auto_ylabel, auto_logy, unity = _axis_spec(
        quantities, [c["header"] for c in curves],
        [c["names"] for c in curves])
    all_positive = all(np.all(c["y"][np.isfinite(c["y"])] > 0) for c in curves)
    if logy is None:
        logy = auto_logy and all_positive
    x_all = np.concatenate([c["x"] for c in curves])
    x_pos = x_all[np.isfinite(x_all) & (x_all > 0)]
    is_cmb = curves[0]["names"][1] == "Dl_muK2"
    if logx is None:
        logx = (not is_cmb and len(x_pos) == len(x_all)
                and x_pos.max() / x_pos.min() > 30)

    if labels is not None:
        if len(labels) != n:
            raise ValueError(f"labels has {len(labels)} entries for {n} files.")
        legend_labels, shared = list(labels), ""
    else:
        raw = [c["label"] for c in curves]
        legend_labels, shared = shorten_labels(raw)
        legend_labels = [prettify_label(s) for s in legend_labels]
        shared = prettify_label(shared)

    want_ratio = {"on": n >= 2, "off": False,
                  "auto": n >= 2 and not unity}[ratio_panel]
    if want_ratio and not 0 <= reference_index < n:
        raise ValueError("reference_index is out of range for the files given.")

    # sequential coloring (scan plots) vs categorical
    sequential = color_values is not None
    if sequential:
        if len(color_values) != n:
            raise ValueError("color_values needs one value per file.")
        vmin, vmax = float(min(color_values)), float(max(color_values))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "seq", plt.get_cmap("Blues")(np.linspace(0.35, 1.0, 256)))
        colors = [cmap(norm(v)) for v in color_values]
        styles = ["-"] * n
    else:
        if n > len(PALETTE):
            raise ValueError(
                f"{n} categorical curves exceed the {len(PALETTE)}-color "
                "palette; split into panels, or pass color_values to use a "
                "sequential colormap keyed on the varied parameter.")
        colors = PALETTE[:n]
        styles = LINESTYLES[:n]

    long_legend = (not sequential) and (
        n > 4 or max(len(re.sub(r"[${}\\^_]", "", s)) for s in legend_labels) > 34)

    with plt.rc_context(rc_params()):
        width = 6.8
        if want_ratio:
            fig, (ax, axr) = plt.subplots(
                2, 1, figsize=(width, 6.2), sharex=True, layout="constrained",
                gridspec_kw={"height_ratios": [2.4, 1]})
        else:
            fig, ax = plt.subplots(figsize=(width, 4.6), layout="constrained")
            axr = None
        if not sequential:
            fig.get_layout_engine().set(h_pad=0.06)

        for i, c in enumerate(curves):
            ax.plot(c["x"], c["y"], color=colors[i], linestyle=styles[i],
                    label=None if sequential else legend_labels[i], zorder=3)
            if c["err"] is not None:
                ax.fill_between(c["x"], c["y"] - c["err"], c["y"] + c["err"],
                                color=colors[i], alpha=0.18, linewidth=0,
                                zorder=2)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if unity:
            ax.axhline(1.0, color=MUTED, linewidth=0.8, zorder=1)
        ax.set_ylabel(_wrap(ylabel or auto_ylabel, 38))

        x_text = xlabel or _x_label(curves[0]["names"][0], curves[0]["header"])
        if axr is not None:
            ref = curves[reference_index]
            for i, c in enumerate(curves):
                if i == reference_index:
                    continue
                denom = np.interp(c["x"], ref["x"], ref["y"],
                                  left=np.nan, right=np.nan)
                axr.plot(c["x"], c["y"] / denom, color=colors[i],
                         linestyle=styles[i], zorder=3)
            axr.axhline(1.0, color=colors[reference_index],
                        linestyle=styles[reference_index], linewidth=1.2)
            axr.set_ylabel("ratio to ref.")
            axr.set_xlabel(x_text)
            if not sequential:
                legend_labels[reference_index] += " (ref.)"
                ax.get_lines()[reference_index].set_label(
                    legend_labels[reference_index])
        else:
            ax.set_xlabel(x_text)

        if title:
            ax.set_title(_wrap(title, 62), loc="left", pad=8)

        if sequential:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, ax=[a for a in (ax, axr) if a is not None],
                                pad=0.02, aspect=30)
            cbar.set_label(colorbar_label or "scanned parameter")
            cbar.ax.minorticks_off()
        elif n > 1 or labels is not None:
            wrapped = [_wrap(s, 48 if long_legend else 34) for s in legend_labels]
            handles = ax.get_lines()[:n]
            legend_title = _wrap(f"All: {shared}", 70 if long_legend else 40) if shared else None
            if long_legend:
                ncol = 1 if max(len(s) for s in legend_labels) > 40 else 2
                leg = fig.legend(handles, wrapped, loc="outside lower center",
                                 ncol=ncol, borderaxespad=0.2,
                                 title=legend_title)
            else:
                leg = ax.legend(handles, wrapped, loc="best",
                                title=legend_title)
            if legend_title:
                leg.get_title().set_fontsize(10)
                leg.get_title().set_color(MUTED)
        if sequential and shared:
            # no legend to carry the shared text: it becomes a muted subtitle
            ax.set_title(_wrap((f"{title}\n" if title else "") + f"All: {shared}", 62),
                         loc="left", pad=8, fontsize=10.5, color=MUTED)

        out = [str(output_path)]
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08,
                    facecolor="white")
        if save_pdf:
            pdf = output_path.with_suffix(".pdf")
            fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
            out.append(str(pdf))
        plt.close(fig)

    return {"files": out, "labels": legend_labels, "shared_caption": shared,
            "ratio_panel": axr is not None, "logx": bool(logx),
            "logy": bool(logy), "quantities": quantities}

