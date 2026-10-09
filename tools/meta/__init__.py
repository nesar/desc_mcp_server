"""Discovery tools: which DESC packages and tool families exist, the server's
skills, and parameter-name conversion between DESC conventions."""

import importlib
import inspect
from typing import Annotated, Literal

from pydantic import Field, validate_call

from ..common import ArtifactResult
from .registry import FAMILIES, PACKAGES, installed_version
from .skills import skill_index, skill_text

__all__ = ["list_desc_packages", "describe_desc_tool_family",
           "list_desc_skills", "load_desc_skill", "convert_cosmology_names"]


@validate_call
def list_desc_packages() -> ArtifactResult:
    """List the LSST DESC packages behind this server and the tool families built on them.

    Call this first. Each entry gives the package's role, which tool-name
    prefix exposes it (ccl_, sacc_, firecrown_, augur_, txpipe_), the version
    installed in this server's environment (None = not importable here, which
    for TXPipe is by design: it is composed and dispatched, never imported),
    and conventions worth knowing (units, parameter names). Then call
    describe_desc_tool_family for the family you need.
    """
    entries = {}
    for name, p in PACKAGES.items():
        entries[name] = {
            "role": p["role"], "families": p["families"], "status": p["status"],
            "installed_version": installed_version(p["import_name"]),
            "reference_clone_version": p["clone_version"], "repo": p["repo"],
        }
        for key in ("units", "parameter_names", "note", "examples"):
            if key in p:
                entries[name][key] = p[key]
    families = {k: v["summary"] for k, v in FAMILIES.items()}
    return ArtifactResult(
        status="success", files=[],
        message=f"{len(entries)} DESC packages; tool families: {', '.join(families)}. "
                "Skills (multi-tool recipes) via list_desc_skills.",
        metadata={"packages": entries, "families": families,
                  "conventions": {
                      "redshift": "tools take z (CCL internally uses a=1/(1+z))",
                      "k_units": "h/Mpc and (Mpc/h)^3 by default (k_units switch exposes CCL's 1/Mpc)",
                      "theta": "arcmin", "ell": "integer multipoles",
                      "parameter_names": "firecrown's: Omega_c Omega_b h n_s sigma8|A_s Omega_k "
                                         "Neff m_nu w0 wa; nuisance src0_delta_z lens0_bias ia_bias ...",
                      "data_exchange": "sacc files between families; CSV + PNG artifacts for humans",
                  }},
    )


@validate_call
def describe_desc_tool_family(
    family: Annotated[Literal["meta", "ccl", "sacc", "firecrown", "augur", "txpipe", "tjpcov", "namaster",
                              "smokescreen", "dispatch"],
                      Field(description="Tool-family key from list_desc_packages.")],
) -> ArtifactResult:
    """Describe one tool family: every tool with its one-line purpose, weight
    (light / dispatchable / heavy), and the family's conventions and caveats.

    Weight tells you where to run it: light tools always run on the server;
    heavy ones (Fisher matrices, chains, TXPipe pipelines) can be sent to
    ALCF/NERSC with set_dispatch or the client-side dispatch pack.
    """
    spec = FAMILIES[family]
    try:
        module = importlib.import_module(spec["module"])
    except Exception as exc:  # family not importable in this env
        return ArtifactResult(
            status="success", files=[],
            message=f"Family '{family}' is not importable in this environment: {exc}",
            metadata={"family": family, "summary": spec["summary"], "tools": {}})
    tools = {}
    for name in getattr(module, "__all__", []):
        fn = getattr(module, name)
        doc = inspect.getdoc(fn) or ""
        tools[name] = {"purpose": doc.split("\n\n")[0].replace("\n", " "),
                       "weight": getattr(fn, "weight", "light")}
    extra = {}
    for key in ("CONVENTIONS", "CAVEATS", "DISPATCH_KERNELS"):
        if hasattr(module, key):
            extra[key.lower()] = getattr(module, key)
    return ArtifactResult(
        status="success", files=[],
        message=f"Family '{family}': {len(tools)} tools. {spec['summary']}.",
        metadata={"family": family, "summary": spec["summary"], "tools": tools, **extra},
    )


@validate_call
def list_desc_skills() -> ArtifactResult:
    """List this server's skills: recipes for multi-tool DESC workflows.

    A skill is procedural know-how - which tools to combine, in what order,
    with what parameters, and how to sanity-check and report the results
    (e.g. an LSST 3x2pt Fisher forecast, or turning a TXPipe measurement
    into a likelihood evaluation). Returns one name + description per skill;
    when a task matches one, call load_desc_skill and follow it. These cover
    this server's tools only; other servers' skills are separate.
    """
    index = skill_index()
    return ArtifactResult(
        status="success", files=[],
        message=f"{len(index)} skills available. Load one with load_desc_skill "
                "when a task matches its description.",
        metadata={"skills": index},
    )


@validate_call
def load_desc_skill(
    name: Annotated[str, Field(min_length=1, description="Skill name from list_desc_skills, e.g. 'lsst-3x2pt-forecast'.")],
) -> ArtifactResult:
    """Load the full instructions of one of this server's skills and follow them."""
    text = skill_text(name)
    if text is None:
        raise ValueError(f"No skill named '{name}'. Available: {', '.join(sorted(skill_index()))}")
    return ArtifactResult(
        status="success", files=[],
        message=f"Loaded skill '{name}'. Follow these instructions.",
        metadata={"instructions": text},
    )


# firecrown/pyccl name -> (cobaya/CAMB name, cosmosis name, emulator-server name)
_NAME_MAP = {
    "Omega_c": ("omch2*h2", "omega_c", "Om-Ob"),
    "Omega_b": ("ombh2*h2", "omega_b", "Ob"),
    "h": ("H0/100", "h0", "h"),
    "n_s": ("ns", "n_s", "ns"),
    "sigma8": ("sigma8", "sigma_8", "sigma8"),
    "A_s": ("As", "A_s", "As"),
    "Omega_k": ("omk", "omega_k", None),
    "Neff": ("nnu", "nnu", None),
    "m_nu": ("mnu", "mnu", "mnu"),
    "w0": ("w", "w", "w0"),
    "wa": ("wa", "wa", "wa"),
    "T_CMB": ("TCMB", "TCMB", None),
}


@validate_call
def convert_cosmology_names(
    params: Annotated[dict[str, float], Field(description="Cosmology in firecrown/pyccl names, e.g. {'Omega_c':0.25,'Omega_b':0.05,'h':0.67,'n_s':0.96,'sigma8':0.81}.")],
    to: Annotated[Literal["cobaya", "cosmosis", "emulator"], Field(description="Target convention: 'cobaya' (CAMB-style ombh2/omch2/H0/ns/As/mnu/w/wa), 'cosmosis' (omega_c/omega_b/h0/n_s/sigma_8/w/wa), 'emulator' (the cosmic-emulator server: Om, Ob, h, ns, sigma8|As, mnu, w0, wa).")] = "emulator",
) -> ArtifactResult:
    """Translate a cosmology between the naming conventions used around this server.

    firecrown/pyccl names are the lingua franca here (Omega_c, Omega_b, h,
    n_s, sigma8|A_s, m_nu, w0, wa). Cobaya/CAMB want physical densities
    (ombh2 = Omega_b h^2, omch2 = Omega_c h^2, H0 = 100 h); CosmoSIS uses
    omega_c/omega_b/h0/sigma_8; the companion emulator server uses the total
    matter density Om = Omega_c + Omega_b (+ Omega_nu). Exact arithmetic,
    no fitting. Note the emulator server needs Om, not Omega_c, and the
    neutrino contribution Omega_nu = m_nu/(93.14 h^2) is included in Om.
    """
    p = dict(params)
    h = p.get("h")
    if h is None:
        raise ValueError("params must include h.")
    out: dict[str, float] = {}
    notes = []
    if to == "cobaya":
        if "Omega_b" in p:
            out["ombh2"] = p["Omega_b"] * h**2
        if "Omega_c" in p:
            out["omch2"] = p["Omega_c"] * h**2
        out["H0"] = 100.0 * h
        for src, dst in (("n_s", "ns"), ("sigma8", "sigma8"), ("A_s", "As"), ("Omega_k", "omk"),
                         ("Neff", "nnu"), ("m_nu", "mnu"), ("w0", "w"), ("wa", "wa"), ("T_CMB", "TCMB")):
            if src in p:
                out[dst] = p[src]
        notes.append("Cobaya+CAMB: omch2 excludes neutrinos (CAMB adds omnuh2 from mnu).")
    elif to == "cosmosis":
        for src, dst in (("Omega_c", "omega_c"), ("Omega_b", "omega_b"), ("h", "h0"), ("n_s", "n_s"),
                         ("sigma8", "sigma_8"), ("A_s", "A_s"), ("Omega_k", "omega_k"), ("Neff", "nnu"),
                         ("m_nu", "mnu"), ("w0", "w"), ("wa", "wa"), ("T_CMB", "TCMB")):
            if src in p:
                out[dst] = p[src]
        notes.append("CosmoSIS [cosmological_parameters] names; omega_m is derived by the consistency module.")
    else:
        omega_nu = p.get("m_nu", 0.0) / (93.14 * h**2)
        if "Omega_c" in p and "Omega_b" in p:
            out["Om"] = p["Omega_c"] + p["Omega_b"] + omega_nu
            out["Ob"] = p["Omega_b"]
        out["h"] = h
        for src, dst in (("n_s", "ns"), ("sigma8", "sigma8"), ("A_s", "As"), ("m_nu", "mnu"),
                         ("w0", "w0"), ("wa", "wa")):
            if src in p:
                out[dst] = p[src]
        notes.append(f"Om includes Omega_nu = {omega_nu:.5f} (m_nu/(93.14 h^2)); Omega_k, Neff, T_CMB have no emulator-server counterpart.")
    return ArtifactResult(
        status="success", files=[],
        message=f"Converted {len(out)} parameters to the '{to}' convention.",
        metadata={"params": {k: round(float(v), 10) for k, v in out.items()},
                  "to": to, "notes": notes},
    )
