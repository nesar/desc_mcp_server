"""The ONE cosmology specification every family shares.

Parameter names are firecrown's (== pyccl's): Omega_c, Omega_b, h, n_s,
sigma8 | A_s, Omega_k, Neff, m_nu, w0, wa, T_CMB. The same object yields
(a) pyccl.Cosmology kwargs for the ccl family, (b) a flat float dict for
firecrown's ParamsMap / augur's `cosmo:` block, and (c) a stable slug for
file names and caches.

Modelling switches (transfer function, nonlinear prescription, baryons,
mu-Sigma MG) live here too so that "the cosmology" an agent passes between
tools is one JSON object. Everything is JSON-safe; the pyccl object is built
lazily and cached per process.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .common import get_cached, param_slug

TransferFunction = Literal["boltzmann_camb", "bbks", "eisenstein_hu",
                           "eisenstein_hu_nowiggles", "boltzmann_class"]
MatterPower = Literal["halofit", "linear", "camb_hmcode", "cosmicemu_mt4",
                      "cosmicemu_mt2", "bacco"]
Baryons = Literal["none", "schneider15", "vandaalen19"]

PRESETS: dict[str, dict] = {
    # pyccl.CosmologyVanillaLCDM
    "vanilla": {"Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67, "n_s": 0.96, "sigma8": 0.81},
    # Planck 2018 TT,TE,EE+lowE+lensing (base LCDM), sigma8 convention
    "planck18": {"Omega_c": 0.2607, "Omega_b": 0.0490, "h": 0.6766, "n_s": 0.9665,
                 "sigma8": 0.8102},
    # DESC SRD fiducial (augur examples/srd_y1_3x2.yml)
    "desc_srd": {"Omega_c": 0.2664, "Omega_b": 0.0492, "h": 0.6727, "n_s": 0.9645,
                 "sigma8": 0.831},
}


class CosmologyParams(BaseModel):
    """Cosmology + modelling choices shared by every DESC tool family.

    Give exactly one of sigma8 / A_s (CCL rule; analytic transfer functions
    bbks / eisenstein_hu REQUIRE sigma8). m_nu is the summed neutrino mass in
    eV. The emulator-backed nonlinear options have training boxes (CosmicEmu:
    sigma8 0.7-0.9, z <= 2; bacco: z <= 1.5) - check the tool's metadata
    'in_training_box'. 'camb_hmcode' = CAMB HMcode-2020 with baryon feedback
    set by hmcode_logT_AGN (7.8 ~ TNG-like); it cannot be combined with
    mu-Sigma modified gravity (CCL refuses).
    """

    model_config = {"extra": "forbid"}

    preset: Annotated[Literal["vanilla", "planck18", "desc_srd"] | None, Field(
        description="Fill Omega_c/Omega_b/h/n_s/sigma8 from a named set; explicit fields override.")] = None
    Omega_c: Annotated[float, Field(ge=0.05, le=0.6)] = 0.25
    Omega_b: Annotated[float, Field(ge=0.02, le=0.08)] = 0.05
    h: Annotated[float, Field(ge=0.5, le=0.9)] = 0.67
    n_s: Annotated[float, Field(ge=0.8, le=1.1)] = 0.96
    sigma8: Annotated[float | None, Field(ge=0.4, le=1.3, description="sigma8 at z=0; give this OR A_s")] = 0.81
    A_s: Annotated[float | None, Field(ge=5e-10, le=5e-9, description="Primordial amplitude; give this OR sigma8")] = None
    Omega_k: Annotated[float, Field(ge=-0.3, le=0.3)] = 0.0
    Neff: Annotated[float, Field(ge=1.0, le=5.0)] = 3.044
    m_nu: Annotated[float, Field(ge=0.0, le=1.0, description="Sum of neutrino masses [eV]; split per mass_split")] = 0.0
    mass_split: Literal["normal", "inverted", "equal", "single"] = "normal"
    w0: Annotated[float, Field(ge=-2.0, le=-0.3)] = -1.0
    wa: Annotated[float, Field(ge=-3.0, le=1.0)] = 0.0
    T_CMB: Annotated[float, Field(ge=2.0, le=3.5)] = 2.7255
    transfer_function: TransferFunction = "boltzmann_camb"
    matter_power_spectrum: MatterPower = "halofit"
    baryons: Baryons = "none"
    baryon_params: Annotated[dict[str, float], Field(
        description="Model parameters: schneider15 {log10Mc=14.079, eta_b=0.5, k_s=55.0 [h/Mpc]}; vandaalen19 {fbar=0.7}")] = {}
    hmcode_logT_AGN: Annotated[float, Field(ge=7.0, le=8.5)] = 7.8
    mg_mu0: Annotated[float, Field(ge=-1.0, le=1.0, description="mu-Sigma MG: mu_0 (0 = GR)")] = 0.0
    mg_sigma0: Annotated[float, Field(ge=-1.0, le=1.0, description="mu-Sigma MG: Sigma_0 (0 = GR)")] = 0.0

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data):
        if isinstance(data, dict) and data.get("preset"):
            base = dict(PRESETS[data["preset"]])
            # an explicit A_s replaces the preset's sigma8
            if data.get("A_s") is not None and "sigma8" not in data:
                base.pop("sigma8", None)
                base["sigma8"] = None
            base.update({k: v for k, v in data.items() if k != "preset"})
            base["preset"] = data["preset"]
            return base
        return data

    @model_validator(mode="after")
    def _check(self):
        if (self.sigma8 is None) == (self.A_s is None):
            raise ValueError("Give exactly one of sigma8 or A_s.")
        if self.A_s is not None and self.transfer_function in ("bbks", "eisenstein_hu",
                                                                "eisenstein_hu_nowiggles"):
            raise ValueError(f"transfer_function={self.transfer_function} requires sigma8 "
                             "(analytic transfer functions cannot use A_s).")
        if (self.mg_mu0 or self.mg_sigma0) and self.matter_power_spectrum == "camb_hmcode":
            raise ValueError("mu-Sigma MG cannot be combined with matter_power_spectrum="
                             "'camb_hmcode' (CCL refuses to rescale CAMB's nonlinear P(k)); "
                             "use 'halofit' or 'linear'.")
        return self

    # ---- views -----------------------------------------------------------
    def firecrown_params(self) -> dict[str, float]:
        """Flat float dict with firecrown/pyccl names (for ParamsMap, augur cosmo:)."""
        out = {"Omega_c": self.Omega_c, "Omega_b": self.Omega_b, "h": self.h,
               "n_s": self.n_s, "Omega_k": self.Omega_k, "Neff": self.Neff,
               "m_nu": self.m_nu, "w0": self.w0, "wa": self.wa, "T_CMB": self.T_CMB}
        if self.sigma8 is not None:
            out["sigma8"] = self.sigma8
        else:
            out["A_s"] = self.A_s
        return {k: float(v) for k, v in out.items()}

    def ccl_kwargs(self) -> dict:
        """Keyword arguments for pyccl.Cosmology (objects built lazily)."""
        kw = dict(self.firecrown_params())
        kw["mass_split"] = self.mass_split
        kw["transfer_function"] = self.transfer_function
        extra = {}
        mps = self.matter_power_spectrum
        if mps == "camb_hmcode":
            kw["matter_power_spectrum"] = "camb"
            extra["camb"] = {"halofit_version": "mead2020_feedback",
                             "HMCode_logT_AGN": self.hmcode_logT_AGN, "kmax": 20.0}
        elif mps in ("halofit", "linear"):
            kw["matter_power_spectrum"] = mps
        else:  # emulator objects are attached in build()
            kw["matter_power_spectrum"] = "halofit"
        if extra:
            kw["extra_parameters"] = extra
        return kw

    def slug(self) -> str:
        return param_slug(self.model_dump())

    def label(self) -> str:
        amp = f"sigma8={self.sigma8:g}" if self.sigma8 is not None else f"A_s={self.A_s:.3g}"
        bits = [f"Om={self.Omega_c + self.Omega_b:.3f}", amp, f"h={self.h:g}"]
        if self.w0 != -1.0 or self.wa != 0.0:
            bits.append(f"w0={self.w0:g}, wa={self.wa:g}")
        if self.m_nu:
            bits.append(f"mnu={self.m_nu:g}eV")
        if self.mg_mu0 or self.mg_sigma0:
            bits.append(f"mu0={self.mg_mu0:g}, Sigma0={self.mg_sigma0:g}")
        return ", ".join(bits)

    # ---- pyccl object ----------------------------------------------------
    def build(self):
        """Cached pyccl.Cosmology for this spec (splines are cached on it)."""
        return get_cached(f"ccl_cosmo_{self.slug()}", lambda: _build_ccl(self))

    def describe(self) -> dict:
        """JSON summary incl. derived parameters (requires pyccl)."""
        cosmo = self.build()
        out = {"params": self.firecrown_params(),
               "modelling": {"transfer_function": self.transfer_function,
                             "matter_power_spectrum": self.matter_power_spectrum,
                             "baryons": self.baryons, "baryon_params": self.baryon_params,
                             "mg_mu0": self.mg_mu0, "mg_sigma0": self.mg_sigma0},
               "derived": {"Omega_m": float(cosmo["Omega_m"]),
                           "Omega_nu_mass": float(cosmo["Omega_nu_mass"]),
                           "Omega_Lambda": float(cosmo["Omega_l"]),
                           "H0_km_s_Mpc": 100.0 * self.h},
               "label": self.label()}
        return out


def _build_ccl(spec: "CosmologyParams"):
    import pyccl as ccl

    kw = spec.ccl_kwargs()
    mps = spec.matter_power_spectrum
    if mps == "cosmicemu_mt4":
        kw["matter_power_spectrum"] = ccl.CosmicemuMTIVPk("tot")
    elif mps == "cosmicemu_mt2":
        kw["matter_power_spectrum"] = ccl.CosmicemuMTIIPk("tot")
    elif mps == "bacco":
        kw["matter_power_spectrum"] = ccl.BaccoemuNonlinear()
    if spec.baryons == "schneider15":
        kw["baryonic_effects"] = ccl.BaryonsSchneider15(**spec.baryon_params)
    elif spec.baryons == "vandaalen19":
        kw["baryonic_effects"] = ccl.BaryonsvanDaalen19(**spec.baryon_params)
    if spec.mg_mu0 or spec.mg_sigma0:
        from pyccl.modified_gravity import MuSigmaMG
        kw["mg_parametrization"] = MuSigmaMG(mu_0=spec.mg_mu0, sigma_0=spec.mg_sigma0)
    return ccl.Cosmology(**kw)
