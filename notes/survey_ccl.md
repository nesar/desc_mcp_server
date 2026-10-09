# Survey: CCL (pyccl) + CCLX for an MCP tool family

Surveyed 2026-10-07. Sources: `CCL/` clone at tag **v3.3.6** (commit 66b05e1, 2026-07-28,
"Merge PR #1305 fix_ept_bug"), `CCLX/` notebooks (24 notebooks + `lsst_galaxy_sample.py`),
readthedocs sources, and a live `pyccl 3.3.6` in the `cosmic-emu` conda env (all signatures
below were introspected with `inspect.signature` against that install and cross-checked
against the clone's source; all timings measured on this Mac, single thread).

---

## 1. Install & import

| Item | Finding |
|---|---|
| Package name | `pyccl` (PyPI + conda-forge), `import pyccl as ccl` |
| Python | `requires-python >= 3.10`; cibuildwheel builds cp310-cp314 (`pyproject.toml`) |
| Compiled code | Yes: C library (GSL, FFTW, CLASS-derived code) wrapped with SWIG into `_ccllib.so`. `pip install pyccl` **builds from source** and needs `cmake` + `swig` + a C compiler (readthedocs `installation.rst`; `[build-system] requires = setuptools, setuptools_scm, cmake, swig, numpy`). manylinux_2_28 x86_64/aarch64 and macOS 11+ wheels are produced by CI; if a wheel matches, pip is a pure download. No Windows. |
| Preferred install | `conda install -c conda-forge pyccl` (also pulls `camb`). Or `pip install pyccl` (+ `pip install camb`). |
| Hard runtime deps | `numpy, packaging, scipy, pyyaml` only |
| Optional deps | `camb` (default `transfer_function='boltzmann_camb'` **fails without it**), `classy` (CLASS), `isitgr`, `fast-pt` (EulerianPT), `velocileptors` (LagrangianPT), `baccoemu` (+tensorflow; Bacco linear/nonlinear/baryons/lbias), `MiraTitanHMFemulator` (Bocquet20 HMF), `dark_emulator` (Nishimichi19 HMF). CosmicEmu MT-II/IV need **nothing extra** (data shipped in `pyccl/emulators/data/*.npz`). Full test env: `CCL/.github/environment.yml`. |
| Local envs | `cosmic-emu`: pyccl **3.3.6**, py3.12, camb 1.6.6, baccoemu 2.3.0, tensorflow 2.21, MiraTitanHMFemulator 0.1.1 present; classy/fastpt/velocileptors/isitgr/dark_emulator **missing**. `llm_env`: pyccl 3.3.1 (py3.10). `hep-genesis`: no pyccl. |
| Import cost | `import pyccl` 0.3 s. Importing from inside `CCL/` fails (circular import; the un-built source tree shadows the installed package) -- run tools from a different cwd. |
| Version string | `pyccl.__version__` via `importlib.metadata` (setuptools_scm dynamic). CHANGELOG top entries: v3.1.2 dynamic versioning fix, v3.1.1 constants update, v3.1 DarkEmulator HMF, v3.0 emulators/baryons/MG structures. |

Clone-vs-install: the local clone (v3.3.6) and the `cosmic-emu` install are the same version, so
signatures quoted here are valid for both.

---

## 2. Core API map (exact signatures, units, runtimes)

**Global unit convention (readthedocs `notation_and_other_cosmological_conventions.rst`):**
*All* units are non-h-inverse: distances in **Mpc**, wavenumbers in **1/Mpc**, P(k) in **Mpc^3**,
masses in **M_sun**, densities in M_sun/Mpc^3, time in Gyr. Scale factor `a` (not z) is the
time argument everywhere. The template server uses h/Mpc; a CCL tool layer must convert
(`k[h/Mpc] = k[1/Mpc]/h`, `P[(Mpc/h)^3] = P[Mpc^3]*h^3`) or label columns explicitly.
Exceptions: `sigma8` is defined at 8 Mpc/h (standard); `BaryonsSchneider15.k_s` is in h/Mpc.

### 2.1 Cosmology

```python
ccl.Cosmology(*, Omega_c=None, Omega_b=None, h=None, n_s=None, sigma8=None, A_s=None,
              Omega_k=0.0, Omega_g=None, Neff=None, m_nu=0.0, mass_split='normal',
              w0=-1.0, wa=0.0, T_CMB=2.7255, T_ncdm=0.71611,
              transfer_function='boltzmann_camb', matter_power_spectrum='halofit',
              baryonic_effects=None, mg_parametrization=None, extra_parameters=None)
```
- Required: `Omega_c, Omega_b, h, n_s` and **exactly one** of `sigma8`/`A_s`
  (`ValueError: Set either A_s or sigma8 but not both.`). `Neff` default 3.044.
- `m_nu`: float (total mass, eV, split per `mass_split` in {'single','equal','normal','inverted'})
  or list of individual masses. Derived `Omega_nu_mass` is added to `Omega_m`
  (`cosmo['Omega_m'] = Omega_b + Omega_c + Omega_nu_mass`, `cosmology.py:457`).
- `transfer_function` (enum `TransferFunctions`, `cosmology.py:39`): `'bbks'`, `'eisenstein_hu'`,
  `'eisenstein_hu_nowiggles'`, `'boltzmann_camb'` (default), `'boltzmann_class'`,
  `'boltzmann_isitgr'`, or an `EmulatorPk` instance (`BaccoemuLinear`).
- `matter_power_spectrum` (enum `MatterPowerSpectra`, `cosmology.py:50`): `'halofit'` (default,
  Takahashi 2012 applied to the CCL linear P(k)), `'linear'`, `'camb'` (take the nonlinear P(k)
  directly from CAMB, incl. HMcode via `extra_parameters`), or an `EmulatorPk` instance
  (`CosmicemuMTIVPk`, `CosmicemuMTIIPk`, `BaccoemuNonlinear`). There is **no** `'hmcode'` string:
  HMcode = `matter_power_spectrum='camb'` + `extra_parameters={'camb': {'halofit_version':
  'mead2020_feedback', 'HMCode_logT_AGN': 7.8, 'kmax': 20.}}` (supported keys: halofit_version,
  HMCode_A_baryon, HMCode_eta_baryon, HMCode_logT_AGN, kmax, lmax, AccuracyTarget, dark_energy_model).
- Derived parameters by item access: `cosmo['Omega_m'], cosmo['sigma8'], cosmo['A_s'] (nan if
  sigma8 given and never computed), cosmo['Omega_nu_mass'], cosmo['Neff'], cosmo['m_nu']`.
- `ccl.CosmologyVanillaLCDM(**kwargs)`: Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
  camb+halofit; kwargs override.
- Serialization: `cosmo.write_yaml(filename, *, sort_keys=False)`, `Cosmology.read_yaml(filename,
  **kwargs)`, `cosmo.to_dict()` (round trip `Cosmology(**cosmo.to_dict()) == cosmo` is True).
  **YAML fails** (`ValueError ... cannot be serialised to YAML`) if `baryonic_effects`,
  `mg_parametrization`, or an emulator object is set -- store those as your own JSON spec.
- Lazy evaluation: distances, growth, P_lin, P_nl, sigma(M) splines are computed on first use and
  cached on the object (`compute_distances/compute_growth/compute_linear_power/
  compute_nonlin_power/compute_sigma`, `has_*` flags).

```python
ccl.CosmologyCalculator(*, Omega_c, Omega_b, h, n_s, sigma8|A_s, Omega_k=0.0, Omega_g=None,
    Neff=None, m_nu=0.0, mass_split='normal', w0=-1.0, wa=0.0, T_CMB=2.7255, T_ncdm=0.71611,
    mg_parametrization=None, background=None, growth=None, pk_linear=None, pk_nonlin=None,
    nonlinear_model=None)
```
Build a cosmology from arrays: `background={'a','chi'[Mpc],'h_over_h0'}`,
`growth={'a','growth_factor','growth_rate'}`, `pk_linear={'a','k'[1/Mpc],
'delta_matter:delta_matter': (n_a,n_k) array}`, `pk_nonlin` same, `nonlinear_model='halofit'`
to halofit the supplied linear P(k). Arrays must be monotonically ascending in `a`; too few `a`
samples (I tried 2) gives a bare `CCLError CCL_ERROR_MEMORY` -- use >= ~10. Construction 0.15 s.

### 2.2 Background (all take `a`, return Mpc / dimensionless; vectorized; ~10-30 ms first call)

```python
ccl.comoving_radial_distance(cosmo, a)        # Mpc
ccl.comoving_angular_distance(cosmo, a)       # Mpc (transverse, curvature-aware)
ccl.angular_diameter_distance(cosmo, a1, a2=None)   # Mpc; a2<a1 for lens->source
ccl.luminosity_distance(cosmo, a)             # Mpc
ccl.distance_modulus(cosmo, a)
ccl.h_over_h0(cosmo, a)                       # E(a) = H(a)/H0
ccl.scale_factor_of_chi(cosmo, chi)           # inverse of chi(a)
ccl.hubble_distance(cosmo, a); ccl.comoving_volume_element(cosmo, a); ccl.comoving_volume(cosmo, a, *, solid_angle=4pi)
ccl.lookback_time(cosmo, a); ccl.age_of_universe(cosmo, a)       # Gyr
ccl.growth_factor(cosmo, a)                   # D(a)/D(1)
ccl.growth_factor_unnorm(cosmo, a)
ccl.growth_rate(cosmo, a)                     # f = dlnD/dlna
ccl.omega_x(cosmo, a, species); ccl.rho_x(cosmo, a, species, *, is_comoving=False)  # M_sun/Mpc^3
ccl.sigma_critical(cosmo, *, a_lens, a_source)   # M_sun/Mpc^2
```
`species` in {'matter','dark_energy','radiation','curvature','neutrinos_rel','neutrinos_massive','critical'}.
Every function is also a method: `cosmo.comoving_radial_distance(a)`.

### 2.3 Power spectra

```python
ccl.linear_matter_power(cosmo, k, a)          # k [1/Mpc] -> P [Mpc^3]
ccl.nonlin_matter_power(cosmo, k, a)          # NOTE: nonlin_, not nonlinear_
ccl.linear_power(cosmo, k, a, *, p_of_k_a='delta_matter:delta_matter')
ccl.nonlin_power(cosmo, k, a, *, p_of_k_a='delta_matter:delta_matter')
ccl.sigma8(cosmo, *, p_of_k_a=...)            # from the stored linear P(k)
ccl.sigmaR(cosmo, R, a=1, *, p_of_k_a=...)    # R in Mpc (NOT Mpc/h)
ccl.sigmaV(cosmo, R, a=1, ...); ccl.sigmaM(cosmo, M, a); ccl.kNL(cosmo, a, ...)
ccl.get_camb_pk_lin(cosmo, *, nonlin=False); ccl.get_class_pk_lin(cosmo); ccl.get_isitgr_pk_lin(cosmo)
cosmo.get_linear_power(name='delta_matter:delta_matter') -> Pk2D ; cosmo.get_nonlin_power(name) -> Pk2D
```
`Pk2D(*, a_arr, lk_arr (ln k, 1/Mpc), pk_arr (na,nk), is_logp=True, extrap_order_lok=1,
extrap_order_hik=2)`, callable `pk(k, a, cosmo=None, *, derivative=False)`, arithmetic
(+,-,*,/,**) between Pk2D objects, `Pk2D.from_model(cosmo, model)` for
'bbks'/'eisenstein_hu'/'eisenstein_hu_nowiggles', `Pk2D.from_function(pkfunc, ...)`,
`pk.apply_halofit(cosmo)`. Default internal spline: k in [5e-5, 50] 1/Mpc (`spline_params.K_MIN,
K_MAX_SPLINE`), a in [0.01,1] (`A_SPLINE_MINLOG_PK..`); evaluation outside extrapolates
(power-law in log P), so a tool should clamp k to [1e-4, 50] 1/Mpc.

Timings (first call builds and caches the spline; second call is ~0 ms):
bbks/eisenstein_hu linear 15 ms, +halofit 40-55 ms; camb linear 100 ms (sigma8) / 330 ms (A_s +
m_nu); camb HMcode2020 nonlinear 270 ms; w0wa + 3 massive nu + curvature camb+halofit 500 ms.

### 2.4 Tracers (`tracers.py:831-1071`; all return a `Tracer`)

```python
ccl.NumberCountsTracer(cosmo, *, dndz, bias=None, mag_bias=None, has_rsd, n_samples=256)
ccl.WeakLensingTracer(cosmo, *, dndz, has_shear=True, ia_bias=None, use_A_ia=True, n_samples=256)
ccl.CMBLensingTracer(cosmo, *, z_source, n_samples=100)        # z_source ~ 1100
ccl.tSZTracer(cosmo, *, z_max=6., n_chi=1024)
ccl.CIBTracer(cosmo, *, z_min=0., z_max=6., n_chi=1024)
ccl.ISWTracer(cosmo, *, z_max=6., n_chi=1024)
ccl.Tracer()  # empty; .add_tracer(cosmo, *, kernel=None, transfer_ka=None, transfer_k=None,
              #   transfer_a=None, der_bessel=0, der_angles=0, is_logt=False, extrap_order_lok=0, extrap_order_hik=2)
ccl.get_density_kernel(cosmo, *, dndz); ccl.get_lensing_kernel(cosmo, *, dndz, mag_bias=None, n_chi=None); ccl.get_kappa_kernel(cosmo, *, z_source, n_samples=100)
```
`dndz=(z_array, nz_array)` -- CCL normalizes n(z) internally. `bias=(z, b(z))`,
`mag_bias=(z, s(z))`, `ia_bias=(z, A_IA(z))` (with `use_A_ia=True` the NLA normalization
`-A_IA C1 rho_crit Omega_m / D(z)` is applied). `has_rsd` is **required** for NumberCountsTracer.
Tracer methods: `get_kernel(chi)`, `get_transfer(lk, a)`, `get_f_ell(ell)`, `chi_min/chi_max`.
WeakLensingTracer build 28 ms.

### 2.5 Angular power spectra and correlation functions

```python
ccl.angular_cl(cosmo, tracer1, tracer2, ell, *, p_of_k_a='delta_matter:delta_matter',
               l_limber=-1, limber_max_error=0.01, limber_integration_method='qag_quad',
               non_limber_integration_method='FKEM', fkem_chi_min=None, fkem_Nchi=None,
               p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
```
- `l_limber=-1` (default): Limber at all ell. `l_limber=N`: exact FKEM non-Limber (N5K winner,
  arXiv:2212.04291) for ell<N, Limber above. `l_limber='auto'`: FKEM picks the switch given
  `limber_max_error`. `return_meta=True` returns `(cl, {'l_limber': ...})`. `p_of_k_a` can be a
  `Pk2D` (halo-model, PT, baryonified ...). CCLX `generate_dv_nonlimber` uses
  `l_limber=np.max(ells), non_limber_integration_method="FKEM"` for all clustering spectra.
- Timing: WL-WL Limber at 60 ells 117 ms first call (includes P_nl build) then **2 ms**;
  NC-NC FKEM for ell=2..499: 33 ms ('auto'), 63 ms (l_limber=100). Non-Limber emits
  `CCLWarning`s about `Nchi`/`chi_min` defaults -- harmless, silence them.

```python
ccl.correlation(cosmo, *, ell, C_ell, theta, type='NN', method='fftlog')
```
`theta` in **degrees**; `type` in {'NN' (0x0), 'NG' (0x2, gamma_t), 'GG+' (xi+), 'GG-' (xi-)};
`method` in {'fftlog' (default, 1 ms), 'legendre' (10 ms), 'bessel'}. Needs C_ell sampled to
high ell (CCLX uses `ell = np.arange(2, 5000)` or more); for theta < ~0.1 deg, raise ell_max.
`method='bessel'` with ell=2..4999 **raised `CCLError Error 18`** in my test -- stick to fftlog.

```python
ccl.correlation_3d(cosmo, *, r, a, p_of_k_a=...)                 # r in Mpc, xi(r); 155 ms
ccl.correlation_multipole(cosmo, *, r, a, beta, ell, p_of_k_a=...)   # beta = f/b; 150 ms
ccl.correlation_3dRsd(cosmo, *, r, a, mu, beta, p_of_k_a=..., use_spline=True)
ccl.correlation_3dRsd_avgmu(cosmo, *, r, a, beta, p_of_k_a=...)
ccl.correlation_pi_sigma(cosmo, *, pi, sigma, a, beta, use_spline=True, p_of_k_a=...)
```
Covariances: `ccl.angular_cl_cov_cNG(cosmo, tracer1, tracer2, *, ell, t_of_kk_a, tracer3=None,
tracer4=None, ell2=None, fsky=1.0, integration_method='qag_quad')`, `angular_cl_cov_SSC(...,
sigma2_B=None, fsky=1.0)`, `sigma2_B_disc(cosmo, a_arr=None, *, fsky=1.0, ...)`; trispectra
from `halos.halomod_Tk3D_{1h,2h,3h,4h,SSC,cNG}`.

### 2.6 Halo model (`pyccl.halos`)

```python
halos.MassDef(Delta, rho_type)   # Delta: float | 'vir' | 'fof'; rho_type: 'critical' | 'matter'
halos.MassDef200m, MassDef200c, MassDef500c, MassDefVir, MassDefFof   # instances
halos.MassDef.from_name('200m'); md.get_radius(cosmo, M, a) [Mpc]; md.get_mass(cosmo, R, a)
halos.mass_translator(*, mass_in, mass_out, concentration) -> callable f(cosmo, M, a)
```
Mass functions (`MassFunc(*, mass_def, mass_def_strict=True)`; `__call__(cosmo, M, a)` returns
**dn/dlog10M in comoving Mpc^-3**, M in M_sun): `Press74, Sheth99, Jenkins01, Tinker08 (default
'200m'), Tinker10, Angulo12, Watson13, Despali16, Bocquet16(hydro=True), Bocquet20 ('200c',
needs MiraTitanHMFemulator), Nishimichi19 (needs dark_emulator)`; `MassFunc.from_name('Tinker08')`.
Halo bias (`__call__(cosmo, M, a)`): `Sheth99, Sheth01, Bhattacharya11, Tinker10`.
Concentration (`__call__(cosmo, M, a)`): `Duffy08(fc_bar=1, *, mass_def='200c'), Bhattacharya13,
Diemer15, Klypin11, Prada12, Ishiyama21, Constant`.
Profiles (`real(cosmo, r[Mpc], M, a)`, `fourier(cosmo, k, M, a)`, `projected`, `cumul2d`,
`convergence`, `shear`, `reduced_shear`, `magnification`):
```python
halos.HaloProfileNFW(*, mass_def, concentration, fourier_analytic=True, projected_analytic=False, cumul2d_analytic=False, truncated=True)
halos.HaloProfileEinasto(*, mass_def, concentration, truncated=False, projected_quad=False, alpha='cosmo')
halos.HaloProfileHernquist(...); halos.HaloProfileHOD(*, mass_def, concentration, log10Mmin_0=12.0, log10Mmin_p=0.0, siglnM_0=0.4, siglnM_p=0.0, log10M0_0=7.0, log10M0_p=0.0, log10M1_0=13.3, log10M1_p=0.0, alpha_0=1.0, alpha_p=0.0, fc_0=1.0, fc_p=0.0, bg_0=1.0, bg_p=0.0, bmax_0=1.0, bmax_p=0.0, a_pivot=1.0, ns_independent=False, is_number_counts=True)
halos.HaloProfilePressureGNFW(*, mass_def, mass_bias=0.8, P0=6.41, c500=1.81, alpha=1.33, alpha_P=0.12, beta=4.13, gamma=0.31, P0_hexp=-1.0, qrange=(0.001, 1000.0), nq=128, x_out=inf)
halos.HaloProfileCIBShang12(*, mass_def, concentration, nu_GHz, alpha=0.36, T0=24.4, beta=1.75, gamma=1.7, s_z=3.6, log10Meff=12.6, siglog10M=0.707, Mmin=1e10, L0=6.4e-08)
halos.SatelliteShearHOD(...)  # IA halo model
halos.Profile2pt(), Profile2ptHOD(), Profile2ptCIB()
```
Calculator and integrals:
```python
halos.HMCalculator(*, mass_function, halo_bias, mass_def=None, log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='simpson')
hmc.number_counts(cosmo, *, selection, a_min=None, a_max=1.0, na=128)   # selection(m, a) -> (len(m), len(a)); 225 ms
halos.halomod_power_spectrum(cosmo, hmc, k, a, prof, *, prof2=None, prof_2pt=None, p_of_k_a=None, get_1h=True, get_2h=True, smooth_transition=None, suppress_1h=None, extrap_pk=False)  # -> (N_a, N_k); 3 ms per a
halos.halomod_Pk2D(cosmo, hmc, prof, *, prof2=None, prof_2pt=None, p_of_k_a=None, get_1h=True, get_2h=True, lk_arr=None, a_arr=None, extrap_order_lok=1, extrap_order_hik=2, smooth_transition=None, suppress_1h=None, extrap_pk=False)  # 565 ms (all a)
halos.halomod_mean_profile_1pt(cosmo, hmc, k, a, prof); halos.halomod_bias_1pt(cosmo, hmc, k, a, prof)
```
`mass_function`/`halo_bias`/`mass_def` accept name strings. HMF first call 70 ms (builds
sigma(M) spline), then ~0.

### 2.7 Baryons (`pyccl.baryons`; all have `boost_factor(cosmo, k, a)` and `include_baryonic_effects(cosmo, pk: Pk2D) -> Pk2D`)

```python
ccl.BaryonsSchneider15(log10Mc=14.079, eta_b=0.5, k_s=55.0)   # BCM; k_s in h/Mpc (!)
ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='500c')           # valid k <= 1 h/Mpc, z=0 only
ccl.BaccoemuBaryons(...)   # kwargs: log10_M_c, log10_eta, log10_beta, log10_M1_z0_cen, log10_theta_out, log10_theta_inn, log10_M_inn (bacco names); needs baccoemu
```
Pass as `Cosmology(baryonic_effects=...)` (applied to P_nl in `_compute_nonlin_power`,
`cosmology.py:645`) **or** apply explicitly. Pitfall: `BaccoemuBaryons` as `baryonic_effects`
**raises** `ValueError: Requested scale factor outside the bounds of the emulator ... (0.4, 1.0)`
because CCL's default P(k) spline starts at a=0.01; CCLX therefore uses
`boost = bacco.boost_factor(cosmo, k, a); pk * boost` (z <= 1.5 only). `boost_factor` needs
array `k` (scalar k crashes inside baccoemu). Schneider15 adds ~150 ms.

### 2.8 Neutrinos

`m_nu` + `mass_split` on `Cosmology` (see 2.1). Helper: `ccl.nu_masses(*, Omega_nu_h2=None,
mass_split, m_nu=None)` (mass_split also accepts 'sum'). Massive neutrinos affect background
(`Omega_nu_mass`), growth and CAMB/CLASS transfer; analytic transfer functions (bbks/EH) ignore
them. CosmicEmu/Bacco accept m_nu through their own parameter boxes.

### 2.9 Modified gravity

```python
from pyccl.modified_gravity import MuSigmaMG
MuSigmaMG(parametrization='mu_Sigma', mu_0=0, sigma_0=0, c1_mg=1, c2_mg=1, lambda_mg=0)
ccl.Cosmology(..., mg_parametrization=MuSigmaMG(mu_0=0.1, sigma_0=0.1), transfer_function='boltzmann_camb'|'boltzmann_isitgr', matter_power_spectrum='linear')
```
mu-Sigma rescales the linear growth and lensing kernels (Planck 2015 Eq. 46-47 style with
`c1_mg, c2_mg, lambda_mg` giving k-dependence). Cannot combine `mu_0 != 0` with
`matter_power_spectrum='camb'` (`ValueError: Can't rescale non-linear power spectrum from CAMB
for mu-Sigma MG`); halofit on top of MG-rescaled linear P(k) is what CCLX does. 90 ms.

### 2.10 Emulators (`pyccl.emulators`; all `EmulatorPk` with `get_pk2d(cosmo)` and `get_pk_at_a(cosmo, a)`)

```python
ccl.CosmicemuMTIVPk(kind='tot'|'cb')   # Mira-Titan IV (Moran+22), 8 params (omega_m, omega_b, sigma8, n_s, h, w0, wa, omega_nu); no extra deps; init 31 ms, P(k) 35 ms
ccl.CosmicemuMTIIPk(kind='tot'|'cb')   # Mira-Titan II (Lawrence+17)
ccl.BaccoemuLinear()                   # needs baccoemu; init 4 ms (after TF import), used as transfer_function
ccl.BaccoemuNonlinear(nonlinear_emu_path=None, nonlinear_emu_details=None, n_sampling_a=100)   # init 265 ms, P(k) 630 ms first call
```
Use: `ccl.Cosmology(..., matter_power_spectrum=ccl.CosmicemuMTIVPk('tot'))` or
`transfer_function=ccl.BaccoemuLinear()`. Parameter boxes are enforced by the emulators
(CosmicEmu: sigma8 0.7-0.9, z <= 2.02, k <= ~5 1/Mpc... ; Bacco: a >= 0.4, k <= 5 h/Mpc) -- a tool
should pre-validate and surface `in_training_box` as the template does.

### 2.11 Perturbation theory (`pyccl.nl_pt`)

```python
pt.EulerianPTCalculator(*, with_NC=False, with_IA=False, with_matter_1loop=True, cosmo=None, log10k_min=-4, log10k_max=2, nk_per_decade=20, a_arr=None, k_cutoff=None, n_exp_cutoff=4, b1_pk_kind='nonlinear', bk2_pk_kind='nonlinear', pad_factor=1.0, low_extrap=-5.0, high_extrap=3.0, P_window=None, C_window=0.75, sub_lowk=False)   # needs fast-pt
pt.LagrangianPTCalculator(*, cosmo=None, log10k_min=-4, log10k_max=2, nk_per_decade=20, a_arr=None, k_cutoff=None, n_exp_cutoff=4, b1_pk_kind='nonlinear', bk2_pk_kind='nonlinear')   # needs velocileptors
pt.BaccoLbiasCalculator(*, cosmo=None, log10k_min=-4, log10k_max=-0.47, nk_per_decade=20, a_arr=None, k_cutoff=None, n_exp_cutoff=4, extrap_lin=True)   # needs baccoemu
pt.PTNumberCountsTracer(b1, b2=None, bs=None, b3nl=None, bk2=None); pt.PTIntrinsicAlignmentTracer(c1, c2=None, cdelta=None); pt.PTMatterTracer()
ptc.update_ingredients(cosmo); ptc.get_biased_pk2d(tracer1, *, tracer2=None, return_ia_bb=False, extrap_order_lok=1, extrap_order_hik=2) -> Pk2D
pt.translate_IA_norm(cosmo, *, z, a1=1.0, a1delta=None, a2=None, Om_m2_for_c2=False, Om_m_fid=0.3)  # (A1, A1delta, A2) -> (c1, cdelta, c2)
```
Bias args can be floats or `(z, b(z))` tuples. The biased Pk2D is then fed to
`angular_cl(..., p_of_k_a=pk_gg, p_of_k_a_lin=pk_gg_lin)` (CCLX `generate_dv_PT_nonlimber`).
Not runnable in `cosmic-emu` (fast-pt / velocileptors absent); k in 1/Mpc.

---

## 3. CCLX notebooks (24) -- what each demonstrates

| Notebook | Demonstrates / key calls |
|---|---|
| **Distance Calculations Example** | Build `ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)`; `comoving_radial_distance, comoving_angular_distance, luminosity_distance, distance_modulus, scale_factor_of_chi` vs z; parameter access `cosmo['Omega_m']`. The canonical starter. |
| **Power spectrum example** | Linear/nonlinear P(k) with `transfer_function` in {'boltzmann_camb','boltzmann_class','bbks','eisenstein_hu'}, `ccl.sigma8(cosmo)` with `A_s`; BCM baryons `BaryonsSchneider15(14.25, 0.5, 37.)`; massive neutrinos `m_nu=0.1, mass_split='equal'` vs list of 3 masses; halo-model P(k) with `MassDef('vir','matter')`, Sheth99 HMF/bias, Duffy08, NFW, `halomod_power_spectrum`; mu-Sigma MG linear P(k). |
| **CellsCorrelations** | The LSST 3x2pt core: Smail n(z) `pz=(z/z0)^2 exp(-z/z0)`, `WeakLensingTracer` (with `ia_bias`), `NumberCountsTracer(has_rsd=False, bias=(z,b))`, `CMBLensingTracer(z_source=1090)`, `angular_cl` for shear/clustering/cross, then `correlation(..., type='GG+'/'GG-'/'NN', method='FFTLog')` with theta in degrees. |
| **Angular cross-correlations** | 6 tracers (galaxy density, shear, CMB kappa, ISW, tSZ, CIB) built from `CosmologyVanillaLCDM`; `halomod_Pk2D` with HOD / GNFW pressure / CIB Shang12 profiles and `Profile2ptHOD/CIB` to supply `p_of_k_a` to `angular_cl` for every pair. |
| **GeneralizedTracers** | Custom `ccl.Tracer().add_tracer(cosmo, kernel=..., transfer_a=..., der_bessel, der_angles)` reproducing standard tracers from `get_density_kernel/get_lensing_kernel`, custom `Pk2D` from arrays; shows growth_factor/growth_rate-based transfer functions. |
| **MCMC Likelihood Analysis** | How CCL feeds a likelihood: mock shear C_ell data from CCL (`transfer_function='bbks'` for speed), diagonal Gaussian cov `C_ell^2 * 2pi/(A_sky * ell * dell)`, `lnprob(theta)` rebuilds `Cosmology(Omega_c, sigma8)` + 2 `WeakLensingTracer`s + `angular_cl` per step, `emcee.EnsembleSampler` 10 walkers x 150 steps. Template for a "fit_cls" tool: one CCL evaluation per likelihood call (~50-150 ms with bbks). |
| **generate_dv_nonlimber** | LSST Y1 3x2pt data-vector generation: Planck18 fiducial, `m_nu=0.1, mass_split='equal'`, `matter_power_spectrum='camb'` + HMcode2020 feedback (`HMCode_logT_AGN=7.8`); 20 log ell bins 20-2000 with tophat `sacc.BandpowerWindow`; SRD lens bins (5 equal-width z=0.2-1.2, sigma_z=0.03(1+z), erf-convolved) and source bins (5 equal-number, sigma_z=0.05(1+z)); bias `b1 = 1.05/D(z_mean)`, mag_bias 0.1; NLA IA `A_IA=(1+z)/(1+0.62))^-1`; `angular_cl(..., l_limber=max(ells), non_limber_integration_method='FKEM')` vs Limber; writes a SACC file (`add_tracer('NZ',...)`, `add_ell_cl('galaxy_density_cl'|'galaxy_shearDensity_cl_e'|'galaxy_shear_cl_ee', ...)`). |
| **generate_dv_PT_nonlimber** | Same pipeline with `EulerianPTCalculator(with_NC=True, with_IA=True)`, `PTNumberCountsTracer(b1,b2,bs,bk2,b3nl)`, `PTIntrinsicAlignmentTracer` via `translate_IA_norm`, biased `Pk2D`s passed as `p_of_k_a` / `p_of_k_a_lin` to non-Limber `angular_cl`; splits GG/GI/II and galaxy-matter terms. Needs fast-pt. |
| **CalculatorMode** | `CosmologyCalculator` from CLASS outputs: `background={'a','chi','h_over_h0'}` (from `classy.get_background()`), `pk_linear/pk_nonlin={'a','k','delta_matter:delta_matter'}` (200 k x 100 a), compared to a native `transfer_function='boltzmann_class'` cosmology for chi, P(k), C_ell. Pattern for ingesting external P(k) (e.g. from the emulator server's CSVs). |
| **Reading-writing-Cosmology-objects** | `cosmo.write_yaml('f.yaml')`, `Cosmology.read_yaml('f.yaml')`, `pickle.dump/load` of a Cosmology (works for freshly built objects; see Gotchas). |
| **LSST_SRD_Redshift_Distributions_and_Binning** | Uses `lsst_galaxy_sample.LSSTGalaxySample(forecast_year in {'1','4','7','10'}, redshift_range)` to produce SRD Smail n(z) for lens/source samples, tomographic bins (`lens_bins`, `source_bins`), bin centers, saving npy/csv to `data_output/`; plots. **No pyccl calls** -- a pure numpy/scipy building block. |
| *lsst_galaxy_sample.py* | Reads `parameters/lsst_desc_parameters.yaml` (per year: lens `z_0, alpha, beta=2, bin_start=0.2, bin_stop=1.2, bin_spacing (0.2 Y1/Y4 -> 5 bins; 0.1 Y7/Y10 -> 10 bins), sigma_z=0.03, n_gal (18/28/38/48 per arcmin^2), galaxy_bias_prefactor (1.05/0.95)`; source `z_0, alpha, n_tomo_bins=5 equal-number, sigma_z=0.05, n_gal (10/15.7/21.3/27)`; sky `frac_sky=0.4363, lsst_sky=18000 deg^2`). Methods: `smail_type_distribution(z, z0, alpha, beta)`, `true_redshift_distribution(nz, zmin, zmax, sigma_z, z_bias)` (Ma-Hu-Huterer erf convolution with scatter `sigma_z(1+z)`), `compute_equal_number_bounds`, `normalize_distribution`, `compute_tomo_bin_centers`. Reads the yaml by **relative path** `parameters/...` so it only works with cwd=CCLX; re-implement (~150 lines) inside the MCP `tools/` package with the yaml values embedded. |
| **Halo-mass-function-example** | `MassDef` variants (200m/200c/500c/vir/fof), all HMFs via class or `from_name`, halo bias, c(M) relations, `mass_translator(mass_in, mass_out, concentration)` to compare Tinker08 at 200m vs 500m. |
| **Halo-model-Pk** | Tinker08/Tinker10/Duffy08 on 200m; NFW matter P(k) 1h/2h via `halomod_power_spectrum(get_1h/get_2h)`; HOD galaxy P(k) with `Profile2ptHOD`; `halomod_Pk2D` -> `angular_cl` for galaxy clustering/shear. |
| **Halo profiles** | `HaloProfileNFW/Einasto/Hernquist.real/fourier/projected/cumul2d`, truncation, custom `HaloProfile` subclass; r in Mpc, M in M_sun. |
| **Baryons_halo_model_power_spectrum** | Builds a Fedeli-2014 style baryonic halo model with custom profiles (stars, gas) and `HMCalculator`; compares to `linear_matter_power/nonlin_matter_power`. |
| **Baryons Modules** | Compares P_nl boosts: `BaryonsSchneider15().include_baryonic_effects(cosmo, cosmo.get_nonlin_power())`, `BaryonsvanDaalen19`, HMcode2020 via `extra_parameters={'camb': {'kmax': 20.0, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}`, `BaccoemuBaryons().boost_factor(cosmo, k, a)`, and `CosmicemuMTIVPk` as DM-only reference. |
| **Cosmological_Emulator** | `BaccoemuLinear` as transfer_function, `BaccoemuNonlinear` as matter_power_spectrum, `BaccoemuBaryons`, `CosmicemuMTIVPk` / `MTIIPk`; linear vs nonlinear comparisons at z=0. |
| **Halo model for IA** | IA halo model: `SatelliteShearHOD` + HOD central profile, `halomod_Pk2D` for GI/II, `halomod_bias_1pt`, fed to `WeakLensingTracer` `angular_cl` and `correlation`; needs `wigner` package. |
| **PerturbationTheoryPk** | `nl_pt` EPT walkthrough: `PTNumberCountsTracer`, `PTIntrinsicAlignmentTracer`, `PTMatterTracer`, `EulerianPTCalculator`, `get_biased_pk2d`, then `angular_cl` with the biased Pk2D; inspects individual PT templates. |
| **MG_mu_sigma_examples** | `MuSigmaMG(mu_0, sigma_0)` scale-independent; P_lin, `growth_factor_unnorm`, C_ells (NC, WL, CMB lensing), `correlation`, `correlation_3d`, comparisons to GR. |
| **MG_mu_Sigma_with_z_dependence_only** / **..._with_z_and_k_dependences** | Same with time-only and (c1_mg, c2_mg, lambda_mg) scale-dependent variants. |

---

## 4. Proposed MCP tool family `ccl` (12 tools)

Common cosmology inputs for every compute tool (JSON scalars, validated with pydantic):
`Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81 | A_s=None (exactly one),
Omega_k=0.0, w0=-1.0, wa=0.0, m_nu=0.0, mass_split='normal', Neff=3.044,
transfer_function='boltzmann_camb' | 'bbks' | 'eisenstein_hu' | 'eisenstein_hu_nowiggles' |
'boltzmann_class' | 'bacco', matter_power_spectrum='halofit' | 'linear' | 'camb_hmcode' |
'cosmicemu_mt4' | 'cosmicemu_mt2' | 'bacco', baryons=None | 'schneider15' | 'vandaalen19' |
'bacco' (+ dict of model params), mg_mu0=0.0, mg_sigma0=0.0, hmcode_logT_AGN=7.8`.
Helper `build_cosmology(spec: dict) -> ccl.Cosmology` in `tools/ccl/common.py` maps these to
CCL objects, caches by `param_slug` (CCL objects cache their splines, so repeated calls across
tools with the same spec cost ~0), and records `cosmo.to_dict()` + the non-YAML-able extras in
metadata. Optionally accept `k_units='1/Mpc'|'h/Mpc'` on P(k) tools (default `h/Mpc` to match the
emulator server; label columns `k_h_per_Mpc, Pk_Mpc_over_h_cubed` vs `k_per_Mpc, Pk_Mpc3`).

| # | Tool | Inputs (beyond cosmology) | Output file / metadata | Wraps |
|---|---|---|---|---|
| 1 | `ccl_describe_cosmology` | cosmology spec | JSON of `to_dict()` + derived (`Omega_m, sigma8, A_s, Omega_nu_mass, Neff`), `sigma8` computed if A_s given, timing; writes `cosmology_<slug>.yaml` when YAML-able | `Cosmology`, `cosmo[...]`, `write_yaml`, `ccl.sigma8` |
| 2 | `ccl_background` | `z_min=0, z_max=3, n_z=100, log_z=False` | CSV: z, a, chi_Mpc, D_A_Mpc, D_L_Mpc, distance_modulus, E(a)=H/H0, H_km_s_Mpc, growth_factor, growth_rate, lookback_Gyr, comoving_volume_element | all of 2.2 |
| 3 | `ccl_matter_pk` | `z=0.0, k_min=1e-4, k_max=50, n_k=300, kind='both'` | CSV: k, P_lin, P_nl (and ratio) in chosen units; metadata `sigma8, kNL`, emulator `in_training_box` | `linear_matter_power`, `nonlin_matter_power`, `kNL` |
| 4 | `ccl_sigma_r` | `R_list [Mpc or Mpc/h], z` or `M_list` | CSV: R, sigma_R (and sigma_M) | `sigmaR`, `sigmaM`, `sigmaV` |
| 5 | `ccl_lsst_srd_nz` | `forecast_year in {1,4,7,10}, sample in {'lens','source'}, z_max=3.5, n_z=500, normalize=True` | CSV `nz_<sample>_Y<yr>.csv`: z, nz_total, bin_0..bin_{N-1}; metadata: bin edges, bin centers, `n_gal_arcmin2` total and per bin (= n_gal/N for sources), `sigma_z`, `galaxy_bias_prefactor`, `f_sky`, SRD reference. Pure numpy (re-implementation of `lsst_galaxy_sample.py` with the yaml table embedded) -- also an HPC-safe kernel. | SRD Appendix D |
| 6 | `ccl_make_tracers` (optional; or fold into 7) | list of tracer specs: `{type: 'wl'|'nc'|'cmb_lensing'|'tsz'|'cib'|'isw', nz_file, column, bias (const or 'srd'), mag_bias, has_rsd, A_ia, eta_ia, z_pivot, z_source=1100}` | JSON spec file + CSV of radial kernels W(chi) for inspection | `WeakLensingTracer`, `NumberCountsTracer`, `CMBLensingTracer`, `tSZTracer`, `CIBTracer`, `ISWTracer`, `Tracer.get_kernel` |
| 7 | `ccl_angular_cls` | tracer specs (as in 6, inline), `pairs='all'|list`, `ell_min=2, ell_max=3000, n_ell=60, ell_spacing='log'|'linear', l_limber=-1 | int | 'auto', pk_source='cosmo'|'<csv path>'` (P(k,a) CSV -> `Pk2D`/`CosmologyCalculator`) | CSV `cls_<slug>.csv`: ell, then one column per pair `cl_<t1>_<t2>`; metadata: pair list, kernel z-ranges, `l_limber` used, runtime | `angular_cl` (+ `return_meta`) |
| 8 | `ccl_correlation_functions` | `cls_file` (from 7) or tracer specs, `theta_min_deg=0.01, theta_max_deg=10, n_theta=30, types auto from spins ('GG+','GG-','NG','NN')` | CSV: theta_deg, theta_arcmin, xi per pair/type; internally re-evaluates C_ell on `ell=2..ELL_MAX_CORR` fine grid | `correlation(method='fftlog')` |
| 9 | `ccl_correlation_3d` | `z, r_min=1, r_max=200 Mpc, n_r=50, multipoles=[0,2,4], beta=None (f/b)` | CSV: r_Mpc, xi_0, xi_2, xi_4 | `correlation_3d`, `correlation_multipole` |
| 10 | `ccl_halo_mass_function` | `z_list, log10M_min=11, log10M_max=15.5, n_M=50, mass_def='200m', mass_function='Tinker08', halo_bias='Tinker10', concentration='Duffy08'` | CSV: M_Msun, dn/dlog10M [Mpc^-3], b(M), c(M), sigma(M) per z; metadata: valid name lists | `halos.MassFunc/HaloBias/Concentration.from_name`, `sigmaM` |
| 11 | `ccl_halo_model_pk` | `z, k grid, profile='nfw'|'hod', hod params, mass_def/hmf/bias/conc names, get_1h, get_2h` | CSV: k, P_1h, P_2h, P_hm, P_nl_ref | `HMCalculator`, `HaloProfileNFW/HOD`, `Profile2ptHOD`, `halomod_power_spectrum` |
| 12 | `ccl_cluster_counts` | `z_edges, log10M_edges, mass_def, mass_function, f_sky (default SRD 0.4363)` | CSV: z_lo, z_hi, log10M_lo, log10M_hi, N_clusters; uses `comoving_volume_element` x HMF integral or `hmc.number_counts(selection=tophat)` | `HMCalculator.number_counts`, `comoving_volume_element` |
| 13 | `ccl_baryon_boost` | `z, k grid, models=['schneider15','vandaalen19','bacco','hmcode2020'] + params` | CSV: k, boost per model (P_bar/P_dmo) -- directly comparable with the emulator server's `compute_baryon_suppression` | `Baryons*.boost_factor`, camb HMcode ratio |
| 14 | `ccl_mock_3x2pt_datavector` (skill-level composite, can be a tool) | `forecast_year, probes {'ss','gs','gg'}, ell binning (SRD: 20 log bins 20-2000), l_limber, IA/bias defaults from SRD` | CSV of all binned C_ell pairs + metadata; optional SACC file if `sacc` installed. Chains 5 -> 7. | per `generate_dv_nonlimber` |

Defaults to expose in docstrings: units (Mpc vs Mpc/h switch), `z` not `a` at the tool boundary
(convert `a=1/(1+z)` internally), k range [1e-4, 50] 1/Mpc, ell >= 2, theta in degrees, the
sigma8-xor-A_s rule, optional-dependency requirements per backend (CAMB default; 'bacco' needs
baccoemu+tensorflow; PT tools need fast-pt), and emulator parameter boxes (CosmicEmu: sigma8
0.7-0.9, z <= 2; Bacco: z <= 1.5, k <= 5 h/Mpc).

Server-side skills that fall out naturally: `lsst-3x2pt-forecast-inputs` (tools 5 -> 7 -> 8),
`pk-ccl-vs-emulators` (tool 3 with several `matter_power_spectrum` values + the emulator server's
`compute_nonlinear_pk`, same k grid in h/Mpc), `halo-model-tour` (10 -> 11 -> 12),
`calculator-mode-ingest` (external P(k) CSV -> `CosmologyCalculator` -> tool 7).

---

## 5. Gotchas

1. **Units**: everything is Mpc / 1/Mpc / Mpc^3 / M_sun, no h anywhere (except `sigma8`'s 8 Mpc/h
   definition and `BaryonsSchneider15.k_s` in h/Mpc). `sigmaR(R)` takes R in **Mpc**: to reproduce
   sigma8 call `sigmaR(cosmo, 8/h)`. The emulator server uses h/Mpc -- convert at the tool layer.
2. **Scale factor, not redshift**: every CCL function takes `a`; `a` arrays for splines must be
   ascending. `angular_diameter_distance(a1, a2)` requires `a2 < a1`.
3. **sigma8 xor A_s**: `ValueError: Set either A_s or sigma8 but not both.` The analytic transfer
   functions (bbks, eisenstein_hu) **require sigma8** (`CCL_ERROR_INCONSISTENT: sigma8 not set,
   required for analytic power spectra`); only CAMB/CLASS/ISiTGR/Bacco accept `A_s`.
   With `sigma8` given, CCL rescales the Boltzmann P(k) to that sigma8 even with MG/neutrinos.
4. **Defaults need CAMB**: `transfer_function='boltzmann_camb'` is the default; without the `camb`
   package the first P(k) call raises `ModuleNotFoundError`. Same for `'boltzmann_class'`
   (`classy`), EPT (`fastpt`), LPT (`velocileptors`), Bacco (`baccoemu`). Catch at tool entry and
   return an actionable message. Default `matter_power_spectrum='halofit'` means
   `nonlin_matter_power` is Takahashi halofit, not HMcode; `'linear'` makes
   `nonlin_matter_power == linear_matter_power` (ratio exactly 1).
5. **Function names**: it is `ccl.nonlin_matter_power`, not `nonlinear_matter_power`;
   `has_rsd` is a required keyword of `NumberCountsTracer`; tracer constructors are
   keyword-only.
6. **First-call cost is the spline build**: Cosmology construction is instant; the first
   P(k)/C_ell call pays 0.1-0.6 s (CAMB) and subsequent calls on the same object are ~ms. Cache
   `Cosmology` objects by parameter slug in the server (`get_cached`). In an MCMC each new
   parameter point is a new object (~50-150 ms with bbks, ~0.3-0.5 s with CAMB+halofit).
7. **Pickling**: a freshly built `Cosmology` pickles (2.5 kB) and `==` round-trips; after
   `compute_distances` it still pickles, but **after a P(k) has been computed pickling fails**
   (`TypeError: cannot pickle 'swig_runtime_data5.SwigPyObject'`). For HPC kernels ship the
   parameter dict (`to_dict()`) and rebuild on the node. `write_yaml` refuses cosmologies with
   `baryonic_effects`, `mg_parametrization`, or emulator objects.
8. **Thread safety**: the C layer is global-state-free by design (readthedocs
   `understanding_the_python_c_interface.rst`), and `pyccl.pyutils.check_openmp_threads()`
   returns 0 in the conda-forge build (no OpenMP). Each `Cosmology` holds its own splines, and
   cached objects are mutated lazily on first use (`unlock_instance`), so share objects across
   threads only after warm-up or guard with a lock (the template's `_CACHE_LOCK` suffices).
9. **Spline ranges**: P(k) spline covers k in [5e-5, 50] 1/Mpc and a in [0.01, 1] by default
   (`ccl.spline_params`); queries outside extrapolate silently (power law). `CosmologyCalculator`
   with too few `a` samples fails with an opaque `CCL_ERROR_MEMORY`.
10. **Non-Limber**: only FKEM is implemented; it warns about `Nchi`/`chi_min` defaults
    (`CCLWarning`), and is ~10-30x slower than Limber but still < 0.1 s per spectrum.
11. **correlation()**: `theta` in degrees; needs C_ell on a dense ell grid to high ell;
    `method='bessel'` raised `CCLError Error 18` for ell=2..4999 -- use `'fftlog'`.
12. **Bacco baryons**: valid only for a >= 0.4 (z <= 1.5) and crash with CCL's default a-grid when
    passed as `baryonic_effects`; apply `boost_factor` manually (array k, not scalar).
13. **MG + CAMB nonlinear**: `mu_0 != 0` with `matter_power_spectrum='camb'` raises; use
    `'halofit'` or `'linear'`.
14. **Running from the CCL source dir** shadows the installed package (circular import) -- set the
    server's cwd elsewhere.
15. `lsst_galaxy_sample.py` uses `np.trapz` (removed in numpy 2.4) and a relative yaml path; CCL
    itself pins nothing, but the test env pins `numpy < 2.4` (for fast-pt) and `scipy < 1.14`
    (for velocileptors).

---

## 6. Pip deps for an HPC compute node (Linux x86_64/aarch64)

Minimal kernel (background, P(k) with bbks/EH/halofit, C_ell, correlations, halo model,
CosmicEmu MT-II/IV, Schneider15/vanDaalen19 baryons, mu-Sigma MG with analytic transfer):
```
pyccl>=3.3          # manylinux_2_28 wheel for py3.10-3.14; if no wheel matches, pip builds from
                    # source and needs cmake, swig, gcc on the node (module load cmake; pip install swig)
numpy scipy pyyaml packaging    # pulled automatically
```
Add for the default `boltzmann_camb` transfer function and HMcode: `camb` (wheel includes
gfortran-built library). Optional per backend: `baccoemu` (+ `tensorflow`, large; also
`tables`), `classy` (compiles CLASS; `cython<3`), `fast-pt` (EPT), `velocileptors @ git+https://github.com/sfschen/velocileptors`
(LPT; needs `pyfftw`), `MiraTitanHMFemulator` (Bocquet20 HMF), `sacc` (data-vector export).
Suggested `DISPATCH_PIP_DEPS` for the template's dispatch manifest:
`{"ccl_core": ["pyccl", "camb"], "ccl_bacco": ["pyccl", "camb", "baccoemu", "tensorflow-cpu"],
"ccl_pt": ["pyccl", "camb", "fast-pt", "numpy<2.4"]}`, with walltime hints of a few minutes
(pyccl import 0.3 s; a full Y1 3x2pt data vector with FKEM is ~5-10 s; tensorflow install is
the slow part for Bacco).
