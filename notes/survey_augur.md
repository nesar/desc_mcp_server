# Survey: LSST DESC `augur` (Fisher forecasting on firecrown + CCL)

Clone surveyed: `/Users/nesar/Projects/Tutorials/desc-mcp-server/augur`
(HEAD `f8ab7cf` "Use sacc fix (#149)", 2026-08-13, tag/version **1.2.4**).
Firecrown cross-reference: `/Users/nesar/Projects/Tutorials/desc-mcp-server/firecrown`
(`v1.16.0-74-g51193df`, 2026-10-06). Nothing in either clone was modified.

Note: augur does **not** implement DALI (grep for "dali" finds nothing). It does
Fisher matrices, Gaussian priors, Omega_m/S8 reparametrisation, Fisher bias
(Amara-style), triangle plots and a w0-wa FoM. ~4600 lines of Python.

---

## 1. Install & import

| Item | Value |
|---|---|
| Python | `requires-python = ">=3.11"` (CI matrix: 3.11 only) |
| Install | `conda env create -f environment.yml` (env name `forecasting`) then `pip install -e .` (or `python -m pip install --no-deps --editable .`); also `conda install lsstdesc-augur -c conda-forge` |
| environment.yml (conda-forge) | `numpy>=2.0`, `lsstdesc-crow`, **`firecrown>=1.14.0`**, `scipy>1.12`, `healpy`, `jinja2`, `numdifftools`, `numpydoc`, `qp-prob`, `tjpcov`, **`pyccl>=3.3.1`**, **`sacc<2.2`**, `pyyaml`, flake8/sphinx/myst-parser |
| pyproject `dependencies` | `click, jinja2, healpy, numpydoc, qp-prob, scipy>1.12, numdifftools, tjpcov, pyccl>=3.3.1, numpy>=2.0, astropy, matplotlib, pandas` — **firecrown is deliberately absent from pyproject** (conda-only); `setup.cfg` (legacy) lists `firecrown>=1.14.0` |
| Optional | `derivkit` (third derivative engine; `from derivkit.calculus_kit import CalculusKit`) |
| Entry point | `[project.scripts] augur = "augur.cli:run"` |
| Env vars | `AUGUR_DIR` is referenced from every example YAML via Jinja2 (`{{ env['AUGUR_DIR'] }}`); `CSL_DIR`, `FIRECROWN_DIR` only for the CosmoSIS example |
| Runtime side effect | `generate()` with `Firecrown_Factory` writes a temp sacc to `tempfile.mkdtemp(prefix="augur_sacc_")` |

**Local environments** (checked from a neutral cwd, 2026-10-07): none of
`cosmic-emu`, `llm_env`, `hep-genesis`, `base` has `firecrown`, `sacc`, `tjpcov`,
`qp` or `augur`. `cosmic-emu` has `pyccl 3.3.6` + `camb 1.6.6`; `llm_env` has
`pyccl 3.3.1`, `camb 1.5.5`, `numdifftools 0.11.1`. No `forecasting` env exists.
So augur is **not importable anywhere locally**; nothing was executed. Beware
that running Python *from* `desc-mcp-server/` makes `import firecrown`/`import augur`
"succeed" as namespace packages (the clone directories) — a false positive.

A dedicated conda env is needed for the MCP server (`firecrown` is conda-forge only;
CAMB via pyccl's `boltzmann_camb` also requires the `camb` package).

---

## 2. Architecture

```
augur/
  cli.py            click entry: parse -> generate -> Analyze.get_fisher_matrix
                    [-> get_fisher_bias] [-> postprocess]
  parser.py         parse(filename): Jinja2(env) render -> yaml.safe_load
  generate.py       generate_sacc_and_stats(), generate(), n(z) registry, cov dispatch
  analyze.py        class Analyze (derivatives, Fisher, priors, Jacobian, bias)
  postprocess.py    postprocess(), draw_fisher_ellipses(), get_FoM_all()
  tracers/two_point.py   ZDist, LensSRD2018, SourceSRD2018, TopHat, Gaussian,
                         ZDistFromFile, WLSource, srd_dndz(), equal_density_zbins()
  utils/config_io.py     parse_array() (restricted eval of "np.geomspace(...)"),
                         parse_config(), validate_amplitude_parameter(), read_fisher_from_file()
  utils/cov_utils.py     get_noise_power(), get_gaus_cov(), get_SRD_cov(), class TJPCovGaus
  utils/diff_utils.py    five_pt_stencil(f, x0, h=1e-4)
  utils/theory_utils.py  compute_new_theory_vector(lk, tools, _sys_pars, _pars, return_all=False)
  utils/firecrown_interface.py  create_modeling_tools(), load_likelihood_from_yaml(),
                         create_twopoint_filter(), registries (transfer fn, PT, HM)
```

`augur/__init__.py` exports `parse, generate, Analyze, postprocess`.

### Key signatures (verbatim)

```python
# augur/generate.py
def generate_sacc_and_stats(config)  # -> S, cosmo, stats, sys_params, tp_filters
def generate(configs, return_all_outputs=False, write_sacc=True, use_sacc=None,
             sacc_path=None, lk=None, tools=None)
    # returns lk                     if return_all_outputs=False
    # returns (lk, S, tools, sys_params)   if use_sacc is None
    # returns (lk, tools, sys_params)      if use_sacc is given   <-- 3-tuple!

# augur/analyze.py
class Analyze(object):
    def __init__(self, config, likelihood=None, tools=None, req_params=None, norm_step=False)
    def f(self, x, labels, pars_fid, sys_fid, donorm=False)          # theory vector at x
    def get_derivatives(self, force=False, method=None, step=None, **kwargs)
    def get_fisher_matrix(self, method=None, save_txt=True, **kwargs)  # -> self.Fij
    def add_gaussian_priors(self, save_txt=True)                        # -> self.Fij_with_gprior
    def get_fisher_bias(self, force=False, method=None, save_txt=True, use_fid=False)  # -> self.bi
    def Jacobian_transform(self)   # Omega_c->Omega_m, sigma8->S8
    def get_Om(self); def get_S8(self)
    def add_external_fisher(...)   # raises NotImplementedError
    # attributes: Fij, Fij_df (pandas), Fij_with_gprior(_df), derivatives, bi, biased_cls,
    #             var_pars, x (pivot), pars_fid, req_params, lk, tools, data_fid

# augur/postprocess.py
def postprocess(config)
def draw_fisher_ellipses(ax, inv_F, facecolors, edgecolors, linestyles, linewidth,
                         mu=[0, 0], CL=0.95, alpha=1.0)
def get_FoM_all(fisher, par1, par2, CL)   # -> FOM, FOM2

# augur/utils/cov_utils.py
def get_noise_power(config, S, tracer_name, return_ndens=False)
def get_gaus_cov(S, lk, cosmo, fsky, config)
def get_SRD_cov(config, S)
class TJPCovGaus(tjpcov.covariance_gaussian_fsky.FourierGaussianFsky)

# augur/utils/firecrown_interface.py
def create_modeling_tools(config)                 # -> (ModelingTools, ccl.Cosmology)
def load_likelihood_from_yaml(config, ccl_factory, S, filters=[])   # S = sacc *path*
def create_twopoint_filter(combo_name, tr1, tr2, cut_low, cut_high)
```

### End-to-end flow (`augur config.yml`)

1. `parse()` renders Jinja2 with `os.environ`, loads YAML.
2. `generate_sacc_and_stats(config)`:
   - `ccl.Cosmology(**config['cosmo'])` (default `transfer_function='boltzmann_camb'`;
     applies `ccl_accuracy.spline_params/gsl_params` globally; MG `mu_Sigma` -> `MuSigmaMG`).
   - For `sources` (tracers `src0..`) and `lenses` (`lens0..`) builds n(z) on
     `z = np.linspace(0.004004, 4.004004, 1000)` via `Nz_type` registry
     (`ZDist, LensSRD2018, SourceSRD2018, ZDistFromFile`; `TopHat`/`Gaussian` exist in
     tracers but are NOT in `NZ_CLASS_REGISTRY`) and `S.add_tracer('NZ', name, z, Nz, quantity=...)`.
   - Firecrown sources: `wl.WeakLensing(sacc_tracer=...)`, `nc.NumberCounts(sacc_tracer=..., derived_scale=True)`.
   - For each `statistics.<sacc_data_type>` and `tracer_combs` pair: ell centres
     `sqrt(edge_i*edge_{i+1})`; scale cut `kmax` -> `lmax = min(kmax * chi(a(zmean_1,2)))`
     via `ccl.comoving_radial_distance` (kmax in **Mpc^-1**, no h) or `lmax` directly;
     adds zero placeholder `S.add_ell_cl(...)` (optionally `TopHat` `sacc.BandpowerWindow`);
     builds `TwoPointBinFilter` per pair (`cut_low=0.0, cut_high=lmax`) unless
     `general.ignore_scale_cuts_likelihood`.
   - Identity placeholder covariance; `sys_params = ParamsMap(config['systematics'])`.
3. `generate()`: `create_modeling_tools(config)` -> `CCLFactory` (+ optional PT/HM calculators)
   wrapped in `ModelingTools`; builds likelihood from `Firecrown_Factory` YAML
   (`TwoPointFactory.model_validate` -> `TwoPointExperiment(..., DataSourceSacc(sacc_data_file=tmp.fits, filters=...)).make_likelihood()`),
   evaluates fiducial theory via `compute_new_theory_vector`, overwrites the sacc data
   vector with `st.get_theory_vector()`, then covariance:
   - `cov_type: gaus_internal` -> `get_gaus_cov`: mode-counting Gaussian,
     `Cov = (C13 C24 + C14 C23) / (Δℓ (2ℓ+1) fsky)` with `ccl.angular_cl` on the firecrown
     CCL tracers and shot/shape noise `N = σ_e²/n̄` (src) or `1/n̄` (lens),
     `n̄ = ndens[arcmin^-2]·(180·60/π)²` per steradian, `ndens` split per bin by n(z) integral.
     Assumes identical ell edges across probes.
   - `cov_type: SRD` -> `get_SRD_cov`: loads `data/Y1_3x2_SRD_cov.npy` (540×540) or
     `Y10_3x2_SRD_cov.npy` (1000×1000), 20 ells per block, hard-coded tracer-pair order
     (Y1/Y10 picked by `'Y10' in path`), re-indexed into the sacc order.
   - `cov_type: tjpcov` -> `TJPCovGaus` (TJPCov `FourierGaussianFsky`), needs `fsky`, `IA`,
     `binning_info.ell_edges` equal to the data-vector edges; "takes a while" per source.
   - Saves `S.save_fits(config['fiducial_sacc_path'])`, then **re-creates** the likelihood
     from that file with the filters (if any filters exist), else
     `ConstGaussian(statistics=stats); lk.read(S)` (legacy path, see gotchas).
4. `Analyze`: pivot `x` from `fisher.var_pars` (fiducial values from `cosmo`/`systematics`)
   or `fisher.parameters: {name: [min, fid, max]}`; derivatives of `f(x)` =
   `compute_new_theory_vector()` (resets `lk`/`tools`, builds a single `ParamsMap` of cosmology
   + systematics, `tools.update(pmap); tools.prepare(); lk.update(pmap); lk.compute_theory_vector(tools)`).
   Methods: `'numdifftools'` (default; `nd.Jacobian(f, step=step, **derivative_args)`),
   `'5pt_stencil'` (`five_pt_stencil`, h=step), `'derivkit'` (`CalculusKit.jacobian`, default
   `adaptive, n_points=27, spacing='1%', base_abs=1e-3, ridge=1e-8`).
   Fisher: `Fij = einsum('il,lm,jm', D, lk.inv_cov, D)`; `Fij = J.T @ Fij @ J`.
   Priors: `+1/σ²` on diagonal (`gaussian_priors`). Bias: `Bj = Δd · C⁻¹ · D`, `bi = F⁻¹ Bj`.
5. `postprocess`: reads `fisher.output` and `fisher.fid_output`, draws
   `chi2.ppf(CL, 2)`-scaled ellipses for all pairs (`triangle_plot`), pair plots to
   `outdir/{p0}--{p1}.pdf`, and a LaTeX FoM table for `(w0, wa)` only.

Number of likelihood evaluations: numdifftools with a fixed `step` = **2n+1** (measured:
5 evals for n=2, 7 for n=3); `5pt_stencil` = 4n+1; numdifftools with `step=None` would be
~30n (Richardson) but augur always passes a step (default 0.01). Fisher bias adds 2.

---

## 3. Config files (examples/)

Files: `config_test.yml`, `srd_y1_3x2.yml`, `srd_y10_3x2.yml`, `srd_y1_3x2_mg.yml`,
`config_guide.md` (authoritative key reference, 670 lines), plus CosmoSIS trio
`srd_y1_3x2_cosmosis.ini`, `srd_y1_3x2_values.ini`, `srd_y1_3x2_like.py`
(`build_likelihood(_)` calls `generate('./config_test.yml', return_all_outputs=True)`).
`augur/tests/test.yaml` is the minimal CI config (uses `cov_type: SRD`). The guide mentions
`examples/cov_srd/` which does not exist in the clone.

Top-level sections and keys:

| Section | Keys (examples) |
|---|---|
| `general` | `ignore_scale_cuts` (bool, default False), `ignore_scale_cuts_likelihood` (bool), `bandpower_windows` (`None`/`TopHat`; `NaMaster` raises) |
| `cosmo` | passed verbatim to `pyccl.Cosmology`: `Omega_c, Omega_b, h, n_s, sigma8 | A_s` (exactly one), `w0, wa, Omega_k, Neff, m_nu, mass_split, T_CMB`, `transfer_function` (`boltzmann_camb` default, `boltzmann_class`, `eisenstein_hu`, `bbks`, `boltzmann_isitgr` for MG), `matter_power_spectrum` (`halofit`, `linear`), `extra_parameters.camb.{dark_energy_model: ppf, halofit_version: takahashi|mead2020_feedback|...}`, `mg_parametrization.mu_Sigma.{mu_0, sigma_0, c1_mg, c2_mg, lambda_mg}` |
| `ccl_accuracy` | `spline_params.K_MAX_SPLINE: 100`, `gsl_params.INTEGRATION_EPSREL: 1e-6`, `INTEGRATION_LIMBER_EPSREL: 1e-2` (set globally on `pyccl`) |
| `systematics` | fiducial nuisance values -> `ParamsMap`: `src{i}_mult_bias`, `src{i}_delta_z`, `ia_bias`, `alphaz`, `z_piv` (global NLA), `lens{i}_bias`, `lens{i}_alphaz`, `lens{i}_z_piv`, `lens{i}_alphag`, `lens{i}_delta_z`, `lens{i}_sigma_z` |
| `Firecrown_Factory` | exactly one key, `TwoPointFactory:` with `correlation_space: harmonic`, `number_counts_factories: [{type_source: default, global_systematics: [], include_rsd: false, per_bin_systematics: [{type: PhotoZShiftandStretchFactory}, {type: LinearBiasSystematicFactory}]}]`, `weak_lensing_factories: [{type_source: default, global_systematics: [{type: LinearAlignmentSystematicFactory, alphag: 1}], per_bin_systematics: [{type: MultiplicativeShearBiasFactory}, {type: PhotoZShiftFactory}]}]`, `cmb_factories: []`, `int_options: null` |
| `sources` | `nbins`, `ndens` [arcmin^-2, total or per-bin list], `ellipticity_error` (σ_e), `Nz_type`, `Nz_kwargs` |
| `lenses` | `nbins`, `ndens` [arcmin^-2], `Nz_type`, `Nz_kwargs`, legacy `delta_z`, `bias_type`, `bias_kwargs.b` |
| `statistics` | keys are sacc data types `galaxy_shear_cl_ee` (src,src), `galaxy_shearDensity_cl_e` (pairs are `[lens_i, src_j]`, mapped to tracers `(src_j, lens_i)`), `galaxy_density_cl` (lens,lens); each has `tracer_combs: [[i,j],...]`, `ell_edges: np.geomspace(20, 15000, 21, endpoint=True)` (string, restricted eval), `kmax` [Mpc^-1] **or** `lmax` (scalar or per-comb list) |
| `fiducial_sacc_path` | output sacc FITS path |
| `cov_options` | `cov_type: gaus_internal` (+`fsky`), `SRD` (+`SRD_cov_path`), `tjpcov` (+`fsky`, `IA`, `binning_info.ell_edges`) |
| `fisher` | `var_pars: [...]` **or** `parameters: {name: [min, fid, max]}`; `step` (default 0.01, absolute), `derivative_method` (`numdifftools` default / `5pt_stencil` / `derivkit`), `derivative_args` (dict), `transform_S8`, `transform_Omega_m` (bool), `gaussian_priors: {name: sigma}`, `output` (Fisher txt path), `fid_output`, `fisher_bias: {biased_dv: '' or path with column dv_sys, bias_params: {name: shifted value}}` |
| `postprocess` | `triangle_plot` (pdf), `latex_table`, `outdir`, `pairplots: [(w0, wa), (Omega_m, S8)]`, `CL: [0.68, 0.95]`, `facecolor`, `linecolor`, `linestyle`, `linewidth`, `size` (default (48,48) in), `labels`, `centers` |
| `pt_calculator`, `hm_calculator`, `cM_relation` | optional pyccl `nl_pt` / `hm.HaloModelCalculator` settings forwarded to `ModelingTools` |

### n(z) parametrisations (tracers/two_point.py)

- `srd_dndz(z, z0, alpha) = (z/z0)^2 exp(-(z/z0)^alpha)`.
- `SourceSRD2018(z, Nz_nbins, Nz_sigmaz, Nz_ibin, Nz_alpha=0.78, Nz_z0=0.13)`: equal-number
  bins (`equal_density_zbins`), erf-smoothed with `σ_z(1+z)`; SRD values `Nz_sigmaz: 0.05`.
- `LensSRD2018(z, Nz_center, Nz_width, Nz_nbins, Nz_sigmaz, Nz_alpha=0.94, Nz_z0=0.26, use_filter=True)`:
  top-hat slices of width `Nz_width: 0.2` at `Nz_center: np.arange(1, 6)*0.2 + 0.1`
  (0.3..1.1), Gaussian-smeared with `Nz_sigmaz: 0.03`.
- `ZDistFromFile(input_file, ibin=None, format='npy'|'ascii')`: ascii = column 0 is z,
  column `ibin+1` is n(z). Shipped files (500 rows, z to 3.5):
  `data/srd_source_bins_y1.txt`, `srd_source_bins_y10.txt` (6 cols = z + 5 bins),
  `srd_lens_bins_y1.txt` (6 cols), `srd_lens_bins_y10.txt` (11 cols = z + 10 bins);
  `.npy` dict variants (`redshift_range`, `bins`).

### How SRD Y1 / Y10 are encoded

| | `srd_y1_3x2.yml` | `srd_y10_3x2.yml` |
|---|---|---|
| cosmo | Omega_c 0.2664, Omega_b 0.0492, h 0.6727, n_s 0.9645, sigma8 0.831, w0 -1, wa 0, Omega_k 0; `eisenstein_hu` + `halofit` | same |
| sources | 5 bins, `ndens: 10`, σ_e 0.26, `ZDistFromFile` y1 | 5 bins, `ndens: 27`, σ_e 0.26, y10 file |
| lenses | 5 bins, `ndens: 18`, biases 1.562362…2.293210 | 10 bins, `ndens: 48`, biases 1.376695…2.118943 |
| IA | `ia_bias 5.717, alphaz -0.47, z_piv 0.3` | same |
| shear | 15 auto+cross pairs, `kmax: None` | 15 pairs |
| GGL | 7 pairs `[0,2],[0,3],[0,4],[1,3],[1,4],[2,4],[3,4]`, `kmax 0.202` | 25 pairs, `kmax 0.201` |
| clustering | 5 autos, `kmax 0.202` | 10 autos, `kmax 0.201` |
| ells | `np.geomspace(20, 15000, 21)` -> 20 bands, centres 23.6…12712 | same |
| cov | `SRD`, `Y1_3x2_SRD_cov.npy` (27 blocks ×20 = 540) | `Y10_3x2_SRD_cov.npy` (50 blocks ×20 = 1000) |
| fisher | `var_pars` 9 (Omega_c, sigma8, n_s, w0, wa, Omega_b, h, lens0_bias, lens1_bias), `step 1e-5`, numdifftools, SRD-table Gaussian priors (Omega_c 0.2, Omega_b 0.006, h 0.063, n_s 0.08, sigma8 0.14, w0 0.8, wa 2.0, lens bias 0.9, lens Δz 0.005, src Δz 0.002, m 0.013, ia_bias 3.9, alphaz 2.3) | same |
| general | `ignore_scale_cuts: True`, `ignore_scale_cuts_likelihood: False` (cuts applied only through firecrown filters) | same |

`config_test.yml` is the "full" variant: CAMB (`boltzmann_camb` + `halofit_version: takahashi`,
`dark_energy_model: ppf`), 27 `var_pars` (7 cosmo + 5 lens bias + 5 m + 10 Δz), `step 1e-2`,
`fisher_bias.bias_params: {Omega_c: 0.27, lens0_bias: 1.3, lens0_delta_z: 0.01, src3_delta_z: 0.005}`.
`srd_y1_3x2_mg.yml`: `mg_parametrization.mu_Sigma`, `transfer_function: boltzmann_isitgr`,
`matter_power_spectrum: linear`, `fisher.parameters` mode with `mg_musigma_mu/_sigma` and `step 0.05`.

---

## 4. CLI & API

```bash
augur config.yml [-v]        # == python -m augur.cli
```
`cli.run`: `generate(parsed_config, return_all_outputs=True)` -> if `fisher` in config:
`Analyze(config, likelihood=, tools=, req_params=).get_fisher_matrix()`; if
`fisher.fisher_bias` present: `get_fisher_bias()`; if `postprocess` in config: `postprocess()`.

Python API (README):
```python
from augur.generate import generate;  lk = generate('cfg.yml', return_all_outputs=False)
from augur.analyze import Analyze;    ao = Analyze('cfg.yml'); ao.get_fisher_bias(method='5pt_stencil'); ao.Fij, ao.bi
from augur.postprocess import postprocess; postprocess('cfg.yml')
```

Outputs written (paths from config, relative to cwd):

| File | Writer | Format |
|---|---|---|
| `{fiducial_sacc_path}` | generate | sacc FITS: tracers NZ, `*_cl*` data, covariance |
| `{fisher.output}` (e.g. `output/fisher.dat`) | `get_fisher_matrix` | `np.savetxt` NxN |
| `{output}.theory_vector`, `{output}.derivatives` (N x Ndata) | same | txt |
| `{output}.priors_only`, `{output}.with_priors` | `add_gaussian_priors` | txt |
| `{fisher.fid_output}` (e.g. `output/fiducials.dat`) | same | astropy ascii, one row, column names = params (S8/Omega_m if transformed) |
| `{fid_output}.biased_params`, `{output}.theory_vector_biased` | `get_fisher_bias` | ascii |
| `{postprocess.triangle_plot}`, `{outdir}/{p0}--{p1}.pdf`, `{postprocess.latex_table}` | postprocess | pdf / LaTeX (CL, FoM, FoM alt, sigma_w0, sigma_wa) |

No marginalised-σ table or `.npy` is written; `Fij_df` (pandas) exists only in memory.
`output/` and `syndata/` in the clone contain only `.gitignore` (`*`) — no shipped results.

**Run-time estimate (not run; no env available).** Per theory-vector evaluation: one
`ccl.Cosmology` + nonlinear P(k) (Eisenstein-Hu+halofit ≈ 0.5-1 s; CAMB ≈ 2-5 s) plus
27-50 Limber `angular_cl` at 20 ells (<0.5 s) inside firecrown. SRD Y1 example: 9 params ->
19 evals + fiducial + 2 bias ≈ 22 evals ≈ **0.5-1 min** on a laptop. `config_test.yml`:
27 params with CAMB -> 55 evals ≈ **3-5 min**. Y10 (1000-point DV): ≈ 1-2 min. `derivkit`
adaptive (27 points/param) and `tjpcov` covariance are substantially slower (several min to
tens of min). These are estimates from the code paths.

---

## 5. Coupling to firecrown / CCL / sacc / TXPipe

Firecrown imports used by augur (all present in the firecrown clone v1.16):
```python
import firecrown.likelihood.weak_lensing as wl      # wl.WeakLensing(sacc_tracer=...)
import firecrown.likelihood.number_counts as nc     # nc.NumberCounts(sacc_tracer=..., derived_scale=True)
from firecrown.likelihood.two_point import TwoPoint # TwoPoint(source0=, source1=, sacc_data_type=)
from firecrown.likelihood.gaussian import ConstGaussian   # ConstGaussian(statistics=stats); lk.read(S)
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools  # ModelingTools(ccl_factory=, pt_calculator=, hm_calculator=, cluster_abundance=, cluster_deltasigma=)
from firecrown.likelihood.factories import DataSourceSacc, TwoPointExperiment, TwoPointFactory
from firecrown.ccl_factory import CCLFactory, CCLCreationMode, CCLPureModeTransferFunction, CAMBExtraParams, PoweSpecAmplitudeParameter
from firecrown.metadata_types import Galaxies       # SHEAR_E, COUNTS
from firecrown.data_functions import TwoPointBinFilterCollection, TwoPointBinFilter   # .from_args(name1, measurement1, name2, measurement2, lower, upper)
```
Likelihood methods used: `lk.reset()`, `lk.update(pmap)`, `lk.compute_theory_vector(tools)`,
`lk.get_data_vector()`, `lk.inv_cov`, `lk.cov`, `lk.data_vector`, `lk.statistics[i].statistic`
(`.source0.sacc_tracer`, `.ells`, `.sacc_indices`, `.get_theory_vector()`, `.tracers[0].ccl_tracer`).
Tools: `tools.update(pmap)`, `tools.prepare()`, `tools.reset()`, `tools.get_ccl_cosmology().to_dict()`,
`tools.ccl_factory`, `tools.ccl_cosmo`, `tools.set_ccl_cosmology()`.

So augur uses the **new factory API** (`TwoPointFactory`/`TwoPointExperiment`/`CCLFactory`,
firecrown >= 1.14, Oct 2025) for the fiducial likelihood, and the **old manual API**
(`ConstGaussian(statistics=[TwoPoint(...)])`) as the fallback. `firecrown.ccl_factory` is a
re-export of `firecrown.modeling_tools._ccl_factory`.

CCL settings: `CCLFactory(creation_mode=PURE_CCL_MODE, pure_ccl_transfer_function=<enum>,
require_nonlinear_pk=(matter_power_spectrum != 'linear'), camb_extra_params=CAMBExtraParams(**extra_parameters.camb),
use_camb_hm_sampling=<halofit_version in mead*>, amplitude_parameter=SIGMA8|AS)`; for MG
`creation_mode=MU_SIGMA_ISITGR`. `TRANSFER_FUNCTION_REGISTRY = {boltzmann_camb, boltzmann_class,
eisenstein_hu, eh, bbks}`. With `boltzmann_camb`, `extra_parameters.camb.halofit_version` is
**required** (ValueError otherwise). `compute_new_theory_vector` strips
`transfer_function`/`matter_power_spectrum`/`extra_parameters` from the ParamsMap and maps
HMCode baryon knobs to `HMCode_logT_AGN (7.8), HMCode_eta_baryon (0.603), HMCode_A_baryon (3.13)`.

Parameter naming shared with firecrown: per-bin systematics are `Updatable(parameter_prefix=sacc_tracer)`
so names are `{tracer}_{param}`: `src0_mult_bias`, `src0_delta_z`, `lens0_bias`, `lens0_delta_z`,
`lens0_sigma_z`, `lens0_alphaz`, `lens0_alphag`, `lens0_z_piv`; global NLA IA is unprefixed
`ia_bias, alphaz, z_piv` (`alphag` fixed by the factory). Cosmology keys are CCL names
(`Omega_c, Omega_b, h, n_s, sigma8/A_s, w0, wa, Omega_k, m_nu, Neff, T_CMB`), MG
`mg_musigma_mu, mg_musigma_sigma, mg_musigma_c1, mg_musigma_c2, mg_musigma_lambda0`.

**Real sacc as input (e.g. TXPipe):** yes via
`generate(config, use_sacc=<sacc.Sacc>, sacc_path='<same file on disk>', return_all_outputs=True)`
with a `Firecrown_Factory` block. Requirements imposed by the code: tracers must have
`quantity` `galaxy_shear` / `galaxy_density` (else no firecrown source is created); tracer
names must be `src{i}` / `lens{i}` because `_get_tracers()` builds `f'src{j}'`/`f'lens{i}'`
and `get_noise_power` splits on the `src`/`lens` prefix; a covariance must be present
(replaces `lk.inv_cov` only if lengths match); `statistics` may use the flat shortcut
`statistics: {tracer_combs: [...]}` when the sacc has a single data type. **TXPipe writes
`source_{i}` / `lens_{i}` (`txpipe/twopoint_fourier.py:748,754`) without `quantity`**, so a
wrapper must rename tracers and set quantities before handing a TXPipe sacc to augur.
Augur's SRD covariances follow the firecrown/sacc convention `(src_j, lens_i)` for GGL (#147).

---

## 6. Proposed MCP tool family `augur_*`

All tools: JSON args, write to `output_dir`, return `{status, files, message, metadata}`
(template `cosmic_emulator_server/tools/common.py`: `ArtifactResult`, `resolve_outdir`,
`write_csv`, `param_slug`). Units in every docstring: `ndens` [arcmin^-2], `kmax` [Mpc^-1],
`ell` dimensionless, `m_nu` [eV], Fisher in parameter units^-2.

| Tool | Args (defaults) | Writes | Heavy? |
|---|---|---|---|
| `list_augur_examples` | — | — ; returns the 4 YAMLs + test.yaml, their survey/probe/param summary, SRD data files, parameter name conventions | no |
| `generate_forecast_config` | `survey_year: "Y1"|"Y10"`, `probes: "3x2pt"|"shear"|"clustering"|"ggl"`, `n_source_bins=5`, `n_lens_bins=5|10`, `ndens_source=10|27`, `ndens_lens=18|48`, `sigma_e=0.26`, `ell_min=20, ell_max=15000, n_ell=20`, `kmax_lens=0.2`, `cov_type="gaus_internal"|"SRD"|"tjpcov"`, `fsky=0.4`, `cosmology dict`, `var_pars list`, `step=0.01`, `derivative_method`, `gaussian_priors dict`, `bias_params dict`, `transfer_function="eisenstein_hu"`, `nz_mode="file"|"analytic"` | `forecast_config.yml` (absolute paths, no Jinja) + JSON echo | no |
| `validate_forecast_config` | `config_path` | report of tracer/combination/param consistency (every `var_pars` name exists in `cosmo`/`systematics`, SRD cov matches 5/5 or 5/10 bins, exactly one of sigma8/A_s, kmax xor lmax) | no |
| `generate_synthetic_datavector` | `config_path`, `write_nz_csv=true` | sacc FITS, `datavector.csv` (stat, tr1, tr2, ell, Cl, sigma), `nz_*.csv`, `cov.npy`, plot of C_ell with error bars | moderate (one theory eval + covariance; tjpcov slow -> dispatchable) |
| `compute_fisher` | `config_path`, `derivative_method`, `step`, `include_priors=true`, `transform_S8/Omega_m` | `fisher.csv` (labelled matrix), `fisher_with_priors.csv`, `derivatives.npy`, `marginalized_sigmas.csv` (σ_i = sqrt(F⁻¹_ii), fiducial, prior), `fom.json` (w0-wa FoM, both `get_FoM_all` conventions + DETF 1/sqrt(det Cov)), sacc | **heavy** (2n+1 or 4n+1 likelihood evals; CAMB makes it minutes) -> dispatch |
| `fisher_bias` | `config_path` or `fisher_dir`, `bias_params dict` or `biased_dv_path`, `use_fid=false` | `biased_params.csv` (b_i, b_i/σ_i), `theory_vector_biased.csv` | heavy if derivatives not cached; reuse `.derivatives` from `compute_fisher` output | 
| `plot_fisher_contours` | `fisher_csv` (one or several for overlays), `params subset`, `CL=[0.68,0.95]`, `labels` | triangle PNG/PDF, pair plots | no |
| `combine_fisher` | list of `fisher.csv`, parameter alignment, optional external priors | summed Fisher + σ table (augur's `add_external_fisher` is NotImplemented; do it in the wrapper) | no |
| `derivative_step_check` (optional) | `config_path`, `param`, `steps=[1e-4,1e-3,1e-2]` | CSV of d(data)/dθ vs step and relative change — stability diagnostic | heavy-ish |

Dispatch: `compute_fisher`, `fisher_bias`, `generate_synthetic_datavector` with
`cov_type=tjpcov` or `derivative_method=derivkit`. Node deps: firecrown+pyccl+camb+sacc<2.2+tjpcov+numdifftools.

---

## 7. Gotchas

1. **firecrown coupling**: requires firecrown >= 1.14 factory API (`TwoPointFactory`,
   `TwoPointExperiment`, `DataSourceSacc(sacc_data_file=<path>)`, `CCLFactory`). The firecrown
   clone is 1.16-dev and still exposes everything augur imports. `DataSourceSacc` needs a
   file **path** (augur writes a temp FITS), so in-memory-only flows are not possible.
2. **sacc pin `< 2.2`** (environment.yml, commit #149) — the conda env must honour it.
3. **Legacy fallback silently drops systematics**: in `generate()`, if the filter list is
   empty (`general.ignore_scale_cuts_likelihood: True`, as in `srd_y1_3x2_mg.yml`) the
   returned likelihood is `ConstGaussian(statistics=stats)` built from bare
   `WeakLensing`/`NumberCounts` sources (no IA, Δz, m), even though the fiducial data vector
   was computed with the `Firecrown_Factory` systematics. Keep `ignore_scale_cuts_likelihood: False`.
4. **Return arity** of `generate(..., return_all_outputs=True)` differs: 4-tuple
   `(lk, S, tools, sys_params)` normally, 3-tuple `(lk, tools, sys_params)` with `use_sacc`.
5. **Step sizes are absolute**, not relative (`step` is passed straight to `nd.Jacobian` /
   `five_pt_stencil`); `norm_step=True` (normalise by `parameters` bounds) is only reachable
   from the Python API, never from the CLI. The SRD examples use `step: 1e-5` with
   numdifftools + CCL splines — potentially noise-dominated; `config_test.yml` uses 1e-2
   (large for Δz ~ 0.002 priors). A step-scan tool is worthwhile. numdifftools with
   `step=None` would cost ~30 evals/param.
6. **Scale cuts**: `kmax` is in Mpc^-1 (no h), converted with `chi(zmean)` of the n(z);
   `kmax` and `lmax` are mutually exclusive; `ignore_scale_cuts: False` with
   `ignore_scale_cuts_likelihood: True` raises. Filters use `TwoPointFilterMethod.SUPPORT`
   with `cut_low=0`.
7. **Postprocess assumes `fisher.var_pars`** (`config["fisher"]["var_pars"]`) — crashes with
   `KeyError` in `parameters` mode (e.g. the MG example); pair plots naming `Omega_m`/`S8`
   are skipped silently unless the transforms are on; the FoM table is only produced when
   both `w0` and `wa` are varied; FoM includes the `chi2.ppf(CL, 2)` scale (not DETF).
   Triangle figure default is 48x48 inches.
8. **Fisher bias** refuses `transform_S8`/`transform_Omega_m`; `bias_params` must name
   existing cosmology or systematics keys.
9. `gaussian_priors` keys must be in `var_pars` (or be `Omega_m`/`S8` with the transform on)
   or `Analyze.__init__` raises — the SRD examples list priors for 30+ params but vary 9,
   so **the shipped `srd_y1_3x2.yml`/`srd_y10_3x2.yml` would raise `ValueError` as written**
   unless the prior list is trimmed; `test.yaml`/`config_test.yml` have no priors.
10. `ccl_accuracy` is applied by mutating global `pyccl.spline_params`/`gsl_params`
    (process-wide, persists across tool calls); `_create_ccl_factory` also `config.pop`s
    sections, so always pass a deep copy of the config dict.
11. `boltzmann_camb` requires `extra_parameters.camb.halofit_version`; `A_s` and `sigma8`
    are mutually exclusive both in `cosmo` and in `var_pars`.
12. `Nz_type` values `TopHat`/`Gaussian` are documented but not in `NZ_CLASS_REGISTRY`
    (`ValueError`). `Nz_center` list length must equal `nbins`.
13. SRD covariance: fixed 20 ells per block and hard-coded tracer-pair lists (Y1: 5+5 bins,
    27 pairs; Y10: 5+10 bins, 50 pairs); any other binning/pairing needs `gaus_internal`
    or `tjpcov`. `get_gaus_cov` assumes identical ell edges for all probes.
14. Example YAMLs reference `{{ env['AUGUR_DIR'] }}`; `AUGUR_DIR` must point at the clone
    (data files) or the wrapper must render absolute paths itself. Output paths like
    `output/fisher.dat` are relative to the cwd.
15. TXPipe/real sacc input: rename `source_i -> srci`, `lens_i -> lensi`, set tracer
    `quantity`, ensure a covariance, keep `sacc_path` consistent with the object passed.
16. `data/` ships only SRD n(z) (y1/y10, txt and npy) and the two SRD covariances;
    `output/`, `syndata/` are empty. CMB lensing (`cmb_lensing`) and NaMaster windows raise
    `NotImplementedError`; `add_external_fisher` is unimplemented; real-space statistics
    are not generated ("Only C_ls available").
