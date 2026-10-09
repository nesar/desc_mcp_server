# Survey: firecrown (LSST DESC likelihood framework) for an MCP tool family

Clone surveyed: `/Users/nesar/Projects/Tutorials/desc-mcp-server/firecrown`
(git `v1.16.0-74-g51193df`, HEAD 2026-10-06, i.e. ~74 commits past the 1.16.0 release).
Companion clone added under `external/`: `sacc` (v2.4, data format). `augur` (DESC forecasting
wrapper around firecrown, shows synthetic-data-vector generation) is at
`/Users/nesar/Projects/Tutorials/desc-mcp-server/augur` (see `notes/survey_augur.md`).
Also already present in `external/`: `desc-python`, `desc-cosmology-env` (NERSC/SLAC
DESC conda stacks that ship firecrown).

All class/function names and signatures below were read from the source in the clone.

---

## 1. Install & import

### What the package needs
`pyproject.toml`: `requires-python >= 3.12`; runtime deps
`astropy, camb<2.0, cobaya, cosmosis>=3.0, fitsio, isitgr, lsstdesc-crow, matplotlib,
numcosmo_py>=0.27, numpy>=2.0,<2.4, pyccl>=2.8.0, pydantic, pyyaml, rich, sacc>=2.4,
scipy>=1.13, typer, typing-extensions`. Console scripts: `firecrown` (typer CLI),
`firecrown-link-checker`.

**Hard import-time dependencies of the core likelihood API** (verified by grepping imports):
- `pyccl`, `sacc`, `pydantic`, `numpy`, `scipy` — obviously.
- **`numcosmo_py`**: `firecrown/generators/_inferred_galaxy_zdist.py:9` does
  `from numcosmo_py import Ncm` at module level; `firecrown/likelihood/_two_point.py:22`
  does `import firecrown.generators as gen`; `firecrown/likelihood/__init__.py` imports
  `_two_point`. So `import firecrown.likelihood` **requires NumCosmo**.
- **`crow`** (`lsstdesc-crow`): `firecrown/likelihood/_binned_cluster.py:6` imports
  `crow.properties.ClusterProperty` at module level and `likelihood/__init__.py`
  re-exports `BinnedClusterNumberCounts`, so `import firecrown.likelihood` **requires crow**
  (crow pulls `clmm`).
- `cobaya`, `cosmosis`, `numcosmo` connectors are only imported from
  `firecrown.connector.{cobaya,cosmosis,numcosmo}`; `firecrown.app.analysis` imports all
  three (`_numcosmo.py` imports `numcosmo_py`, `_cobaya.py` imports
  `firecrown.connector.cobaya.likelihood` -> `cobaya`). The `firecrown` CLI therefore needs
  cobaya installed; the library API does not.

### Distribution channels (checked live)
- **PyPI: firecrown is NOT published** (`pip index versions firecrown` -> no distribution;
  pypi.org/pypi/firecrown/json -> no `info`). `numcosmo_py`/`numcosmo` are **not on PyPI**
  either (404). `sacc` 2.4, `pyccl` 3.3.6, `lsstdesc-crow` 1.0.12, `isitgr` 1.6.5 are on PyPI.
- **conda-forge: `firecrown` 1.16.0** (noarch, `depends: firecrown-deps >=1.16.0, python >=3.12`),
  **`numcosmo` 0.28.0** for linux-64, linux-aarch64, osx-64, osx-arm64 (nompi and mpi builds).
  augur's README confirms: "firecrown ... is only available via conda".
- Developer install (tutorial `_developer_installation.qmd`):
  `conda env create --name firecrown_developer --file firecrown/environment.yml`, set
  `CSL_DIR=${CONDA_PREFIX}/cosmosis-standard-library FIRECROWN_DIR=${PWD}/firecrown`,
  build CSL (`cosmosis-build-standard-library`), then `python -m pip install --no-deps --editable .`
  Note `environment.yml` is *generated* from `dependencies.yaml` (`make deps-sync`) and includes
  dev tooling (sphinx, quarto, pylint...). Cobaya and isitgr come via `pip:` inside the env file.
- Version at runtime comes from `importlib.metadata.version("firecrown")`
  (`firecrown/__init__.py`), so the package must be *installed* (pip/conda), not just on
  `sys.path`.

### Local conda environments (probed)
`conda env list`: base, agent_systems, cosmic-emu, gal_env, graph_env, hep-genesis, llm_env,
mica, open-webui, opencosmo_env, paraview_mcp, spectra-tutorial.
- `cosmic-emu` (py 3.12.13): pyccl 3.3.6, camb 1.6.6 present; **no sacc, cobaya, cosmosis,
  numcosmo_py, isitgr, crow, firecrown**.
- `llm_env` (py 3.10.9): pyccl 3.3.1, camb 1.5.5; nothing else. Python too old (<3.12).
- `hep-genesis` (py 3.12.13): none of the stack.
(`import firecrown` "succeeded" with version `?` in cosmic-emu/llm_env only because cwd
contained the `firecrown/` clone dir -> namespace-package false positive. Not usable.)

**Conclusion:** no existing env can run firecrown; a new conda env (or micromamba on HPC) is
required. See section 7 for the minimal recipe and the scratch-env smoke test.

---

## 2. Architecture

### 2.1 Updatable / parameters (`firecrown/updatable/`)
- `class Updatable(ABC)`: `__init__(self, parameter_prefix: None|str=None)`.
  Assigning `self.x = register_new_updatable_parameter(default_value=...)` registers a
  **SamplerParameter** whose full name is `f"{prefix}_{name}"` (or `name` if prefix is
  None/"" or `shared=True`). Passing a float `register_new_updatable_parameter(value, default_value=...)`
  makes an **InternalParameter** (fixed, not settable by the sampler).
  Lifecycle (final methods): `update(params: ParamsMap, updated_record=None)`,
  `reset()`, `is_updated()`, `required_parameters() -> RequiredParameters`,
  `get_derived_parameters()`, `get_params_names()`. Subclasses hook `_update(params)` and `_reset()`.
  Nested Updatables/collections assigned as attributes are auto-tracked and updated recursively.
- `class ParamsMap(*args: Mapping[str,float], **kwargs: float)`: dict-like str->float with
  `get_from_prefix_param(prefix, param)`, `get_from_full_name`, `union`, `update(dict)`,
  `use_lower_case_keys(bool)`, `get_unused_keys()`. Values must be float or list[float].
  `update()` raises `MissingSamplerParameterError(fullname)` for any missing sampler parameter.
- `RequiredParameters.get_params_names()`, `.get_default_values() -> dict[str,float]`;
  helpers `get_default_params(*updatables) -> dict`, `get_default_params_map(*updatables) -> ParamsMap`
  (`firecrown.updatable`). `handle_unused_params(params, updated_records, raise_on_unused)`.
- `firecrown.parameters` is a deprecated alias module for `firecrown.updatable`.

### 2.2 ModelingTools and the pyccl bridge
`firecrown.modeling_tools` (re-exports `CCLFactory` etc.; `firecrown.ccl_factory` is the same):
```python
class ModelingTools(Updatable):
    def __init__(self, *, pt_calculator: None|pyccl.nl_pt.EulerianPTCalculator=None,
                 hm_calculator: None|pyccl.halos.HMCalculator=None, cM_relation: None|str=None,
                 pk_modifiers: None|Collection[PowerspectrumModifier]=None,
                 ccl_factory: None|CCLFactory=None)      # default CCLFactory() if None
    def prepare(self, *, calculator_args: None|CCLCalculatorArgs=None) -> None  # builds pyccl.Cosmology
    def get_ccl_cosmology(self) -> pyccl.Cosmology
    def add_pk(name, pk2d); get_pk(name); has_pk(name); get_pt_calculator(); get_hm_calculator()
```
`prepare()` raises if not updated / already prepared; `reset()` clears cosmology and pk table.

```python
class CCLFactory(Updatable, BaseModel):   # pydantic fields are frozen/config, parameters are sampler params
    require_nonlinear_pk: bool = False
    amplitude_parameter: PoweSpecAmplitudeParameter = SIGMA8     # "as" | "sigma8"
    mass_split: NeutrinoMassSplits = NORMAL                       # pyccl enum; also "sum","single","equal","list"
    num_neutrino_masses: int|None = None                          # only for LIST/SUM
    creation_mode: CCLCreationMode = DEFAULT                      # "default" | "mu_sigma_isitgr" | "pure_ccl_mode"
    pure_ccl_transfer_function: CCLPureModeTransferFunction = BOLTZMANN_CAMB  # bbks, boltzmann_camb, boltzmann_class, eisenstein_hu, eisenstein_hu_nowiggles
    use_camb_hm_sampling: bool = False; allow_multiple_camb_instances: bool = False
    camb_extra_params: CAMBExtraParams|None = None; ccl_spline_params: CCLSplineParams|None = None
    def create(self, calculator_args: CCLCalculatorArgs|None=None) -> pyccl.Cosmology
    def get(self) -> pyccl.Cosmology
```
Sampler parameters registered by `CCLFactory.__init__` (names **exactly as pyccl**):
`Omega_c, Omega_b, h, n_s, Omega_k, Neff, m_nu, w0, wa, T_CMB` plus `sigma8` **or** `A_s`
(depending on `amplitude_parameter`). Defaults come from `pyccl.CosmologyVanillaLCDM()`
(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81, Omega_k=0, Neff=3.044, m_nu=0,
w0=-1, wa=0, T_CMB=2.7255); `A_s` default 2.1e-9. With `mass_split in (LIST, SUM)` the
parameters are `m_nu, m_nu_2, ... m_nu_{num_neutrino_masses}`.
- `creation_mode=DEFAULT`: if `prepare(calculator_args=...)` is given (CosmoSIS/Cobaya pass
  CAMB background + linear P(k)), pyccl runs in **calculator mode**; otherwise plain `pyccl.Cosmology`.
- `PURE_CCL_MODE`: CCL computes everything itself (transfer function selectable) -> **the mode
  to use for a standalone Python tool** (no Boltzmann code from a sampler framework needed).
  `require_nonlinear_pk=True` is needed whenever a statistic asks for nonlinear P(k) (3x2pt).
- `MU_SIGMA_ISITGR`: mu-Sigma MG via isitgr; adds a `MuSigmaModel` with params
  `mg_musigma_mu, mg_musigma_sigma, mg_musigma_c1, mg_musigma_c2, mg_musigma_lambda0`.

### 2.3 Likelihood classes (`firecrown/likelihood/`, base in `firecrown/likelihood_base.py`)
- `class Likelihood(Updatable)`: abstract `read(sacc_data: sacc.Sacc)`,
  `compute_loglike(tools: ModelingTools) -> float`,
  `compute_loglike_for_sampling(tools)` (returns `-inf` on `CCL_ERROR` integration errors),
  `make_realization_vector()`, `make_realization(sacc_data, add_noise=True, strict=True) -> sacc.Sacc`.
- `class GaussFamily(Likelihood)`: `__init__(self, statistics: Sequence[Statistic], use_cholesky: bool=True)`;
  `@classmethod create_ready(cls, statistics, covariance: ndarray)`; `read(sacc)` (requires
  `sacc_data.covariance`), `get_data_vector()`, `compute_theory_vector(tools)`,
  `get_theory_vector()`, `compute_chisq(tools)`, `get_cov()`, `get_sacc_indices()`,
  `make_realization(sacc_data, add_noise=True, strict=True)`.
  Enforced **state machine** (`State` enum INITIALIZED -> READY -> UPDATED -> COMPUTED): `read()` before
  `update()`; `update()` before `compute_*`; `compute_theory_vector()` before `get_theory_vector()`.
- `class ConstGaussian(GaussFamily)`: `compute_loglike = -0.5*chisq`;
  `make_realization_vector()` = theory + Cholesky * N(0,1).
- `class StudentT(GaussFamily)`: `__init__(statistics, nu: None|float=None)` -> sampler param `nu`
  (default 3.0); loglike `-0.5*nu*log(1+chi2/(nu-1))` (Sellentin & Heavens).
- `class ConstGaussianPM(ConstGaussian)`: point-mass marginalisation for gamma_t.
- Loaders (`firecrown.likelihood`): `load_likelihood(likelihood_name: str, build_parameters: NamedParameters)
  -> tuple[Likelihood, ModelingTools]` (tries script path, then module name),
  `load_likelihood_from_script(filename, build_parameters)`, `load_likelihood_from_module(module, build_parameters)`,
  `load_likelihood_from_module_type(module, build_parameters, build_likelihood_name="build_likelihood")`.
- `class NamedParameters(dict-like)`: `get_string/get_int/get_float/get_bool(name, default)`,
  `get_int_array/get_float_array`, `convert_to_basic_dict()`, `set_from_basic_dict()`.

**"Likelihood factory file" convention:** a python file (or module) defining
`def build_likelihood(build_parameters: NamedParameters) -> tuple[Likelihood, ModelingTools]`.
All three connectors call `load_likelihood(path_or_module, build_parameters)`; build parameters
come from the sampler config (CosmoSIS `[firecrown_likelihood]` keys, Cobaya `build_parameters:`,
NumCosmo `likelihood_build_parameters`). The old convention (module-level `likelihood = ...`,
see `docs/non-developer-mode-example/des_y1_3x2pt.py`) is deprecated (`lkscript_old.py` test).
The YAML factory is itself exposed as a module: `firecrown.likelihood.factories.build_two_point_likelihood(build_parameters)`
reads `build_parameters.get_string("likelihood_config")` -> a `TwoPointExperiment` YAML.

### 2.4 Statistics
- `class Statistic(Updatable)`: `__init__(parameter_prefix=None)`, `read(sacc)`, `get_data_vector() -> DataVector`,
  `compute_theory_vector(tools) -> TheoryVector`, `get_theory_vector()`, attribute `sacc_indices`.
- `class TwoPoint(Statistic)` (`likelihood/_two_point.py`):
  ```python
  TwoPoint(sacc_data_type: str, source0: Source, source1: Source, *,
           interp_ells_gen: LogLinearElls = LogLinearElls(), ell_or_theta: None|EllOrThetaConfig = None,
           tracers: None|TracerNames = None, int_options: ClIntegrationOptions|None = None,
           apply_interp: ApplyInterpolationWhen = DEFAULT, normalize_window: bool = True)
  ```
  Supported `sacc_data_type`: `galaxy_density_cl`, `galaxy_density_xi`, `galaxy_shearDensity_cl_e`,
  `galaxy_shearDensity_xi_t`, `galaxy_shear_cl_ee`, `galaxy_shear_xi_minus`, `galaxy_shear_xi_plus`,
  `cmbGalaxy_convergenceDensity_xi`, `cmbGalaxy_convergenceShear_xi_t` (+ CMB cl types via CMBConvergence).
  `ell_or_theta` dict keys: `minimum, maximum, n, binning ("log"|"lin")` -- used to *generate* ells/thetas
  when the sacc file has none (theory-only mode). Properties: `ells`, `thetas`, `cells`, `window`, `sacc_tracers`.
  Class constructors: `TwoPoint.from_metadata(metadata_seq: Sequence[TwoPointHarmonic|TwoPointReal], tp_factory)`
  (theory-only, ready, no data), `TwoPoint.from_measurement(measurements: Sequence[TwoPointMeasurement], tp_factory)`
  (ready with data), `TwoPoint.from_metadata_index(indices, tp_factory)` (deprecated; needs `read()`).
  Internals in `firecrown.models.two_point.TwoPointTheory`; harmonic C_ell via `pyccl.angular_cl`
  with `ClIntegrationOptions(method: ClIntegrationMethod [LIMBER|FKEM_AUTO|FKEM_L_LIMBER],
  limber_method: ClLimberMethod [GSL_QAG_QUAD|GSL_SPLINE], l_limber, limber_max_error, fkem_chi_min, fkem_Nchi)`
  (`firecrown.utils`). Real-space xi via `pyccl.correlation` on ells from
  `LogLinearElls(minimum=2, midpoint=50, maximum=60000, n_log=200)`.
- `class Supernova(Statistic)`: `__init__(sacc_tracer: str)`; sampler param `{tracer}_M` (default -19.2);
  reads `supernova_distance_mu` data points of a sacc `MiscTracer`, tag `z`; theory = `5 log10(d_L) + 25 + M`
  style distance modulus from `pyccl.luminosity_distance`.
- `class BinnedClusterNumberCounts(BinnedCluster)` / `BinnedClusterShearProfile`:
  `BinnedCluster(cluster_properties: crow.properties.ClusterProperty, survey_name: str, cluster_recipe: crow recipe)`;
  theory delegated to **crow** (`ClusterAbundance`, `ExactBinnedClusterRecipe`, `MurataBinned` mass-richness).
  Needs sacc with `bin_z`, `bin_richness`, `survey` tracers and `cluster_counts` / `cluster_mean_log_mass` data types.
- `class CMBConvergence(Source)` + `CMBConvergenceFactory(type_source, z_source=1100.0, scale=1.0)`.
- `TrivialStatistic` (tests), `UpdatableClusterObjects` (wraps crow objects as Updatables).

### 2.5 Sources and systematics
- `class Source(Updatable)`: `__init__(sacc_tracer: str)`; `read(sacc)`, `get_tracers(tools) -> Sequence[Tracer]`, `get_scale()`.
  `SourceGalaxy[ArgsT](Source)` holds `tracer_args` (z, dndz, scale, ...), `systematics: UpdatableCollection`.
  `SourceGalaxySystematic.apply(tools, tracer_arg) -> tracer_arg'` is the systematic contract.
- `firecrown.likelihood.weak_lensing`:
  `WeakLensing(*, sacc_tracer: str, scale: float=1.0, systematics=None)`;
  `WeakLensing.create_ready(tomographic_bin: TomographicBin, systematics=None)`.
  Systematics (parameter prefix = `sacc_tracer`; `None`/`""` prefix -> global, unprefixed name):
  | class | sampler parameters (defaults) |
  |---|---|
  | `PhotoZShift(sacc_tracer, active=True)` | `{t}_delta_z` (0.0) |
  | `PhotoZShiftandStretch(sacc_tracer, active=True)` | `{t}_delta_z` (0.0), `{t}_sigma_z` (1.0) |
  | `MultiplicativeShearBias(sacc_tracer)` | `{t}_mult_bias` (1.0 default! DES uses ~0.012) |
  | `LinearAlignmentSystematic(sacc_tracer=None, alphag=1.0)` | `ia_bias` (0.5), `alphaz` (0.0), `z_piv` (0.5); `alphag` is internal (fixed) unless passed `None` |
  | `MassDependentLinearAlignmentSystematic(sacc_tracer=None)` | `ia_amplitude` (5.74, shared), `ia_mass_scaling` (0.44, shared), `{t}_red_fraction` (1.0), `{t}_log10_average_halo_mass` (13.5), `pivot_log10_halo_mass` |
  | `TattAlignmentSystematic(sacc_tracer=None, include_z_dependence=False)` | `ia_a_1` (1.0), `ia_zpiv_1` (0.62), `ia_alphaz_1` (0), `ia_a_2` (0.5), `ia_zpiv_2`, `ia_alphaz_2`, `ia_a_d` (0.5), `ia_zpiv_d`, `ia_alphaz_d` -- needs `ModelingTools(pt_calculator=EulerianPTCalculator(...))` |
  | `HMAlignmentSystematic()` | `ia_a_1h` (1e-4), `ia_a_2h` (1.0) -- needs `hm_calculator` |
  | `SelectField(field="delta_matter")` | none (picks which P(k) to use) |
  Formulas (from `apply()`): shear `scale *= (1 + mult_bias)`; linear IA
  `A_IA(z) = ia_bias * ((1+z)/(1+z_piv))**alphaz * D(z)**(alphag-1)` fed to `pyccl.WeakLensingTracer(ia_bias=...)`;
  linear bias `b(z) = {t}_bias * ((1+z)/(1+z_piv))**alphaz * D(z)**alphag`; photo-z shift `dndz(z - delta_z)`
  (active) with optional stretch `sigma_z`.
- `firecrown.likelihood.number_counts`:
  `NumberCounts(*, sacc_tracer: str, has_rsd: bool=False, derived_scale: bool=False, scale: float=1.0, systematics=None)`
  registers **`{t}_bias`** (default 1.5) itself. Systematics:
  | class | parameters (defaults) |
  |---|---|
  | `PhotoZShift`, `PhotoZShiftandStretch` | `{t}_delta_z`, `{t}_sigma_z` |
  | `LinearBiasSystematic(sacc_tracer)` | `{t}_alphaz` (0.0), `{t}_alphag` (0.0), `{t}_z_piv` (0.5) -- b(z) evolution on top of `{t}_bias` |
  | `PTNonLinearBiasSystematic(sacc_tracer=None)` | `{t}_b_2` (1.0), `{t}_b_s` (1.0) -- needs pt_calculator |
  | `MagnificationBiasSystematic(sacc_tracer)` | `{t}_r_lim` (24.0), `{t}_sig_c` (9.83), `{t}_eta` (19.0), `{t}_z_c` (0.39), `{t}_z_m` (0.055) |
  | `ConstantMagnificationBiasSystematic(sacc_tracer)` | `{t}_mag_bias` (1.0) |
  Factories (pydantic, discriminated by `type`): `PhotoZShiftFactory`, `PhotoZShiftandStretchFactory`
  (in `likelihood_base`), `MultiplicativeShearBiasFactory`, `LinearAlignmentSystematicFactory(alphag: float|None=1.0)`,
  `TattAlignmentSystematicFactory(include_z_dependence=False)`, `LinearBiasSystematicFactory`,
  `PTNonLinearBiasSystematicFactory`, `MagnificationBiasSystematicFactory`,
  `ConstantMagnificationBiasSystematicFactory`. Each has `create(bin_name) -> systematic` and
  `create_global()` (raises for per-bin-only systematics like mult_bias, linear bias, magnification).

### 2.6 The factory / YAML API (`firecrown.likelihood.factories`)
```python
class WeakLensingFactory(BaseModel):          # likelihood/weak_lensing/_factories.py
    type_source: TypeSource = TypeSource.DEFAULT
    per_bin_systematics: Sequence[WeakLensingSystematicFactory] = []
    global_systematics:  Sequence[WeakLensingSystematicFactory] = []
    def create(self, tomographic_bin: ProjectedField) -> WeakLensing
    def create_from_metadata_only(self, sacc_tracer: str) -> WeakLensing
class NumberCountsFactory(BaseModel):          # likelihood/number_counts/_factories.py
    type_source: TypeSource = DEFAULT; per_bin_systematics; global_systematics; include_rsd: bool = False
class TwoPointFactory(BaseModel):              # likelihood/_two_point.py (extra="forbid", frozen)
    correlation_space: TwoPointCorrelationSpace            # "real" | "harmonic"
    weak_lensing_factories: list[WeakLensingFactory] = []
    number_counts_factories: list[NumberCountsFactory] = []
    cmb_factories: list[CMBConvergenceFactory] = []
    int_options: ClIntegrationOptions | None = None
    def get_factory(measurement, type_source=DEFAULT); from_measurement(tpms); from_metadata(metadata_seq)
class DataSourceSacc(BaseModel):               # likelihood/factories/_models.py
    sacc_data_file: str; filters: TwoPointBinFilterCollection|None = None; normalize_window: bool = True
    def get_sacc_data(self) -> sacc.Sacc       # path resolved relative to the YAML file's dir
class TwoPointExperiment(BaseModel):
    two_point_factory: TwoPointFactory; data_source: DataSourceSacc; ccl_factory: CCLFactory|None = None
    @classmethod load_from_yaml(cls, file: str|Path) -> TwoPointExperiment   # strict pydantic validation
    def make_likelihood(self) -> Likelihood    # ConstGaussian over all 2pt data in the sacc file
def build_two_point_likelihood(build_parameters: NamedParameters) -> tuple[Likelihood, ModelingTools]
    # build_parameters["likelihood_config"] = path to the YAML; tools = ModelingTools(ccl_factory=exp.ccl_factory)
```
Example experiment YAML (generated by `firecrown examples des_y1_3x2pt --factory-type yaml_pure_ccl`):
```yaml
data_source:
  sacc_data_file: des_y1_3x2pt.sacc
two_point_factory:
  correlation_space: real
  number_counts_factories:
    - type_source: default
      global_systematics: []
      per_bin_systematics:
        - type: PhotoZShiftFactory
  weak_lensing_factories:
    - type_source: default
      global_systematics:
        - {type: LinearAlignmentSystematicFactory, alphag: 1}
      per_bin_systematics:
        - type: MultiplicativeShearBiasFactory
        - type: PhotoZShiftFactory
ccl_factory:
  creation_mode: 'pure_ccl_mode'
  require_nonlinear_pk: true
```
Scale cuts: `data_functions.TwoPointBinFilterCollection(filters: list[TwoPointBinFilter], require_filter_for_all=False, allow_empty=False)`,
`TwoPointBinFilter(spec: list[TwoPointTracerSpec(name, measurement)], interval: (lo, hi), method: TwoPointFilterMethod=SUPPORT)`,
helpers `TwoPointBinFilter.from_args(name1, m1, name2, m2, lower, upper)` / `from_args_auto(name, m, lower, upper)`.
Any pydantic model <-> YAML: `firecrown.utils.base_model_to_yaml(model)`, `base_model_from_yaml(cls, yaml_str)`.
Also useful: `load_sacc_data(filepath) -> sacc.Sacc` (auto-detects HDF5 vs FITS, `firecrown.utils` /
`firecrown.likelihood.factories`), `save_to_sacc(sacc_data, data_vector, indices, strict=True) -> sacc.Sacc`.

### 2.7 Metadata types / functions / generators
`firecrown.metadata_types`:
- `Galaxies` enum: `COUNTS, SHEAR_E, SHEAR_T, SHEAR_MINUS, SHEAR_PLUS, PART_OF_XI_MINUS, PART_OF_XI_PLUS`;
  `CMB.CONVERGENCE`; `Clusters.COUNTS`; `Measurement = Galaxies|CMB|Clusters`.
- `@dataclass TomographicBin(bin_name: str, z: ndarray, dndz: ndarray, measurements: set[Measurement], type_source: TypeSource=DEFAULT)`
  (alias **`InferredGalaxyZDist`** = old name); `CMBLensing(bin_name, z_lss, measurements, type_source)`;
  Protocol `ProjectedField`.
- `TwoPointXY(x: ProjectedField, y: ProjectedField, x_measurement, y_measurement)`,
  `TwoPointHarmonic(XY, ells: ndarray[int], window=None, window_ells=None)`, `TwoPointReal(XY, thetas)`;
  `TwoPointCorrelationSpace` (REAL/HARMONIC), `TwoPointFilterMethod`, `TracerNames(name1, name2)`,
  bin-pair selectors (`AutoBinPairSelector`, `CrossBinPairSelector`, `ThreeTwoBinPairSelector(lens_dist, source_dist, source_lens_dist)`, ...).
`firecrown.metadata_functions`:
- `extract_all_tracers_tomographic_bins(sacc_data, allow_mixed_types=False) -> list[TomographicBin]`
  (**this is the `extract_all_tracers_inferred_galaxy_zdists` of older versions**),
  `extract_all_tracers_projected_fields(sacc_data, ...)` (NZ + Map tracers),
  `extract_all_harmonic_metadata(sacc_data, allowed_data_type=None, allow_mixed_types=False, bin_pair_selector=None, normalize=True) -> list[TwoPointHarmonic]`,
  `extract_all_real_metadata(sacc_data, ...) -> list[TwoPointReal]`,
  `extract_all_real_metadata_indices / extract_all_harmonic_metadata_indices` (deprecated path),
  `make_all_photoz_bin_combinations(bins) -> list[TwoPointXY]`, `make_binned_two_point_filtered(bins, selector)`,
  `make_measurement(s)`, `make_correlation_space`.
`firecrown.data_functions`: `extract_all_harmonic_data(sacc_data, allow_mixed_types=False, allowed_data_type=None, normalize=True) -> list[TwoPointMeasurement]`,
`extract_all_real_data(sacc_data, ...)` (both require a dense covariance), `check_two_point_consistence_harmonic/real(list)`.
`firecrown.data_types`: `TwoPointMeasurement(data, indices, covariance_name, metadata)`, `DataVector`, `TheoryVector`.

`firecrown.generators` (LSST SRD n(z); **uses NumCosmo splines**):
```python
class ZDistLSSTSRD:
    def __init__(self, alpha: float, beta: float, z0: float, max_z: float=5.0, use_autoknot: bool=False,
                 autoknots_reltol: float=1e-4, autoknots_abstol: float=1e-15)
    @classmethod year_1_lens(alpha=Y1_LENS_ALPHA, beta=Y1_LENS_BETA, z0=Y1_LENS_Z0, **opts) ; year_1_source ; year_10_lens ; year_10_source
    def distribution(self, z) ; distribution_zp(zp, sigma_z)
    def equal_area_bins(self, n_bins: int, sigma_z: float, last_z: float, use_true_distribution=False) -> edges
    def binned_distribution(self, *, zpl: float, zpu: float, sigma_z: float, z: ndarray, name: str,
                            measurements: set[Measurement]) -> TomographicBin
class ZDistLSSTSRDBin(BaseModel): zpl, zpu, sigma_z, z: LinearGrid1D(start,end,num)|RawGrid1D(values), bin_name, measurements
class ZDistLSSTSRDBinCollection(BaseModel): alpha, beta, z0, bins: list[ZDistLSSTSRDBin], max_z=5.0, use_autoknot=True, ...
    def generate(self) -> list[TomographicBin]
get_y1_lens_bins() -> {"edges": linspace(0.2,1.2,6), "sigma_z": 0.03}; get_y1_source_bins() (5 equal-area bins, sigma_z 0.05, last_z 3.5)
get_y10_lens_bins() (10 bins); get_y10_source_bins()
LSST_Y1_LENS_HARMONIC_BIN_COLLECTION, LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION, LSST_Y10_* (module constants, lazily built)
    # bin names: lsst_y1_lens{i} (Galaxies.COUNTS), lsst_y1_source{i} (Galaxies.SHEAR_E)
class LogLinearElls(BaseModel): minimum=2, midpoint=50, maximum=60000, n_log=200; generate(min_ell=None, max_ell=None)
generate_bin_centers(*, minimum, maximum, n, binning="log"); generate_ells_cells(cfg); generate_reals(cfg)
```

---

## 3. Connectors (`firecrown/connector/`)

| framework | entry point | config | cosmology source |
|---|---|---|---|
| **CosmoSIS** | `firecrown/connector/cosmosis/likelihood.py` (module with `setup/execute/cleanup`; class `FirecrownLikelihood(config: datablock)`) | `.ini`: `[firecrown_likelihood] file=<site-packages>/firecrown/connector/cosmosis/likelihood.py; likelihood_source=<factory.py or module>; sampling_parameters_sections=<section names>; <extra keys -> build_parameters>`; values in `[cosmological_parameters]` (CosmoSIS names `omega_c omega_b h0 n_s sigma_8/A_s omega_k w wa mnu nnu tau`) + model section(s) with firecrown names (`src0_delta_z = lo start hi`) | DEFAULT mode: CAMB module output read from datablock via `MappingCosmoSIS.calculate_ccl_args(sample)` -> pyccl calculator mode; PURE_CCL mode: CCL computes (ini must not have a `[camb]` section unless `allow_multiple_camb_instances`) |
| **Cobaya** | `firecrown.connector.cobaya.likelihood.LikelihoodConnector(cobaya.likelihood.Likelihood)`; options `firecrownIni` (factory path/module), `input_style: "CAMB"`, `build_parameters: dict`, `derived_parameters: list`, `distance_max_z: 4.0` | `.yaml`: `likelihood: {name: {external: !!python/name:firecrown.connector.cobaya.likelihood.LikelihoodConnector '', firecrownIni: ..., build_parameters: {...}, input_style: CAMB}}`; params are CAMB-style (`H0 ombh2 omch2 mnu nnu tau YHe As|sigma8 ns w wa`) + firecrown names (`src0_delta_z`, `lens0_bias`, ...) | DEFAULT mode: requests `Pk_grid`, `comoving_radial_distance`, `Hubble` from Cobaya's theory (CAMB) -> `MappingCAMB`; PURE_CCL mode: no `theory:` block, params passed straight (CCL names `Omega_c` ...) |
| **NumCosmo** | `firecrown.connector.numcosmo`: `NumCosmoFactory(likelihood_source, build_parameters, mapping: MappingNumCosmo|None, model_list)`, `NumCosmoData`, `NumCosmoGaussCov`, `MappingNumCosmo` | serialized experiment YAML (`numcosmo run test exp.yaml`, `numcosmo from-cosmosis x.ini --matter-ps eisenstein_hu`); generated by `firecrown examples ... --target-framework numcosmo` -> `numcosmo_{prefix}.yaml` + `.builders.yaml` | NumCosmo `Nc.HICosmo` + CLASS/EH P(k) mapped to pyccl calculator mode |

Notes:
- `docs/non-developer-mode-example/cobaya_evaluate.yaml` references `firecrown.connector.cobaya.ccl.CCLConnector`;
  that module **no longer exists** (only `cobaya/likelihood.py`). Current generator (`app/analysis/_cobaya.py`)
  emits only the `LikelihoodConnector` block with `input_style: CAMB`.
- Every connector per sample does: `likelihood.update(params); tools.update(params); tools.prepare(...);
  loglike = likelihood.compute_loglike_for_sampling(tools); likelihood.reset(); tools.reset()`.
- `firecrown.app.analysis` generates ready-to-run configs for all three frameworks from
  `(factory, build_parameters, models, CCLCosmologySpec)`; builder pattern
  `get_generator(framework, output_path, prefix, use_absolute_path, required_cosmology)`.
  Name maps: CCL->Cobaya `Omega_c->omch2 (x h^2), Omega_b->ombh2, h->H0 (x100), Neff->nnu, m_nu->mnu, w0->w, A_s->As, n_s->ns`;
  CCL->CosmoSIS `Omega_c->omega_c, Omega_b->omega_b, h->h0, sigma8->sigma_8`.

**Simplest sampler route from a Python function with no extra install:**
*none of the three frameworks is pip-installable alongside firecrown without conda*, but given a
conda env with firecrown-deps, **Cobaya in PURE_CCL_MODE** is the lightest: pure Python,
`from cobaya.run import run; run(info_dict)` (no CAMB, no CSL build, no GObject). Info dict:
```python
info = {"likelihood": {"fc": {"external": LikelihoodConnector, "firecrownIni": "/path/factory.py",
                               "build_parameters": {"sacc_file": "...", ...}}},   # no input_style, no theory
        "params": {"Omega_c": {"prior": {"min": .2, "max": .35}, "ref": .25, "proposal": .01}, "Omega_b": .05, "h": .67,
                   "n_s": .96, "sigma8": {"prior": {...}}, "Omega_k": 0., "Neff": 3.044, "m_nu": 0., "w0": -1., "wa": 0., "T_CMB": 2.7255,
                   "trc0_delta_z": {...}},
        "sampler": {"mcmc": {"max_samples": 2000}}, "output": "chains/run"}
```
(`firecrown examples cosmic_shear --target-framework cobaya` writes such a YAML; test
`tests/example/test_cosmic_shear.py` runs it with `cobaya-run --no-mpi -f cobaya_cosmic_shear.yaml`.)
CosmoSIS needs the CSL built (compilers) -> avoid on the MCP host; NumCosmo needs GObject
introspection (conda-only) and its CLI runs in a subprocess for GType safety.
For a **no-sampler** tool (loglike / theory vector / 1-D profile) nothing beyond firecrown is needed.

---

## 4. Examples & tutorials

### 4.1 `examples/` in the clone
Only `examples/cluster_number_counts/` remains (des_y1_3x2pt, cosmicshear, srd_sn were moved into
the `firecrown examples` CLI, see 4.2). Contents: factory scripts
`cluster_redshift_richness.py` (`ConstGaussian([BinnedClusterNumberCounts(average_on, "numcosmo_simulated_redshift_richness", ExactBinnedClusterRecipe(...))])`,
reads `${FIRECROWN_DIR}/examples/cluster_number_counts/cluster_redshift_richness_sacc_data.fits`),
`cluster_redshift_richness_deltasigma.py`, `cluster_redshift_richness_reduced_shear.py`,
`cluster_SDSS_redshift_richness.py`; data generators `generate_rich_mean_mass_sacc_data[_withshear].py`,
`generate_SDSS_ClusterCountsMass_sacc_data.py` (uses `cosmodc2_redmapper_data.fits`, 1.8 MB, shipped);
CosmoSIS inis `cluster_counts_redshift_richness.ini` etc. (sampler=test, modules `consistency camb firecrown_likelihood`,
`sampling_parameters_sections = firecrown_number_counts`, build params `use_cluster_counts`, `use_mean_log_mass`).
Run: `cd examples/cluster_number_counts && python generate_rich_mean_mass_sacc_data.py && cosmosis cluster_counts_redshift_richness.ini`
(requires CSL + `FIRECROWN_DIR`, `CSL_DIR`). Needs crow; theory per eval is seconds (HMF integrals).
README.md in that dir is stale (refers to `number_counts.ini`).

### 4.2 `firecrown examples` CLI (`firecrown/app/examples/`) -- the maintained examples
`firecrown examples <name> OUTPUT_DIR [--prefix P] [--target-framework cosmosis|cobaya|numcosmo]
[--sacc-format hdf5|fits] [--use-absolute-path/--no-use-absolute-path] [--cosmology-spec cosmo.yaml]`
writes: `{prefix}.sacc`, `{prefix}_factory.py` (or an experiment YAML), and
`cosmosis_{prefix}.ini + _values.ini (+ _priors.ini)` / `cobaya_{prefix}.yaml` / `numcosmo_{prefix}.yaml + .builders.yaml`
(sampler = `test`/`evaluate`, i.e. a single likelihood evaluation).
| name | data | statistic / sources | nuisance params |
|---|---|---|---|
| `cosmic_shear` | **synthetic, generated locally with pyccl** (Gaussian n(z) bins `trc{i}`, `galaxy_shear_cl_ee`, diagonal cov = (noise_level*Cl)^2); opts `--n-bins 2 --ell-min 10 --ell-max 10000 --n-ell-points 10 --noise-level 0.01 --sigma-z 0.25 --z-max 2.0 --n-z-points 600 --seed 42` | `WeakLensing(trc{i}, [PhotoZShift])` x all pairs, `ConstGaussian`; `CCLFactory(require_nonlinear_pk=True)` | `trc{i}_delta_z` |
| `cmb_cross` | synthetic: NZ tracers + `Map` tracer with `metadata={"z_lss": 1100}`; `galaxy_shear_cl_ee`, `cmbGalaxy_convergenceShear_cl_e`...; YAML factory | TwoPointExperiment with `cmb_factories` | `trc{i}_delta_z` |
| `sn_srd` | **download** `https://github.com/LSSTDESC/firecrown/releases/download/files-v1.0.0/srd-y1.sacc` (cached in `~/.cache`-style dir, see `_download.py`) | `Supernova(sacc_tracer="sn_ddf_sample")`; `CCLFactory(require_nonlinear_pk=False, amplitude_parameter=AS)` | `sn_ddf_sample_M` (prior N(-19.4, 0.05)) |
| `des_y1_3x2pt` | **download** `https://github.com/LSSTDESC/firecrown/releases/download/files-v1.1.0/des_y1_3x2pt.sacc` (real DES Y1 3x2pt, tracers `src0..3`, `lens0..4`, real-space xi+/xi-/gamma_t/w(theta), 457 data points); `--factory-type standard|pt|tatt|hmia|pk_modifier|yaml_default|yaml_pure_ccl|yaml_mu_sigma` | STANDARD: hand-written factory (4 WL sources w/ `LinearAlignmentSystematic("")`, `MultiplicativeShearBias`, `PhotoZShift`; 5 NC sources w/ `PhotoZShift`, `derived_scale=True`; 10 xip+10 xim+20 gammat+5 wtheta TwoPoints); YAML_*: `TwoPointExperiment` YAML shown in 2.6 | `ia_bias, alphaz, (alphag=1 fixed), z_piv, lens{i}_bias, lens{i}_delta_z, src{i}_delta_z, src{i}_mult_bias` (TATT: `ia_a_1, ia_a_2, ia_a_d, ia_zpiv_*, ia_alphaz_*`; HMIA: `ia_a_1h, ia_a_2h`; MU_SIGMA: `mg_musigma_*`) |
`tests/example/test_*.py` run exactly these via subprocess (`cosmosis x.ini`, `cobaya-run --no-mpi -f x.yaml`,
`numcosmo run test x.yaml`) -- markers `@pytest.mark.example`. Reference parameter values/bounds live in
`ExampleDESY13x2pt.get_models()` and `docs/non-developer-mode-example/cobaya_evaluate.yaml`
(e.g. `lens0_bias 1.4, lens1 1.6, lens2 1.6, lens3 1.9, lens4 2.0; src{0..3}_delta_z -0.001,-0.019,0.009,-0.018; mult_bias ~0.012`).
A shipped **test data vector** usable offline: `tests/sacc_data.hdf5` (1.6 MB; same DES-Y1-like layout:
NZ tracers `lens0-4`, `src0-3` with 400 z points; data `galaxy_shear_xi_plus` 167, `xi_minus` 60,
`galaxy_shearDensity_xi_t` 176, `galaxy_density_xi` 54; dense covariance 457x457). Also `tests/legacy_sacc_data.fits`,
`tests/old_format_{real,harmonic}.sacc`, `tests/bug_398.sacc.gz`.
Other CLI commands: `firecrown sacc view FILE [--plot-covariance]`, `firecrown sacc transform FILE --output-format hdf5|fits`,
`firecrown experiment view EXP.yaml`, `firecrown cosmology OUT.yaml [--cosmology vanilla_lcdm|vanilla_lcdm_with_neutrinos] [priors]`
(writes a `CCLCosmologySpec` YAML; `COSMO_DESC` defaults: Omega_c 0.25 [0.06,0.46], Omega_b 0.05 [0.03,0.07],
h 0.67, n_s 0.96, sigma8 0.81 [0.6,1.0], Omega_k 0, Neff 3.046, m_nu 0, w0 -1, wa 0, T_CMB 2.7255).

### 4.3 `tutorial/` (Quarto `.qmd`, rendered on readthedocs "Tutorial Site")
`introduction_to_firecrown.qmd` (slides: concepts, Updatable, likelihood factory), `two_point_workflow.qmd`
(roadmap), `two_point_framework.qmd`, `two_point_generators.qmd` (LSST Y1 metadata from generators),
`two_point_sacc_data.qmd` (extract from sacc), `two_point_factory_basics.qmd` (**the canonical no-sampler
loglike recipe**, below), `two_point_bin_selectors.qmd`, `two_point_scale_cuts.qmd`, `two_point_integration.qmd`
(Limber/FKEM options), `inferred_zdist_generators.qmd`, `inferred_zdist_serialization.qmd`, `tomographic_bin.qmd`,
`systematics.qmd`, `development_example.qmd`. All run in seconds (ells ~128 points, no sampling).

### 4.4 Simplest end-to-end, no sampler (from `two_point_factory_basics.qmd`, verified against source)
**(a) log-likelihood of a sacc data vector at a given cosmology + nuisance values**
```python
from firecrown.likelihood import ConstGaussian, TwoPoint, TwoPointFactory
from firecrown.likelihood.factories import load_sacc_data
from firecrown.data_functions import extract_all_real_data, check_two_point_consistence_real   # or *_harmonic_*
from firecrown.modeling_tools import ModelingTools, CCLFactory
from firecrown.updatable import get_default_params_map, ParamsMap
from firecrown.utils import base_model_from_yaml

sacc_data = load_sacc_data("tests/sacc_data.hdf5")
meas = extract_all_real_data(sacc_data); check_two_point_consistence_real(meas)
tp_factory = base_model_from_yaml(TwoPointFactory, two_point_yaml)        # yaml as in 2.6 (without data_source/ccl_factory)
stats = TwoPoint.from_measurement(meas, tp_factory)
like = ConstGaussian.create_ready(stats, sacc_data.covariance.dense)      # READY state, no read() needed
tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True, creation_mode="pure_ccl_mode"))
params = get_default_params_map(tools, like)                              # all required names w/ defaults
for k, v in {"Omega_c": 0.26, "sigma8": 0.80, "lens0_bias": 1.4, "src0_delta_z": -0.001}.items():
    params[k] = float(v)   # NOT params.update(): it raises ValueError for keys already present
tools.update(params); tools.prepare(); like.update(params)
loglike = like.compute_loglike(tools)        # float; chisq = like.compute_chisq(tools) is cached w/ theory
theory = like.get_theory_vector(); data = like.get_data_vector(); idx = like.get_sacc_indices()
like.reset(); tools.reset()                  # mandatory before the next parameter point
```
Equivalent one-liner via the YAML experiment: `lk, tools = build_two_point_likelihood(NamedParameters({"likelihood_config": "exp.yaml"}))`
then `like.read()` already done inside -> same update/prepare/compute cycle. Or any factory file:
`lk, tools = load_likelihood("factory.py", NamedParameters({"sacc_file": "..."}))`.
Required parameter names are discoverable: `list((like.required_parameters() + tools.required_parameters()).get_params_names())`.

**(b) theory data vector / synthetic sacc from theory**
- With data: after (a), `new_sacc = like.make_realization(sacc_data, add_noise=False)` returns a copy of
  the sacc with the data replaced by the theory vector (`add_noise=True` draws from the covariance);
  `new_sacc.save_hdf5(path)` / `save_fits(path)`.
- From scratch (no data, forecast): generate `TomographicBin`s (`LSST_Y1_*_HARMONIC_BIN_COLLECTION.generate()`
  or `ZDistLSSTSRD.binned_distribution(...)` or your own z/dndz arrays), `xy = make_all_photoz_bin_combinations(bins)`
  (or filtered), `cells = [TwoPointHarmonic(XY=x, ells=ells) for x in xy]`,
  `tps = TwoPoint.from_metadata(cells, tp_factory)`, then `tools.update(p); tools.prepare(); tps.update(p)`;
  `tp.compute_theory_vector(tools)` per TwoPoint (`tp.ells`, `tp.sacc_tracers`, `tp.sacc_data_type`).
  Writing to sacc is manual (firecrown has no helper): `S = sacc.Sacc(); S.add_tracer("NZ", name, z, dndz);
  S.add_ell_cl(dtype, t1, t2, ells, cl)` (or `add_theta_xi`), `S.add_covariance(cov)`, `S.save_hdf5(...)`.
  That is exactly what `firecrown/app/examples/_cosmic_shear.py` and augur `generate.py` do
  (augur also computes a Gaussian covariance analytically or via TJPCov).

---

## 5. Proposed MCP tool family `firecrown` (8-10 tools)

All tools: JSON-safe args, write to `output_dir`, return `{status, files, message, metadata}`; a
module-level cache (`get_cached`) for the loaded `(Likelihood, ModelingTools)` keyed by factory+sacc path.
Units: redshift dimensionless; `ell` dimensionless integer; `theta` in **arcmin** (sacc convention for
`theta_xi`); C_ell dimensionless; distance modulus mag; `m_nu` eV; `T_CMB` K; `h` dimensionless; `H0` km/s/Mpc
only in Cobaya/CosmoSIS translations.

1. `list_firecrown_examples()` -> table of `firecrown examples` names, what data they use (synthetic vs
   download URL), statistics, parameter names; plus shipped offline test sacc files. Pure metadata; also
   can run `ExampleCosmicShear(output_path=..., target_framework=...)` to materialise an example dir
   (`generate_firecrown_example(name, output_dir, framework="cobaya", n_bins=2, ...)`).
2. `inspect_sacc_file(path)` (sacc only, no firecrown import): tracers (type, name, n_z, z range, mean z),
   data types with counts and tracer pairs, ell/theta ranges, covariance presence/shape/condition number,
   writes `tracers.csv`, `nz_<tracer>.csv`, `datavector.csv`, optional covariance-correlation PNG.
   Uses `sacc.Sacc.load(path)` (auto-detect), `get_tracer_combinations()`, `get_ell_cl/get_theta_xi`.
3. `generate_lsst_srd_nz(year=1|10, sample="lens"|"source", n_bins=None, sigma_z=None, z_max=3.5, n_z=200)`
   -> CSV of z, dndz per bin + optional sacc with NZ tracers; wraps `ZDistLSSTSRD` / bin collections
   (needs numcosmo -> only in the firecrown env).
4. `build_firecrown_likelihood(sacc_path, correlation_space, wl_per_bin=[...], wl_global=[...], nc_per_bin=[...],
   nc_global=[...], include_rsd=False, ccl: {creation_mode, require_nonlinear_pk, amplitude_parameter, transfer_function},
   scale_cuts=[...])` -> writes `experiment.yaml` (`TwoPointExperiment`) validated with pydantic, loads it once,
   and returns the **required parameter list with defaults** (`get_default_params(tools, like)`), data-vector
   size, statistics list. This is the "spec -> factory yaml" step; systematics given by factory `type` names.
5. `compute_loglike(experiment_yaml, params: dict)` -> loglike, chi2, n_data, per-statistic chi2 contributions
   (from `like.get_sacc_indices()` + inv cov blocks), unused-parameter warnings
   (`params.get_unused_keys()`), writes `theory_vs_data.csv`. Cosmology keys are CCL names
   (`Omega_c, Omega_b, h, n_s, sigma8|A_s, Omega_k, Neff, m_nu, w0, wa, T_CMB`), nuisance keys as in 2.5.
6. `compute_theory_data_vector(experiment_yaml | {bins, ells/thetas, factory}, params, write_sacc=True, add_noise=False, seed=None)`
   -> CSV (`statistic, tracer1, tracer2, ell_or_theta, theory[, data, sigma]`) + sacc via `make_realization`
   or manual sacc assembly for the forecast (metadata-only) path; optional plot of all C_ell / xi(theta).
7. `scan_parameter(experiment_yaml, param_name, values | (min,max,n), fixed_params)` -> 1-D loglike profile
   CSV + PNG (delta chi2, 1-sigma crossing), reusing the cached likelihood with update/reset per point.
   (A 2-D grid variant is the same loop.)
8. `fisher_forecast(experiment_yaml, params, free_params, step_frac=0.01)` -> finite-difference derivative of
   the theory vector, Fisher matrix F = dT^T C^-1 dT, marginalised sigmas; CSV + optional ellipse plot.
   Cheap (n_free x 2 theory evaluations) and the natural "forecast" tool (augur does this).
9. `run_firecrown_chain(experiment_yaml, params_with_priors, sampler="cobaya_mcmc"|"cobaya_evaluate"|"cosmosis_emcee",
   n_samples, output_dir)` -- **heavy, dispatch to NERSC/ALCF**: writes the Cobaya YAML (PURE_CCL mode, no
   theory block) via `firecrown.app.analysis` generators or a minimal template and runs `cobaya-run`;
   returns chain files + GetDist summary. Requires conda env on the compute node (see 7).
10. `convert_cosmology_names(params, to="cobaya"|"cosmosis"|"ccl")` -- small helper using the connector
    NAME_MAPs (Omega_c h^2 <-> omch2, h <-> H0, sigma8 <-> sigma_8 ...), useful when mixing with the
    emulator server's conventions (`Om`, `Ob`, `As`).

Parameter-naming summary for tool docs: `{tracer}_{param}` for per-bin systematics (`src0_delta_z`,
`src0_mult_bias`, `lens0_bias`, `lens0_delta_z`, `lens0_sigma_z`, `lens0_alphaz`, `lens0_mag_bias`),
unprefixed for global systematics (`ia_bias, alphaz, z_piv`, `ia_a_1, ...`, `ia_a_1h, ia_a_2h`), cosmology
unprefixed pyccl names, Supernova `{tracer}_M`, StudentT `nu`. A ParamsMap **must contain every**
sampler parameter of `like` and `tools` (else `MissingSamplerParameterError`); extra keys are tolerated but
reported by `get_unused_keys()` (and `raise_on_unused_parameter` on the likelihood can make it fatal).

---

## 6. Gotchas

- **Install**: firecrown and numcosmo are **conda-forge only**; the core import needs `numcosmo_py`
  (GObject introspection, GLib) and `crow` even for a plain 3x2pt likelihood. Python >= 3.12 pinned
  (`match` statements, `StrEnum`); numpy `>=2.0,<2.4`; `camb<2.0`; pyccl `>=2.8` but the code uses
  pyccl 3.x APIs (`NeutrinoMassSplits`, `pyccl.nl_pt.EulerianPTCalculator`, calculator mode) -> use pyccl 3.3.x.
  Mac arm64 conda builds exist for numcosmo (nompi) so a local env is feasible.
- **Import time**: pyccl + numcosmo (GI) + pydantic model building typically several seconds; cache the
  `(Likelihood, ModelingTools)` per process and never re-import per call.
- **Updatable lifecycle**: order is `tools.update(p) -> tools.prepare() -> like.update(p) -> compute -> like.reset(); tools.reset()`.
  Calling `update()` twice without `reset()` is a silent no-op (`_updated` flag) -> stale parameters;
  `prepare()` twice raises; `GaussFamily` state machine raises if `get_theory_vector()` precedes
  `compute_theory_vector()`/`compute_loglike()`. `make_realization` needs the COMPUTED state.
- **Parameter completeness**: `ParamsMap` must include all CCLFactory cosmology params too (use
  `get_default_params_map(tools, like)` then override). Values **must be `float` or `list[float]`**:
  `validate_params_map_value` raises `TypeError` for ints/numpy scalars, so JSON ints from an MCP call
  must be cast with `float(x)` before building the `ParamsMap`. **`ParamsMap.update(d)` refuses keys that
  already exist** (`ValueError: Key ... is already present`) -- override defaults with `params[k] = v` or
  build `ParamsMap(defaults_dict | overrides)`.
- **Cosmology names**: pyccl convention `Omega_c` (not `Omega_m`), `sigma8` xor `A_s` chosen at
  `CCLFactory` construction (`amplitude_parameter`), `m_nu` in eV (sum for NORMAL split), `Neff` total,
  `T_CMB`; Cobaya/CosmoSIS translations handled only inside the connectors.
  `sigma8` with `PURE_CCL_MODE` + `boltzmann_camb` is fine; `Supernova` example uses `A_s` to avoid P(k).
- **`require_nonlinear_pk`**: in DEFAULT mode without calculator args the nonlinear P(k) comes from CCL
  (`halofit`) anyway; in calculator mode (CosmoSIS/Cobaya) `require_nonlinear_pk=True` makes firecrown
  request nonlinear P(k) from the Boltzmann code. Missing it gives the "CCL failed to compute the nonlinear
  power spectrum ... calculator mode" RuntimeError in `ModelingTools.get_pk`.
- **sacc conventions**: tracer type `NZ` (z, nz arrays), names free but firecrown's regexes
  `SOURCE_REGEX`/`LENS_REGEX` and type inference (`extract_all_measured_types`) decide whether a tracer is
  shear (`src*`) or counts (`lens*`) when a tracer appears only in ambiguous data types; mixed-type tracers
  need `allow_mixed_types=True`. Real-space `theta` in arcmin; data types from `sacc.standard_types`
  (`galaxy_shear_xi_plus`, `galaxy_shearDensity_xi_t`, `galaxy_density_cl`, ...). For gamma_t the
  (shear, lens) tracer order matters -- firecrown swaps/warns (`_sacc_convention_warning`). Covariance
  must be dense and present for `extract_all_*_data` and `GaussFamily.read`. HDF5 (`save_hdf5`) is the
  current default format; FITS still readable (`load_sacc_data` auto-detects; `sacc.Sacc.load`).
- **Windows / bandpowers**: harmonic data with `BandpowerWindow` are handled (`normalize_window`), the
  theory is computed on `window_ells` and convolved; `TwoPointHarmonic.window` shape checks are strict.
- **Interpolation**: real-space xi uses C_ell on `LogLinearElls` up to 60000 then `pyccl.correlation`;
  per-evaluation cost for the DES Y1 3x2pt set is O(seconds) (45 TwoPoints), harmonic LSST Y1 similar.
- **Deprecated modules still importable** (emit `DeprecationWarning`): `firecrown.likelihood.two_point`,
  `firecrown.likelihood.gaussian`, `firecrown.parameters`, `ModelingTools(cluster_abundance=...)`,
  `TwoPoint.from_metadata_index`, `InferredGalaxyZDist` alias. Prefer `firecrown.likelihood` top-level exports.
- `firecrown examples des_y1_3x2pt` / `sn_srd` **download** data from GitHub releases (network, cached);
  `cosmic_shear`/`cmb_cross` are fully synthetic and offline.
- CosmoSIS connector needs `FIRECROWN_DIR`/`CSL_DIR`-style paths and the CSL compiled; `tau` must be set
  in values (CAMB), and `sampling_parameters_sections` must list the sections holding firecrown params.
- Thread safety: pyccl and `Updatable` objects are not thread-safe; serialise tool calls or use one
  likelihood instance per worker process.

---

## 7. Pip/conda deps for an HPC node

There is **no pure-pip path** (firecrown, numcosmo absent from PyPI). Options, in order of preference:

1. **Use the DESC-maintained stack already on NERSC**: `source /global/common/software/lsst/common/miniconda/setup_current_python.sh`
   then `conda activate desc-python` (lock 2026-05-28 for `hpc-linux-64`: python 3.12.13, **firecrown 1.14.3**,
   numcosmo 0.26.0, pyccl 3.3.4, sacc 2.1.2, cosmosis 3.25.2, cobaya) -- see `external/desc-python`.
   Also `source $CFS/lsst/groups/MCP/setup-cosmology.sh` -> `desc-cosmology` env (firecrown 1.14.3, numcosmo 0.27.0,
   `module load PrgEnv-gnu cpu cray-mpich-abi`; sets `COSMOSIS_NO_SUBPROCESS=1`) -- see `external/desc-cosmology-env`.
   Caveat: 1.14.3 predates some names used here (e.g. `TomographicBin`, `extract_all_tracers_tomographic_bins`
   are 1.15/1.16 renames; 1.14 has `InferredGalaxyZDist`/`extract_all_tracers_inferred_galaxy_zdists`). Pin tool code to
   the installed version or vendor the 1.16 clone with `pip install --no-deps` on top (deps are satisfied by the stack).
2. **Self-contained micromamba env on the node** (works on NERSC/ALCF, ~5 min, linux-64 nompi builds):
   ```bash
   micromamba create -y -p $SCRATCH/fc-env -c conda-forge --override-channels \
       python=3.12 "pyccl>=3.3" "sacc>=2.4" "numcosmo>=0.27" lsstdesc-crow "camb<2.0" \
       "numpy>=2.0,<2.4" scipy astropy fitsio pydantic pyyaml rich typer typing_extensions matplotlib-base
   $SCRATCH/fc-env/bin/pip install --no-deps /path/to/firecrown_clone      # or: micromamba install firecrown  (pulls firecrown-deps incl. cosmosis)
   # samplers (optional): micromamba install cobaya  (pure python; add `getdist` for post-processing)
   ```
   pip-only extras that *are* on PyPI if one starts from a conda base that already has numcosmo+pyccl:
   `sacc>=2.4 pydantic pyyaml rich typer typing_extensions lsstdesc-crow clmm cobaya getdist`.
3. Container: `lsstdesc/desc-cosmology:slac-latest-2026-07-23` (docker/apptainer; `source /opt/desc/bin/activate`).

Local (this Mac) smoke test: see the "Verification" section below.

---

## 8. Verification (scratch env on this Mac, 2026-10-07)

Built a throwaway env in the session scratchpad (not under `anaconda3/envs`, the clone was not modified;
`git status` clean):
```bash
conda create -p <scratch>/fc-env -c conda-forge --override-channels python=3.12 "pyccl>=3.3" "sacc>=2.4" \
    "numcosmo>=0.27" pydantic pyyaml rich typer "camb<2.0" "numpy>=2.0,<2.4" scipy astropy fitsio typing_extensions lsstdesc-crow
# classic conda solver: ~6 min solve+install on osx-arm64 (micromamba would be ~1-2 min)
cp -R firecrown <scratch>/fc-src && fc-env/bin/pip install --no-deps --no-build-isolation <scratch>/fc-src
```
Result: python 3.12.15, pyccl 3.3.6, sacc 2.4, camb 1.6.5, numcosmo 0.28, crow present; `firecrown.__version__`
reports `0.0.0` because `--no-build-isolation` skips setuptools-scm (use normal isolation or
`SETUPTOOLS_SCM_PRETEND_VERSION=1.16.0` for a correct version string).

`scratchpad/smoke_firecrown.py` (the recipe in 4.4, real-space DES-Y1-like `tests/sacc_data.hdf5`,
`TwoPointFactory` with mult-bias + photo-z + linear IA, PURE_CCL_MODE, `require_nonlinear_pk=True`):
- import numpy+pyccl+sacc 0.97 s; `import firecrown.likelihood` (+ numcosmo, crow, pydantic) **0.56-1.0 s** (warm cache).
- `extract_all_real_data` -> 45 TwoPoint measurements, 457 data points, 9 NZ tracers (`lens0-4`, `src0-3`).
- `get_default_params_map(tools, like)` -> 32 required parameters: `Neff Omega_b Omega_c Omega_k T_CMB h m_nu n_s
  sigma8 w0 wa` + `ia_bias alphaz z_piv` + `lens{0-4}_bias lens{0-4}_delta_z src{0-3}_delta_z src{0-3}_mult_bias`.
- One full 3x2pt evaluation (pyccl cosmology + 45 real-space correlations): **0.41 s first, 0.27 s repeated**
  (`loglike = -270.04`, chi2 = 540.1 at defaults with DES-like biases; deterministic between calls).
- `like.make_realization(sacc_data, add_noise=False).save_hdf5(...)` wrote a theory-vector sacc; CSV of
  `(sacc_index, data, theory)` written.
- Theory-only harmonic path from generators (`LSST_Y1_*_HARMONIC_BIN_COLLECTION.generate()`, 1 lens + 1 source bin,
  57 ells): 3 C_ell spectra in **0.19 s** (`galaxy_shear_cl_ee`, `galaxy_shearDensity_cl_e`, `galaxy_density_cl`).
Only `DeprecationWarning`s were emitted (none from the recipe itself). The scratch env lives in the session
scratchpad and can be deleted; to make it permanent, re-create it under `anaconda3/envs` (e.g. `desc-firecrown`)
with the same spec plus `cobaya getdist matplotlib` for the chain/plot tools.
