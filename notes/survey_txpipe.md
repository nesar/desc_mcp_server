# TXPipe survey (for the DESC MCP server)

Surveyed: `/Users/nesar/Projects/Tutorials/desc-mcp-server/TXPipe` @ `efad606` (merge of PR #510,
2026-10-06). ceci cloned for reference to `/Users/nesar/Projects/Tutorials/desc-mcp-server/external/ceci`
(HEAD 2026-10-06, `lsstdesc-ceci` 2.4.x pinned by TXPipe). TXPipe clone was NOT modified.
Note: TXPipe's `submodules/WLMassMap` and `submodules/pyfsb` are empty (clone was not `--recurse-submodules`);
`TXConvergenceMaps` (needs `WLMassMap`) and `HOSFSB` (needs `pyfsb`) will fail until
`git submodule update --init` is run.

---

## 1. What TXPipe is & how it is installed

TXPipe is the LSST DESC **3x2pt pipeline**: it ingests shear + photometry catalogs, selects/tomographically
bins source and lens samples, estimates n(z), makes maps/masks/randoms, measures real-space (TreeCorr) and
Fourier-space (NaMaster) two-point functions, computes Gaussian covariances (TJPCov / NaMaster), runs
null/PSF/systematics tests, blinds, and writes **SACC** files + PNG plots. Every stage is a Python class
derived from `ceci.PipelineStage` (TXPipe wraps it in `txpipe/base_stage.py::PipelineStage`, which adds
provenance, `memory_report`, `combined_iterators`, and MPI-aware `open_output`). The pipeline engine is
**ceci** (`ceci <pipeline.yml>`; launchers `mini` / `parsl` / `cwl`).

### Install paths (all from the repo)

| Target | Mechanism | Notes |
|---|---|---|
| Laptop (Linux/mac) | `./bin/install.sh` -> downloads Miniforge into `./conda`, `mamba env update -f bin/environment-local.yml` | README still says "M1 macs not supported", but CI runs `macos-14` (Apple Silicon) successfully. Activate: `source ./conda/bin/activate`. Adds `KMP_DUPLICATE_LIB_OK=TRUE` workaround for healpy/libomp. |
| NERSC Perlmutter | `./bin/install.sh` detects `NERSC_HOST=perlmutter` -> `bin/perlmutter-install.sh`: `module load python; module load cray-mpich; mamba env create -f bin/environment-perlmutter.yml -p ./conda`; firecrown v1.7.5 installed from git with `numcosmo` stripped; mpi4py rebuilt against cray-mpich (`MPI4PY_BUILD_MPIABI=1 MPICC="mpicc -shared" pip install --no-binary=mpi4py mpi4py`). Activation hook sets `MPI4PY_RC_RECV_MPROBE=False`, `HDF5_USE_FILE_LOCKING=FALSE`, `MPICH_GPU_SUPPORT_ENABLED=0`. Activate: `module load python; conda activate ./conda`. |
| NERSC container | `shifter --image=joezuntz/txpipe ceci examples/metadetect/pipeline.yml` (`.gitlab-ci.yml`); `examples/2.2i/pipeline.yml` uses `site: {image: ghcr.io/lsstdesc/txpipe, volume: ${PWD}:/opt/txpipe}`. The docs call these "deprecated/legacy" in favour of the conda install. |
| CC-IN2P3 | `bin/environment-in2p3.yml`, `bin/pip-in2p3.sh`, `docs/src/installation/ccin2p3.rst` |
| Jupyter kernel at NERSC | `bin/setup-jupyter-kernel.sh <name> "<Display Name>"` writes a kernel.json that `module load python; conda activate` before launching ipykernel. |

### Python version & heavy dependencies (`bin/environment-local.yml`, pinned)

- **Python 3.12** locally, **3.13** on Perlmutter (`environment-perlmutter.yml`). `requirements.txt`/`conda.txt`
  are the older unpinned lists; the `environment-*.yml` files are authoritative.
- Core: `numpy=2.3`, `scipy=1.16`, `h5py=3.15 (mpi_mpich build)`, `mpi4py=4.1`, `mpich=4.3`, `astropy=7.1`,
  `pandas=2.3`, `dask=2024.5` + `dask-mpi`, `psutil`, `matplotlib=3.10`, `scikit-learn=1.7`, `jax/jaxlib=0.7`.
- Cosmology/2pt: `treecorr=5.1`, `namaster=2.5` (pymaster), `healpy=1.18`, `healsparse=1.15`, `skyproj=2.5`,
  `pyccl=3.2`, `camb=1.6`, `cosmosis=3.24`, `firecrown=1.7`, `tjpcov=0.4`, `sacc[all]==2.2`, `qp-prob=1.0`,
  `tables-io-full`, `photerr`, `dsigma==1.2` (delta-sigma), `glass==2023.7` (lognormal sims), `CLMM 1.14.6`,
  `hybrideb`, `hyperbolic`, `parallel-statistics==0.13`, `mockmpi`, `pygraphviz` (flow charts), `somoclu`.
- Photo-z: `pz-rail[algos] @ RAIL v1.1.2` + `pzflow>=4` (RAIL stages BPZliteInformer/Estimator, NZDirInformer,
  FlowCreator, GridSelection are run as ceci stages inside TXPipe pipelines).
- Framework: `lsstdesc-ceci=2.4.*` (deps: pyyaml, ruamel.yaml, psutil, networkx, jinja2; `parsl` optional).
- Perlmutter-only extras: `lsstdesc-gcr-catalogs`, `lsstdesc-dc2-dm-data`, `psycopg2`, `pyarrow`, `pytables`,
  `dataregistry` (DESC data registry lookups in pipeline `inputs`).
- The env is ~60 conda packages + ~12 pip/git packages; expect a multi-GB environment and 10-30 min solve/install.
  `pip install txpipe` is not a thing - TXPipe is run from the repo checkout (`python -m txpipe`, `sys.path`
  includes cwd via ceci `main.py`).

### Local conda environments checked (none usable)

`conda env list`: cosmic-emu (py3.12), llm_env (py3.10), hep-genesis (py3.12), plus others. In all three:
`ceci`, `txpipe`, `sacc`, `treecorr`, `healpy`, `pymaster`, `mpi4py`, `parsl`, `qp`, `rail`, `tjpcov`, `healsparse`
are **MISSING**. Only `pyccl`, `camb`, `h5py` exist in cosmic-emu/llm_env. A dedicated `txpipe` env
(via `./bin/install.sh` or `mamba env create -f bin/environment-local.yml`) is required to run anything.
`hep-genesis` env does have the `hep_genesis` dispatch backend (`run_codes_on_perlmutter`, `run_on_perlmutter`).

---

## 2. Pipeline structure

### Stage anatomy (introspectable)

Every stage class declares class attributes we can read without running anything (`txpipe/base_stage.py`,
`docs/src/structure.rst`):

- `name` (string used in YAML and CLI; note e.g. `TXSourceSelectorBase.name == "TXSourceSelector"`,
  `TXTwoPointSelfCalibrationIA.name == "TXTwoPointSCIA"`)
- `inputs = [(tag, FileTypeClass), ...]`, `outputs = [(tag, FileTypeClass), ...]`
- `config_options = {key: StageParameter(dtype, default, required=..., msg=...)}` (newer stages) or
  `{key: default_value_or_type}` (older/extension stages; a bare type means required)
- `parallel = True/False` (MPI capable) and `dask_parallel = True` (dask-based mapping stages)
- ceci provides `PipelineStage.get_stage(name)`, `cls.pipeline_stages` (registry of all imported stages),
  `cls.describe_configuration()` / `_describe_configuration_text()`, `find_outputs(outdir)` and
  `generate_command(inputs, config, outputs, aliases, instance_name)`.

Output filenames are `{output_dir}/{tag}.{suffix}` (`FileType.make_name`), written as `inprogress_{tag}.{suffix}`
and renamed on success, which is how `resume` detects completed stages.

File types (`txpipe/data_types.py`): `HDFFile`, `PhotometryCatalog`, `ShearCatalog`, `TomographyCatalog`,
`BinnedCatalog`, `RandomsCatalog`, `MapsFile` (+`LensingNoiseMaps`, `ClusteringNoiseMaps`), `PhotozPDFFile`,
`QPPDFFile`, `QPNOfZFile`, `QPMultiFile`, `SACCFile` (suffix `.sacc`, FITS inside), `CSVFile`, `FiducialCosmology`
(YAML), plus ceci's `TextFile`, `YamlFile`, `PNGFile`, `PickleFile`, `ParquetFile`, `FileCollection`, `Directory`.

### Stage inventory (161 classes found by AST scan of `txpipe/`, incl. bases)

Grouped by purpose (names are class `name`s usable in YAML):

- **Ingest / mocks** (`txpipe/ingest/`, `simulation.py`): `TXCosmoDC2Mock`, `TXBuzzardMock`, `TXGaussianSimsMock`,
  `TXSimpleMock` (ASCII -> shear catalog), `TXMockTruthPZ`, `TXIngestRedmagic`, `TXMetacalGCRInput`, `TXIngestStars`,
  `TXExposureInfo`, `TXIngestDataPreview02`, `TXIngestDataPreview1` (DP1 via LSST butler; NERSC only),
  `TXIngestRubinMetaDetect`, `TXIngestDESI`, `TXIngestDESIRandoms`, `TXIngestFlagshipMocks/Masks`,
  `TXIngestDESY3Gold/Footprint/SpeczCat/Shear/SourceRedshift`, `TXIngestCatalogFits/H5`, `TXIngestMapsHsp`,
  SSI: `TXIngestSSIGCR`, `TXMatchSSI`, `TXIngestSSI*DESBalrog`; `TXLogNormalGlass` (GLASS lognormal sims).
- **Photo-z / n(z)**: `TXPhotozStack`, `TXTruePhotozStack`, `TXMockGaussianPhotozStack`, `TXPhotozPlot`,
  `PZRailSummarize`, `PZRailPZSummarize`, `PZRealizationsPlot`, `TXParqetToHDF`; plus RAIL stages imported via
  `modules:` (`BPZliteInformer`, `BPZliteEstimator`, `NZDirInformer`, `FlowCreator`, `GridSelection`).
- **Selection / tomography**: `TXSourceSelector` (base), `TXSourceSelectorMetacal`, `TXSourceSelectorMetadetect`,
  `TXSourceSelectorMetadetectDP2`, `TXSourceSelectorLensfit`, `TXSourceSelectorHSC`, `TXSourceSelectorSimple`,
  `TXSourceSelectorDESY3`, `TXSourceTomography` (random-forest classifier); lens: `TXTruthLensSelector`,
  `TXMeanLensSelector`, `TXModeLensSelector`, `TXCustomLensSelector`, `TXRandomForestLensSelector`,
  `TXDESISelector`, `TXDESIMockSelector`, `TXDESIMultiMockSelector`.
- **Calibration / splitting**: `TXShearCalibration`, `TXLensCatalogSplitter`, `TXTruthLensCatalogSplitter`,
  `TXTruthLensCatalogSplitterWeighted`, `TXExternalLensCatalogSplitter`, `TXStarCatalogSplitter`,
  `TXCutCatalog`, `TXCutShearCatalog`.
- **LSS weights**: `TXLSSWeightsUnit`, `TXLSSWeightsLinBinned`, `TXLSSWeightsLinPix`, `TXLSSDensityNullTests`.
- **Maps / masks / randoms**: `TXSourceMaps`, `TXLensMaps`, `TXDensityMaps`, `TXPSFMaps`, `TXFlagMaps`,
  `TXDepthMaps`, `TXBrightObjectMaps`, `TXUniformDepthMap`, `TXAuxiliarySSIMaps`, `TXPredictDensitySelectionFunction`,
  `TXSourceNoiseMaps`, `TXLensNoiseMaps`, `TXExternalLensNoiseMaps`, `TXNoiseMapsJax`, `TXSimpleMask`,
  `TXSimpleMaskSource`, `TXSimpleMaskFrac`, `TXCustomMask`, `TXJointMask`, `TXDESIJointMask`, `TXRandomCat`,
  `TXSubsampleRandoms`, `TXJackknifeCenters`, `TXJackknifeCentersSource`, `TXConvergenceMaps`, `TXTracerMetadata`.
- **Two-point**: `TXTwoPoint` (TreeCorr real space), `TXTwoPointPixel`, `TXTwoPointPixelExtCross`,
  `TXTwoPointFourier` (NaMaster), `TXTwoPointFourierCatalog`, `TXTwoPointTheoryReal`, `TXTwoPointTheoryFourier`,
  `TXDeltaSigma`, `TXDeltaSigmaTheory`.
- **Covariance**: `TXFourierGaussianCovariance`, `TXRealGaussianCovariance` (TJPCov, f_sky only),
  `TXFourierTJPCovariance` (mask geometry, reuses NaMaster workspaces), `TXFourierNamasterCovariance`,
  `TXRealNamasterCovariance`.
- **Blinding**: `TXBlinding` (Muir et al.; uses firecrown theory shift), `TXNullBlinding`.
- **Null tests / PSF diagnostics**: `TXGammaTFieldCenters`, `TXGammaTStars`, `TXGammaTRandoms`, `TXApertureMass`,
  `TXRoweStatistics`, `TXTauStatistics`, `TXPSFDiagnostics`, `TXPSFMomentCorr`, `TXGalaxyStarShear`,
  `TXGalaxyStarDensity`, `TXBrighterFatterPlot`, `TXFocalPlanePlot`, `TXMapCorrelations`.
- **Plots / diagnostics**: `TXDiagnosticQuantiles`, `TXSourceDiagnosticPlots`, `TXLensDiagnosticPlots`,
  `TXMapPlots`, `TXMapPlotsSSI`, `TXConvergenceMapPlots`, `TXTwoPointPlots`, `TXTwoPointPlotsFourier`,
  `TXTwoPointPlotsTheory`, `TXDeltaSigmaPlots`, `TXPhotozPlot`, `PZRealizationsPlot`.
- **Extensions** (`txpipe/extensions/`, must be listed in `modules:` - not auto-imported): cluster counts
  `CLIngestRedmapper`, `CLClusterBinningRedshiftRichness`, `CLClusterShearCatalogs`, `CLClusterEnsembleProfiles`,
  `CLClusterSACC`, `TXTwoPointRLens`; CMB lensing `TXIngestQuaia`, `TXIngestPlanckLensingMaps`,
  `TXTwoPointFourierCMBLensingCrossDensity`, `TXCMBLensingCrossMonteCarloCorrection`; `TXTwoPointCluster`;
  IA self-calibration `TXTwoPointSCIA`, `TXTwoPointSourcePixel`, `TXTwoPointSCIAArc`; HOS `HOSFSB`.

### Key stages: inputs / outputs / config (quoted from class attributes)

- `TXSourceSelectorMetadetect` (base `TXSourceSelector`): in `shear_catalog: ShearCatalog`,
  `shear_tomography_classifier: PickleFile`; out `shear_tomography_catalog: TomographyCatalog`; config
  `T_cut` (req), `s2n_cut` (req), `source_zbin_edges` (req), `delta_gamma` (req), `bands=[r,i,z]`, `input_pz`,
  `true_z`, `do_tomography`, `chunk_rows=10000`.
- `TXSourceTomography`: in `spectroscopic_catalog`; out `shear_tomography_classifier`; config `bands`,
  `source_zbin_edges` (req), `spec_mag_column_format`, `spec_redshift_column`, `random_seed`.
- `TXMeanLensSelector` (base `TXBaseLensSelector`): in `photometry_catalog`, `lens_photoz_pdfs`; out
  `lens_tomography_catalog_unweighted`; config `lens_zbin_edges` (req), BOSS cuts `cperp_cut, r_cpar_cut,
  r_lo_cut, r_hi_cut, i_lo_cut, i_hi_cut, r_i_cut`, `selection_type='boss'`, `maglim_band/limit`.
- `TXShearCalibration`: in `shear_catalog`, `shear_tomography_catalog`, `fiducial_cosmology`; out
  `binned_shear_catalog`; config `use_true_shear`, `subtract_mean_shear=True`, `copy_redshift`,
  `add_fiducial_distance`, `redshift_name`, `shear_catalog_type`.
- `TXLensCatalogSplitter`: in `lens_tomography_catalog_unweighted`, `photometry_catalog`, `fiducial_cosmology`,
  `lens_photoz_pdfs`; out `binned_lens_catalog_unweighted`; config `redshift_column='zmean'`.
- `TXLSSWeightsUnit`: in `binned_lens_catalog_unweighted`, `lens_tomography_catalog_unweighted`, `mask`;
  out `lss_weight_summary`, `weighted_density_correlation`, `lss_weight_maps`, `binned_lens_catalog`,
  `lens_tomography_catalog`.
- `TXPhotozStack`: in `photoz_pdfs: QPPDFFile`, `tomography_catalog`, `weights_catalog`; out
  `photoz_stack: QPNOfZFile`. `TXTruePhotozStack`: in `tomography_catalog`, `catalog`, `weights_catalog`
  (used via aliases for source/lens). `PZRailSummarize`: in `binned_catalog`, `model`; config
  `summarizer='NZDirSummarizer'`, `module='rail.estimation.algos.nz_dir'`, `catalog_group`, `tomography_name`, `bands`.
- `TXSourceMaps` (dask): in `binned_shear_catalog`; out `source_maps`. `TXLensMaps` (dask): in
  `binned_lens_catalog`; out `lens_maps`. `TXDensityMaps`: in `lens_maps`, `mask`; out `density_maps`.
  Map stages share `map_config_options` (`nside`, `pixelization`, `sparse`, ... from `global:`).
- `TXSimpleMaskFrac`: in `depth_map`, `bright_object_map`; out `mask`; config `depth_cut=23.5`,
  `bright_object_max=10`, `supreme_map_file`, `frac_cut`.
- `TXRandomCat`: in `depth_map`, `mask`, `lens_photoz_stack`, `fiducial_cosmology`; out `random_cats`,
  `binned_random_catalog`, `binned_random_catalog_sub`; config `density=100` (per sq arcmin), `Mstar`, `alpha`,
  `method='quadrilateral'`, `sample_rate=0.5`.
- `TXJackknifeCenters`: in `random_cats`; out `patch_centers: TextFile`, `jk: PNGFile`; config `npatch=10`, `every_nth=100`.
- `TXTracerMetadata`: in `shear_catalog`, `shear_tomography_catalog`, `lens_tomography_catalog`, `mask`;
  out `tracer_metadata: HDFFile`, `tracer_metadata_yml: YamlFile`.
- `TXTwoPoint`: in `binned_shear_catalog`, `binned_lens_catalog`, `binned_random_catalog`,
  `binned_random_catalog_sub`, `shear_photoz_stack`, `lens_photoz_stack`, `patch_centers`, `tracer_metadata`;
  out `twopoint_data_real_raw: SACCFile`, `twopoint_gamma_x: SACCFile`; config = `TREECORR_CONFIG`
  (`min_sep=0.5`, `max_sep=300`, `nbins=9`, `bin_slop=0.0`, `sep_units='arcmin'`, `flip_g1`, `flip_g2=True`,
  `cores_per_task=20`, `verbose`) + `calcs=[0,1,2]`, `source_bins`, `lens_bins`, `do_shear_shear/pos/pos_pos`,
  `auto_only`, `var_method='jackknife'`, `use_randoms`, `low_mem`, `patch_dir='./cache/patches'`, `metric`,
  `use_subsampled_randoms`.
- `TXTwoPointFourier`: in `shear_photoz_stack`, `lens_photoz_stack`, `fiducial_cosmology`, `tracer_metadata`,
  `lens_maps`, `source_maps`, `density_maps`, `mask`, `source_noise_maps`, `lens_noise_maps`; out
  `twopoint_data_fourier: SACCFile`; config `ell_min=100`, `ell_max=1500`, `n_ell=20`, `ell_spacing='log'`,
  `cache_dir='./cache/twopoint_fourier'`, `flip_g2`, `analytic_noise`, `b0`, `compute_theory=True`, `do_*`.
- `TXBlinding`: in `twopoint_data_real_raw`; out `twopoint_data_real`; config `seed=1972`, `Omega_b`,
  `Omega_c`, `w0`, `h`, `sigma8`, `n_s` as `[fiducial, sigma]`, `b0`, `delete_unblinded`.
  `TXNullBlinding`: same tags, no-op copy (use for sims).
- `TXTwoPointTheoryReal`: in `twopoint_data_real`, `fiducial_cosmology`; out `twopoint_theory_real`; config
  `galaxy_bias=[0.0]`, `smooth`. Fourier variant: `twopoint_data_fourier` -> `twopoint_theory_fourier`.
- `TXRealGaussianCovariance`: in `fiducial_cosmology`, `twopoint_data_real`, `tracer_metadata`; out
  `summary_statistics_real: SACCFile`; config `min_sep`, `max_sep`, `nbins`, `pickled_wigner_transform`,
  `galaxy_bias`. `TXFourierTJPCovariance`: in `fiducial_cosmology`, `twopoint_data_fourier`,
  `tracer_metadata_yml`, `mask`, `density_maps`, `source_maps`; out `summary_statistics_fourier`; config
  `cov_type=['FourierGaussianNmt','FourierSSCHaloModel']`, `IA=0.5`, `cache_dir`.
- `TXTwoPointPlots`: in `twopoint_data_real`, `twopoint_gamma_x`; out PNGs `shear_xi_plus`, `shear_xi_minus`,
  `shearDensity_xi`, `density_xi`, `shearDensity_xi_x`.
- `TXSourceDiagnosticPlots`: 14 outputs (PNG histograms + text), config `g_min/max`, `psfT_min/max`,
  `T_min/max`, `s2n_min/max`, `nbins`, `bands`.
- `TXMapPlots`: in 7 map files; out 6 PNGs; config `projection='McBryde'`.
- `TXIngestDataPreview1`: no inputs; out `photometry_catalog`, `shear_catalog`, `exposures`,
  `survey_property_maps`; config `butler_config_file='/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml'`,
  `collections='LSSTComCam/DP1'`, `cosmology_tracts_only`, `select_field` (NERSC + LSST stack only).
- `TXSimpleMock`: in `mock_shear_catalog: TextFile`; out `shear_catalog` (the laptop-friendly mock entry point).

### How ceci runs a pipeline

Two YAML files. **Pipeline file** keys (quoted from `examples/metadetect/pipeline.yml`; semantics in
`docs/src/running.rst`, `ceci/pipeline/pipeline.py`):

```yaml
stages:                       # ordered list; ceci builds the DAG from input/output tags
  - name: TXTwoPoint
    threads_per_process: 2    # OMP_NUM_THREADS / NUMBA_NUM_THREADS
    nprocess: 2               # MPI ranks (-> "mpirun -n 2 ... --mpi" locally; "srun -u -n 2" at NERSC)
    nodes: 1                  # NERSC only
  - name: PZEstimatorLens     # instance name; class chosen by classname
    classname: BPZliteEstimator
    aliases: {model: lens_photoz_model, input: photometry_catalog, output: lens_photoz_pdfs}
modules: >                    # python modules to import to find stages
    txpipe
    txpipe.extensions
    rail.estimation.algos.bpz_lite
python_paths:                 # prepended to PYTHONPATH
  - submodules/WLMassMap/python/desc/
output_dir: data/example/outputs_metadetect
launcher: {name: mini, interval: 1.0}        # mini | parsl | cwl
site: {name: local, max_threads: 2}          # local | nersc-interactive | nersc-batch | cori-interactive | cori-batch | cc-parallel
config: examples/metadetect/config.yml       # per-stage options file
inputs:                                      # overall inputs: tag -> path (or {name:/id:} for data registry)
    shear_catalog: data/example/inputs/metadetect_shear_catalog.hdf5
    fiducial_cosmology: data/fiducial_cosmology.yml
    binned_random_catalog_source: None
resume: true                                 # true|resume -> skip stages whose outputs exist; false|restart -> rerun all; refuse -> error
log_dir: data/example/logs_metadetect        # one "<StageInstanceName>.out" per stage (stdout+stderr)
pipeline_log: data/example/log.txt
registry: {}                                 # optional DESC data-registry config (NERSC)
```

Site options (`ceci/sites/*.py`, `running.rst`): all sites accept `image` (docker locally / shifter at NERSC) and
`volume`; `local`/`nersc-interactive` accept `max_threads` (and `max_processes`); `nersc-batch` (parsl) accepts
`mpi_command` (default `srun -u -n`), `queue: debug`, `max_jobs: 2`, `account: m1727`, `walltime: 00:30:00`,
`cpu_type`, `setup`. The NERSC site refuses `nprocess > 1` outside a SLURM job (`SLURM_JOB_ID` unset) unless dry-run.
`mini` launcher reads `SLURM_JOB_NODELIST`/`SLURM_CPUS_ON_NODE` to build its node list (256 -> 128 procs/node on
Perlmutter CPU). Pipeline YAML is Jinja2-templated (`-t key=value`, e.g. `examples/dp1/pipeline.yml` uses `{{ suffix }}`).

**Config file** (`examples/metadetect/config.yml`): a `global:` section inherited by all stages (`chunk_rows: 100000`,
`pixelization: healpix`, `sparse: true`, `nside`) and one section per stage **instance name** (e.g. `TXTwoPoint:`,
`PZRailSummarizeLens:`), resolved in `ceci.PipelineStage.read_config` as defaults < global < stage section < CLI flags.
YAML anchors are used for nsides (`_nsides: {nside_high: &nside_high 256, ...}`).

**CLI** (`ceci/main.py`): `ceci pipeline.yml [--dry-run] [--flow-chart out.png] [-t k=v ...] [key=value overrides]`.
Overrides use dotted paths: `ceci examples/metadetect/pipeline.yml site.name=nersc-interactive resume=restart
inputs.shear_catalog=/path/file.hdf5` (both forms appear in `bin/nersc-test.sub` and `.github/workflows/ci.yml`).
`--dry-run` prints the exact per-stage command, e.g.
`OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONPATH=... mpirun -n 2 python3 -m txpipe TXTwoPoint --binned_shear_catalog=... --config=examples/metadetect/config.yml --twopoint_data_real_raw=... --mpi`.
Single stage by hand (`docs/src/structure.rst`): `python -m txpipe <StageName> --<input_tag>=path ... --config=config.yml --<output_tag>=path [--name=InstanceName] [--mpi] [--pdb] [--<option>=value]`.
Python API (`notebooks/Welcome to TXPipe.ipynb`): `ceci.Pipeline.read(yml)`; `cfg = ceci.Pipeline.build_config(yml);
cfg['stages'].append({...}); ceci.Pipeline.create(cfg).run()`; `pipeline.overall_inputs`, `pipeline.pipeline_outputs`.

---

## 3. Examples, test data, notebooks

### `examples/` (29 directories, 70+ pipeline/config YAMLs; stage counts in parentheses)

| Dir | Pipelines | Site | Data | Notes |
|---|---|---|---|---|
| `metadetect/` | `pipeline.yml` (61 stages) | local, max_threads 2 | `data/example/inputs/*` | **The canonical laptop/CI test** (README, docs, CI on ubuntu + macos-14). Includes RAIL BPZ photo-z + NZDir, maps, real+Fourier 2pt, blinding, theory, null tests, delta-sigma, SCIA, FSB. Covariance stages commented out as too slow. |
| `metadetect_source_only/` | (28) | local | example | Shear-only subset; CI sets `TX_DASK_DEBUG=1`. |
| `metacal/` | (52) | local | example | Metacal variant (`TXSourceSelectorMetacal`). CI. |
| `lensfit/` | (43) | local | `lensfit_*` example files | KiDS-style. CI. |
| `redmagic/` | (38) | local | example + `cosmoDC2_v1.1.4_run_redmagic_highdens.fit` | CI. |
| `mock_shear/` | (6) | local | `mock_nfw_shear_catalog.txt`, `mock_single_cluster_catalog.hdf5`, `mock_spectroscopic_catalog.hdf5` | **Smallest pipeline**: `TXSimpleMock -> TXSourceSelectorSimple -> TXSourceTomography -> TXShearCalibration -> TXMockTruthPZ -> CLClusterShearCatalogs`. CI checks `binned_shear_catalog.hdf5`. |
| `cmb/` | `quaia.yml` (3) | local, 4 threads | `example-cmb-lensing.tar.gz` (189 MB) | Quaia x Planck lensing C_ell. CI. |
| `cosmodc2/` | `pipeline.yml` (45), `pipeline-20deg2-laptop.yml` (38), `pipeline-20deg2-nersc.yml` (38), `pipeline_redmagic*.yml`, `pipeline-pz.yml`, `ingest-small-pipeline.yml`, `Cluster_pipelines/*` (CL cluster pipelines for laptop/nersc/in2p3 + `.sub` scripts) | local / `cori-interactive` / `nersc-interactive` | `cosmodc2-20deg2.tar.gz` (**13.6 GB**) or DESC data registry (`inputs: shear_catalog: {name: txpipe_cosmodc2_20deg2_shear_catalog.hdf5}`) | The "mini" 20 deg^2 dataset; laptop version uses `nprocess: 12`. |
| `2.2i/` | `pipeline.yml` (37), `pipeline_1tract.yml`, `cori-2.2i.sub` | `cori-interactive` + shifter `ghcr.io/lsstdesc/txpipe` | NERSC paths under `/global/projecta/...` | DC2 Run2.2i DR6; 2 nodes, 9 h sbatch. |
| `dp1/`, `dp0.2/` | `ingest.yml` (DP1 butler ingest, NERSC+LSST stack), `pipeline.yml`, `pipeline-edfs.yml`, `pipeline_ww.yml` | local | NERSC | CI only does `ceci --dry-run examples/dp1/pipeline-edfs.yml`. |
| `desy1/`, `desy3/`, `hscy1/`, `hscy3/`, `kids-1000/`, `desi_dr1_stage3/`, `desi_mocks/` | real Stage-III survey pipelines | `nersc-interactive`, `max_threads: 128` | NERSC inputs | DES Y3 uses RAIL DNF etc. |
| `buzzard/`, `skysim/`, `gaussian_sims/`, `lognormal/`, `ssi/`, `lssweights/`, `ext_cross_corr/`, `star-challenge/`, `randoms/`, `clustering_test/`, `twopoint_cluster/` | specialty | mixed | NERSC/mock | |

### `data/` shipped in the repo (tiny)

`data/fiducial_cosmology.yml` (598 B; pyccl YAML used as `fiducial_cosmology` input everywhere),
`data/DESY1-R-model.hdf5` (21 KB; metacal response model for `TXCosmoDC2Mock`), `data/bpz_riz.columns`,
`data/bpz_ugrizy.columns` (BPZ column files), `data/testing/test1.hdf`, `test2.hdf` (~4.5 KB each, unit tests).

### Example dataset (NOT in repo; there is no `bin/get-example-data` script - download is manual)

```bash
curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz   # 347 MB (same bytes as example-v10.tar.gz; CI pins EXAMPLE_DATA_FILE_VERSION=v10)
tar -zxvf example.tar.gz                                               # -> data/example/inputs/
```
Index at `https://portal.nersc.gov/cfs/lsst/txpipe/data/` also lists `example-v8/v9/v10`, `example-old`,
`example-jj*`, `example-cmb-lensing.tar.gz` (189 MB), `cosmodc2-20deg2.tar.gz` (13.6 GB), `flexcode_model.pkl`.
Files referenced by the example pipelines under `data/example/inputs/`: `metadetect_shear_catalog.hdf5`,
`shear_catalog.hdf5` (metacal), `lensfit_shear_catalog.hdf5`, `photometry_catalog.hdf5`,
`lensfit_photometry_catalog.hdf5`, `star_catalog.hdf5`, `lensfit_star_catalog.hdf5`, `exposures.hdf5`,
`example_flow.pkl` (RAIL FlowCreator), `hsc_ratios_and_specz.hdf5`, `HSC_grid_settings.pkl`, `rail-bpz-inputs/`,
`supreme_dc2_dr6d_v2_g_nexp_sum_1deg2.hs`, `supreme/`, `wigner.pkl`, `photoz_pdfs.hdf5`,
`cosmoDC2_trees_i25.3.npy`, `sample_cosmodc2_w10year_errors.dat`, `cluster_catalog.hdf5`,
`mock_nfw_shear_catalog.txt`, `mock_single_cluster_catalog.hdf5`, `mock_spectroscopic_catalog.hdf5`,
`cosmoDC2_v1.1.4_run_redmagic_highdens.fit`. The example is "1 square degree of simulated sky ... too small to
check any numerical results, just designed to test that the code runs" (`docs/src/example.rst`).

### `notebooks/`

`Welcome to TXPipe.ipynb` (runs `examples/metadetect/pipeline.yml` via `ceci.Pipeline.read(...).run()`
- "This will take a few minutes" - then explores HDF5 catalogs, `lens_photoz_pdfs.hdf5`, `QPNOfZFile`
n(z), `MapsFile`, SACC files, and appends `TXFourierTJPCovariance` programmatically),
`Reading TXPipe Outputs.ipynb` (data_types API: `TomographyCatalog.read_nbin/read_zbins`,
`MapsFile.list_maps/read_map`, `QPNOfZFile.get_nbin/read_ensemble`, `SACCFile`), `Syst3x2pt.ipynb`,
`Cluster_twopoint_test.ipynb`, `fsb.ipynb`, `cluster_counts/*.ipynb` (CL pipelines on 20 deg^2).

### Minimal laptop pipelines

1. **Smoke test (seconds-minutes, no photo-z, no 2pt)**: `ceci examples/mock_shear/pipeline.yml`
   -> `data/example/outputs_mock_shear/{shear_catalog, shear_tomography_classifier, shear_tomography_catalog,
   binned_shear_catalog, source_photoz_pdfs, cluster_shear_catalogs}.hdf5`.
2. **Full 3x2pt test (CI-grade)**: `ceci examples/metadetect/pipeline.yml` (61 stages, `max_threads: 2`,
   a few stages with `nprocess: 2` / `threads_per_process: 2`). Expect roughly 5-20 min on a laptop
   (notebook says "a few minutes"; CI runs it on 2-4 core GitHub runners). Outputs in
   `data/example/outputs_metadetect/`, logs in `data/example/logs_metadetect/<Stage>.out`; CI success check is
   `test -f data/example/outputs_metadetect/shear_xi_plus.png`. Requires `cache/workspaces` dir
   (`mkdir -p cache/workspaces` in CI) for NaMaster workspaces, and submodules for `TXConvergenceMaps`/`HOSFSB`.
3. Dry run / introspection only (no data needed beyond paths): `ceci --dry-run examples/metadetect/pipeline.yml`,
   `ceci --flow-chart chart.png examples/metadetect/pipeline.yml` (needs pygraphviz), `python bin/flow_chart.py ...`.
4. Single stage: `python -m txpipe TXTwoPointPlots --twopoint_data_real=data/example/outputs_metadetect/twopoint_data_real.sacc --twopoint_gamma_x=.../twopoint_gamma_x.sacc --config=examples/metadetect/config.yml --shear_xi_plus=out/shear_xi_plus.png ...`
   (copy the exact line from `--dry-run`).

---

## 4. Outputs

Final science products (all in `output_dir`, each HDF5 carries a `provenance` group with config, input UUIDs,
git hash/diff, module versions - `base_stage.gather_provenance`):

- **SACC files** (`.sacc`, FITS): `twopoint_data_real_raw.sacc` (unblinded TreeCorr xi+/xi-/gamma_t/w with
  `galaxy_shear_xi_plus/minus`, `galaxy_shearDensity_xi_t`, `galaxy_density_xi` data types; tracers `source_i`,
  `lens_j` with n(z) attached from the QP stacks), `twopoint_data_real.sacc` (blinded or copied),
  `twopoint_gamma_x.sacc`, `twopoint_data_fourier.sacc` (NaMaster `galaxy_shear_cl_ee/bb`,
  `galaxy_shearDensity_cl_e/b`, `galaxy_density_cl`), `twopoint_theory_real.sacc` / `twopoint_theory_fourier.sacc`
  (CCL+firecrown theory at `fiducial_cosmology`), `summary_statistics_real.sacc` / `summary_statistics_fourier.sacc`
  (data + covariance, from the covariance stages), `delta_sigma.sacc`, `gammat_field_center.sacc`,
  `cluster_sacc_catalog.sacc`.
- **Catalog/tomography HDF5**: `shear_tomography_catalog.hdf5`, `lens_tomography_catalog.hdf5`,
  `binned_shear_catalog.hdf5`, `binned_lens_catalog.hdf5`, `binned_random_catalog(.sub).hdf5`, `random_cats.hdf5`,
  `lens_photoz_pdfs.hdf5`, `shear_photoz_stack.hdf5` / `lens_photoz_stack.hdf5` (QP n(z) ensembles),
  `*_photoz_realizations.hdf5`, `tracer_metadata.hdf5` + `tracer_metadata_yml.yml`, `patch_centers.txt`.
- **Maps HDF5** (`MapsFile`: healpix pixel+value arrays, `list_maps()`): `source_maps`, `lens_maps`,
  `density_maps`, `mask`, `psf_maps`, `depth_map`, `bright_object_map`, `source_noise_maps`, `lens_noise_maps`,
  `convergence_maps`, `lss_weight_maps`.
- **PNG plots**: `shear_xi_plus/minus.png`, `shearDensity_xi.png`, `density_xi.png`, Fourier equivalents,
  `nz_source.png`/`nz_lens.png`, `*_map_plot.png`, `g_psf_T.png`, `g1_hist.png`, `rowe134.png`, `jk.png`,
  `brighter_fatter_plot.png`, convergence map plots, etc.
- No HTML report is generated by the pipeline itself; `bin/make_output_report.py` (summarises sacc + HDF5 with
  `tabulate`) and `bin/generate-web-page.py` (writes a run page to `/global/cfs/cdirs/lsst/www/txpipe/runs`)
  are separate NERSC utilities. `bin/h5show.py` and `bin/mapshow.py` are inspection helpers;
  `bin/sacc_to_twopoint.py` converts sacc -> CosmoSIS `twopoint` FITS.

**Feeding firecrown / augur**: the TXPipe SACC files are already in the firecrown input format (tracers with
n(z) + data types + optional covariance). TXPipe itself uses firecrown internally (`txpipe/utils/theory.py::theory_3x2pt`
builds a `firecrown.likelihood.gauss_family` `TwoPoint`/`ConstGaussian` likelihood via
`load_likelihood_from_script(theory_model, {"sacc_data": sacc})`, `tools.prepare(cosmo)`,
`likelihood.compute_theory_vector(tools)`) for `TXTwoPointTheory*` and `TXBlinding`. For inference, hand
`summary_statistics_real.sacc` (or `_fourier`) - the version WITH a covariance - to a firecrown likelihood factory
(`sacc_data` build parameter) / augur Fisher forecast. Files lacking a covariance (`twopoint_data_*.sacc` from the
example run) cannot be used for likelihoods without first running one of the covariance stages.

---

## 5. HPC usage (Perlmutter)

How TXPipe runs at NERSC today (`docs/src/nersc.rst`, `bin/nersc-test.sub`, `examples/*/*.sub`, ceci sites):

- Environment: `module load python; conda activate ./conda` (repo-local env from `perlmutter-install.sh`), or
  legacy `shifter --image=joezuntz/txpipe` / `site: {image: ghcr.io/lsstdesc/txpipe, volume: ${PWD}:/opt/txpipe}`.
- The whole pipeline runs **inside one SLURM allocation**; ceci's `mini` launcher is the scheduler and emits
  `srun -u -n <nprocess> --cpus-per-task=<threads> --nodes <nodes> ... python3 -m txpipe <Stage> ... --mpi`
  for every stage (NerscSite.command). `sbatch` scripts must NOT call srun themselves. `site.name` must be
  `nersc-interactive` (works for both `salloc` and `sbatch`) - the `nersc-batch` parsl/SlurmProvider mode is legacy.
- Example batch script (`bin/nersc-test.sub`):
  ```bash
  #SBATCH --qos=debug --time=00:30:00 --nodes=2 --account=m1727 --constraint=cpu
  module load python; conda activate ./conda
  export OMP_PROC_BIND=true; export OMP_PLACES=threads
  ceci examples/metadetect/pipeline.yml site.name=nersc-interactive resume=restart
  ```
  `examples/cosmodc2/Cluster_pipelines/20deg2-nersc.sub`: `-C cpu --qos=debug --nodes=1 --ntasks-per-node=32`.
  Older Cori scripts: `examples/2.2i/cori-2.2i.sub` (2 Haswell nodes, 9 h regular), `examples/skysim/cori-skysim.sub` (8 nodes).
- Typical per-stage resources for 20 deg^2 (`pipeline-20deg2-nersc.yml`): selection/calibration `nprocess: 32`;
  map making `nprocess: 8` (dask); noise maps `nprocess: 4`; `TXTwoPoint` `nprocess: 1, threads_per_process: 64`;
  `TXTwoPointFourier` `nprocess: 2, threads_per_process: 64`; covariances `threads_per_process: 32`.
  Full DC2 (`examples/2.2i`) uses 2 nodes x 9 h. `docs/src/parallel.rst`: ingestion = 1 process; selection = many
  MPI processes with Lustre striping; mapping ~8 procs/node; 2pt/covariance = max threads, can split over nodes.
- Inputs at NERSC: explicit CFS paths or DESC **data registry** (`registry: {}` + `inputs: {tag: {name: ...}}`),
  resolved by `Pipeline.data_registry_lookup` (needs `dataregistry` package). DP1 ingestion needs the LSST stack
  (`docs/src/lsst.rst`, `examples/dp1/ingest.yml`).
- Monitoring: `txpipe/ui/nersc_monitor.py` and `pipeline_monitor.py` ssh to NERSC (paramiko, still hardcoded
  `cori.nersc.gov`) and poll `log_dir`; experimental. Practical status = list `output_dir` for final vs
  `inprogress_*` files and tail `log_dir/<Stage>.out`.

**What the MCP server needs to submit a TXPipe pipeline as a facility job** (hep-genesis model, see
`cosmic_emulator_server/mcp_server/dispatch.py`; `hep_genesis.iri.nersc.dispatch.run_on_perlmutter(job_name,
files: dict[str,str], pip_deps, script_name='compute.py', result_file='results.json', duration, nodes, qos,
constraint, artifact_dir)` and `run_codes_on_perlmutter(function, args, pip_deps, codes, duration, nodes, ...)`):

1. The "kernel + pip deps" venv bootstrap (`python -m venv --system-site-packages` + `pip install deps`) **cannot
   build the TXPipe environment** (namaster, treecorr, healpy, mpi4py-against-cray-mpich, h5py-mpi, firecrown,
   RAIL...). The job must instead activate a **pre-built TXPipe env on Perlmutter**: either a conda env created
   once with `bin/perlmutter-install.sh` under `$CFS/<project>/txpipe/conda`, or `shifter --image=ghcr.io/lsstdesc/txpipe`.
   So the dispatched kernel should be a thin Python/shell wrapper (`pip_deps=None` -> "using module python as-is")
   that runs `module load python; conda activate <TXPIPE_ENV>; cd <TXPIPE_CHECKOUT>; ceci <pipeline.yml> site.name=nersc-interactive ...`
   via `subprocess`, then writes `results.json` (status, list of output files, tail of logs).
2. Stage in: generated `pipeline.yml` + `config.yml` (small; `files=` dict of text), with `inputs:` pointing at
   CFS paths or registry names; `output_dir`/`log_dir` under the facility `NERSC_WORKDIR`. A TXPipe checkout must
   exist on CFS (clone once; or ship the ~1 MB `txpipe/` package in the pack with `--recurse-submodules` content).
3. Resources: `nodes>=1`, `constraint=cpu`, `qos=debug` (30 min) for the example/20 deg^2 subsets, `regular` with
   hours for full pipelines; `duration` must exceed the sum of stage times since ceci runs the DAG serially/concurrently
   inside the allocation. Pipeline YAML `nprocess`/`threads_per_process` must be consistent with the node count.
4. Fetch back: SACC files, PNGs and `tracer_metadata_yml` are small and should be returned as artifacts; HDF5
   catalogs/maps can be GB-scale and should stay on CFS (return paths + `h5show`-style summaries). Note the IRI
   (Globus-free) path skips binary artifacts - PNG retrieval needs the Globus path or base64-in-results.json.
5. Resume: re-submitting with `resume: true` and the same `output_dir` continues where the last job stopped.

---

## 6. Proposed MCP tool family (`txpipe_*`)

| # | Tool | Weight | What it does |
|---|---|---|---|
| 1 | `list_txpipe_stages(group=None, include_extensions=True)` | light | Import `txpipe` (+`txpipe.extensions`) and return `ceci.PipelineStage.pipeline_stages` as `{name, module, doc_first_line, parallel, dask_parallel, n_inputs, n_outputs}`, grouped like `docs/make-stages.py`. Fallback without the env: AST scan of `txpipe/` (the approach used for this survey). |
| 2 | `describe_txpipe_stage(name)` | light | `inputs`/`outputs` with file-type names, `config_options` with dtype/default/required/help (`StageParameter` fields; `cls._describe_configuration_text()`), the single-stage CLI template from `generate_command`, and which example pipelines use it. |
| 3 | `list_txpipe_examples()` / `describe_txpipe_pipeline(path)` | light | Parse an `examples/**/pipeline.yml`: stages, site, launcher, inputs (and whether they exist locally), output_dir, missing-input report; optional `ceci --flow-chart` PNG. |
| 4 | `generate_txpipe_pipeline_config(spec, output_dir)` | light | From a high-level spec (`catalog_type: metadetect|metacal|lensfit|mock`, `stages` or `preset: source_only|3x2pt_real|3x2pt_fourier|mock_shear`, `inputs: {...}`, `source_zbin_edges`, `lens_zbin_edges`, `nside`, 2pt binning, `site: local|perlmutter`, `nprocess/threads`) write `pipeline.yml` + `config.yml`, starting from the matching example and overriding keys. Returns the files and a dependency check (every stage input is an overall input or another stage's output). |
| 5 | `validate_txpipe_pipeline(pipeline_yml, overrides=[])` | light (needs txpipe env) | Run `ceci --dry-run <yml> <overrides>`; return the per-stage command list, unresolved input tags, missing files, and the DAG order. Pure Python alternative: `ceci.Pipeline.build_config` + `Pipeline.create` with `dry_run=True`. |
| 6 | `run_txpipe_pipeline(pipeline_yml, overrides=[], site='local'|'perlmutter', resume=True, walltime, nodes)` | **heavy** | Local: `subprocess` `ceci ...` in the txpipe env, stream `log_dir`, return `{status, files: [final outputs], metadata: {stage_times, failed_stage, log_tail}}`. Remote: dispatch as described in section 5 (shell wrapper kernel, pre-built env). Should refuse `nprocess>1` locally unless `mpirun` and MPI-h5py are available. |
| 7 | `run_txpipe_stage(stage, inputs: {tag: path}, outputs_dir, config: {...}, nprocess=1, threads=1, site)` | heavy-ish | Build the `python -m txpipe <Stage> --tag=path ... --config=tmp.yml` command (`generate_command`) and run it locally or remotely. Useful for re-running a plotting/theory/covariance stage on existing outputs. |
| 8 | `get_txpipe_run_status(output_dir, log_dir)` | light | Classify each stage from `output_dir` (`inprogress_*` vs final) + `log_dir/<Stage>.out` tails; for remote jobs poll the hep-genesis job id. |
| 9 | `inspect_txpipe_outputs(path_or_dir)` | light (needs sacc/h5py/qp) | For `.sacc`: tracers, data types, tracer combinations, n points, has_covariance, n(z) summary (what `bin/make_output_report.py::summarize_sacc` does). For HDF5: group/dataset table (`bin/h5show.py`), `MapsFile.list_maps()`, `TomographyCatalog.read_nbin/read_zbins`, provenance attrs. Returns JSON + optional CSV extracts. |
| 10 | `plot_txpipe_results(sacc_path(s), kind='xi'|'cl'|'nz'|'map', theory_sacc=None)` | light | `txpipe.plotting.full_3x2pt_plots(sacc_files, labels, cosmo)` (`bin/plot_example.py`), `TXPhotozPlot`-style n(z), `healpy`/`skyproj` map PNGs (`bin/mapshow.py`). |
| 11 | `convert_txpipe_sacc(sacc_path, to='twopoint_fits'|'csv')` | light | `bin/sacc_to_twopoint.py` logic; CSV of (type, tracer1, tracer2, theta/ell, value, err). |
| 12 | `fetch_txpipe_example_data(which='example'|'cmb'|'cosmodc2-20deg2', dest)` | light I/O but 347 MB / 189 MB / 13.6 GB | Download + extract `portal.nersc.gov/cfs/lsst/txpipe/data/*.tar.gz` with size check and version pin (`v10`). |

Heavy tools (6, 7) are the only ones needing dispatch; everything else runs anywhere a `txpipe` env (or at
least `sacc`, `h5py`, `qp`, `healpy`, `matplotlib`) is present, and tools 1, 3, 4 can run with no env at all via AST
+ YAML parsing.

---

## 7. Gotchas

- **Dependency weight**: ~60 conda + 12 pip packages incl. compiled NaMaster, TreeCorr, healpy, MPI-enabled h5py,
  cosmosis, camb, jax, RAIL; env creation takes a long time and several GB. No wheel for `txpipe` itself - it is run
  from the checkout with cwd on `sys.path`; the MCP server must `cd` into the TXPipe directory (relative
  `python_paths`, `data/fiducial_cosmology.yml`, `cache/` are all cwd-relative).
- **Python pins**: 3.12 local / 3.13 Perlmutter, `numpy 2.3`; TXPipe cannot share the `cosmic-emu` (3.12, numpy
  pinned for emulators) or `llm_env` (3.10) envs. None of the local envs has ceci/txpipe.
- **Submodules**: `WLMassMap` and `pyfsb` are empty in this clone; `TXConvergenceMaps` and `HOSFSB` in the
  metadetect example will fail without `git submodule update --init --recursive`.
- **Data**: nothing runs without the 347 MB `example.tar.gz` (extracts to `data/example/inputs/`); the 20 deg^2
  set is 13.6 GB; real-survey and DC2 pipelines hardcode NERSC CFS paths or registry names. There is no
  `bin/get-example-data` script - the download is a `curl` in the README/CI.
- **MPI**: `nprocess > 1` requires `mpirun` + `mpi4py` + MPI-enabled h5py (base_stage raises
  "h5py module is not MPI-enabled" otherwise). On NERSC, `nprocess>1` is refused on login nodes (`SLURM_JOB_ID`
  check); mpi4py must be rebuilt against cray-mpich (handled by `perlmutter-install.sh`). Dask stages
  (`dask_parallel=True`) use dask-mpi when `nprocess>1`.
- **Ordering/resume**: ceci orders stages by tag dependencies, not list order; `resume: true` skips a stage only if
  ALL its outputs exist (final names, not `inprogress_*`); `resume: restart` reruns everything; `refuse` errors on
  existing outputs. Changing config does NOT invalidate outputs - callers must delete downstream files
  (`bin/delete_downstream.py`) or use `restart`. Aliases let one class run under several instance names
  (`name:` + `classname:`); the config section key is the instance name.
- **Logs**: one file per stage instance, `{log_dir}/{InstanceName}.out` (stdout+stderr merged); on failure ceci
  prints "Standard output and error streams in {log_dir}/{job}.out". `pipeline_log` is only used by parsl.
  Provenance (config, versions, git diff) is embedded in every HDF5 output under `provenance/`.
- **Caches**: NaMaster workspaces in `cache_dir` (`./cache/workspaces`, `./cache/twopoint_fourier`) and TreeCorr
  patches in `patch_dir` (`./cache/patches`) - CI pre-creates `cache/workspaces`; these can be large and should
  live on scratch at NERSC.
- **Blinding**: `TXBlinding` with `delete_unblinded: true` removes the raw sacc; sims should use `TXNullBlinding`.
- **Covariance stages are slow**: excluded from the example pipeline; `TXFourierTJPCovariance`/
  `TXFourierNamasterCovariance` need NaMaster + mask + workspace cache; the Welcome notebook shows adding
  `TXFourierTJPCovariance` afterwards via the Python API.
- **Container images** (`joezuntz/txpipe`, `ghcr.io/lsstdesc/txpipe`) are called legacy in the docs; may lag the
  conda env pins.
- **Platform quirk**: `KMP_DUPLICATE_LIB_OK=TRUE` is required for healpy on conda (set by `install.sh`);
  `HDF5_USE_FILE_LOCKING=FALSE` is required on Lustre.
