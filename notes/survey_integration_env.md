# Cross-cutting survey: integration repos + environment feasibility

Date: 2026-10-07. Scope: LSSTDESC repos that show CCL / TXPipe / firecrown / augur
working together, plus the local conda situation and a plan for the server env.
The four primary clones were treated read-only.

Local clone state (for reference):

| Clone | HEAD | Version |
|---|---|---|
| CCL | 66b05e1 (2026-07-28) | v3.3.6 |
| CCLX | 738bee0 (2025-04-21) | (notebooks, no version) |
| TXPipe | efad606 (2026-10-06) | framework-paper-ir-2322-gefad6069 (not pip-installable; no setup.py/pyproject — run from repo via PYTHONPATH + ceci) |
| firecrown | 51193df (2026-10-06) | v1.16.0-74 (dev past 1.16.0) |
| augur | f8ab7cf (2026-08-13) | 1.2.4 |

---

## Part A — LSSTDESC org survey (488 public repos scanned)

Source: `gh api orgs/LSSTDESC/repos` pages 1-5. Sizes are GitHub `size` (KB).

### A.1 Candidates (relevance-ranked)

| Repo | Pushed | Size | Stars | Why it matters | Cloned? |
|---|---|---|---|---|---|
| **sacc** | 2026-07-02 | 3.3 MB | 15 | The data-format glue: TXPipe writes it, TJPCov adds covariance to it, firecrown/augur read it. v2.4. Ships `examples/example-txpipe-sacc1.sacc` (a real TXPipe-produced file) + notebooks (`Create_Sacc`, `SACC_read/write`, DES/KiDS converters). | yes |
| **ceci** | 2026-10-06 | 0.8 MB | 13 | TXPipe's pipeline runner (`ceci pipeline.yml`). conda-forge name `lsstdesc-ceci` (2.5.1). Has a toy pipeline (`ceci_example/`, `tests/test.yml`) showing stage/alias/site/launcher YAML syntax — the template for any MCP tool that composes TXPipe pipelines. | yes |
| **TJPCov** | 2026-10-02 | 63 MB | 14 | Covariance calculator used by both TXPipe (`TXFourierTJPCovariance`, `covariance.py`) and augur (`cov_utils.py`, `cov_type: tjpcov`). `run_tjpcov.py config.yml -o out.sacc`; examples for full 3x2pt fsky and NaMaster covariances; SLURM MPI script. v0.5.1 on conda-forge. | yes |
| **minimal_mcp_pipelines** | 2026-10-07 | 0.9 MB | 4 | "MCP" here = DESC *Modeling & Combined Probes* WG, not Model Context Protocol — but it is exactly the end-to-end recipe we want: sacc -> TJPCov covariance -> firecrown likelihood -> CosmoSIS/nautilus MCMC on Perlmutter. Includes a fiducial 3x2pt sacc and SLURM jobscript. | yes |
| **desc-cosmology-env** | 2026-10-01 | 1.2 MB | 0 | Official DESC conda env that holds firecrown + lsstdesc-augur + pyccl + sacc + tjpcov + cosmosis + namaster together; conda-lock files, Dockerfile (`lsstdesc/desc-cosmology`), NERSC setup scripts (`source $CFS/lsst/groups/MCP/setup-cosmology.sh`). The canonical answer to "can these coexist". | yes |
| **desc-python** | 2026-10-07 | 5.4 MB | 7 | The broader NERSC `desc-python` env + jupyter kernel docs; useful for HPC dispatch (what is already installed at NERSC). | yes |
| **txpipe-docker** | 2024-09-05 | 0.1 MB | 1 | Dockerfiles for the `joezuntz/txpipe` / `ghcr.io/lsstdesc/txpipe` images used via shifter at NERSC. Shows the firecrown v1.7.5-with-numcosmo-stripped hack TXPipe needs. | yes |
| **firecrownx** | 2023-07-20 | 5 MB | 0 | "Examples of the use of Firecrown" — but the clone contains only a README (examples moved into firecrown's `firecrown examples` CLI and `tutorial/`). Low value. | yes (tiny) |
| **fisherA2Z** | 2026-09-29 | 21 MB | 2 | Standalone 3x2pt Fisher forecast (CCL-based, photo-z systematics). Alternative/cross-check to augur's Fisher. | yes |
| **crowkit** | 2026-04-02 | 18 KB | 0 | Fisher manager on CROW + TJPCov + DerivKit (augur 1.2.x also supports derivkit). README + pyproject only. | yes (tiny) |
| **sl_forecast** | 2026-10-05 | 51 KB | 1 | Strong-lensing forecast modules "to be called from Firecrown" — skeleton only. | yes (tiny) |
| **desc-example-env** | 2024-10-01 | 5 KB | 0 | Minimal conda-pack + Dockerfile pattern (mpich external_* trick) used by DESC env builders. | yes (tiny) |
| 6x2pt_LSST_and_ext_Spec | 2025-11-03 | 78 MB | 4 | 6x2pt (LSST + DESI/4MOST) Fisher toolkit; outputs sacc files. Not cloned (size, overlaps augur/fisherA2Z). | no |
| forecasting_validation | 2025-01-28 | 206 MB | 0 | Forecasting TT validation of CCL C_ell's; mentions firecrown/augur/TJPCov as future targets. Not cloned (size, notebooks). | no |
| txpipe-cosmodc2 | 2022-11-16 | 283 MB | 1 | TXPipe-on-cosmoDC2 paper repo; large data. Not cloned. | no |
| nulltests_txpipe | 2025-04-21 | 88 MB | 1 | Notebooks for TXPipe null-test reanalysis. Not cloned. | no |
| txpipe-reanalysis | 2023-10-30 | 55 MB | 3 | DES Y1/HSC/KiDS through TXPipe then CosmoSIS/firecrown; README documents the shifter workflow. Not cloned. | no |
| legacy_blinding | 2023-12-08 | 7.5 MB | 1 | Muir et al. 2pt blinding with CosmoSIS (TXPipe's `TXBlinding` stage implements the same). Not cloned. | no |
| cluster_mor_firecrown_modeling, CLxTJP, rail_ccl | — | tiny | 0 | Stubs / empty. Not cloned. | no |
| CCLX | 2025-04-20 | 21 MB | 34 | Already present. `generate_dv_nonlimber.ipynb` and `generate_dv_PT_nonlimber.ipynb` build a **sacc** 3x2pt data vector with CCL tracers + `sacc.BandpowerWindow` — the CCL -> sacc bridge. | existing |

No LSSTDESC repo named like "DESC_Y1_pipeline" or "txpipe-firecrown-example" exists; the
closest end-to-end artefacts are `minimal_mcp_pipelines`, TXPipe's own `examples/*/pipeline.yml`,
and augur's `examples/srd_y1_3x2*`.

### A.2 What was cloned into `external/`

All `git clone --depth 1 https://github.com/LSSTDESC/<name>`:

```
external/sacc                   4.5M   v2.4
external/ceci                   916K   2026-10-06 HEAD (lsstdesc-ceci 2.5.1 on conda-forge)
external/TJPCov                  37M   2026-05-01 HEAD (0.5.1 on conda-forge)
external/minimal_mcp_pipelines  6.7M
external/desc-cosmology-env      12M
external/desc-python             24M
external/txpipe-docker          200K
external/firecrownx             128K   (README only)
external/fisherA2Z              102M   (large because of bundled data)
external/crowkit                148K
external/sl_forecast            276K
external/desc-example-env       140K
```

Not cloned by this survey: `external/augur` (12 MB, full clone of LSSTDESC/augur at the same HEAD
f8ab7cf as the top-level `augur/`) appeared at 14:42 during the survey — presumably created by the
augur package-survey agent. It duplicates `../augur` and can be removed.

### A.3 Concrete cross-package workflows

**W1. TXPipe -> sacc -> firecrown theory (inside TXPipe)**
Files: `TXPipe/txpipe/twopoint.py` (`TXTwoPoint`, outputs `twopoint_data_real_raw` as `SACCFile`,
`S.save_fits(...)` line 458), `TXPipe/txpipe/blinding.py` (`TXBlinding`, imports pyccl+firecrown+sacc,
Muir blinding of the sacc), `TXPipe/txpipe/theory.py` (`TXTwoPointTheoryReal/Fourier`: loads the sacc,
builds a CCL cosmology from `fiducial_cosmology.yml`, calls `theory_3x2pt`), and
`TXPipe/txpipe/utils/theory.py` + `utils/theory_model.py` which call
`firecrown.likelihood.likelihood.load_likelihood_from_script` with a `build_likelihood()` that
makes `wl.WeakLensing` / `nc.NumberCounts` sources + `TwoPoint` stats + `ConstGaussian`, then
`likelihood.compute_theory_vector(tools)` to fill a theory sacc.
Run: `ceci examples/metadetect/pipeline.yml` (needs test data:
`curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz`). Stage order in
`examples/metacal/pipeline.yml`: `TXTwoPoint -> TXBlinding -> TXTwoPointTheoryReal -> TXTwoPointPlotsTheory`.
**Caveat:** `theory_model.py` imports `firecrown.likelihood.gauss_family.*`, a module tree removed
in firecrown v1.10.0 (commit 702b486, 2025-02-21). TXPipe therefore pins `firecrown=1.7.*`
(`bin/environment-local.yml`) and its Docker image installs firecrown v1.7.5 with numcosmo stripped.
TXPipe's theory/blinding stages will NOT run against firecrown 1.16.

**W2. TXPipe -> TJPCov covariance -> sacc**
`TXPipe/txpipe/covariance.py`: `TXFourierGaussianCovariance` / `TXRealGaussianCovariance` use
`tjpcov.wigner_transform` for Gaussian fsky covariances; `TXFourierTJPCovariance` (line 620) builds a
`tjpcov.covariance_calculator.CovarianceCalculator({"tjpcov": ..., "GaussianFsky": {"fsky": fsky}})`
from the pipeline config's `tjpcov` section and writes the covariance into the sacc.
Standalone equivalent: `python external/TJPCov/run_tjpcov.py external/TJPCov/examples/full_3x2pt_cov_example.yml -o cov.sacc`
(cosmology in `examples/3x2pt_cov_cosmology.yaml`; input sacc from `tests/benchmarks/32_DES_tjpcov_bm/cls_cov.fits`).

**W3. augur: CCL + firecrown factories + TJPCov -> fiducial sacc -> Fisher**
`augur/augur/generate.py`: builds `sacc.Sacc()` with `NZ` tracers (SRD n(z) from `augur/data/*.txt`),
computes the fiducial data vector through firecrown (`firecrown.likelihood.two_point.TwoPoint`,
`ConstGaussian`, `ParamsMap`), attaches a covariance from either the SRD `.npy` (`cov_type: SRD`),
an internal Gaussian (`gaus_internal`), or TJPCov (`cov_type: tjpcov` -> `FourierGaussianFsky`,
lines 748-795), and saves `fiducial_sacc_path`. `augur/augur/utils/firecrown_interface.py` wraps
`firecrown.ccl_factory.CCLFactory` (pure-CCL mode, CAMB extras, mu-Sigma MG) and
`TwoPointFactory`/`TwoPointExperiment`. `augur/augur/analyze.py` does Fisher via numdifftools / derivkit.
Run: `export AUGUR_DIR=$PWD/augur; augur augur/examples/srd_y1_3x2.yml -v` (CLI in `augur/cli.py`:
generate -> fisher -> postprocess). Needs no CosmoSIS.

**W4. augur likelihood -> CosmoSIS sampler**
`augur/examples/srd_y1_3x2_cosmosis.ini` runs `consistency camb firecrown_likelihood` with
`file = ${FIRECROWN_DIR}/firecrown/connector/cosmosis/likelihood.py` and
`likelihood_source = ${AUGUR_DIR}/examples/srd_y1_3x2_like.py`, where `build_likelihood()` just
calls `augur.generate.generate('./config_test.yml', return_all_outputs=True)`.
Requires `CSL_DIR` (cosmosis-standard-library built via `cosmosis-build-standard-library`),
`FIRECROWN_DIR`, `AUGUR_DIR`. Run: `cd augur/examples && cosmosis srd_y1_3x2_cosmosis.ini`.

**W5. minimal_mcp_pipelines: sacc -> TJPCov -> firecrown -> CosmoSIS/nautilus at NERSC**
`external/minimal_mcp_pipelines/1_make_covmat/DESY6BAO_wtheta_example/`: `cov_calculator.py` runs
`CovarianceCalculator('config_wtheta_DESY6BAO.yml')` (Jinja-templated per-bin `Ngal_lens{i}`,
`bias_lens{i}`, `RealGaussianFsky`, `use_mpi: True`) on a sacc in `wtheta_sacc/`; SLURM wrapper
`run_cov_wtheta.sh`.
`2_run_mcmc/forecast_3x2pt.py` is a firecrown `build_likelihood` (5 lens + 5 source bins, LinearAlignment
IA, PhotoZShift, magnification, linear bias) over
`paul_auto_forecast_fid_3x2pt_linear_sys_limber_20_log_bins.sacc`; `forecast_3x2pt.ini` =
`consistency camb firecrown_likelihood`, `sampler = nautilus`; `jobscript_perlmutter.sh` =
`source /global/cfs/cdirs/lsst/groups/MCP/setup-cosmology-dev.sh; srun cosmosis --mpi forecast_3x2pt.ini`.
This is the best template for the server's NERSC dispatch path.

**W6. firecrown built-in example generators (CCL -> sacc -> likelihood factory)**
`firecrown/firecrown/app/examples/` registers `cosmic_shear`, `cmb_cross`, `sn_srd`, `des_y1_3x2pt`;
each has `generate_sacc()` (pyccl tracers -> sacc with covariance) and `generate_factory()` (YAML
`TwoPointFactory` config). CLI: `firecrown examples <name> ...`, `firecrown sacc view <file>`,
`firecrown cosmology`, `firecrown experiment`. Tutorials: `firecrown/tutorial/two_point_workflow.qmd`,
`two_point_sacc_data.qmd` (`load_sacc_data`, `extract_all_real_data`, `TwoPointFactory`).
Test fixtures: `firecrown/tests/sacc_data.hdf5`, `tests/legacy_sacc_data.fits`.

**W7. CCLX -> sacc**
`CCLX/generate_dv_nonlimber.ipynb` / `generate_dv_PT_nonlimber.ipynb`: CCL `NumberCountsTracer` /
`WeakLensingTracer` -> `cosmo.angular_cl` -> `s.add_ell_cl('galaxy_density_cl'|'galaxy_shear_cl_ee'|...,
'lens%d','src%d', ell, cl, window=sacc.BandpowerWindow(...))` -> sacc file consumable by firecrown/augur.
No CCLX notebook imports firecrown directly.

**W8. firecrown CI tests augur downstream**
`firecrown/.github/workflows/ci-reusable.yml` stage 3 clones `lsstdesc/augur`, `conda install jinja2 tjpcov`,
`pip install -e .`, `export AUGUR_DIR=$PWD; pytest augur/tests` — confirms augur is expected to track
firecrown master.

Cross-reference greps: firecrown mentions "txpipe" nowhere; TXPipe mentions firecrown in
`blinding.py, covariance*.py, utils/theory*.py, bin/environment-*.yml, docs/src/conf.py`; augur never
mentions TXPipe (it consumes sacc only); CCL mentions firecrown only in `readthedocs/source/terms.rst`.

---

## Part B — environment feasibility

### B.1 Local machine and existing envs

macOS 26.6.2 (Darwin 25.6.0), **arm64 (Apple Silicon)**, conda 23.7.4 at `/Users/nesar/anaconda3`,
channels = `defaults` only, no mamba/micromamba. Probed with each env's python directly:

| env | python | numpy | pyccl | camb | sacc / firecrown / augur / ceci / txpipe / tjpcov / cosmosis / healpy / treecorr / pymaster |
|---|---|---|---|---|---|
| cosmic-emu | 3.12.13 | 2.4.6 | 3.3.6 | 1.6.6 | all MISSING (h5py 3.14 present) |
| llm_env | 3.10.9 | 1.26.4 | 3.3.1 | 1.5.5 | all MISSING |
| hep-genesis | 3.12.13 | 2.5.2 | MISSING | MISSING | all MISSING; has `mcp`, `hep_genesis 0.3.12` |
| base | 3.11.5 | 1.24.3 | MISSING | MISSING | all MISSING |

(Importing `firecrown`/`augur` from inside the project dir falsely "succeeds" as namespace packages
because `firecrown/` and `augur/` clone directories exist — always test imports from another cwd.)

Nothing beyond pyccl exists locally; a new env is required regardless.

**hep-genesis-agent**: `/Users/nesar/Projects/Tutorials/HEPKE/hep-genesis-agent`
(backend package `/Users/nesar/Projects/Tutorials/HEPKE/hep-genesis-agent/backend/src/hep_genesis/__init__.py`,
version 0.3.12, editable-installed in env `hep-genesis`). Dispatch template:
`/Users/nesar/Projects/Tutorials/cosmic_emulator_server` (branch `dispatch`, `mcp_server/dispatch.py`
imports `hep_genesis` lazily only when `set_dispatch("polaris"|"perlmutter")` is called; install with
`pip install -e <hep-genesis-agent>/backend[iri]` into the server env).

### B.2 Declared dependencies

| | CCL (pyproject) | firecrown 1.16 (pyproject / environment.yml) | augur 1.2.4 (pyproject / environment.yml) | TXPipe (bin/environment-local.yml) |
|---|---|---|---|---|
| python | >=3.10 (wheels cp310-cp314) | **>=3.12** | >=3.11 | 3.12.* (perlmutter: 3.13.*) |
| numpy | any | **>=2.0,<2.4** | >=2.0 | 2.3.* |
| scipy | any | >=1.13 | >1.12 | 1.16.* |
| pyccl | (is 3.3.6) | >=2.8.0 | >=3.3.1 | **3.2.*** |
| sacc | — | **>=2.4** | env.yml: **<2.2**; pyproject/conda recipe: unpinned | **sacc[all]==2.2.*** (pip) |
| firecrown | — | — | >=1.14.0 | **1.7.*** (conda) / v1.7.5 (docker, numcosmo stripped) |
| tjpcov | — | — | yes | 0.4.* |
| camb | — | <2.0 | (via firecrown) | 1.6.* |
| cosmosis | — | >=3.0 (+ cosmosis-build-standard-library, compilers) | (via firecrown) | 3.24.* |
| numcosmo | — | >=0.27 | (via firecrown) | **removed on purpose** |
| cobaya, isitgr | — | pip | — | — |
| lsstdesc-crow, clmm | — | yes | crow | CLMM 1.14.6 via git pip |
| ceci | — | — | — | lsstdesc-ceci 2.4.* |
| heavy extras | — | — | healpy, qp-prob, numdifftools | namaster 2.5, treecorr 5.1, healpy 1.18, healsparse, h5py mpi_mpich, mpi4py, mpich, dask(+dask-mpi), jax 0.7, pz-rail 1.1.2 (git), pzflow>=4, hyperbolic, hybrideb, dsigma, glass, dsf (git), photerr, somoclu, pygraphviz, skyproj, tables-io-full, qp-prob |

conda-forge today (api.anaconda.org): pyccl 3.3.6 (osx-arm64 yes), firecrown 1.16.0 (noarch, via
`firecrown-deps` metapackage), lsstdesc-augur 1.2.3 (noarch; depends `firecrown>=1.14, pyccl>=3.3.1,
tjpcov, qp-prob, healpy, numdifftools, ...` — **no sacc pin**), lsstdesc-ceci 2.5.1, sacc 2.4
(`numpy>=2`), tjpcov 0.5.1 (depends `namaster>2, mpi4py, camb, healpy`), namaster 3.0.1 (osx-arm64
yes), treecorr 5.1.4, cosmosis 3.26, numcosmo 0.28.0, camb 2.0.4 (must cap `<2.0`), mpich 5.0.2.

### B.3 Compatibility analysis

1. **Python**: intersection of CCL(>=3.10) ∩ firecrown(>=3.12) ∩ augur(>=3.11) ∩ TXPipe(3.12.*) = **3.12**.
   (desc-cosmology-env uses 3.13; TXPipe/perlmutter uses 3.13 — 3.13 also works for the light set.)
2. **pyccl**: firecrown >=2.8 and augur >=3.3.1 are both satisfied by the CCL clone (v3.3.6) and by
   conda-forge 3.3.6. TXPipe pins **3.2.***, which violates augur's >=3.3.1 — conflict #1
   (soft: TXPipe's pin is a lock-style pin, not a known incompatibility; CCL 3.3 is API-compatible).
3. **numpy**: everyone is numpy 2. firecrown caps `<2.4`; sacc 2.4 needs `>=2`. Use numpy 2.3.x.
   Note `cosmic-emu` is on numpy 2.4.6, so firecrown cannot be dropped into cosmic-emu as-is.
4. **sacc**: firecrown 1.16 requires **sacc>=2.4**, but augur's `environment.yml` says **sacc<2.2**
   (commit f8ab7cf "Use sacc fix (#149)", 2026-08-13 — a workaround for a sacc 2.2/2.3 bug) and TXPipe
   pins **2.2.***. The conda-forge augur 1.2.3 recipe carries no sacc pin, and DESC's own
   `desc-cosmology-env` SLAC lock (2026-10-01) ships **firecrown 1.16.0 + lsstdesc-augur 1.2.3 +
   sacc 2.4 + pyccl 3.3.6 + tjpcov 0.5.1 + numpy 2.3.5 + python 3.13** together, so sacc 2.4 is the
   validated choice for the light set; treat augur's `<2.2` as stale. Conflict #2 is only with TXPipe.
5. **firecrown API split (the hard conflict)**: TXPipe's `utils/theory_model.py` imports
   `firecrown.likelihood.gauss_family.*`, deleted in firecrown 1.10 (2025-02). TXPipe pins firecrown
   1.7.* and even strips numcosmo from it. augur needs firecrown >=1.14 (uses `firecrown.ccl_factory`,
   `TwoPointFactory`, `data_functions`). **No firecrown version satisfies both** -> one env cannot hold
   TXPipe(theory/blinding stages) and augur simultaneously. TXPipe's measurement stages (TXTwoPoint,
   maps, covariance via TJPCov) do not import firecrown, so a TXPipe env without firecrown is viable
   if the theory/blinding stages are skipped — but then TXPipe's "pipeline.yml" examples must be edited.
6. **Heavy compiled/MPI stack (TXPipe only)**: namaster, treecorr, healpy, healsparse, h5py=*=mpi_mpich_*,
   mpi4py+mpich, dask-mpi, jax 0.7 pinned, RAIL 1.1.2 from git, 5 other git pips. TXPipe README still says
   "M1 macs are not yet supported"; `bin/install.sh` installs a private Miniforge into `./conda` (we must
   not do that inside the read-only clone — use `-p` elsewhere or a copy of the yml). TJPCov's conda
   package pulls namaster+mpi4py even for the light env (osx-arm64 builds exist, fine).
7. **CosmoSIS**: firecrown/augur conda installs pull `cosmosis` + `cosmosis-build-standard-library`;
   the CSL must then be built (`source $CONDA_PREFIX/bin/cosmosis-configure; cosmosis-build-standard-library`)
   and `CSL_DIR`, `FIRECROWN_DIR`, `AUGUR_DIR` exported (augur README). Not needed for augur's Fisher path
   (W3), only for W4/W5 sampling.
8. **Channel hygiene**: the local conda uses `defaults` only; everything above is conda-forge. Create the
   env with `--override-channels -c conda-forge` (or install miniforge/mamba) to avoid mixed solves.

### B.4 Recommendation: split envs

- **`desc-mcp` (light, local + server runtime)**: python 3.12, conda-forge: pyccl 3.3.6, sacc 2.4,
  firecrown 1.16, lsstdesc-augur 1.2.x (or `pip install -e augur --no-deps` from the clone to get 1.2.4),
  tjpcov 0.5.1, lsstdesc-ceci (pure python — lets the server parse/validate/emit ceci pipeline YAML and
  run ceci's dry-run/flow-chart without TXPipe's heavy deps), cosmosis + CSL, camb<2, numpy<2.4,
  plus `mcp[cli]`, pydantic, matplotlib, and `pip install -e <hep-genesis-agent>/backend[iri]`.
  Mirrors `external/desc-cosmology-env/conda/lock/environment.yml` minus diffsky/lsdb/opencosmo/rubin-sim.
  This covers CCL tools, sacc inspection, firecrown likelihood/theory, augur generate+Fisher, TJPCov
  covariances, and composing TXPipe pipeline YAML.
- **TXPipe (heavy)**: do not install locally on arm64 for now. Run at NERSC via the official container
  (`shifter --image ghcr.io/lsstdesc/txpipe` / `joezuntz/txpipe`, see `external/txpipe-docker`) or the
  `bin/perlmutter-install.sh` env; the MCP server's TXPipe tools generate `pipeline.yml`/`config.yml` and
  dispatch `ceci pipeline.yml` through hep-genesis to Perlmutter. For a laptop TXPipe later:
  `conda env create -p ~/envs/txpipe -f TXPipe/bin/environment-local.yml` (separate from `desc-mcp`;
  it pins firecrown 1.7.* and sacc 2.2.*).
- **NERSC side**: `source $CFS/lsst/groups/MCP/setup-cosmology.sh` (desc-cosmology env: firecrown,
  augur, pyccl, sacc, tjpcov, cosmosis, namaster, mpi4py) for light workloads; TXPipe shifter image or
  TXPipe's `./conda` for pipelines. `external/minimal_mcp_pipelines/2_run_mcmc/jobscript_perlmutter.sh`
  is the SLURM template.

### B.5 Proposed `scripts/setup_env.sh` (sketch — NOT executed)

```bash
#!/usr/bin/env bash
# Create the light "desc-mcp" env: pyccl + sacc + firecrown + augur + tjpcov + ceci + cosmosis.
# TXPipe is deliberately NOT here (needs firecrown 1.7.*, sacc 2.2.*, MPI/namaster/RAIL stack);
# it runs at NERSC via shifter or TXPipe/bin/perlmutter-install.sh. See notes/survey_integration_env.md.
set -euo pipefail
ENV_NAME="${1:-desc-mcp}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEP_GENESIS="${HEP_GENESIS_DIR:-/Users/nesar/Projects/Tutorials/HEPKE/hep-genesis-agent}"
CONDA="${CONDA_EXE:-conda}"   # prefer mamba/micromamba if present
command -v mamba >/dev/null && CONDA=mamba

if ! $CONDA env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  $CONDA create -n "$ENV_NAME" -y --override-channels -c conda-forge \
    "python=3.12" "numpy>=2.0,<2.4" "scipy>=1.13,<1.18" "camb<2.0" \
    "pyccl>=3.3.1" "sacc>=2.4" "firecrown>=1.16" "tjpcov>=0.5" "lsstdesc-ceci>=2.5" \
    "lsstdesc-augur>=1.2.3" "lsstdesc-crow" "qp-prob" "healpy" "numdifftools" "jinja2" \
    "cosmosis>=3.0" "cosmosis-build-standard-library" "compilers" \
    "matplotlib-base" "h5py" "astropy" "pyyaml" "rich" "typer" "pip"
fi
PREFIX="$($CONDA info --base)/envs/$ENV_NAME"; PIP="$PREFIX/bin/pip"; PY="$PREFIX/bin/python"

# pip-only pieces (cobaya/isitgr are firecrown connectors; derivkit is augur's optional Fisher engine)
$PIP install cobaya isitgr derivkit "mcp[cli]>=1.27,<2" "pydantic>=2,<3"

# Track the local clones' versions of augur (1.2.4 > conda 1.2.3); firecrown/pyccl stay conda-built.
$PIP install --no-deps -e "$REPO_DIR/augur"

# CosmoSIS standard library (needed only for sampling workflows W4/W5)
( source "$PREFIX/bin/cosmosis-configure"; cd "$PREFIX" && cosmosis-build-standard-library )
FIRECROWN_DIR="$($PY -c 'import firecrown,os;print(os.path.dirname(os.path.dirname(firecrown.__file__)))')"
$CONDA env config vars set -n "$ENV_NAME" \
  CSL_DIR="$PREFIX/cosmosis-standard-library" FIRECROWN_DIR="$FIRECROWN_DIR" AUGUR_DIR="$REPO_DIR/augur" \
  COSMOSIS_NO_SUBPROCESS=1

# HPC dispatch layer (optional, server-side mode only)
[ -d "$HEP_GENESIS/backend" ] && $PIP install -e "$HEP_GENESIS/backend[iri]"

# smoke test (run from outside the repo so firecrown/ and augur/ dirs are not shadowing)
(cd /tmp && $PY -c "import pyccl,sacc,firecrown,augur,tjpcov,ceci,cosmosis; \
  print(pyccl.__version__,sacc.__version__,firecrown.__version__,augur.__version__,tjpcov.__version__,ceci.__version__)")
```

Open risks to verify when the env is actually built: (a) `lsstdesc-augur` 1.2.3 vs sacc 2.4 behaviour
(DESC's own lock does this, so expected OK); (b) numcosmo on osx-arm64 (conda-forge has builds);
(c) `tjpcov` drags `mpi4py`+`namaster` — fine on arm64 per conda-forge, but if the solve fights, install
tjpcov with `pip install tjpcov` (minimal deps) instead; (d) CCL clone v3.3.6 == conda pyccl 3.3.6,
so no local build of CCL is needed (building from `CCL/` requires cmake, swig, gsl, fftw).

---

## .gitignore for the eventual server repo

```
# upstream DESC package clones (read-only, never modified here)
CCL/
CCLX/
TXPipe/
firecrown/
augur/
# shallow clones of integration/example repos
external/
# local junk
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
output/
*.sacc
*.fits
.DS_Store
```

(Keep `notes/`, `scripts/`, `skills/`, `mcp_server/`, `tools/` tracked.)
