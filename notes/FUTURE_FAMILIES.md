# Future tool families — LSSTDESC repositories that fit this server

Survey date: 2026-10-08 (GitHub org LSSTDESC: 488 public repos, 480 active,
8 archived; listing with stars / last push / language kept by the agent
session that produced this note). Added on 2026-10-09 from this list:
**TJPCov**, **NaMaster**, **Smokescreen** (families `tjpcov_*`, `namaster_*`,
`smokescreen_*`). Everything below is NOT built yet; it is the ranked
backlog for the next rounds, with what each would need.

## Tier 1 — same 3x2pt / likelihood family, cheap to add

| Repo | What it is | Why it fits | Notes for the build |
|---|---|---|---|
| `TJPCov` (done 2026-10-09) | covariance calculator | sacc contract | NaMaster-coupled types need pymaster < 3 (TJPCov 0.5.1 API); see ENVIRONMENT.md |
| `NaMaster` (done 2026-10-09) | pseudo-C_ell estimator | measurement side of TXPipe Fourier mode | catalog-based fields (`NmtFieldCatalog`), flat-sky fields and a native NaMaster Gaussian covariance tool (`NmtCovarianceWorkspace.from_fields` + CCL theory from the sacc n(z)) are the natural next steps — the latter would replace TJPCov's broken-under-NaMaster-3 `FourierGaussianNmt` |
| `Smokescreen` (done 2026-10-09) | data-vector concealment | firecrown-based | posterior-level concealment is "not yet developed" upstream; follow the repo |
| `CLMM` + `CLMMx` | cluster weak-lensing mass modelling (pip `clmm`, 32 stars, active) | new "clusters" family; already in the `desc-cosmology-env` lock | tools: profile models (NFW/Einasto/Hernquist, miscentering, boost), DeltaSigma / gamma_t predictions, stacked-profile fits; examples in CLMMx notebooks |
| `crow` (+ `crowkit`) | cluster observables (`pip install lsstdesc-crow`), firecrown cluster likelihood ingredients | listed as deferred in DESIGN.md; firecrown already imports crow | tools: cluster number counts / mass-richness integrals via firecrown's cluster likelihood; `crowkit` = Fisher manager over CROW + TJPCov + DerivKit |
| `CLPipe` | Cluster working group pipeline (active) | ships NERSC + IN2P3 shared envs (separate firecrown and TXPipe/TJPCov envs) | mostly orchestration; good source of `env_setup` candidates for cluster work |

## Tier 2 — forecasting cousins (cross-checks of augur, not new tools first)

| Repo | What | Use here |
|---|---|---|
| `dsf` | DeltaSigma galaxy-galaxy lensing forecasts (early development, active 2026-10) | compare with augur's 3x2pt Fisher for the ggl part once stable |
| `BAO_forecast` | BAO Fisher (2026-06) | reference numbers |
| `6x2pt_LSST_and_ext_Spec` | 6x2pt (LSST + DESI/4MOST) Fisher toolkit, writes sacc (78 MB) | cross-check; not cloned (size, overlaps augur/fisherA2Z) |
| `fisherA2Z` (cloned, `external/`) | standalone 3x2pt Fisher with photo-z systematics | a second Fisher engine for `augur_compute_fisher` validation |
| `forecasting_validation` | TT validation of CCL C_ell (206 MB notebooks) | validation data for `ccl_angular_cls` |

## Tier 3 — separate servers (different stacks, different data)

- `rail` family (`rail_base`, `rail_ccl`, ~30 `rail_*` repos): photo-z
  estimation and n(z) characterisation; heavy ML dependencies (pz-rail 1.1,
  pzflow, deep models). Would produce the `nz_csv` inputs this server
  consumes — a natural upstream server, not a family here.
- `imSim` + `skyCatalogs` (+ `skyCatalogs_creator`): GalSim image
  simulation; strong HPC fit, entirely different stack.
- Catalog access: `gcr-catalogs`, `descqa`, `lsstdesc-diffsky`,
  `Cat_NERSC_LSDB` — the data lives at NERSC; a facility-side data server.
- `dsigma`: galaxy-galaxy lensing measurement from catalogs (2026-07) —
  real-space counterpart of the NaMaster family; could join a
  "measurement" family with NaMaster catalog fields and TreeCorr.

## Tier 4 — reference material (skills, validation sets, env candidates)

- `minimal_mcp_pipelines` (cloned): sacc -> TJPCov -> firecrown ->
  CosmoSIS/nautilus MCMC on Perlmutter with job scripts — model for a
  second chain backend if Cobaya limits.
- `N5K`: the non-Limber benchmark — validation set for `ccl_angular_cls`
  non-Limber mode.
- `txpipe-reanalysis`, `nulltests_txpipe`: worked TXPipe analyses -> skills.
- `cluster_mor_firecrown_modeling`: firecrown cluster MOR example -> skill
  for the clusters family.
- `nersc` (docs repo), `desc-python`, `desc-cosmology-env` (cloned):
  `env_setup` candidates and allocation notes.
- `rubin_globus_compute`: Rubin pipelines at ALCF through Globus Compute —
  relevant to the dispatch design, not a tool.

## Not worth wrapping

- `CosmoAPI` ("Press Enter for Cosmology"): preliminary, no API surface yet.
- `firecrownx`: README only (examples moved into firecrown).
- `mgemu` (2023, f(R) boost emulator): duplicates the emulator server.

## Known upstream gaps found while building the three new families

- TJPCov 0.5.1 vs NaMaster 3: `FourierGaussianNmt` calls the NaMaster 2
  constructor; fix upstream or ship a native NaMaster covariance tool here.
- TJPCov real-space `CovarianceCalculator` path: interleaved xi_+/xi_-
  assumption and the unfilled xi_- auto block (worked around in
  `tools/inner/tjpcov_cov.py`); report upstream.
- Smokescreen embeds the seed in the concealed file's metadata
  (`seed_smokescreen`); flagged by `smokescreen_inspect`; discuss upstream.
