# desc-mcp-server — design synthesis (step 1 output)

Date: 2026-10-07. Inputs: the five surveys in this directory
(`survey_ccl.md`, `survey_txpipe.md`, `survey_firecrown.md`, `survey_augur.md`,
`survey_integration_env.md`) and the template server
`~/Projects/Tutorials/cosmic_emulator_server` (branch `dispatch`).

Constraints (from `instructions.txt`): the four package clones are read-only
references; all code lives in this repo; only LSSTDESC-org repos are cloned;
follow the template's conventions (they are matched by name in hep-genesis).

---

## 1. Decisions

| # | Decision | Reason (survey evidence) |
|---|---|---|
| D1 | **Two execution environments, not one.** Light conda env `desc-mcp` (pyccl 3.3.6, sacc 2.4, firecrown 1.16, augur 1.2.4, TJPCov 0.5.1, ceci) runs the server and every CCL/firecrown/augur tool. TXPipe is never installed into it. | TXPipe pins firecrown 1.7 (its theory/blinding stages import `firecrown.likelihood.gauss_family`, deleted in 1.10); augur needs firecrown >= 1.14. No version satisfies both. DESC's own `desc-cosmology-env` lock validates the light combination. |
| D2 | **TXPipe tools are "compose + dispatch", never "import txpipe".** The server parses stage metadata from the clone's source (AST) and ceci YAML, writes `pipeline.yml` + `config.yml`, and runs `ceci` either in an optional local TXPipe env or as a facility job (env-kernel, D3). The TXPipe python package (127 files, 1.3 MB text) and ceci (28 files) travel **inside the dispatch pack**, so no TXPipe checkout is required on the facility; only its compiled dependencies must exist in the user-chosen facility environment. | TXPipe's env is ~60 conda pkgs incl. NaMaster, TreeCorr, MPI h5py, RAIL; not pip-installable; README says arm64 unsupported. All stage classes expose `inputs/outputs/config_options` as class attributes, so introspection does not need the heavy deps. Pack limits are 200 files / 10 MB. |
| D3 | **Two dispatch kernel styles, both through the hep-genesis engine (IRI + Globus), never SLURM scripts or SSH.** (a) *pip-kernel*: pure-pyccl kernels ship with `pip_deps=["pyccl","camb"]` exactly like the template. (b) *env-kernel*: a stdlib-only kernel that takes an `env_setup` shell snippet **as a kernel argument supplied by the client**, writes an inner script + params, runs `bash -lc "<env_setup>; python inner.py"` on the compute node and returns the JSON it wrote. Used for firecrown, augur, TJPCov and TXPipe jobs. | firecrown and numcosmo are conda-forge only (not on PyPI), so the engine's venv+pip bootstrap cannot build them. The engine runs the facility's module python as-is when `pip_deps=None`, which is enough to launch a subprocess under a facility-resident environment. Which environment is the user's choice (their collaboration env, their own conda env, a container); the server only ships documented *candidates* in the pack manifest. |
| D3b | **The server holds no facility identity or configuration.** No NERSC/ALCF project, username, workdir, token, Globus endpoint, environment path or allocation appears in server code or server config. These belong to the client's hep-genesis harness (its facility servers / sidecar), exactly as in the template and the spectra/gaia servers. The same rule applies to the facility-side software environment: it is a per-call argument. | This is the fixed client-server model for all the HEP-KE servers; a hosted server serves many users with different projects. |
| D4 | **sacc is the data contract between families.** Every family reads/writes sacc files; a `sacc` tool family (inspect, convert, rename tracers, attach covariance) sits between them. | TXPipe -> sacc -> firecrown/augur is the only cross-package path that exists upstream (W1–W5 in the integration survey). TXPipe writes `source_i/lens_i` without `quantity`; augur requires `src{i}/lens{i}` with quantities and a covariance. |
| D5 | **Units at the tool boundary: redshift z, k in h/Mpc, P(k) in (Mpc/h)^3, theta in arcmin, ell integer.** Columns are labelled; a `k_units` switch exposes CCL's native 1/Mpc. | CCL is h-free Mpc everywhere and takes scale factor a; the template server and emulators use h/Mpc. Mixed conventions are the #1 way to get a silent factor of h^3. |
| D6 | **Parameter names are firecrown's.** Cosmology: `Omega_c, Omega_b, h, n_s, sigma8|A_s, Omega_k, Neff, m_nu, w0, wa`. Nuisance: `{tracer}_{param}` (`src0_delta_z`, `src0_mult_bias`, `lens0_bias`, `lens0_delta_z`, `lens0_sigma_z`) and global IA `ia_bias, alphaz, z_piv`. A `convert_cosmology_names` tool maps to Cobaya/CosmoSIS/emulator-server names. | firecrown and augur share these exactly; CCL uses the same cosmology keys. |
| D7 | **Skills encode the cross-package pipelines**; tools stay single-purpose. | Mirrors the template. The pipelines (forecast, TXPipe-to-likelihood, CCL-vs-emulator) are multi-tool by nature. |

---

## 2. Repository layout (mirrors the template)

```
desc-mcp-server/
  pyproject.toml            [tool.mcp-server] tool_modules list; deps: mcp[cli], pydantic, numpy, scipy, matplotlib, pyyaml
  README.md                 quick start, tool table, skills, hosting, HPC, EXAMPLE QUERIES
  ENVIRONMENT.md            how desc-mcp is built; version pins and why; TXPipe/NERSC env notes
  scripts/setup_env.sh      creates conda env desc-mcp (conda-forge, override-channels)
  mcp_server/               __init__, __main__, cli.py, server.py, dispatch.py  (copied/adapted from template)
  tools/
    common.py               ArtifactResult, write_csv/read_csv, get_cached, param_slug, summary_stats (template) + z<->a, h-unit helpers
    plotting.py             template plotting + new quantity axes (C_ell, xi(theta), n(z), Fisher ellipses)
    meta/                   list_desc_packages, describe_desc_tool_family, list_desc_skills, load_desc_skill, convert_cosmology_names
    ccl/                    __init__.py (wrappers) + kernels.py (pip-kernel) + srd_nz.py (pure-numpy SRD n(z))
    sacc_tools/             inspect/convert/rename/attach-covariance (package name avoids shadowing `sacc`)
    firecrown_tools/        __init__.py (wrappers) + kernels.py (env-kernel inner scripts)
    augur_tools/            __init__.py + kernels.py
    txpipe_tools/           __init__.py + stage_index.py (AST index of the clone) + pipeline_compose.py + kernels.py (ceci job wrapper)
    envkernel.py            the generic env-kernel runner (stdlib only) shared by firecrown/augur/txpipe
  skills/                   *.md with name/description frontmatter
  tests/                    smoke_env.py, test_tools.py, test_dispatch_pack.py, smoke_server.py
  notes/                    surveys + this file
  external/                 shallow LSSTDESC clones (gitignored)
  CCL/ CCLX/ TXPipe/ firecrown/ augur/   upstream clones (gitignored, read-only)
```

Clone paths are configuration, not code: `DESC_CCLX_DIR`, `DESC_TXPIPE_DIR`,
`DESC_AUGUR_DIR`, `DESC_FIRECROWN_DIR` env vars default to the sibling directories.
Tools that read them (TXPipe stage index, augur SRD n(z) files, example configs) say
so in metadata and never write into them. These are the ONLY server-side settings
beyond the template's `MCP_PUBLIC` / `MCP_OUTPUT_ROOT` / `MCP_ARTIFACT_URL`; nothing
facility-related is configured on the server (D3b).

---

## 3. Tool families (first release ~32 tools)

Weight: L = light (ms–s, runs anywhere the light env exists); M = seconds–minute,
dispatchable; H = heavy, dispatch recommended.

### meta (5, L)
`list_desc_packages` (versions, status, which families they back), `describe_desc_tool_family`,
`list_desc_skills`, `load_desc_skill`, `convert_cosmology_names(params, to=ccl|cobaya|cosmosis|emulator)`.

### ccl (10) — pyccl 3.3.6, kernels pip-dispatchable (`pyccl`, `camb`)
| Tool | W | Writes |
|---|---|---|
| `ccl_describe_cosmology` | L | derived params (Omega_m, sigma8<->A_s, Omega_nu), YAML when serialisable |
| `ccl_background` | L | CSV: z, chi, D_A, D_L, mu, E(z), D(z), f(z), t_lookback |
| `ccl_matter_pk` | L | CSV: k, P_lin, P_nl (+ratio); `matter_power_spectrum` in {halofit, linear, camb_hmcode, cosmicemu_mt4, bacco}; `transfer_function` in {boltzmann_camb, bbks, eisenstein_hu, boltzmann_class}; baryons {none, schneider15, vandaalen19, bacco}; mu-Sigma MG |
| `ccl_lsst_srd_nz` | L | CSV n(z) per tomographic bin for Y1/Y4/Y7/Y10 lens/source (pure numpy re-implementation of CCLX `lsst_galaxy_sample.py`; also usable as sacc NZ tracers) |
| `ccl_angular_cls` | L/M | CSV: ell + one column per tracer pair; tracer specs inline (wl/nc/cmb_lensing/tsz/cib/isw, n(z) from file or SRD, bias, IA, mag bias); Limber or FKEM non-Limber |
| `ccl_correlation_functions` | L | CSV: theta_arcmin, xi+/xi-/gamma_t/w per pair (fftlog) |
| `ccl_correlation_3d` | L | CSV: r, xi_0/2/4 |
| `ccl_halo_mass_function` | L | CSV: M, dn/dlog10M, b(M), c(M), sigma(M); 11 HMFs, 4 biases, 7 c(M) |
| `ccl_halo_model_pk` | M | CSV: k, P_1h, P_2h, P_hm (NFW or HOD) |
| `ccl_baryon_boost` | L | CSV: k, P_bar/P_dmo per model — directly comparable with the emulator server's `compute_baryon_suppression` |

### sacc_tools (4, L) — `sacc` 2.4 only
`inspect_sacc` (tracers, data types, pairs, ell/theta ranges, covariance presence, n(z) CSVs),
`sacc_to_csv`, `prepare_sacc_for_firecrown` (rename `source_i`->`src{i}`, `lens_i`->`lens{i}`,
set `quantity`, keep only requested data types, check covariance; writes a new file, never edits input),
`attach_gaussian_covariance` (mode-counting Gaussian with f_sky, sigma_e, n_gal — augur's `gaus_internal`
formula — for sacc files that lack a covariance, e.g. TXPipe `twopoint_data_*.sacc`).

### firecrown_tools (6) — firecrown 1.16 factory API, env-kernel dispatch
| Tool | W | Notes |
|---|---|---|
| `list_firecrown_examples` | L | the 4 CLI examples + offline test sacc; parameter names |
| `build_firecrown_likelihood` | L | spec (sacc path, harmonic/real, per-bin & global systematics by factory type, scale cuts, CCL mode) -> validated `experiment.yaml` + required parameter list with defaults |
| `compute_loglike` | L | loglike, chi2, per-statistic chi2, theory-vs-data CSV; casts ints to float; full update/prepare/reset cycle |
| `compute_theory_data_vector` | L/M | theory C_ell/xi for a sacc (with or without data) + optional noiseless/noisy realization sacc via `make_realization` |
| `scan_loglike` | M | 1-D (or 2-D grid) profile; CSV + PNG |
| `run_firecrown_chain` | H | Cobaya MCMC in PURE_CCL mode (no theory block); dispatch; returns chain + summary |

### augur_tools (6) — augur 1.2.4, env-kernel dispatch
`list_augur_examples`, `generate_forecast_config` (Y1/Y10, probes, bins, ndens, sigma_e, ell binning,
kmax, cov_type gaus_internal|SRD|tjpcov, var_pars, step, priors -> absolute-path YAML, no Jinja),
`validate_forecast_config` (catches the known traps: priors not in var_pars, sigma8 xor A_s,
kmax xor lmax, SRD cov vs bin count), `generate_synthetic_datavector` (M; sacc + CSV + plot),
`compute_fisher` (H; Fisher CSV, marginalised sigmas, FoM json, derivatives), `plot_fisher_contours` (L).

### txpipe_tools (8) — compose + dispatch
`list_txpipe_stages` (L, AST index of `TXPipe/txpipe`), `describe_txpipe_stage` (L: inputs/outputs/
config_options with dtype/default/required), `list_txpipe_examples` (L), `generate_txpipe_pipeline`
(L: preset `mock_shear|source_only|3x2pt_real|3x2pt_fourier` or explicit stage list; inputs; z-bin edges;
nside; 2pt binning; threads -> `pipeline.yml` + `config.yml` + dependency check), `validate_txpipe_pipeline`
(L if ceci installed: `ceci --dry-run` command list), `run_txpipe_pipeline` (H: local TXPipe env if
`DESC_TXPIPE_ENV` set, else env-kernel job via the dispatch engine running `ceci pipeline.yml site.name=local`
on the compute node with `max_threads` = node cores; one node, single allocation; returns sacc/PNG/yml
artifacts in-band and leaves large HDF5 outputs in the facility job directory, reporting their paths),
`txpipe_run_status` (L: final vs `inprogress_*` outputs, log tails, from a returned results manifest),
`fetch_txpipe_example_data` (L I/O: the 347 MB example tarball, version-pinned).
The pipeline always uses ceci's `local` site inside the job, so ceci never issues scheduler commands
itself; multi-node TXPipe runs are out of scope for the first release.

### dispatch (4) — verbatim template contract
`set_dispatch(site, artifact_dir)`, `get_dispatch`, `auth_status`, `export_dispatch_pack`.
The pack manifest gains, per env-kernel, an `env_setup_required: true` flag and a list of
`env_setup_candidates` (documented, public DESC environment activation snippets the user may pick
from or replace with their own). The server never selects one.

Deferred to a later release (documented, not built): CCL perturbation theory (needs fast-pt),
cluster number counts via crow, firecrown CosmoSIS/NumCosmo runners, TXPipe single-stage runner,
TJPCov standalone covariance tool (augur's `cov_type: tjpcov` covers the forecasting need).

---

## 4. HPC execution model

Everything below is the template's model (`cosmic_emulator_server/mcp_server/dispatch.py`,
hep-genesis `make-tools-dispatchable` contract R1–R5), unchanged. Access to ALCF Polaris and
NERSC Perlmutter is **exclusively through the hep-genesis engine**: job submission over the IRI
compute API, payload staging over Globus Transfer (or IRI `filesystem/upload` on NERSC), results
read back over Globus or IRI `filesystem/view`. The server writes no batch scripts and opens no
SSH sessions; the scheduler wrapper (`run.sh`) is generated by the engine, not by this server.

```
agent ──MCP──> desc-mcp-server (light env) ──┬── local: tools run in-process
                                             └── remote: the dispatch engine stages tools/ + runner.py + params.json
                                                   ├─ pip-kernel  (tools/ccl/kernels.py)   pip_deps=[pyccl, camb]  (template style)
                                                   └─ env-kernel  (tools/envkernel.py)      pip_deps=None
                                                        args: {env_setup: "<client-supplied shell snippet>",
                                                               inner: "<family>.<function>", params: {...}}
                                                        on the node: bash -lc "<env_setup>; python inner.py"
                                                        returns: the JSON the inner script wrote (results.json)
```

**Who provides what**

| Item | Provided by | Never by |
|---|---|---|
| Facility choice (`polaris` / `perlmutter`) | client (`set_dispatch` on its facility server, or the server's own `set_dispatch` in server-side mode) | server defaults |
| IRI token, Globus transfer token, Globus endpoint | client's hep-genesis sign-in (`~/.globus` on the client machine) | server |
| Project / allocation, workdir, QoS, queue | client's hep-genesis `.env` (`NERSC_PROJECT`, `NERSC_WORKDIR`, …) | server code or config |
| Facility-side software environment (`env_setup`) | client, as a per-call kernel argument; the pack manifest lists public candidates (e.g. the DESC collaboration env activation script, a `desc-python` kernel, a user's own conda env) | server |
| Kernels, inner scripts, TXPipe + ceci python sources | this server's `tools/` pack (`export_dispatch_pack`) | — |

**Two dispatch modes (as in the template)**

- *Client-side (hosted deployments, recommended)*: `export_dispatch_pack` hands over `tools/` plus a
  manifest; the client's facility server runs `run_pack_kernel(pack=..., function=..., args=...,
  pip_deps=...)` under the user's credentials. A VM deployment needs no hep-genesis install.
- *Server-side (server on the user's own machine)*: `set_dispatch("perlmutter")` routes the heavy tools
  through `run_codes_on_perlmutter(function, args, codes=TOOLS_DIR, pip_deps, duration)` in-process;
  `auth_status` reports the user's sign-in. Facility settings still come from the user's hep-genesis
  `.env`, which the engine loads; the server passes nothing of its own.

**env-kernel contract** (`tools/envkernel.py`, stdlib only so it runs under any module python)

- Arguments: `env_setup` (str, required; the tool raises a clear "this call needs a facility
  environment — pass env_setup or pick one of the manifest candidates" error before dispatch if missing),
  `inner` (dotted name of an inner script inside `tools/`), `params` (JSON-safe dict).
- Behaviour: write `params.json` and the inner script into the job CWD, run
  `bash -lc "<env_setup>; python <inner>.py"`, capture stdout/stderr tails, read `inner_result.json`,
  return `{"result": ..., "stdout_tail": ..., "stderr_tail": ..., "env_check": {python, firecrown,
  pyccl, sacc versions}}`. A failure returns a string beginning with `Error` per the engine's convention.
- Inner scripts only use firecrown/augur APIs that exist across 1.14–1.16 (the facility env may be older
  than the server's), and write every result as JSON; sacc files they produce are written in the job CWD
  so the engine returns them as `artifact_files` when the staging path supports binary fetch-back (Globus),
  and as base64 inside the JSON when it does not (Globus-free IRI path) and they are small.
- TXPipe inner script: unpack the shipped `txpipe/` + `ceci/` sources onto `sys.path`, write
  `pipeline.yml`/`config.yml`, run `ceci pipeline.yml site.name=local site.max_threads=<n>` as a subprocess,
  collect sacc/PNG/yml outputs and a per-stage status manifest. Inputs (catalogs) are referenced by
  facility paths the user supplies; the server never assumes where data lives.

**Cheap checks before dispatch** (template rule): argument validation, YAML validation, stage
dependency checks and `ceci --dry-run` all run on the server in milliseconds so a bad request never
costs queue time.

---

## 5. Server-side skills (first set)

| Skill | Tools chained | What it answers |
|---|---|---|
| `desc-tour` | one call per family | "show me what this server can do" |
| `lsst-3x2pt-forecast` | ccl_lsst_srd_nz -> augur generate_forecast_config -> validate -> generate_synthetic_datavector -> compute_fisher -> plot_fisher_contours | "Fisher forecast for LSST Y1/Y10 3x2pt; how do w0-wa constraints change if ..." |
| `txpipe-to-likelihood` | generate_txpipe_pipeline -> run (dispatch) -> inspect_sacc -> prepare_sacc_for_firecrown -> attach_gaussian_covariance -> build_firecrown_likelihood -> compute_loglike / scan | "measure 2pt functions from this catalog and evaluate the likelihood at Planck" |
| `ccl-datavector-to-sacc` | ccl_lsst_srd_nz -> ccl_angular_cls -> (sacc write) -> inspect_sacc -> build_firecrown_likelihood | the CCLX `generate_dv_nonlimber` recipe as tools; also "compare Limber vs non-Limber" |
| `pk-ccl-vs-emulators` | ccl_matter_pk (several backends) + the emulator server's compute_nonlinear_pk on the same h/Mpc grid | cross-server P(k) consistency, honest spread |
| `hpc-dispatch-handoff` | export_dispatch_pack -> client run_pack_kernel | template skill adapted: which kernels are pip-kernels vs env-kernels, how the user picks `env_setup`, verifying the `host` field |
| `likelihood-sanity-checks` | inspect_sacc -> build -> compute_loglike at fiducial -> scan_loglike | chi2/dof at fiducial, parameter completeness, unused-parameter warnings before any chain |

---

## 6. Environment plan

`scripts/setup_env.sh` creates `desc-mcp` (python 3.12, conda-forge with `--override-channels`):
`pyccl>=3.3.1 sacc>=2.4 firecrown>=1.16 lsstdesc-ceci tjpcov lsstdesc-crow qp-prob healpy numdifftools
jinja2 "camb<2" "numpy>=2,<2.4" matplotlib h5py astropy pyyaml` + pip `mcp[cli] pydantic cobaya derivkit`
+ `pip install --no-deps -e augur` (clone, 1.2.4) + optional `pip install -e <hep-genesis>/backend[iri]`.
TXPipe locally is optional and separate (`conda env create -p ~/envs/txpipe -f TXPipe/bin/environment-local.yml`),
pointed to by `DESC_TXPIPE_ENV`. Open risks to verify when building: numcosmo on osx-arm64, tjpcov pulling
namaster+mpi4py (fallback `pip install tjpcov`), augur 1.2.4 vs sacc 2.4 (DESC's lock says fine).

Facility environments are not part of this plan: the server documents candidates in the pack manifest
and in `ENVIRONMENT.md` (what each needs to contain: pyccl, sacc, firecrown >= 1.14 for the likelihood
kernels; the TXPipe compiled stack for pipeline kernels), and the user chooses per call.

---

## 7. README example queries (to test from hep-genesis-agent and Claude Desktop)

1. "Using CCL, compute the comoving distance, angular diameter distance and growth factor from z=0 to 3 for Planck 2018 and plot them."
2. "Compare the nonlinear matter power spectrum at z=0.5 from CCL halofit, CCL HMcode2020 and CosmicEmu on the same k grid in h/Mpc, and report the maximum fractional difference below k=1 h/Mpc."
3. "Generate the LSST Y1 SRD source and lens n(z) bins, compute all 3x2pt angular power spectra for ell 20–2000, and write them to a sacc file."
4. "Run an LSST Y1 3x2pt Fisher forecast with augur with the SRD covariance, varying Omega_c, sigma8, w0, wa, n_s, h and the lens biases; give the marginalised 1-sigma errors and the w0-wa figure of merit, and plot the contours."
5. "Repeat the forecast for Y10 and tell me by how much the w0-wa FoM improves."
6. "Inspect this TXPipe sacc file: list the tracers, data types and whether it has a covariance; then prepare it for firecrown, attach a Gaussian covariance with f_sky=0.1, sigma_e=0.26 and n_gal=10 per arcmin^2, and compute the log-likelihood at the vanilla LCDM cosmology."
7. "Scan the log-likelihood in sigma8 between 0.7 and 0.9 for the DES Y1 3x2pt test data vector with fiducial nuisance parameters, and report the 1-sigma interval."
8. "Build a TXPipe pipeline for the metadetect example data that produces real-space 3x2pt measurements, show me the stage list and the dry-run commands, and then run it on Perlmutter."
9. "Describe the TXTwoPoint stage: its inputs, outputs and configuration options with defaults."
10. "Set dispatch to Perlmutter and run the Y10 Fisher forecast there; confirm the host it ran on."

---

## 8. Resolutions (user answers of 2026-10-07, applied to the plan)

- **`env_setup` default candidates.** The env-kernel's `env_setup` argument becomes optional with a
  documented default per facility, overridable per call:
  - NERSC: the DESC `desc-python` stack, `source /global/common/software/lsst/common/miniconda/setup_current_python.sh`
    (lock of 2026-09-01: firecrown 1.15.2, pyccl 3.3.6, sacc 2.4, tjpcov 0.5.1, numcosmo 0.27). This is a
    public collaboration path, not a user identity, so it may ship as a default candidate. The
    `desc-cosmology` stack (`source $CFS/lsst/groups/MCP/setup-cosmology.sh`) is the second candidate.
  - ALCF: no DESC-maintained stack exists. The manifest's candidate is "a user-built env from
    `external/desc-cosmology-env/slac/env-nobuild-*.yml` (or the `desc-python` hpc lock) under the user's
    own Polaris area, activated with `source <path>/bin/activate`"; `ENVIRONMENT.md` carries the recipe.
    The user supplies the resulting `env_setup` string.
  - TXPipe kernels have no default: the user names their TXPipe env (or container) per call.
  All inner scripts therefore target firecrown >= 1.14 APIs (desc-python is at 1.15.2).
- **Example data**: the hosted deployment does not carry the TXPipe example tarball; `fetch_txpipe_example_data`
  stays for local use.
- **augur**: installed from the clone (1.2.4), from a temporary copy so the clone directory stays pristine
  (pip would otherwise write `augur.egg-info` into it).
- **Dispatch phrasing**: hep-genesis-agent routes "set dispatch to Perlmutter" to the right server; the
  README shows the hosted and the server-side phrasing.
- **"Compose + dispatch, never import txpipe" clarified**: nothing is lost. TXPipe cannot live in the
  server's Python process (its firecrown 1.7 pin and compiled MPI/NaMaster/TreeCorr stack cannot coexist
  with firecrown 1.16 and augur). So the server (1) reads TXPipe's source files as text to learn every
  stage's inputs, outputs and options, (2) writes the ceci pipeline and config YAML, and (3) runs the real
  TXPipe as a separate process: in a local TXPipe conda env if the user has one, or on a facility node
  under the user's TXPipe environment via the dispatch engine. Full pipelines run unchanged; they just
  run outside the server's interpreter, exactly as `ceci pipeline.yml` would from a shell.

## 8a. Open questions (historical, now answered above)

- `env_setup` ergonomics: is a per-call string argument acceptable, or should the client's hep-genesis
  harness learn a per-user "facility environment" setting it injects automatically (a client-side change,
  outside this server)? The server design works with either; the first release takes the argument.
  Answer: Best use desc-python on NERSC (NERSC: source /global/common/software/lsst/common/miniconda/setup_current_python.sh) ..details given here: https://github.com/LSSTDESC/desc-python. We may not have a similar one on ALCF but maybe this repo itself is good? 
- Should the hosted deployment carry the 347 MB TXPipe example data pre-fetched? 
- Answer: not needed to carry the examples to deployment. 
- Keep augur at the clone's 1.2.4 (editable, no-deps) or the conda-forge 1.2.3? Default: clone.
Answer: clone 
- Example query 10 in section 7 says "set dispatch to Perlmutter"; in a hosted setting that call goes to
  the client's facility server, not this one. The README will show both phrasings.
  Answer: OK. The hep-genesis-agent should be able to handle this. 
Also I didnt fully understnad that **TXPipe tools are "compose + dispatch", never "import txpipe".. Hopefully it's not an issue.**
