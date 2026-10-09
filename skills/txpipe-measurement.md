---
name: txpipe-measurement
description: Measure 3x2pt two-point functions with TXPipe - pick a preset, compose pipeline.yml + config.yml (txpipe_generate_pipeline), validate the DAG and dry-run commands (txpipe_validate_pipeline), run ceci in the user's TXPipe environment locally or as a facility job (txpipe_run_pipeline with env_setup), check status, and hand the resulting sacc to the likelihood skill
---

# TXPipe measurement recipe (compose -> validate -> run -> sacc)

TXPipe is never imported by this server. The tools read TXPipe's source for
stage metadata, write ceci YAML, and run `ceci pipeline.yml` in a SEPARATE
TXPipe environment: either a local one (server env var `DESC_TXPIPE_ENV`)
or one on a facility (per-call `env_setup`). Science products are sacc
files; everything downstream (inspection, covariance, firecrown) is the
`txpipe-sacc-to-likelihood` skill.

## Dispatch rule (read first)

- The server never knows the user's facility project, allocation, paths or
  software environment. For a facility run the CLIENT supplies
  `env_setup` - the snippet that activates the USER's TXPipe env and enters
  its checkout, e.g. `module load python; conda activate /path/to/TXPipe/conda; cd /path/to/TXPipe`
  (built once with `TXPipe/bin/perlmutter-install.sh`). Never invent one;
  if the user has none, say how to build it (TXPipe/bin/install.sh locally,
  perlmutter-install.sh at NERSC) and stop.
- Facility inputs are facility paths the USER gives in `inputs`
  (shear_catalog, photometry_catalog, ...). They will show as "missing
  locally" in validation - expected.
- Local runs need `DESC_TXPIPE_ENV` set in the server's environment (e.g.
  `source ~/TXPipe/conda/bin/activate; cd ~/TXPipe`). Without either,
  `txpipe_run_pipeline` refuses with an actionable message - relay it.

## Steps

1. **Choose a preset** with `txpipe_list_examples` / the table below, and
   confirm the catalog flavour (`catalog_type`):

   | preset | what it measures | stages | base example | runtime (1 deg^2 example) |
   |---|---|---|---|---|
   | `mock_shear` | smoke test: mock -> selection -> calibrated shear catalog (no 2pt) | 6 | examples/mock_shear | seconds-minutes |
   | `source_only` | cosmic shear xi+/xi- (TreeCorr), maps, PSF/Rowe tests | 28 | metadetect_source_only | ~5-10 min |
   | `3x2pt_real` | xi+/-, gamma_t, w(theta) + n(z) + theory + plots | 27 | metadetect/metacal/lensfit | ~5-20 min |
   | `3x2pt_fourier` | NaMaster C_ell (EE/BB, gE, gg) + theory | 28 | same, Fourier branch | ~10-30 min |
   | `custom` | your own `stages` list, config pulled from the base example | n | by catalog_type | - |

   For stage questions use `txpipe_describe_stage("TXTwoPoint")` (inputs,
   outputs, options with defaults: min_sep 0.5 / max_sep 300 arcmin / nbins 9).
   Example data: `txpipe_fetch_example_data(dest)` (347 MB, local only).

2. **Compose**: `txpipe_generate_pipeline(output_dir, preset, catalog_type,
   inputs={tag: path}, source_zbin_edges=[...], lens_zbin_edges=[...],
   nside, min_sep, max_sep, nbins | ell_min, ell_max, n_ell, threads,
   blinding="null"|"muir", include_covariance)`.
   - Simulations: `blinding="null"` (TXNullBlinding copies the raw sacc).
     Real data: `"muir"` (deletes the unblinded file).
   - `include_covariance=true` adds TXRealGaussianCovariance /
     TXFourierGaussianCovariance -> `summary_statistics_*.sacc` (data +
     covariance). It is slow; without it attach a covariance later with
     the sacc tools.
   - Report the stage list, `missing_input_files` and `main_stage_command`.
     The pipeline is always `site: local`, `launcher: mini`.

3. **Validate**: `txpipe_validate_pipeline(pipeline_yml)`. Require
   `ok: true` (no unknown stages, no unresolved input tags, no cycle).
   Show the topological `order` and the TXTwoPoint/TXTwoPointFourier
   command. If `DESC_TXPIPE_ENV` is set the real `ceci --dry-run` also runs;
   treat its failure as authoritative. Missing input files are only a
   problem for LOCAL runs.

4. **Run** (`txpipe_run_pipeline`, heavy):
   - Local: `txpipe_run_pipeline(output_dir, pipeline_yml, max_threads=<cores>, walltime_s=1800)`.
   - Facility: `set_dispatch("perlmutter"|"polaris")` (or the client-side
     pack: kernel `txpipe_run`, `function="envkernel.run_in_env"`), then
     `txpipe_run_pipeline(..., env_setup="<user's snippet>", max_threads=<node cores>, walltime_s=3600, nodes=1)`.
     The inner script writes the YAMLs into the job dir, clones TXPipe
     (depth 1) if `txpipe` is not importable there, sets
     `KMP_DUPLICATE_LIB_OK=TRUE` and `HDF5_USE_FILE_LOCKING=FALSE`, and runs
     `ceci pipeline.yml site.name=local site.max_threads=N`.
   - Expect minutes for the 1 deg^2 example, hours for the 20 deg^2
     cosmoDC2 set; `walltime_s` must cover the whole DAG. Re-run with
     `resume=true` (default) to continue after a timeout.
   - The result lists per-stage status, produced files (sacc/PNG/yml with
     sizes) and returns small sacc files in-band (decoded into
     `output_dir/txpipe_products_*`); HDF5 catalogs and maps stay where the
     pipeline ran (paths in the manifest). Verify `ran_on`/host names a
     compute node before saying it ran on the facility. If `env_check`
     shows `txpipe`/`ceci` as null, the env_setup did not activate a
     TXPipe environment - fix that first; do not blind-retry.

5. **Status / debugging**: `txpipe_run_status(manifest_json=...)` for a
   returned manifest, or `txpipe_run_status(run_output_dir, log_dir, pipeline_yml)`
   for a local run: `complete` / `inprogress` (an `inprogress_<tag>` file
   means the stage died mid-write) / `failed_or_incomplete` (log exists, no
   outputs - read its tail). Common causes: missing example data, MPI
   requested without mpirun (keep `allow_mpi=false`), submodules missing for
   TXConvergenceMaps/HOSFSB (not in the presets), NaMaster cache dir.

6. **Hand over**: the measurement is `twopoint_data_real.sacc`
   (`galaxy_shear_xi_plus/minus`, `galaxy_shearDensity_xi_t`,
   `galaxy_density_xi`; tracers `source_i`, `lens_j` with n(z)) or
   `twopoint_data_fourier.sacc` (`galaxy_shear_cl_ee/bb`, ...), plus
   `twopoint_theory_*.sacc` and, with covariance, `summary_statistics_*.sacc`.
   Run `sacc_inspect` on it and continue with the
   `txpipe-sacc-to-likelihood` skill (rename tracers, attach/check the
   covariance, build a firecrown likelihood).

## Sanity checks to quote

- 3x2pt_real with the example data: 4 source bins (edges 0.5-2.0) and 2-3
  lens bins; xi+ at ~10 arcmin of order 1e-5-1e-4 for the simulated DC2
  patch; gamma_t positive at small scales; the example is "too small to
  check any numerical results" - say so.
- A sacc without covariance cannot feed a likelihood; check
  `has_covariance` in `sacc_inspect` before promising constraints.

## Report

Preset + catalog type, number of stages and the order, where it ran (local
/ facility host), wall time, the per-stage status counts, the list of
sacc/PNG products with paths, and the one next call (`sacc_inspect` on the
measurement file).
