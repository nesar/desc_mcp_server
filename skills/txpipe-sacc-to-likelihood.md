---
name: txpipe-sacc-to-likelihood
description: Turn a TXPipe two-point sacc file (tracers source_i/lens_i, often no covariance) into a firecrown likelihood - inspect, rename/relabel tracers, attach a Gaussian covariance for harmonic data (or route real-space to TJPCov/augur), build the experiment, evaluate and profile the log-likelihood
---

# TXPipe sacc -> firecrown likelihood (the data half of the pipeline)

The measurement half is `txpipe_run_pipeline` (txpipe family): it runs ceci
on a facility and returns `twopoint_data_fourier.sacc` (C_ell) and/or
`twopoint_data_real.sacc` (xi) plus a `summary_statistics` sacc. This skill
starts from such a file (or any sacc written outside firecrown) and ends
with a validated likelihood evaluation. All steps are local and quick.

## 1. Inspect

`sacc_inspect(output_dir, sacc_path)`. TXPipe files look like:
- tracers `source_0..n` and `lens_0..m` (NZ), `quantity = generic`;
- harmonic: `galaxy_shear_cl_ee` plus `galaxy_shear_cl_bb` / `_eb` B-mode
  null tests, `galaxy_shearDensity_cl_e` (+ `_b`), `galaxy_density_cl`,
  bandpower windows in the tags; real: `galaxy_shear_xi_plus/minus`,
  `galaxy_shearDensity_xi_t`, `galaxy_density_xi` (theta in arcmin);
- `covariance.present` false unless the TXTwoPointFourier/TXPipe covariance
  stages ran (then it is present and you skip step 3).
Note `inferred_kind` per tracer (shear/density, from the data types) and the
ell/theta ranges.

## 2. Prepare for firecrown

`sacc_prepare_for_firecrown(output_dir, sacc_path,
keep_data_types=["galaxy_shear_cl_ee", "galaxy_shearDensity_cl_e", "galaxy_density_cl"])`
(real: `[..._xi_plus, ..._xi_minus, ..._xi_t, galaxy_density_xi]`).
It renames `source_i -> src{i}`, `lens_i -> lens{i}`, sets quantities, drops
B-modes and other null-test types (the covariance, if any, is sliced), and
writes a NEW file. Check `metadata.renamed` and `quantities`; use
`tracer_map` for non-standard names. `metadata.next_step` tells you whether
a covariance is still needed.

## 3. Covariance

- Harmonic, no covariance: `sacc_attach_gaussian_covariance(output_dir,
  prepared_sacc, f_sky, n_gal, sigma_e=0.26, galaxy_bias, cosmology)`.
  Inputs you must know from the TXPipe run: `f_sky` (footprint area / 41253
  deg^2 - the TXPipe example data covers ~1e-3 to 1e-2; LSST Y1 ~0.3),
  `n_gal` per tracer in arcmin^-2 (TXPipe's tomography/summary stages
  report effective densities; the SRD Y1 total is ~10 for sources, ~18 for
  lenses), `galaxy_bias` per lens bin (1.2-2 for LSST-like lenses). The
  result is a disconnected Gaussian covariance (augur's gaus_internal);
  say so in the report. Expect `snr_total` of tens to hundreds for a
  survey-sized footprint, less for the example data.
- Real-space, no covariance: this server cannot build one. Options: run
  TXPipe's covariance stages (TXTwoPointTheoryReal + TXFourierGaussianCovariance
  / TXRealGaussianCovariance via `txpipe_run_pipeline`), TJPCov directly, or
  augur (`augur_generate_forecast_config` with `cov_type: tjpcov`) on a
  synthetic vector with the same binning. Then come back to step 4.
- Covariance present: re-run `sacc_inspect` on the prepared file and confirm
  `positive_definite`.

## 4. Build and evaluate

`firecrown_build_likelihood(output_dir, prepared_sacc)` (correlation_space is
auto-detected; add `scale_cuts` for ell_max / theta_min; e.g. keep
`ell` in [20, 2000] for shear, [20, 600] for density if no k_max cut was
applied in TXPipe). Then `firecrown_compute_loglike(output_dir,
experiment_yaml, cosmology, nuisance)` with explicit nuisance values
(mult_bias ~0, delta_z 0, lens biases as used for the covariance). For a
TXPipe mock (e.g. the CosmoDC2 example data) the cosmology of the simulation
should give chi2/dof ~ 1 with a matched covariance; a strongly larger value
usually means a wrong `n_gal`/`f_sky` (covariance too small) or a bias
mismatch. Follow `likelihood-sanity-checks` for the per-statistic and scan
checks, then `firecrown_scan_loglike` on sigma8 and Omega_c.

## 5. Report

Give the file lineage (TXPipe sacc -> prepared -> with covariance ->
experiment YAML), the tracer renaming and dropped data types, the covariance
inputs (f_sky, n_gal, sigma_e, bias, cosmology) and its caveats, n_data after
cuts, chi2/dof at the evaluated point with the warnings, and the scan
summary. Keep the prepared sacc and the experiment YAML: they are the inputs
for `firecrown_run_chain` and for augur forecasts.
