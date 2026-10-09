---
name: likelihood-sanity-checks
description: Before any chain - inspect the sacc, build the firecrown experiment, evaluate chi2/dof at the fiducial, check parameter completeness and unused-parameter warnings, then profile one parameter with firecrown_scan_loglike
---

# Likelihood sanity checks (sacc -> firecrown, no sampler)

Goal: establish in a few minutes that a sacc data vector + a firecrown
likelihood are consistent and that the parameter set is complete, BEFORE
spending hours on an MCMC. Everything runs locally (one likelihood
evaluation is ~0.4 s for a DES-Y1-sized 3x2pt). Use one `output_dir`.

## 1. Inspect the data

`sacc_inspect(output_dir, sacc_path)`. Read the metadata:
- `space` real or harmonic; `n_points`; data types and tracer pairs.
- `covariance.present` must be true and `covariance.positive_definite` true;
  `condition_number` of 1e9-1e10 is normal for 3x2pt with mixed units
  (xi+ ~1e-4, w(theta) ~1e-1); above ~1e14 expect numerical trouble.
- `tracer_names_ready`: src{i}/lens{i}. If false -> `sacc_prepare_for_firecrown`
  (TXPipe writes source_i/lens_i). `quantities_ready` false is fine for
  firecrown (augur needs it set).
- No covariance: harmonic -> `sacc_attach_gaussian_covariance`; real-space ->
  TJPCov/augur (this server cannot build xi covariances).

## 2. Build the experiment

`firecrown_build_likelihood(output_dir, sacc_path)` with the systematics you
intend to sample (defaults: WL MultiplicativeShearBias + PhotoZShift per bin,
LinearAlignment global; NC PhotoZShift per bin + the always-present
{lens}_bias). Add `scale_cuts` here, never by editing the sacc.
Read `metadata.required_parameters`: EVERY name the likelihood needs with its
firecrown default. Note firecrown's own defaults are not all "no systematic"
(`src{i}_mult_bias` = 1.0 would double the shear; `ia_bias` = 0.5); the tools
therefore substitute NEUTRAL values for unspecified mult_bias / ia_bias /
delta_z (0) and report them under "neutral defaults applied". `lens{i}_bias`
keeps firecrown's 1.5 and `z_piv` 0.5. Decide a value for each nuisance
parameter now (the DES-Y1 fiducials for the offline test file are in
`firecrown_list_examples().metadata.des_y1_fiducial`).

## 3. chi2 at the fiducial

`firecrown_compute_loglike(output_dir, experiment_yaml, cosmology, nuisance)`.
Checks, in order:
- `metadata.warnings`: "ignored parameters not used by this likelihood" means a
  typo or a systematic you did not enable (e.g. passing `lens0_alphaz` without
  LinearBiasSystematicFactory). Fix before continuing - firecrown would
  silently use the default otherwise. "neutral defaults applied" and "using
  firecrown defaults for" list what you did not set. A "transfer_function"
  warning means the experiment YAML was built with a different transfer
  function than the cosmology you passed: firecrown uses the YAML's; rebuild
  with `transfer_function=` matching if the data vector came from CCL.
- `chi2_per_dof`: a real data vector at a reasonable point gives 0.8-1.5
  (offline DES-Y1-like file at the example fiducials: chi2 = 552/457 = 1.2).
  >> 2: wrong nuisance values (mult_bias = 1.0?), wrong tracer order for
  gamma_t, theta/ell unit mismatch, or a covariance from a different data
  vector. << 0.5: covariance too large or the data IS the theory (a noiseless
  realization gives exactly 0).
- `per_statistic`: `chi2_block / n` per tracer pair; one pair far above the
  rest points at a bad bin (photo-z, bias) rather than cosmology.
  `max_abs_pull` > 5 flags single outliers. Look at the PNG.
- `loglike` = -chi2/2 exactly (ConstGaussian) - if not, the likelihood is not
  Gaussian (StudentT) and chi2 interpretation changes.

Closure test when in doubt: `firecrown_theory_data_vector(..., add_noise=False)`
writes a theory sacc; building a likelihood on it and evaluating at the same
point must give chi2 = 0 (and ~n_data with `add_noise=True`).

## 4. Profile before sampling

`firecrown_scan_loglike(output_dir, experiment_yaml, parameter="sigma8",
min_value, max_value, n_points=11, cosmology, nuisance)` (then Omega_c, then
the nuisance you worry about). Expect: a single smooth minimum near the
fiducial, `one_sigma_lower/upper` both found (else widen the grid), and
`parabolic_best_fit.sigma` of the right order (DES-Y1 3x2pt: sigma8 to
~0.01-0.03 conditional). A flat profile means the parameter is unconstrained
(fix it or add a prior); a minimum at the grid edge means your prior range
or fiducial is off. Only then run `firecrown_run_chain` - locally with
`max_samples` ~100-300 as a smoke test, on a facility (set_dispatch +
env_setup) for a converged posterior.

## Report

Quote: n_data, chi2 and chi2/dof at the fiducial, the 3 worst statistics by
chi2_block/n, every warning from step 3, and per scanned parameter the grid
minimum, the parabolic best fit +/- sigma and the 1-sigma crossings. Attach
theory_vs_data and scan PNGs. State the systematics model and the fixed
nuisance values explicitly - they define what the numbers mean.
