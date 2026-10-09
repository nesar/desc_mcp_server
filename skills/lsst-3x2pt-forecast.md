---
name: lsst-3x2pt-forecast
description: Fisher forecast for LSST Y1 or Y10 3x2pt (or shear / clustering / GGL) with augur - generate a validated config, synthetic data vector, Fisher matrix with priors, marginalised sigmas, w0-wa FoM, contour plots; compare Y1 vs Y10 or two model choices
---

# LSST 3x2pt Fisher forecast (augur)

Use when the user asks for forecast constraints ("how well will LSST Y1/Y10
measure w0, wa, sigma8 ...", "FoM", "what if we add priors / drop GGL /
change kmax"). Everything is augur 1.2.4 on firecrown + CCL; parameter names
are firecrown's (Omega_c, Omega_b, h, n_s, sigma8|A_s, w0, wa, m_nu;
nuisance lens{i}_bias, src{i}_delta_z, src{i}_mult_bias, ia_bias, alphaz).
Units: ndens arcmin^-2, kmax Mpc^-1 (no h), ells dimensionless, step ABSOLUTE.
Use one `output_dir` for the whole forecast.

## Steps

1. **(optional) `augur_list_examples`** - shows the shipped SRD Y1/Y10 YAMLs,
   their encodings and that they do NOT run as shipped (priors listed for
   parameters they do not vary). Do not copy them; generate instead.
2. **`augur_generate_forecast_config`** - presets:
   - Y1: `survey_year="Y1", probes="3x2pt"` -> 5 source + 5 lens bins, ndens
     10/18 arcmin^-2, sigma_e 0.26, 20 ells in 20..15000, kmax 0.202 Mpc^-1
     on GGL + clustering, SRD lens biases 1.56..2.29, IA (5.717, -0.47, 0.3),
     eisenstein_hu + halofit at the SRD cosmology (Omega_c 0.2664, Omega_b
     0.0492, h 0.6727, n_s 0.9645, sigma8 0.831).
   - Y10: `survey_year="Y10"` -> 5 + 10 bins, ndens 27/48, kmax 0.201, 10 lens biases.
   - `cov_type="SRD"` when the layout is the SRD one (default bins, 3x2pt, 20
     ells; metadata `srd_covariance_layout_match` tells you), otherwise
     `gaus_internal` with `fsky` (0.3 Y1 / 0.4 Y10 by default). `tjpcov` is slow.
   - `var_pars`: default 7 cosmology + 2 lens biases. For the SRD-like
     question use `["Omega_c","sigma8","n_s","w0","wa","Omega_b","h"]` plus
     nuisance as needed; `srd_priors=True` adds the SRD prior widths for
     exactly the varied parameters (avoids augur's prior/var_pars trap).
   - `step=0.01` (absolute) with `numdifftools` (2n+1 evaluations). Use
     `fisher_bias_params={"Omega_c": 0.27}` to probe a systematic shift.
   - Non-SRD layouts (e.g. 3 source bins, 8 lens bins) need `nz_mode="analytic"`.
   Note the metadata `estimated_likelihood_evaluations` for the cost.
3. **`augur_validate_forecast_config(config_path)`** - must be `go`. Fix
   every error before running (they are augur runtime exceptions in waiting).
4. **`augur_generate_synthetic_datavector(output_dir, config_path)`** -
   fiducial C_ell + covariance -> sacc, CSV, n(z), PNG. Check
   `n_data_in_likelihood` (points surviving the kmax cut: Y1 3x2pt 540 ->
   ~410) and `total_snr` (Y1 3x2pt SRD cov ~7000; shear-only few hundred).
5. **`augur_compute_fisher(output_dir, config_path)`** - HEAVY. Local is
   fine for eisenstein_hu Y1 (3x2pt, 9 params: ~20 s); CAMB or Y10 with
   many nuisance parameters or `derivkit`: dispatch. On a facility pass
   `env_setup` (NERSC desc-python candidate from `export_dispatch_pack`),
   via `set_dispatch` server-side or the client's `run_pack_kernel` with
   `function="envkernel.run_in_env", inner="augur_forecast"`. The n(z)
   tables are embedded in the kernel params; the SRD covariance too when
   <= 6 MB (Y1 ok; Y10 needs gaus_internal remotely).
   Read `sigma_no_prior`, `sigma_with_prior`, `fom.no_prior.detf_fom`,
   `fom.with_prior`, `derived` (Omega_m, S8) and `checks.positive_definite`.
6. **`augur_plot_fisher_contours(output_dir, [fisher.csv, ...], params=[...])`**
   - triangle plot at 68/95%; overlay no-prior vs with-prior, or Y1 vs Y10.
   `fom_ratio_to_first` gives the FoM ratio directly.

## Sanity checks (what numbers to expect)

- Fisher matrix symmetric and positive definite; condition number < 1e12.
  Non-PD or absurdly small sigmas: degenerate var_pars (e.g. A_s with an
  analytic transfer function, or h with no CMB prior) or a noisy step.
- Y1 shear-only, 2 bins, 3 cosmological parameters, no nuisance (the test
  setup, ~2 s): sigma(sigma8) ~0.011, sigma(w0) ~0.033, S/N ~230.
- SRD layout, SRD covariance, 9 varied parameters (7 cosmology + lens0/1
  bias) with SRD priors - measured with this server (Y1 8.6 s, Y10 11 s
  local, eisenstein_hu): Y1 sigma(sigma8) ~0.002, sigma(w0) ~0.03,
  sigma(wa) ~0.15, DETF FoM ~900; Y10 sigma(w0) ~0.018, sigma(wa) ~0.08,
  FoM ~3000; Y10/Y1 FoM ratio ~3.3; S/N ~7000 (Y1). These are far above
  the SRD's quoted FoM (Y1 ~40, Y10 ~100+) because the SRD marginalises
  over ~30 nuisance parameters (all lens biases, every src/lens delta_z,
  multiplicative biases, IA amplitude and slope) with their priors: adding
  them to `var_pars` (with `srd_priors=True`) brings sigma(w0) to ~0.1-0.2
  and the FoM to O(10-100). State which nuisance set was marginalised
  whenever you quote a FoM. The Y10/Y1 ratio is the robust comparison
  (2-4 for identical parameter sets).
- Omega_m from Omega_c+Omega_b: sigma(Omega_m) ~ sigma(Omega_c) when
  Omega_b has a tight prior; S8 = sigma8 sqrt(Omega_m/0.3) is better
  constrained than sigma8 (lensing degeneracy direction).
- Fisher bias: |shift/sigma| < 1 means the probed systematic shift is
  tolerable at 1 sigma.

## Comparing two forecasts

Generate both configs with the same `var_pars`, `step`, priors and
cosmology (only survey_year / probes / cov differ), run steps 3-5 for each,
then `augur_plot_fisher_contours` with both fisher.csv files and quote
`fom_ratio_to_first` plus the per-parameter sigma ratios from
`per_file[*].sigma`. Say explicitly which parameters were marginalised and
which priors were applied; forecasts are only comparable with identical
parameter sets.

## Pitfalls

- `step` is ABSOLUTE in parameter units: 0.01 is fine for Omega_c/sigma8/w0,
  too large for Omega_b (0.049) or src_delta_z (prior 0.002); use 1e-3 for
  those. 1e-5 (the shipped examples) is spline-noise territory.
- Every `gaussian_priors` key must be in `var_pars` (or Omega_m/S8 with
  the transform); augur raises otherwise. The generator trims for you.
- `cov_type="SRD"`: only the exact SRD layout (bins, pair lists, 3 statistics,
  20 ells). Anything else -> `gaus_internal`. The SRD covariance already
  encodes its own fsky/noise, so ndens/sigma_e only matter for gaus_internal/tjpcov.
- Keep `general.ignore_scale_cuts_likelihood=False`; True silently removes
  all systematics from the likelihood.
- Exactly one of sigma8/A_s in the cosmology and in var_pars; eisenstein_hu
  needs sigma8. boltzmann_camb needs extra_parameters.camb.halofit_version
  (the generator adds takahashi) and is ~5x slower per evaluation.
- augur mutates global pyccl accuracy parameters in-process; ccl_* tools in
  the same server process afterwards use K_MAX_SPLINE=100.

## Reporting

Give: survey/probes/covariance, varied parameters and priors, data-vector
size inside the cuts, the marginalised sigma table (no prior / with prior),
the DETF w0-wa FoM (and augur's CL-scaled values if asked), derived
sigma(S8)/sigma(Omega_m), where it ran (`computed_on`), and the file paths
(fisher.csv, marginalized_sigmas.csv, fom.json, the contour PNG).
