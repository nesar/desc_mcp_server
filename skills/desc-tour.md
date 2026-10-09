---
name: desc-tour
description: One representative call per tool family (CCL theory, sacc inspection, firecrown likelihood, augur Fisher forecast, TXPipe pipeline composition) - the demo path that shows what this server does end to end in a few minutes, all local
---

# DESC server tour

Goal: exercise every family once, cheaply, and leave the user with concrete
artifacts and a sense of how the pieces chain (theory -> data vector -> sacc
-> likelihood -> forecast; measurement pipeline -> sacc). Everything below
runs locally in well under five minutes. Use one `output_dir` for the tour.

1. **Discover**: `list_desc_packages`, then `describe_desc_tool_family("ccl")`.
   Report the installed versions (pyccl, sacc, firecrown, augur) in one line.
2. **CCL theory**: `ccl_background` (z 0-3, vanilla preset) and
   `ccl_matter_pk` at z=0.5 with `matter_power_spectrum="halofit"`. Quote
   chi(z=1) ~ 3300 Mpc and P_nl(k=1 h/Mpc) ~ 800-900 (Mpc/h)^3 as checks.
3. **Data vector**: `ccl_lsst_srd_nz` (Y1, source and lens) ->
   `ccl_angular_cls` with the SRD bins, `pairs="auto"`, `write_sacc=true`.
   This is the CCLX "generate data vector" recipe in two calls.
4. **sacc**: `sacc_inspect` on that file: tracers, data types, covariance
   presence (none yet). Then `sacc_attach_gaussian_covariance` with
   f_sky=0.4, sigma_e=0.26, n_gal from the SRD metadata.
5. **Likelihood**: `firecrown_build_likelihood` on the covariance-bearing
   sacc (harmonic, per-bin PhotoZShift + MultiplicativeShearBias, global
   LinearAlignment) -> `firecrown_compute_loglike` at the same cosmology.
   Build it with the SAME `transfer_function` the data vector used (the
   experiment YAML fixes it; the cosmology passed later cannot change it).
   Expect chi2 ~ 0 (theory == data by construction) - that is the point:
   a non-zero chi2 here means a convention mismatch (transfer function,
   tracer order, ell windows, a nuisance parameter left at a non-neutral
   value: unspecified mult_bias/ia_bias/delta_z are neutral by design).
6. **Forecast**: `augur_generate_forecast_config` (Y1, shear-only, 3 var_pars)
   -> `augur_validate_forecast_config` -> `augur_compute_fisher` (local is
   fine for shear-only Y1; ~1 min) -> `augur_plot_fisher_contours`.
   Quote the marginalised sigma(sigma8) and sigma(w0).
7. **Measurement pipeline (compose only)**: `txpipe_list_stages(group="two-point")`,
   `txpipe_describe_stage("TXTwoPoint")`, then `txpipe_generate_pipeline(preset="3x2pt_real", ...)`
   and `txpipe_validate_pipeline`. Show the stage list and the dry-run
   command for TXTwoPoint; explain that running it needs a TXPipe
   environment (local `DESC_TXPIPE_ENV` or a facility `env_setup`).
8. **Wrap up**: list the files produced, the one-line numbers from steps
   2, 5 and 6, and point to `list_desc_skills` for the deeper recipes
   (`lsst-3x2pt-forecast`, `txpipe-sacc-to-likelihood`, `pk-ccl-vs-emulators`,
   `hpc-dispatch-handoff`).
