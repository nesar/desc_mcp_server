---
name: maps-to-likelihood
description: From masked HEALPix maps (or a simulated test bed) to a likelihood - NaMaster bandpowers with windows -> sacc -> TJPCov covariance (Gaussian f_sky, optionally super-sample) -> firecrown chi2 at the fiducial, with the cross-checks that catch binning and noise mistakes
---

# Maps -> bandpowers -> covariance -> likelihood

Goal: measure angular power spectra from maps with NaMaster, give them an
analysis-grade covariance with TJPCov, and evaluate the firecrown
likelihood - checking at each step that binning, noise and tracer
conventions line up. Use one `output_dir`. With a simulated test bed the
whole chain runs locally in about a minute at nside 64-128.

1. **Inputs.** Real data: one `FieldSpec` per tracer - mask (FITS, or the
   TXPipe HDF5 maps file with `hdf5_names` + `nside`), map(s) (one for
   density, Q/U for shear), `n_gal_arcmin2` (+ `sigma_e`) for the
   analytic white-noise subtraction, and `nz_csv` so the sacc gets NZ
   tracers (firecrown and TJPCov need the n(z)). No data? Make a test bed:
   `namaster_simulate_maps` (nside 64-128, `f_sky` 0.2-0.4, two or three
   tracers with Gaussian n(z)) writes maps, mask, n(z) CSVs, the input
   theory C_ell and a ready `fields_json`.
2. **Mask.** `namaster_mask_properties` on the mask: note `fsky_eff`
   (= <w>^2/<w^2>) - that is the f_sky for every Gaussian covariance
   below. Apodize (0.5-2 deg, C1) if the mask has sharp edges or if you
   will purify B-modes.
3. **Measure.** `namaster_compute_cls(fields_json=..., binning={scheme:
   'linear', nlb: 16-32, ell_min: 8-30, ell_max: <= 3*nside-1},
   include_b_modes=false for a likelihood file, theory_csv=<input theory>
   on a simulation)`. Check: (a) `windows_in_sacc` true; (b) on a
   simulation the binned theory overlays the bandpowers in the PNG within
   the scatter; (c) `fields.<name>.noise_coupled` is nonzero when n_gal
   was given. Keep `n_ell_coupled` in the tracer metadata - TJPCov reads it.
4. **Covariance.** `tjpcov_generate_config(sacc_path=<measured sacc>,
   cov_types=['FourierGaussianFsky'] (+ 'FourierSSCHaloModelFsky' for a
   wide-area survey), f_sky=<fsky_eff>, n_gal={...}, sigma_e, galaxy_bias,
   cosmology=<same transfer function as the theory>)` then
   `tjpcov_compute_covariance`. The config tool reads the bandpower edges
   namaster wrote into the sacc metadata, so TJPCov bins exactly as the
   measurement did. Require `positive_definite: true`.
   Cross-check: `sacc_attach_gaussian_covariance` with the same f_sky and
   noise, then `tjpcov_compare_covariances(knox, tjpcov)` - the median
   sigma ratio must be ~1 (0.95-1.05); a factor of 2 or more means a
   binning/noise/f_sky mismatch, not physics. SSC adds a few percent on
   the diagonal at low ell for f_sky ~0.3 and more correlation.
   (FourierGaussianNmt - true mode coupling - needs TJPCov under NaMaster
   2.x: run it remotely with env_setup, see hpc-dispatch-handoff.)
5. **Likelihood.** `firecrown_build_likelihood(sacc_path=<TJPCov sacc>,
   transfer_function=<same as step 4>)` -> `firecrown_compute_loglike`
   at the input cosmology with the simulation's biases as `nuisance`.
   Expect chi2/n_data ~ 1 (0.7-1.3) on a simulation; chi2/n >> 1 means a
   convention slip (theta/ell, tracer order, noise double-counted, a
   nuisance left at a non-neutral default). Then `firecrown_scan_loglike`
   in sigma8 to see the constraint, or `likelihood-sanity-checks`.
6. **Report** the f_sky, the number of bandpowers and pairs, the sigma
   ratio from the cross-check, the covariance condition number, and
   chi2/n at the fiducial - with the file paths. Large maps (nside >= 1024)
   go to a facility: the same `namaster_compute_cls` call with
   `env_setup` and facility map paths.
