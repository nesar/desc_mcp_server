---
name: ccl-datavector-to-sacc
description: Build an LSST 3x2pt theory data vector with CCL - SRD n(z) bins -> angular C_ell (Limber and FKEM non-Limber) -> sacc file with src{i}/lens{i} tracers -> sacc_inspect -> ready for a firecrown likelihood; the CCLX generate_dv_nonlimber recipe as tools
---

# CCL data vector to sacc

Goal: a sacc file holding the LSST Y1 (or Y4/Y7/Y10) shear + clustering +
galaxy-galaxy lensing C_ell computed by pyccl, with tracers and data types
named the way firecrown and augur expect, and a Limber vs non-Limber
comparison for the clustering spectra. Everything is local and takes well
under a minute. Use one `output_dir` throughout.

1. **Cosmology**: agree the cosmology with the user. Default is the
   `desc_srd` preset (`{"preset": "desc_srd"}`: Omega_c 0.2664, Omega_b 0.0492,
   h 0.6727, n_s 0.9645, sigma8 0.831); CCLX's own notebook uses Planck18
   with `m_nu=0.1, mass_split="equal", matter_power_spectrum="camb_hmcode"`.
   Confirm with `ccl_describe_cosmology` (quote Omega_m and sigma8). camb is
   needed for the default transfer function; for a quick dry run use
   `transfer_function="eisenstein_hu"`.
2. **n(z)**: `ccl_lsst_srd_nz(forecast_year=1, sample="source")` and
   `ccl_lsst_srd_nz(forecast_year=1, sample="lens")`. Sanity: source edges
   ~[0, 0.35, 0.55, 0.77, 1.11, 3.5] (equal numbers, 2 gal/arcmin^2 each);
   lens edges 0.2..1.2 in steps of 0.2 (Y1/Y4) or 0.1 (Y7/Y10), 18/arcmin^2
   total, `galaxy_bias_prefactor` 1.05 (Y1/Y4) or 0.95 (Y7/Y10). Keep the
   two CSV paths; the metadata also gives `n_gal_per_bin_arcmin2`, `sigma_z`
   and `f_sky` (0.4363), which the covariance step needs later.
3. **Tracer list**: one `TracerSpec` per bin -
   `{"type":"wl","name":"src<i>","nz_file":<source csv>,"nz_column":"bin_<i>","A_ia":1.0,"eta_ia":-1.0,"z_piv_ia":0.62}`
   for i in 0..4 and
   `{"type":"nc","name":"lens<j>","nz_file":<lens csv>,"nz_column":"bin_<j>","bias":"srd","mag_bias":0.1}`
   for j in 0..4 (10 bins for Y7/Y10). `bias="srd"` is the SRD rule
   b = prefactor / D(z_mean); the resolved values are returned in
   `metadata.bias_values` (Y1: ~1.24, 1.36, 1.50, 1.65, 1.80). Names MUST be
   `src{i}` / `lens{j}` for augur and the firecrown examples.
4. **C_ell, non-Limber** (the CCLX default): `ccl_angular_cls(tracers=...,
   pairs="all", ell_min=20, ell_max=2000, n_ell=20, l_limber=2000,
   non_limber_pairs="clustering", write_sacc=true)`. This runs exact FKEM for
   every pair with a lens tracer and Limber for shear-shear. Expect ~55 pairs
   for 5+5 bins and a few seconds. Check `metadata.l_limber_used` (2000 or
   more for clustering pairs, -1 for shear-shear) and `metadata.kernel_z_range`.
5. **C_ell, Limber**: repeat with `l_limber=-1` and `write_sacc=false`. Compare
   the `cl_lens0_lens0` columns of the two CSVs: non-Limber differs from Limber
   by several percent at ell < 50 for the lowest lens bins and agrees to < 1%
   above ell ~ 100. Quote the largest fractional difference and the ell where
   the two agree within 1%; that is the user's answer to "does non-Limber
   matter for my ell range".
6. **Inspect the sacc**: `sacc_inspect(<cls_*.sacc>)`. Expect tracers src0-4
   (quantity galaxy_shear) and lens0-4 (galaxy_density); data types
   `galaxy_shear_cl_ee`, `galaxy_shearDensity_cl_e` (shear tracer first), and
   `galaxy_density_cl`; no covariance; no bandpower windows (C_ell at the
   listed ells).
7. **Covariance and likelihood** (optional continuation): attach a Gaussian
   covariance with the sacc tools (f_sky 0.4363, sigma_e 0.26, n_gal per bin
   from step 2's metadata), then `firecrown_build_likelihood` on the result and
   `firecrown_compute_loglike` at the same cosmology: chi2 ~ 0 by construction
   (theory == data). A non-zero chi2 means a convention mismatch (bias model,
   IA model, tracer order, ell grid) - re-check the TracerSpecs.
8. **Report**: the cosmology, the bin edges, the number of pairs and the ell
   grid, the non-Limber vs Limber summary from step 5, the sacc path and its
   contents, and the files produced (CSV + PNG per step, the .sacc).

Variations: real-space data vectors via `ccl_correlation_functions` with the
same tracer list (xi+/xi-, gamma_t, w(theta) on 1-300 arcmin); a CMB-lensing
cross by adding `{"type":"cmb_lensing","name":"ck"}` (sacc type
cmbGalaxy_convergenceShear_cl_e / cmbGalaxy_convergenceDensity_cl); Y10 by
`forecast_year=10` (10 lens bins, prefactor 0.95). Heavy ell grids or many
cosmologies can be dispatched: `ccl_angular_cls` is a pip-kernel
(`ccl.kernels.compute_angular_cls`, deps pyccl + camb); see
hpc-dispatch-handoff.
