---
name: pk-ccl-vs-emulators
description: Cross-check the nonlinear matter P(k) from CCL's backends (halofit, HMcode-2020, CosmicEmu MT-IV, bacco, linear) on one h/Mpc grid and against the companion cosmic-emulator server's compute_nonlinear_pk at the same cosmology; report the maximum fractional differences and where the backends agree
---

# P(k): CCL backends vs the emulator server

Goal: an honest spread of nonlinear P(k) predictions at one cosmology and
redshift, computed on the SAME k grid in h/Mpc, and the maximum fractional
difference of each backend relative to a reference. Units are the trap:
CCL is h-free (1/Mpc, Mpc^3), the emulator server is h/Mpc, (Mpc/h)^3. The
ccl_ tools default to h/Mpc so no conversion is needed on the CCL side.

1. **Cosmology**: fix it with the user (defaults: vanilla Omega_c 0.25,
   Omega_b 0.05, h 0.67, n_s 0.96, sigma8 0.81, z 0 or 0.5). Emulator boxes:
   CosmicEmu MT-IV needs sigma8 in 0.7-0.9, z <= 2.02; bacco z <= 1.5 (and the
   baccoemu package, often absent); CAMB (camb_hmcode) must be installed.
   Call `ccl_describe_cosmology` once and quote Omega_m, sigma8.
2. **CCL runs**, one `ccl_matter_pk` call per backend, identical grid
   (`k_min=0.01, k_max=4.5, n_k=200, k_units="h/Mpc"`, same z):
   - `matter_power_spectrum="halofit"` (Takahashi 2012, default)
   - `"camb_hmcode"` (HMcode-2020 with feedback; set `hmcode_logT_AGN=7.8`;
     use `7.6` for weaker feedback, and note this is NOT dark-matter-only)
   - `"cosmicemu_mt4"` (Mira-Titan IV, HACC N-body; the only emulator with
     no extra dependency)
   - `"bacco"` if importable (`optional dependency` error otherwise - say so)
   - `"linear"` as the no-nonlinear baseline.
   Read `metadata.box.in_training_box` for each emulator run; outside the
   box the tool refuses (parameters) or flags (z) - report it, do not retry
   blindly. `metadata.at_k_nearest_1` gives P_nl at k ~ 1 h/Mpc: ~400
   (Mpc/h)^3 at z=0 for vanilla halofit.
3. **Emulator server**: convert names first with
   `convert_cosmology_names(params=<firecrown dict>, to="emulator")` - it
   returns `Om` (= Omega_c + Omega_b + Omega_nu), `Ob`, `h`, `ns`, `sigma8`,
   `mnu`, `w0`, `wa` - then call that server's `compute_nonlinear_pk` with
   the same `k_min/k_max/n_points` and `z` for each of its backends (its
   `miratitan` is the same Mira-Titan IV emulator as CCL's cosmicemu_mt4 and
   should agree to < 1% - that is the units cross-check; if it is off by
   h^3 ~ 0.3 or the k axis is shifted by h, a unit convention slipped).
4. **Compare**: load each CSV's `Pk_nl_*` column (CCL files) / P(k) column
   (emulator files) on the common k; interpolate in log k if grids differ.
   With halofit as reference compute for each backend: max |P/P_ref - 1|
   and the k where it occurs, plus the k range where all agree within 2%.
   Typical z=0 LCDM: halofit vs HMcode-2020 3-5% at k ~ 1-3 h/Mpc; halofit
   vs Mira-Titan 2-5% for k < 2; emulators among themselves 1-3%; linear
   vs nonlinear 2x by k ~ 1 h/Mpc.
5. **Plot**: the emulator server's `plot_emulator_curves` overlays the CSVs
   (CCL files carry `quantity: power_spectrum` and `k_h_per_Mpc` so they are
   understood directly), ratio panel on, `reference_index` at halofit.
6. **Report**: table backend -> max fractional difference (and k), the
   agreement range, which runs were out of box or skipped (missing package),
   and a recommendation: emulator (cosmicemu_mt4 / server miratitan) for
   k <= 2-5 h/Mpc inside the box; camb_hmcode when baryon feedback must be
   folded in; halofit only as the legacy reference. Add `ccl_baryon_boost`
   (S(k) for schneider15 / vandaalen19 / hmcode2020) if the user asks for
   baryons; its files overlay the emulator server's
   `compute_baryon_suppression` output (same `quantity: suppression`).
