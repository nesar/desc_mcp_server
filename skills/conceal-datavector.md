---
name: conceal-datavector
description: Conceal (blind) a sacc data vector with Smokescreen before the cosmology analysis - validate the likelihood, choose ranges and a seed, conceal, record what was done without the hidden values, lock the original away, then analyse only the concealed file until the unblinding protocol says otherwise
---

# Conceal a data vector (Smokescreen)

Goal: run the cosmology analysis on a data vector whose cosmology nobody on
the team knows, following the Muir et al. (2021) scheme DESC adopted:
d_concealed = d + t(theta_hidden) - t(theta_ref). The server never records
or prints the hidden values or the seed; it is the user's job to keep the
seed and the original file out of reach. Use one `output_dir`.

1. **Validate the likelihood first** (`likelihood-sanity-checks`): the
   experiment YAML from `firecrown_build_likelihood` must cover EVERY point
   of the sacc file (no scale cuts: Smokescreen compares the data vector
   with the file). If you need cuts, first write a reduced file with
   `sacc_prepare_for_firecrown(keep_data_types=...)` and build on that.
   Note the experiment's amplitude parameter (sigma8 or A_s).
2. **Choose what to hide.** Ranges are ABSOLUTE parameter values, e.g.
   `{"Omega_c": [0.20, 0.32], "sigma8": [0.72, 0.90], "w0": [-1.3, -0.7]}`
   with `shift_distribution='flat'`; or `[mean, std]` with `'gaussian'`.
   Shift the amplitude the experiment samples. `shift_type='add'` for
   3x2pt (multiplicative divides by the reference theory and breaks where
   it crosses zero). A single number per parameter is a deterministic
   offset: reproducible, so only for pipeline tests.
3. **Seed.** Ask the user for a seed string they will not reuse and will
   keep privately; do not invent one yourself and do not repeat it back.
   The same seed + ranges + reference reproduce the hidden cosmology.
4. **Conceal.** `smokescreen_conceal_datavector(experiment_yaml, shifts,
   seed, shift_type, shift_distribution, reference_cosmology=<team
   fiducial>, nuisance=<fiducial nuisance values>, encrypt_original=true)`.
   Outputs: `<stem>_concealed.<ext>` (metadata concealed=true), the
   record JSON (parameters, ranges, distribution, type, reference - never
   the drawn values), the generated likelihood module, and with
   encrypt_original the `.encrpt` + `.key` of the original. The original
   file itself is untouched - the user moves or deletes it per protocol
   (the server never deletes).
5. **Check the concealed file** with `smokescreen_inspect`: concealed=true,
   creator, creation. It also reports that Smokescreen embeds the seed in
   the file's metadata (upstream behaviour): with the ranges, the file
   alone reproduces the hidden cosmology - keep the ranges/record and the
   file apart if that matters. Never call `compare_with` on real data; it
   quantifies the shift.
6. **Analyse the concealed file**: `firecrown_build_likelihood(sacc_path=
   <concealed>)`, scans and chains as usual. Report results as
   "concealed"; parameter values are shifted by construction.
7. **Unblinding** happens outside this server: when the protocol allows,
   `smokescreen_decrypt_file(<.encrpt>, <.key>)` recovers the original,
   and the analysis is rerun on it.
