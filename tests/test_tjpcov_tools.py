"""tjpcov_tools: config validation, Gaussian f_sky covariances (harmonic with
top-hat windows derived from NaMaster-style dense windows, and real space
assembled block by block), SSC term, and the comparison tool.

Synthetic sacc files are built here (seconds); the real-space run uses a
small lmax so the Wigner projection stays fast.
"""

import numpy as np
import pytest

pytest.importorskip("tjpcov")
sacc = pytest.importorskip("sacc")
ccl = pytest.importorskip("pyccl")

from tools.sacc_tools import sacc_attach_gaussian_covariance  # noqa: E402
from tools.tjpcov_tools import (  # noqa: E402
    COV_TYPES,
    tjpcov_compare_covariances,
    tjpcov_compute_covariance,
    tjpcov_generate_config,
    tjpcov_list_covariance_types,
)

Z = np.linspace(0.01, 3.0, 150)
NZ_S = np.exp(-0.5 * ((Z - 0.9) / 0.25) ** 2)
NZ_L = np.exp(-0.5 * ((Z - 0.5) / 0.10) ** 2)
COSMO = {"transfer_function": "eisenstein_hu"}


def _theory():
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
                          transfer_function="eisenstein_hu")
    ts = ccl.WeakLensingTracer(cosmo, dndz=(Z, NZ_S))
    tl = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(Z, NZ_L), bias=(Z, 1.6 * np.ones_like(Z)))
    return cosmo, ts, tl


def make_harmonic_sacc(path, dense_windows=True):
    """src0 x src0, src0 x lens0, lens0 x lens0 bandpowers with windows that
    mimic NaMaster's (dense, peaked in the bin) or exact top-hats."""
    cosmo, ts, tl = _theory()
    edges = np.array([10, 30, 50, 70, 90, 110])
    lmax = int(edges[-1]) - 1
    values = np.arange(lmax + 1)
    ell_eff = 0.5 * (edges[:-1] + edges[1:])
    weight = np.zeros((values.size, len(ell_eff)))
    for k in range(len(ell_eff)):
        inside = (values >= edges[k]) & (values < edges[k + 1])
        if dense_windows:
            weight[:, k] = np.exp(-0.5 * ((values - ell_eff[k]) / 8.0) ** 2)
        else:
            weight[inside, k] = 1.0
        weight[:, k] /= weight[:, k].sum()
    win = sacc.BandpowerWindow(values, weight)
    ellf = np.arange(2, lmax + 1)
    s = sacc.Sacc()
    s.add_tracer("NZ", "src0", Z, NZ_S, quantity="galaxy_shear", metadata={"n_ell_coupled": 1e-10})
    s.add_tracer("NZ", "lens0", Z, NZ_L, quantity="galaxy_density", metadata={"n_ell_coupled": 1e-8})
    for dt, a, b, tr1, tr2 in [("galaxy_shear_cl_ee", ts, ts, "src0", "src0"),
                               ("galaxy_shearDensity_cl_e", ts, tl, "src0", "lens0"),
                               ("galaxy_density_cl", tl, tl, "lens0", "lens0")]:
        cl = np.interp(ell_eff, ellf, ccl.angular_cl(cosmo, a, b, ellf))
        s.add_ell_cl(dt, tr1, tr2, ell_eff, cl, window=win)
    s.metadata["binning/ell_edges"] = ",".join(str(int(e)) for e in edges)
    s.save_hdf5(str(path), overwrite=True)
    return str(path)


def make_real_sacc(path):
    cosmo, ts, tl = _theory()
    ell = np.unique(np.geomspace(2, 2000, 200).astype(int))
    theta = np.geomspace(20, 120, 4)
    s = sacc.Sacc()
    s.add_tracer("NZ", "src0", Z, NZ_S, quantity="galaxy_shear")
    s.add_tracer("NZ", "lens0", Z, NZ_L, quantity="galaxy_density")
    cl_ss, cl_sl, cl_ll = (ccl.angular_cl(cosmo, ts, ts, ell), ccl.angular_cl(cosmo, ts, tl, ell),
                           ccl.angular_cl(cosmo, tl, tl, ell))
    s.add_theta_xi("galaxy_shear_xi_plus", "src0", "src0", theta,
                   ccl.correlation(cosmo, ell=ell, C_ell=cl_ss, theta=theta / 60, type="GG+"))
    s.add_theta_xi("galaxy_shear_xi_minus", "src0", "src0", theta,
                   ccl.correlation(cosmo, ell=ell, C_ell=cl_ss, theta=theta / 60, type="GG-"))
    s.add_theta_xi("galaxy_shearDensity_xi_t", "src0", "lens0", theta,
                   ccl.correlation(cosmo, ell=ell, C_ell=cl_sl, theta=theta / 60, type="NG"))
    s.add_theta_xi("galaxy_density_xi", "lens0", "lens0", theta,
                   ccl.correlation(cosmo, ell=ell, C_ell=cl_ll, theta=theta / 60, type="NN"))
    s.save_hdf5(str(path), overwrite=True)
    return str(path)


def test_list_types():
    r = tjpcov_list_covariance_types()
    assert "FourierGaussianFsky" in r.metadata["supported_here"]
    assert set(r.metadata["types"]) == set(COV_TYPES)


def test_generate_config_validates(tmp_path):
    harm = make_harmonic_sacc(tmp_path / "h.hdf5")
    r = tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, cov_types=["FourierGaussianFsky"],
                               f_sky=0.3, n_gal={"src0": 8, "lens0": 4}, galaxy_bias=1.6, cosmology=COSMO)
    m = r.metadata
    assert m["space"] == "harmonic" and m["has_windows"] is True and m["terms"] == ["gauss"]
    assert m["tracer_kinds"] == {"src0": "shear", "lens0": "density"}
    import yaml
    doc = yaml.safe_load(open(m["config_path"]))
    assert doc["tjpcov"]["Ngal_src0"] == 8 and doc["tjpcov"]["sigma_e_src0"] == 0.26
    assert doc["tjpcov"]["bias_lens0"] == 1.6 and doc["tjpcov"]["cosmo"] == "set" and "parameters" in doc
    # guardrails
    with pytest.raises(ValueError):  # f_sky required by Fsky types
        tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, cov_types=["FourierGaussianFsky"], n_gal=8)
    with pytest.raises(ValueError):  # wrong space
        tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, cov_types=["RealGaussianFsky"],
                               f_sky=0.3, n_gal=8)
    with pytest.raises(ValueError):  # missing n_gal for a tracer
        tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, cov_types=["FourierGaussianFsky"],
                               f_sky=0.3, n_gal={"src0": 8})
    with pytest.raises(ValueError):  # unknown type
        tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, cov_types=["Nope"], f_sky=0.3, n_gal=8)
    # NaMaster type needs masks; cNG needs an HOD (example used, warned)
    with pytest.raises(ValueError):
        tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, cov_types=["FourierGaussianNmt"], n_gal=8)
    r2 = tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, f_sky=0.3, n_gal=8, galaxy_bias=1.6,
                                cov_types=["FourierGaussianFsky", "FouriercNGHaloModelFsky"], name="cng")
    assert any("HOD" in w for w in r2.metadata["warnings"])
    assert "HOD" in yaml.safe_load(open(r2.metadata["config_path"]))


def test_gaussian_fsky_matches_knox_with_dense_windows(tmp_path):
    harm = make_harmonic_sacc(tmp_path / "h.hdf5", dense_windows=True)
    cfg = tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, cov_types=["FourierGaussianFsky"],
                                 f_sky=0.3, n_gal={"src0": 8, "lens0": 4}, galaxy_bias=1.6, cosmology=COSMO)
    cov = tjpcov_compute_covariance(output_dir=str(tmp_path), config_path=cfg.metadata["config_path"])
    m = cov.metadata
    assert m["computed_on"] == "local" and m["n_points"] == 15
    assert m["covariance"]["positive_definite"] is True and m["snr_total"] > 0
    s = sacc.Sacc.load(m["output_sacc"])
    assert s.covariance is not None and s.metadata["tjpcov_cov_types"] == "FourierGaussianFsky"
    # the dense windows survive on the output (only TJPCov's copy got top-hats)
    idx = s.indices("galaxy_density_cl", ("lens0", "lens0"))
    assert (s.get_bandpower_windows(idx).weight > 1e-6).sum(axis=0).min() > 20
    knox = sacc_attach_gaussian_covariance(output_dir=str(tmp_path), sacc_path=harm, f_sky=0.3,
                                           n_gal={"src0": 8, "lens0": 4}, galaxy_bias=1.6, cosmology=COSMO)
    cmp = tjpcov_compare_covariances(output_dir=str(tmp_path), sacc_a=knox.files[0], sacc_b=m["output_sacc"],
                                     label_a="knox", label_b="tjpcov")
    for dt, row in cmp.metadata["per_data_type"].items():
        assert 0.8 < row["median_ratio"] < 1.25, (dt, row)
    assert cmp.metadata["max_abs_corr_difference"] < 0.1
    assert any(f.endswith(".csv") for f in cmp.files) and any(f.endswith(".png") for f in cmp.files)


def test_ssc_term_and_term_files(tmp_path):
    harm = make_harmonic_sacc(tmp_path / "h.hdf5", dense_windows=False)
    cfg = tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=harm, f_sky=0.3, n_gal={"src0": 8, "lens0": 4},
                                 galaxy_bias=1.6, cov_types=["FourierGaussianFsky", "FourierSSCHaloModelFsky"],
                                 cosmology=COSMO, name="ssc")
    cov = tjpcov_compute_covariance(output_dir=str(tmp_path), config_path=cfg.metadata["config_path"])
    terms = cov.metadata["terms"]
    assert set(terms) == {"gauss", "SSC"}
    assert 0 < terms["SSC"]["frac_of_total_diag"] < 0.5 and terms["gauss"]["frac_of_total_diag"] > 0.5
    assert sum(f.endswith("_SSC.hdf5") for f in cov.files) == 1 and sum(f.endswith("_gauss.hdf5") for f in cov.files) == 1
    assert cov.metadata["covariance"]["positive_definite"] is True


def test_real_space_block_assembly(tmp_path):
    real = make_real_sacc(tmp_path / "r.hdf5")
    cfg = tjpcov_generate_config(output_dir=str(tmp_path), sacc_path=real, cov_types=["RealGaussianFsky"],
                                 f_sky=0.1, n_gal={"src0": 8, "lens0": 4}, galaxy_bias=1.6, lmax=600, cosmology=COSMO)
    assert cfg.metadata["space"] == "real" and cfg.metadata["lmax"] == 600
    cov = tjpcov_compute_covariance(output_dir=str(tmp_path), config_path=cfg.metadata["config_path"], plot=False)
    s = sacc.Sacc.load(cov.metadata["output_sacc"])
    c = s.covariance.dense
    d = np.diag(c)
    assert c.shape == (16, 16) and np.all(d > 0), d          # every block filled (xi_- auto included)
    assert np.allclose(c, c.T)
    corr = c / np.sqrt(np.outer(d, d))
    assert np.max(np.abs(corr)) <= 1.0 + 1e-9
    # the xi_+ x xi_- cross block is filled and bounded (TJPCov's own path leaves it broken)
    ip = s.indices("galaxy_shear_xi_plus", ("src0", "src0"))
    im = s.indices("galaxy_shear_xi_minus", ("src0", "src0"))
    assert np.all(np.isfinite(corr[np.ix_(ip, im)])) and np.max(np.abs(corr[np.ix_(ip, im)])) > 0.01
    assert cov.metadata["covariance"]["positive_definite"] is True


def test_compare_refuses_mismatched_layouts(tmp_path):
    a = make_harmonic_sacc(tmp_path / "a.hdf5", dense_windows=False)
    b = make_real_sacc(tmp_path / "b.hdf5")
    ka = sacc_attach_gaussian_covariance(output_dir=str(tmp_path), sacc_path=a, f_sky=0.3,
                                         n_gal={"src0": 8, "lens0": 4}, galaxy_bias=1.6, cosmology=COSMO)
    with pytest.raises(ValueError):
        tjpcov_compare_covariances(output_dir=str(tmp_path), sacc_a=ka.files[0], sacc_b=b)
