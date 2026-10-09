"""ccl family: tools called in-process against direct pyccl calls (small grids, < 90 s)."""

import json

import numpy as np
import pytest

ccl = pytest.importorskip("pyccl")

from tools.ccl import (ccl_angular_cls, ccl_background, ccl_baryon_boost, ccl_correlation_3d,  # noqa: E402
                       ccl_correlation_functions, ccl_describe_cosmology, ccl_halo_mass_function,
                       ccl_halo_model_pk, ccl_lsst_srd_nz, ccl_matter_pk, DISPATCH_KERNELS, TracerSpec)
from tools.ccl.kernels import compute_angular_cls, compute_matter_pk, emulator_box_check  # noqa: E402
from tools.ccl.srd_nz import SRD_PARAMS, srd_sample  # noqa: E402
from tools.common import read_csv  # noqa: E402
from tools.cosmology import CosmologyParams  # noqa: E402

BBKS = CosmologyParams(transfer_function="bbks")  # fast analytic transfer; sigma8-normalised
VANILLA = ccl.CosmologyVanillaLCDM(transfer_function="bbks")


def test_describe_cosmology(tmp_path):
    res = ccl_describe_cosmology(str(tmp_path), cosmology=BBKS)
    assert res.status == "success" and len(res.files) == 2  # json + yaml
    d = res.metadata["derived"]
    assert abs(d["Omega_m"] - 0.30) < 1e-6 and abs(d["sigma8"] - 0.81) < 1e-3
    assert 13 < d["age_Gyr"] < 15
    # baryon models cannot be serialised to YAML -> json only, still success
    res2 = ccl_describe_cosmology(str(tmp_path), cosmology=CosmologyParams(transfer_function="bbks", baryons="schneider15"),
                                  compute_sigma8=False)
    assert len(res2.files) == 1 and res2.metadata["yaml"].startswith("not written")


def test_background_matches_pyccl(tmp_path):
    res = ccl_background(str(tmp_path), cosmology=BBKS, z_min=0.0, z_max=3.0, n_z=31)
    header, cols = read_csv(res.files[0])
    assert header["quantity"] == "background"
    i = np.argmin(np.abs(cols["z"] - 1.0))
    chi_direct = ccl.comoving_radial_distance(VANILLA, 0.5)
    assert abs(cols["chi_Mpc"][i] / chi_direct - 1) < 1e-6  # CSV keeps 8 significant digits
    assert abs(chi_direct - 3300) / 3300 < 0.1  # sanity: ~3.3-3.5 Gpc
    assert np.isnan(cols["distance_modulus"][0]) and cols["growth_factor"][0] == pytest.approx(1.0)
    assert any(f.endswith(".png") for f in res.files)


def test_matter_pk_bbks_units(tmp_path):
    h = BBKS.h
    res = ccl_matter_pk(str(tmp_path), cosmology=BBKS, z=0.0, k_min=0.01, k_max=10.0, n_k=30, make_plot=False)
    header, cols = read_csv(res.files[0])
    k = cols["k_h_per_Mpc"]
    direct = ccl.nonlin_matter_power(VANILLA, k * h, 1.0) * h**3
    assert np.allclose(cols["Pk_nl_Mpc3_over_h3"], direct, rtol=1e-6)
    assert res.metadata["box"]["in_training_box"] is None
    # CCL-native units
    res2 = ccl_matter_pk(str(tmp_path), cosmology=BBKS, k_min=0.5, k_max=2.0, n_k=5, k_units="1/Mpc", make_plot=False)
    _, cols2 = read_csv(res2.files[0])
    assert np.allclose(cols2["Pk_nl_Mpc3"], ccl.nonlin_matter_power(VANILLA, cols2["k_per_Mpc"], 1.0), rtol=1e-6)


def test_matter_pk_camb_sanity(tmp_path):
    pytest.importorskip("camb")
    res = ccl_matter_pk(str(tmp_path), cosmology=CosmologyParams(), z=0.0, k_min=0.9, k_max=1.1, n_k=5, make_plot=False)
    _, cols = read_csv(res.files[0])
    p1 = np.interp(1.0, cols["k_h_per_Mpc"], cols["Pk_nl_Mpc3_over_h3"])
    assert 350 < p1 < 450  # ~400 (Mpc/h)^3 with halofit
    assert abs(res.metadata["sigma8"] - 0.81) < 1e-3


def test_matter_pk_cosmicemu_box(tmp_path):
    emu = CosmologyParams(matter_power_spectrum="cosmicemu_mt4")
    res = ccl_matter_pk(str(tmp_path), cosmology=emu, z=0.5, k_min=0.01, k_max=4.0, n_k=20, make_plot=False)
    assert res.metadata["box"]["in_training_box"] is True
    _, cols = read_csv(res.files[0])
    assert np.all(cols["ratio_nl_lin"][-5:] > 1.0)
    with pytest.raises(ValueError, match="training box"):
        ccl_matter_pk(str(tmp_path), cosmology=CosmologyParams(matter_power_spectrum="cosmicemu_mt4", sigma8=0.95), n_k=5)
    flagged = emulator_box_check(emu.model_dump(), 3.0)
    assert flagged["in_training_box"] is False and not flagged["parameter_violations"]


def test_srd_nz_table_and_bins(tmp_path):
    for yr in (1, 4, 7, 10):
        assert SRD_PARAMS["lens"][yr]["n_tomo_bins"] == (5 if yr in (1, 4) else 10)
        s = srd_sample(yr, "source", n_z=200)
        frac = np.array(s["bin_fraction_of_sample"])
        assert len(s["bins"]) == 5 and np.allclose(frac, frac[0], rtol=0.15)  # equal-number bins
    lens = srd_sample(10, "lens", n_z=200)
    assert lens["edges"] == pytest.approx(np.linspace(0.2, 1.2, 11).tolist())
    res = ccl_lsst_srd_nz(str(tmp_path), forecast_year=1, sample="lens", n_z=200)
    header, cols = read_csv(res.files[0])
    assert list(cols)[:3] == ["z", "nz_total", "bin_0"] and len(cols) == 7
    assert np.isclose(np.trapezoid(cols["bin_2"], cols["z"]), 1.0, atol=1e-6)
    assert header["galaxy_bias_prefactor"] == "1.05" and abs(sum(res.metadata["n_gal_per_bin_arcmin2"]) - 18.0) < 1e-9
    assert res.metadata["f_sky"] == 0.4363


def _specs(src_file, lens_file):
    return [TracerSpec(type="wl", name="src0", nz_file=src_file, nz_column="bin_0", A_ia=1.0, eta_ia=-1.0),
            TracerSpec(type="nc", name="lens0", nz_file=lens_file, nz_column="bin_0", bias="srd", mag_bias=0.1)]


def test_angular_cls_vs_pyccl_and_sacc(tmp_path):
    sacc = pytest.importorskip("sacc")
    src = ccl_lsst_srd_nz(str(tmp_path), forecast_year=1, sample="source", n_z=150, make_plot=False)
    lens = ccl_lsst_srd_nz(str(tmp_path), forecast_year=1, sample="lens", n_z=150, make_plot=False,
                           write_sacc=True, sacc_file=str(tmp_path / "nz.sacc"))
    specs = _specs(src.files[0], lens.files[0])
    res = ccl_angular_cls(str(tmp_path), tracers=specs, cosmology=BBKS, pairs="all", ell_min=10, ell_max=1000,
                          n_ell=15, l_limber=-1, write_sacc=True, make_plot=False)
    header, cols = read_csv(res.files[0])
    assert list(cols) == ["ell", "cl_src0_src0", "cl_src0_lens0", "cl_lens0_lens0"]
    # direct pyccl check of the shear auto-spectrum (same n(z), same IA model)
    _, nzc = read_csv(src.files[0])
    z, nz = nzc["z"], nzc["bin_0"]
    ia = (z, 1.0 * ((1 + z) / 1.62) ** -1.0)
    wl = ccl.WeakLensingTracer(VANILLA, dndz=(z, nz), ia_bias=ia)
    assert np.allclose(cols["cl_src0_src0"], ccl.angular_cl(VANILLA, wl, wl, cols["ell"]), rtol=1e-5)
    b = res.metadata["bias_values"]["lens0"]
    assert 1.1 < b < 1.5  # 1.05 / D(z_mean ~ 0.3)
    assert 1e-10 < np.max(cols["cl_src0_src0"]) < 1e-6
    # sacc round trip
    s = sacc.Sacc.load_fits(res.metadata["sacc_file"])
    assert set(s.tracers) == {"src0", "lens0"}
    assert s.tracers["src0"].quantity == "galaxy_shear" and s.tracers["lens0"].quantity == "galaxy_density"
    assert set(s.get_data_types()) == {"galaxy_shear_cl_ee", "galaxy_shearDensity_cl_e", "galaxy_density_cl"}
    assert ("src0", "lens0") in s.get_tracer_combinations("galaxy_shearDensity_cl_e")  # shear first
    ell_s, cl_s = s.get_ell_cl("galaxy_density_cl", "lens0", "lens0")
    assert np.allclose(cl_s, cols["cl_lens0_lens0"])
    # the n(z)-only sacc from ccl_lsst_srd_nz
    s2 = sacc.Sacc.load_fits(str(tmp_path / "nz.sacc"))
    assert [f"lens{i}" for i in range(5)] == list(s2.tracers)


def test_angular_cls_nonlimber(tmp_path):
    lens = ccl_lsst_srd_nz(str(tmp_path), forecast_year=1, sample="lens", n_z=150, make_plot=False)
    spec = [TracerSpec(type="nc", name="lens0", nz_file=lens.files[0], nz_column="bin_0", bias=1.3)]
    res = ccl_angular_cls(str(tmp_path), tracers=spec, cosmology=BBKS, pairs="auto", ell_min=2, ell_max=300,
                          n_ell=12, l_limber=100, make_plot=False)
    assert res.metadata["l_limber_used"]["lens0|lens0"] >= 100
    lim = ccl_angular_cls(str(tmp_path), tracers=spec, cosmology=BBKS, pairs="auto", ell_min=2, ell_max=300,
                          n_ell=12, l_limber=-1, make_plot=False)
    _, a = read_csv(res.files[0])
    _, b = read_csv(lim.files[0])
    # non-Limber and Limber differ at low ell, agree at high ell
    assert abs(a["cl_lens0_lens0"][0] / b["cl_lens0_lens0"][0] - 1) > 0.02
    assert abs(a["cl_lens0_lens0"][-1] / b["cl_lens0_lens0"][-1] - 1) < 0.05


def test_correlation_functions(tmp_path):
    src = ccl_lsst_srd_nz(str(tmp_path), forecast_year=1, sample="source", n_z=150, make_plot=False)
    lens = ccl_lsst_srd_nz(str(tmp_path), forecast_year=1, sample="lens", n_z=150, make_plot=False)
    res = ccl_correlation_functions(str(tmp_path), tracers=_specs(src.files[0], lens.files[0]), cosmology=BBKS,
                                    theta_min_arcmin=1, theta_max_arcmin=300, n_theta=8, make_plot=False)
    _, cols = read_csv(res.files[0])
    assert list(cols) == ["theta_arcmin", "xip_src0_src0", "xim_src0_src0", "gammat_src0_lens0", "w_lens0_lens0"]
    assert res.metadata["types"]["xim_src0_src0"] == "GG-"
    xip = cols["xip_src0_src0"]
    assert 1e-7 < xip[0] < 1e-3 and xip[0] > xip[-1] > 0


def test_correlation_3d_bao(tmp_path):
    res = ccl_correlation_3d(str(tmp_path), cosmology=BBKS, z=0.0, r_min=5, r_max=150, n_r=20, make_plot=False)
    _, cols = read_csv(res.files[0])
    assert list(cols) == ["r_Mpc_over_h", "xi_r", "xi_0", "xi_2", "xi_4"]
    direct = ccl.correlation_3d(VANILLA, r=cols["r_Mpc_over_h"] / 0.67, a=1.0)
    assert np.allclose(cols["xi_r"], direct, rtol=1e-5)
    assert 0.4 < res.metadata["beta"] < 0.6  # f(z=0) ~ 0.51 for Omega_m=0.3


def test_halo_mass_function(tmp_path):
    res = ccl_halo_mass_function(str(tmp_path), cosmology=BBKS, z_list=[0.0], log10M_min=13, log10M_max=15, n_M=9, make_plot=False)
    _, cols = read_csv(res.files[0])
    hmf = ccl.halos.MassFuncTinker08(mass_def="200m")
    assert np.allclose(cols["dndlog10M_Mpc3_z0"], hmf(VANILLA, cols["M_Msun"], 1.0), rtol=1e-6)
    n14 = res.metadata["at_M_nearest_1e14"]["z0"]["dndlog10M_Mpc3"]
    assert 1e-5 < n14 < 1e-4
    with pytest.raises(ValueError, match="mass_def"):
        ccl_halo_mass_function(str(tmp_path), cosmology=BBKS, mass_def="banana", n_M=5)


def test_halo_model_pk(tmp_path):
    res = ccl_halo_model_pk(str(tmp_path), cosmology=BBKS, k_min=0.01, k_max=5, n_k=12, n_M=64, make_plot=False)
    _, cols = read_csv(res.files[0])
    tot = cols["Pk_hm_Mpc3_over_h3"]
    assert np.allclose(tot, cols["Pk_1h_Mpc3_over_h3"] + cols["Pk_2h_Mpc3_over_h3"])
    # 2-halo dominates on large scales and tracks the reference within ~30%
    assert abs(tot[0] / cols["Pk_nl_ref_Mpc3_over_h3"][0] - 1) < 0.3


def test_baryon_boost(tmp_path):
    res = ccl_baryon_boost(str(tmp_path), cosmology=BBKS, models=["schneider15", "vandaalen19"], k_min=0.1, k_max=10, n_k=12, make_plot=False)
    header, cols = read_csv(res.files[0])
    assert header["quantity"] == "suppression"
    s15 = cols["boost_schneider15"]
    assert s15[0] > 0.98 and 0.6 < s15.min() < 0.95


def test_kernel_json_roundtrip(tmp_path):
    args = {"cosmo_params": BBKS.model_dump(), "z": 0.5, "k_list": [0.01, 0.1, 1.0], "k_units": "h/Mpc"}
    args = json.loads(json.dumps(args))
    out = json.loads(json.dumps(compute_matter_pk(**args)))
    assert len(out["pk_nl"]) == 3 and out["box"]["in_training_box"] is None
    direct = ccl.nonlin_matter_power(VANILLA, np.array(args["k_list"]) * 0.67, 1 / 1.5) * 0.67**3
    assert np.allclose(out["pk_nl"], direct, rtol=1e-6)
    z = np.linspace(0, 2, 40)
    nz = z**2 * np.exp(-((z / 0.5) ** 1.5))
    targs = json.loads(json.dumps({
        "cosmo_params": BBKS.model_dump(),
        "tracers": [{"type": "wl", "name": "s", "z": z.tolist(), "nz": nz.tolist()},
                    {"type": "nc", "name": "g", "z": z.tolist(), "nz": nz.tolist(), "bias_value": 1.5, "has_rsd": False}],
        "pairs": [["s", "s"], ["g", "g"]], "ell_list": [10, 100, 1000], "l_limber": -1}))
    cout = json.loads(json.dumps(compute_angular_cls(**targs)))
    assert set(cout["cls"]) == {"s|s", "g|g"} and len(cout["ell"]) == 3
    assert cout["kernel_z_range"]["g"][1] == pytest.approx(2.0, abs=0.05)
    assert DISPATCH_KERNELS["ccl_matter_pk"]["function"] == "ccl.kernels.compute_matter_pk"
    assert "pyccl" in DISPATCH_KERNELS["ccl_angular_cls"]["pip_deps"]
