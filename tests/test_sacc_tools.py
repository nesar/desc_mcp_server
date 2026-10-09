"""sacc_tools: inspect / export / prepare-for-firecrown / Gaussian covariance.

Uses firecrown/tests/sacc_data.hdf5 (DES-Y1-like real-space 3x2pt, copied to
tmp_path) and a tiny TXPipe-style harmonic file built here (the sacc repo's
example-txpipe-sacc1.sacc is a pre-2.0 file that sacc 2.4 cannot read).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

sacc = pytest.importorskip("sacc")

from tools.common import clone_dir  # noqa: E402
from tools.sacc_tools import (sacc_attach_gaussian_covariance, sacc_inspect,  # noqa: E402
                              sacc_prepare_for_firecrown, sacc_to_csv)

_FC = clone_dir("firecrown")
_DES = _FC / "tests" / "sacc_data.hdf5" if _FC else None


@pytest.fixture
def des_sacc(tmp_path) -> str:
    if _DES is None or not _DES.is_file():
        pytest.skip("firecrown clone with tests/sacc_data.hdf5 not available")
    dst = tmp_path / "sacc_data.hdf5"
    shutil.copy(_DES, dst)
    return str(dst)


def make_txpipe_like_sacc(path: Path) -> Path:
    """TXPipe-style harmonic file: source_i / lens_i, quantity 'generic',
    B-modes included, no covariance."""
    z = np.linspace(0.01, 2.5, 120)

    def nz(z0, s):
        return np.exp(-0.5 * ((z - z0) / s) ** 2)

    s = sacc.Sacc()
    s.add_tracer("NZ", "source_0", z, nz(0.5, 0.15))
    s.add_tracer("NZ", "source_1", z, nz(0.9, 0.2))
    s.add_tracer("NZ", "lens_0", z, nz(0.4, 0.1))
    ell = np.array([100.0, 180.0, 320.0, 560.0, 1000.0])
    rng = np.random.default_rng(0)
    for a, b in [("source_0", "source_0"), ("source_0", "source_1"), ("source_1", "source_1")]:
        s.add_ell_cl("galaxy_shear_cl_ee", a, b, ell, 1e-8 * (ell / 100.0) ** -1.0 * (1 + 0.05 * rng.standard_normal(5)))
        s.add_ell_cl("galaxy_shear_cl_bb", a, b, ell, 1e-10 * rng.standard_normal(5))
    for a in ["source_0", "source_1"]:
        s.add_ell_cl("galaxy_shearDensity_cl_e", a, "lens_0", ell, 3e-8 * (ell / 100.0) ** -1.0)
    s.add_ell_cl("galaxy_density_cl", "lens_0", "lens_0", ell, 2e-6 * (ell / 100.0) ** -1.0)
    s.save_fits(str(path), overwrite=True)
    return path


@pytest.fixture
def txpipe_sacc(tmp_path) -> str:
    return str(make_txpipe_like_sacc(tmp_path / "twopoint_data_fourier.fits"))


def test_inspect_des_y1_like(des_sacc, tmp_path):
    r = sacc_inspect(output_dir=str(tmp_path / "out"), sacc_path=des_sacc)
    assert r.status == "success"
    m = r.metadata
    assert m["n_points"] == 457 and m["space"] == "real"
    assert sorted(t["name"] for t in m["tracers"]) == [f"lens{i}" for i in range(5)] + [f"src{i}" for i in range(4)]
    assert {d["data_type"]: d["n_points"] for d in m["data_types"]} == {
        "galaxy_density_xi": 54, "galaxy_shearDensity_xi_t": 176,
        "galaxy_shear_xi_minus": 60, "galaxy_shear_xi_plus": 167}
    assert m["covariance"]["present"] and m["covariance"]["positive_definite"]
    assert m["covariance"]["shape"] == [457, 457] and not m["covariance"]["is_diagonal"]
    assert m["tracer_names_ready"] and not m["quantities_ready"]  # names fine, quantity 'generic'
    names = {Path(f).name for f in r.files}
    assert "sacc_data_tracers.csv" in names and "sacc_data_datavector.csv" in names
    assert "sacc_data_nz_src0.csv" in names and "sacc_data_covariance_corr.png" in names
    src0 = next(t for t in m["tracers"] if t["name"] == "src0")
    assert src0["n_z"] == 400 and 0.0 < src0["z_mean"] < 1.5 and src0["inferred_kind"] == "shear"
    json.dumps(m)  # JSON-safe metadata (the MCP boundary)


def test_to_csv(des_sacc, tmp_path):
    r = sacc_to_csv(output_dir=str(tmp_path), sacc_path=des_sacc, data_types=["galaxy_density_xi"])
    assert len(r.files) == 1 and r.metadata["counts"] == {"galaxy_density_xi": 54}
    text = Path(r.files[0]).read_text().splitlines()
    assert text[0].startswith("# label:") and "theta_arcmin" in text[3]
    with pytest.raises(ValueError):
        sacc_to_csv(output_dir=str(tmp_path), sacc_path=des_sacc, data_types=["nope"])


def test_inspect_txpipe_like(txpipe_sacc, tmp_path):
    r = sacc_inspect(output_dir=str(tmp_path), sacc_path=txpipe_sacc)
    m = r.metadata
    assert m["space"] == "harmonic" and not m["covariance"]["present"]
    assert not m["firecrown_ready"] and "sacc_prepare_for_firecrown" in r.message
    kinds = {t["name"]: t["inferred_kind"] for t in m["tracers"]}
    assert kinds == {"source_0": "shear", "source_1": "shear", "lens_0": "density"}


@pytest.mark.parametrize("fmt", ["hdf5", "fits"])
def test_prepare_for_firecrown(txpipe_sacc, tmp_path, fmt):
    keep = ["galaxy_shear_cl_ee", "galaxy_shearDensity_cl_e", "galaxy_density_cl"]
    r = sacc_prepare_for_firecrown(output_dir=str(tmp_path / fmt), sacc_path=txpipe_sacc,
                                   keep_data_types=keep, output_format=fmt)
    m = r.metadata
    assert m["renamed"] == {"source_0": "src0", "source_1": "src1", "lens_0": "lens0"}
    assert m["quantities"] == {"src0": "galaxy_shear", "src1": "galaxy_shear", "lens0": "galaxy_density"}
    assert m["dropped_data_types"] == ["galaxy_shear_cl_bb"] and m["n_points"] == 30
    assert m["firecrown_ready"] and not m["covariance"]["present"]
    out = Path(r.files[0])
    assert out.suffix == ("." + fmt) and out.is_file()
    s = sacc.Sacc.load(str(out))
    assert set(s.tracers) == {"src0", "src1", "lens0"} and set(s.get_data_types()) == set(keep)
    assert s.tracers["lens0"].quantity == "galaxy_density"
    # the input was not touched
    assert set(sacc.Sacc.load(txpipe_sacc).tracers) == {"source_0", "source_1", "lens_0"}
    # explicit mapping wins
    r2 = sacc_prepare_for_firecrown(output_dir=str(tmp_path / "map"), sacc_path=txpipe_sacc,
                                    tracer_map={"source_0": "src7"}, output_name="renamed")
    assert r2.metadata["renamed"]["source_0"] == "src7" and Path(r2.files[0]).name == "renamed.hdf5"


def test_attach_gaussian_covariance(txpipe_sacc, tmp_path):
    pytest.importorskip("pyccl")
    prep = sacc_prepare_for_firecrown(
        output_dir=str(tmp_path), sacc_path=txpipe_sacc,
        keep_data_types=["galaxy_shear_cl_ee", "galaxy_shearDensity_cl_e", "galaxy_density_cl"]).files[0]
    r = sacc_attach_gaussian_covariance(output_dir=str(tmp_path), sacc_path=prep, f_sky=0.1,
                                        n_gal={"src0": 5.0, "src1": 5.0, "lens0": 10.0}, galaxy_bias=1.5)
    m = r.metadata
    assert m["covariance"]["present"] and m["covariance"]["positive_definite"]
    assert m["covariance"]["shape"] == [30, 30] and not m["covariance"]["is_diagonal"]
    assert m["snr_total"] > 0 and m["tracer_kinds"] == {"lens0": "density", "src0": "shear", "src1": "shear"}
    s = sacc.Sacc.load(r.files[0])
    cov = s.covariance.dense
    assert np.allclose(cov, cov.T)
    ells = np.array([d.get_tag("ell") for d in s.data])
    # no coupling between different ell bins; coupling within one ell bin
    diff = ells[:, None] != ells[None, :]
    assert np.all(cov[diff] == 0.0)
    assert np.any(np.abs(cov[~diff] - np.diag(np.diag(cov))[~diff]) > 0)
    # hand check of one auto-shear diagonal element: 2 (C + N)^2 / ((2l+1) dl f_sky)
    i = s.indices("galaxy_shear_cl_ee", ("src0", "src0"))[0]
    arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2
    noise = 0.26 ** 2 / (5.0 * arcmin2_per_sr)
    assert cov[i, i] > 2 * noise ** 2 / ((2 * 100 + 1) * 80 * 0.1) * 0.99  # signal only adds
    # refusals
    with pytest.raises(ValueError, match="already has a covariance"):
        sacc_attach_gaussian_covariance(output_dir=str(tmp_path), sacc_path=r.files[0], f_sky=0.1, n_gal=5.0)
    with pytest.raises(ValueError, match="n_gal missing"):
        sacc_attach_gaussian_covariance(output_dir=str(tmp_path), sacc_path=prep, f_sky=0.1, n_gal={"src0": 5.0})
    with pytest.raises(ValueError, match="unsupported data types"):
        sacc_attach_gaussian_covariance(output_dir=str(tmp_path), sacc_path=txpipe_sacc, f_sky=0.1, n_gal=5.0)
    json.dumps(m)


def test_attach_refuses_real_space(des_sacc, tmp_path):
    with pytest.raises(ValueError, match="real-space"):
        sacc_attach_gaussian_covariance(output_dir=str(tmp_path), sacc_path=des_sacc, f_sky=0.1, n_gal=5.0)


def test_missing_file_error(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        sacc_inspect(output_dir=str(tmp_path), sacc_path=str(tmp_path / "nope.hdf5"))
