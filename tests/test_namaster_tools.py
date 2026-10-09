"""namaster_tools: mask properties, simulated maps, bandpowers -> sacc with windows.

Everything runs on a small nside-64 simulation built in tmp_path (seconds).
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymaster")
pytest.importorskip("healpy")
sacc = pytest.importorskip("sacc")

from tools.common import read_csv  # noqa: E402
from tools.namaster_tools import namaster_compute_cls, namaster_mask_properties, namaster_simulate_maps  # noqa: E402

TRACERS = [{"name": "src0", "kind": "shear", "z_mean": 0.9, "z_sigma": 0.25, "n_gal_arcmin2": 8.0},
           {"name": "lens0", "kind": "density", "z_mean": 0.5, "z_sigma": 0.1, "bias": 1.6, "n_gal_arcmin2": 4.0}]
COSMO = {"transfer_function": "eisenstein_hu"}


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    r = namaster_simulate_maps(output_dir=str(out), nside=64, f_sky=0.3, seed=1, tracers=TRACERS, cosmology=COSMO)
    assert r.status == "success"
    return r


def test_simulate_writes_everything(sim):
    m = sim.metadata
    assert m["nside"] == 64 and m["tracers"] == ["src0", "lens0"]
    assert abs(m["f_sky"] - 0.3) < 0.02
    spec = json.loads(Path(m["fields_json"]).read_text())
    assert [f["name"] for f in spec["fields"]] == ["src0", "lens0"]
    assert len(spec["fields"][0]["map_paths"]) == 2 and len(spec["fields"][1]["map_paths"]) == 1
    for f in spec["fields"]:
        assert Path(f["mask_path"]).is_file() and all(Path(p).is_file() for p in f["map_paths"])
        assert Path(f["nz_csv"]).is_file()
    _, cols = read_csv(m["theory_cls_csv"])
    assert {"ell", "src0|src0", "src0|lens0", "lens0|lens0"} <= set(cols)
    assert cols["src0|src0"][2:].min() > 0  # E-mode shear power is positive


def test_mask_properties(sim, tmp_path):
    r = namaster_mask_properties(output_dir=str(tmp_path), mask_path=sim.metadata["mask"], apodize_deg=3.0)
    raw, apod = r.metadata["raw"], r.metadata["apodized"]
    assert abs(raw["fsky_raw"] - 0.3) < 0.02 and abs(raw["fsky_eff"] - raw["fsky_raw"]) < 1e-9  # binary mask
    assert 0 < apod["fsky_eff"] < raw["fsky_eff"]  # apodization costs area
    assert any(f.endswith(".fits") for f in r.files) and any(f.endswith(".png") for f in r.files)


def test_compute_cls_sacc_and_theory(sim, tmp_path):
    # default ell_max = 2*nside = 128 (above it pixelization biases the top bins)
    r = namaster_compute_cls(output_dir=str(tmp_path), fields_json=sim.metadata["fields_json"],
                             binning={"scheme": "linear", "nlb": 14, "ell_min": 8},
                             theory_csv=sim.metadata["theory_cls_csv"])
    m = r.metadata
    assert m["ell_edges"][0] == 8 and m["ell_edges"][-1] == 129 and m["lmax"] == 128
    assert m["n_bins"] == len(m["ell_edges"]) - 1 >= 4
    assert m["windows_in_sacc"] is True and m["computed_on"] == "local"
    # white-noise subtraction recorded for both fields
    assert m["fields"]["src0"]["noise_coupled"] > 0 and m["fields"]["lens0"]["noise_coupled"] > 0
    s = sacc.Sacc.load(m["sacc"])
    assert sorted(s.tracers) == ["lens0", "src0"]
    assert s.tracers["src0"].quantity == "galaxy_shear" and s.tracers["lens0"].quantity == "galaxy_density"
    assert "n_ell_coupled" in s.tracers["src0"].metadata and "binning/ell_edges" in s.metadata
    types = s.get_data_types()
    assert {"galaxy_shear_cl_ee", "galaxy_shear_cl_bb", "galaxy_shearDensity_cl_e", "galaxy_density_cl"} <= set(types)
    # shear first in the cross type; windows attached and normalised
    combo = s.get_tracer_combinations("galaxy_shearDensity_cl_e")[0]
    assert tuple(combo) == ("src0", "lens0")
    idx = s.indices("galaxy_density_cl", ("lens0", "lens0"))
    w = s.get_bandpower_windows(idx)
    assert w is not None and w.weight.shape[1] == len(idx)
    assert np.allclose(w.weight.sum(axis=0), 1.0, atol=0.05)
    # measured density bandpowers agree with the binned input theory within the
    # Gaussian scatter (loose: nside 64, f_sky 0.3)
    pair = next(p for p in m["pairs"] if p["a"] == "lens0" and p["b"] == "lens0")
    _, cols = read_csv(pair["csv"])
    meas, theo = cols["TT"], cols["theory_binned"]
    ratio = meas[1:] / theo[1:]
    assert np.all(np.abs(ratio - 1.0) < 0.5), ratio
    # B-modes of a pure E-mode simulation are small compared to EE
    ee = s.mean[s.indices("galaxy_shear_cl_ee", ("src0", "src0"))]
    bb = s.mean[s.indices("galaxy_shear_cl_bb", ("src0", "src0"))]
    assert np.median(np.abs(bb)) < 0.5 * np.median(np.abs(ee))


def test_compute_cls_without_b_modes_and_explicit_fields(sim, tmp_path):
    spec = json.loads(Path(sim.metadata["fields_json"]).read_text())["fields"]
    fields = [{k: v for k, v in f.items() if k in ("name", "kind", "mask_path", "map_paths", "n_gal_arcmin2",
                                                   "sigma_e", "nz_csv", "nz_column")} for f in spec]
    r = namaster_compute_cls(output_dir=str(tmp_path), fields=fields, pairs="auto", include_b_modes=False,
                             binning={"scheme": "log", "n_bins": 4, "ell_min": 8, "ell_max": 120}, plot=False)
    s = sacc.Sacc.load(r.metadata["sacc"])
    assert set(s.get_data_types()) == {"galaxy_shear_cl_ee", "galaxy_density_cl"}
    with pytest.raises(ValueError):
        namaster_compute_cls(output_dir=str(tmp_path), fields=fields, fields_json=sim.metadata["fields_json"])
    with pytest.raises(ValueError):
        namaster_compute_cls(output_dir=str(tmp_path), fields=fields, pairs=[["src0", "nope"]])
