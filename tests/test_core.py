"""Core contracts: cosmology spec, env-kernel runner, meta tools, dispatch pack."""

import json

import pytest

from tools.cosmology import CosmologyParams, PRESETS
from tools.envkernel import ENV_SETUP_CANDIDATES, run_in_env
from tools.meta import convert_cosmology_names, list_desc_packages, list_desc_skills, load_desc_skill


def test_cosmology_presets_and_views():
    spec = CosmologyParams(preset="planck18")
    p = spec.firecrown_params()
    assert p["Omega_c"] == PRESETS["planck18"]["Omega_c"]
    assert "sigma8" in p and "A_s" not in p
    assert spec.ccl_kwargs()["transfer_function"] == "boltzmann_camb"
    # explicit field overrides the preset
    assert CosmologyParams(preset="planck18", h=0.7).h == 0.7
    # JSON round trip (the MCP boundary)
    again = CosmologyParams.model_validate(json.loads(spec.model_dump_json()))
    assert again.slug() == spec.slug()


def test_cosmology_validation():
    with pytest.raises(ValueError):
        CosmologyParams(A_s=2e-9)  # both amplitudes
    with pytest.raises(ValueError):
        CosmologyParams(A_s=2e-9, sigma8=None, transfer_function="bbks")
    with pytest.raises(ValueError):
        CosmologyParams(mg_mu0=0.1, matter_power_spectrum="camb_hmcode")
    hm = CosmologyParams(matter_power_spectrum="camb_hmcode")
    assert hm.ccl_kwargs()["extra_parameters"]["camb"]["halofit_version"] == "mead2020_feedback"


def test_cosmology_builds_pyccl():
    pytest.importorskip("pyccl")
    spec = CosmologyParams(transfer_function="eisenstein_hu")
    d = spec.describe()
    assert abs(d["derived"]["Omega_m"] - 0.30) < 1e-6
    assert spec.build() is spec.build()  # cached


def test_env_kernel_roundtrip(tmp_path):
    out = run_in_env("true", "noop", {"x": 1, "y": [1.5, 2.5]}, job_dir=str(tmp_path))
    assert isinstance(out, dict), out
    assert out["result"]["echo"] == {"x": 1, "y": [1.5, 2.5]}
    assert "python" in out["env_check"]


def test_env_kernel_requires_env_setup(tmp_path):
    out = run_in_env("", "noop", {}, job_dir=str(tmp_path))
    assert isinstance(out, str) and out.startswith("Error")


def test_env_kernel_failure_is_error_string(tmp_path):
    out = run_in_env("true", "does_not_exist", {}, job_dir=str(tmp_path))
    assert isinstance(out, str) and out.startswith("Error")


def test_env_candidates_hold_no_identity():
    blob = json.dumps(ENV_SETUP_CANDIDATES)
    assert "nesar" not in blob.lower()
    for site, cands in ENV_SETUP_CANDIDATES.items():
        for c in cands:
            assert {"name", "env_setup", "provides", "suitable_for"} <= set(c)


def test_meta_tools():
    res = list_desc_packages()
    assert "pyccl" in res.metadata["packages"]
    skills = list_desc_skills().metadata["skills"]
    assert "hpc-dispatch-handoff" in skills and "desc-tour" in skills
    text = load_desc_skill("desc-tour").metadata["instructions"]
    assert text.startswith("---")
    with pytest.raises(ValueError):
        load_desc_skill("no-such-skill")


def test_convert_names():
    base = {"Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67, "n_s": 0.96, "sigma8": 0.81, "m_nu": 0.06}
    emu = convert_cosmology_names(base, to="emulator").metadata["params"]
    assert abs(emu["Om"] - (0.30 + 0.06 / (93.14 * 0.67**2))) < 1e-9
    cob = convert_cosmology_names(base, to="cobaya").metadata["params"]
    assert abs(cob["ombh2"] - 0.05 * 0.67**2) < 1e-12 and cob["H0"] == 67.0
    cs = convert_cosmology_names(base, to="cosmosis").metadata["params"]
    assert cs["sigma_8"] == 0.81 and cs["h0"] == 0.67
