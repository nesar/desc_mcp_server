"""augur_tools: config generation, validation, synthetic data vector, Fisher, contours.

Runs in-process with the desc-mcp python from the repo root
(`python -m pytest tests/test_augur_tools.py -q`); the whole file takes ~15 s
(eisenstein_hu + halofit, 2 source bins, 3 varied parameters).
"""

import json
import sys

import numpy as np
import pytest

pytest.importorskip("augur")
pytest.importorskip("firecrown")

from tools.augur_tools import (  # noqa: E402
    augur_compute_fisher, augur_generate_forecast_config, augur_generate_synthetic_datavector,
    augur_list_examples, augur_plot_fisher_contours, augur_validate_forecast_config,
    read_fisher_csv, DISPATCH_KERNELS, CONVENTIONS, CAVEATS,
)
from tools.common import clone_dir, read_csv  # noqa: E402


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("augur")


@pytest.fixture(scope="module")
def shear_config(workdir):
    """Y1 shear-only, 2 bins, eisenstein_hu + halofit, gaus_internal, 3 var_pars."""
    r = augur_generate_forecast_config(
        str(workdir), survey_year="Y1", probes="shear", n_source_bins=2,
        var_pars=["Omega_c", "sigma8", "w0"], step=0.01, derivative_method="numdifftools",
        cov_type="gaus_internal", srd_priors=True, fisher_bias_params={"Omega_c": 0.27},
        label="y1shear")
    assert r.status == "success"
    assert r.metadata["estimated_likelihood_evaluations"] == 2 * 3 + 1 + 1 + 2
    assert set(r.metadata["gaussian_priors"]) == {"Omega_c", "sigma8", "w0"}  # trimmed to var_pars
    return r.metadata["config_path"]


def test_list_examples_flags_shipped_prior_trap():
    r = augur_list_examples()
    ex = r.metadata["examples"]
    assert "srd_y1_3x2.yml" in ex and "srd_y10_3x2.yml" in ex
    assert ex["srd_y1_3x2.yml"]["runs_as_shipped"] is False
    assert "lens2_bias" in ex["srd_y1_3x2.yml"]["priors_not_in_var_pars"]
    assert ex["srd_y1_3x2.yml"]["cov_type"] == "SRD"
    assert r.metadata["data_files"]["Y1_3x2_SRD_cov.npy"]["shape"] == [540, 540]
    assert CONVENTIONS["kmax"].startswith("Mpc^-1")
    # lock-kernel (contract R8): runs in the pack's own environment, env_setup only overrides
    assert DISPATCH_KERNELS["augur_forecast"]["env_setup_required"] is False
    assert DISPATCH_KERNELS["augur_forecast"]["function"] == "inner.augur_forecast.main"
    assert len(CAVEATS) >= 5


def test_generated_config_is_jinja_free_and_absolute(shear_config):
    import yaml
    text = open(shear_config, encoding="utf-8").read()
    assert "{{" not in text
    cfg = yaml.safe_load(text)
    assert cfg["sources"]["Nz_kwargs"]["input_file"].startswith("/")
    assert cfg["fiducial_sacc_path"].startswith("/")
    assert cfg["cosmo"]["transfer_function"] == "eisenstein_hu"
    assert cfg["cosmo"]["matter_power_spectrum"] == "halofit"
    assert "sigma8" in cfg["cosmo"] and "A_s" not in cfg["cosmo"]
    assert cfg["general"]["ignore_scale_cuts_likelihood"] is False
    assert cfg["fisher"]["step"] == 0.01
    assert "kmax" not in cfg["statistics"]["galaxy_shear_cl_ee"]


def test_validate_go(shear_config):
    v = augur_validate_forecast_config(shear_config)
    assert v.metadata["verdict"] == "go", v.metadata["errors"]
    assert v.metadata["n_data_before_cuts"] == 3 * 20
    assert v.metadata["estimated_likelihood_evaluations"] == 10


def test_validator_catches_traps(workdir):
    import yaml
    base = yaml.safe_load(open(augur_generate_forecast_config(
        str(workdir), survey_year="Y1", probes="3x2pt", label="trap").metadata["config_path"]))
    # 1) prior not in var_pars
    bad = json.loads(json.dumps(base))
    bad["fisher"]["gaussian_priors"] = {"lens3_bias": 0.9}
    # 2) both sigma8 and A_s
    bad["cosmo"]["A_s"] = 2.1e-9
    # 3) kmax and lmax keys together
    bad["statistics"]["galaxy_density_cl"]["lmax"] = 1000
    # 4) silent-systematics-drop flag
    bad["general"]["ignore_scale_cuts_likelihood"] = True
    # 5) camb without halofit_version
    bad["cosmo"]["transfer_function"] = "boltzmann_camb"
    # 6) unknown Nz_type
    bad["lenses"]["Nz_type"] = "TopHat"
    p = workdir / "bad.yml"
    p.write_text(yaml.safe_dump(bad))
    v = augur_validate_forecast_config(str(p))
    errs = " | ".join(v.metadata["errors"])
    assert v.metadata["verdict"] == "no-go"
    for needle in ("gaussian_priors", "sigma8 / A_s", "kmax", "ignore_scale_cuts_likelihood",
                   "halofit_version", "Nz_type"):
        assert needle in errs, needle
    # SRD covariance with a non-SRD layout
    srd = json.loads(json.dumps(base))
    srd["cov_options"] = {"cov_type": "SRD",
                          "SRD_cov_path": str(clone_dir("augur") / "data" / "Y1_3x2_SRD_cov.npy")}
    srd["sources"]["nbins"] = 3
    p2 = workdir / "bad_srd.yml"
    p2.write_text(yaml.safe_dump(srd))
    v2 = augur_validate_forecast_config(str(p2))
    assert v2.metadata["verdict"] == "no-go"
    assert any("SRD" in e and "bins" in e for e in v2.metadata["errors"])


def test_validator_rejects_shipped_srd_y1_example(workdir):
    clone = clone_dir("augur")
    if clone is None:
        pytest.skip("augur clone not available")
    text = (clone / "examples" / "srd_y1_3x2.yml").read_text(encoding="utf-8")
    text = text.replace("{{ env['AUGUR_DIR'] }}", str(clone))
    p = workdir / "srd_y1_3x2_rendered.yml"
    p.write_text(text, encoding="utf-8")
    v = augur_validate_forecast_config(str(p))
    assert v.metadata["verdict"] == "no-go"
    assert any("gaussian_priors" in e and "lens2_bias" in e for e in v.metadata["errors"])
    # and the SRD layout itself is fine (only the prior list is wrong)
    assert not any("SRD" in e for e in v.metadata["errors"])


def test_srd_layout_config_matches_covariance(workdir):
    r = augur_generate_forecast_config(str(workdir), survey_year="Y10", probes="3x2pt",
                                       cov_type="SRD", label="y10srd")
    assert r.metadata["srd_covariance_layout_match"] is True
    assert r.metadata["n_data_before_cuts"] == 50 * 20
    assert r.metadata["n_lens_bins"] == 10
    v = augur_validate_forecast_config(r.metadata["config_path"])
    assert v.metadata["verdict"] == "go", v.metadata["errors"]


def test_generate_config_rejects_bad_inputs(workdir):
    with pytest.raises(ValueError):
        augur_generate_forecast_config(str(workdir), probes="shear", var_pars=["lens0_bias"])
    with pytest.raises(ValueError):
        augur_generate_forecast_config(str(workdir), var_pars=["Omega_c"], gaussian_priors={"w0": 0.8})
    with pytest.raises(ValueError):
        augur_generate_forecast_config(str(workdir), n_source_bins=7, nz_mode="file")


def test_synthetic_datavector(workdir, shear_config):
    r = augur_generate_synthetic_datavector(str(workdir), shear_config, plot=True)
    assert r.status == "success"
    assert r.metadata["n_data"] == 60 and r.metadata["n_data_in_likelihood"] == 60
    assert r.metadata["computed_on"] == "local"
    assert 50 < r.metadata["total_snr"] < 2000
    csv = [f for f in r.files if "datavector_" in f and f.endswith(".csv")][0]
    lines = [ln for ln in open(csv) if not ln.startswith("#")]
    assert lines[0].strip() == "stat,tr1,tr2,ell,cl,sigma,in_likelihood"
    assert len(lines) == 61
    cov = np.load([f for f in r.files if f.endswith(".npy")][0])
    assert cov.shape == (60, 60) and np.all(np.diag(cov) > 0)
    nz = [f for f in r.files if "nz_sources" in f][0]
    _, cols = read_csv(nz)
    assert "nz_src0" in cols and "nz_src1" in cols and cols["z"].max() > 3
    assert any(f.endswith(".png") for f in r.files)
    assert any(f.endswith(".sacc") for f in r.files)


def test_compute_fisher_and_contours(workdir, shear_config):
    r = augur_compute_fisher(str(workdir), shear_config)
    assert r.status == "success"
    m = r.metadata
    assert m["params"] == ["Omega_c", "sigma8", "w0"]
    assert m["checks"]["positive_definite"] and m["checks"]["symmetric_rel_err"] < 1e-10
    assert m["n_evaluations"] == 7
    assert 0.005 <= m["sigma_no_prior"]["sigma8"] <= 0.1
    assert m["sigma_with_prior"]["sigma8"] <= m["sigma_no_prior"]["sigma8"]
    assert m["fisher_bias"]["bias_params"] == {"Omega_c": 0.27}
    assert m["fom"]["no_prior"] is None  # wa not varied -> no w0-wa FoM
    assert "S8" in m["derived"]["no_prior"] and m["derived"]["no_prior"]["S8"]["sigma"] > 0
    fcsv = [f for f in r.files if "/fisher_" in f and "with_priors" not in f and "bias" not in f][0]
    names, F, fid, header = read_fisher_csv(fcsv)
    assert names == m["params"] and F.shape == (3, 3)
    assert abs(fid["sigma8"] - 0.831) < 1e-9
    assert np.allclose(F, F.T)
    assert np.all(np.linalg.eigvalsh(F) > 0)
    sig = np.sqrt(np.diag(np.linalg.inv(F)))
    assert abs(sig[1] - m["sigma_no_prior"]["sigma8"]) < 1e-6
    ms = [f for f in r.files if "marginalized_sigmas" in f][0]
    lines = [ln for ln in open(ms) if not ln.startswith("#")]
    assert lines[0].strip() == "param,fiducial,sigma_no_prior,sigma_with_prior,prior"
    der = np.load([f for f in r.files if "derivatives" in f][0])
    assert der.shape == (3, 60)
    fom = json.load(open([f for f in r.files if "fom_" in f][0]))
    assert "DETF" in fom["fom_definition"]
    fp = [f for f in r.files if "fisher_with_priors" in f][0]
    p = augur_plot_fisher_contours(str(workdir), [fcsv, fp], params=["Omega_c", "sigma8", "w0"],
                                   labels=["no prior", "SRD priors"])
    assert p.files[0].endswith(".png")
    assert len(p.metadata["per_file"]) == 2
    assert abs(p.metadata["per_file"][0]["sigma"]["w0"] - m["sigma_no_prior"]["w0"]) < 1e-6
    # subset + fiducial override
    p2 = augur_plot_fisher_contours(str(workdir), [fcsv], params=["sigma8", "w0"], fiducials={"w0": -0.99})
    assert p2.status == "success"


def test_inner_kernel_json_roundtrip(workdir, shear_config):
    """The env-kernel path: tools.inner.augur_forecast.main through tools.envkernel.run_in_env."""
    import yaml
    from tools.envkernel import run_in_env
    cfg = yaml.safe_load(open(shear_config))
    cfg["fiducial_sacc_path"] = str(workdir / "kernel_fid.sacc")
    cfg["fisher"]["output"] = str(workdir / "kernel_out" / "fisher.dat")
    cfg["fisher"]["fid_output"] = str(workdir / "kernel_out" / "fiducials.dat")
    cfg["fisher"].pop("fisher_bias", None)
    params = {"config": cfg, "mode": "fisher", "max_cov_return": 0}
    # env_setup that makes `python` resolve to this interpreter (no facility env here)
    env_setup = f'python() {{ "{sys.executable}" "$@"; }}'
    out = run_in_env(env_setup, "augur_forecast", params, job_dir=str(workdir), timeout_s=600)
    assert isinstance(out, dict), out
    res = out["result"]
    assert out["env_check"]["augur"] is not None
    assert res["param_names"] == ["Omega_c", "sigma8", "w0"]
    F = np.asarray(res["fisher"])
    assert F.shape == (3, 3) and np.all(np.linalg.eigvalsh(F) > 0)
    assert "cov" not in res and len(res["cov_diag"]) == 60
    json.dumps(res)  # JSON-safe
    # relocate_outputs + inline files (the remote branch) also work locally
    import os
    from tools.augur_tools import _inline_files_for_remote
    from tools.inner.augur_forecast import main
    params2 = {"config": cfg, "mode": "generate", "inline_files": _inline_files_for_remote(cfg),
               "relocate_outputs": True, "job_dir": str(workdir / "remote_like")}
    os.makedirs(workdir / "remote_like", exist_ok=True)
    res2 = main(params2)
    assert res2["sacc_path"].startswith(str(workdir / "remote_like"))
    assert "srd_source_bins_y1.txt" in res2["inline_files_written"]
    assert res2["n_data"] == 60
