"""txpipe_tools: AST stage index, pipeline composition/validation, run refusal.

Nothing here imports or runs TXPipe (it is not installed, by design). The
inner script is exercised with a fake `ceci` on PATH so its JSON round-trip
and stage classification are tested without the TXPipe stack.
"""

import json
import os
import stat

import pytest

from tools.common import clone_dir

pytestmark = pytest.mark.skipif(clone_dir("txpipe") is None, reason="TXPipe clone not available")

from tools import txpipe_tools as T  # noqa: E402
from tools.txpipe_tools.stage_index import find_stage, get_index  # noqa: E402


def test_index_size_and_metadata():
    idx = get_index()
    assert len(idx["stages"]) >= 150
    meta = idx["metadata"]
    assert len(meta["clone_commit"]) >= 7
    assert meta["parse_errors"] == []
    assert meta["groups"]["two-point"] >= 5 and meta["groups"]["extensions"] >= 5
    assert idx["examples"] and all("error" not in e for e in idx["examples"])


def test_describe_txtwopoint():
    r = T.txpipe_describe_stage("TXTwoPoint")
    tags_in = [t for t, _ in r.metadata["inputs"]]
    tags_out = [t for t, _ in r.metadata["outputs"]]
    assert "binned_shear_catalog" in tags_in
    assert "twopoint_data_real_raw" in tags_out
    cfg = r.metadata["config_options"]
    for key in ("min_sep", "max_sep", "nbins"):
        assert key in cfg and cfg[key]["dtype"] in ("float", "int")
    assert cfg["min_sep"]["default"] == 0.5 and cfg["nbins"]["default"] == 9
    assert "--binned_shear_catalog=" in r.metadata["cli_template"]
    assert r.metadata["output_filenames"]["twopoint_data_real_raw"] == "twopoint_data_real_raw.sacc"
    assert any(p.endswith("metadetect/pipeline.yml") for p in r.metadata["used_in_examples"])


def test_inheritance_and_name_tags():
    # subclass without its own inputs inherits them; required bare options
    sel = find_stage("TXSourceSelectorMetadetect")
    assert [t for t, _ in sel["inputs"]] == ["shear_catalog", "shear_tomography_classifier"]
    assert sel["config_options"]["source_zbin_edges"]["required"] is True
    # tag given by a class attribute name (output_main) resolves
    assert find_stage("TXTwoPointFourier")["outputs"] == [["twopoint_data_fourier", "SACCFile"]]
    # map stages pull the shared module-level dict
    assert "nside" in find_stage("TXSourceMaps")["config_options"]
    assert find_stage("TXSourceMaps")["dask_parallel"] is True


def test_list_stages_and_examples():
    r = T.txpipe_list_stages(group="covariance")
    assert {s["name"] for s in r.metadata["stages"]} >= {"TXRealGaussianCovariance", "TXFourierGaussianCovariance"}
    with pytest.raises(ValueError):
        T.txpipe_list_stages(group="nonsense")
    ex = T.txpipe_list_examples().metadata["examples"]
    mock = next(e for e in ex if e["path"] == "examples/mock_shear/pipeline.yml")
    assert mock["n_stages"] == 6 and mock["site"] == "local"
    assert "mock_shear_catalog" in mock["inputs"]


@pytest.mark.parametrize("preset,kwargs,n_min", [
    ("mock_shear", {}, 6),
    ("3x2pt_real", {"min_sep": 2.5, "max_sep": 100.0, "nbins": 12, "source_zbin_edges": [0.3, 0.6, 0.9, 1.2]}, 20),
])
def test_generate_and_validate_presets(tmp_path, preset, kwargs, n_min):
    r = T.txpipe_generate_pipeline(str(tmp_path), preset=preset, **kwargs)
    assert len(r.files) == 3 and all(os.path.isfile(f) for f in r.files)
    assert r.metadata["validation_ok"] and not r.metadata["unresolved_inputs"]
    assert len(r.metadata["stages"]) >= n_min
    # example data files are not present locally -> reported missing
    missing_tags = {m["tag"] for m in r.metadata["missing_input_files"]}
    expect = {"mock_shear_catalog", "spectroscopic_catalog"} if preset == "mock_shear" else {"shear_catalog", "photometry_catalog"}
    assert expect <= missing_tags
    assert all("data/example/inputs" in m["path"] for m in r.metadata["missing_input_files"])
    v = T.txpipe_validate_pipeline(r.metadata["pipeline_yml"], run_ceci_dry_run=False)
    assert v.metadata["ok"] and v.metadata["cycle"] is False
    assert len(v.metadata["commands"]) == len(r.metadata["stages"]) == len(v.metadata["order"])
    cmds = {c["stage"]: c["command"] for c in v.metadata["commands"]}
    assert all("python3 -m txpipe" in c or "python3 -m ceci" in c for c in cmds.values())
    if preset == "3x2pt_real":
        assert "TXTwoPoint" in cmds and "--twopoint_data_real_raw=" in cmds["TXTwoPoint"]
        assert v.metadata["order"].index("TXTwoPoint") < v.metadata["order"].index("TXNullBlinding")
        cfg = json.load(open(r.files[2]))["spec"]
        assert cfg["min_sep"] == 2.5
        import yaml
        config = yaml.safe_load(open(r.files[1]))
        assert config["TXTwoPoint"]["nbins"] == 12 and config["TXTwoPoint"]["sep_units"] == "arcmin"
        assert config["TXSourceSelectorMetadetect"]["source_zbin_edges"] == [0.3, 0.6, 0.9, 1.2]
        assert "nprocess" not in json.dumps(yaml.safe_load(open(r.files[0]))["stages"])
    # site/launcher are always local/mini
    import yaml
    p = yaml.safe_load(open(r.files[0]))
    assert p["site"] == {"name": "local", "max_threads": 2} and p["launcher"]["name"] == "mini"


def test_fourier_preset_with_covariance(tmp_path):
    r = T.txpipe_generate_pipeline(str(tmp_path), preset="3x2pt_fourier", ell_min=50, ell_max=300,
                                   n_ell=8, include_covariance=True)
    assert r.metadata["validation_ok"]
    assert {"TXTwoPointFourier", "TXFourierGaussianCovariance", "TXTwoPointPlotsFourier"} <= set(r.metadata["stages"])


def test_custom_preset_reports_unresolved(tmp_path):
    r = T.txpipe_generate_pipeline(str(tmp_path), preset="custom", stages=["TXTwoPointPlots"],
                                   inputs={"twopoint_data_real": "/nowhere/a.sacc"})
    assert not r.metadata["validation_ok"]
    assert any(u["tag"] == "twopoint_gamma_x" for u in r.metadata["unresolved_inputs"])


def test_validate_example_pipeline():
    path = clone_dir("txpipe") / "examples" / "metadetect" / "pipeline.yml"
    v = T.txpipe_validate_pipeline(str(path), run_ceci_dry_run=False)
    assert v.metadata["ok"] and v.metadata["n_stages"] == 61 and v.metadata["unknown_stages"] == []


def test_run_refuses_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("DESC_TXPIPE_ENV", raising=False)
    r = T.txpipe_generate_pipeline(str(tmp_path), preset="mock_shear")
    with pytest.raises(RuntimeError) as exc:
        T.txpipe_run_pipeline(str(tmp_path), r.metadata["pipeline_yml"])
    msg = str(exc.value)
    assert "DESC_TXPIPE_ENV" in msg and "env_setup" in msg and "set_dispatch" in msg


def test_inner_script_json_roundtrip(tmp_path, monkeypatch):
    """tools/inner/txpipe_run.main with a fake `ceci`: JSON in/out, manifest shape."""
    from tools.inner.txpipe_run import main
    from tools.txpipe_tools import _expected_outputs
    from tools.txpipe_tools.pipeline_compose import compose

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ceci = fake_bin / "ceci"
    ceci.write_text("#!/bin/bash\n"
                    "OUT=$(grep '^output_dir:' \"$1\" | sed 's/output_dir: //'); LOG=$(grep '^log_dir:' \"$1\" | sed 's/log_dir: //')\n"
                    "mkdir -p \"$OUT\" \"$LOG\"; echo x > \"$OUT/shear_catalog.hdf5\"; printf 'FITS' > \"$OUT/twopoint_data_real.sacc\"\n"
                    "echo x > \"$OUT/inprogress_shear_tomography_catalog.hdf5\"; echo ok > \"$LOG/TXSimpleMock.out\"; echo fake-ceci\n")
    ceci.chmod(ceci.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    c = compose({"preset": "mock_shear", "output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs")})
    params = json.loads(json.dumps({"pipeline": c["pipeline"], "config": c["config"], "max_threads": 2,
                                    "txpipe_dir": str(clone_dir("txpipe")), "job_dir": str(tmp_path / "job"),
                                    "stage_outputs": _expected_outputs(c["pipeline"])}))
    res = json.loads(json.dumps(main(params)))
    assert res["status"] == "success" and res["txpipe_source"] == "existing checkout"
    assert res["stages"]["TXSimpleMock"]["status"] == "complete"
    assert res["stages"]["TXSourceSelectorSimple"]["status"] == "inprogress"
    assert "twopoint_data_real.sacc" in res["inline_files"] and res["inline_files"]["twopoint_data_real.sacc"]["encoding"] == "base64"
    assert (tmp_path / "job" / "pipeline.yml").is_file() and (tmp_path / "job" / "config.yml").is_file()
    # the status tool reads the same layout
    # without output_dir/log_dir params the inner script keeps products in the job dir (facility behaviour)
    assert res["output_dir"] == str(tmp_path / "job" / "outputs")
    s = T.txpipe_run_status(run_output_dir=res["output_dir"], log_dir=res["log_dir"],
                            pipeline_yml=str(tmp_path / "job" / "pipeline.yml"))
    assert s.metadata["counts"]["complete"] == 1 and "twopoint_data_real.sacc" in s.metadata["sacc_files"]


def test_dispatch_kernel_manifest():
    k = T.DISPATCH_KERNELS["txpipe_run"]
    assert k["function"] == "envkernel.run_in_env" and k["env_setup_required"] is True
    assert k["inner"] == "txpipe_run" and os.path.isfile(os.path.join("tools", "inner", "txpipe_run.py"))
    assert T.txpipe_run_pipeline.weight == "heavy"
