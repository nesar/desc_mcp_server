"""smokescreen_tools: concealment through a firecrown experiment, inspection,
encryption round trip. Uses the DES-Y1-like real-space test data vector
(firecrown/tests/sacc_data.hdf5) like the firecrown tests.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("smokescreen")
sacc = pytest.importorskip("sacc")
pytest.importorskip("firecrown")

from tools.common import clone_dir  # noqa: E402
from tools.firecrown_tools import firecrown_build_likelihood, firecrown_compute_loglike  # noqa: E402
from tools.smokescreen_tools import (  # noqa: E402
    smokescreen_conceal_datavector,
    smokescreen_decrypt_file,
    smokescreen_encrypt_file,
    smokescreen_inspect,
)

_FC = clone_dir("firecrown")
_DES = _FC / "tests" / "sacc_data.hdf5" if _FC else None
if _DES is None or not _DES.is_file():
    pytest.skip("firecrown clone with tests/sacc_data.hdf5 not available", allow_module_level=True)

NUISANCE = {"lens0_bias": 1.4, "lens1_bias": 1.6, "lens2_bias": 1.6, "lens3_bias": 1.9, "lens4_bias": 2.0}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    work = tmp_path_factory.mktemp("conceal")
    shutil.copy(_DES, work / "sacc_data.hdf5")
    r = firecrown_build_likelihood(output_dir=str(work), sacc_path=str(work / "sacc_data.hdf5"))
    yaml_file = next(f for f in r.files if f.endswith(".yaml"))
    return work, yaml_file


def test_conceal_hides_values_and_changes_data(experiment):
    work, yaml_file = experiment
    r = smokescreen_conceal_datavector(output_dir=str(work / "out"), experiment_yaml=yaml_file,
                                       shifts={"Omega_c": [0.20, 0.32], "sigma8": [0.72, 0.90]},
                                       seed="unit-test-seed", nuisance=NUISANCE, encrypt_original=True)
    m = r.metadata
    concealed = Path(m["concealed_file"])
    assert concealed.is_file() and concealed.name.endswith("_concealed.hdf5")
    assert m["shifted_parameters"] == ["Omega_c", "sigma8"] and m["deterministic_parameters"] == []
    # nothing hidden leaks: no seed, no drawn values anywhere in message/metadata/record
    blob = json.dumps(m) + r.message
    assert "unit-test-seed" not in blob
    record = json.loads(Path(next(f for f in r.files if f.endswith("_concealment_record.json"))).read_text())
    assert record["seed"].startswith("withheld") and "unit-test-seed" not in json.dumps(record)
    assert set(record["ranges"]["Omega_c"]) == {"kind", "range"}
    # the data vector changed, the original did not
    orig = sacc.Sacc.load(str(work / "sacc_data.hdf5"))
    conc = sacc.Sacc.load(str(concealed))
    assert len(conc.data) == len(orig.data) == 457
    assert not np.allclose(conc.mean, orig.mean)
    assert np.allclose(sacc.Sacc.load(str(work / "sacc_data.hdf5")).mean, orig.mean)
    assert conc.metadata["concealed"] is True or conc.metadata["concealed"] == "True"
    # encrypted copy + key written, original kept
    assert any(f.endswith(".encrpt") for f in r.files) and any(f.endswith(".key") for f in r.files)
    assert (work / "sacc_data.hdf5").is_file()
    # the generated likelihood module is importable firecrown glue
    mod = next(f for f in r.files if f.endswith(".py"))
    assert "def build_likelihood" in Path(mod).read_text()
    # the concealed file feeds the normal firecrown path
    r2 = firecrown_build_likelihood(output_dir=str(work / "out"), sacc_path=str(concealed), name="concealed")
    yaml2 = next(f for f in r2.files if f.endswith(".yaml"))
    ll = firecrown_compute_loglike(output_dir=str(work / "out"), experiment_yaml=yaml2, nuisance=NUISANCE)
    assert np.isfinite(ll.metadata["chi2"])


def test_conceal_is_deterministic_in_seed(experiment):
    work, yaml_file = experiment
    kw = dict(experiment_yaml=yaml_file, shifts={"sigma8": [0.72, 0.90]}, nuisance=NUISANCE)
    a = smokescreen_conceal_datavector(output_dir=str(work / "s1"), seed="same", **kw)
    b = smokescreen_conceal_datavector(output_dir=str(work / "s2"), seed="same", **kw)
    c = smokescreen_conceal_datavector(output_dir=str(work / "s3"), seed="other", **kw)
    ma, mb, mc = (sacc.Sacc.load(x.metadata["concealed_file"]).mean for x in (a, b, c))
    assert np.allclose(ma, mb) and not np.allclose(ma, mc)


def test_conceal_guardrails(experiment):
    work, yaml_file = experiment
    with pytest.raises(ValueError):  # not a cosmological parameter
        smokescreen_conceal_datavector(output_dir=str(work), experiment_yaml=yaml_file, seed=1,
                                       shifts={"lens0_bias": [1.0, 2.0]})
    with pytest.raises(ValueError):  # wrong amplitude parameter for a sigma8 experiment
        smokescreen_conceal_datavector(output_dir=str(work), experiment_yaml=yaml_file, seed=1,
                                       shifts={"A_s": [1e-9, 3e-9]})
    with pytest.raises(ValueError):  # lo >= hi
        smokescreen_conceal_datavector(output_dir=str(work), experiment_yaml=yaml_file, seed=1,
                                       shifts={"sigma8": [0.9, 0.7]})
    # an experiment with scale cuts does not cover the file -> refused with guidance
    cut = firecrown_build_likelihood(output_dir=str(work), sacc_path=str(work / "sacc_data.hdf5"), name="cut",
                                     scale_cuts=[{"tracer": "src0", "measurement": "shear", "lower": 10, "upper": 60}])
    with pytest.raises(ValueError, match="cover"):
        smokescreen_conceal_datavector(output_dir=str(work), seed=1, shifts={"sigma8": [0.7, 0.9]},
                                       experiment_yaml=next(f for f in cut.files if f.endswith(".yaml")))


def test_inspect_and_encrypt_roundtrip(experiment, tmp_path):
    work, _ = experiment
    plain = str(work / "sacc_data.hdf5")
    ins = smokescreen_inspect(output_dir=str(tmp_path), sacc_path=plain)
    assert ins.metadata["concealed"] is False and ins.metadata["encrypted"] is False
    enc = smokescreen_encrypt_file(output_dir=str(tmp_path), path=plain)
    epath = next(f for f in enc.files if f.endswith(".encrpt"))
    kpath = next(f for f in enc.files if f.endswith(".key"))
    assert Path(plain).is_file()  # never deleted
    assert smokescreen_inspect(output_dir=str(tmp_path), sacc_path=epath).metadata["encrypted"] is True
    dec = smokescreen_decrypt_file(output_dir=str(tmp_path), path=epath, key_path=kpath)
    assert dec.metadata["format"] == "hdf5"
    assert hashlib.md5(Path(dec.files[0]).read_bytes()).hexdigest() == hashlib.md5(Path(plain).read_bytes()).hexdigest()
