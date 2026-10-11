"""The environment the pack carries (contract R8): environment.yml <-> conda-lock.yml
consistency, both platforms, python 3.12, and the engine-side detection."""

import re
from pathlib import Path

import pytest
import yaml

ENV = Path(__file__).resolve().parents[1] / "tools" / "env"


@pytest.fixture(scope="module")
def lock():
    path = ENV / "conda-lock.yml"
    if not path.is_file():
        pytest.fail("tools/env/conda-lock.yml missing: run `bash scripts/env.sh --relock`")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _spec_name(spec: str) -> str:
    return re.split(r"[<>=!~ @\[]", spec.strip(), 1)[0].lower()


def test_lock_covers_both_platforms(lock):
    assert set(lock["metadata"]["platforms"]) >= {"linux-64", "osx-arm64"}


def test_every_source_spec_is_locked_on_every_platform(lock):
    src = yaml.safe_load((ENV / "environment.yml").read_text(encoding="utf-8"))
    conda_specs, pip_specs = [], []
    for dep in src["dependencies"]:
        if isinstance(dep, dict):
            pip_specs += [_spec_name(s) for s in dep.get("pip", [])]
        else:
            conda_specs.append(_spec_name(dep))
    for platform in ("linux-64", "osx-arm64"):
        names = {p["name"].lower() for p in lock["package"] if p["platform"] == platform}
        missing = [n for n in conda_specs + pip_specs if n not in names]
        assert not missing, f"{platform}: not in the lock: {missing}"
        python = [p for p in lock["package"] if p["platform"] == platform and p["name"] == "python"]
        assert python and python[0]["version"].startswith("3.12."), platform


def test_lock_is_text_and_fits_the_pack_cap():
    text = (ENV / "conda-lock.yml").read_text(encoding="utf-8")
    assert len(text.encode()) < 8 * 1024 * 1024
    assert "conda-forge" in text


def test_engine_detects_the_lock():
    hg = pytest.importorskip("hep_genesis.iri.dispatch")
    from hep_genesis.iri.dispatch.envsetup import ENV_LOCK_RELPATH
    from hep_genesis.iri.dispatch.runner import pack_env_lock, pack_payload

    pack, files = pack_payload(str(ENV.parent))
    assert pack_env_lock(pack, files) == f"{pack.remote_name}/{ENV_LOCK_RELPATH}"
    assert hg is not None
