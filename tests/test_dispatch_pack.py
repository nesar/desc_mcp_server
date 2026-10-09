"""export_dispatch_pack: the client-side HPC handoff contract.

The pack must be stageable by any hep-genesis client that receives it:
sanitized relative paths, engine size caps, a manifest naming kernel entry
points, node-side pip requirements (pip-kernels) or the env_setup requirement
with public candidates (env-kernels) - and nothing identity-shaped.
"""

import json

from mcp_server.dispatch import _PACK_MAX_BYTES, _PACK_MAX_FILES, export_dispatch_pack


def test_pack_shape_and_caps():
    pack = export_dispatch_pack()["dispatch_pack"]
    assert pack["name"] == "tools"
    assert pack["server"] == "desc-mcp-server"
    assert 0 < pack["file_count"] <= _PACK_MAX_FILES
    assert 0 < pack["bytes"] <= _PACK_MAX_BYTES
    assert pack["file_count"] == len(pack["files"])
    json.dumps(pack)  # one JSON tool result


def test_paths_are_safe_and_rooted():
    files = export_dispatch_pack()["dispatch_pack"]["files"]
    for rel in files:
        assert rel.startswith("tools/"), rel
        parts = rel.split("/")
        assert all(p and p != ".." and not p.startswith(".") for p in parts), rel
        assert "__pycache__" not in rel, rel
    assert "tools/common.py" in files
    assert "tools/envkernel.py" in files
    assert "tools/inner/__init__.py" in files
    assert "tools/inner/noop.py" in files


def test_inner_scripts_are_facility_safe():
    """Inner scripts run under a facility env that only guarantees the DESC
    stack: no server-only imports allowed there."""
    import ast

    banned_roots = {"pydantic", "matplotlib", "mcp", "tools"}
    files = export_dispatch_pack()["dispatch_pack"]["files"]
    for rel, text in files.items():
        if not rel.startswith("tools/inner/"):
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level >= 2:  # reaches back into the server package
                    raise AssertionError(f"{rel}: relative import '{'.' * node.level}{node.module}'")
                if node.level == 1:  # a sibling inner script: fine
                    continue
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                assert root not in banned_roots, f"{rel} imports '{name}'"


def test_manifest_has_no_identity_and_env_candidates():
    pack = export_dispatch_pack()["dispatch_pack"]
    blob = json.dumps({k: v for k, v in pack.items() if k != "files"}).lower()
    for forbidden in ("nesar", "m1727", "nersc_project", "pscratch/sd"):
        assert forbidden not in blob
    env = pack["env_kernel"]
    assert env["function"] == "envkernel.run_in_env"
    assert "perlmutter" in env["env_setup_candidates"]
    for name, k in pack["kernels"].items():
        assert "function" in k, name
        if k["function"] == "envkernel.run_in_env":
            assert k.get("env_setup_required") is True, name
            assert "inner" in k, name
            assert f"tools/inner/{k['inner']}.py" in pack["files"], name
        else:
            assert "pip_deps" in k, name
