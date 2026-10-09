"""Live-server smoke test: call tools from every family through a real MCP session.

Unlike smoke_env.py (environment) and tests/test_*.py (in-process), this
exercises the SERVED path - transport, schema, sandboxing, filesystem
permissions. Run it after every deployment; a tool that works in-process
can still fail under a hosting sandbox.

Usage:
    python tests/smoke_server.py                          # spawns stdio server
    python tests/smoke_server.py http://127.0.0.1:8000/mcp
    python tests/smoke_server.py https://desc.example.org/mcp

Exit code 0 only if every call (including the chained ones) succeeds.
"""

import json
import os
import sys
import tempfile

import anyio
from mcp.client.session import ClientSession

OUT = (tempfile.mkdtemp(prefix="desc_smoke_") if len(sys.argv) < 2
       else "/srv/artifacts/smoke-server")

RESULTS = []


async def call(s, name, args, expect_error=False):
    try:
        res = await s.call_tool(name, args)
        if res.isError:
            text = res.content[0].text if res.content else ""
            if expect_error:
                RESULTS.append((name, "OK", "expected error raised"))
                return None
            RESULTS.append((name, "FAIL", text[:160]))
            return None
        text = res.content[0].text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"message": text}
        if expect_error:
            RESULTS.append((name, "FAIL", "expected an error, got success"))
            return payload
        RESULTS.append((name, "OK", str(payload.get("message", payload))[:110]))
        return payload
    except Exception as e:  # transport-level failure
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:120]}"))
        return None


def first_file(payload, suffix):
    for f in (payload or {}).get("files", []):
        if f.endswith(suffix):
            return f
    return None


async def run(session: ClientSession):
    await session.initialize()
    tools = await session.list_tools()
    prompts = await session.list_prompts()
    print(f"server: {len(tools.tools)} tools, {len(prompts.prompts)} prompts; outputs in {OUT}")
    s = session

    # --- discovery
    await call(s, "list_desc_packages", {})
    await call(s, "describe_desc_tool_family", {"family": "ccl"})
    await call(s, "list_desc_skills", {})
    await call(s, "load_desc_skill", {"name": "desc-tour"})
    await call(s, "convert_cosmology_names",
               {"params": {"Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67, "n_s": 0.96, "sigma8": 0.81},
                "to": "emulator"})
    await call(s, "get_dispatch", {})
    await call(s, "auth_status", {})

    # --- ccl
    await call(s, "ccl_background", {"output_dir": OUT, "z_max": 2.0, "n_z": 20})
    await call(s, "ccl_matter_pk", {"output_dir": OUT, "z": 0.5, "n_k": 40,
                                    "cosmology": {"transfer_function": "eisenstein_hu"}})
    nz = await call(s, "ccl_lsst_srd_nz", {"output_dir": OUT, "forecast_year": 1,
                                           "sample": "source", "write_sacc": False})
    nz_csv = first_file(nz, ".csv")
    cls = None
    if nz_csv:
        cls = await call(s, "ccl_angular_cls",
                         {"output_dir": OUT, "n_ell": 12, "ell_max": 2000, "pairs": "auto",
                          "write_sacc": True,
                          "cosmology": {"transfer_function": "eisenstein_hu"},
                          "tracers": [{"type": "wl", "name": "src0", "nz_file": nz_csv, "nz_column": "bin_0"},
                                      {"type": "wl", "name": "src1", "nz_file": nz_csv, "nz_column": "bin_1"}]})
    await call(s, "ccl_halo_mass_function", {"output_dir": OUT, "n_M": 10,
                                             "cosmology": {"transfer_function": "eisenstein_hu"}})

    # --- sacc + firecrown chain on the CCL data vector
    sacc_file = first_file(cls, ".sacc") or first_file(cls, ".hdf5") or first_file(cls, ".fits")
    if sacc_file:
        await call(s, "sacc_inspect", {"output_dir": OUT, "sacc_path": sacc_file})
        cov = await call(s, "sacc_attach_gaussian_covariance",
                         {"output_dir": OUT, "sacc_path": sacc_file, "f_sky": 0.4,
                          "sigma_e": 0.26, "n_gal": 2.0,
                          "cosmology": {"transfer_function": "eisenstein_hu"}})
        cov_file = first_file(cov, ".sacc") or first_file(cov, ".hdf5") or first_file(cov, ".fits")
        if cov_file:
            # same transfer function as the data vector: the closure must be exact
            exp = await call(s, "firecrown_build_likelihood",
                             {"output_dir": OUT, "sacc_path": cov_file,
                              "correlation_space": "harmonic",
                              "transfer_function": "eisenstein_hu"})
            yaml_file = first_file(exp, ".yaml") or first_file(exp, ".yml")
            if yaml_file:
                ll = await call(s, "firecrown_compute_loglike",
                                {"output_dir": OUT, "experiment_yaml": yaml_file,
                                 "cosmology": {"transfer_function": "eisenstein_hu"}})
                chi2 = (ll or {}).get("metadata", {}).get("chi2")
                if chi2 is not None:
                    ok = chi2 < 1e-3
                    RESULTS.append(("closure chi2 (CCL data == firecrown theory)",
                                    "OK" if ok else "FAIL", f"chi2 = {chi2:.3g} (expect ~0)"))

    # --- txpipe (compose + validate only; never runs TXPipe)
    await call(s, "txpipe_list_stages", {"group": "two-point"})
    await call(s, "txpipe_describe_stage", {"name": "TXTwoPoint"})
    pipe = await call(s, "txpipe_generate_pipeline",
                      {"output_dir": OUT, "preset": "mock_shear"})
    pyml = first_file(pipe, "pipeline.yml")
    if pyml:
        await call(s, "txpipe_validate_pipeline", {"pipeline_yml": pyml})

    # --- augur (config + validation only here; Fisher is covered by tests)
    cfg = await call(s, "augur_generate_forecast_config",
                     {"output_dir": OUT, "survey_year": "Y1", "probes": "shear",
                      "var_pars": ["Omega_c", "sigma8"]})
    cfg_yml = first_file(cfg, ".yml") or first_file(cfg, ".yaml")
    if cfg_yml:
        await call(s, "augur_validate_forecast_config", {"config_path": cfg_yml})

    # --- dispatch handoff
    pack = await call(s, "export_dispatch_pack", {})
    if pack and "dispatch_pack" in pack:
        RESULTS.append(("export_dispatch_pack(files)", "OK",
                        f"{pack['dispatch_pack']['file_count']} files, "
                        f"{len(pack['dispatch_pack']['kernels'])} kernels"))

    # --- guardrails must guard
    await call(s, "load_desc_skill", {"name": "no-such-skill"}, expect_error=True)
    await call(s, "ccl_matter_pk", {"output_dir": OUT, "cosmology": {"A_s": 2e-9}},
               expect_error=True)  # both amplitudes


async def main():
    if len(sys.argv) > 1:
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(sys.argv[1]) as (r, w, _):
            async with ClientSession(r, w) as session:
                await run(session)
    else:
        from mcp.client.stdio import StdioServerParameters, stdio_client
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"],
                                       env=dict(os.environ))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await run(session)


if __name__ == "__main__":
    anyio.run(main)
    width = max(len(n) for n, *_ in RESULTS) + 2
    fails = 0
    for name, status, detail in RESULTS:
        print(f"{status:6s}{name:{width}s}{detail}")
        fails += status == "FAIL"
    print(f"\n{len(RESULTS)} calls: {len(RESULTS) - fails} OK, {fails} FAIL")
    sys.exit(1 if fails else 0)
