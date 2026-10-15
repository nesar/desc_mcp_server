import asyncio
import functools
import importlib
import inspect
import os
import time
from pathlib import Path
import tomllib
from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from mcp_server.dispatch import CALL_OUTPUT_DIR

Transport = Literal["stdio", "streamable-http"]

# Hosted deployments set these (same convention as the cosmic/spectra/gaia servers):
#   MCP_OUTPUT_ROOT   e.g. /srv/artifacts — every output_dir an agent passes
#                     is remapped under this directory
#   MCP_ARTIFACT_URL  e.g. https://files.example.org — returned messages then
#                     include a browsable URL for each file written there
OUTPUT_ROOT = os.environ.get("MCP_OUTPUT_ROOT")
ARTIFACT_URL = (os.environ.get("MCP_ARTIFACT_URL") or "").rstrip("/")


def publish_outputs(func):
    """Confine a tool's output_dir to OUTPUT_ROOT and add URLs to its result."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        requested = bound.arguments.get("output_dir")
        if requested is not None:
            root = Path(OUTPUT_ROOT)
            path = Path(str(requested))
            if not path.is_relative_to(root):
                safe = "-".join(part for part in path.parts if part != "/")
                bound.arguments["output_dir"] = str(root / (safe or "output"))
        # dispatched jobs fetch into the confined dir, not the client's path
        out_key = CALL_OUTPUT_DIR.set(bound.arguments.get("output_dir") or None)
        try:
            result = func(*bound.args, **bound.kwargs)
        finally:
            CALL_OUTPUT_DIR.reset(out_key)
        if ARTIFACT_URL and getattr(result, "files", None):
            urls = [f.replace(OUTPUT_ROOT, ARTIFACT_URL, 1)
                    for f in result.files if f.startswith(OUTPUT_ROOT)]
            if urls:
                result.message += " View: " + "  ".join(urls)
        return result

    return wrapper


# Progress heartbeat period for heavy tools. The hep-genesis desktop attaches
# a progress token to every call and resets its 45-min per-request timeout on
# each notification (24 h cap); a silent call is given up after 45 min and the
# result of a job the user paid for is lost (observed with a 6-hour local
# chain and an hour-long Perlmutter queue wait, 2026-10-08).
HEARTBEAT_S = float(os.environ.get("MCP_HEARTBEAT_S", "30"))
# Heavy tools share process state (experiment caches, dispatch env vars) and
# were serialized implicitly by blocking the event loop; keep them serialized.
# Every other tool runs off the loop too, serialized per weight class: a
# "dispatchable" tool used to run ON the event loop, and when it dispatched
# (a CCL P(k) job behind a 1.5-h env build, 2026-10-14) the whole server froze
# for hours - the running chain's heartbeats stopped, the client gave up after
# 45 min and lost the job's handle.
_LOCKS: dict[str, asyncio.Lock] = {}
_HOLDERS: dict[str, str] = {}  # weight class -> name of the tool holding its lock


def _lock(kind: str = "heavy") -> asyncio.Lock:
    if kind not in _LOCKS:
        _LOCKS[kind] = asyncio.Lock()
    return _LOCKS[kind]


def _heavy_lock() -> asyncio.Lock:
    return _lock("heavy")


async def acquire_heavy(ctx, label: str, heartbeat_s: float | None = None, kind: str = "heavy") -> None:
    """Take the ``kind`` lock (default: the heavy-tool lock), heartbeating
    while queued behind another call of that class. A queued call used to stay
    silent until the lock freed; the client then gave up after its 45-min idle
    timeout and recycled the whole session, which also killed the call that
    held the lock (a running Perlmutter chain, 2026-10-14)."""
    period = HEARTBEAT_S if heartbeat_s is None else heartbeat_s
    lock = _lock(kind)
    started = time.monotonic()
    while True:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=period)
            break
        except asyncio.TimeoutError:
            if ctx is None:
                continue
            waited = int(time.monotonic() - started)
            text = (f"{label}: queued for {waited // 60} min behind {_HOLDERS.get(kind) or 'another tool'} "
                    f"(this server runs one {kind} tool at a time)")
            try:
                await ctx.report_progress(waited, None, text)
            except Exception:  # noqa: BLE001 - progress is advisory
                pass
    _HOLDERS[kind] = label


def release_heavy(kind: str = "heavy") -> None:
    _HOLDERS[kind] = ""
    _lock(kind).release()


async def run_with_heartbeat(ctx, func, *args, label: str = "", heartbeat_s: float | None = None, **kwargs):
    """Run blocking ``func`` in a worker thread, sending an MCP progress
    notification every ``heartbeat_s`` seconds while it runs (no-op without
    a request context / progress token)."""
    period = HEARTBEAT_S if heartbeat_s is None else heartbeat_s
    # This call's own engine progress slot (to_thread copies the context, so
    # the dispatch running in the worker reports into it). The process-wide
    # slot showed another call's job - or nothing, once that call's dispatch
    # cleared it: "86 min elapsed (still running)" for a Perlmutter chain,
    # 2026-10-14.
    try:
        from hep_genesis.iri.dispatch.progress import open_slot as _open_slot

        slot = _open_slot()
    except Exception:  # noqa: BLE001 - engine absent or old
        slot = None
    task = asyncio.ensure_future(asyncio.to_thread(func, *args, **kwargs))
    started = time.monotonic()
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=period)
            if done:
                return task.result()
            if ctx is not None:
                elapsed = int(time.monotonic() - started)
                text = f"{label}: {elapsed // 60} min elapsed (still running)"
                try:
                    # server-side dispatch: the engine knows the facility job's
                    # state, elapsed time and stdout digest - say that instead
                    from hep_genesis.iri.dispatch.progress import message as _job_message

                    live = _job_message(slot) if slot is not None else _job_message()
                    if live:
                        text = f"{label}: {live}"
                except Exception:  # noqa: BLE001 - engine absent or old
                    pass
                try:
                    await ctx.report_progress(elapsed, None, text)
                except Exception:  # noqa: BLE001 - progress is advisory
                    pass
    finally:
        if not task.done():
            task.cancel()


def offload_heavy(func, kind: str | None = None):
    """Wrap a tool so it runs off the event loop under its weight class's
    lock (``kind``: default the tool's ``weight``, else "heavy") and
    heartbeats progress while it runs. The MCP schema is unchanged - the
    extra ``ctx`` parameter is the FastMCP Context, injected by the server
    and hidden from clients."""
    kind = kind or getattr(func, "weight", "heavy")
    # eval_str: tool modules using `from __future__ import annotations` carry
    # string annotations that only resolve in THEIR globals - the wrapper
    # lives here, so hand FastMCP/pydantic the resolved objects.
    sig = inspect.signature(func, eval_str=True)
    params = [p for p in sig.parameters.values() if p.kind is not inspect.Parameter.VAR_KEYWORD]
    params.append(inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Context))

    @functools.wraps(func)
    async def wrapper(*args, ctx: Context | None = None, **kwargs):
        await acquire_heavy(ctx, func.__name__, kind=kind)
        out_key = CALL_OUTPUT_DIR.set(kwargs.get("output_dir") or None)
        try:
            # partial: a tool parameter named like ours (augur's `label`) must
            # reach the tool, not collide with run_with_heartbeat's keywords
            return await run_with_heartbeat(ctx, functools.partial(func, *args, **kwargs), label=func.__name__)
        finally:
            CALL_OUTPUT_DIR.reset(out_key)
            release_heavy(kind)

    wrapper.__signature__ = sig.replace(parameters=params)
    annotations = {p.name: p.annotation for p in params if p.annotation is not inspect.Parameter.empty}
    if sig.return_annotation is not inspect.Signature.empty:
        annotations["return"] = sig.return_annotation
    wrapper.__annotations__ = annotations
    wrapper.weight = kind
    return wrapper


def pyproject_toml() -> Path:
    for directory in Path(__file__).resolve().parents:
        path = directory / "pyproject.toml"
        if path.exists():
            return path
    raise FileNotFoundError("Could not find pyproject.toml")


def configured_tool_module_names() -> list[str]:
    path = pyproject_toml()
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        return list(config["tool"]["mcp-server"]["tool_modules"])
    except KeyError as exc:
        raise RuntimeError(
            f"{path} must contain a [tool.mcp-server] section with a "
            "tool_modules list."
        ) from exc


def load_tool_modules():
    return [importlib.import_module(name)
            for name in configured_tool_module_names()]


INSTRUCTIONS = (
    "LSST DESC cosmology server: the public DESC analysis stack as tools. "
    "Families: ccl_* (pyccl theory: distances, growth, P(k), angular C_ell, "
    "correlation functions, halo model, LSST SRD n(z)); sacc_* (the data-vector "
    "file format shared by every DESC code: inspect, convert, prepare for "
    "likelihoods, attach covariances); firecrown_* (likelihoods: build an "
    "experiment from a sacc file + systematics, log-likelihood, theory data "
    "vectors, parameter scans, chains); augur_* (Fisher forecasts for LSST "
    "Y1/Y10 3x2pt); txpipe_* (the DESC 3x2pt measurement pipeline: stage "
    "catalog, compose and validate pipelines, run them locally or on a "
    "facility); tjpcov_* (analysis-grade covariances for a sacc file: "
    "Gaussian f_sky or NaMaster mode-coupled, super-sample and connected "
    "non-Gaussian terms, real space); namaster_* (pseudo-C_ell bandpowers "
    "from masked HEALPix maps -> sacc with windows, mask properties, "
    "simulated test maps); smokescreen_* (data-vector concealment/blinding "
    "through a firecrown likelihood, file encryption - never reveals the "
    "hidden shift). Start with list_desc_packages, then describe_desc_tool_family "
    "for the family you need. The server carries skills (named multi-tool "
    "recipes): call list_desc_skills, and when a task matches one, "
    "load_desc_skill and follow it. Conventions: redshift z (never scale "
    "factor) at the tool boundary; k in h/Mpc and P(k) in (Mpc/h)^3 unless a "
    "tool's k_units says otherwise; theta in arcmin; parameter names are "
    "firecrown's (Omega_c, Omega_b, h, n_s, sigma8 or A_s, w0, wa, m_nu; "
    "nuisance src0_delta_z, lens0_bias, ia_bias ...) - use "
    "convert_cosmology_names to translate. Tools that write files take an "
    "output_dir and return structured artifact metadata; pass file paths "
    "between tools, never raw arrays. Heavy tools (Fisher, chains, TXPipe "
    "pipelines) can run on ALCF/NERSC through the hep-genesis dispatch engine; "
    "this server holds no facility credentials or project settings - those "
    "belong to the client."
)


def create_server(*, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    transport_security = None
    if os.environ.get("MCP_PUBLIC"):
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False)

    instructions = INSTRUCTIONS
    if OUTPUT_ROOT:
        instructions += (
            f" This is a hosted server: all output files are stored under "
            f"{OUTPUT_ROOT} on the server (any other output_dir is remapped "
            "there), and results include browsable URLs - share those URLs "
            "with the user instead of trying to read or recreate the files."
        )

    mcp = FastMCP(
        "DESC Cosmology MCP Server",
        instructions=instructions,
        host=host,
        port=port,
        transport_security=transport_security,
    )

    for tool_module in load_tool_modules():
        if not hasattr(tool_module, "__all__"):
            raise RuntimeError(
                f"Tool module '{tool_module.__name__}' must define __all__.")
        for name in tool_module.__all__:
            tool_function = getattr(tool_module, name)
            if OUTPUT_ROOT:
                tool_function = publish_outputs(tool_function)
            if not inspect.iscoroutinefunction(tool_function):
                # heavy / dispatchable / light: each class serialized, all off the loop
                tool_function = offload_heavy(tool_function, getattr(tool_function, "weight", "light"))
            mcp.tool()(tool_function)

    register_skill_prompts(mcp)
    return mcp


def register_skill_prompts(mcp: FastMCP) -> None:
    """Expose every skills/*.md file as a native MCP prompt.

    The skills are primarily served through the list_desc_skills /
    load_desc_skill tools (which every MCP client supports); registering
    them as prompts as well lets prompt-capable clients surface the same
    recipes in their UI with zero tool calls.
    """
    from tools.meta.skills import skill_index, skill_text

    for name, description in skill_index().items():
        def make_prompt(skill_name: str):
            def prompt() -> str:
                return skill_text(skill_name)
            return prompt

        mcp.prompt(name=name, description=description)(make_prompt(name))


def run_server(*, transport: Transport = "stdio",
               host: str = "127.0.0.1", port: int = 8000) -> None:
    mcp = create_server(host=host, port=port)
    mcp.run(transport=transport)
