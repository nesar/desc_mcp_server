import functools
import importlib
import inspect
import os
from pathlib import Path
import tomllib
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

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
        result = func(*bound.args, **bound.kwargs)
        if ARTIFACT_URL and getattr(result, "files", None):
            urls = [f.replace(OUTPUT_ROOT, ARTIFACT_URL, 1)
                    for f in result.files if f.startswith(OUTPUT_ROOT)]
            if urls:
                result.message += " View: " + "  ".join(urls)
        return result

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
