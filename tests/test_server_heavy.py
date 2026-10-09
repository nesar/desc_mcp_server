"""Heavy tools run off the event loop with progress heartbeats (contract R7)."""

import asyncio
import inspect
import time

import pytest

from mcp_server.server import offload_heavy, run_with_heartbeat


class _Ctx:
    def __init__(self):
        self.calls = []

    async def report_progress(self, progress, total=None, message=None):
        self.calls.append((progress, message))


def test_run_with_heartbeat_ticks_and_returns():
    ctx = _Ctx()
    out = asyncio.run(run_with_heartbeat(ctx, lambda x: (time.sleep(0.7), x * 2)[1], 21,
                                         label="t", heartbeat_s=0.2))
    assert out == 42 and len(ctx.calls) >= 2
    assert all("t:" in m for _, m in ctx.calls)


def test_run_with_heartbeat_without_context():
    assert asyncio.run(run_with_heartbeat(None, lambda: "ok", heartbeat_s=0.1)) == "ok"


def test_offload_heavy_hides_ctx_from_schema_and_injects_it():
    from mcp.server.fastmcp import Context
    from mcp.server.fastmcp.tools.base import Tool

    def heavy(output_dir: str, n: int = 3) -> dict:
        """doc"""
        return {"n": n, "out": output_dir}

    heavy.weight = "heavy"
    wrapped = offload_heavy(heavy)
    assert inspect.iscoroutinefunction(wrapped) and wrapped.weight == "heavy"
    tool = Tool.from_function(wrapped)
    assert tool.context_kwarg == "ctx" and tool.is_async
    assert set(tool.parameters["properties"]) == {"output_dir", "n"}
    assert tool.description == "doc"
    res = asyncio.run(tool.run({"output_dir": "x", "n": 5}, context=None))
    assert res == {"n": 5, "out": "x"}
    assert Context is not None


def test_create_server_wraps_only_heavy_tools():
    pytest.importorskip("pyccl")
    from mcp_server.server import create_server

    mcp = create_server()
    heavy = mcp._tool_manager.get_tool("firecrown_run_chain")
    light = mcp._tool_manager.get_tool("firecrown_chain_status")
    assert heavy.is_async and heavy.context_kwarg == "ctx" and "ctx" not in heavy.parameters["properties"]
    assert not light.is_async and light.context_kwarg is None
