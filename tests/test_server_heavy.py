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


def test_create_server_runs_every_sync_tool_off_the_loop():
    pytest.importorskip("pyccl")
    from mcp_server.server import create_server

    mcp = create_server()
    heavy = mcp._tool_manager.get_tool("firecrown_run_chain")
    light = mcp._tool_manager.get_tool("firecrown_chain_status")
    assert heavy.is_async and heavy.context_kwarg == "ctx" and "ctx" not in heavy.parameters["properties"]
    # light/dispatchable tools too: one ran ON the loop and froze the server
    # (and every running chain's heartbeat) while its facility job ran
    assert light.is_async and light.context_kwarg == "ctx" and "ctx" not in light.parameters["properties"]
    disp = mcp._tool_manager.get_tool("ccl_matter_pk")
    assert disp.is_async and disp.fn.weight == "dispatchable"


def test_queued_heavy_call_heartbeats_while_waiting_for_the_lock():
    """A heavy call queued behind another must keep the client's idle timer
    alive: a silent queued call made the desktop recycle the session, which
    killed the running call too (2026-10-14)."""
    import mcp_server.server as srv

    async def scenario():
        srv._LOCKS.clear()  # fresh locks on this event loop
        first, second = _Ctx(), _Ctx()

        def slow():
            time.sleep(0.8)
            return "first"

        def quick():
            return "second"

        slow.__name__, quick.__name__ = "slow_chain", "quick_loglike"
        a = asyncio.ensure_future(srv.acquire_heavy(first, "slow_chain", heartbeat_s=0.1))
        await a
        holder = srv._HOLDERS["heavy"]
        b = asyncio.ensure_future(srv.acquire_heavy(second, "quick_loglike", heartbeat_s=0.1))
        await asyncio.sleep(0.45)
        assert not b.done()  # still queued
        srv.release_heavy()
        await b
        srv.release_heavy()
        return holder, second.calls

    holder, calls = asyncio.run(scenario())
    assert holder == "slow_chain"
    assert len(calls) >= 2 and all("queued" in m and "slow_chain" in m for _, m in calls)
    assert not srv._heavy_lock().locked() and srv._HOLDERS["heavy"] == ""


def test_weight_classes_do_not_block_each_other():
    """A long heavy call must not hold up a dispatchable or light one: each
    class has its own lock (2026-10-14: a frozen loop lost a chain's handle)."""
    import mcp_server.server as srv

    async def scenario():
        srv._LOCKS.clear()
        await srv.acquire_heavy(None, "chain", kind="heavy")
        t0 = time.monotonic()
        await asyncio.wait_for(srv.acquire_heavy(None, "pk", kind="dispatchable"), timeout=1)
        await asyncio.wait_for(srv.acquire_heavy(None, "inspect", kind="light"), timeout=1)
        quick = time.monotonic() - t0
        for kind in ("light", "dispatchable", "heavy"):
            srv.release_heavy(kind)
        return quick

    assert asyncio.run(scenario()) < 0.5


def test_heartbeat_reports_this_calls_job_not_another_calls():
    """Two dispatches in one process: each heartbeat shows its own job."""
    pytest.importorskip("hep_genesis")
    from hep_genesis.iri.dispatch import progress

    def dispatch(job_id, hold):
        progress.report(site="perlmutter", job_id=job_id, job_name=job_id, state="running")
        time.sleep(hold)
        return job_id

    async def scenario():
        a_ctx, b_ctx = _Ctx(), _Ctx()
        a = asyncio.ensure_future(run_with_heartbeat(a_ctx, dispatch, "111", 0.7, label="chain", heartbeat_s=0.2))
        await asyncio.sleep(0.05)
        b = asyncio.ensure_future(run_with_heartbeat(b_ctx, dispatch, "222", 0.7, label="pk", heartbeat_s=0.2))
        await asyncio.gather(a, b)
        return a_ctx.calls, b_ctx.calls

    a_calls, b_calls = asyncio.run(scenario())
    assert a_calls and all("job 111" in m for _, m in a_calls)
    assert b_calls and all("job 222" in m for _, m in b_calls)
