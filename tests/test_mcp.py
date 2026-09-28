"""The MCP server, driven by a real MCP client over stdio."""
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .test_example import load_example

ROOT = Path(__file__).resolve().parents[1]
SIM = str(ROOT / "examples" / "refund_bot.py") + ":simulated_model"


async def test_mcp_tools_end_to_end(tmp_path):
    load_example().main(tmp_path / "traces" / "run.jsonl")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "runtape", "mcp", "--model-fn", SIM],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            assert {"list_traces", "summary", "timeline", "show_event", "context", "grep", "why_dry", "why", "rerun"} <= names

            async def call(name, **args):
                res = await s.call_tool(name, args)
                return "".join(getattr(c, "text", "") for c in res.content), res.isError

            out, _ = await call("list_traces")
            assert "run.jsonl" in out
            out, _ = await call("grep", term="any amount")
            assert "entered here" in out
            out, _ = await call("why_dry", event="tool:issue_refund")
            assert "#9 search_kb result[1].text sentence 2" in out
            out, _ = await call("why", event="tool:issue_refund")
            assert "CAUSE  #9 search_kb result[1].text sentence 2" in out
            out, _ = await call("rerun", event="30", drop=["9[1]"])
            assert "5/5  calls escalate_to_manager" in out
            out, err = await call("show_event", event="999")
            assert err and "No event #999" in out
