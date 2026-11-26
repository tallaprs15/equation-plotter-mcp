"""MCP end to end: launch `equation-plotter-mcp` over stdio like a real client and call the tool."""

import base64
import json
import os
import sys

import anyio
import pytest

pytest.importorskip("mcp")
from mcp.client import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "equation_plotter.mcp_server"],
                               env={"PYTHONPATH": os.pathsep.join(sys.path)})


async def _session(fn):
    async with Client(SERVER) as client:
        return await fn(client)


def test_list_and_call():
    async def go(client):
        tools = (await client.list_tools()).tools
        assert [t.name for t in tools] == ["plot_equations"]
        assert tools[0].annotations.read_only_hint is True
        res = await client.call_tool("plot_equations",
                                     {"equations": ["y = x^2 - 3x + 2"], "x_range": [-2, "4"]})
        assert not res.is_error
        text = next(c for c in res.content if c.type == "text")
        img = next(c for c in res.content if c.type == "image")
        data = json.loads(text.text)
        assert data["equations"][0]["classification"] == "quadratic"
        assert data["viewport"]["x"] == [-2.0, 4.0]
        assert data["image"]["data"] is None
        assert img.mime_type == "image/png"
        assert base64.b64decode(img.data)[:4] == b"\x89PNG"

        bad = await client.call_tool("plot_equations", {"equations": ["__import__('os')"]})
        assert bad.is_error
        assert "invalid_character" in bad.content[0].text

    anyio.run(_session, go)
