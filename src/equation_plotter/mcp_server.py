"""MCP server exposing ``analyse_and_plot`` as the ``plot_equations`` tool.

Run locally over stdio (what Claude Desktop / Claude Code / most agent frameworks launch):

    equation-plotter-mcp

or over Streamable HTTP for remote clients (binds to localhost unless told otherwise):

    equation-plotter-mcp --transport streamable-http --host 127.0.0.1 --port 8000   # -> /mcp
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
from typing import Annotated, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__
from .core import analyse_and_plot
from .ratelimit import Access

# Server-side caps: callers can ask for less, never more.
MAX_CONCURRENT = int(os.environ.get("PLOTTER_MCP_MAX_CONCURRENT", "4"))
MAX_TIMEOUT = float(os.environ.get("PLOTTER_MCP_MAX_TIMEOUT", "15"))
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)

INSTRUCTIONS = """\
Use plot_equations to graph and analyse mathematical equations. It returns a JSON analysis
(classification, domain/range, intercepts, turning points, asymptotes, symmetry, ...) and a
PNG of the graph. Every number has an exact form (e.g. "sqrt(2)/2") and a float; prefer the
exact form when explaining results. Check `warnings` and each equation's `methods` map: values
marked numeric_estimate are approximations. Several equations in one call share one set of axes
and their intersections are reported."""

DESCRIPTION = """\
Analyse and plot one or more equations on a single set of axes.

Accepted inputs (one string per equation):
- explicit: "y = x^2 - 3x + 2", "f(x) = 1/(x-2)", bare "x^3 - x"
- implicit: "x^2 + y^2 = 25", "x^2/9 - y^2/4 = 1", "xy = 1"
- parametric: "x = cos(t), y = sin(t)"   - polar: "r = 1 + cos(theta)", "r^2 = 4cos(2theta)"
- inequalities (shaded): "y > x^2", "x^2 + y^2 <= 9"
- LaTeX subset: "y = \\frac{1}{x-2}"
Syntax: ^ for powers, implicit multiplication (3x, 2sin x), constants pi and e, functions sin cos
tan sec csc cot asin acos atan sinh cosh tanh exp ln log (base 10) sqrt cbrt abs floor ceil.

Returns JSON text (schema version "1.0"; errors come back as {"error": {code, message}}) plus
the graph as an image. Ranges are [min, max]; values may be numbers or strings like "2pi"."""

Range = Annotated[list[float | str] | None,
                  Field(description="[min, max]; numbers or expressions like '-2pi'",
                        min_length=2, max_length=2)]

# subscriptions=False: the tool list never changes, so there is nothing to push. Without this,
# 2026-07-28 clients hold a `subscriptions/listen` stream open on every connection, which ties
# up a request slot and keeps a Cloud Run instance billed while idle.
mcp = MCPServer(name="equation-plotter", title="Equation Plotter", version=__version__,
                instructions=INSTRUCTIONS, subscriptions=False)


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


@mcp.tool(
    name="plot_equations",
    title="Plot and analyse equations",
    description=DESCRIPTION,
    annotations=ToolAnnotations(title="Plot and analyse equations", readOnlyHint=True,
                                idempotentHint=True, openWorldHint=False),
    structured_output=False,
)
def plot_equations(
    equations: Annotated[list[str], Field(description="Equations to plot together (1-8)",
                                          min_length=1, max_length=8)],
    x_range: Range = None,
    y_range: Range = None,
    t_range: Annotated[list[float | str] | None,
                       Field(description="Parameter range for parametric curves",
                             min_length=2, max_length=2)] = None,
    theta_range: Annotated[list[float | str] | None,
                           Field(description="Theta range for polar curves",
                                 min_length=2, max_length=2)] = None,
    image_format: Annotated[Literal["png", "svg", "none"],
                            Field(description="png (shown inline), svg (returned base64 "
                                              "inside the JSON), or none")] = "png",
    timeout: Annotated[float | None, Field(description=f"Symbolic analysis budget in seconds "
                                                       f"(max {MAX_TIMEOUT:g})", gt=0)] = None,
):
    budget = min(timeout or MAX_TIMEOUT, MAX_TIMEOUT)
    if not _slots.acquire(timeout=30):
        raise ToolError(json.dumps({"error": {"code": "busy",
                                              "message": "Server is busy; try again shortly"}}))
    try:
        result = analyse_and_plot(equations, x_range=x_range, y_range=y_range,
                                  image_format=image_format, t_range=t_range,
                                  theta_range=theta_range, timeout=budget)
    finally:
        _slots.release()

    if "error" in result:
        raise ToolError(json.dumps(result, ensure_ascii=False))

    image = result.get("image")
    blocks = []
    if image and image["format"] == "png":
        # Send the PNG once, as a real image block; keep the JSON small for the model.
        png = base64.b64decode(image["data"])
        image["data"] = None
        image["note"] = "PNG returned as a separate image content block"
        blocks.append(json.dumps(result, ensure_ascii=False))
        blocks.append(Image(data=png, format="png"))
    else:
        blocks.append(json.dumps(result, ensure_ascii=False))
    return blocks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="equation-plotter-mcp", description="Equation Plotter MCP server")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1",
                    help="HTTP bind address (default localhost; put auth in front before exposing)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")),
                    help="HTTP port (default $PORT, else 8000)")
    args = ap.parse_args(argv)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        import uvicorn

        # proxy_headers off: the client address is resolved in ratelimit.Access, which only
        # trusts the hops the operator declares (X-Forwarded-For is otherwise forgeable).
        uvicorn.run(http_app(args.host), host=args.host, port=args.port,
                    log_level="info", proxy_headers=False)


def http_app(host: str = "127.0.0.1"):
    """Stateless Streamable HTTP app. GET /mcp (a server-push stream) gets 405 straight away:
    the server is stateless and has nothing to push, and an open stream would only hang
    clients and connection checkers."""
    app = mcp.streamable_http_app(stateless_http=True, json_response=True, host=host)
    access = Access.from_env()

    async def wrapped(scope, receive, send):
        if (scope["type"] == "http" and scope["method"] == "GET"
                and scope["path"].rstrip("/") == "/mcp"):
            body = json.dumps({"jsonrpc": "2.0", "id": None, "error": {
                "code": -32000, "message": "Method Not Allowed: use POST"}}).encode()
            await send({"type": "http.response.start", "status": 405,
                        "headers": [(b"allow", b"POST"), (b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        if scope["type"] != "http" or scope["method"] != "POST":
            await app(scope, receive, send)
            return
        await _logged(app, scope, receive, send, access)

    return wrapped


def _rpc_summary(raw: bytes) -> list[str]:
    """Method names only (plus the tool name for tools/call); never arguments."""
    try:
        msgs = json.loads(raw)
    except ValueError:
        return ["<unparseable>"]
    out = []
    for m in msgs if isinstance(msgs, list) else [msgs]:
        if not isinstance(m, dict):
            out.append("<non-object>")
        elif "method" in m:
            name = str(m["method"])[:64]
            if name == "tools/call":
                name += ":" + str((m.get("params") or {}).get("name"))[:64]
            out.append(name)
        else:
            out.append("<response>" if "result" in m or "error" in m else "<unknown>")
    return out


async def _send_json(send, status, payload, extra_headers=()):
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()), *extra_headers]})
    await send({"type": "http.response.body", "body": body})


def _request_id(raw: bytes):
    try:
        msg = json.loads(raw)
        return msg.get("id") if isinstance(msg, dict) else None
    except ValueError:
        return None


async def _logged(app, scope, receive, send, access):
    """Apply access control / rate limits to tool calls, then log one structured line per POST
    (Cloud Run reads JSON lines from stdout)."""
    import sys
    import time

    chunks, more = [], True
    while more:
        msg = await receive()
        if msg["type"] != "http.request":
            break
        chunks.append(msg.get("body", b""))
        more = msg.get("more_body", False)
    raw = b"".join(chunks)
    replayed = False

    async def replay():
        nonlocal replayed
        if not replayed:
            replayed = True
            return {"type": "http.request", "body": raw, "more_body": False}
        return await receive()

    status = {"code": None}

    async def capture(msg):
        if msg["type"] == "http.response.start":
            status["code"] = msg["status"]
        await send(msg)

    headers = {k.decode().lower(): v.decode(errors="replace") for k, v in scope["headers"]}
    methods = _rpc_summary(raw)

    decision = access.check(headers, scope.get("client"),
                            costly=any(m.startswith("tools/call") for m in methods))
    if decision.status == 401:
        print(json.dumps({"severity": "WARNING", "message": "rejected: invalid API key"}),
              flush=True)
        await _send_json(send, 401, {"jsonrpc": "2.0", "id": _request_id(raw), "error": {
            "code": -32001, "message": "Invalid API key"}})
        return
    if decision.status == 429:
        print(json.dumps({"severity": "WARNING", "message": "rate limited",
                          "caller": decision.caller_tag}), flush=True)
        # A tool *result* flagged as an error, so the calling model sees why and can tell the
        # user, instead of the client reporting a broken connection.
        text = json.dumps({"error": {"code": "rate_limited", "message":
                           f"Rate limit reached for this caller; retry in about "
                           f"{decision.retry_after} s"}})
        await _send_json(send, 200, {"jsonrpc": "2.0", "id": _request_id(raw), "result": {
            "content": [{"type": "text", "text": text}], "isError": True}},
            [(b"retry-after", str(decision.retry_after).encode())])
        return
    start = time.monotonic()
    print(json.dumps({"severity": "INFO", "message": f"rpc start {','.join(methods)}",
                      "rpc": methods, "protocol": headers.get("mcp-protocol-version"),
                      "caller": decision.caller_tag,
                      "accept": headers.get("accept")}), file=sys.stdout, flush=True)
    try:
        await app(scope, replay, capture)
    finally:
        print(json.dumps({"severity": "INFO",
                          "message": f"rpc end {','.join(methods)} {status['code']} "
                                     f"{(time.monotonic() - start) * 1000:.0f}ms",
                          "rpc": methods, "status": status["code"],
                          "ms": round((time.monotonic() - start) * 1000)}),
              file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
