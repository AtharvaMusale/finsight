"""Run the FinSight MCP server.

Run from the finsight/ folder:
  uv run python -m finsight.mcp_server                                # stdio (Claude Desktop/Code)
  uv run python -m finsight.mcp_server --transport streamable-http    # http://127.0.0.1:8001/mcp

Logging goes to stderr: under stdio, stdout is the protocol channel and must stay clean.
"""

import argparse
import logging
import sys

from finsight.config import get_settings
from finsight.mcp_server.server import build_server
from finsight.observability import configure_langsmith
from finsight.tracing import configure as configure_tracing


def main() -> None:
    ap = argparse.ArgumentParser(prog="finsight-mcp")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1", help="streamable-http only; local by default")
    ap.add_argument("--port", type=int, default=8001, help="streamable-http only")
    args = ap.parse_args()

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s %(message)s"
    )
    settings = get_settings()
    configure_tracing("mcp", settings.trace_dir, settings.tracing)
    configure_langsmith(settings)
    server = build_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
