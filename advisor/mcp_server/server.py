"""MCP server exposing the advisor's financial tools + RAG search.

Run:  python -m advisor.mcp_server.server              (stdio, for ADK / LangGraph / Claude Desktop)
      python -m advisor.mcp_server.server --http 8080  (streamable HTTP, e.g. on Cloud Run)
"""
from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from advisor.tools import TOOLS

mcp = FastMCP("financial-advisor-tools", log_level="WARNING")
for _fn in TOOLS.values():
    mcp.add_tool(_fn)


@mcp.resource("advisor://disclaimer")
def disclaimer() -> str:
    from advisor.graph.compliance import DISCLAIMER

    return DISCLAIMER


def main() -> None:
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8080
        mcp.settings.host, mcp.settings.port = "0.0.0.0", port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
