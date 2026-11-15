"""Tool executors used by the LangGraph agent: in-process, or over MCP (stdio)."""
from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack

from advisor.tools import TOOLS
from advisor.tools.schema import function_schema


class LocalToolExecutor:
    """Calls the Python functions directly (fastest; used in tests and by default)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def declarations(self) -> list[dict]:
        return [function_schema(f) for f in TOOLS.values()]

    async def call(self, name: str, args: dict) -> dict:
        if name not in TOOLS:
            raise ValueError(f"unknown tool {name}")
        return TOOLS[name](**args)


class MCPToolExecutor:
    """Discovers and calls tools through the MCP server over stdio."""

    def __init__(self, command: str = sys.executable, args: list[str] | None = None):
        self.command = command
        self.args = args or ["-m", "advisor.mcp_server.server"]
        self._stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self.command, args=self.args, env=dict(os.environ))
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def declarations(self) -> list[dict]:
        tools = (await self.session.list_tools()).tools
        return [{"name": t.name, "description": t.description or "", "parameters": t.inputSchema} for t in tools]

    async def call(self, name: str, args: dict) -> dict:
        res = await self.session.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.isError:
            raise RuntimeError(text)
        if getattr(res, "structuredContent", None):
            sc = res.structuredContent
            return sc.get("result", sc) if isinstance(sc, dict) else sc
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}


def get_executor(kind: str | None = None):
    kind = (kind or os.getenv("TOOL_TRANSPORT", "local")).lower()
    return MCPToolExecutor() if kind == "mcp" else LocalToolExecutor()
