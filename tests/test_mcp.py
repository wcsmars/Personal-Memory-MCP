"""Real subprocess round-trip, exercised when the official MCP extra is installed."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from personal_memory.store import MemoryStore


def test_official_mcp_stdio_roundtrip(tmp_path):
    pytest.importorskip("mcp")
    from mcp import Client, StdioServerParameters
    database = tmp_path / "memory.db"
    with MemoryStore(database, vector_backend="python") as store:
        store.remember(namespace="test", memory_id="r", content="equity research", known_at="2025-01-01")
        store.remember(namespace="test", memory_id="r", content="trading", known_at="2025-03-01")
    env = {**os.environ, "MEMORY_DB": str(database), "MEMORY_NAMESPACE": "test",
           "MEMORY_VECTOR_BACKEND": "python", "MEMORY_ENABLE_WRITES": "false",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}

    async def check():
        params = StdioServerParameters(command=sys.executable, args=["-m", "personal_memory.server", "--sdk"], env=env)
        async with Client(params) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert "memory_search" in names
            assert "memory_remember" not in names
            result = await client.call_tool("memory_search", {"query": "equity", "as_of": "2025-02-01"})
            assert not result.is_error
            assert result.structured_content["results"][0]["content"] == "equity research"

    asyncio.run(check())
