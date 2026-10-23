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


def _client_env(tmp_path, writes):
    database = tmp_path / "memory.db"
    with MemoryStore(database, vector_backend="python") as store:
        store.remember(namespace="test", memory_id="r", content="equity research", known_at="2025-01-01")
    return {**os.environ, "MEMORY_DB": str(database), "MEMORY_NAMESPACE": "test",
            "MEMORY_VECTOR_BACKEND": "python", "MEMORY_ENABLE_WRITES": "true" if writes else "false",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}


@pytest.mark.parametrize("adapter", [[], ["--sdk"]], ids=["stdlib", "sdk"])
def test_official_client_gets_same_behaviour_from_both_adapters(tmp_path, adapter):
    pytest.importorskip("mcp")
    from mcp import Client, StdioServerParameters
    env = _client_env(tmp_path, writes=True)

    async def check():
        params = StdioServerParameters(command=sys.executable, args=["-m", "personal_memory.server", *adapter], env=env)
        async with Client(params) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert set(tools) == {"memory_search", "memory_history", "memory_status", "memory_remember", "memory_forget"}
            assert tools["memory_search"].annotations.read_only_hint
            assert tools["memory_remember"].annotations.destructive_hint is False
            assert tools["memory_search"].input_schema["properties"]["limit"]["maximum"] == 100
            found = await client.call_tool("memory_search", {"query": "equity", "as_of": "2025-02-01"})
            assert not found.is_error and found.structured_content["results"][0]["content"] == "equity research"
            # Store errors reach the model with their reason, not a generic failure.
            bad = await client.call_tool("memory_search", {"query": "x", "as_of": "not-a-date"})
            assert bad.is_error and "Invalid time 'not-a-date'" in bad.content[0].text
            missing = await client.call_tool("memory_forget", {"memory_id": "missing"})
            assert missing.is_error and "No memory with that identity" in missing.content[0].text
            for arguments in ({"query": "x", "limit": 101}, {"query": "x", "limit": True}, {"query": None}):
                assert (await client.call_tool("memory_search", arguments)).is_error
            too_long = await client.call_tool("memory_remember", {"memory_id": "t", "content": "x", "title": "t" * 1001})
            assert too_long.is_error
            written = await client.call_tool("memory_remember", {"memory_id": "n", "content": "new fact"})
            assert not written.is_error and written.structured_content["deleted"] is False

    asyncio.run(check())


def test_sdk_launcher_reports_configuration_errors_in_one_line(tmp_path):
    import subprocess

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "MEMORY_NAMESPACE": " "}
    result = subprocess.run([sys.executable, "-m", "personal_memory.server", "--sdk"], input="",
                            capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 2
    assert result.stderr.startswith("personal-memory configuration error:") and "Traceback" not in result.stderr
