"""Real subprocess checks for the dependency-free MCP tools transport."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from personal_memory.store import MemoryStore, instant
from personal_memory.stdio import PROTOCOL_VERSION


def request(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def handshake(version=PROTOCOL_VERSION):
    return [request("initialize", {"protocolVersion": version, "capabilities": {},
                                    "clientInfo": {"name": "test-client", "version": "1"}}),
            {"jsonrpc": "2.0", "method": "notifications/initialized"}]


def call(name, arguments=None, request_id=2):
    return request("tools/call", {"name": name, "arguments": arguments or {}}, request_id)


def run_server(tmp_path, messages, *, writes=False, namespace="test", module="personal_memory.stdio"):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
           "MEMORY_DB": str(tmp_path / "memory.db"), "MEMORY_NAMESPACE": namespace,
           "MEMORY_VECTOR_BACKEND": "python", "MEMORY_ENABLE_WRITES": "true" if writes else "false"}
    payload = "\n".join(item if isinstance(item, str) else json.dumps(item) for item in messages) + "\n"
    result = subprocess.run([sys.executable, "-m", module], input=payload,
                            text=True, capture_output=True, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stderr == "", result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


def test_stdio_handshake_discovery_temporal_search_and_namespace(tmp_path):
    with MemoryStore(tmp_path / "memory.db", vector_backend="python") as store:
        store.remember(namespace="test", memory_id="role", content="equity research career", known_at="2025-01-01")
        store.remember(namespace="test", memory_id="role", content="quantitative trading career", known_at="2025-03-01")
        store.remember(namespace="other", memory_id="private", content="secret career information", known_at="2025-01-01")
    replies = run_server(tmp_path, handshake() + [request("tools/list", request_id=2),
                        call("memory_search", {"query": "career", "as_of": "2025-02-01"}, 3),
                        call("memory_search", {"query": "career", "as_of": "2025-04-01"}, 4),
                        call("memory_status", request_id=5),
                        call("memory_history", {"memory_id": "role", "as_of": "2025-02-01"}, 6)])
    assert [item["id"] for item in replies] == [1, 2, 3, 4, 5, 6]
    assert replies[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert replies[0]["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert {tool["name"] for tool in replies[1]["result"]["tools"]} == {"memory_search", "memory_history", "memory_status"}
    assert replies[2]["result"]["structuredContent"]["results"][0]["content"] == "equity research career"
    assert replies[3]["result"]["structuredContent"]["results"][0]["content"] == "quantitative trading career"
    status = replies[4]["result"]["structuredContent"]
    assert status["revisions"] == 2 and status["namespace"] == "test" and not status["writes_enabled"]
    assert "namespaces" not in status
    assert len(replies[5]["result"]["structuredContent"]["revisions"]) == 1
    assert "secret" not in json.dumps(replies)


def test_stdio_negotiates_supported_version(tmp_path):
    replies = run_server(tmp_path, handshake("2099-01-01") + [request("ping", request_id="probe")])
    assert replies[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert replies[1] == {"jsonrpc": "2.0", "id": "probe", "result": {}}


def test_default_server_launcher_works_without_optional_sdk(tmp_path):
    replies = run_server(tmp_path, handshake() + [request("tools/list", request_id=2)], module="personal_memory.server")
    assert replies[-1]["result"]["tools"]


def test_stdio_oversized_line_is_bounded_and_recovers(tmp_path):
    from personal_memory.stdio import MAX_MESSAGE_BYTES

    replies = run_server(tmp_path, ["x" * (MAX_MESSAGE_BYTES + 5)] + handshake() + [request("ping", request_id=2)])
    assert replies[0]["error"]["code"] == -32600
    assert replies[-1]["result"] == {}


def test_stdio_initialization_is_required(tmp_path):
    replies = run_server(tmp_path, [request("tools/list", request_id=0), request("ping", request_id=1),
                        handshake()[0], request("tools/list", request_id=3), handshake()[1],
                        request("tools/list", request_id=4), handshake()[0]])
    assert replies[0]["error"]["code"] == -32002
    assert replies[1]["result"] == {}
    assert replies[3]["error"]["code"] == -32002
    assert "tools" in replies[4]["result"]
    assert replies[5]["error"]["code"] == -32600


def test_stdio_protocol_errors_and_notification_silence(tmp_path):
    messages = handshake() + ["{broken", "[]", '{"jsonrpc":"2.0","id":NaN,"method":"ping"}',
               '{"jsonrpc":"2.0","id":2,"id":3,"method":"ping"}',
               request("unknown", request_id=10), request("tools/call", [], 11),
               call("unknown_tool", request_id=12),
               {"jsonrpc": "2.0", "method": "unknown_notification"},
               {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "memory_remember", "arguments": {"memory_id": "bad", "content": "must not write"}}},
               call("memory_status", request_id=13)]
    replies = run_server(tmp_path, messages, writes=True)
    assert len(replies) == 9
    assert [r["error"]["code"] for r in replies[1:8]] == [-32700, -32600, -32700, -32700, -32601, -32602, -32602]
    assert replies[-1]["result"]["structuredContent"]["revisions"] == 0


@pytest.mark.parametrize("arguments", [{"query": 12}, {"query": "career", "limit": True},
                                      {"query": "career", "limit": 101}, {"query": "career", "namespace": "other"},
                                      {"query": "career", "as_of": "not-a-date"}])
def test_stdio_tool_validation_uses_error_result(tmp_path, arguments):
    replies = run_server(tmp_path, handshake() + [call("memory_search", arguments)])
    assert replies[-1]["result"]["isError"] is True
    assert "error" not in replies[-1]


def test_stdio_read_only_rejects_write_tool(tmp_path):
    replies = run_server(tmp_path, handshake() + [call("memory_remember", {"memory_id": "role", "content": "example"})])
    assert replies[-1]["error"]["code"] == -32602


def test_stdio_writes_are_opt_in_and_cannot_backdate_knowledge(tmp_path):
    before = instant()
    messages = handshake() + [request("tools/list", request_id=2),
               call("memory_remember", {"memory_id": "role", "content": "fictional research role", "known_at": "2000-01-01"}, 3),
               call("memory_remember", {"memory_id": "role", "content": "fictional research role"}, 4),
               call("memory_search", {"query": "research"}, 5),
               call("memory_forget", {"memory_id": "role"}, 6),
               call("memory_search", {"query": "research"}, 7),
               call("memory_history", {"memory_id": "role"}, 8)]
    replies = run_server(tmp_path, messages, writes=True)
    assert len(replies[1]["result"]["tools"]) == 5
    assert replies[2]["result"]["isError"] is True
    created = replies[3]["result"]["structuredContent"]
    assert before <= created["known_at"] <= instant()
    assert replies[4]["result"]["structuredContent"]["results"]
    assert replies[5]["result"]["structuredContent"]["deleted"] is True
    assert not replies[6]["result"]["structuredContent"]["results"]
    assert len(replies[7]["result"]["structuredContent"]["revisions"]) == 2


def test_stdio_execution_failure_remains_tool_result(tmp_path):
    replies = run_server(tmp_path, handshake() + [call("memory_forget", {"memory_id": "missing"})], writes=True)
    assert replies[-1]["result"]["isError"] is True


def test_stdio_rejects_invalid_utf8_without_crashing(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
           "MEMORY_DB": str(tmp_path / "memory.db"), "MEMORY_VECTOR_BACKEND": "python"}
    result = subprocess.run([sys.executable, "-m", "personal_memory.stdio"],
                            input=b"\xff\n", capture_output=True, env=env, timeout=10)
    assert result.returncode == 0
    assert json.loads(result.stdout)["error"]["code"] == -32700
