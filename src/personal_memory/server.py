"""MCP stdio launcher with an optional official SDK v2 adapter."""

import argparse
import sys
from contextlib import contextmanager
from typing import Annotated, Any

from . import __version__
from .stdio import INSTRUCTIONS, STORE_ERRORS, settings, tool_definitions
from .store import MAX_ID_CHARS, MAX_TITLE_CHARS, MemoryStore


def make_server(env=None):
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
        from pydantic import Field
    except ImportError as exc:
        raise RuntimeError("Install the MCP v2 extra: python -m pip install -e '.[mcp]'") from exc
    config = settings(env)
    database, namespace, backend, writes = (config[k] for k in ("database", "namespace", "backend", "writes"))
    server = MCPServer("personal-memory", version=__version__, instructions=INSTRUCTIONS)
    # Share descriptions and hints with the dependency-free adapter so both stay in step.
    specs = {tool["name"]: tool for tool in tool_definitions(writes=True)}

    def tool(name: str):
        hints = specs[name]["annotations"]
        annotations = ToolAnnotations(read_only_hint=hints["readOnlyHint"], destructive_hint=hints["destructiveHint"],
                                      idempotent_hint=hints["idempotentHint"], open_world_hint=False)
        # SDK v2 does not infer structured output from a bare dict return annotation.
        return server.tool(name=name, description=specs[name]["description"], annotations=annotations,
                           structured_output=True)

    @contextmanager
    def opened():
        # The SDK withholds the text of unexpected exceptions; ToolError passes it to the model.
        try:
            with MemoryStore(database, vector_backend=backend, read_only=not writes) as store:
                yield store
        except STORE_ERRORS as exc:
            raise ToolError(str(exc)) from exc

    # Same bounds as the stdio adapter's input schemas.
    Text = Annotated[str, Field(max_length=10_000, strict=True)]
    Identity = Annotated[str, Field(max_length=MAX_ID_CHARS, strict=True)]
    Title = Annotated[str, Field(max_length=MAX_TITLE_CHARS, strict=True)]
    Content = Annotated[str, Field(max_length=2_000_000, strict=True)]
    Time = Annotated[str, Field(max_length=100, strict=True)] | None
    Limit = Annotated[int, Field(ge=1, le=100, strict=True)]

    @tool("memory_search")
    def memory_search(query: Text, as_of: Time = None, valid_at: Time = None, limit: Limit = 8) -> dict[str, Any]:
        with opened() as store:
            return store.search(query, namespace=namespace, as_of=as_of, valid_at=valid_at, limit=limit,
                                candidate_limit=max(50, limit))

    @tool("memory_history")
    def memory_history(memory_id: Identity, as_of: Time = None) -> dict[str, Any]:
        with opened() as store:
            return {"namespace": namespace, "revisions": store.history(namespace=namespace, memory_id=memory_id, as_of=as_of)}

    @tool("memory_status")
    def memory_status() -> dict[str, Any]:
        with opened() as store:
            status = store.stats()
            status.pop("namespaces", None)
            status["revisions"] = store.connection.execute(
                "SELECT count(*) FROM revisions WHERE namespace=?", (namespace,)).fetchone()[0]
            return {**status, "namespace": namespace, "writes_enabled": writes, "transport": "stdio-sdk"}

    if writes:
        @tool("memory_remember")
        def memory_remember(memory_id: Identity, content: Content, title: Title = "",
                            valid_from: Time = None, valid_to: Time = None) -> dict[str, Any]:
            with opened() as store:
                return store.remember(namespace=namespace, memory_id=memory_id, content=content, title=title,
                                      valid_from=valid_from, valid_to=valid_to, source={"kind": "mcp-sdk"})

        @tool("memory_forget")
        def memory_forget(memory_id: Identity) -> dict[str, Any]:
            with opened() as store:
                return store.forget(namespace=namespace, memory_id=memory_id)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory-mcp", description="Local memory MCP stdio server")
    parser.add_argument("--sdk", action="store_true", help="Use optional official MCP SDK v2 adapter")
    args = parser.parse_args(argv)
    if not args.sdk:
        from .stdio import main as run_stdio
        return run_stdio()
    try:
        server = make_server()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"personal-memory configuration error: {exc}", file=sys.stderr, flush=True)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
