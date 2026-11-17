"""Expose oncall's five reading tools over MCP."""

from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import oncall_tools as ot
from opsworld import new_world

ops = new_world("downstream-tax")
tools = ot.definitions(ops)
ServiceName = Literal.__getitem__(tuple(sorted(ops.services)))
mcp = MCPServer("opsworld-tools", version="0.4.0")
READ_HINTS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


@mcp.tool(
    description=tools["list_services"].description,
    annotations=READ_HINTS,
)
async def list_services() -> str:
    try:
        return ot.call(ops, "list_services", "{}")
    except ot.ToolError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(
    description=tools["recent_changes"].description,
    annotations=READ_HINTS,
)
async def recent_changes(service: ServiceName) -> str:
    try:
        return ot.recent_changes(ops, service)
    except ot.ToolError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(
    description=tools["search_logs"].description,
    annotations=READ_HINTS,
)
async def search_logs(
    service: ServiceName,
    query: str,
    detail: Literal["concise", "detailed"] = "concise",
) -> str:
    try:
        return ot.search_logs(ops, service, query, detail)
    except ot.ToolError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(
    description=tools["get_metrics"].description,
    annotations=READ_HINTS,
)
async def get_metrics(
    service: ServiceName,
    detail: Literal["concise", "detailed"] = "concise",
) -> str:
    try:
        return ot.get_metrics(ops, service, detail)
    except ot.ToolError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(
    description=tools["search_runbooks"].description,
    annotations=READ_HINTS,
)
async def search_runbooks(query: str) -> str:
    try:
        return ot.search_runbooks(ops, query)
    except ot.ToolError as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    mcp.run()
