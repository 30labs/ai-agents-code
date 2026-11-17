"""Exercise the MCP server over an in-memory client session."""

import pytest

mcp = pytest.importorskip("mcp")

import anyio

from mcp.client import Client

from mcp_server import mcp as server


async def exercise_server() -> None:
    async with Client(server, raise_exceptions=True) as client:
        listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} == {
            "list_services",
            "recent_changes",
            "search_logs",
            "get_metrics",
            "search_runbooks",
        }
        assert all(tool.annotations.read_only_hint for tool in listed.tools)
        metrics = next(
            tool for tool in listed.tools if tool.name == "get_metrics"
        )
        assert metrics.input_schema["properties"]["service"]["enum"]
        reading = await client.call_tool("list_services", {})
        assert reading.is_error is False
        assert reading.content

        bad = await client.call_tool(
            "get_metrics", {"service": "missing-service"}
        )
        assert bad.is_error is True
        assert bad.model_dump(by_alias=True)["isError"] is True
        assert bad.content[0].text.startswith(
            "Error executing tool get_metrics"
        )


def test_mcp_lists_calls_and_reports_tool_errors() -> None:
    anyio.run(exercise_server)
