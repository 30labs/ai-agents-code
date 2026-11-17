"""Offline smoke tests for the framework rebuilds."""

from typing import Any

import pytest


def test_pydantic_ai_runs_with_test_model(capsys: Any) -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    from frameworks.pydantic_ai_oncall import run_scenario

    answer, passed, reason = run_scenario(
        "downstream-tax", model=TestModel(call_tools=["list_services"])
    )
    assert answer
    assert passed is False
    assert "action history" in reason
    assert capsys.readouterr().out == ""


def test_langgraph_runs_with_fake_chat_model() -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from frameworks.langgraph_oncall import run_scenario

    class FakeChatModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools: list[Any]) -> "FakeChatModel":
            assert len(tools) == 5
            return self

        def invoke(self, messages: list[Any]) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_services",
                            "args": {},
                            "id": "call-list-services",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Investigation complete.")

    answer, passed, reason = run_scenario(
        "downstream-tax", model=FakeChatModel()  # type: ignore[arg-type]
    )
    assert answer == "Investigation complete."
    assert passed is False
    assert "action history" in reason


def test_langgraph_returns_tool_errors_to_the_model() -> None:
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage, ToolMessage

    from frameworks.langgraph_oncall import run_scenario

    seen: list[str] = []

    class FakeChatModel:
        def bind_tools(self, tools: list[Any]) -> "FakeChatModel":
            return self

        def invoke(self, messages: list[Any]) -> AIMessage:
            if isinstance(messages[-1], ToolMessage):
                seen.append(str(messages[-1].content))
                return AIMessage(content="Done.")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_metrics",
                        "args": {"service": "no-such-service"},
                        "id": "call-bad",
                        "type": "tool_call",
                    }
                ],
            )

    answer, _, _ = run_scenario(
        "downstream-tax", model=FakeChatModel()  # type: ignore[arg-type]
    )
    assert answer == "Done."
    assert seen and seen[0].startswith("ERROR:")
