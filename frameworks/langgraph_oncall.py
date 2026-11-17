"""Run the on-call investigation with LangGraph."""

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import oncall_tools as ot
from loop_parts import SYSTEM, opening
from opsworld import World, check, new_world


def oncall_tools(world: World) -> list:
    descriptions = ot.definitions(world)

    @tool(description=descriptions["list_services"].description)
    def list_services() -> str:
        return ot.call(world, "list_services", "{}")

    @tool(description=descriptions["recent_changes"].description)
    def recent_changes(service: str) -> str:
        args = {"service": service}
        return ot.call(world, "recent_changes", json.dumps(args))

    @tool(description=descriptions["search_logs"].description)
    def search_logs(
        service: str,
        query: str,
        detail: Literal["concise", "detailed"] = "concise",
    ) -> str:
        args = {"service": service, "query": query, "detail": detail}
        return ot.call(world, "search_logs", json.dumps(args))

    @tool(description=descriptions["get_metrics"].description)
    def get_metrics(
        service: str,
        detail: Literal["concise", "detailed"] = "concise",
    ) -> str:
        args = {"service": service, "detail": detail}
        return ot.call(world, "get_metrics", json.dumps(args))

    @tool(description=descriptions["search_runbooks"].description)
    def search_runbooks(query: str) -> str:
        args = {"query": query}
        return ot.call(world, "search_runbooks", json.dumps(args))

    return [
        list_services,
        recent_changes,
        search_logs,
        get_metrics,
        search_runbooks,
    ]


def build_app(world: World, model: BaseChatModel):
    tools = oncall_tools(world)
    llm = model.bind_tools(tools)

    def think(state: MessagesState) -> MessagesState:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=InMemorySaver())


def run_scenario(
    scenario: str, model: BaseChatModel | None = None
) -> tuple[str, bool, str]:
    world = new_world(scenario)
    if model is None:
        model = ChatOpenAI(
            model="qwen3:8b",
            api_key="ollama",
            base_url="http://localhost:11434/v1/",
            temperature=0,
            seed=7,
        )
    app = build_app(world, model)
    state = app.invoke(
        {"messages": opening(world)},
        {
            "recursion_limit": 24,
            "configurable": {"thread_id": "inc-1"},
        },
    )
    answer = str(state["messages"][-1].content)
    verdict = check(world)
    return answer, verdict.passed, verdict.reason


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    args = parser.parse_args()
    answer, passed, reason = run_scenario(args.scenario)
    print(answer)
    print(f"opsworld check: {'PASS' if passed else 'FAIL'} — {reason}")


if __name__ == "__main__":
    main()
