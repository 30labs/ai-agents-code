"""Run the on-call investigation with Pydantic AI."""

from typing import Literal

from pydantic_ai import Agent, ModelRetry, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

import oncall_tools as ot
from loop_parts import SYSTEM, opening
from opsworld import World, check, new_world


def build_agent(world: World, model: Model) -> Agent[None, str]:
    agent = Agent(
        model,
        instructions=SYSTEM,
        model_settings={"temperature": 0, "seed": 7},
    )

    @agent.tool_plain(description=ot.definitions(world)["list_services"].description)
    def list_services() -> str:
        return ot.call(world, "list_services", "{}")

    @agent.tool_plain(description=ot.definitions(world)["recent_changes"].description)
    def recent_changes(service: str) -> str:
        try:
            return ot.recent_changes(world, service)
        except ot.ToolError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool_plain(description=ot.definitions(world)["search_logs"].description)
    def search_logs(
        service: str,
        query: str,
        detail: Literal["concise", "detailed"] = "concise",
    ) -> str:
        try:
            return ot.search_logs(world, service, query, detail)
        except ot.ToolError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool_plain(description=ot.definitions(world)["get_metrics"].description)
    def get_metrics(
        service: str,
        detail: Literal["concise", "detailed"] = "concise",
    ) -> str:
        try:
            return ot.get_metrics(world, service, detail)
        except ot.ToolError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool_plain(description=ot.definitions(world)["search_runbooks"].description)
    def search_runbooks(query: str) -> str:
        return ot.search_runbooks(world, query)

    return agent


def run_scenario(scenario: str, model: Model | None = None) -> tuple[str, bool, str]:
    world = new_world(scenario)
    if model is None:
        provider = OllamaProvider(base_url="http://localhost:11434/v1/")
        model = OllamaModel("qwen3:8b", provider=provider)
    result = build_agent(world, model).run_sync(
        str(opening(world)[1]["content"]),
        usage_limits=UsageLimits(
            request_limit=12, total_tokens_limit=40_000
        ),
    )
    verdict = check(world)
    return str(result.output), verdict.passed, verdict.reason


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
