"""Parts of chapter 6's planners that the chapter describes."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from openai.types.shared_params import (
    ResponseFormatJSONObject,
)
from pydantic import BaseModel

from loop_parts import SYSTEM, Model, Turn, cached
from oncall_tools import definitions
from opsworld import (
    Remedy,
    World,
    check,
    rollback_deploy,
    scale_service,
    set_flag,
)

REMEDY_RULE = (
    "End the answer with one line that names the remedy:"
    " 'REMEDY: rollback_deploy <deploy id>',"
    " 'REMEDY: set_flag <flag> off',"
    " 'REMEDY: scale_service <service> <replicas>',"
    " or 'REMEDY: none' when no action is needed."
)

PLAN_SHAPE = (
    '{"steps": [{"id": 1, "tool": "<name>",'
    ' "arguments": {...}, "after": []}]}'
)

FINISH_SHAPE = '{"answer": "<text>", "more": [<steps>]}'


def react_opening(world: World) -> list[dict[str, Any]]:
    "Chapter 5's opening, with the remedy line the suite reads."
    alert = world.alert
    return [
        {
            "role": "system",
            "content": f"{SYSTEM} {REMEDY_RULE}",
        },
        {
            "role": "user",
            "content": f"Alert on {alert.service}: "
            f"{alert.message}.",
        },
    ]


def plan_opening(world: World) -> list[dict[str, Any]]:
    "The planner's prompt: the tools, the shape, the alert."
    tools = "\n".join(
        json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters["properties"],
            }
        )
        for tool in definitions(world).values()
    )
    system = (
        "You are on call. Before any tool runs, write the"
        " whole investigation as a plan. Reply with JSON"
        f" only, shaped like {PLAN_SHAPE}. 'after' lists the"
        " ids of steps that must finish first; leave it"
        " empty when a step needs nothing earlier. Every"
        " argument must be known now, from the alert or the"
        f" tool descriptions. Tools, one per line:\n{tools}"
    )
    alert = world.alert
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Alert on {alert.service}: "
            f"{alert.message}.",
        },
    ]


def results_message(
    steps: list[Any], results: dict[int, str]
) -> dict[str, Any]:
    "Every result of a plan, sent back in one message."
    lines = ["Results:"]
    for step in steps:
        args = json.dumps(step.arguments, sort_keys=True)
        lines.append(f"[{step.id}] {step.tool} {args}")
        lines.append(results[step.id])
    lines.append(
        f"Reply with JSON only, shaped like {FINISH_SHAPE}."
        " Either answer with the cause, the evidence and"
        " one remedy, leaving 'more' empty, or leave"
        " 'answer' empty and list in 'more' the steps"
        f" still needed, as in the plan. {REMEDY_RULE}"
    )
    return {"role": "user", "content": "\n".join(lines)}


def ask[T: BaseModel](
    model: Model,
    messages: list[dict[str, Any]],
    kind: type[T],
    valid: Callable[[T], object],
    tries: int = 2,
) -> T | None:
    """Ask for JSON of one shape; send the error back once.

    `valid` raises ValueError for a value of the right
    shape that still cannot be used.
    """
    for _ in range(tries):
        turn = model(messages)
        messages.append(
            {"role": "assistant", "content": turn.text}
        )
        try:
            value = kind.model_validate_json(turn.text)
            valid(value)
        except ValueError as exc:
            first = str(exc).splitlines()[0][:200]
            messages.append(
                {
                    "role": "user",
                    "content": f"ERROR: {first}. Reply with"
                    " JSON only, in the shape asked for.",
                }
            )
            continue
        return value
    return None


def local_json(
    client: OpenAI,
    model: str = "qwen3:8b",
    seed: int = 7,
    temperature: float = 0,
) -> Model:
    "A local model asked for JSON, with no tools offered."

    json_only: ResponseFormatJSONObject = {
        "type": "json_object"
    }

    def answer(messages: list[dict[str, Any]]) -> Turn:
        reply = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format=json_only,
            temperature=temperature,
            seed=seed,
        )
        choice = reply.choices[0]
        extra = choice.message.model_extra or {}
        usage = reply.usage
        return Turn(
            choice.finish_reason,
            choice.message.content or "",
            [],
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            str(extra.get("reasoning") or ""),
            cached(usage),
        )

    return answer


@dataclass
class PlanRun:
    "What a planned run did, for the report and the suite."

    answer: str = ""
    reason: str = ""
    rounds: int = 0
    read: list[str] = field(default_factory=list)


@dataclass
class Meter:
    "Counts what a model was asked and what it cost."

    model: Model
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_turns: int = 0
    peak_input: int = 0
    cached_tokens: int = 0

    def __call__(self, messages: list[dict[str, Any]]) -> Turn:
        turn = self.model(messages)
        self.calls += 1
        self.input_tokens += turn.input_tokens
        self.peak_input = max(
            self.peak_input, turn.input_tokens
        )
        self.output_tokens += turn.output_tokens
        self.cached_tokens += turn.cached_tokens
        self.tool_turns += 1 if turn.calls else 0
        return turn


_REMEDY = re.compile(r"^\W*REMEDY:\s*(.+?)\W*$", re.M)


def parse_remedy(answer: str) -> Remedy | None:
    """The remedy an answer's last REMEDY line names.

    None means the answer proposes no action; a missing
    or unreadable line raises ValueError.
    """
    found = _REMEDY.findall(answer)
    if not found:
        raise ValueError("no REMEDY line")
    words = found[-1].replace("`", "").split()
    match words:
        case ["none"]:
            return None
        case ["rollback_deploy", deploy_id]:
            return Remedy(
                "rollback_deploy", {"deploy_id": deploy_id}
            )
        case ["set_flag", name, "off" | "on" as state]:
            return Remedy(
                "set_flag",
                {"name": name, "enabled": state == "on"},
            )
        case ["scale_service", service, count] if (
            count.isdigit()
        ):
            return Remedy(
                "scale_service",
                {"service": service, "replicas": int(count)},
            )
    raise ValueError(f"unreadable REMEDY line {found[-1]!r}")


def judge(world: World, answer: str) -> tuple[bool, str, str]:
    """Apply the proposed remedy as an approver would, then
    check the world. Returns (passed, reason, remedy)."""
    try:
        remedy = parse_remedy(answer)
    except ValueError as exc:
        return False, str(exc), ""
    text = "none"
    if remedy is not None:
        args = remedy.arguments
        text = f"{remedy.tool} {json.dumps(args)}"
        try:
            if remedy.tool == "rollback_deploy":
                rollback_deploy(world, args["deploy_id"])
            elif remedy.tool == "set_flag":
                set_flag(world, args["name"], args["enabled"])
            else:
                scale_service(
                    world, args["service"], args["replicas"]
                )
        except ValueError as exc:
            return False, f"remedy failed: {exc}", text
    outcome = check(world)
    return outcome.passed, outcome.reason, text
