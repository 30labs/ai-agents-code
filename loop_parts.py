"""The parts of chapter 5's loop that the chapter describes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from openai.types import CompletionUsage

from opsworld import World


@dataclass(frozen=True)
class Call:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Turn:
    """One model response, from a provider or a stand-in."""

    finish: str
    text: str
    calls: list[Call] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning: str = ""
    cached_tokens: int = 0

    def message(self) -> dict[str, Any]:
        "The assistant message to append to the history."
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.text,
        }
        if self.calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in self.calls
            ]
        return message


Model = Callable[[list[dict[str, Any]]], Turn]


def cached(usage: CompletionUsage | None) -> int:
    "Prompt tokens the server reported reusing from cache."
    if usage is None or usage.prompt_tokens_details is None:
        return 0
    return usage.prompt_tokens_details.cached_tokens or 0


def local_model(
    client: OpenAI,
    tools: list[Any],
    model: str = "qwen3:8b",
    seed: int = 7,
    temperature: float = 0,
) -> Model:
    "Chapter 2's request, returning a Turn."

    def ask(messages: list[dict[str, Any]]) -> Turn:
        reply = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,
            temperature=temperature,
            seed=seed,
        )
        choice = reply.choices[0]
        calls = [
            Call(
                item.id,
                item.function.name,
                item.function.arguments,
            )
            for item in choice.message.tool_calls or []
            if item.type == "function"
        ]
        extra = choice.message.model_extra or {}
        usage = reply.usage
        return Turn(
            choice.finish_reason,
            choice.message.content or "",
            calls,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            str(extra.get("reasoning") or ""),
            cached(usage),
        )

    return ask


Step = tuple[str, dict[str, Any]] | str


def stand_in(
    script: Iterable[Step], writes: int = 150
) -> Model:
    """A scripted model: each step is a call or an answer.

    Input tokens are the characters it was sent over four,
    so a long history costs roughly what it would.
    """
    steps = iter(script)

    def ask(messages: list[dict[str, Any]]) -> Turn:
        step = next(steps)
        size = len(json.dumps(messages)) // 4
        if isinstance(step, str):
            return Turn("stop", step, [], size, writes)
        name, args = step
        call = Call(
            f"call_{len(messages)}",
            name,
            json.dumps(args, separators=(",", ":")),
        )
        return Turn("tool_calls", "", [call], size, writes)

    return ask


SYSTEM = (
    "You are on call. Read evidence with the tools"
    " before you answer. When you know the cause,"
    " answer with it, the evidence for it, and one"
    " remedy: the deploy id to roll back, the flag"
    " to turn off, or the service to scale."
)


def opening(world: World) -> list[dict[str, Any]]:
    "The system prompt and the alert, as the run starts."
    alert = world.alert
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Alert on {alert.service}: "
            f"{alert.message}.",
        },
    ]


@dataclass
class Run:
    "What one run did, for the report and the incident note."

    reason: str = ""
    answer: str = ""
    steps: int = 0
    tokens: int = 0
    repeats: int = 0
    read: list[str] = field(default_factory=list)


def key(call: Call) -> str:
    "The same call however its arguments were spelled."
    try:
        args = json.loads(call.arguments)
    except json.JSONDecodeError:
        return call.name + call.arguments
    return call.name + json.dumps(args, sort_keys=True)


def report(run: Run) -> str:
    "Built by code from the record, never by the model."
    head = "answered" if run.answer else "stopped"
    lines = [
        f"{head}: {run.reason}",
        f"{run.steps} steps, {run.tokens} tokens",
    ]
    if run.read:
        lines.append("read:")
    lines += [f"  {item}" for item in run.read]
    if not run.answer:
        lines.append("not done: no cause, no remedy")
    return "\n".join(lines)


def traced(model: Model) -> Model:
    "Print each turn's stop reason, calls and tokens."

    def ask(messages: list[dict[str, Any]]) -> Turn:
        turn = model(messages)
        calls = [call.name for call in turn.calls] or ["-"]
        print(
            turn.finish,
            *calls,
            turn.input_tokens,
            turn.output_tokens,
        )
        return turn

    return ask
