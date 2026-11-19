"""Chapter 5's loop, as printed there."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loop_parts import Model, Run, Turn, key
from oncall_tools import call
from opsworld import World


@dataclass
class Budget:
    steps: int = 12
    tokens: int = 40_000
    seconds: float = 300
    errors: int = 3


def run(
    model: Model, world: World,
    messages: list[Any], budget: Budget,
    halted: Callable[[], bool] = lambda: False,
) -> Run:
    r = Run()
    start, last_in, errors = time.monotonic(), 0, 0
    seen: dict[str, int] = {}
    while True:
        why = limit(budget, r, last_in, errors,
                    time.monotonic() - start)
        if why or halted():
            r.reason = why or "stopped by a person"
            return r
        turn = model(messages)
        r.steps += 1
        last_in = turn.input_tokens
        r.tokens += last_in + turn.output_tokens
        messages.append(turn.message())
        if turn.calls:
            bad = act(world, turn, r,
                      seen, messages)
            errors = errors + 1 if bad else 0
            continue
        if turn.finish == "stop":
            r.answer = turn.text
            r.reason = "answered"
            return r
        r.reason = f"model stopped: {turn.finish}"
        return r


def limit(
    b: Budget, r: Run, last_in: int,
    errors: int, spent: float,
) -> str:
    if r.steps >= b.steps:
        return f"step limit of {b.steps}"
    if r.tokens + last_in > b.tokens:
        return f"token budget of {b.tokens}"
    if spent > b.seconds:
        return f"time limit of {b.seconds:.0f}s"
    if r.repeats > 1:
        return "repeated a call after a warning"
    if errors >= b.errors:
        return f"{errors} failed steps in a row"
    return ""


def act(
    world: World, turn: Turn, r: Run,
    seen: dict[str, int], messages: list[Any],
) -> bool:
    bad = True
    for c in turn.calls:
        if (k := key(c)) in seen:
            r.repeats += 1
            text = ("ERROR: you made this call"
                    f" at step {seen[k]}."
                    " Make another.")
        else:
            seen[k] = r.steps
            r.read.append(f"{c.name} {c.arguments}")
            text = call(world, c.name, c.arguments)
        bad = bad and text.startswith("ERROR")
        messages.append({"role": "tool",
                         "tool_call_id": c.id,
                         "content": text})
    return bad
