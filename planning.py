import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel

from loop_parts import Model
from oncall_tools import call
from opsworld import World
from plan_parts import PlanRun, ask, results_message


class Step(BaseModel):
    id: int
    tool: str
    arguments: dict[str, Any]
    after: list[int] = []


class Plan(BaseModel):
    steps: list[Step]


class Finish(BaseModel):
    answer: str = ""
    more: list[Step] = []


def waves(steps: list[Step]) -> list[list[Step]]:
    done: set[int] = set()
    left, out = list(steps), []
    while left:
        ready = [s for s in left
                 if set(s.after) <= done]
        if not ready:
            raise ValueError("a step waits forever")
        out.append(ready)
        done |= {s.id for s in ready}
        left = [s for s in left
                if s.id not in done]
    return out


def execute(
    world: World, steps: list[Step], parallel: bool,
) -> tuple[dict[int, str], int]:
    order = waves(steps)
    if not parallel:
        order = [[s] for w in order for s in w]
    out: dict[int, str] = {}

    def one(s: Step) -> str:
        if any(out[d].startswith("ERROR")
               for d in s.after):
            return "ERROR: a step it needs failed"
        args = json.dumps(s.arguments)
        return call(world, s.tool, args)

    with ThreadPoolExecutor() as pool:
        for wave in order:
            texts = list(pool.map(one, wave))
            for i, s in enumerate(wave):
                out[s.id] = texts[i]
    return out, len(order)


def plan_run(
    model: Model, world: World,
    messages: list[Any], parallel: bool,
    replans: int = 2,
) -> PlanRun:
    r = PlanRun()
    plan = ask(model, messages, Plan,
               lambda p: waves(p.steps))
    if plan is None:
        r.reason = "no valid plan"
        return r
    steps = plan.steps

    for _ in range(replans + 1):
        results, rounds = execute(
            world, steps, parallel)
        r.rounds += rounds
        for s in steps:
            args = json.dumps(s.arguments)
            r.read.append(f"{s.tool} {args}")
        messages.append(
            results_message(steps, results))
        end = ask(model, messages, Finish,
                  lambda f: waves(f.more))
        if end is None:
            r.reason = "no valid finish"
            return r
        if not end.more:
            r.answer = end.answer
            r.reason = "answered"
            return r
        steps = end.more
    r.reason = f"still planning after {replans}"
    return r
