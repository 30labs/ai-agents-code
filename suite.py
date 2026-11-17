"""Run one strategy over the opsworld scenarios, k trials each.

Writes one record per trial to runs.json and one run card per
invocation to run_card.json, in the format of the book's plan.
The command line runs only when this file is run directly; the
book's build imports nothing from it that calls a model.

    python suite.py react --trials 5
    python suite.py parallel --scenarios checkout-deploy
    python suite.py team --trials 5
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from loop_parts import Model, local_model
from oncall_loop import Budget, run
from oncall_tools import definitions
from opsworld import World, new_world, scenarios
from plan_parts import (
    Meter,
    judge,
    local_json,
    plan_opening,
    react_opening,
)
from planning import plan_run
from team import (
    MOST_SUBAGENTS,
    SUB_BUDGET,
    Factory,
    lead_tools,
    team_opening,
    team_run,
)

STRATEGIES = ("react", "plan", "parallel", "team")
HERE = Path(__file__).resolve().parent


@dataclass
class Record:
    "One trial: the plan's fields, plus what reading needs."

    card: str
    strategy: str
    scenario: str
    trial: int
    seed: int
    passed: bool
    check: str
    remedy: str
    ended: str
    model_calls: int
    tool_calls: int
    rounds: int
    input_tokens: int
    output_tokens: int
    seconds: float
    read: list[str]
    answer: str
    peak_input: int
    lead_tokens: int
    subagents: int
    cached_tokens: int


def run_trial(
    strategy: str,
    world: World,
    model: Model,
    budget: Budget,
    factory: Factory | None = None,
) -> dict[str, Any]:
    """One trial of one strategy on one world.

    `model` is the tool-calling model for react, the JSON
    model for the two planners and the lead for team, whose
    subagents' models come from `factory`.
    """
    meter = Meter(model)
    subs: list[Meter] = []
    start = time.monotonic()
    if strategy == "react":
        r = run(meter, world, react_opening(world), budget)
        answer, ended, read = r.answer, r.reason, r.read
        tool_calls = len(r.read) + r.repeats
        rounds = meter.tool_turns
    elif strategy in ("plan", "parallel"):
        p = plan_run(
            meter,
            world,
            plan_opening(world),
            parallel=strategy == "parallel",
        )
        answer, ended, read = p.answer, p.reason, p.read
        tool_calls, rounds = len(p.read), p.rounds
    elif strategy == "team":
        if factory is None:
            raise ValueError("team needs a subagent factory")
        lead, team = team_run(
            meter, factory, world, team_opening(world), budget
        )
        answer, ended = lead.answer, lead.reason
        read = lead.read + [
            f"{s.kind}> {item}"
            for s in team
            for item in s.run.read
        ]
        tool_calls = len(lead.read) + lead.repeats
        tool_calls += sum(
            len(s.run.read) + s.run.repeats for s in team
        )
        rounds = meter.tool_turns
        subs = [s.meter for s in team]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    seconds = time.monotonic() - start
    if answer:
        passed, reason, remedy = judge(world, answer)
    else:
        passed, reason, remedy = False, "no answer", ""
    meters = [meter, *subs]
    return {
        "passed": passed,
        "check": reason,
        "remedy": remedy,
        "ended": ended,
        "model_calls": sum(m.calls for m in meters),
        "tool_calls": tool_calls,
        "rounds": rounds,
        "input_tokens": sum(m.input_tokens for m in meters),
        "output_tokens": sum(m.output_tokens for m in meters),
        "seconds": round(seconds, 2),
        "read": read,
        "answer": answer,
        "peak_input": max(m.peak_input for m in meters),
        "lead_tokens": meter.input_tokens + meter.output_tokens,
        "subagents": len(subs),
        "cached_tokens": sum(m.cached_tokens for m in meters),
    }


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as reply:
        data = json.load(reply)
    if not isinstance(data, dict):
        raise ValueError(f"{url} did not return an object")
    return data


def _gpu() -> str:
    if not shutil.which("nvidia-smi"):
        return "none found"
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return "; ".join(out.stdout.strip().splitlines())


def environment(base_url: str, model: str) -> dict[str, Any]:
    "Model, server and hardware facts for the run card."
    root = base_url.rstrip("/").removesuffix("/v1")
    server, digest = "unknown", "unknown"
    try:
        server = _get_json(f"{root}/api/version")["version"]
        for item in _get_json(f"{root}/api/tags")["models"]:
            if item.get("name") == model:
                digest = item.get("digest", "unknown")
    except (OSError, KeyError, ValueError):
        pass
    return {
        "model": model,
        "model_digest": digest,
        "server": f"Ollama {server}",
        "base_url": base_url,
        "openai_sdk": metadata.version("openai"),
        "python": platform.python_version(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "gpu": _gpu(),
        },
    }


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must hold a JSON list")
    return data


def _save(path: Path, items: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=1) + "\n")
    tmp.replace(path)


def save_record(path: Path, record: dict[str, Any]) -> None:
    "Add a record, replacing an older one of the same trial."
    same = ("strategy", "scenario", "trial")
    kept = [
        item
        for item in _load(path)
        if any(item.get(k) != record[k] for k in same)
    ]
    _save(path, [*kept, record])


def models_for(
    strategy: str,
    world: World,
    base_url: str,
    model: str,
    seed: int,
    temperature: float,
) -> Model:
    "The local model, set up for one strategy and trial."
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="ollama")
    if strategy in ("react", "team"):
        found = (
            lead_tools if strategy == "team" else definitions
        )
        tools = [
            tool.definition("openai")
            for tool in found(world).values()
        ]
        return local_model(
            client, tools, model, seed, temperature
        )
    return local_json(client, model, seed, temperature)


def factory_for(
    base_url: str, model: str, seed: int, temperature: float
) -> Factory:
    "Subagent models: the lead's model and seed, own tools."
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="ollama")

    def make(tools: list[Any]) -> Model:
        return local_model(
            client, tools, model, seed, temperature
        )

    return make


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=STRATEGIES)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=sorted(scenarios()),
        help="scenario ids (default: all)",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument(
        "--base-url", default="http://localhost:11434/v1/"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="trial t uses seed + t",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="above zero, so trials differ (default 0.6)",
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=40_000)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE,
        help="directory for runs.json and run_card.json",
    )
    args = parser.parse_args(argv)
    unknown = set(args.scenarios) - set(scenarios())
    if unknown:
        parser.error(f"unknown scenarios: {sorted(unknown)}")
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    budget = Budget(args.steps, args.tokens, args.seconds)
    started = datetime.now(UTC)
    card_id = f"{args.strategy}-{started:%Y%m%dT%H%M%SZ}"
    card: dict[str, Any] = {
        "id": card_id,
        "strategy": args.strategy,
        "date": started.isoformat(timespec="seconds"),
        **environment(args.base_url, args.model),
        "temperature": args.temperature,
        "seeds": f"{args.seed} + trial",
        "scenarios": args.scenarios,
        "trials": args.trials,
        "budget": asdict(budget),
        **(
            {
                "sub_budget": asdict(SUB_BUDGET),
                "most_subagents": MOST_SUBAGENTS,
            }
            if args.strategy == "team"
            else {}
        ),
        "cost_usd": 0.0,
        "cost_note": "local model; no API spend",
        "finished": None,
    }
    runs = args.out / "runs.json"
    cards = args.out / "run_card.json"
    args.out.mkdir(parents=True, exist_ok=True)
    _save(cards, [*_load(cards), card])

    for scenario in args.scenarios:
        for trial in range(args.trials):
            seed = args.seed + trial
            world = new_world(scenario)
            model = models_for(
                args.strategy,
                world,
                args.base_url,
                args.model,
                seed,
                args.temperature,
            )
            factory = (
                factory_for(
                    args.base_url,
                    args.model,
                    seed,
                    args.temperature,
                )
                if args.strategy == "team"
                else None
            )
            result = run_trial(
                args.strategy, world, model, budget, factory
            )
            record = Record(
                card=card_id,
                strategy=args.strategy,
                scenario=scenario,
                trial=trial,
                seed=seed,
                **result,
            )
            save_record(runs, asdict(record))
            print(
                f"{scenario} t{trial}: "
                f"{'pass' if record.passed else 'fail'} "
                f"({record.check}; {record.ended}) "
                f"{record.model_calls} calls, "
                f"{record.input_tokens}+"
                f"{record.output_tokens} tokens, "
                f"{record.seconds}s",
                flush=True,
            )

    card["finished"] = datetime.now(UTC).isoformat(
        timespec="seconds"
    )
    _save(
        cards,
        [c for c in _load(cards) if c["id"] != card_id]
        + [card],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
