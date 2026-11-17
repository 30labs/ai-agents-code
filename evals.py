"""Chapter 13's readers: pass@k, pass^k and failure codes.

Every function here is pure. The suite's records come from
`suite.py`, run by hand; these only read them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from statistics import mean
from typing import Any

from opsworld import Scenario, scenarios

Record = dict[str, Any]


def pass_at(n: int, c: int, k: int) -> float | None:
    """Chance that at least one of k trials, drawn from n
    of which c passed, passes. None when k exceeds n."""
    if k > n:
        return None
    return 1 - comb(n - c, k) / comb(n, k)


def pass_hat(n: int, c: int, k: int) -> float | None:
    """Chance that all k trials, drawn from n of which c
    passed, pass (pass^k). None when k exceeds n."""
    if k > n:
        return None
    return comb(c, k) / comb(n, k)


def by_scenario(
    runs: list[Record], strategy: str
) -> dict[str, list[Record]]:
    "One strategy's records, grouped by scenario id."
    found: dict[str, list[Record]] = defaultdict(list)
    for r in runs:
        if r["strategy"] == strategy:
            found[r["scenario"]].append(r)
    return dict(sorted(found.items()))


def suite_rate(
    runs: list[Record], strategy: str, k: int, hat: bool
) -> float | None:
    """pass^k (hat) or pass@k over a strategy's scenarios,
    each estimated from its own trials, then averaged.
    None when any scenario has fewer than k trials."""
    rates = []
    for rs in by_scenario(runs, strategy).values():
        n = len(rs)
        c = sum(r["passed"] for r in rs)
        rate = (pass_hat if hat else pass_at)(n, c, k)
        if rate is None:
            return None
        rates.append(rate)
    return mean(rates) if rates else None


def reference(scenario_id: str) -> str:
    "The REMEDY line a correct answer ends with."
    remedy = scenarios()[scenario_id].remedy
    if remedy is None:
        return "REMEDY: none"
    a = remedy.arguments
    if remedy.tool == "rollback_deploy":
        rest = a["deploy_id"]
    elif remedy.tool == "set_flag":
        rest = f"{a['name']} {'on' if a['enabled'] else 'off'}"
    else:
        rest = f"{a['service']} {a['replicas']}"
    return f"REMEDY: {remedy.tool} {rest}"


def target(scenario: Scenario) -> str:
    "The service the scenario's remedy acts on, or ''."
    remedy = scenario.remedy
    if remedy is None:
        return ""
    if remedy.tool == "rollback_deploy":
        deploy = remedy.arguments["deploy_id"]
        for item in scenario.deploys:
            if item.id == deploy:
                return item.service
    if remedy.tool == "scale_service":
        return str(remedy.arguments["service"])
    return scenario.alert.service


def services_read(record: Record) -> set[str]:
    "Every service a trial's tool calls named."
    found = set()
    for item in record["read"]:
        raw = item[item.find("{") :]
        try:
            args = json.loads(raw)
        except ValueError:
            continue
        if isinstance(args, dict) and "service" in args:
            found.add(str(args["service"]))
    return found


def code(record: Record) -> str:
    "A first failure code for one trial, from its record."
    if record["passed"]:
        return "passed"
    if record["ended"] != "answered":
        return "stopped by a limit"
    if not record["remedy"]:
        return "no usable remedy line"
    scenario = scenarios()[record["scenario"]]
    if scenario.remedy is None:
        return "acted on a false alarm"
    if target(scenario) not in services_read(record):
        return "never read the cause"
    return "read the cause, chose wrong"


def short(item: str) -> str:
    "One tool call or remedy as a line of at most 48."
    name, _, raw = item.partition(" ")
    try:
        args = json.loads(raw)
    except ValueError:
        return item[:48]
    if not isinstance(args, dict):
        return item[:48]
    values = " ".join(str(v) for v in args.values())
    return f"{name} {values}"[:48]


def steps(record: Record) -> list[str]:
    "A trial's tool calls, one short line each."
    return [short(item) for item in record["read"]]
