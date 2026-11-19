"""The remedy checks printed in chapters 7 and 11."""

from dataclasses import dataclass

from opsworld import Remedy, World


@dataclass(frozen=True)
class Verdict:
    ok: bool
    why: str


def onset(world: World) -> str:
    "When the alerting service's errors first rose."
    series = world.metrics[world.alert.service]
    base = series[0].error_rate
    return next((p.time for p in series
                 if p.error_rate > 2 * base), "")


def changed_at(world: World, r: Remedy) -> str:
    "When the change this remedy undoes was made."
    a = r.arguments
    if r.tool == "rollback_deploy":
        return next((d.time for d in world.deploys
                     if d.id == a["deploy_id"]), "")
    flags = world.flags
    if r.tool == "set_flag" and a["name"] in flags:
        return flags[a["name"]].changed_at
    return ""


def before(world: World, r: Remedy) -> Verdict:
    "Could the undone change be the cause?"
    when, start = changed_at(world, r), onset(world)
    if not when:
        return Verdict(True, "no change to date")
    if when > start:
        return Verdict(False, f"{when} is after the"
                       f" errors began at {start}")
    return Verdict(True, f"{when} is not after the"
                   f" errors began at {start}")


def note(r: Remedy, v: Verdict) -> str:
    "The next attempt's note, from a failed check."
    vals = r.arguments.values()
    args = " ".join(str(x) for x in vals)
    return (f"Your remedy {r.tool} {args} failed"
            f" a check: {v.why}. Propose another"
            " remedy, or none.")


def after(
    world: World, limit: float = 0.02,
) -> Verdict:
    "Is the alerting service healthy again?"
    last = world.metrics[world.alert.service][-1]
    rate = (f"errors {last.error_rate:.1%}"
            f" at {last.time}")
    return Verdict(last.error_rate <= limit, rate)



from math import ceil

from opsworld import Remedy, World


def window(world: World) -> tuple[str, str]:
    "First metric sample to the onset of errors."
    series = world.metrics[world.alert.service]
    base = series[0].error_rate
    onset = next((p.time for p in series
                  if p.error_rate > 2 * base), "")
    return series[0].time, onset


def fixed_set(world: World) -> list[Remedy]:
    "Writes trusted state allows, before any log."
    first, onset = window(world)
    name = world.alert.service
    deps = world.services[name].dependencies
    scope = {name, *deps}
    out = [Remedy("rollback_deploy",
                  {"deploy_id": d.id})
           for d in world.deploys
           if d.service in scope
           and first <= d.time <= onset]
    out += [Remedy("set_flag",
                   {"name": f.name,
                    "enabled": not f.enabled})
            for f in world.flags.values()
            if first <= f.changed_at <= onset]
    for s in world.services.values():
        u = s.utilization_percent
        if s.name in scope and u > 60:
            n = ceil(s.replicas * u / 60)
            out.append(Remedy("scale_service", {
                "service": s.name, "replicas": n}))
    return out


def gate(c: Remedy, fixed: list[Remedy],
         tainted: list[str]) -> str:
    "What code does with a write the model chose."
    if c in fixed:
        return "ask"
    if tainted:
        return f"refuse: read {tainted[0]}"
    return "ask, flagged: not in the fixed set"
