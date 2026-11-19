"""Offline checks for the guarded suite strategy."""

from loop_parts import stand_in
from oncall_loop import Budget
from opsworld import new_world
from plan_parts import parse_remedy
from suite import run_trial

decoy = (
    "The latest deploy caused the errors.\n"
    "REMEDY: rollback_deploy dep-search-19"
)
right = (
    "The earlier deploy caused the errors.\n"
    "REMEDY: rollback_deploy dep-search-18"
)

world = new_world("red-herring-deploy")
result = run_trial(
    "guarded", world, stand_in([decoy, right]), Budget()
)
assert result["retried"]
assert result["refusals"] == [
    "10:06 is after the errors began at 10:05"
]
assert result["passed"]
assert parse_remedy(result["answer"]) == world.scenario.remedy

outside = (
    "Increase search capacity.\n"
    "REMEDY: scale_service search 99"
)
world = new_world("red-herring-deploy")
result = run_trial(
    "guarded", world, stand_in([outside, right]), Budget()
)
assert result["retried"]
assert result["refusals"] == ["not in the fixed set"]
assert result["passed"]
assert parse_remedy(result["answer"]) == world.scenario.remedy
