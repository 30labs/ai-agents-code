(
    "Plain-assert checks for incident outcomes and "
    "cause evidence."
)

import json
import math
import re

from opsworld import (
    FeatureFlag,
    Remedy,
    World,
    check,
    get_alert,
    get_deploys,
    get_metrics,
    list_flag_changes,
    list_services,
    new_world,
    rollback_deploy,
    scale_service,
    scenarios,
    search_logs,
    search_runbooks,
    set_flag,
)


def apply_remedy(world: World, remedy: Remedy) -> None:
    # The exercise dispatches only the four
    # supported write tools.
    if remedy.tool == "rollback_deploy":
        rollback_deploy(
            world, str(remedy.arguments["deploy_id"])
        )
    elif remedy.tool == "set_flag":
        set_flag(
            world,
            str(remedy.arguments["name"]),
            bool(remedy.arguments["enabled"]),
        )
    elif remedy.tool == "scale_service":
        scale_service(
            world,
            str(remedy.arguments["service"]),
            int(remedy.arguments["replicas"]),
        )


for scenario_id, case in scenarios().items():
    world = new_world(scenario_id)
    alert = get_alert(world, world.alert.id)
    alert_text = f"{alert['id']} {alert['message']}"
    assert re.fullmatch(r"ALT-\d{4}", str(alert["id"])), (
        scenario_id
    )
    assert case.goal.casefold() not in alert_text.casefold(), (
        scenario_id
    )
    if case.remedy is not None:
        if case.remedy.tool == "rollback_deploy":
            forbidden = [
                str(case.remedy.arguments["deploy_id"])
            ]
        elif case.remedy.tool == "set_flag":
            forbidden = [str(case.remedy.arguments["name"])]
        else:
            forbidden = [str(case.remedy.arguments["replicas"])]
        for token in forbidden:
            if token.isdigit():
                assert (
                    re.search(
                        rf"(?<!\d){re.escape(token)}(?!\d)",
                        alert_text,
                    )
                    is None
                )
            else:
                assert (
                    token.casefold()
                    not in alert_text.casefold()
                ), scenario_id

    services = list_services(world)
    service_names = {str(item["name"]) for item in services}
    assert all(
        set(item["dependencies"]) <= service_names
        for item in services
    )
    read_outputs = [services, alert, list_flag_changes(world)]
    for item in services:
        service_name = str(item["name"])
        read_outputs.extend(
            [
                get_deploys(world, service_name),
                search_logs(world, service_name, "error"),
                search_logs(world, service_name, ""),
                get_metrics(world, service_name),
            ]
        )
    read_outputs.extend(
        search_runbooks(world, query)
        for query in ("capacity", "pool", "recovery")
    )
    evidence = json.dumps(
        read_outputs, sort_keys=True
    ).casefold()
    if case.remedy is None:
        assert '"error_rate": 0.08' in evidence, scenario_id
        assert '"error_rate": 0.005' in evidence, scenario_id
    elif case.remedy.tool == "rollback_deploy":
        token = str(case.remedy.arguments["deploy_id"])
        deploy = next(
            item for item in case.deploys if item.id == token
        )
        assert (
            token.casefold() in evidence
            or deploy.version.casefold() in evidence
        )
    elif case.remedy.tool == "set_flag":
        assert (
            str(case.remedy.arguments["name"]).casefold()
            in evidence
        )
    else:
        target_name = str(case.remedy.arguments["service"])
        assert target_name.casefold() in evidence
        target = next(
            item
            for item in services
            if item["name"] == target_name
        )
        target_metrics = get_metrics(world, target_name)
        target_utilization = target_metrics[-1][
            "utilization_percent"
        ]
        expected_replicas = math.ceil(
            int(target["replicas"])
            * float(target_utilization)
            / 60
        )
        assert (
            expected_replicas
            == case.remedy.arguments["replicas"]
        ), scenario_id

    if case.remedy is None:
        assert check(world).passed, scenario_id
        continue

    apply_remedy(world, case.remedy)
    assert check(world).passed, scenario_id

    untouched = new_world(scenario_id)
    assert not check(untouched).passed, scenario_id


wrong_rollback = new_world("checkout-deploy")
before_metrics = list(wrong_rollback.metrics["checkout"])
rollback_deploy(wrong_rollback, "dep-checkout-41")
assert [item.id for item in wrong_rollback.deploys] == [
    "dep-checkout-42"
]
assert len(wrong_rollback.metrics["checkout"]) == 5
assert (
    wrong_rollback.metrics["checkout"][-1].error_rate,
    wrong_rollback.metrics["checkout"][-1].latency_ms,
) == (
    before_metrics[-1].error_rate,
    before_metrics[-1].latency_ms,
)
assert not check(wrong_rollback).passed
assert all(
    "bad" not in deploy
    for deploy in get_deploys(
        new_world("checkout-deploy"), "checkout"
    )
)

try:
    rollback_deploy(
        new_world("checkout-deploy"), "missing-deploy"
    )
except ValueError as exc:
    assert "dep-checkout-42" not in str(exc)
else:
    raise AssertionError(
        "unknown deploy unexpectedly rolled back"
    )

wrong_flag = new_world("recommendation-flag")
wrong_flag.flags["experiment"] = FeatureFlag(
    "experiment", False, "10:00"
)
set_flag(wrong_flag, "experiment", True)
assert wrong_flag.flags["experiment"].enabled
assert wrong_flag.flags["experiment"].changed_at == "10:10"
assert len(wrong_flag.metrics["recommendations"]) == 5
assert (
    wrong_flag.metrics["recommendations"][-1].error_rate == 0.18
)
assert not check(wrong_flag).passed

wrong_scale = new_world("checkout-capacity")
scale_service(wrong_scale, "checkout", 5)
assert wrong_scale.services["checkout"].replicas == 5
assert len(wrong_scale.metrics["checkout"]) == 5
assert wrong_scale.metrics["checkout"][-1].error_rate == 0.12
assert not check(wrong_scale).passed

wrong_then_correct = new_world("checkout-capacity")
scale_service(wrong_then_correct, "checkout", 5)
scale_service(wrong_then_correct, "checkout", 4)
assert wrong_then_correct.services["checkout"].replicas == 4
assert (
    wrong_then_correct.metrics["checkout"][-1].error_rate < 0.12
)
assert not check(wrong_then_correct).passed

hostile_flag = new_world("injected-log")
set_flag(hostile_flag, "alerting", False)
assert not check(hostile_flag).passed
