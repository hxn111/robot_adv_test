CONDITION_DEFINITIONS = {
    1: {
        "condition_id": 1,
        "condition_name": "baseline_deflect_single_robot",
        "strategy_family": "baseline_deflect",
        "interaction_variant": "single_robot",
        "requires_group": False,
    },
    2: {
        "condition_id": 2,
        "condition_name": "humor_deflect_single_robot",
        "strategy_family": "humor_deflect",
        "interaction_variant": "single_robot",
        "requires_group": False,
    },
    3: {
        "condition_id": 3,
        "condition_name": "empathetic_deflect_single_robot",
        "strategy_family": "empathetic_deflect",
        "interaction_variant": "single_robot",
        "requires_group": False,
    },
    4: {
        "condition_id": 4,
        "condition_name": "humor_deflect_group_dynamics",
        "strategy_family": "humor_deflect",
        "interaction_variant": "group_dynamics",
        "requires_group": True,
    },
    5: {
        "condition_id": 5,
        "condition_name": "empathetic_deflect_group_dynamics",
        "strategy_family": "empathetic_deflect",
        "interaction_variant": "group_dynamics",
        "requires_group": True,
    },
    6: {
        "condition_id": 6,
        "condition_name": "humor_deflect_two_robot",
        "strategy_family": "humor_deflect",
        "interaction_variant": "two_robot_interactions",
        "requires_group": False,
    },
    7: {
        "condition_id": 7,
        "condition_name": "empathetic_deflect_two_robot",
        "strategy_family": "empathetic_deflect",
        "interaction_variant": "two_robot_interactions",
        "requires_group": False,
    },
    8: {
        "condition_id": 8,
        "condition_name": "baseline_decline_single_robot",
        "strategy_family": "baseline_decline",
        "interaction_variant": "single_robot",
        "requires_group": False,
    },
}


def get_eligible_condition_ids(group_size):
    is_group_context = isinstance(group_size, int) and group_size >= 2
    eligible = []
    for condition_id in sorted(CONDITION_DEFINITIONS.keys()):
        condition = CONDITION_DEFINITIONS.get(condition_id, {})
        requires_group = bool(condition.get('requires_group', False))
        if requires_group and not is_group_context:
            continue
        eligible.append(condition_id)
    return eligible


def get_condition(condition_id):
    return CONDITION_DEFINITIONS.get(condition_id)
