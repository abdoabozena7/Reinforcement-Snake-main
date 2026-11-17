"""Shared, simulation-time arena configuration (one tick is 120 ms)."""

SIMULATION_DT = 0.12
MODEL_VERSION = "arena-3d-v3"
ENVIRONMENT_VERSION = "survival-balanced-v8"
REWARD_VERSION = "survival-objectives-v1"
SHAPING_GAMMA = 0.99

REWARD_CONFIG = {
    "food_value": 6.0,
    "death": -30.0,
    "living": -0.05,
    "risk_cost": -0.08,
    "resource_potential": 6.0,
    "safety_potential": 2.0,
    "race_potential": 1.0,
    "contested_loss": -0.5,
}

ABILITY_CONFIG = {
    "dash_distance": 3,
    "dash_cooldown": 24,
    "shield_duration": 4,
    "shield_cooldown": 60,
}

ESCAPE_CONFIG = {
    "near_distance": 4,
    "separation_distance": 8,
    "separation_steps": 6,
    "cooldown": 60,
    "reward": 0.75,
}

OBSTACLE_CHANGE_CONFIG = {
    "warning_ticks": 16,
    "min_interval": 48,
    "max_interval": 88,
}


def _difficulty(name: str, **changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "step_limit": 320,
        "obstacle_count": 72,
        "patrols": 0,
        "hunters": 0,
        "rivals": 0,
        "ambushers": 0,
        "patrol_interval": 3,
        "hunter_interval": 1,
        "hunter_range": 12,
        "rival_interval": 1,
        "ambush_range": 6,
        "ambush_warning": 3,
        "ambush_burst": 3,
        "ambush_recovery": 10,
        "gates": 0,
        "gate_open_ticks": 24,
        "gate_warning_ticks": 6,
        "gate_closed_ticks": 16,
        "moving_walls": 0,
        "wall_interval": 10,
        "wall_warning_ticks": 3,
        "maze_walls": 0,
        "maze_interval": 45,
        "maze_warning_ticks": 8,
        "shrinking_zone": False,
        "zone_interval": 150,
        "zone_warning_ticks": 30,
        "zone_max_inset": 4,
        "gold": False,
        "gold_interval": 75,
        "gold_ttl": 55,
        "abilities": False,
        "evaluationGoals": {"food": 2, "survivalSteps": 100, "escapes": 0},
        "evaluationMinEpisodes": 8,
        "evaluationThreshold": 0.7,
    }
    result.update(changes)
    return result


DIFFICULTIES = {
    1: _difficulty(
        "Collect & Survive", step_limit=260, obstacle_count=48,
        evaluationGoals={"food": 2, "survivalSteps": 90, "escapes": 0},
    ),
    2: _difficulty(
        "Moving Danger", patrols=1,
        evaluationGoals={"food": 2, "survivalSteps": 110, "escapes": 0},
    ),
    3: _difficulty(
        "The Hunter", step_limit=420, obstacle_count=96, patrols=1, hunters=1, abilities=True,
        evaluationGoals={"food": 2, "survivalSteps": 140, "escapes": 0},
    ),
    4: _difficulty(
        "Risk & Reward", step_limit=480, obstacle_count=128, patrols=1, hunters=1, rivals=1,
        gold=True, abilities=True,
        evaluationGoals={"food": 3, "survivalSteps": 160, "escapes": 0},
    ),
    5: _difficulty(
        "Survival Arena", step_limit=560, obstacle_count=168, patrols=2,
        hunters=1, rivals=1, ambushers=1, gates=2,
        moving_walls=1, maze_walls=1, gold=True, abilities=True,
        evaluationGoals={"food": 3, "survivalSteps": 200, "escapes": 1},
    ),
    6: _difficulty(
        "Master Mode", step_limit=640, obstacle_count=208, patrols=2,
        hunters=2, rivals=1, ambushers=1, gates=2,
        moving_walls=2, maze_walls=1, shrinking_zone=True, gold=True,
        abilities=True,
        evaluationGoals={"food": 4, "survivalSteps": 240, "escapes": 1},
    ),
}
