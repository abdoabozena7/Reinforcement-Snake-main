from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import json
import heapq
import math
import random
import unittest
from unittest.mock import patch

from backend.config import (
    ABILITY_CONFIG, DIFFICULTIES, ESCAPE_CONFIG, OBSTACLE_CHANGE_CONFIG,
    REWARD_CONFIG, REWARD_VERSION, SHAPING_GAMMA,
)
from backend.environment import (
    ACTION_SIZE, DIRECTIONS, OBSERVATION_LABELS, OBSERVATION_SIZE,
    Enemy, Food, Gate, MovingWall, ObstacleChange, Point, SnakeEnvironment,
)


def p(x: int, z: int, y: int = 11) -> Point:
    return Point(x, y, z)


class EnvironmentTest(unittest.TestCase):
    def arena(self, **kwargs: object) -> SnakeEnvironment:
        options = dict(seed=17, level=6, grid_size=22, obstacle_count=0, step_limit=1000)
        options.update(kwargs)
        env = SnakeEnvironment(**options)
        env.enemies.clear()
        env.gates.clear()
        env.moving_walls.clear()
        env.obstacles.clear()
        env.foods = [Food("normal-test", "normal", p(19, 19))]
        env.config["gold"] = False
        env.config["shrinking_zone"] = False
        env.events.clear()
        return env

    def place_player(
        self, env: SnakeEnvironment, cells: list[Point], direction: Point = DIRECTIONS[0],
    ) -> None:
        env.snake = cells
        env.direction = direction
        env._recent = deque([cells[0]], maxlen=24)

    def cycle(self, env: SnakeEnvironment, count: int) -> list[dict[str, object]]:
        return [env.step((1, 0, 3, 2)[i % 4]) for i in range(count)]

    def features(self, env: SnakeEnvironment) -> dict[str, float]:
        values = env.get_state()
        self.assertEqual(len(values), OBSERVATION_SIZE)
        self.assertTrue(all(math.isfinite(v) and -1 <= v <= 1 for v in values))
        return dict(zip(OBSERVATION_LABELS, values))

    def test_seeded_six_levels_and_full_mode_spawn_connected_and_disjoint(self) -> None:
        for seed in range(8):
            for level in DIFFICULTIES:
                with self.subTest(seed=seed, level=level):
                    a = SnakeEnvironment(seed=seed, level=level)
                    b = SnakeEnvironment(seed=seed, level=level)
                    self.assertEqual(a.board(), b.board())
                    self.assertEqual(a.get_state(), b.get_state())
                    all_body = a.snake + [part for enemy in a.enemies for part in enemy.snake]
                    self.assertEqual(len(all_body), len(set(all_body)))
                    self.assertTrue(all(a._inside(part) for part in all_body))
                    self.assertFalse(set(all_body) & a._solid_cells())
                    self.assertFalse(set(all_body) & {part for wall in a.moving_walls for part in wall.next_cells})
                    self.assertTrue(a._floor_connected(a._solid_cells()))
                    for food in a.foods:
                        self.assertNotIn(food.position, set(all_body) | a._reserved_paths() | a._solid_cells())
                    self.features(a)
                    json.dumps(a.board(), allow_nan=False)
        full = SnakeEnvironment(seed=7, full_survival=True)
        self.assertEqual(full.level, 6)
        self.assertEqual({e.type for e in full.enemies}, {"hunter", "patrol", "rival", "ambush"})
        self.assertTrue(full.gates)
        self.assertEqual({w.type for w in full.moving_walls}, {"moving", "maze"})
        self.assertTrue(full.config["shrinking_zone"])
        self.assertTrue(any(f.type == "gold" for f in full.foods))

    def test_local_rng_observation_and_board_are_read_only(self) -> None:
        random.seed(771)
        global_before = random.getstate()
        env = SnakeEnvironment(seed=6, full_survival=True)
        before = env.rng.getstate(), deepcopy(env.board())
        for _ in range(4):
            self.features(env)
            env.board()
        self.assertEqual(before, (env.rng.getstate(), env.board()))
        self.assertEqual(global_before, random.getstate())

    def test_identical_seed_and_actions_reproduce_all_transition_data(self) -> None:
        rng = random.Random(901)
        for level in (1, 3, 6):
            a, b = SnakeEnvironment(seed=39, level=level), SnakeEnvironment(seed=39, level=level)
            for _ in range(120):
                if a.done:
                    self.assertTrue(b.done)
                    seed = rng.randrange(100000)
                    a.reset(seed=seed)
                    b.reset(seed=seed)
                action = rng.choice([i for i, legal in enumerate(a.action_mask()) if legal])
                self.assertEqual(a.step(action), b.step(action))
                self.assertEqual(a.board(), b.board())
            expected = a.reset(seed=39, level=level, full_survival=False)
            self.assertEqual(expected, SnakeEnvironment(seed=39, level=level).get_state())

    def test_volumetric_adaptive_grid_and_explicit_configuration_errors(self) -> None:
        for size in (8, 12, 32):
            env = SnakeEnvironment(grid_size=size, seed=3, level=1, obstacle_count=3)
            self.assertEqual(env.board()["gridSize"], size)
            self.assertEqual(env.snake[0], Point(size // 2, size // 2, size // 2))
            self.assertTrue(all(env._inside(part) for part in env.snake + env.obstacles))
            self.assertTrue(any(part.y != size // 2 for part in env.obstacles))
            self.features(env)
        for kwargs in ({"grid_size": 7}, {"grid_size": 65}, {"level": 0}, {"step_limit": 0}):
            with self.assertRaises(ValueError):
                SnakeEnvironment(**kwargs)

    def test_six_by_three_mapping_reverse_and_unavailable_abilities(self) -> None:
        for ability in range(3):
            for direction, delta in enumerate(DIRECTIONS):
                with self.subTest(ability=ability, direction=direction):
                    env = self.arena()
                    result = env.step(ability * len(DIRECTIONS) + direction)
                    if direction == 2:
                        self.assertFalse(env.action_mask()[2] if env.direction == DIRECTIONS[0] else True)
                        self.assertEqual(env.snake[0], p(11, 10))
                        self.assertTrue(result["info"]["invalidAction"])
                        self.assertIsNone(result["info"]["abilityUsed"])
                    else:
                        distance = 3 if ability == 1 else 1
                        self.assertEqual(env.snake[0], Point(
                            11 + delta.x * distance, 11 + delta.y * distance, 11 + delta.z * distance))
                        self.assertFalse(result["done"])
        env = SnakeEnvironment(level=1, obstacle_count=0)
        self.assertTrue(all(not v for v in env.action_mask()[6:]))
        initial = env.snake[0]
        result = env.step(7)
        self.assertEqual(env.snake[0], Point(initial.x + 1, initial.y, initial.z))
        self.assertIsNone(result["info"]["abilityUsed"])
        self.assertTrue(result["info"]["invalidAction"])

    def test_mask_is_not_a_safety_controller(self) -> None:
        env = self.arena()
        env.obstacles = [p(12, 11)]
        self.assertTrue(env.action_mask()[1])
        self.assertTrue(env.action_mask()[7])
        result = env.step(1)
        self.assertTrue(result["done"])
        self.assertEqual(env.last_death, "obstacle")

    def test_invalid_action_is_rejected_without_advancing(self) -> None:
        env = self.arena()
        before = env.board()
        for action in (-1, ACTION_SIZE, 1.5, None, True):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.board(), before)

    def test_tail_vacates_but_neck_and_growth_do_not(self) -> None:
        env = self.arena()
        self.place_player(env, [p(4, 4), p(4, 5), p(3, 5), p(3, 4)])
        self.assertFalse(env.step(3)["done"])
        self.assertEqual(env.snake[0], p(3, 4))
        env = self.arena()
        self.place_player(env, [p(4, 4), p(4, 5), p(3, 5), p(3, 4)])
        env.food = p(3, 4)
        self.assertTrue(env.step(3)["done"])
        self.assertEqual(env.last_death, "self")

    def test_terminal_is_idempotent_including_metrics_events_and_rng(self) -> None:
        env = self.arena()
        env.obstacles = [p(11, 10)]
        first = env.step(0)
        self.assertTrue(first["done"])
        board, rng = deepcopy(env.board()), env.rng.getstate()
        for action in (0, 5, 11):
            again = env.step(action)
            self.assertEqual(again["reward"], 0)
            self.assertTrue(again["info"]["terminalAlreadyProcessed"])
            self.assertEqual(env.board(), board)
            self.assertEqual(env.rng.getstate(), rng)
        self.assertEqual(sum(event["type"] == "death" for event in board["events"]), 1)

    def test_time_limit_is_exact_and_shaping_has_zero_terminal_potential(self) -> None:
        env = self.arena()
        env.step_limit = 2
        self.assertFalse(env.step(1)["done"])
        terminal = env.step(0)
        self.assertTrue(terminal["done"])
        self.assertEqual(env.steps, 2)
        self.assertEqual(env.last_death, "time_limit")
        self.assertTrue(terminal["info"]["truncated"])
        self.assertEqual(terminal["info"]["potentialAfter"], 0)
        self.assertEqual(terminal["info"]["rewardComponents"]["death"], 0)
        self.assertIsNone(env.board()["contact"])
        self.assertEqual(terminal["info"]["events"][-1]["type"], "episode_end")

    def test_timeout_after_a_real_shield_block_has_no_terminal_physical_contact(self) -> None:
        env = self.arena(step_limit=1)
        env.enemies = [Enemy("hunter-test", "hunter", [p(12, 11)], DIRECTIONS[3],
                             next_move=100)]
        result = env.step(13)
        self.assertTrue(result["done"])
        self.assertEqual(env.last_death, "time_limit")
        self.assertIsNone(env.board()["contact"])
        self.assertNotEqual(env.board()["emotion"], "dead")
        blocked = next(event for event in result["info"]["events"] if event["type"] == "shield_block")
        self.assertEqual(blocked["contact"]["outcome"], "blocked")
        self.assertFalse(any(event["type"] == "death" for event in result["info"]["events"]))
        self.assertEqual(result["info"]["rewardComponents"]["death"], 0)

    def test_dash_checks_intermediate_obstacle_and_awards_food_once(self) -> None:
        env = self.arena()
        env.food = p(12, 11)
        env.obstacles = [p(13, 11)]
        result = env.step(7)
        self.assertTrue(result["done"])
        self.assertEqual(env.snake[0], p(12, 11))
        self.assertEqual(env.death_point, p(13, 11))
        self.assertEqual((env.score, env.food_collected, env.steps), (1, 1, 1))
        self.assertEqual(result["info"]["rewardComponents"]["food"], 6)
        self.assertEqual([x["type"] for x in result["info"]["events"]], ["dash", "food", "death"])
        contact = env.board()["contact"]
        self.assertEqual((contact["kind"], contact["substep"]), ("obstacle", 2))
        self.assertEqual(contact["victim"]["from"], {"x": 12, "y": 11, "z": 11})
        self.assertEqual(contact["victim"]["to"], {"x": 13, "y": 11, "z": 11})
        self.assertEqual(contact["position"], {"x": 12.5, "y": 11, "z": 11})
        self.assertEqual(contact["timeFraction"], 0.5)
        self.assertEqual(result["info"]["events"][-1]["contact"], contact)
        env.step(7)
        self.assertEqual(env.food_collected, 1)

    def test_dash_cannot_skip_intermediate_enemy(self) -> None:
        env = self.arena()
        env.enemies = [Enemy("hunter-test", "hunter", [p(13, 11)], DIRECTIONS[3],
                             next_move=100, detection_range=12)]
        self.assertTrue(env.step(7)["done"])
        self.assertEqual(env.last_death, "hunter")
        self.assertEqual(env.snake[0], p(12, 11))
        contact = env.board()["contact"]
        self.assertEqual((contact["kind"], contact["substep"]), ("head-on", 2))
        self.assertEqual(contact["attacker"]["from"], contact["attacker"]["to"])
        self.assertEqual(contact["victim"]["from"], {"x": 12, "y": 11, "z": 11})

    def test_dash_cooldown_downgrades_unavailable_action_without_hidden_move(self) -> None:
        env = self.arena()
        env.step(7)
        self.assertFalse(env.action_mask()[6])
        self.assertEqual(env.dash_cooldown, ABILITY_CONFIG["dash_cooldown"] - 1)
        result = env.step(6)
        self.assertEqual(env.snake[0], p(14, 10))
        self.assertIsNone(result["info"]["abilityUsed"])
        self.assertTrue(result["info"]["invalidAction"])
        for i in range(22):
            self.assertFalse(env.step((3, 2, 1, 0)[i % 4])["done"])
        self.assertEqual(env.dash_cooldown, 0)
        self.assertTrue(env.action_mask()[7])

    def test_shield_absorbs_one_enemy_contact_and_never_teleports(self) -> None:
        env = self.arena()
        env.enemies = [Enemy("hunter-test", "hunter", [p(12, 11)], DIRECTIONS[3],
                             next_move=100, detection_range=12)]
        first = env.step(13)
        self.assertFalse(first["done"])
        self.assertEqual(env.snake[0], p(11, 11))
        self.assertEqual(env.enemies[0].snake[0], p(12, 11))
        self.assertEqual(env.shield_charge, 0)
        self.assertFalse(env.board()["abilities"]["shield"]["active"])
        self.assertTrue(any(e["type"] == "shield_block" for e in first["info"]["events"]))
        self.assertTrue(env.step(1)["done"])
        self.assertEqual(env.last_death, "hunter")

    def test_shield_blocks_laser_but_not_solid_wall_self_or_zone(self) -> None:
        env = self.arena()
        env.gates = [Gate("g", [p(12, 11)], 2, 2, 20, phase=5, state="closed")]
        self.assertFalse(env.step(13)["done"])
        self.assertEqual(env.snake[0], p(11, 11))
        self.assertTrue(env.step(1)["done"])
        self.assertEqual(env.last_death, "laser")
        for kind in ("obstacle", "moving_wall", "maze_wall", "wall", "zone", "self"):
            with self.subTest(kind=kind):
                env = self.arena()
                if kind == "obstacle":
                    env.obstacles = [p(12, 11)]
                elif kind.endswith("wall") and kind != "wall":
                    env.moving_walls = [MovingWall("w", "maze" if kind == "maze_wall" else "moving",
                                                   [[p(12, 11)], [p(13, 11)]], 10, 3, 10)]
                elif kind == "wall":
                    self.place_player(env, [p(21, 11)], DIRECTIONS[0])
                elif kind == "zone":
                    env.zone_inset = 1
                    self.place_player(env, [p(20, 11)], DIRECTIONS[0])
                else:
                    self.place_player(env, [p(11, 11), p(11, 12), p(12, 12), p(12, 11), p(13, 11)])
                self.assertTrue(env.step(13)["done"])
                self.assertEqual(env.last_death, kind)

    def test_shield_expires_without_contact_and_has_real_cooldown(self) -> None:
        env = self.arena()
        env.step(13)
        for action in (0, 3, 2):
            self.assertFalse(env.step(action)["done"])
        self.assertEqual(env.shield_ticks, 0)
        self.assertEqual(env.shield_charge, 0)
        self.assertFalse(any(env.action_mask()[12:]))
        self.assertEqual(env.shield_cooldown, ABILITY_CONFIG["shield_cooldown"] - 4)

    def test_patrol_moves_on_schedule_and_future_occupancy_predicts_contact(self) -> None:
        env = self.arena()
        route = [p(13, 11), p(12, 11), p(12, 12), p(13, 12)]
        enemy = Enemy("patrol-test", "patrol", [route[0]], DIRECTIONS[3],
                      interval=2, next_move=1, route=route)
        env.enemies = [enemy]
        features = self.features(env)
        self.assertEqual(features["Current danger E"], 0)
        self.assertEqual(features["Next tick danger E"], 1)
        self.assertTrue(env.step(1)["done"])
        self.assertEqual(env.last_death, "patrol")
        env = self.arena()
        route = [p(3, 3), p(4, 3), p(4, 4), p(3, 4)]
        enemy = Enemy("patrol-test", "patrol", [route[0]], DIRECTIONS[1],
                      interval=2, next_move=2, route=route)
        env.enemies = [enemy]
        for i in range(8):
            self.assertFalse(env.step((1, 0, 3, 2)[i % 4])["done"])
            self.assertEqual(enemy.snake[0], route[((i + 1) // 2) % 4])

    def test_opposing_crossing_is_lethal_even_when_both_old_heads_vacate(self) -> None:
        env = self.arena()
        self.place_player(env, [p(11, 11)])
        route = [p(12, 11), p(11, 11), p(11, 12), p(12, 12)]
        env.enemies = [Enemy("patrol-test", "patrol", [route[0]], DIRECTIONS[3],
                             interval=1, next_move=1, route=route)]
        self.assertTrue(env.step(1)["done"])
        self.assertEqual(env.last_death, "patrol")

    def test_head_contacts_record_real_paths_and_never_commit_terminal_intent(self) -> None:
        for action in (1, 4):
            delta = DIRECTIONS[action]
            head = Point(11, 11, 11)
            for kind, distance, fraction in (("crossing", 1, 0.5), ("head-on", 2, 1.0)):
                with self.subTest(action=action, kind=kind):
                    env = self.arena()
                    self.place_player(env, [head])
                    start = Point(head.x + distance * delta.x, head.y + distance * delta.y,
                                  head.z + distance * delta.z)
                    enemy = Enemy("actual-hunter", "hunter", [start], DIRECTIONS[3],
                                  interval=1, next_move=1, detection_range=12)
                    env.enemies = [enemy]
                    before = deepcopy(env.board())
                    result = env.step(action)
                    board = env.board()
                    contact = board["contact"]
                    self.assertTrue(result["done"])
                    self.assertEqual((contact["schemaVersion"], contact["kind"], contact["cause"]),
                                     (1, kind, "hunter"))
                    self.assertEqual((contact["outcome"], contact["step"], contact["substep"]),
                                     ("death", 1, 1))
                    self.assertEqual(contact["sourceId"], enemy.id)
                    self.assertEqual(contact["timeFraction"], fraction)
                    self.assertEqual(contact["position"], {
                        "x": head.x + delta.x * fraction, "y": head.y + delta.y * fraction,
                        "z": head.z + delta.z * fraction,
                    })
                    self.assertEqual(contact["victim"]["id"], "player")
                    self.assertEqual(contact["victim"]["segmentIndex"], 0)
                    self.assertEqual(contact["victim"]["before"], before["snake"])
                    self.assertEqual(contact["attacker"]["before"], before["enemies"][0]["snake"])
                    self.assertEqual(contact["victim"]["attempted"][0], contact["victim"]["to"])
                    self.assertEqual(contact["attacker"]["attempted"][0], contact["attacker"]["to"])
                    self.assertEqual(board["snake"], before["snake"])
                    self.assertEqual(board["enemies"][0]["snake"], before["enemies"][0]["snake"])
                    self.assertEqual(result["info"]["events"][-1]["contact"], contact)
                    frozen = json.dumps(board, sort_keys=True, allow_nan=False)
                    board["contact"]["victim"]["before"][0]["x"] = -999
                    board["events"][-1]["contact"]["attacker"]["to"]["y"] = -999
                    env.get_state()
                    env.step(action)
                    self.assertEqual(json.dumps(env.board(), sort_keys=True, allow_nan=False), frozen)

    def test_body_strike_and_enemy_body_keep_distinct_contacted_segments(self) -> None:
        env = self.arena()
        self.place_player(env, [p(11, 11), p(11, 12), p(12, 12), p(13, 12)])
        route = [p(10, 12), p(11, 12), p(11, 13), p(10, 13)]
        env.enemies = [Enemy("body-attacker", "patrol", [route[0]], DIRECTIONS[1],
                             interval=1, next_move=1, route=route)]
        before = deepcopy(env.board())
        self.assertTrue(env.step(1)["done"])
        contact = env.board()["contact"]
        self.assertEqual((contact["kind"], contact["victim"]["segmentIndex"]), ("body-strike", 2))
        self.assertEqual(contact["position"], {"x": 11, "y": 11, "z": 12})
        self.assertEqual(contact["victim"]["attempted"][2], contact["position"])
        self.assertEqual(contact["attacker"]["to"], contact["position"])
        self.assertEqual(env.board()["snake"], before["snake"])
        env = self.arena()
        env.enemies = [Enemy("stationary-body", "hunter", [p(14, 11), p(13, 11), p(12, 11)],
                             DIRECTIONS[1], next_move=100)]
        self.assertTrue(env.step(1)["done"])
        contact = env.board()["contact"]
        self.assertEqual((contact["kind"], contact["victim"]["segmentIndex"]), ("enemy-body", 0))
        self.assertEqual(contact["attacker"]["segmentIndex"], 2)
        self.assertEqual(contact["attacker"]["from"], contact["attacker"]["to"])
        self.assertEqual(contact["position"], contact["attacker"]["attempted"][2])

    def test_shield_contact_retains_uncancelled_intent_then_clears_on_next_tick(self) -> None:
        env = self.arena()
        env.enemies = [Enemy("shield-hunter", "hunter", [p(12, 11)], DIRECTIONS[3],
                             interval=1, next_move=1, detection_range=12)]
        before = deepcopy(env.board())
        result = env.step(13)
        contact = env.board()["contact"]
        self.assertFalse(result["done"])
        self.assertEqual((contact["kind"], contact["outcome"]), ("crossing", "blocked"))
        self.assertEqual(contact["position"], {"x": 11.5, "y": 11, "z": 11})
        self.assertEqual(contact["attacker"]["to"], before["snake"][0])
        self.assertEqual(contact["victim"]["to"], before["enemies"][0]["snake"][0])
        self.assertEqual(env.board()["snake"], before["snake"])
        self.assertEqual(env.board()["enemies"][0]["snake"], before["enemies"][0]["snake"])
        self.assertEqual(result["info"]["events"][-1]["contact"], contact)
        self.assertEqual(result["info"]["events"][-1]["type"], "shield_block")
        self.assertFalse(env.step(0)["done"])
        self.assertIsNone(env.board()["contact"])
        self.assertEqual(sum(event["type"] == "shield_block" for event in env.events), 1)

    def test_geometry_contacts_identify_actual_hazard_and_boundary(self) -> None:
        for cause in ("obstacle", "moving_wall", "maze_wall", "laser", "wall", "zone", "self"):
            with self.subTest(cause=cause):
                env = self.arena()
                source = None
                if cause == "obstacle":
                    env.obstacles = [p(12, 11)]
                elif cause in ("moving_wall", "maze_wall"):
                    source = "moving-source" if cause == "moving_wall" else "maze-source"
                    env.moving_walls = [MovingWall(
                        source, "moving" if cause == "moving_wall" else "maze",
                        [[p(12, 11)], [p(13, 11)]], 10, 3, 10,
                    )]
                elif cause == "laser":
                    source = "actual-gate"
                    env.gates = [Gate(source, [p(12, 11)], 2, 2, 20, phase=5, state="closed")]
                elif cause == "wall":
                    self.place_player(env, [p(21, 11)])
                elif cause == "zone":
                    env.zone_inset = 1
                    self.place_player(env, [p(20, 11)])
                else:
                    self.place_player(env, [p(11, 11), p(11, 12), p(12, 12), p(12, 11), p(13, 11)])
                before = deepcopy(env.board()["snake"])
                self.assertTrue(env.step(1)["done"])
                contact = env.board()["contact"]
                self.assertEqual(contact["cause"], cause)
                self.assertEqual(contact["kind"], "obstacle" if cause.endswith("_wall") else cause)
                self.assertEqual(contact["victim"]["before"], before)
                self.assertEqual(contact["victim"]["segmentIndex"], 0)
                self.assertEqual(contact["outcome"], "death")
                if cause != "self":
                    self.assertIsNone(contact["attacker"])
                    self.assertEqual(contact["timeFraction"], 0.5)
                    self.assertEqual(contact["position"]["x"], before[0]["x"] + 0.5)
                if source is not None:
                    self.assertEqual(contact["sourceId"], source)
                self.assertEqual(env.board()["snake"], before)
                self.assertEqual(env.board()["events"][-1]["contact"], contact)

    def test_enemy_attack_of_body_and_shield_cancellation_preserve_disjointness(self) -> None:
        env = self.arena()
        route = [p(10, 11), p(11, 11), p(11, 10), p(10, 10)]
        env.enemies = [Enemy("patrol-test", "patrol", [route[0]], DIRECTIONS[1],
                             interval=1, next_move=1, route=route)]
        self.assertTrue(env.step(1)["done"])
        self.assertEqual(env.last_death, "patrol")
        env = self.arena()
        env.enemies = [
            Enemy("contact", "hunter", [p(12, 11)], DIRECTIONS[3], next_move=100),
            Enemy("tail", "patrol", [p(10, 13)], DIRECTIONS[1], interval=1, next_move=1,
                  route=[p(10, 13), p(11, 13), p(11, 14), p(10, 14)]),
        ]
        self.assertFalse(env.step(13)["done"])
        occupied = env.snake + [part for e in env.enemies for part in e.snake]
        self.assertEqual(len(occupied), len(set(occupied)))

    def test_enemy_equal_destination_does_not_overlap(self) -> None:
        env = self.arena()
        first = [p(3, 3), p(4, 3), p(4, 4), p(3, 4)]
        second = [p(5, 3), p(4, 3), p(4, 2), p(5, 2)]
        env.enemies = [
            Enemy("p1", "patrol", [first[0]], DIRECTIONS[1], interval=1, next_move=1, route=first),
            Enemy("p2", "patrol", [second[0]], DIRECTIONS[3], interval=1, next_move=1, route=second),
        ]
        self.assertFalse(env.step(1)["done"])
        self.assertEqual([e.snake[0] for e in env.enemies], [p(3, 3), p(5, 3)])

    def test_hunter_navigates_barrier_cardinally_and_intercepts_previous_heading(self) -> None:
        env = self.arena()
        env.obstacles = [p(8, z) for z in range(5, 17)]
        hunter = Enemy("hunter-test", "hunter", [p(7, 11), p(6, 11), p(5, 11)],
                       DIRECTIONS[1], interval=1, next_move=1, detection_range=20)
        env.enemies = [hunter]
        visited = [hunter.snake[0]]
        for tick in range(1, 24):
            plans = env._plan_enemies(now=tick)
            next_head = plans[hunter.id]
            self.assertEqual(abs(next_head.x - visited[-1].x) + abs(next_head.y - visited[-1].y)
                             + abs(next_head.z - visited[-1].z), 1)
            self.assertNotIn(next_head, env.obstacles)
            env.steps = tick
            env._commit_enemies(plans, tick)
            visited.append(next_head)
            if next_head.x > 8:
                break
        self.assertGreater(hunter.snake[0].x, 8)
        self.assertTrue(any(part.y != 11 or part.z in (4, 17) for part in visited))
        env = self.arena()
        env.enemies = [Enemy("h", "hunter", [p(5, 11), p(4, 11), p(3, 11)],
                             DIRECTIONS[1], interval=1, next_move=1, detection_range=12)]
        other = deepcopy(env)
        env.step(1)
        other.step(3)
        self.assertEqual(env.enemies[0].snake[0], p(5, 10))
        self.assertEqual(env.enemies[0].snake, other.enemies[0].snake)

    def test_hunter_is_slower_than_player_and_cannot_detect_outside_range(self) -> None:
        env = self.arena()
        hunter = Enemy("h", "hunter", [p(3, 11), p(2, 11), p(1, 11)], DIRECTIONS[1],
                       interval=4, next_move=4, detection_range=4)
        env.enemies = [hunter]
        env._refresh_alerts()
        self.assertFalse(hunter.alert)
        for action in (1, 0, 3):
            env.step(action)
            self.assertEqual(hunter.snake[0], p(3, 11))
        env.step(2)
        self.assertEqual(hunter.snake[0], p(4, 11))
        self.assertFalse(hunter.alert)

    def test_rival_collects_same_resource_grows_and_prioritizes_value_per_distance(self) -> None:
        env = self.arena()
        rival = Enemy("rival-test", "rival", [p(4, 4), p(3, 4), p(2, 4)],
                      DIRECTIONS[1], interval=2, next_move=1)
        env.enemies = [rival]
        env.food = p(5, 4)
        result = env.step(0)
        self.assertFalse(result["done"])
        self.assertEqual((rival.snake[0], len(rival.snake), rival.resources), (p(5, 4), 4, 1))
        self.assertEqual((env.score, env.food_collected), (0, 0))
        self.assertNotEqual(env.food, p(5, 4))
        self.assertTrue(any(e["type"] == "rival_food" for e in result["info"]["events"]))
        env = self.arena()
        env.enemies = [Enemy("r", "rival", [p(4, 4), p(3, 4), p(2, 4)],
                             DIRECTIONS[1], interval=1, next_move=1)]
        env.foods = [Food("normal", "normal", p(5, 4)),
                     Food("gold", "gold", p(4, 2), 3, expires_at=50)]
        env.step(0)
        self.assertEqual(env.enemies[0].snake[0], p(4, 3))

    def test_ambush_warns_waits_bursts_and_recovers(self) -> None:
        env = self.arena()
        ambush = Enemy("a", "ambush", [p(8, 11), p(7, 11), p(6, 11)], DIRECTIONS[1],
                       interval=1, next_move=1, detection_range=6, mode="waiting")
        env.enemies = [ambush]
        env.step(1)
        self.assertEqual(ambush.mode, "warning")
        self.assertEqual(ambush.snake[0], p(8, 11))
        self.assertEqual(ambush.next_move, 4)
        env.step(0)
        env.step(3)
        self.assertEqual(ambush.snake[0], p(8, 11))
        for action in (0, 1, 1):
            self.assertFalse(env.step(action)["done"])
        self.assertEqual(ambush.mode, "recover")
        recovered_head = ambush.snake[0]
        self.place_player(env, [p(17, 4), p(17, 5), p(17, 6)])
        for i in range(int(env.config["ambush_recovery"])):
            self.assertFalse(env.step((1, 0, 3, 2)[i % 4])["done"])
            self.assertEqual(ambush.snake[0], recovered_head)
        self.assertEqual(ambush.mode, "waiting")

    def test_laser_schedule_warns_before_closing_and_reopens(self) -> None:
        env = self.arena()
        gate = Gate("g", [p(4, 4)], open_ticks=2, warning_ticks=2, closed_ticks=3)
        env.gates = [gate]
        actual = []
        for action in (1, 0, 3, 2, 1, 0, 3):
            self.assertFalse(env.step(action)["done"])
            actual.append(gate.state)
        self.assertEqual(actual, ["open", "warning", "warning", "closed", "closed", "closed", "open"])
        warning = [e for e in env.events if e["type"] == "gate_warning"]
        closed = [e for e in env.events if e["type"] == "gate_closed"]
        self.assertEqual(closed[0]["step"] - warning[0]["step"], 2)

    def test_public_active_gate_agrees_with_collision_and_policy_observation(self) -> None:
        env = self.arena()
        gate = Gate("g", [p(12, 11)], 2, 2, 20, phase=5, state="closed")
        env.gates = [gate]
        self.assertEqual(env.board()["gates"][0]["state"], "active")
        self.assertEqual(gate.state, "closed")
        features = self.features(env)
        self.assertEqual(features["Gate closed"], 1)
        self.assertEqual(features["Current danger E"], 1)
        self.assertTrue(env.step(1)["done"])
        self.assertEqual(env.last_death, "laser")
        self.assertEqual(env.board()["gates"][0]["state"], "active")

    def test_gate_defers_for_every_occupied_body_cell(self) -> None:
        env = self.arena()
        gate = Gate("g", [p(11, 11)], 1, 1, 5, phase=1, state="warning")
        env.gates = [gate]
        for action in (1, 0):
            self.assertFalse(env.step(action)["done"])
            self.assertEqual(gate.state, "warning")
            self.assertEqual(gate.ticks_until_change, 1)
        env.step(3)
        self.assertEqual(gate.state, "closed")
        self.assertNotIn(p(11, 11), env.snake)

    def test_moving_and_maze_walls_defer_without_telefrag_then_move(self) -> None:
        for kind in ("moving", "maze"):
            with self.subTest(kind=kind):
                env = self.arena()
                wall = MovingWall("w", kind, [[p(10, 11)], [p(11, 11)]], 3, 1, 1)
                env.moving_walls = [wall]
                for action in (1, 0):
                    self.assertFalse(env.step(action)["done"])
                    self.assertEqual(wall.index, 0)
                    self.assertEqual(wall.next_move, env.steps + 1)
                self.assertFalse(env.step(1)["done"])
                self.assertEqual(wall.cells, [p(11, 11)])
                self.assertFalse(set(wall.cells) & env._occupied())
                self.assertTrue(any(e["type"] == ("maze_shift" if kind == "maze" else "wall_move")
                                    for e in env.events))

    def test_reconfiguration_cannot_seal_last_legal_exit_or_split_floor(self) -> None:
        env = self.arena()
        self.place_player(env, [p(2, 2), p(2, 3)])
        env.obstacles = [p(1, 2), p(3, 2), p(2, 2, 10), p(2, 2, 12)]
        gate = Gate("g", [p(2, 1)], 1, 1, 3, phase=1, state="warning")
        env.gates = [gate]
        env._update_hazards()
        self.assertEqual(gate.state, "warning")
        self.assertFalse(env._safe_configuration({Point(8, y, z) for y in range(22) for z in range(22)}))
        self.assertFalse(env._safe_configuration(env._solid_cells() | {p(2, 1)}))

    def test_reconfiguration_does_not_count_an_enemy_body_as_an_escape_route(self) -> None:
        env = self.arena()
        self.place_player(env, [p(2, 2), p(2, 3)])
        env.obstacles = [p(3, 2), p(2, 2, 10), p(2, 2, 12)]
        env.enemies = [Enemy("stationary", "ambush", [p(1, 2)], DIRECTIONS[1],
                             next_move=100, mode="waiting")]
        gate = Gate("last-exit", [p(2, 1)], 1, 1, 3, phase=1, state="warning")
        env.gates = [gate]
        env._update_hazards()
        self.assertEqual(gate.state, "warning")
        self.assertTrue(env._safe_configuration(env._solid_cells()))
        env.enemies.clear()
        env._update_hazards()
        self.assertEqual(gate.state, "closed")

    def test_zone_has_warning_defers_for_body_then_changes_real_collision_boundary(self) -> None:
        env = self.arena()
        env.config.update(shrinking_zone=True, zone_interval=5, zone_warning_ticks=2, zone_max_inset=1)
        env.zone_next_step = 5
        self.cycle(env, 3)
        self.assertEqual(env.board()["zone"]["ticksUntilShrink"], 2)
        self.assertTrue(env.board()["zone"]["warning"])
        env.step(2)
        env.step(1)
        self.assertEqual(env.zone_inset, 1)
        self.place_player(env, [p(1, 5), p(1, 6), p(1, 7)])
        self.assertTrue(env.step(3)["done"])
        self.assertEqual(env.last_death, "zone")
        env = self.arena()
        env.config.update(shrinking_zone=True, zone_interval=5, zone_warning_ticks=2, zone_max_inset=1)
        env.zone_next_step = 1
        self.place_player(env, [p(0, 5), p(0, 6), p(0, 7)])
        for _ in range(2):
            self.assertFalse(env.step(1)["done"])
            self.assertEqual(env.zone_inset, 0)
            self.assertTrue(env.board()["zone"]["warning"])
        self.assertFalse(env.step(1)["done"])
        self.assertEqual(env.zone_inset, 1)

    def test_gold_lifetime_respawn_value_and_score_are_actual(self) -> None:
        env = self.arena()
        env.config.update(gold=True, gold_interval=4)
        env.next_gold_step = 4
        env.foods.append(Food("gold-test", "gold", p(4, 4), 3, expires_at=2))
        env.step(1)
        self.assertEqual(next(f for f in env.board()["foods"] if f["type"] == "gold")["ttl"], 1)
        env.step(0)
        self.assertFalse(any(f.type == "gold" for f in env.foods))
        env.step(3)
        env.step(2)
        self.assertTrue(any(f.type == "gold" for f in env.foods))
        self.assertEqual(env.score, 0)
        env = self.arena()
        env.foods.append(Food("gold-test", "gold", p(12, 11), 3, expires_at=20))
        result = env.step(1)
        self.assertEqual((env.score, env.food_collected, len(env.snake)), (3, 1, 4))
        self.assertEqual(result["info"]["rewardComponents"]["food"], 18)
        self.assertTrue(any(e["type"] == "gold_food" for e in result["info"]["events"]))

    def test_shaping_telescopes_and_proximity_loop_cannot_earn_positive_return(self) -> None:
        env = self.arena()
        env.food = p(13, 11)
        initial = env._potential()
        rewards, shapings = [], []
        for i in range(20):
            result = env.step((1, 0, 3, 2)[i % 4])
            self.assertFalse(result["done"])
            self.assertEqual(env.food_collected, 0)
            rewards.append(result["reward"])
            shapings.append(result["info"]["rewardComponents"]["potential"])
        discounted = sum(SHAPING_GAMMA ** i * reward for i, reward in enumerate(shapings))
        self.assertAlmostEqual(discounted, SHAPING_GAMMA ** len(shapings) * env._potential() - initial)
        self.assertLess(discounted, 0)
        self.assertLess(sum(rewards), 0)
        self.assertLess(rewards[-1], rewards[3])

    def test_escape_requires_real_near_encounter_sustained_separation_and_cooldown(self) -> None:
        env = self.arena()
        hunter = Enemy("h", "hunter", [p(8, 11)], DIRECTIONS[1], next_move=1000,
                       detection_range=20, alert=True)
        env.enemies = [hunter]
        self.assertEqual(env._update_escape(), 0)
        self.assertEqual(env._encounter, hunter.id)
        self.place_player(env, [p(17, 11)])
        for _ in range(int(ESCAPE_CONFIG["separation_steps"]) - 1):
            self.assertEqual(env._update_escape(), 0)
        self.place_player(env, [p(12, 11)])
        self.assertEqual(env._update_escape(), 0)
        self.assertEqual(env._separation_ticks, 0)
        self.place_player(env, [p(17, 11)])
        for _ in range(int(ESCAPE_CONFIG["separation_steps"]) - 1):
            self.assertEqual(env._update_escape(), 0)
        self.assertEqual(env._update_escape(), ESCAPE_CONFIG["reward"])
        self.assertEqual(env.enemy_escapes, 1)
        self.place_player(env, [p(11, 11)])
        self.assertEqual(env._update_escape(), 0)
        self.assertIsNone(env._encounter)
        self.assertGreater(env._encounter_cooldown, 0)
        env = self.arena()
        env.enemies = [Enemy("h", "hunter", [p(1, 1)], DIRECTIONS[1], next_move=1000,
                             detection_range=20, alert=True)]
        for _ in range(20):
            self.assertEqual(env._update_escape(), 0)
        self.assertEqual(env.enemy_escapes, 0)

    def test_policy_observes_enemy_type_motion_gate_wall_food_zone_and_abilities(self) -> None:
        env = self.arena()
        env.enemies = [Enemy("h", "hunter", [p(14, 11)], DIRECTIONS[3], detection_range=12)]
        before = self.features(env)
        env.enemies[0].direction = DIRECTIONS[0]
        after = self.features(env)
        self.assertNotEqual(before["Hunter velocity X"], after["Hunter velocity X"])
        env.enemies[0].type = "rival"
        typed = self.features(env)
        self.assertNotEqual(after["Hunter present"], typed["Hunter present"])
        self.assertNotEqual(after["Rival present"], typed["Rival present"])
        env.gates = [Gate("g", [p(12, 11)], 2, 2, 3, phase=3, state="warning")]
        warning = self.features(env)
        env.gates[0].state, env.gates[0].phase = "closed", 4
        closed = self.features(env)
        self.assertNotEqual(warning["Gate closed"], closed["Gate closed"])
        self.assertNotEqual(warning["Gate warning"], closed["Gate warning"])
        env.moving_walls = [
            MovingWall("w", "moving", [[p(8, 8)], [p(9, 8)]], 10, 3, 10),
            MovingWall("m", "maze", [[p(6, 6)], [p(6, 7)]], 45, 8, 45),
        ]
        later = self.features(env)
        env.moving_walls[0].next_move = 1
        env.moving_walls[1].next_move = 1
        imminent = self.features(env)
        self.assertNotEqual(later["Moving wall warning"], imminent["Moving wall warning"])
        self.assertNotEqual(later["Maze wall timer"], imminent["Maze wall timer"])
        env.foods.append(Food("g", "gold", p(12, 12), 3, 50))
        long_life = self.features(env)
        env.foods[-1].expires_at = 1
        short_life = self.features(env)
        self.assertNotEqual(long_life["Gold lifetime"], short_life["Gold lifetime"])
        self.assertEqual(long_life["Gold value"], 1)
        env.config["shrinking_zone"] = True
        env.zone_next_step = 1
        zone = self.features(env)
        self.assertNotEqual(short_life["Zone warning"], zone["Zone warning"])
        env.dash_cooldown, env.shield_cooldown = 12, 20
        abilities = self.features(env)
        self.assertNotEqual(zone["Dash ready"], abilities["Dash ready"])
        self.assertNotEqual(zone["Shield cooldown"], abilities["Shield cooldown"])

    def test_multiseed_full_mode_random_legal_rollouts_remain_finite_and_nonoverlapping(self) -> None:
        rng = random.Random(1729)
        causes: Counter[str] = Counter()
        transitions = 0
        for seed in (101, 202, 303, 404, 505, 606, 707, 808):
            env = SnakeEnvironment(seed=seed, full_survival=True)
            for i in range(160):
                if env.done:
                    causes[env.last_death] += 1
                    env.reset(seed=seed * 1000 + i)
                mask = env.action_mask()
                action = rng.choice([index for index in range(ACTION_SIZE) if mask[index]])
                result = env.step(action)
                self.assertTrue(math.isfinite(result["reward"]))
                self.features(env)
                board = env.board()
                self.assertEqual(len(board["events"]), len({e["id"] for e in board["events"]}))
                occupied = env.snake + [part for enemy in env.enemies for part in enemy.snake]
                self.assertEqual(len(occupied), len(set(occupied)))
                self.assertFalse(set(occupied) & env._solid_cells())
                self.assertTrue(all(env._inside(part) for part in occupied))
                for enemy in env.enemies:
                    self.assertTrue(all(abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) == 1
                                        for a, b in zip(enemy.snake, enemy.snake[1:])))
                transitions += 1
        self.assertEqual(transitions, 1280)
        self.assertGreater(sum(causes.values()), 0)

    def test_vertical_food_obstacle_boundary_and_reverse_are_real(self) -> None:
        env = self.arena()
        env.food = p(11, 11, 12)
        self.assertFalse(env.step(4)["done"])
        self.assertEqual((env.snake[0], env.food_collected), (p(11, 11, 12), 1))
        self.assertFalse(any(env.action_mask()[i] for i in (5, 11, 17)))
        result = env.step(5)
        self.assertTrue(result["info"]["invalidAction"])
        self.assertEqual(env.snake[0], p(11, 11, 13))
        env = self.arena()
        env.obstacles = [p(11, 11, 12)]
        self.assertEqual(self.features(env)["Current danger Up"], 1)
        self.assertTrue(env.action_mask()[4], "Hazards must not be masked.")
        self.assertTrue(env.step(4)["done"])
        self.assertEqual(env.last_death, "obstacle")
        for y, action in ((0, 5), (21, 4)):
            env = self.arena()
            self.place_player(env, [p(11, 11, y)])
            self.assertTrue(env.step(action)["done"])
            self.assertEqual(env.last_death, "wall")

    def test_vertical_dash_and_shield_check_every_actual_cell(self) -> None:
        env = self.arena()
        env.food, env.obstacles = p(11, 11, 12), [p(11, 11, 13)]
        self.assertTrue(env.step(10)["done"])
        self.assertEqual(env.snake[0], p(11, 11, 12))
        self.assertEqual(env.death_point, p(11, 11, 13))
        self.assertEqual(env.food_collected, 1)
        env = self.arena()
        env.enemies = [Enemy("above", "hunter", [p(11, 11, 12)], DIRECTIONS[5], next_move=100)]
        self.assertFalse(env.step(16)["done"])
        self.assertEqual(env.snake[0], p(11, 11))
        self.assertTrue(env.step(4)["done"])
        self.assertEqual(env.last_death, "hunter")

    def test_enemy_navigation_and_observations_include_height(self) -> None:
        env = self.arena()
        env.food = p(11, 11, 16)
        hunter = Enemy("above", "hunter", [p(11, 11, 14)], DIRECTIONS[5],
                       detection_range=12, interval=1, next_move=1)
        env.enemies = [hunter]
        features = self.features(env)
        self.assertGreater(features["Food relative Y"], 0)
        self.assertGreater(features["Hunter relative Y"], 0)
        self.assertLess(features["Hunter velocity Y"], 0)
        self.assertEqual(env._plan_enemies()["above"], p(11, 11, 13))
        env.obstacles = [p(11, 11, 13)]
        destination = env._plan_enemies()["above"]
        self.assertNotEqual(destination, hunter.snake[0])
        self.assertNotIn(destination, env.obstacles)
        self.assertEqual(sum(abs(a - b) for a, b in zip(
            (destination.x, destination.y, destination.z), (11, 14, 11))), 1)

    def test_span_connectivity_matches_independent_cell_flood_fill(self) -> None:
        env = self.arena(grid_size=8)
        rng = random.Random(324)
        cells = {Point(x, y, z) for x in range(8) for y in range(8) for z in range(8)}
        scenarios = [set(), {Point(4, y, z) for y in range(8) for z in range(8)}]
        scenarios += [{cell for cell in sorted(cells, key=lambda p: (p.x, p.y, p.z))
                       if rng.random() < 0.4} for _ in range(4)]
        for blocked in scenarios:
            available = cells - blocked
            seen = set()
            queue = deque([next(iter(available))]) if available else deque()
            while queue:
                cell = queue.popleft()
                if cell in seen:
                    continue
                seen.add(cell)
                queue.extend(Point(cell.x + d.x, cell.y + d.y, cell.z + d.z) for d in DIRECTIONS
                             if Point(cell.x + d.x, cell.y + d.y, cell.z + d.z) in available - seen)
            self.assertEqual(env._floor_connected(blocked), bool(available) and seen == available)

    def test_astar_far_target_is_direct_cached_and_replans_for_changed_solids(self) -> None:
        env = self.arena(grid_size=32)
        start, goal = Point(2, 2, 2), Point(28, 27, 29)
        with patch("backend.environment.heapq.heappop", wraps=heapq.heappop) as pop:
            first = env._astar_next(start, goal, set())
            self.assertLessEqual(pop.call_count, 80, "Manhattan ties must not flood the whole 3D volume.")
            calls = pop.call_count
            self.assertEqual(env._astar_next(start, goal, set()), first)
            self.assertEqual(pop.call_count, calls)
        self.assertIsNotNone(first)
        changed = env._astar_next(start, goal, {first})
        self.assertNotEqual(changed, first)
        self.assertIsNone(env._astar_next(start, goal, {goal}))

    def test_hunter_and_rival_search_move_every_tick_without_teleporting(self) -> None:
        env = SnakeEnvironment(seed=73, level=4)
        enemies = [enemy for enemy in env.enemies if enemy.type in {"hunter", "rival"}]
        self.assertEqual(len(enemies), 2)
        for tick in range(1, 7):
            previous = {enemy.id: enemy.snake[0] for enemy in enemies}
            plans = env._plan_enemies(now=tick)
            env.steps = tick
            env._commit_enemies(plans, tick)
            for enemy in enemies:
                old, new = previous[enemy.id], enemy.snake[0]
                self.assertEqual(abs(old.x - new.x) + abs(old.y - new.y) + abs(old.z - new.z), 1)
                self.assertNotIn(new, env._solid_cells())
                self.assertEqual(enemy.next_move, tick + 1)

    def test_central_baffles_preserve_spawn_space_and_broad_six_axis_passages(self) -> None:
        for level in (1, 4, 6):
            env = SnakeEnvironment(seed=73, level=level)
            blocked = set(env.obstacles)
            self.assertEqual(len(blocked), env.obstacle_count)
            self.assertLess(len(blocked) / env.grid_size**3, 0.008)
            connected = {p for p in blocked if any(
                Point(p.x + d.x, p.y + d.y, p.z + d.z) in blocked for d in DIRECTIONS)}
            self.assertEqual(connected, blocked, "Obstacles should be structures, not isolated single cubes.")
            self.assertTrue(env._floor_connected(env._solid_cells()))
            self.assertFalse(blocked & env._occupied())
            head = env.snake[0]
            self.assertFalse(blocked & env._spawn_clearance(env.snake, env.direction))
            self.assertGreater(env.arena["centralObstacleCount"], 0)
            self.assertTrue(any(max(abs(p.x - head.x), abs(p.y - head.y), abs(p.z - head.z)) <= 5
                                for p in blocked), "The starting play area must contain real visible baffles.")
            footprints = env._solid_cells() | env._reserved_paths()
            for point in footprints:
                offsets = sorted((abs(point.x - head.x), abs(point.y - head.y), abs(point.z - head.z)))
                self.assertGreater(offsets[1], 2, "All three broad through-passages must remain open.")
            levels = {point.y for point in blocked}
            self.assertTrue(any(y < head.y - 5 for y in levels))
            if level == 6:
                self.assertTrue(any(y > head.y + 5 for y in levels))
            self.assertEqual(env.board()["arena"]["layout"], "open-chambers")
            for enemy in env.enemies:
                self.assertFalse(any(
                    max(abs(solid.x - part.x), abs(solid.y - part.y), abs(solid.z - part.z)) <= 1
                    for part in enemy.snake for solid in blocked
                ))
        a = SnakeEnvironment(seed=73, level=6)
        b = SnakeEnvironment(seed=74, level=6)
        self.assertNotEqual(a.obstacles, b.obstacles)

    def test_enemy_intent_is_the_actual_planner_target_and_next_step(self) -> None:
        env = self.arena()
        self.place_player(env, [p(11, 11), p(10, 11), p(9, 11)], DIRECTIONS[1])
        hunter = Enemy("hunter", "hunter", [p(6, 11), p(5, 11)], DIRECTIONS[1],
                       interval=1, detection_range=12, next_move=1)
        rival = Enemy("rival", "rival", [p(16, 11), p(17, 11)], DIRECTIONS[3],
                      interval=1, next_move=1)
        env.enemies = [hunter, rival]
        env.foods = [Food("near", "normal", p(18, 11)), Food("valuable", "gold", p(16, 16), 3)]
        snapshot = env.board()
        actors = {e["id"]: e for e in snapshot["enemies"]}
        self.assertEqual(actors["hunter"]["intent"]["behavior"], "intercept")
        self.assertEqual(actors["hunter"]["intent"]["target"], {"x": 13, "y": 11, "z": 11})
        self.assertEqual(actors["rival"]["intent"]["targetId"], "valuable")
        self.assertEqual(actors["rival"]["intent"]["target"], {"x": 16, "y": 11, "z": 16})
        plans = env._plan_enemies()
        for enemy in env.enemies:
            self.assertEqual(actors[enemy.id]["intent"]["nextPosition"], actors[enemy.id]["nextPosition"])
        before = env.rng.getstate(), deepcopy(snapshot), env.steps
        self.assertEqual(env.board(), snapshot)
        self.assertEqual((env.rng.getstate(), env.board(), env.steps), before)
        env._commit_enemies(plans, 1)
        env.steps = 1
        for enemy in env.enemies:
            self.assertEqual(enemy.snake[0].__dict__, actors[enemy.id]["intent"]["nextPosition"])
        hunter.detection_range = 0
        self.assertEqual(env.board()["enemies"][0]["intent"]["behavior"], "search")
        hunter.stunned_until = 10
        stopped = env.board()["enemies"][0]
        self.assertEqual(stopped["intent"]["behavior"], "stunned")
        self.assertEqual(stopped["intent"]["nextPosition"], stopped["snake"][0])


    def test_value_and_safety_choose_reachable_food_without_forcing_a_gold_suicide(self) -> None:
        env = self.arena()
        env.foods = [Food("normal", "normal", p(15, 11)),
                     Food("gold", "gold", p(11, 7), 3, expires_at=100)]
        self.assertEqual(env._objective_state()["target"]["id"], "gold")
        to_gold, to_normal = deepcopy(env), deepcopy(env)
        self.assertGreater(to_gold.step(0)["reward"], to_normal.step(1)["reward"])
        env.enemies = [Enemy("hunter", "hunter", [p(11, 6), p(11, 5)], DIRECTIONS[2],
                             next_move=1000, detection_range=12)]
        self.assertEqual(env._objective_state()["target"]["id"], "normal",
                         "Nearby unsafe gold must not override a safer reachable normal food.")
        env.enemies.clear()
        env.foods[1].expires_at = 2
        self.assertEqual(env._objective_state()["target"]["id"], "normal",
                         "Gold that expires before arrival is not a viable reward objective.")
        env.foods[1].expires_at = 100
        env.foods[1].position = p(9, 11)
        env.obstacles = [Point(10, y, z) for y in range(env.grid_size) for z in range(env.grid_size)]
        self.assertEqual(env._objective_state()["target"]["id"], "normal",
                         "An unreachable high-value resource must not attract the policy's shaping.")

    def test_route_reward_uses_true_detour_distance_not_just_heading_toward_food(self) -> None:
        env = self.arena()
        self.place_player(env, [p(11, 11), p(11, 11, 10), p(11, 11, 9)], DIRECTIONS[4])
        env.foods = [Food("normal", "normal", p(14, 11))]
        env.obstacles = [Point(12, y, z) for y in range(env.grid_size) for z in range(11, env.grid_size)]
        start, goal = env.snake[0], env.food
        blocked = env._solid_cells() | set(env.snake[1:-1])
        seen, queue = {start}, deque([(start, 0)])
        expected = None
        while queue:
            cell, distance = queue.popleft()
            if cell == goal:
                expected = distance
                break
            for direction in DIRECTIONS:
                neighbor = Point(cell.x + direction.x, cell.y + direction.y, cell.z + direction.z)
                if env._inside(neighbor) and neighbor not in blocked and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        self.assertEqual(env._objective_state()["target"]["distance"], expected)
        self.assertGreater(expected, 3, "The barrier makes Manhattan distance underestimate this route.")
        around, away = deepcopy(env), deepcopy(env)
        self.assertGreater(around.step(0)["reward"], away.step(2)["reward"],
                           "Two moves equally far in straight-line distance need different route feedback.")

    def test_rival_arrival_changes_food_value_and_only_real_contestable_loss_is_penalized(self) -> None:
        env = self.arena()
        env.foods = [Food("normal", "normal", p(11, 7)),
                     Food("gold", "gold", p(15, 11), 3, expires_at=100)]
        env.enemies = [Enemy("rival", "rival", [p(16, 11), p(17, 11)], DIRECTIONS[3],
                             interval=1, next_move=1)]
        objective = env._objective_state()
        self.assertEqual(objective["target"]["id"], "normal",
                         "An unwinnable gold race must not dominate another obtainable food.")
        gold = next(food for food in objective["foods"] if food["id"] == "gold")
        self.assertEqual(gold["rivalArrival"], 1)
        self.assertGreater(gold["racePressure"], 0)
        env.foods = [Food("race", "normal", p(13, 11))]
        env.enemies[0].snake = [p(14, 11), p(15, 11)]
        result = env.step(0)
        self.assertEqual(result["info"]["rewardComponents"]["competition"], REWARD_CONFIG["contested_loss"])
        self.assertEqual(env.score, 0, "Training penalties must not change the visible food score.")
        self.assertEqual(env.rival_food_collected, 1)

    def test_survival_reward_prefers_clearance_and_death_outweighs_a_single_gold_dash(self) -> None:
        env = self.arena()
        env.enemies = [Enemy("hunter", "hunter", [p(8, 11), p(7, 11)], DIRECTIONS[1],
                             next_move=1000, detection_range=12)]
        away, toward = deepcopy(env), deepcopy(env)
        self.assertGreater(away.step(1)["reward"], toward.step(3)["reward"])
        env = self.arena()
        env.foods = [Food("gold", "gold", p(12, 11), 3, expires_at=100)]
        env.obstacles = [p(13, 11)]
        result = env.step(7)
        self.assertTrue(result["done"])
        self.assertEqual(env.score, 3)
        self.assertEqual(result["info"]["rewardComponents"]["food"], 18)
        self.assertLess(result["reward"], 0, "A gold pickup must not make an otherwise fatal dash profitable.")
        self.assertEqual(result["info"]["potentialAfter"], 0)
        self.assertAlmostEqual(sum(result["info"]["potentialComponents"].values()),
                               result["info"]["rewardComponents"]["potential"])
        self.assertEqual(env.board()["trainingReward"]["version"], REWARD_VERSION)
        self.assertAlmostEqual(env.board()["trainingReward"]["total"], result["reward"])

    def test_random_obstacles_warn_then_really_disappear_and_appear_without_read_side_effects(self) -> None:
        env = SnakeEnvironment(seed=73, level=1, step_limit=1000)
        change = deepcopy(env.obstacle_change)
        self.assertIsNotNone(change)
        before = set(env.obstacles)
        self.assertFalse(set(change.added) & (before | env._occupied() | {f.position for f in env.foods}))
        self.assertTrue(set(change.removed) <= before)
        frozen = env.rng.getstate(), deepcopy(env.board())
        for _ in range(3):
            env.get_state()
            env.board()
        self.assertEqual((env.rng.getstate(), env.board()), frozen)
        self.cycle(env, OBSTACLE_CHANGE_CONFIG["warning_ticks"] - 1)
        self.assertFalse(env.done)
        self.assertEqual(set(env.obstacles), before)
        self.assertEqual(env.board()["obstacleChange"]["ticksUntilChange"], 1)
        env.step(2)
        self.assertFalse(env.done)
        self.assertIsNone(env.obstacle_change)
        self.assertFalse(set(change.removed) & set(env.obstacles))
        self.assertTrue(set(change.added) <= set(env.obstacles))
        self.assertEqual(len(env.obstacles), len(before))
        self.assertFalse(env._solid_cells() & env._occupied())
        self.assertTrue(env._floor_connected(env._solid_cells()))
        event = next(e for e in env.events if e["type"] == "obstacles_changed")
        self.assertEqual(event["step"], OBSTACLE_CHANGE_CONFIG["warning_ticks"])

    def test_obstacle_appearance_defers_for_bodies_food_and_the_last_escape(self) -> None:
        for occupied in ("player", "enemy", "food", "last-exit"):
            with self.subTest(occupied=occupied):
                env = self.arena()
                old = p(18, 18)
                env.obstacles = [old]
                candidate = env.snake[-1] if occupied == "player" else p(14, 14)
                if occupied == "enemy":
                    env.enemies = [Enemy("guard", "patrol", [candidate], DIRECTIONS[0])]
                elif occupied == "food":
                    env.foods = [Food("blocked-food", "normal", candidate)]
                elif occupied == "last-exit":
                    self.place_player(env, [p(2, 2), p(2, 3)])
                    env.obstacles += [p(1, 2), p(3, 2), p(2, 2, 10), p(2, 2, 12)]
                    candidate = p(2, 1)
                env.obstacle_change = ObstacleChange([old], [candidate], 1)
                env.steps = 1
                before = set(env.obstacles)
                env._update_obstacles()
                self.assertEqual(set(env.obstacles), before)
                self.assertEqual(env.obstacle_change.change_at, 2)
                self.assertNotIn(candidate, env.obstacles)

    def test_random_obstacle_schedule_replays_and_pending_cells_reach_policy_prediction(self) -> None:
        a = SnakeEnvironment(seed=73, level=1, step_limit=1000)
        b = SnakeEnvironment(seed=73, level=1, step_limit=1000)
        other = SnakeEnvironment(seed=74, level=1, step_limit=1000)
        self.assertNotEqual(a.obstacle_change.added, other.obstacle_change.added)
        changes = 0
        for i in range(144):
            action = (1, 0, 3, 2)[i % 4]
            result = a.step(action)
            self.assertEqual(result, b.step(action))
            self.assertEqual(a.board(), b.board())
            self.assertFalse(a.done)
            changes += any(e["type"] == "obstacles_changed" for e in result["info"]["events"])
        self.assertGreaterEqual(changes, 2)
        env = self.arena()
        point = p(12, 11)
        env.obstacle_change = ObstacleChange([p(18, 18)], [point], 1)
        features = self.features(env)
        self.assertEqual(features["Current danger E"], 0)
        self.assertEqual(features["Next tick danger E"], 1)
        self.assertNotIn(point, env._food_candidates())
        self.assertTrue(env.action_mask()[1], "A warning must not become a hidden safety controller.")


if __name__ == "__main__":
    unittest.main()
