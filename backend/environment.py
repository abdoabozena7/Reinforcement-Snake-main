from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
import heapq
from itertools import count, product
from numbers import Integral
import random

from backend.config import (
    ABILITY_CONFIG, DIFFICULTIES, ENVIRONMENT_VERSION, ESCAPE_CONFIG, OBSTACLE_CHANGE_CONFIG,
    REWARD_CONFIG, REWARD_VERSION, SHAPING_GAMMA,
)


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    z: int


DIRECTIONS = (
    Point(0, 0, -1), Point(1, 0, 0), Point(0, 0, 1),
    Point(-1, 0, 0), Point(0, 1, 0), Point(0, -1, 0),
)
DIRECTION_NAMES = ("N", "E", "S", "W", "Up", "Down")
DIRECTION_COUNT = len(DIRECTIONS)
ACTION_LABELS = tuple(
    f"{direction}{suffix}"
    for suffix in ("", " + Dash", " + Shield")
    for direction in DIRECTION_NAMES
)
ACTION_SIZE = len(ACTION_LABELS)
ENEMY_TYPES = ("hunter", "patrol", "rival", "ambush")
NEIGHBOR_OFFSETS = tuple(p for p in product((-1, 0, 1), repeat=3) if p != (0, 0, 0))

OBSERVATION_LABELS = tuple(
    [f"Heading {d}" for d in DIRECTION_NAMES]
    + ["Head X", "Head Y", "Head Z", "Body length", "Episode remaining", "Level", "Full survival"]
    + [f"{kind} {feature}" for kind in ("Food", "Gold")
       for feature in ("present", "relative X", "relative Y", "relative Z", "distance", "value", "lifetime")]
    + [f"Solid clearance {d}" for d in DIRECTION_NAMES]
    + [f"Current danger {d}" for d in DIRECTION_NAMES]
    + [f"Next tick danger {d}" for d in DIRECTION_NAMES]
    + [f"Dash path danger {d}" for d in DIRECTION_NAMES]
    + [f"Dynamic now {i}" for i in range(len(NEIGHBOR_OFFSETS))]
    + [f"Dynamic next {i}" for i in range(len(NEIGHBOR_OFFSETS))]
    + [f"{kind.title()} {feature}" for kind in ENEMY_TYPES
       for feature in ("present", "relative X", "relative Y", "relative Z",
                       "velocity X", "velocity Y", "velocity Z",
                       "move timer", "alert", "body distance", "head distance")]
    + [f"Gate {feature}" for feature in
       ("present", "relative X", "relative Y", "relative Z", "closed", "warning", "timer", "next closed")]
    + [f"{kind} {feature}" for kind in ("Moving wall", "Maze wall")
       for feature in ("present", "relative X", "relative Y", "relative Z", "next relative X",
                       "next relative Y", "next relative Z", "timer", "warning")]
    + ["Zone inset", "Zone next inset", "Zone timer", "Zone warning",
       "Dash ready", "Dash cooldown", "Shield ready", "Shield cooldown",
       "Shield active", "Shield remaining", "Threat", "Recent cell visits",
       "Time since food", "Pursuit encounter", "Escape separation",
       "Gold spawn timer", "Zone clearance"]
    + [f"Food toward {d}" for d in DIRECTION_NAMES]
)
OBSERVATION_SIZE = len(OBSERVATION_LABELS)


@dataclass
class Food:
    id: str
    type: str
    position: Point
    value: int = 1
    expires_at: int | None = None


@dataclass
class Enemy:
    id: str
    type: str
    snake: list[Point]
    direction: Point
    interval: int = 3
    detection_range: int = 0
    next_move: int = 3
    route: list[Point] = field(default_factory=list)
    route_index: int = 0
    mode: str = "patrol"
    burst_remaining: int = 0
    resources: int = 0
    stunned_until: int = 0
    alert: bool = False


@dataclass
class Gate:
    id: str
    cells: list[Point]
    open_ticks: int = 24
    warning_ticks: int = 6
    closed_ticks: int = 16
    phase: int = 0
    state: str = "open"

    @property
    def period(self) -> int:
        return self.open_ticks + self.warning_ticks + self.closed_ticks

    @property
    def ticks_until_change(self) -> int:
        end = (self.open_ticks if self.state == "open" else
               self.open_ticks + self.warning_ticks if self.state == "warning" else self.period)
        return max(1, end - self.phase)


@dataclass
class MovingWall:
    id: str
    type: str
    layouts: list[list[Point]]
    interval: int
    warning_ticks: int
    next_move: int
    index: int = 0
    travel: int = 1

    @property
    def cells(self) -> list[Point]:
        return self.layouts[self.index]

    @property
    def next_index(self) -> int:
        candidate = self.index + self.travel
        if candidate < 0 or candidate >= len(self.layouts):
            candidate = self.index - self.travel
        return candidate if len(self.layouts) > 1 else self.index

    @property
    def next_cells(self) -> list[Point]:
        return self.layouts[self.next_index]


@dataclass
class ObstacleChange:
    removed: list[Point]
    added: list[Point]
    change_at: int


def _add(a: Point, b: Point, times: int = 1) -> Point:
    return Point(a.x + b.x * times, a.y + b.y * times, a.z + b.z * times)


def _distance(a: Point, b: Point) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)


def _json_point(p: Point) -> dict[str, int]:
    return {"x": p.x, "y": p.y, "z": p.z}


class SnakeEnvironment:
    """Deterministic, Python-authoritative six-direction 3D survival arena.

    Each tick plans enemies from the *previous* player heading, then resolves
    simultaneous player/enemy movement (including swaps) on dash substep one.
    Remaining dash cells are checked individually against the committed world.
    Food is collected after contacts, timers advance once, then announced
    geometry changes are admitted only when disjoint and escapable. A shield
    absorbs one enemy/laser contact transition by cancelling that movement; it cannot
    traverse solids, self, the zone boundary, moving walls or maze walls.
    """

    def __init__(
        self, seed: int = 0, level: int = 1, full_survival: bool = False,
        grid_size: int = 32, step_limit: int | None = None, **legacy: object,
    ) -> None:
        if not isinstance(grid_size, Integral) or not 8 <= grid_size <= 64:
            raise ValueError("grid_size must be an integer from 8 to 64")
        if set(legacy) - {"block_size", "obstacle_count"}:
            raise TypeError(f"Unknown environment options: {sorted(set(legacy) - {'block_size', 'obstacle_count'})}")
        if step_limit is not None and (not isinstance(step_limit, Integral) or step_limit < 1):
            raise ValueError("step_limit must be positive")
        self.grid_size = int(grid_size)
        self.block_size = 1
        self._step_limit_override = step_limit
        self._obstacle_override = legacy.get("obstacle_count")
        self.seed = int(seed)
        self.level = int(level)
        self.full_survival = bool(full_survival)
        self.reset(seed=seed, level=level, full_survival=full_survival)

    def reset(
        self, seed: int | None = None, level: int | None = None,
        full_survival: bool | None = None,
    ) -> list[float]:
        requested_level = self.level if level is None else int(level)
        if requested_level not in DIFFICULTIES:
            raise ValueError("level must be from 1 to 6")
        if seed is not None:
            self.seed = int(seed)
        if full_survival is not None:
            self.full_survival = bool(full_survival)
        self.level = 6 if self.full_survival else requested_level
        self.config = deepcopy(DIFFICULTIES[self.level])
        self.step_limit = int(self._step_limit_override or self.config["step_limit"])
        self.obstacle_count = int(
            round(int(self.config["obstacle_count"]) * (self.grid_size / 32) ** 2)
            if self._obstacle_override is None else self._obstacle_override
        )
        if self.obstacle_count < 0:
            raise ValueError("obstacle_count cannot be negative")
        self.rng = random.Random(self.seed)
        c = self.grid_size // 2
        self.snake = [Point(c, c, c), Point(c, c, c + 1), Point(c, c, c + 2)]
        self.direction = DIRECTIONS[0]
        self.steps = self.score = self.food_collected = self.enemy_escapes = 0
        self.gold_collected = self.rival_food_collected = 0
        self.danger_exposure = 0.0
        self.last_reward: dict[str, object] | None = None
        self.done = False
        self.last_death: str | None = None
        self.death_point: Point | None = None
        self.contact: dict[str, object] | None = None
        self.dash_cooldown = self.shield_cooldown = self.shield_ticks = 0
        self.shield_charge = 0
        self.zone_inset = 0
        self.zone_next_step = int(self.config["zone_interval"])
        self._last_food_step, self._last_escape_step = 0, -100
        self._encounter: str | None = None
        self._separation_ticks = self._encounter_cooldown = 0
        self._recent: deque[Point] = deque([self.snake[0]], maxlen=24)
        self._event_serial = self._food_serial = 0
        self.events: deque[dict[str, object]] = deque(maxlen=16)
        self._step_events: list[dict[str, object]] = []
        self.foods: list[Food] = []
        self.enemies: list[Enemy] = []
        self.gates: list[Gate] = []
        self.moving_walls: list[MovingWall] = []
        self.obstacles: list[Point] = []
        self.obstacle_change: ObstacleChange | None = None
        self.next_obstacle_warning = 0
        self._path_cache: dict[tuple[Point, Point, frozenset[Point], int], tuple[Point | None, int | None]] = {}
        self._build_arena()
        self._spawn_normal()
        self.next_gold_step = int(self.config["gold_interval"])
        if self.config["gold"]:
            self._spawn_gold()
        self._prepare_obstacle_change()
        self._refresh_alerts()
        return self.get_state()

    @property
    def food(self) -> Point | None:
        return next((f.position for f in self.foods if f.type == "normal"), None)

    @food.setter
    def food(self, position: Point | None) -> None:
        self.foods = [f for f in self.foods if f.type != "normal"]
        if position is not None:
            self.foods.insert(0, Food(f"food-{self._food_serial}", "normal", position))
            self._food_serial += 1

    def _scaled(self, x: int, z: int, y: int | None = None) -> Point:
        scale = (self.grid_size - 1) / 21
        return Point(round(x * scale), self.grid_size // 2 if y is None else round(y * scale),
                     round(z * scale))

    def _rectangle(self, x0: int, z0: int, x1: int, z1: int, y: int | None = None) -> list[Point]:
        a, b = self._scaled(x0, z0, y), self._scaled(x1, z1, y)
        return (
            [Point(x, a.y, a.z) for x in range(a.x, b.x)]
            + [Point(b.x, a.y, z) for z in range(a.z, b.z)]
            + [Point(x, a.y, b.z) for x in range(b.x, a.x, -1)]
            + [Point(a.x, a.y, z) for z in range(b.z, a.z, -1)]
        )

    def _readable_space(self) -> set[Point]:
        c, radius = self.grid_size // 2, max(1, self.grid_size // 6)
        passage = self.grid_size // 12
        self.arena = {
            "layout": "open-chambers",
            "centralChamber": {
                "min": _json_point(Point(c - radius, c - radius, c - radius)),
                "max": _json_point(Point(c + radius, c + radius, c + radius)),
            },
            "passageHalfWidth": passage,
            "structureCount": 0,
        }
        protected = {Point(x, y, z)
                     for x in range(c - radius, c + radius + 1)
                     for y in range(c - radius, c + radius + 1)
                     for z in range(c - radius, c + radius + 1)}
        for along in range(self.grid_size):
            for a in range(c - passage, c + passage + 1):
                for b in range(c - passage, c + passage + 1):
                    protected.update((Point(along, a, b), Point(a, along, b), Point(a, b, along)))
        return protected

    def _spawn_clearance(self, body: list[Point], direction: Point) -> set[Point]:
        centers = body + [_add(body[0], direction, i) for i in range(1, 4)]
        return {Point(p.x + dx, p.y + dy, p.z + dz)
                for p in centers for dx, dy, dz in product((-1, 0, 1), repeat=3)
                if self._inside(Point(p.x + dx, p.y + dy, p.z + dz), 0)}

    def _build_arena(self) -> None:
        reserved = self._readable_space() | self._spawn_clearance(self.snake, self.direction)
        patrol_templates = ((2, 3, 6, 6, 7), (15, 15, 19, 19, 14))
        for i in range(int(self.config["patrols"])):
            route = self._rectangle(*patrol_templates[i % len(patrol_templates)])
            if len(route) < 4 or set(route) & reserved:
                continue
            body = [route[0], route[-1], route[-2]]
            direction = Point(body[0].x - body[1].x, body[0].y - body[1].y, body[0].z - body[1].z)
            interval = int(self.config["patrol_interval"]) + i
            self.enemies.append(Enemy(f"patrol-{i}", "patrol", body, direction,
                                      interval=interval, next_move=interval, route=route))
            reserved.update(route)
            reserved.update(self._spawn_clearance(body, direction))

        first_gate, second_gate = self._scaled(6, 14, 9), self._scaled(15, 7, 13)
        gate_templates = (
            [_add(first_gate, DIRECTIONS[2], offset) for offset in (-1, 0, 1)],
            [_add(second_gate, DIRECTIONS[1], offset) for offset in (-1, 0, 1)],
        )
        for i in range(int(self.config["gates"])):
            cells = list(dict.fromkeys(gate_templates[i % 2]))
            if set(cells) & reserved or any(not self._inside(p, 0) for p in cells):
                continue
            gate = Gate(
                f"gate-{i}", cells, int(self.config["gate_open_ticks"]),
                int(self.config["gate_warning_ticks"]), int(self.config["gate_closed_ticks"]),
            )
            gate.phase = i * 7
            self.gates.append(gate)
            reserved.update(cells)

        first_wall, second_wall = self._scaled(4, 16, 6), self._scaled(16, 4, 15)
        wall_templates = (
            [[Point(first_wall.x + travel, first_wall.y, first_wall.z + offset)
              for offset in (-1, 0, 1)] for travel in range(4)],
            [[Point(second_wall.x + offset, second_wall.y, second_wall.z + travel)
              for offset in (-1, 0, 1)] for travel in range(4)],
        )
        for i in range(int(self.config["moving_walls"])):
            self._add_wall(f"wall-{i}", "moving", wall_templates[i % 2],
                           int(self.config["wall_interval"]) + i * 3,
                           int(self.config["wall_warning_ticks"]), reserved)
        if self.config["maze_walls"]:
            corner = self._scaled(10, 4, 14)
            layouts = [
                [_add(corner, DIRECTIONS[1], offset) for offset in (-1, 0, 1, 2)],
                [Point(corner.x - 1, corner.y, corner.z + offset) for offset in range(4)],
            ]
            self._add_wall("maze-0", "maze", layouts, int(self.config["maze_interval"]),
                           int(self.config["maze_warning_ticks"]), reserved)

        templates = (
            ("hunter", (6, 11, 8), DIRECTIONS[1]),
            ("hunter", (16, 10, 13), DIRECTIONS[3]),
            ("rival", (17, 4, 7), DIRECTIONS[3]),
            ("ambush", (5, 17, 15), DIRECTIONS[0]),
        )
        counts = Counter()
        for kind, position, direction in templates:
            limit = int(self.config[{"hunter": "hunters", "rival": "rivals", "ambush": "ambushers"}[kind]])
            if counts[kind] >= limit:
                continue
            body = self._find_spawn(self._scaled(*position), direction, reserved)
            if body is None:
                continue
            interval = int(self.config[f"{kind}_interval"]) if kind != "ambush" else 1
            detection = int(self.config["hunter_range"] if kind == "hunter" else
                            self.config["ambush_range"] if kind == "ambush" else 0)
            self.enemies.append(Enemy(
                f"{kind}-{counts[kind]}", kind, body, direction, interval=interval,
                detection_range=detection, next_move=interval,
                mode="waiting" if kind == "ambush" else "forage" if kind == "rival" else "hunt",
            ))
            reserved.update(self._spawn_clearance(body, direction))
            counts[kind] += 1

        self._build_static_obstacles(reserved)

    def _build_static_obstacles(self, reserved: set[Point]) -> None:
        width, depth = max(2, self.grid_size // 5), max(2, self.grid_size // 7)
        height = 1 if self.grid_size < 16 else 2 if self.grid_size < 48 else 3
        shape = ([(x, 0, z) for z in range(depth) for x in range(width)]
                 + [(x, y, 0) for y in range(1, height) for x in range(width)])
        c, radius = self.grid_size // 2, max(1, self.grid_size // 6)
        passage = int(self.arena["passageHalfWidth"])
        central_budget = min(36, self.obstacle_count // 3) if self.grid_size >= 16 else 0
        if central_budget >= 3:
            chamber_corners = {
                Point(x, y, z)
                for x in range(c - radius, c + radius + 1)
                for y in range(c - radius, c + radius + 1)
                for z in range(c - radius, c + radius + 1)
                if sum(abs(v - c) <= passage for v in (x, y, z)) < 2
            }
            reserved = reserved - chamber_corners
            reserved |= self._spawn_clearance(self.snake, self.direction) | self._reserved_paths()
            for enemy in self.enemies:
                reserved |= self._spawn_clearance(enemy.snake, enemy.direction)
        central_shape = [(x, y, z) for y in range(3) for x, z in ((0, 0), (1, 0), (0, 1))]
        offset = passage + 2
        central_sites = [Point(c + x * offset, c - 1, c + z * offset)
                         for x, z in ((-1, -1), (1, 1), (-1, 1), (1, -1))]
        edge = self.grid_size - 1
        low, high = round(edge * 0.2), round(edge * 0.8)
        lower = [Point(x, round(edge * 0.24), z) for x, z in product((low, high), repeat=2)]
        upper = [Point(x, round(edge * 0.76), z) for x, z in product((low, high), repeat=2)]
        self.rng.shuffle(lower)
        self.rng.shuffle(upper)
        sites = ([(point, True) for point in central_sites if central_budget >= 3]
                 + [(point, False) for pair in zip(lower, upper) for point in pair])
        blocked = self._solid_cells()
        occupied: set[Point] = set()
        central_count = 0
        for site, central in sites:
            remaining = self.obstacle_count - len(occupied)
            if remaining <= 0:
                break
            if central:
                remaining = min(remaining, central_budget - central_count)
                if remaining < 3:
                    continue
            offsets = (central_shape if central else shape)[:remaining]
            for _ in range(self.rng.randrange(4)):
                offsets = [(z, y, -x) for x, y, z in offsets]
            minimum = [min(p[axis] for p in offsets) for axis in range(3)]
            extents = [max(p[axis] for p in offsets) - minimum[axis] + 1 for axis in range(3)]
            shifts = list(product((-1, 0, 1), repeat=3))
            self.rng.shuffle(shifts)
            for dx, dy, dz in shifts:
                centers = (site.x + dx, site.y + dy, site.z + dz)
                anchor = [max(1, min(self.grid_size - extents[axis] - 1,
                                     centers[axis] - (extents[axis] // 2 if axis != 1 else 0)))
                          for axis in range(3)]
                cells = {Point(*(anchor[axis] + p[axis] - minimum[axis] for axis in range(3)))
                         for p in offsets}
                if cells & (reserved | blocked) or any(not self._inside(p, 1) for p in cells):
                    continue
                if self._floor_connected(blocked | cells):
                    occupied.update(cells)
                    blocked.update(cells)
                    self.arena["structureCount"] += 1
                    if central:
                        central_count += len(cells)
                    break
        self.obstacles = sorted(occupied, key=lambda p: (p.x, p.y, p.z))
        self.arena["obstacleBudget"] = self.obstacle_count
        self.arena["obstacleCount"] = len(self.obstacles)
        self.arena["centralObstacleCount"] = central_count

    def _add_wall(
        self, identifier: str, kind: str, layouts: list[list[Point]],
        interval: int, warning: int, reserved: set[Point],
    ) -> None:
        layouts = [list(dict.fromkeys(cells)) for cells in layouts]
        all_cells = {p for layout in layouts for p in layout}
        if all_cells & reserved or any(not self._inside(p, 0) for p in all_cells):
            return
        self.moving_walls.append(MovingWall(identifier, kind, layouts, interval, warning, interval))
        reserved.update(all_cells)

    def _schedule_obstacle_warning(self) -> None:
        self.next_obstacle_warning = self.steps + self.rng.randint(
            OBSTACLE_CHANGE_CONFIG["min_interval"], OBSTACLE_CHANGE_CONFIG["max_interval"],
        )

    def _prepare_obstacle_change(self) -> None:
        if not self.obstacles or self.obstacle_change is not None:
            return
        current = set(self.obstacles)
        first = self.rng.choice(sorted(current, key=lambda p: (p.x, p.y, p.z)))
        component, queue = {first}, deque([first])
        while queue:
            point = queue.popleft()
            for direction in DIRECTIONS:
                neighbor = _add(point, direction)
                if neighbor in current and neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        minimum = tuple(min(getattr(p, axis) for p in component) for axis in ("x", "y", "z"))
        offsets = [Point(p.x - minimum[0], p.y - minimum[1], p.z - minimum[2]) for p in component]
        extents = tuple(max(getattr(p, axis) for p in offsets) + 1 for axis in ("x", "y", "z"))
        inset = max(1, self.zone_inset)
        forbidden = self._occupied() | self._reserved_paths() | self._solid_cells()
        forbidden.update(food.position for food in self.foods)
        center, passage = self.grid_size // 2, self.grid_size // 12
        removed = sorted(component, key=lambda p: (p.x, p.y, p.z))
        if all(extent <= self.grid_size - inset * 2 for extent in extents):
            for _ in range(64):
                anchor = tuple(self.rng.randint(inset, self.grid_size - inset - extent) for extent in extents)
                added = {Point(p.x + anchor[0], p.y + anchor[1], p.z + anchor[2]) for p in offsets}
                if added & forbidden or any(
                    sum(abs(value - center) <= passage for value in (p.x, p.y, p.z)) >= 2 for p in added
                ):
                    continue
                if not self._safe_configuration((self._solid_cells() - component) | added):
                    continue
                self.obstacle_change = ObstacleChange(
                    removed, sorted(added, key=lambda p: (p.x, p.y, p.z)),
                    self.steps + OBSTACLE_CHANGE_CONFIG["warning_ticks"],
                )
                self._event("obstacle_warning", self.obstacle_change.added[0],
                            ticks=OBSTACLE_CHANGE_CONFIG["warning_ticks"])
                return
        self._event("obstacle_change_deferred", self.snake[0], reason="no_safe_location")
        self._schedule_obstacle_warning()

    def _update_obstacles(self) -> None:
        change = self.obstacle_change
        if change is None:
            if self.steps >= self.next_obstacle_warning:
                self._prepare_obstacle_change()
            return
        if self.steps < change.change_at:
            return
        current = set(self.obstacles)
        if not set(change.removed) <= current or any(not self._inside(p, max(1, self.zone_inset)) for p in change.added):
            self._event("obstacle_change_cancelled", change.added[0], reason="layout_or_zone_changed")
            self.obstacle_change = None
            self._schedule_obstacle_warning()
            return
        blocked = (self._solid_cells() - set(change.removed)) | set(change.added)
        occupied = self._occupied() | {food.position for food in self.foods}
        if set(change.added) & occupied or not self._safe_configuration(blocked):
            change.change_at = self.steps + 1
            return
        self.obstacles = sorted((current - set(change.removed)) | set(change.added), key=lambda p: (p.x, p.y, p.z))
        self.arena["obstacleCount"] = len(self.obstacles)
        center, radius = self.grid_size // 2, max(1, self.grid_size // 6)
        self.arena["centralObstacleCount"] = sum(
            max(abs(value - center) for value in (p.x, p.y, p.z)) <= radius for p in self.obstacles
        )
        self._path_cache.clear()
        self._event("obstacles_changed", change.added[0],
                    removed=[_json_point(p) for p in change.removed], added=[_json_point(p) for p in change.added])
        self.obstacle_change = None
        self._schedule_obstacle_warning()

    def _find_spawn(
        self, preferred: Point, direction: Point, reserved: set[Point],
    ) -> list[Point] | None:
        candidates, seen = deque([preferred]), {preferred}
        while candidates:
            head = candidates.popleft()
            body = [_add(head, direction, -i) for i in range(3)]
            if (all(self._inside(p) and p not in reserved for p in body)
                    and min(_distance(p, self.snake[0]) for p in body) >= 4):
                return body
            for delta in DIRECTIONS:
                neighbor = _add(head, delta)
                if self._inside(neighbor) and neighbor not in seen:
                    seen.add(neighbor)
                    candidates.append(neighbor)
        return None

    def _inside(self, point: Point, inset: int | None = None) -> bool:
        inset = self.zone_inset if inset is None else inset
        return (inset <= point.y < self.grid_size - inset
                and inset <= point.x < self.grid_size - inset
                and inset <= point.z < self.grid_size - inset)

    def _occupied(self) -> set[Point]:
        return set(self.snake) | {p for e in self.enemies for p in e.snake}

    def _solid_cells(self, excluding: object = None) -> set[Point]:
        blocked = set(self.obstacles)
        blocked.update(p for g in self.gates if g is not excluding and g.state == "closed" for p in g.cells)
        blocked.update(p for wall in self.moving_walls if wall is not excluding for p in wall.cells)
        return blocked

    def _reserved_paths(self) -> set[Point]:
        return ({p for g in self.gates for p in g.cells}
                | {p for w in self.moving_walls for layout in w.layouts for p in layout}
                | {p for e in self.enemies for p in e.route}
                | set(self.obstacle_change.added if self.obstacle_change else []))

    def _floor_connected(self, blocked: set[Point], inset: int | None = None) -> bool:
        inset = self.zone_inset if inset is None else inset
        end = self.grid_size - inset - 1
        blocked_rows: dict[tuple[int, int], list[int]] = {}
        for point in blocked:
            if self._inside(point, inset):
                blocked_rows.setdefault((point.y, point.z), []).append(point.x)
        # Flood free X-spans rather than all N^3 cells; connectivity is still exact.
        rows: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for y in range(inset, end + 1):
            for z in range(inset, end + 1):
                cuts = [inset - 1, *sorted(blocked_rows.get((y, z), [])), end + 1]
                rows[y, z] = [(a + 1, b - 1) for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        nodes = {(y, z, i) for (y, z), spans in rows.items() for i in range(len(spans))}
        if not nodes:
            return False
        start = min(nodes)
        seen, queue = {start}, deque([start])
        while queue:
            y, z, index = queue.popleft()
            lo, hi = rows[y, z][index]
            for ny, nz in ((y - 1, z), (y + 1, z), (y, z - 1), (y, z + 1)):
                for i, (other_lo, other_hi) in enumerate(rows.get((ny, nz), [])):
                    neighbor = (ny, nz, i)
                    if neighbor not in seen and lo <= other_hi and other_lo <= hi:
                        seen.add(neighbor)
                        queue.append(neighbor)
        return len(seen) == len(nodes)

    def _safe_configuration(self, blocked: set[Point], inset: int | None = None) -> bool:
        inset = self.zone_inset if inset is None else inset
        occupied = self._occupied()
        if occupied & blocked or any(not self._inside(p, inset) for p in occupied):
            return False
        if not self._floor_connected(blocked, inset):
            return False
        body = set(self.snake[1:-1]) | {p for enemy in self.enemies for p in enemy.snake}
        reverse = Point(-self.direction.x, -self.direction.y, -self.direction.z)
        return any((len(self.snake) == 1 or d != reverse)
                   and self._inside(_add(self.snake[0], d), inset)
                   and _add(self.snake[0], d) not in blocked | body for d in DIRECTIONS)

    def _food_candidates(self) -> list[Point]:
        forbidden = self._occupied() | self._solid_cells() | self._reserved_paths()
        forbidden.update(f.position for f in self.foods)
        return [Point(x, y, z)
                for x in range(self.zone_inset, self.grid_size - self.zone_inset)
                for y in range(self.zone_inset, self.grid_size - self.zone_inset)
                for z in range(self.zone_inset, self.grid_size - self.zone_inset)
                if Point(x, y, z) not in forbidden]

    def _spawn_normal(self) -> None:
        if any(f.type == "normal" for f in self.foods):
            return
        candidates = self._food_candidates()
        safe = [p for p in candidates if all(_distance(p, e.snake[0]) >= 3 for e in self.enemies)]
        if safe:
            candidates = safe
        if self.steps == 0:
            near = [p for p in candidates if 4 <= _distance(p, self.snake[0]) <= 9]
            if near:
                candidates = near
        if candidates:
            position = self.rng.choice(candidates)
            self.foods.append(Food(f"food-{self._food_serial}", "normal", position))
            self._food_serial += 1

    def _spawn_gold(self) -> None:
        if any(f.type == "gold" for f in self.foods):
            return
        candidates = self._food_candidates()
        candidates = [p for p in candidates
                      if all(_distance(p, e.snake[0]) >= 2 for e in self.enemies)]
        risky = [p for p in candidates
                 if any(3 <= _distance(p, e.snake[0]) <= 6 for e in self.enemies)]
        if risky:
            candidates = risky
        if candidates:
            position = self.rng.choice(candidates)
            self.foods.append(Food(f"gold-{self._food_serial}", "gold", position, 3,
                                   self.steps + int(self.config["gold_ttl"])))
            self._food_serial += 1
            self._event("gold_spawn", position)

    def action_mask(self) -> list[bool]:
        result = []
        for action in range(ACTION_SIZE):
            direction = DIRECTIONS[action % DIRECTION_COUNT]
            reverse = len(self.snake) > 1 and direction == Point(
                -self.direction.x, -self.direction.y, -self.direction.z)
            available = (action < DIRECTION_COUNT or bool(self.config["abilities"]) and (
                self.dash_cooldown == 0 if action < 2 * DIRECTION_COUNT else self.shield_cooldown == 0))
            result.append(not reverse and available)
        return result

    def _astar_route(
        self, start: Point, goal: Point, blocked: set[Point],
    ) -> tuple[Point | None, int | None]:
        if start == goal:
            return start, 0
        if goal in blocked or not self._inside(goal):
            return None, None
        key = (start, goal, frozenset(blocked), self.zone_inset)
        if key in self._path_cache:
            return self._path_cache[key]
        if len(self._path_cache) >= 64:
            self._path_cache.clear()
        serial = count()
        distance = _distance(start, goal)
        queue = [(distance, distance, next(serial), 0, start)]
        first: dict[Point, Point] = {}
        distances = {start: 0}
        while queue:
            _, _, _, cost, point = heapq.heappop(queue)
            if cost != distances[point]:
                continue
            if point == goal:
                self._path_cache[key] = first[point], cost
                return self._path_cache[key]
            for direction in DIRECTIONS:
                neighbor = _add(point, direction)
                next_cost = cost + 1
                if (not self._inside(neighbor) or neighbor in blocked
                        or next_cost >= distances.get(neighbor, self.grid_size ** 3)):
                    continue
                distances[neighbor] = next_cost
                first[neighbor] = neighbor if point == start else first[point]
                distance = _distance(neighbor, goal)
                heapq.heappush(queue, (next_cost + distance, distance,
                                      next(serial), next_cost, neighbor))
        self._path_cache[key] = None, None
        return None, None

    def _astar_next(self, start: Point, goal: Point, blocked: set[Point]) -> Point | None:
        return self._astar_route(start, goal, blocked)[0]

    def _wander(self, enemy: Enemy, blocked: set[Point], inset: int | None = None) -> Point:
        direction_index = DIRECTIONS.index(enemy.direction)
        for offset in range(DIRECTION_COUNT):
            point = _add(enemy.snake[0], DIRECTIONS[(direction_index + offset) % DIRECTION_COUNT])
            if self._inside(point, inset) and point not in blocked:
                return point
        return enemy.snake[0]

    def _enemy_navigation(
        self, now: int | None = None,
    ) -> tuple[dict[str, Point], dict[str, dict[str, object]]]:
        now = self.steps + 1 if now is None else now
        solid = self._solid_cells()
        bodies = {p for e in self.enemies for p in e.snake}
        plans: dict[str, Point] = {}
        intents: dict[str, dict[str, object]] = {}
        zone = self._zone_snapshot()
        for enemy in self.enemies:
            head = enemy.snake[0]
            plans[enemy.id] = head
            intent: dict[str, object] = {
                "objective": ("collect-food" if enemy.type == "rival" else
                              "patrol" if enemy.type == "patrol" else "catch-player"),
                "behavior": "waiting", "target": None, "targetId": None,
            }
            intents[enemy.id] = intent
            if now < enemy.next_move or now < enemy.stunned_until:
                if now < enemy.stunned_until:
                    intent["behavior"] = "stunned"
                continue
            if enemy.type == "ambush" and enemy.mode not in {"warning", "attack"}:
                intent["behavior"] = enemy.mode
                continue
            blocked = solid | (bodies - {head, enemy.snake[-1]})
            candidate: Point | None = None
            if zone["warning"] and any(not self._inside(p, int(zone["nextInset"])) for p in enemy.snake):
                c = self.grid_size // 2
                candidate = self._astar_next(head, Point(c, c, c), blocked)
                intent.update(behavior="leave-zone", target=_json_point(Point(c, c, c)))
            elif enemy.type == "patrol":
                point = enemy.route[(enemy.route_index + 1) % len(enemy.route)]
                target_inset = int(zone["nextInset"]) if zone["warning"] else self.zone_inset
                if self._inside(point, target_inset) and point not in blocked and _distance(head, point) == 1:
                    candidate = point
                    intent.update(behavior="patrol", target=_json_point(point))
                else:
                    candidate = self._wander(enemy, blocked, target_inset)
                    intent["behavior"] = "detour"
            elif enemy.type == "rival":
                blocked |= set(self.snake)
                for food in sorted(self.foods, key=lambda f: (_distance(head, f.position) / f.value, f.id)):
                    candidate = self._astar_next(head, food.position, blocked)
                    if candidate is not None:
                        intent.update(behavior="seek-food", target=_json_point(food.position), targetId=food.id)
                        break
            elif _distance(head, self.snake[0]) <= enemy.detection_range:
                target = self.snake[0]
                if enemy.type == "hunter" and _distance(head, target) >= 5:
                    intercept = _add(target, self.direction, 2)
                    if self._inside(intercept) and intercept not in blocked:
                        candidate = self._astar_next(head, intercept, blocked)
                        if candidate is not None:
                            intent.update(behavior="intercept", target=_json_point(intercept), targetId="player")
                if candidate is None:
                    candidate = self._astar_next(head, target, blocked)
                    if candidate is not None:
                        intent.update(behavior="chase", target=_json_point(target), targetId="player")
            if candidate is None:
                candidate = self._wander(enemy, blocked)
                intent.update(behavior="search", target=None, targetId=None)
            if candidate not in blocked and self._inside(candidate):
                plans[enemy.id] = candidate

        # Equal destinations and opposing swaps never overlap two enemies.
        for i, first in enumerate(self.enemies):
            for second in self.enemies[i + 1:]:
                a, b = plans[first.id], plans[second.id]
                if a == b or a == second.snake[0] and b == first.snake[0]:
                    plans[first.id], plans[second.id] = first.snake[0], second.snake[0]
                    intents[first.id]["behavior"] = intents[second.id]["behavior"] = "yield"
        for enemy in self.enemies:
            intents[enemy.id]["nextPosition"] = _json_point(plans[enemy.id])
        return plans, intents

    def _plan_enemies(self, now: int | None = None) -> dict[str, Point]:
        return self._enemy_navigation(now)[0]

    def _enemy_future(self, enemy: Enemy, point: Point) -> list[Point]:
        if point == enemy.snake[0]:
            return list(enemy.snake)
        growing = enemy.type == "rival" and any(f.position == point for f in self.foods) and len(enemy.snake) < 10
        return [point] + enemy.snake[:len(enemy.snake) if growing else len(enemy.snake) - 1]

    def _commit_enemies(self, plans: dict[str, Point], now: int) -> None:
        for enemy in self.enemies:
            head = plans[enemy.id]
            if head != enemy.snake[0]:
                previous = enemy.snake[0]
                enemy.snake = self._enemy_future(enemy, head)
                enemy.direction = Point(head.x - previous.x, head.y - previous.y, head.z - previous.z)
                if enemy.route:
                    try:
                        enemy.route_index = enemy.route.index(head)
                    except ValueError:
                        pass
                if enemy.type == "rival":
                    for food in list(self.foods):
                        if food.position == head:
                            enemy.resources += food.value
                            self.rival_food_collected += 1
                            self.foods.remove(food)
                            self._event("rival_food", head, enemyId=enemy.id, value=food.value, foodId=food.id)
            if now >= enemy.next_move and (
                    enemy.type != "ambush" or enemy.mode in {"warning", "attack"}):
                enemy.next_move = now + enemy.interval
                if enemy.type == "ambush" and enemy.mode in {"warning", "attack"}:
                    if enemy.mode == "warning":
                        enemy.burst_remaining = int(self.config["ambush_burst"])
                    enemy.mode = "attack"
                    enemy.burst_remaining -= 1
                    if enemy.burst_remaining <= 0:
                        enemy.mode = "recover"
                        enemy.next_move = now + int(self.config["ambush_recovery"])

    def _refresh_alerts(self) -> None:
        for enemy in self.enemies:
            enemy.alert = (
                _distance(enemy.snake[0], self.snake[0]) <= enemy.detection_range
                if enemy.type == "hunter" else enemy.mode in {"warning", "attack"}
            )

    def _update_ambush(self) -> None:
        for enemy in self.enemies:
            if enemy.type != "ambush":
                continue
            if enemy.mode == "recover" and self.steps >= enemy.next_move:
                enemy.mode = "waiting"
            if (enemy.mode == "waiting"
                    and _distance(enemy.snake[0], self.snake[0]) <= enemy.detection_range):
                enemy.mode = "warning"
                enemy.next_move = self.steps + int(self.config["ambush_warning"])
                self._event("ambush_warning", enemy.snake[0], enemyId=enemy.id)
        self._refresh_alerts()

    def _geometry_collision(self, point: Point, growing: bool) -> str | None:
        if not self._inside(point, 0):
            return "wall"
        if not self._inside(point):
            return "zone"
        if point in self.snake[:len(self.snake) if growing else -1]:
            return "self"
        if point in self.obstacles:
            return "obstacle"
        for wall in self.moving_walls:
            if point in wall.cells:
                return "maze_wall" if wall.type == "maze" else "moving_wall"
        if any(g.state == "closed" and point in g.cells for g in self.gates):
            return "laser"
        return None

    def _enemy_contacts(
        self, target: Point, player_future: list[Point], plans: dict[str, Point],
    ) -> list[Enemy]:
        contacts = []
        for enemy in self.enemies:
            future = self._enemy_future(enemy, plans[enemy.id])
            crossing = target == enemy.snake[0] and plans[enemy.id] == self.snake[0]
            attacks_body = plans[enemy.id] != enemy.snake[0] and plans[enemy.id] in player_future
            if target in future or crossing or attacks_body:
                contacts.append(enemy)
        return contacts

    @staticmethod
    def _contact_actor(
        identifier: str, kind: str, before: list[Point], attempted: list[Point], segment: int = 0,
    ) -> dict[str, object]:
        return {
            "id": identifier, "type": kind,
            "from": _json_point(before[0]), "to": _json_point(attempted[0]),
            "before": [_json_point(p) for p in before],
            "attempted": [_json_point(p) for p in attempted],
            "segmentIndex": segment,
        }

    def _contact_record(
        self, cause: str, target: Point, player_future: list[Point],
        plans: dict[str, Point], enemy: Enemy | None, substep: int, outcome: str,
    ) -> dict[str, object]:
        """Record uncommitted intent; substeps are 1-based, geometry stops at its cell face."""
        kind = "obstacle" if cause in {"moving_wall", "maze_wall"} else cause
        source: str | None = None
        attacker = None
        segment, fraction, position = 0, 0.5, None
        if enemy is not None:
            future = self._enemy_future(enemy, plans[enemy.id])
            source = enemy.id
            enemy_segment = 0
            if target == enemy.snake[0] and plans[enemy.id] == self.snake[0]:
                kind = "crossing"
            elif target == plans[enemy.id]:
                kind, fraction, position = "head-on", 1.0, _json_point(target)
            elif target in future:
                kind, fraction, position = "enemy-body", 1.0, _json_point(target)
                enemy_segment = future.index(target)
            else:
                kind, fraction, position = "body-strike", 1.0, _json_point(plans[enemy.id])
                segment = player_future.index(plans[enemy.id])
            attacker = self._contact_actor(enemy.id, enemy.type, enemy.snake, future, enemy_segment)
        elif cause == "self":
            source, fraction, position = "player", 1.0, _json_point(target)
            attacker = self._contact_actor(
                "player", "player", self.snake, player_future, player_future.index(target, 1),
            )
        elif cause == "laser":
            source = next(g.id for g in self.gates if g.state == "closed" and target in g.cells)
        elif cause in {"moving_wall", "maze_wall"}:
            source = next(w.id for w in self.moving_walls if target in w.cells)
        elif cause == "obstacle":
            source = f"obstacle-{target.x}-{target.y}-{target.z}"
        elif cause == "zone":
            source = "zone"
        if position is None:
            head = self.snake[0]
            position = {
                "x": head.x + (target.x - head.x) * fraction,
                "y": head.y + (target.y - head.y) * fraction,
                "z": head.z + (target.z - head.z) * fraction,
            }
        record: dict[str, object] = {
            "schemaVersion": 1,
            "id": f"{self.seed}:{self.steps}:{substep}:{self._event_serial}:contact",
            "seed": self.seed, "step": self.steps, "substep": substep,
            "kind": kind, "cause": cause, "outcome": outcome,
            "position": position, "timeFraction": fraction, "sourceId": source,
            "attacker": attacker,
            "victim": self._contact_actor("player", "player", self.snake, player_future, segment),
        }
        if enemy is None and cause != "self":
            record["geometry"] = {
                "type": cause, "cell": _json_point(target),
                "normal": _json_point(Point(-self.direction.x, -self.direction.y, -self.direction.z)),
            }
        return record

    def _event(
        self, kind: str, position: Point | dict[str, float], **extra: object,
    ) -> None:
        event: dict[str, object] = {
            "id": f"{self.seed}:{self.steps}:{self._event_serial}:{kind}",
            "type": kind, "step": self.steps,
            "position": _json_point(position) if isinstance(position, Point) else dict(position),
            **deepcopy(extra),
        }
        self._event_serial += 1
        self.events.append(event)
        self._step_events.append(event)

    def _position_risk(self, point: Point, destination: bool = False) -> float:
        risk = 0.0
        for enemy in self.enemies:
            if destination and enemy.type not in {"hunter", "ambush"}:
                continue
            distance = min(_distance(point, part) for part in enemy.snake)
            weight = 1.0 if enemy.type in {"hunter", "ambush"} else 0.6
            risk = max(risk, weight * max(0.0, 1.0 - distance / 6))
        if destination:
            return risk
        blocked = self._solid_cells() | set(self.snake[1:-1])
        blocked.update(part for enemy in self.enemies for part in enemy.snake)
        reverse = Point(-self.direction.x, -self.direction.y, -self.direction.z)
        exits = [direction for direction in DIRECTIONS if len(self.snake) == 1 or direction != reverse]
        unsafe = sum(not self._inside(_add(point, direction)) or _add(point, direction) in blocked for direction in exits)
        risk = max(risk, unsafe / len(exits) * 0.75)
        warning = set(self.obstacle_change.added if self.obstacle_change else [])
        warning.update(part for gate in self.gates if gate.state == "warning" for part in gate.cells)
        if warning:
            risk = max(risk, max(0.0, 1.0 - min(_distance(point, cell) for cell in warning) / 3) * 0.8)
        zone = self._zone_snapshot()
        if zone["warning"] and not self._inside(point, int(zone["nextInset"])):
            risk = max(risk, 0.9)
        return risk

    def _objective_state(self) -> dict[str, object]:
        terms = {"resource": 0.0, "safety": 0.0, "race": 0.0}
        if self.done:
            return {"terms": terms, "target": None, "foods": [], "risk": 0.0}
        head = self.snake[0]
        risk = self._position_risk(head)
        terms["safety"] = -REWARD_CONFIG["safety_potential"] * risk
        solid = self._solid_cells()
        player_blocked = solid | set(self.snake[1:-1])
        enemy_bodies = {part for enemy in self.enemies for part in enemy.snake}
        foods: list[dict[str, object]] = []
        for food in self.foods:
            _, distance = self._astar_route(head, food.position, player_blocked)
            if distance is None or food.expires_at is not None and distance > food.expires_at - self.steps:
                continue
            rival_arrivals = []
            for rival in self.enemies:
                if rival.type != "rival":
                    continue
                blocked = solid | (enemy_bodies - {rival.snake[0], rival.snake[-1]}) | set(self.snake)
                _, rival_distance = self._astar_route(rival.snake[0], food.position, blocked)
                if rival_distance is not None:
                    rival_arrivals.append(max(0, rival.next_move - self.steps - 1) + rival_distance * rival.interval)
            rival_steps = min(rival_arrivals) if rival_arrivals else None
            pressure = max(0.0, (distance - rival_steps + 1) / (distance + 1)) if rival_steps is not None else 0.0
            destination_risk = self._position_risk(food.position, destination=True)
            resource = REWARD_CONFIG["resource_potential"] * food.value / (distance + food.value)
            resource *= (1 - destination_risk) ** 2 * (1 - pressure)
            race = -REWARD_CONFIG["race_potential"] * pressure
            foods.append({
                "id": food.id, "type": food.type, "value": food.value, "distance": distance,
                "rivalArrival": rival_steps, "racePressure": pressure, "destinationRisk": destination_risk,
                "resourcePotential": resource, "racePotential": race,
            })
        target = max(foods, key=lambda food: (food["resourcePotential"] + food["racePotential"], food["id"])) if foods else None
        if target:
            terms["resource"] = target["resourcePotential"]
            terms["race"] = target["racePotential"]
        return {"terms": terms, "target": target, "foods": foods, "risk": risk}

    def _potential(self) -> float:
        return sum(self._objective_state()["terms"].values())

    def _collect_food(self) -> float:
        reward = 0.0
        for food in list(self.foods):
            if food.position != self.snake[0]:
                continue
            self.foods.remove(food)
            self.food_collected += 1
            self.gold_collected += int(food.type == "gold")
            self.score += food.value
            reward += REWARD_CONFIG["food_value"] * food.value
            self._last_food_step = self.steps
            self._event("gold_food" if food.type == "gold" else "food", food.position, value=food.value)
        return reward

    def _consume_shield(self, source: str, contact: dict[str, object]) -> None:
        self.shield_charge = self.shield_ticks = 0
        self._event("shield_block", contact["position"], source=source, contact=contact)

    def _update_hazards(self) -> None:
        occupied = self._occupied() | {f.position for f in self.foods}
        for gate in self.gates:
            phase = (gate.phase + 1) % gate.period
            state = ("open" if phase < gate.open_ticks else
                     "warning" if phase < gate.open_ticks + gate.warning_ticks else "closed")
            if state == "closed" and gate.state != "closed":
                blocked = self._solid_cells(excluding=gate) | set(gate.cells)
                if set(gate.cells) & occupied or not self._safe_configuration(blocked):
                    phase, state = gate.open_ticks + gate.warning_ticks - 1, "warning"
            if state != gate.state:
                self._event("gate_" + state, gate.cells[0], gateId=gate.id)
            gate.phase, gate.state = phase, state
        for wall in self.moving_walls:
            if self.steps < wall.next_move:
                continue
            blocked = self._solid_cells(excluding=wall) | set(wall.next_cells)
            if set(wall.next_cells) & occupied or not self._safe_configuration(blocked):
                wall.next_move = self.steps + 1
                continue
            new_index = wall.next_index
            if new_index < wall.index:
                wall.travel = -1
            elif new_index > wall.index:
                wall.travel = 1
            wall.index = new_index
            wall.next_move = self.steps + wall.interval
            self._event("maze_shift" if wall.type == "maze" else "wall_move",
                        wall.cells[0], wallId=wall.id)
        self._update_obstacles()
        zone = self._zone_snapshot()
        if self.config["shrinking_zone"] and self.steps >= self.zone_next_step and zone["nextInset"] > self.zone_inset:
            inset = int(zone["nextInset"])
            if self._safe_configuration(self._solid_cells(), inset):
                self.zone_inset = inset
                self.zone_next_step = self.steps + int(self.config["zone_interval"])
                for food in list(self.foods):
                    if not self._inside(food.position):
                        self.foods.remove(food)
                        self._event("food_expired", food.position, source="zone")
                self._event("zone_shrink", self.snake[0], inset=inset)
            else:
                self.zone_next_step = self.steps + 1

    def _update_foods(self) -> None:
        for food in list(self.foods):
            if food.expires_at is not None and self.steps >= food.expires_at:
                self.foods.remove(food)
                self._event("food_expired", food.position, source="lifetime")
        self._spawn_normal()
        if self.config["gold"] and self.steps >= self.next_gold_step:
            self._spawn_gold()
            self.next_gold_step = self.steps + int(self.config["gold_interval"])

    def _update_escape(self) -> float:
        if self._encounter_cooldown:
            self._encounter_cooldown -= 1
        pursuers = [e for e in self.enemies if e.type in {"hunter", "ambush"}]
        near = [e for e in pursuers if e.alert and _distance(e.snake[0], self.snake[0])
                <= int(ESCAPE_CONFIG["near_distance"])]
        if self._encounter is None and not self._encounter_cooldown and near:
            self._encounter = min(near, key=lambda e: (_distance(e.snake[0], self.snake[0]), e.id)).id
            self._separation_ticks = 0
        encounter = next((e for e in pursuers if e.id == self._encounter), None)
        if encounter is None:
            self._encounter = None
            self._separation_ticks = 0
            return 0.0
        if _distance(encounter.snake[0], self.snake[0]) >= int(ESCAPE_CONFIG["separation_distance"]) and not near:
            self._separation_ticks += 1
        else:
            self._separation_ticks = 0
        if self._separation_ticks < int(ESCAPE_CONFIG["separation_steps"]):
            return 0.0
        self.enemy_escapes += 1
        self._last_escape_step = self.steps
        self._event("escape", self.snake[0], enemyId=encounter.id)
        self._encounter = None
        self._separation_ticks = 0
        self._encounter_cooldown = int(ESCAPE_CONFIG["cooldown"])
        return float(ESCAPE_CONFIG["reward"])

    def step(self, action_index: int) -> dict[str, object]:
        if self.done:
            return {"next_state": self.get_state(), "reward": 0.0, "done": True,
                    "info": {"terminalAlreadyProcessed": True, "deathCause": self.last_death,
                             "steps": self.steps, "events": []}}
        if isinstance(action_index, bool) or not isinstance(action_index, Integral) or not 0 <= action_index < ACTION_SIZE:
            raise ValueError(f"action must be an integer from 0 to {ACTION_SIZE - 1}")
        action_index = int(action_index)
        mask = self.action_mask()
        objectives_before = self._objective_state()
        potential_before = sum(objectives_before["terms"].values())
        plans = self._plan_enemies()
        self.contact = None
        self._step_events = []
        requested_direction = DIRECTIONS[action_index % DIRECTION_COUNT]
        reverse = len(self.snake) > 1 and requested_direction == Point(
            -self.direction.x, -self.direction.y, -self.direction.z)
        if not reverse:
            self.direction = requested_direction
        ability = action_index // DIRECTION_COUNT
        ability_used: str | None = None
        components = {"living": REWARD_CONFIG["living"], "food": 0.0, "escape": 0.0, "potential": 0.0,
                      "risk": 0.0, "competition": 0.0, "loop": 0.0, "ability": 0.0, "invalid": 0.0, "death": 0.0}
        if not mask[action_index]:
            components["invalid"] = -0.1
        if mask[action_index] and ability:
            ability_used = "dash" if ability == 1 else "shield"
            if ability == 1:
                self.dash_cooldown = int(ABILITY_CONFIG["dash_cooldown"])
            else:
                self.shield_cooldown = int(ABILITY_CONFIG["shield_cooldown"])
                self.shield_ticks = int(ABILITY_CONFIG["shield_duration"])
                self.shield_charge = 1
            components["ability"] = -0.04
        self.steps += 1
        if ability_used:
            self._event(ability_used, self.snake[0])
        substeps = int(ABILITY_CONFIG["dash_distance"]) if ability_used == "dash" else 1
        for substep in range(substeps):
            target = _add(self.snake[0], self.direction)
            growing = any(f.position == target for f in self.foods)
            collision = self._geometry_collision(target, growing)
            player_future = [target] + self.snake[:len(self.snake) if growing else -1]
            current_plans = plans if substep == 0 else {e.id: e.snake[0] for e in self.enemies}
            contacts = self._enemy_contacts(target, player_future, current_plans)
            if collision is None and contacts:
                collision = contacts[0].type
            if collision:
                supported = collision == "laser" or collision in ENEMY_TYPES
                blocked = bool(supported and self.shield_ticks and self.shield_charge)
                self.contact = self._contact_record(
                    collision, target, player_future, current_plans,
                    contacts[0] if collision in ENEMY_TYPES else None,
                    substep + 1, "blocked" if blocked else "death",
                )
                if blocked:
                    self._consume_shield(collision, self.contact)
                    for enemy in contacts:
                        current_plans[enemy.id] = enemy.snake[0]
                        enemy.stunned_until = self.steps + 2
                    for enemy in self.enemies:
                        if current_plans[enemy.id] in self.snake:
                            current_plans[enemy.id] = enemy.snake[0]
                    if substep == 0:
                        self._commit_enemies(current_plans, self.steps)
                    break
                self.done = True
                self.last_death, self.death_point = collision, target
                components["death"] = REWARD_CONFIG["death"]
                self._event("death", self.contact["position"], cause=collision, contact=self.contact)
                break
            self.snake = player_future
            components["food"] += self._collect_food()
            if substep == 0:
                self._commit_enemies(current_plans, self.steps)
            self._spawn_normal()

        if not self.done:
            self._update_ambush()
            self._update_hazards()
            self._update_foods()
            self._refresh_alerts()
            components["escape"] = self._update_escape()
            visits = self._recent.count(self.snake[0])
            if visits >= 2:
                components["loop"] = -0.08 * min(visits - 1, 3)
            self._recent.append(self.snake[0])
            if self.steps >= self.step_limit:
                self.done, self.last_death = True, "time_limit"
                self.contact = None
                self._event("episode_end", self.snake[0], cause="time_limit")
        self.dash_cooldown = max(0, self.dash_cooldown - 1)
        self.shield_cooldown = max(0, self.shield_cooldown - 1)
        self.shield_ticks = max(0, self.shield_ticks - 1)
        if not self.shield_ticks:
            self.shield_charge = 0
        objectives_after = self._objective_state()
        potential_after = sum(objectives_after["terms"].values())
        components["potential"] = SHAPING_GAMMA * potential_after - potential_before
        components["risk"] = REWARD_CONFIG["risk_cost"] * objectives_before["risk"]
        target = objectives_before["target"]
        if target:
            components["competition"] = sum(
                REWARD_CONFIG["contested_loss"] * event["value"] for event in self._step_events
                if event["type"] == "rival_food" and event["foodId"] == target["id"]
                and (target["rivalArrival"] is None or target["distance"] <= target["rivalArrival"] + 2)
            )
        self.danger_exposure += objectives_before["risk"]
        shaping = {
            name: SHAPING_GAMMA * objectives_after["terms"][name] - value
            for name, value in objectives_before["terms"].items()
        }
        self.last_reward = {
            "version": REWARD_VERSION, "total": sum(components.values()), "components": dict(components),
            "shaping": shaping, "objectiveBefore": objectives_before, "objectiveAfter": objectives_after,
        }
        return {
            "next_state": self.get_state(), "reward": sum(components.values()), "done": self.done,
            "info": {
                "score": self.score, "steps": self.steps, "foodCollected": self.food_collected,
                "enemyEscapes": self.enemy_escapes, "deathCause": self.last_death,
                "requestedAction": action_index, "invalidAction": not mask[action_index],
                "abilityUsed": ability_used, "rewardComponents": components,
                "potentialBefore": potential_before, "potentialAfter": potential_after,
                "shapingGamma": SHAPING_GAMMA, "truncated": self.last_death == "time_limit",
                "potentialComponents": shaping, "rewardVersion": REWARD_VERSION,
                "objectiveBefore": objectives_before, "objectiveAfter": objectives_after,
                "events": deepcopy(self._step_events),
            },
        }

    def _zone_snapshot(self) -> dict[str, object]:
        maximum = min(int(self.config["zone_max_inset"]), max(0, self.grid_size // 2 - 4))
        next_inset = min(self.zone_inset + 1, maximum) if self.config["shrinking_zone"] else self.zone_inset
        remaining = max(0, self.zone_next_step - self.steps) if next_inset > self.zone_inset else 0
        return {
            "inset": self.zone_inset, "nextInset": next_inset,
            "warning": next_inset > self.zone_inset and remaining <= int(self.config["zone_warning_ticks"]),
            "ticksUntilShrink": remaining,
        }

    def _threat(self) -> float:
        values = [max(0.0, 1.0 - _distance(self.snake[0], e.snake[0]) / 12)
                  for e in self.enemies]
        if any(g.state == "warning" and min(_distance(self.snake[0], p) for p in g.cells) <= 2
               for g in self.gates):
            values.append(0.7)
        return max(values, default=0.0)

    def _dynamic_cells(self, future: bool, plans: dict[str, Point]) -> set[Point]:
        cells = {p for e in self.enemies
                 for p in (self._enemy_future(e, plans[e.id]) if future else e.snake)}
        for gate in self.gates:
            if gate.state == "closed" or future and gate.state == "warning" and gate.ticks_until_change <= 1:
                cells.update(gate.cells)
        for wall in self.moving_walls:
            cells.update(wall.next_cells if future and wall.next_move <= self.steps + 1 else wall.cells)
        if future and self.obstacle_change and self.obstacle_change.change_at <= self.steps + 1:
            cells.update(self.obstacle_change.added)
        return cells

    def _danger(self, point: Point, dynamic: set[Point], blocked: set[Point], inset: int) -> float:
        return float(not self._inside(point, inset) or point in blocked or point in dynamic)

    def get_state(self) -> list[float]:
        head, n = self.snake[0], self.grid_size - 1
        result = [float(self.direction == direction) for direction in DIRECTIONS]
        result += [head.x / n, head.y / n, head.z / n, min(1.0, len(self.snake) / (self.grid_size ** 3)),
                   max(0.0, 1.0 - self.steps / self.step_limit), self.level / 6, float(self.full_survival)]
        for kind in ("normal", "gold"):
            food = next((f for f in self.foods if f.type == kind), None)
            result += ([1.0, (food.position.x - head.x) / n, (food.position.y - head.y) / n,
                        (food.position.z - head.z) / n,
                        _distance(food.position, head) / (3 * n), food.value / 3,
                        1.0 if food.expires_at is None else
                        max(0.0, (food.expires_at - self.steps) / int(self.config["gold_ttl"]))]
                       if food else [0.0] * 7)
        solid = self._solid_cells()
        body = set(self.snake[:-1])
        ray_blocked = solid | body
        for direction in DIRECTIONS:
            distance = 1
            while distance < self.grid_size:
                point = _add(head, direction, distance)
                if not self._inside(point) or point in ray_blocked:
                    break
                distance += 1
            result.append((distance - 1) / n)
        plans = self._plan_enemies()
        now, future = self._dynamic_cells(False, plans), self._dynamic_cells(True, plans)
        zone = self._zone_snapshot()
        future_inset = int(zone["nextInset"]) if zone["warning"] and zone["ticksUntilShrink"] <= 1 else self.zone_inset
        static = set(self.obstacles) | body
        result += [self._danger(_add(head, d), now, static, self.zone_inset) for d in DIRECTIONS]
        result += [self._danger(_add(head, d), future, static, future_inset) for d in DIRECTIONS]
        result += [float(any(self._danger(_add(head, d, i), now | future, static, future_inset)
                             for i in range(1, int(ABILITY_CONFIG["dash_distance"]) + 1)))
                   for d in DIRECTIONS]
        for cells in (now, future):
            result += [float(Point(head.x + dx, head.y + dy, head.z + dz) in cells)
                       for dx, dy, dz in NEIGHBOR_OFFSETS]
        for kind in ENEMY_TYPES:
            enemies = [e for e in self.enemies if e.type == kind]
            enemy = min(enemies, key=lambda e: (_distance(head, e.snake[0]), e.id)) if enemies else None
            if enemy is None:
                result += [0.0] * 11
                continue
            pos = enemy.snake[0]
            stationary = enemy.type == "ambush" and enemy.mode in {"waiting", "recover"} or self.steps < enemy.stunned_until
            velocity = 0.0 if stationary else 1 / enemy.interval
            result += [1.0, (pos.x - head.x) / n, (pos.y - head.y) / n, (pos.z - head.z) / n,
                       enemy.direction.x * velocity, enemy.direction.y * velocity, enemy.direction.z * velocity,
                       min(1.0, max(0, enemy.next_move - self.steps) / 60),
                       float(enemy.alert), min(_distance(head, p) for p in enemy.snake) / (3 * n),
                       _distance(head, pos) / (3 * n)]
        gate = min(self.gates, key=lambda g: (min(_distance(head, p) for p in g.cells), g.id)) if self.gates else None
        if gate:
            pos = min(gate.cells, key=lambda p: (_distance(head, p), p.x, p.z))
            result += [1.0, (pos.x - head.x) / n, (pos.y - head.y) / n, (pos.z - head.z) / n,
                       float(gate.state == "closed"), float(gate.state == "warning"),
                       min(1.0, gate.ticks_until_change / gate.period),
                       float(gate.state == "closed" or gate.state == "warning" and gate.ticks_until_change <= 1)]
        else:
            result += [0.0] * 8
        for kind in ("moving", "maze"):
            walls = [w for w in self.moving_walls if w.type == kind]
            wall = min(walls, key=lambda w: (min(_distance(head, p) for p in w.cells), w.id)) if walls else None
            if wall:
                pos = min(wall.cells, key=lambda p: (_distance(head, p), p.x, p.z))
                next_pos = min(wall.next_cells, key=lambda p: (_distance(head, p), p.x, p.z))
                remaining = max(0, wall.next_move - self.steps)
                result += [1.0, (pos.x - head.x) / n, (pos.y - head.y) / n, (pos.z - head.z) / n,
                           (next_pos.x - head.x) / n, (next_pos.y - head.y) / n, (next_pos.z - head.z) / n,
                           min(1.0, remaining / wall.interval), float(remaining <= wall.warning_ticks)]
            else:
                result += [0.0] * 9
        threat = self._threat()
        clearance = min(head.x - self.zone_inset, head.y - self.zone_inset, head.z - self.zone_inset,
                        self.grid_size - 1 - self.zone_inset - head.x,
                        self.grid_size - 1 - self.zone_inset - head.y,
                        self.grid_size - 1 - self.zone_inset - head.z)
        result += [
            self.zone_inset / max(1, self.grid_size // 2),
            int(zone["nextInset"]) / max(1, self.grid_size // 2),
            min(1.0, int(zone["ticksUntilShrink"]) / int(self.config["zone_interval"])),
            float(zone["warning"]), float(bool(self.config["abilities"]) and not self.dash_cooldown),
            self.dash_cooldown / int(ABILITY_CONFIG["dash_cooldown"]),
            float(bool(self.config["abilities"]) and not self.shield_cooldown),
            self.shield_cooldown / int(ABILITY_CONFIG["shield_cooldown"]),
            float(bool(self.shield_ticks and self.shield_charge)),
            self.shield_ticks / int(ABILITY_CONFIG["shield_duration"]), threat,
            min(1.0, self._recent.count(head) / 4),
            min(1.0, max(0, self.steps - self._last_food_step) / self.step_limit),
            float(self._encounter is not None),
            self._separation_ticks / int(ESCAPE_CONFIG["separation_steps"]),
            min(1.0, max(0, self.next_gold_step - self.steps) / int(self.config["gold_interval"]))
            if self.config["gold"] else 0.0,
            max(0.0, clearance / n),
        ]
        food = self.food
        result += [float(food is not None and
                         (food.x - head.x) * d.x + (food.y - head.y) * d.y + (food.z - head.z) * d.z > 0)
                   for d in DIRECTIONS]
        return [max(-1.0, min(1.0, float(value))) for value in result]

    def board(self) -> dict[str, object]:
        threat = self._threat()
        emotion = ("relieved" if self.done and self.last_death == "time_limit" else
                   "dead" if self.done else "happy" if self.food_collected and self.steps - self._last_food_step <= 3 else
                   "relieved" if self.steps - self._last_escape_step <= 5 else
                   "terrified" if threat >= 0.7 else "alert" if threat >= 0.35 else "safe")
        nearest = min(self.enemies, key=lambda e: (_distance(self.snake[0], e.snake[0]), e.id)) if self.enemies else None
        look_target = nearest.snake[0] if nearest and threat >= 0.35 else self.food
        if look_target is None:
            look_target = _add(self.snake[0], self.direction)
        plans, intents = self._enemy_navigation()
        return {
            "gridSize": self.grid_size, "blockSize": 1, "seed": self.seed,
            "environmentVersion": ENVIRONMENT_VERSION,
            "rewardVersion": REWARD_VERSION, "trainingReward": deepcopy(self.last_reward),
            "arena": deepcopy(self.arena),
            "level": self.level, "levelName": str(self.config["name"]), "fullSurvival": self.full_survival,
            "snake": [_json_point(p) for p in self.snake], "direction": _json_point(self.direction),
            "obstacles": [_json_point(p) for p in self.obstacles],
            "obstacleChange": {
                "warning": True, "ticksUntilChange": max(0, self.obstacle_change.change_at - self.steps),
                "addedCells": [_json_point(p) for p in self.obstacle_change.added],
                "removedCells": [_json_point(p) for p in self.obstacle_change.removed],
            } if self.obstacle_change else None,
            "food": _json_point(self.food) if self.food else None,
            "foods": [{"id": f.id, "type": f.type, "position": _json_point(f.position), "value": f.value,
                       "ttl": None if f.expires_at is None else max(0, f.expires_at - self.steps)}
                      for f in self.foods],
            "enemies": [{
                "id": e.id, "type": e.type, "snake": [_json_point(p) for p in e.snake],
                "direction": _json_point(e.direction), "nextMoveIn": max(0, e.next_move - self.steps),
                "nextPosition": _json_point(plans[e.id]), "detectionRange": e.detection_range,
                "alert": e.alert, "mode": e.mode, "resources": e.resources,
                "navigation": "A*" if e.type in ("hunter", "rival", "ambush") else "patrol route",
                "moveInterval": e.interval,
                "route": [_json_point(p) for p in e.route],
                "intent": intents[e.id],
            } for e in self.enemies],
            "gates": [{"id": g.id, "cells": [_json_point(p) for p in g.cells],
                       "state": "active" if g.state == "closed" else g.state,
                       "ticksUntilChange": g.ticks_until_change} for g in self.gates],
            "movingWalls": [{
                "id": w.id, "type": w.type, "cells": [_json_point(p) for p in w.cells],
                "nextCells": [_json_point(p) for p in w.next_cells],
                "nextMoveIn": max(0, w.next_move - self.steps),
                "warning": w.next_move - self.steps <= w.warning_ticks,
            } for w in self.moving_walls],
            "zone": self._zone_snapshot(),
            "abilities": {
                "dash": {"ready": bool(self.config["abilities"]) and self.dash_cooldown == 0,
                         "cooldown": self.dash_cooldown, "active": any(
                             e["step"] == self.steps and e["type"] == "dash" for e in self.events)},
                "shield": {"ready": bool(self.config["abilities"]) and self.shield_cooldown == 0,
                           "cooldown": self.shield_cooldown,
                           "active": bool(self.shield_ticks and self.shield_charge),
                           "ticks": self.shield_ticks, "charges": self.shield_charge},
            },
            "events": deepcopy(list(self.events)), "contact": deepcopy(self.contact),
            "emotion": emotion, "threat": threat,
            "lookTarget": _json_point(look_target), "steps": self.steps, "score": self.score,
            "foodCollected": self.food_collected, "enemyEscapes": self.enemy_escapes,
            "goldCollected": self.gold_collected, "rivalFoodCollected": self.rival_food_collected,
            "dangerExposure": self.danger_exposure,
            "done": self.done, "lastDeath": self.last_death,
            "deathPoint": _json_point(self.death_point) if self.death_point else None,
        }
