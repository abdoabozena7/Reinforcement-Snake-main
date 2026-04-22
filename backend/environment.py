from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


class SnakeEnvironment:
    def __init__(
        self,
        grid_size: int = 20,
        block_size: int = 20,
        step_limit: int = 500,
        obstacle_count: int = 8,
    ) -> None:
        self.grid_size = grid_size
        self.block_size = block_size
        self.step_limit = step_limit
        self.obstacle_count = obstacle_count
        self.reset()

    def reset(self) -> list[float]:
        self.snake = [Point(10, 10), Point(10, 11), Point(10, 12)]
        self.direction = Point(0, -1)
        self.obstacles = self._spawn_obstacles()
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.last_death: str | None = None
        self.death_point: Point | None = None
        return self.get_state()

    def _spawn_obstacles(self) -> list[Point]:
        obstacles: list[Point] = []
        forbidden = {
            (part.x, part.y)
            for part in self.snake
        }
        forbidden.update(
            {
                (x, y)
                for x in range(7, 14)
                for y in range(7, 14)
            }
        )

        while len(obstacles) < self.obstacle_count:
            obstacle = Point(
                random.randrange(self.grid_size),
                random.randrange(self.grid_size),
            )
            if (obstacle.x, obstacle.y) in forbidden:
                continue
            if any(item.x == obstacle.x and item.y == obstacle.y for item in obstacles):
                continue
            obstacles.append(obstacle)
        return obstacles

    def _spawn_food(self) -> Point:
        while True:
            food = Point(
                random.randrange(self.grid_size),
                random.randrange(self.grid_size),
            )
            if any(part.x == food.x and part.y == food.y for part in self.snake):
                continue
            if any(obstacle.x == food.x and obstacle.y == food.y for obstacle in self.obstacles):
                continue
            return food

    def get_state(self) -> list[float]:
        head = self.snake[0]
        return [
            float(self._is_hazard(head.x, head.y - 1)),
            float(self._is_hazard(head.x, head.y + 1)),
            float(self._is_hazard(head.x - 1, head.y)),
            float(self._is_hazard(head.x + 1, head.y)),
            1.0 if self.food.y < head.y else 0.0,
            1.0 if self.food.y > head.y else 0.0,
            1.0 if self.food.x < head.x else 0.0,
            1.0 if self.food.x > head.x else 0.0,
        ]

    def _is_hazard(self, x: int, y: int) -> int:
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return 1
        if any(part.x == x and part.y == y for part in self.snake):
            return 1
        return 1 if any(obstacle.x == x and obstacle.y == y for obstacle in self.obstacles) else 0

    def step(self, action_index: int) -> dict[str, object]:
        directions = [
            Point(0, -1),
            Point(0, 1),
            Point(-1, 0),
            Point(1, 0),
        ]
        new_direction = directions[action_index]
        is_reverse = (
            len(self.snake) > 1
            and new_direction.x == -self.direction.x
            and new_direction.y == -self.direction.y
        )
        if not is_reverse:
            self.direction = new_direction

        head = self.snake[0]
        new_head = Point(head.x + self.direction.x, head.y + self.direction.y)

        reward = 0.0
        done = False

        hit_wall = (
            new_head.x < 0
            or new_head.x >= self.grid_size
            or new_head.y < 0
            or new_head.y >= self.grid_size
        )
        hit_self = not hit_wall and any(
            part.x == new_head.x and part.y == new_head.y for part in self.snake
        )
        hit_obstacle = not hit_wall and any(
            obstacle.x == new_head.x and obstacle.y == new_head.y for obstacle in self.obstacles
        )

        if hit_wall or hit_self or hit_obstacle:
            if hit_wall:
                self.last_death = "wall"
            elif hit_obstacle:
                self.last_death = "obstacle"
            else:
                self.last_death = "self"
            self.death_point = new_head
            reward = -15.0 if hit_obstacle else -10.0
            if hit_obstacle and self.score > 0:
                self.score //= 2
            done = True
        else:
            self.snake.insert(0, new_head)
            if new_head.x == self.food.x and new_head.y == self.food.y:
                reward = 10.0
                self.score += 1
                self.food = self._spawn_food()
            else:
                self.snake.pop()
                distance = abs(new_head.x - self.food.x) + abs(new_head.y - self.food.y)
                reward = 1.0 if distance < 5 else 0.0

        self.steps += 1
        if self.steps > self.step_limit:
            done = True

        return {
            "next_state": self.get_state(),
            "reward": reward,
            "done": done,
        }

    def board(self) -> dict[str, object]:
        return {
            "gridSize": self.grid_size,
            "blockSize": self.block_size,
            "snake": [{"x": part.x, "y": part.y} for part in self.snake],
            "obstacles": [{"x": item.x, "y": item.y} for item in self.obstacles],
            "food": {"x": self.food.x, "y": self.food.y},
            "direction": {"x": self.direction.x, "y": self.direction.y},
            "lastDeath": self.last_death,
            "deathPoint": (
                {"x": self.death_point.x, "y": self.death_point.y}
                if self.death_point is not None
                else None
            ),
        }
