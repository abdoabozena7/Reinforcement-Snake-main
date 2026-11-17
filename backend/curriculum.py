from __future__ import annotations

import copy
import math

from backend.config import DIFFICULTIES


class Curriculum:
    """Promotion is evidence-driven and is applied only at an episode boundary."""

    def __init__(self, level: int = 3, enabled: bool = False) -> None:
        self.level = self._level(level)
        self.enabled = bool(enabled)
        self.pending_level: int | None = None
        self.last_passed = False
        self.reason = "Manual difficulty. Choose curriculum to start at level 1."

    @staticmethod
    def _level(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value not in DIFFICULTIES:
            raise ValueError("Difficulty must be an integer from 1 to 6.")
        return value

    @staticmethod
    def goals(level: int) -> dict[str, int]:
        configuration = DIFFICULTIES[level]
        supplied = configuration.get("evaluationGoals", {})
        return {
            "food": int(supplied.get("food", 1)),
            "survivalSteps": int(supplied.get("survivalSteps", 60 + 15 * (level - 1))),
            "escapes": int(supplied.get("escapes", 0)),
        }

    @staticmethod
    def successful(result: dict[str, object], goals: dict[str, int]) -> bool:
        return (
            int(result.get("food", 0)) >= goals["food"]
            and int(result.get("steps", 0)) >= goals["survivalSteps"]
            and int(result.get("enemyEscapes", 0)) >= goals["escapes"]
        )

    def configure(self, level: int, enabled: bool) -> None:
        self.level = self._level(level)
        self.enabled = bool(enabled)
        self.pending_level = None
        self.last_passed = False
        self.reason = (
            "Awaiting a held-out evaluation suite; episode count never unlocks a level."
            if self.enabled
            else "Manual difficulty; automatic promotion is disabled."
        )

    def consider(self, evaluation: dict[str, object]) -> bool:
        if not self.enabled or evaluation.get("fullSurvival") or evaluation.get("level") != self.level:
            return False
        self.pending_level = None
        minimum = int(DIFFICULTIES[self.level].get("evaluationMinEpisodes", 8))
        threshold = float(DIFFICULTIES[self.level].get("evaluationThreshold", 0.7))
        results = evaluation.get("results", [])
        seeds = evaluation.get("seeds", [])
        episodes = evaluation.get("episodes")
        if (
            not isinstance(results, list)
            or not isinstance(seeds, list)
            or isinstance(episodes, bool)
            or not isinstance(episodes, int)
            or episodes < minimum
            or episodes != len(results)
            or episodes != len(seeds)
            or len(set(seeds)) != episodes
        ):
            self.last_passed = False
            self.reason = f"Need at least {minimum} distinct held-out episodes at this difficulty."
            return False
        goals = self.goals(self.level)
        successes = sum(self.successful(result, goals) for result in results)
        if evaluation.get("successes") != successes:
            self.last_passed = False
            self.reason = "Evaluation totals do not match the recorded episode outcomes."
            return False
        required = math.ceil(threshold * episodes)
        self.last_passed = successes >= required
        if self.last_passed:
            if self.level < max(DIFFICULTIES):
                self.pending_level = self.level + 1
                self.reason = (
                    f"Held-out goals passed ({successes}/{episodes}); "
                    f"level {self.pending_level} begins at the next episode boundary."
                )
            else:
                self.reason = f"Master Mode goals passed on {successes}/{episodes} held-out episodes."
        else:
            self.reason = (
                f"Held-out goals passed on {successes}/{episodes}; "
                f"need {required}/{episodes}. Continue training."
            )
        return self.last_passed

    def apply_pending(self) -> bool:
        if not self.enabled or self.pending_level is None:
            return False
        self.level = self.pending_level
        self.pending_level = None
        self.last_passed = False
        self.reason = f"Promoted to {DIFFICULTIES[self.level]['name']} after measured evaluation."
        return True

    def snapshot(self) -> dict[str, object]:
        configuration = DIFFICULTIES[self.level]
        return {
            "enabled": self.enabled,
            "level": self.level,
            "currentLevel": self.level,
            "levelName": configuration["name"],
            "pendingLevel": self.pending_level,
            "goals": self.goals(self.level),
            "minEvaluationEpisodes": int(configuration.get("evaluationMinEpisodes", 8)),
            "successThreshold": float(configuration.get("evaluationThreshold", 0.7)),
            "evaluationPassed": self.last_passed,
            "reason": self.reason,
            "levels": [
                {"level": level, "name": values["name"], "goals": self.goals(level)}
                for level, values in sorted(DIFFICULTIES.items())
            ],
        }

    def checkpoint(self) -> dict[str, object]:
        return {
            "level": self.level,
            "enabled": self.enabled,
            "pendingLevel": self.pending_level,
            "lastPassed": self.last_passed,
            "reason": self.reason,
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, object]) -> Curriculum:
        level = cls._level(data["level"])
        if not isinstance(data.get("enabled"), bool) or not isinstance(data.get("lastPassed"), bool):
            raise ValueError("Invalid curriculum checkpoint.")
        pending = data.get("pendingLevel")
        if pending is not None and (
            isinstance(pending, bool) or not isinstance(pending, int)
            or level == 6 or pending != level + 1 or not data["enabled"]
        ):
            raise ValueError("Invalid pending curriculum promotion.")
        if not isinstance(data.get("reason"), str):
            raise ValueError("Invalid curriculum reason.")
        curriculum = cls(level, data["enabled"])
        curriculum.pending_level = pending
        curriculum.last_passed = data["lastPassed"]
        curriculum.reason = copy.copy(data["reason"])
        return curriculum
