from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from backend.agent import CHECKPOINT_SCHEMA, DQNAgent, Experience, ReplayBuffer, _integer
from backend.champion import (
    CHAMPION_CHECK_INTERVAL, CHAMPION_SEEDS, ChampionManager, evaluation_rollout,
)
from backend.config import DIFFICULTIES, ENVIRONMENT_VERSION, MODEL_VERSION, REWARD_VERSION, SIMULATION_DT
from backend.curriculum import Curriculum
from backend.environment import ACTION_LABELS, ACTION_SIZE, OBSERVATION_SIZE, SnakeEnvironment

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.environ.get("SNAKE_STORAGE_DIR", str(BASE_DIR / "storage")))
READY_MODEL_FILE = BASE_DIR / "storage" / "pretrained_survival.npz"
HISTORY_LIMIT = 240
EVALUATION_HISTORY_LIMIT = 40
MAX_BATCH_STEPS = 2048
HELD_OUT_SEED_BASE = 2**31


def _board_hash(board: dict[str, object]) -> str:
    encoded = json.dumps(board, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _seed(value: object) -> int:
    return _integer(value, "seed (expected 0..4294967295)", 0, 2**32 - 1)


class TrainingSession:
    """Keep replayable runs separate across arena semantics, with compatible v3 policies."""

    def __init__(
        self, storage_dir: Path | str | None = None, seed: int = 0, load_saved: bool = True
    ) -> None:
        self.lock = threading.RLock()
        root = Path(storage_dir) if storage_dir is not None else STORAGE_DIR
        self.storage_dir = root / MODEL_VERSION / ENVIRONMENT_VERSION
        self.model_file = self.storage_dir / "latest.npz"
        self.manual_model_file = self.storage_dir / "manual.npz"
        self.best_model_file = self.storage_dir / "best.npz"
        self.batch_size = 64
        self.min_buffer_for_training = 128
        self.train_every = 4
        self.save_every_steps = 5_000
        self._advance_epoch = 0
        self._champion = ChampionManager(self.storage_dir)
        self._initialize(_seed(seed), level=4)
        if load_saved:
            for candidate in (self.model_file, self.manual_model_file, self.best_model_file):
                if candidate.exists() and self.load_checkpoint(candidate):
                    self.level, self.full_survival, self.mode = 4, False, "train"
                    self.curriculum.configure(4, False)
                    self.paused, self.intro_active = True, False
                    self._start_new_episode()
                    self.log_message = (
                        f"Resumed {MODEL_VERSION} learner; paused Hunter and Rival presentation. "
                        "Older checkpoints remain untouched."
                    )
                    break

    def _initialize(self, seed: int, level: int = 4) -> None:
        if seed in CHAMPION_SEEDS:
            raise ValueError("This seed is reserved for held-out best-policy evaluation.")
        self.seed = seed
        self.level, self.full_survival = level, False
        self.mode = "train"
        self.paused, self.intro_active = True, False
        self.rng = np.random.default_rng(seed + 101)
        self.agent = DQNAgent(self.model_file, seed=seed)
        self.buffer = ReplayBuffer(seed=seed + 202)
        self.curriculum = Curriculum(level, False)
        self.completed_episodes = 0
        self.current_episode = 0
        self.high_score = 0
        self.total_steps = 0
        self.training_steps = 0
        self._reset_best_schedule()
        self._policy_source: dict[str, object] = {
            "source": "random", "seed": seed, "policyHash": self.agent.parameter_hash(),
        }
        self.training_seeds: set[int] = set()
        self._prior_held_out_seeds: set[int] = set()
        self.chart_points: list[dict[str, object]] = []
        self.score_history: list[int] = []
        self.evaluation_history: list[dict[str, object]] = []
        self.evaluation: dict[str, object] | None = None
        self.pending_reset = False
        self.transitions_per_second = 0.0
        self.last_action: int | None = None
        self.last_exploring = False
        self._last_chart_point: dict[str, object] | None = None
        self._best_context_scores: dict[str, int] = {}
        self._start_new_episode(seed)
        self.log_message = (
            f"Paused 3D Hunter and Rival arena. Randomly initialized {MODEL_VERSION} Double DQN. "
            "Use Load Ready Weights for pretrained evaluation playback."
        )

    @property
    def auto_curriculum(self) -> bool:
        return self.curriculum.enabled

    @property
    def held_out_seeds(self) -> set[int]:
        seeds = self._prior_held_out_seeds.union(CHAMPION_SEEDS)
        if self.evaluation is not None:
            seeds.update(self.evaluation.get("reservedSeeds", self.evaluation["seeds"]))
        return seeds

    def _next_episode_seed(self) -> int:
        candidate = int(self.rng.integers(0, HELD_OUT_SEED_BASE))
        while candidate in self.held_out_seeds:
            candidate = int(self.rng.integers(0, HELD_OUT_SEED_BASE))
        return candidate

    def _start_new_episode(self, seed: int | None = None) -> None:
        if self.curriculum.apply_pending():
            self.level = self.curriculum.level
            self.full_survival = False
        self.episode_seed = self._next_episode_seed() if seed is None else seed
        self.env = SnakeEnvironment(
            seed=self.episode_seed, level=self.level, full_survival=self.full_survival
        )
        self.current_state = self.env.get_state()
        self.current_episode += 1
        self.pending_reset = False
        self._episode_actions: list[int] = []
        self._episode_modes: set[str] = set()
        self.agent.last_decision = None
        self.last_action = None
        self.last_exploring = False
        if self.mode == "train":
            self.training_seeds.add(self.episode_seed)

    def _recent_points(self) -> list[dict[str, object]]:
        return [
            point
            for point in self.chart_points
            if point["mode"] == self.mode
            and point["level"] == self.level
            and point["fullSurvival"] == self.full_survival
            and not point.get("mixedMode", False)
        ][-20:]

    def _average(self, key: str) -> float:
        points = self._recent_points()
        return sum(float(point[key]) for point in points) / len(points) if points else 0.0

    def _snapshot(self, include_network: bool = True) -> dict[str, object]:
        network = (
            self.agent.network_snapshot(self.current_state, self.env.action_mask())
            if include_network
            else None
        )
        cached = self.agent.last_decision
        last_action = cached["lastAction"] if cached is not None else None
        exploring = bool(cached["isExploring"]) if cached is not None else False
        return {
            "episode": self.current_episode,
            "episodeKey": f"{ENVIRONMENT_VERSION}:{self.current_episode}:{self.episode_seed}",
            "completedEpisodes": self.completed_episodes,
            "highScore": self.high_score,
            "epsilon": 0.0 if self.mode == "eval" else self.agent.epsilon,
            "policyEpsilon": self.agent.epsilon,
            "currentScore": self.env.score,
            "avgScore": round(self._average("score"), 4),
            "bufferSize": self.buffer.size(),
            "logMessage": self.log_message,
            "board": self.env.board(),
            "network": network,
            "episodeEnded": self.pending_reset,
            "delayMs": 1200 if self.pending_reset else 0,
            "chartPoint": copy.deepcopy(self._last_chart_point),
            "chartPoints": copy.deepcopy(self.chart_points),
            "totalSteps": self.total_steps,
            "trainingSteps": self.training_steps,
            "gradientSteps": self.agent.train_steps,
            "loss": self.agent.last_loss,
            "paused": self.paused,
            "introActive": self.intro_active,
            "mode": self.mode,
            "level": self.level,
            "levelName": DIFFICULTIES[self.level]["name"],
            "autoCurriculum": self.auto_curriculum,
            "fullSurvival": self.full_survival,
            "foodAverage": round(self._average("food"), 4),
            "survivalSeconds": round(self.env.steps * SIMULATION_DT, 4),
            "avgSurvival": round(self._average("survivalSeconds"), 4),
            "enemyEscapes": self.env.enemy_escapes,
            "deathCause": self.env.last_death,
            "evaluation": copy.deepcopy(self.evaluation),
            "evaluationHistory": copy.deepcopy(self.evaluation_history),
            "curriculum": self.curriculum.snapshot(),
            "transitionsPerSecond": round(self.transitions_per_second, 2),
            "modelVersion": MODEL_VERSION,
            "environmentVersion": ENVIRONMENT_VERSION,
            "rewardVersion": REWARD_VERSION,
            "readyWeightsAvailable": READY_MODEL_FILE.is_file(),
            "bestPolicy": self._champion.snapshot(),
            "lastAction": last_action,
            "actionLabel": ACTION_LABELS[last_action] if last_action is not None else "No decision yet",
            "isExploring": exploring,
            "metricWindow": 20,
            "episodeModes": sorted(self._episode_modes),
            "escapeDefinition": (
                "6 consecutive steps at least 8 cells from a hunter/ambush encountered "
                "within 4 cells, with no other pursuer within 4; 60-step encounter cooldown."
            ),
        }

    def get_state(self, include_network: bool = True) -> dict[str, object]:
        with self.lock:
            return self._snapshot(include_network)

    def _reset_best_schedule(self) -> None:
        self._best_check_training_steps = self.training_steps
        self._best_check_gradient_steps = self.agent.train_steps

    def _queue_best_policy(self) -> None:
        if self._champion.queue(
            self.agent, self.training_seeds,
            {
                "trainingSteps": self.training_steps, "gradientSteps": self.agent.train_steps,
                "totalSteps": self.total_steps,
            },
            self._policy_source, READY_MODEL_FILE,
        ):
            self._reset_best_schedule()

    def check_best_policy(self) -> dict[str, object]:
        """Queue a bounded frozen-policy check without pausing or replacing this learner."""
        with self.lock:
            self._queue_best_policy()
            return self._snapshot()

    def close(self) -> None:
        """Cooperatively cancel and join the background best-policy worker."""
        self._champion.close()

    @staticmethod
    def _bounded_steps(steps: int) -> int:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer.")
        return min(steps, MAX_BATCH_STEPS)

    def _finish_episode(self) -> None:
        self.completed_episodes += 1
        self.high_score = max(self.high_score, self.env.score)
        point: dict[str, object] = {
            "episode": self.current_episode,
            "seed": self.episode_seed,
            "score": self.env.score,
            "epsilon": 0.0 if self.mode == "eval" else self.agent.epsilon,
            "food": self.env.food_collected,
            "survivalSeconds": round(self.env.steps * SIMULATION_DT, 4),
            "steps": self.env.steps,
            "enemyEscapes": self.env.enemy_escapes,
            "deathCause": self.env.last_death,
            "mode": self.mode,
            "episodeModes": sorted(self._episode_modes),
            "mixedMode": len(self._episode_modes) > 1,
            "level": self.level,
            "levelName": DIFFICULTIES[self.level]["name"],
            "fullSurvival": self.full_survival,
            "environmentVersion": ENVIRONMENT_VERSION,
            "trainingSteps": self.training_steps,
            "window": 20,
        }
        recent = self._recent_points()[-19:]
        point["avgScore"] = round(
            (sum(float(item["score"]) for item in recent) + self.env.score) / (len(recent) + 1),
            4,
        )
        self.chart_points.append(point)
        del self.chart_points[:-HISTORY_LIMIT]
        self.score_history.append(self.env.score)
        del self.score_history[:-HISTORY_LIMIT]
        self._last_chart_point = point
        self.pending_reset = True
        self.log_message = (
            f"{self.mode.capitalize()} episode {self.current_episode}: "
            f"{self.env.food_collected} food, {self.env.steps} steps, "
            f"{self.env.last_death or 'completed'}."
        )
        context = f"{self.level}:{int(self.full_survival)}"
        if (
            self.mode == "train"
            and not point["mixedMode"]
            and self.env.score > self._best_context_scores.get(context, 0)
        ):
            self._best_context_scores[context] = self.env.score
            try:
                self._save_checkpoint(self.best_model_file, "best-training-score")
            except Exception as error:
                self.log_message += f" Checkpoint not written: {error}"

    def _advance(self, steps: int) -> dict[str, object]:
        requested_epoch = self._advance_epoch
        was_blocked = self.intro_active or self.paused
        count = self._bounded_steps(steps)
        with self.lock:
            # Requests that queued during an intro stay invalid even after it is unlocked.
            if (
                was_blocked
                or requested_epoch != self._advance_epoch
                or self.intro_active
                or self.paused
            ):
                return self._snapshot()
            started = time.perf_counter()
            for index in range(count):
                if self.pending_reset:
                    self._start_new_episode()
                state = self.current_state
                mask = self.env.action_mask()
                action = self.agent.act(
                    state,
                    mask,
                    explore=self.mode == "train",
                    capture_decision=index == count - 1,
                )
                if index == count - 1 and self.agent.last_decision is not None:
                    self.agent.last_decision["decisionStep"] = self.env.steps
                    self.agent.last_decision["decisionEpisode"] = self.current_episode
                    self.agent.last_decision["decisionSeed"] = self.episode_seed
                    self.agent.last_decision["decisionMode"] = self.mode
                    self.agent.last_decision["decisionTrainingSteps"] = self.training_steps
                transition = self.env.step(action)
                self._episode_actions.append(action)
                self._episode_modes.add(self.mode)
                self.total_steps += 1
                self.last_action = action
                self.last_exploring = bool(
                    self.agent.last_decision and self.agent.last_decision["isExploring"]
                )
                self.current_state = list(transition["next_state"])
                if self.mode == "train":
                    self.training_seeds.add(self.episode_seed)
                    self.buffer.add(
                        Experience(
                            state,
                            action,
                            float(transition["reward"]),
                            self.current_state,
                            bool(transition["done"]),
                            mask,
                            self.env.action_mask(),
                        )
                    )
                    self.training_steps += 1
                    self.agent.record_training_step()
                    if (
                        self.buffer.size() >= self.min_buffer_for_training
                        and self.training_steps % self.train_every == 0
                    ):
                        self.agent.train_batch(self.buffer.sample(self.batch_size))
                if transition["done"]:
                    self._finish_episode()
                if self.mode == "train" and self.training_steps % self.save_every_steps == 0:
                    try:
                        self._save_checkpoint(self.model_file, "automatic")
                    except Exception as error:
                        self.log_message = f"Training continues; automatic checkpoint failed: {error}"
            elapsed = max(time.perf_counter() - started, 1e-9)
            self.transitions_per_second = count / elapsed
            if (
                self.mode == "train"
                and self.training_steps - self._best_check_training_steps >= CHAMPION_CHECK_INTERVAL
                and self.agent.train_steps > self._best_check_gradient_steps
            ):
                self._queue_best_policy()
            if not self.pending_reset:
                self.log_message = (
                    f"{'Learning' if self.mode == 'train' else 'Greedy evaluation playback'} "
                    f"at {self.transitions_per_second:.0f} transitions/s; "
                    f"level {self.level}, episode {self.current_episode}."
                )
            return self._snapshot()

    def tick(self, steps: int = 1) -> dict[str, object]:
        return self._advance(steps)

    def train_steps(self, steps: int = 256) -> dict[str, object]:
        return self._advance(steps)

    def control(self, options: dict[str, object]) -> dict[str, object]:
        if not isinstance(options, dict):
            raise ValueError("Control options must be an object.")
        with self.lock:
            for key in ("paused", "intro", "autoCurriculum", "fullSurvival", "newEpisode"):
                if key in options and not isinstance(options[key], bool):
                    raise ValueError(f"{key} must be a boolean.")
            mode = options.get("mode", self.mode)
            if mode not in ("train", "eval"):
                raise ValueError("Mode must be 'train' or 'eval'.")
            level = options.get("level", self.level)
            Curriculum._level(level)
            selected_seed = _seed(options["seed"]) if "seed" in options else None
            mutation = any(
                key in options
                for key in ("mode", "level", "autoCurriculum", "fullSurvival", "seed", "newEpisode")
            )
            if self.intro_active and mutation and options.get("intro", True):
                raise ValueError("Cancel the active intro before changing the run.")
            enabled = bool(options.get("autoCurriculum", self.auto_curriculum))
            full = bool(options.get("fullSurvival", self.full_survival))
            if "level" in options and "autoCurriculum" not in options:
                enabled = False
            if enabled and not self.auto_curriculum:
                level, full = 1, False
            if full:
                level, enabled = 6, False
            context_changed = (
                level != self.level or full != self.full_survival or enabled != self.auto_curriculum
            )
            if selected_seed is not None and selected_seed in self.held_out_seeds:
                raise ValueError("This seed is reserved for held-out evaluation; choose a training seed.")
            self._advance_epoch += 1
            self.mode = mode
            self.level, self.full_survival = level, full
            if context_changed:
                self.curriculum.configure(level, enabled)
            if selected_seed is not None:
                self.seed = selected_seed
                self.rng = np.random.default_rng(selected_seed + 101)
            if context_changed or selected_seed is not None or options.get("newEpisode", False):
                self._start_new_episode(selected_seed)
            if "intro" in options:
                self.intro_active = bool(options["intro"])
            if "paused" in options:
                self.paused = bool(options["paused"])
            if self.intro_active:
                self.paused = True
                self.log_message = "Cinematic intro locked: simulation, replay and learning are frozen."
            elif self.paused:
                self.log_message = "Simulation paused; policy and replay are preserved."
            else:
                self.log_message = (
                    "Greedy evaluation playback: exploration and learning are disabled."
                    if self.mode == "eval"
                    else "Training resumed; actions come from the learned/exploring policy."
                )
            return self._snapshot()

    def default_evaluation_seeds(self, episodes: int = 8) -> list[int]:
        base = HELD_OUT_SEED_BASE + (self.seed % 100_000) * 64
        candidates: list[int] = []
        cursor = base
        while len(candidates) < episodes:
            if cursor not in self.training_seeds:
                candidates.append(cursor)
            cursor += 1
        return candidates

    def evaluate(
        self, episodes: int = 8, seeds: list[int] | tuple[int, ...] | None = None
    ) -> dict[str, object]:
        with self.lock:
            _integer(episodes, "evaluation episode count", 1, 32)
            suite = self.default_evaluation_seeds(episodes) if seeds is None else list(seeds)
            if not 1 <= len(suite) <= 32 or len(set(suite)) != len(suite):
                raise ValueError("Use 1..32 distinct evaluation seeds.")
            suite = [_seed(seed) for seed in suite]
            overlap = self.training_seeds.intersection(suite)
            if overlap:
                raise ValueError(f"Evaluation seeds overlap training: {sorted(overlap)}.")
            live_agent = self.agent
            policy = self.agent.clone()
            level, full = self.level, self.full_survival
            training_steps, gradient_steps, total_steps = (
                self.training_steps, self.agent.train_steps, self.total_steps,
            )
            self._prior_held_out_seeds.update(suite)
            reserved_seeds = sorted(self.held_out_seeds)
            goals = Curriculum.goals(level)
        started = time.perf_counter()
        results = [evaluation_rollout(policy, seed, level, full) for seed in suite]
        death_causes: dict[str, int] = {}
        for result in results:
            cause = result["deathCause"] or "completed"
            death_causes[cause] = death_causes.get(cause, 0) + 1
        count = len(results)
        successes = sum(bool(result["success"]) for result in results)
        record: dict[str, object] = {
            "episodes": count,
            "seeds": suite,
            "level": level,
            "levelName": DIFFICULTIES[level]["name"],
            "fullSurvival": full,
            "mode": "eval",
            "epsilon": 0.0,
            "policyHash": policy.parameter_hash(),
            "modelVersion": MODEL_VERSION,
            "environmentVersion": ENVIRONMENT_VERSION,
            "rewardVersion": REWARD_VERSION,
            "trainingSteps": training_steps,
            "gradientSteps": gradient_steps,
            "totalSteps": total_steps,
            "successes": successes,
            "failures": count - successes,
            "successRate": successes / count,
            "meanScore": sum(float(result["score"]) for result in results) / count,
            "meanFood": sum(float(result["food"]) for result in results) / count,
            "meanSteps": sum(int(result["steps"]) for result in results) / count,
            "meanSurvivalSeconds": sum(float(result["survivalSeconds"]) for result in results) / count,
            "meanEscapes": sum(float(result["enemyEscapes"]) for result in results) / count,
            "deathCauses": death_causes,
            "goals": goals,
            "results": results,
            "reservedSeeds": reserved_seeds,
            "seconds": time.perf_counter() - started,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            if self.agent is not live_agent:
                self.log_message = "Evaluation finished for a replaced learner; current run was preserved."
                return self._snapshot()
            record["reservedSeeds"] = sorted(self.held_out_seeds.union(suite))
            self.evaluation = record
            self.evaluation_history.append(record)
            del self.evaluation_history[:-EVALUATION_HISTORY_LIMIT]
            self.curriculum.consider(record)
            return self._snapshot()

    def _session_metadata(self) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        replay, arrays = self.buffer.checkpoint()
        metadata: dict[str, object] = {
            "environmentVersion": ENVIRONMENT_VERSION,
            "rewardVersion": REWARD_VERSION,
            "seed": self.seed,
            "level": self.level,
            "fullSurvival": self.full_survival,
            "mode": self.mode,
            "paused": self.paused,
            "completedEpisodes": self.completed_episodes,
            "currentEpisode": self.current_episode,
            "highScore": self.high_score,
            "totalSteps": self.total_steps,
            "trainingSteps": self.training_steps,
            "chartPoints": self.chart_points,
            "scoreHistory": self.score_history,
            "evaluationHistory": self.evaluation_history,
            "evaluation": self.evaluation,
            "curriculum": self.curriculum.checkpoint(),
            "sessionRngState": copy.deepcopy(self.rng.bit_generator.state),
            "trainingSeeds": sorted(self.training_seeds),
            "policySource": self._policy_source,
            "heldOutSeeds": sorted(self.held_out_seeds),
            "episodeSeed": self.episode_seed,
            "episodeActions": self._episode_actions,
            "episodeModes": sorted(self._episode_modes),
            "pendingReset": self.pending_reset,
            "lastDecision": self.agent.last_decision,
            "boardHash": _board_hash(self.env.board()),
            "bestContextScores": self._best_context_scores,
            "replay": replay,
        }
        return metadata, arrays

    def _save_checkpoint(self, path: Path, label: str) -> None:
        metadata, arrays = self._session_metadata()
        self.agent.save_to(
            path,
            {"label": label, "savedAt": datetime.now(timezone.utc).isoformat(), "session": metadata,
             "environmentVersion": ENVIRONMENT_VERSION, "rewardVersion": REWARD_VERSION},
            arrays,
        )

    def save_checkpoint(self, path: Path | str, label: str = "headless") -> Path:
        with self.lock:
            destination = Path(path)
            self._save_checkpoint(destination, label)
            return destination

    def save_manual_checkpoint(self) -> dict[str, object]:
        with self.lock:
            try:
                self._save_checkpoint(self.manual_model_file, "manual")
                self.log_message = (
                    f"Saved {MODEL_VERSION} policy, target, Adam, RNGs, replay and run "
                    f"at training step {self.training_steps}."
                )
            except Exception as error:
                self.log_message = f"Checkpoint save failed; previous file preserved: {error}"
            return self._snapshot()

    @staticmethod
    def _validated_decision(
        decision: object, agent: DQNAgent
    ) -> dict[str, object] | None:
        if decision is None:
            return None
        if not isinstance(decision, dict) or decision.get("dimensions") != agent.dimensions:
            raise ValueError("Invalid saved last-decision network.")
        layers, weights = decision.get("layers", []), decision.get("weights", [])
        indices = decision.get("sampleIndices", [])
        if len(layers) != 4 or len(weights) != 3 or len(indices) != 4:
            raise ValueError("Invalid saved network layers.")
        sizes = [min(20, size) for size in agent.dimensions]
        if any(len(layer) != size for layer, size in zip(layers, sizes)):
            raise ValueError("Invalid saved network sampling.")
        for index, matrix in enumerate(weights):
            if len(matrix) != sizes[index] or any(len(row) != sizes[index + 1] for row in matrix):
                raise ValueError("Invalid saved network weights.")
            if not np.isfinite(np.asarray(matrix, dtype=np.float64)).all():
                raise ValueError("Non-finite saved network weights.")
        for size, index, dimension in zip(sizes, indices, agent.dimensions):
            if len(index) != size or any(
                isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < dimension
                for item in index
            ):
                raise ValueError("Invalid saved network sample indexes.")
        if len(decision.get("qValues", [])) != ACTION_SIZE or len(decision.get("actionMask", [])) != ACTION_SIZE:
            raise ValueError("Invalid saved action visualization.")
        if any(not np.isfinite(np.asarray(layer, dtype=np.float64)).all() for layer in layers):
            raise ValueError("Non-finite saved network activations.")
        if not np.isfinite(np.asarray(decision["qValues"], dtype=np.float64)).all():
            raise ValueError("Non-finite saved Q values.")
        action = _integer(decision.get("lastAction"), "last action", 0, ACTION_SIZE - 1)
        if (
            any(not isinstance(value, bool) for value in decision["actionMask"])
            or not decision["actionMask"][action]
            or not isinstance(decision.get("isExploring"), bool)
            or decision.get("inputLabels") != [agent.input_labels[index] for index in indices[0]]
            or decision.get("outputLabels") != [agent.output_labels[index] for index in indices[-1]]
        ):
            raise ValueError("Saved action/labels do not align with the decision.")
        return copy.deepcopy(decision)

    @staticmethod
    def _number(value: object, name: str, minimum: float = 0.0, maximum: float = 1e15) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"Invalid saved {name}.")
        return float(value)

    @classmethod
    def _validate_chart_point(cls, point: object) -> None:
        if not isinstance(point, dict):
            raise ValueError("Invalid saved chart point.")
        _integer(point.get("episode"), "chart episode", 1)
        _integer(point.get("score"), "chart score", 0)
        _integer(point.get("food"), "chart food", 0)
        _integer(point.get("trainingSteps"), "chart budget", 0)
        Curriculum._level(point.get("level"))
        cls._number(point.get("survivalSeconds"), "chart survival")
        cls._number(point.get("avgScore"), "chart moving average")
        cls._number(point.get("epsilon"), "chart epsilon", 0.0, 1.0)
        if (
            point.get("mode") not in ("train", "eval")
            or not isinstance(point.get("fullSurvival"), bool)
            or not isinstance(point.get("mixedMode"), bool)
            or (point["fullSurvival"] and point["level"] != 6)
        ):
            raise ValueError("Invalid saved chart context.")

    @classmethod
    def _validate_evaluation(cls, record: object) -> None:
        if not isinstance(record, dict):
            raise ValueError("Invalid saved evaluation.")
        count = _integer(record.get("episodes"), "evaluation suite size", 1, 32)
        level = Curriculum._level(record.get("level"))
        seeds, results, goals = record.get("seeds"), record.get("results"), record.get("goals")
        if (
            not isinstance(seeds, list) or not isinstance(results, list)
            or len(seeds) != count or len(results) != count or len(set(seeds)) != count
            or not isinstance(goals, dict)
            or not isinstance(record.get("fullSurvival"), bool)
            or record.get("mode") != "eval" or record.get("epsilon") != 0.0
            or record.get("modelVersion") != MODEL_VERSION
            or (record["fullSurvival"] and level != 6)
        ):
            raise ValueError("Invalid saved evaluation context.")
        for key in ("food", "survivalSteps", "escapes"):
            _integer(goals.get(key), "evaluation goal", 0)
        for seed, result in zip(seeds, results):
            _seed(seed)
            if not isinstance(result, dict) or result.get("seed") != seed:
                raise ValueError("Invalid saved evaluation result attribution.")
            for key in ("score", "food", "steps", "enemyEscapes"):
                _integer(result.get(key), f"evaluation {key}", 0)
            cls._number(result.get("survivalSeconds"), "evaluation survival")
            if (
                not isinstance(result.get("success"), bool)
                or result["success"] != Curriculum.successful(result, goals)
            ):
                raise ValueError("Evaluation success does not match its recorded goals.")
        if "reservedSeeds" in record:
            if not isinstance(record["reservedSeeds"], list):
                raise ValueError("Invalid held-out seed registry.")
            for seed in record["reservedSeeds"]:
                _seed(seed)
            if not set(seeds).issubset(record["reservedSeeds"]):
                raise ValueError("Incomplete held-out seed registry.")
        successes = sum(result["success"] for result in results)
        if record.get("successes") != successes or record.get("failures") != count - successes:
            raise ValueError("Invalid saved evaluation totals.")
        rate = cls._number(record.get("successRate"), "evaluation success rate", 0.0, 1.0)
        if abs(rate - successes / count) > 1e-9:
            raise ValueError("Invalid saved evaluation rate.")
        for key, result_key in (
            ("meanFood", "food"), ("meanScore", "score"),
            ("meanSurvivalSeconds", "survivalSeconds"), ("meanEscapes", "enemyEscapes"),
        ):
            mean = cls._number(record.get(key), key)
            if abs(mean - sum(result[result_key] for result in results) / count) > 1e-5:
                raise ValueError("Evaluation means do not match its episode outcomes.")
        if "meanSteps" in record:
            mean_steps = cls._number(record["meanSteps"], "mean evaluation steps")
            if abs(mean_steps - sum(result["steps"] for result in results) / count) > 1e-5:
                raise ValueError("Evaluation mean steps do not match its episode outcomes.")
        budget = _integer(record.get("trainingSteps"), "evaluation training budget", 0)
        _integer(record.get("totalSteps"), "evaluation total budget", budget)
        _integer(record.get("gradientSteps"), "evaluation gradient budget", 0)
        policy_hash = record.get("policyHash")
        if (
            not isinstance(policy_hash, str) or len(policy_hash) != 64
            or any(character not in "0123456789abcdef" for character in policy_hash)
        ):
            raise ValueError("Invalid evaluation policy identity.")

    def _prepare_checkpoint(self, path: Path) -> dict[str, object]:
        metadata, arrays = self.agent.read_checkpoint(path)
        saved = metadata["trainingMetadata"].get("session")
        if not isinstance(saved, dict):
            raise ValueError(f"Checkpoint does not contain a complete {MODEL_VERSION} training session.")
        if saved.get("environmentVersion") != ENVIRONMENT_VERSION:
            raise ValueError(
                f"Checkpoint episode uses {saved.get('environmentVersion')!r}, "
                f"not {ENVIRONMENT_VERSION!r}. "
                "It cannot be replayed as a current session; the saved file is preserved. "
                "Use the compatible ready policy or start fresh training."
            )
        seed, episode_seed = _seed(saved.get("seed")), _seed(saved.get("episodeSeed"))
        level = Curriculum._level(saved.get("level"))
        mode = saved.get("mode")
        if mode not in ("train", "eval"):
            raise ValueError("Invalid saved mode.")
        for key in ("fullSurvival", "paused", "pendingReset"):
            if not isinstance(saved.get(key), bool):
                raise ValueError(f"Invalid saved {key}.")
        full = saved["fullSurvival"]
        if full and level != 6:
            raise ValueError("Full survival checkpoint must be level 6.")
        curriculum = Curriculum.from_checkpoint(saved["curriculum"])
        if curriculum.level != level or (full and curriculum.enabled):
            raise ValueError("Saved curriculum does not match the live difficulty.")
        training_steps = _integer(saved.get("trainingSteps"), "session training steps", 0)
        if training_steps != metadata["trainingSteps"]:
            raise ValueError("Session and learner training counters disagree.")
        total_steps = _integer(saved.get("totalSteps"), "total steps", training_steps)
        completed = _integer(saved.get("completedEpisodes"), "completed episodes", 0)
        episode = _integer(saved.get("currentEpisode"), "episode", 1)
        if episode < completed + (0 if saved["pendingReset"] else 1):
            raise ValueError("Invalid saved episode counters.")
        high_score = _integer(saved.get("highScore"), "high score", 0)
        for key, maximum in (
            ("chartPoints", HISTORY_LIMIT),
            ("scoreHistory", HISTORY_LIMIT),
            ("evaluationHistory", EVALUATION_HISTORY_LIMIT),
        ):
            if not isinstance(saved.get(key), list) or len(saved[key]) > maximum:
                raise ValueError(f"Invalid saved {key}.")
        for point in saved["chartPoints"]:
            self._validate_chart_point(point)
        for score in saved["scoreHistory"]:
            _integer(score, "score history entry", 0)
        for record in saved["evaluationHistory"]:
            self._validate_evaluation(record)
            if record["trainingSteps"] > training_steps or record["totalSteps"] > total_steps:
                raise ValueError("Saved evaluation claims a future training budget.")
        if saved.get("evaluation") is not None:
            self._validate_evaluation(saved["evaluation"])
            if not saved["evaluationHistory"] or saved["evaluation"] != saved["evaluationHistory"][-1]:
                raise ValueError("Latest evaluation does not match its history.")
        training_seeds = {_seed(value) for value in saved["trainingSeeds"]}
        held_out_seeds = {_seed(value) for value in saved["heldOutSeeds"]}.union(CHAMPION_SEEDS)
        if training_seeds & held_out_seeds:
            raise ValueError("Saved training and evaluation seeds overlap.")
        if any(
            not set(record.get("reservedSeeds", record["seeds"])).issubset(held_out_seeds)
            for record in saved["evaluationHistory"]
        ):
            raise ValueError("Saved held-out seed provenance is incomplete.")
        replay_metadata = saved.get("replay", {})
        if (
            replay_metadata.get("stateSize") != OBSERVATION_SIZE
            or replay_metadata.get("actionSize") != ACTION_SIZE
            or replay_metadata.get("size", -1) > training_steps
        ):
            raise ValueError("Replay does not match the learner.")
        buffer = ReplayBuffer.from_checkpoint(replay_metadata, arrays)
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["sessionRngState"]
        environment = SnakeEnvironment(seed=episode_seed, level=level, full_survival=full)
        actions = saved.get("episodeActions")
        if not isinstance(actions, list) or len(actions) > environment.step_limit:
            raise ValueError("Invalid saved episode action history.")
        observation = environment.get_state()
        for action in actions:
            _integer(action, "saved action", 0, ACTION_SIZE - 1)
            if environment.done or not environment.action_mask()[action]:
                raise ValueError("Saved episode contains a terminal or illegal continuation.")
            observation = environment.step(action)["next_state"]
        if environment.done != saved["pendingReset"] or _board_hash(environment.board()) != saved.get("boardHash"):
            raise ValueError("Saved environment does not reproduce under current arena semantics.")
        modes = saved.get("episodeModes")
        if not isinstance(modes, list) or any(value not in ("train", "eval") for value in modes):
            raise ValueError("Invalid saved episode modes.")
        candidate = self.agent.clone()
        candidate.restore_checkpoint(metadata, arrays)
        candidate.model_path = self.model_file
        candidate.last_decision = self._validated_decision(saved.get("lastDecision"), candidate)
        best_context_scores = saved.get("bestContextScores", {})
        if not isinstance(best_context_scores, dict):
            raise ValueError("Invalid best-score metadata.")
        for score in best_context_scores.values():
            _integer(score, "context high score", 0)
        return {
            "agent": candidate,
            "buffer": buffer,
            "rng": rng,
            "env": environment,
            "current_state": list(observation),
            "seed": seed,
            "level": level,
            "full_survival": full,
            "mode": mode,
            "curriculum": curriculum,
            "training_steps": training_steps,
            "total_steps": total_steps,
            "completed_episodes": completed,
            "current_episode": episode,
            "high_score": high_score,
            "training_seeds": training_seeds,
            "_policy_source": {
                "source": "checkpoint", "file": str(path.resolve()),
                "policyHash": candidate.parameter_hash(),
            },
            "_prior_held_out_seeds": held_out_seeds,
            "chart_points": copy.deepcopy(saved["chartPoints"]),
            "score_history": copy.deepcopy(saved["scoreHistory"]),
            "evaluation_history": copy.deepcopy(saved["evaluationHistory"]),
            "evaluation": copy.deepcopy(saved.get("evaluation")),
            "episode_seed": episode_seed,
            "_episode_actions": list(actions),
            "_episode_modes": set(modes),
            "pending_reset": saved["pendingReset"],
            "_last_chart_point": copy.deepcopy(saved["chartPoints"][-1]) if saved["chartPoints"] else None,
            "_best_context_scores": dict(best_context_scores),
            "last_action": candidate.last_decision["lastAction"] if candidate.last_decision else None,
            "last_exploring": bool(candidate.last_decision and candidate.last_decision["isExploring"]),
            "transitions_per_second": 0.0,
        }

    def load_checkpoint(self, path: Path | str) -> bool:
        with self.lock:
            if self.intro_active:
                self.log_message = "Cancel the cinematic intro before loading a checkpoint."
                return False
            try:
                prepared = self._prepare_checkpoint(Path(path))
            except Exception as error:
                self.log_message = (
                    f"Checkpoint load failed; live learner preserved: {error}. "
                    "Legacy models require retraining."
                )
                return False
            self.__dict__.update(prepared)
            self._reset_best_schedule()
            self._advance_epoch += 1
            self.log_message = f"Loaded validated {MODEL_VERSION} checkpoint at step {self.training_steps}."
            return True

    def load_saved_or_best_checkpoint(self) -> dict[str, object]:
        with self.lock:
            if self.intro_active:
                self.log_message = "Cancel the cinematic intro before loading a checkpoint."
                return self._snapshot()
            attempted: list[str] = []
            for path in (self.manual_model_file, self.model_file, self.best_model_file):
                if path.exists():
                    if self.load_checkpoint(path):
                        return self._snapshot()
                    attempted.append(path.name)
            if not attempted:
                self.log_message = (
                    f"No compatible {MODEL_VERSION} checkpoint exists. Current random/learned policy "
                    "is unchanged; no starter policy is injected. Legacy models require retraining."
                )
            return self._snapshot()

    def load_ready_weights(self) -> dict[str, object]:
        with self.lock:
            if self.intro_active:
                raise ValueError("Cancel the active intro before loading ready weights.")
            loaded = self._champion.load_best(READY_MODEL_FILE)
            candidate = loaded.agent
            candidate.model_path = self.model_file
            seed, level, full = self.seed, self.level, self.full_survival
            self._initialize(seed, level)
            self.agent = candidate
            self._policy_source = dict(loaded.identity)
            self.training_seeds.update(loaded.training_seeds)
            self.mode, self.paused = "eval", False
            self.full_survival = full
            self.training_steps = candidate.environment_steps
            self.total_steps = self.training_steps
            self.curriculum.configure(level, False)
            self.current_episode = 0
            self._start_new_episode(seed)
            self._reset_best_schedule()
            self._advance_epoch += 1
            label = "Loaded trained best-policy champion" if loaded.identity["source"] == "trained" else "Loaded ready 3D weights"
            self.log_message = (
                f"{label} ({candidate.train_steps:,} training updates). "
                f"Greedy evaluation playback on {ENVIRONMENT_VERSION}: no exploration or learning. "
                "Safety-first reward profile; hard levels are not solved. Saved checkpoints are untouched."
            )
            return self._snapshot()

    def reset_training(self) -> dict[str, object]:
        with self.lock:
            files: list[Path] = []
            unrecognized: list[str] = []
            for path in (self.model_file, self.manual_model_file, self.best_model_file):
                if not path.exists():
                    continue
                try:
                    with np.load(path, allow_pickle=False) as data:
                        metadata = json.loads(str(data["metadata"].item()))
                    recognized = (
                        metadata.get("schemaVersion") == CHECKPOINT_SCHEMA
                        and metadata.get("modelVersion") == MODEL_VERSION
                    )
                except Exception:
                    recognized = False
                if recognized:
                    files.append(path)
                else:
                    unrecognized.append(path.name)
            if files:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                archive = self.storage_dir / "archive" / f"{stamp}-{uuid.uuid4().hex[:8]}"
                archive.mkdir(parents=True, exist_ok=False)
                for path in files:
                    path.replace(archive / path.name)
            seed, level, full = self.seed, self.level, self.full_survival
            enabled = self.auto_curriculum
            self._initialize(seed, level)
            self.full_survival = full
            self.curriculum.configure(level, enabled)
            if full:
                self._start_new_episode(seed)
            self._advance_epoch += 1
            self.paused = False
            self.log_message = (
                "Learner reset with random weights; new-format checkpoints were archived, "
                "not deleted. Legacy and ready weights are untouched. Training resumed automatically."
            )
            if unrecognized:
                self.log_message += f" Unrecognized files preserved: {', '.join(unrecognized)}."
            return self._snapshot()


def _environment_seed() -> int:
    try:
        return _seed(int(os.environ.get("SNAKE_SEED", "0")))
    except (ValueError, TypeError):
        return 0


training_session = TrainingSession(seed=_environment_seed())
