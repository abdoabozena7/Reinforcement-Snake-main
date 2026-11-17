"""Frozen, held-out policy comparisons; never owns or updates the live learner."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

from backend.agent import DQNAgent, _integer
from backend.config import DIFFICULTIES, ENVIRONMENT_VERSION, MODEL_VERSION, REWARD_VERSION, SIMULATION_DT
from backend.curriculum import Curriculum
from backend.environment import SnakeEnvironment


CHAMPION_SEEDS = (3_600_008_001, 3_600_008_002, 3_600_008_003, 3_600_008_004)
CHAMPION_LEVELS = (1, 4, 6)
CHAMPION_CHECK_INTERVAL = 4096
LOGGER = logging.getLogger(__name__)
_STORAGE_LOCKS: dict[str, threading.RLock] = {}
_STORAGE_LOCKS_GUARD = threading.Lock()


def suite_specification() -> dict[str, object]:
    return {
        "version": "champion-food-safety-v1",
        "seeds": list(CHAMPION_SEEDS),
        "levels": list(CHAMPION_LEVELS),
        "fullSurvival": False,
        "mode": "eval",
        "epsilon": 0.0,
        "modelVersion": MODEL_VERSION,
        "environmentVersion": ENVIRONMENT_VERSION,
        "rewardVersion": REWARD_VERSION,
    }


class EvaluationCancelled(Exception):
    pass


def _check_cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise EvaluationCancelled("Best-policy check cancelled; current best retained.")


def evaluation_rollout(
    policy: DQNAgent, seed: int, level: int, full_survival: bool,
    *, cancel: threading.Event | None = None, step_delay: float = 0.0,
) -> dict[str, object]:
    """The ordinary evaluation rollout, with only legal-action (not hazard) masks."""
    _check_cancelled(cancel)
    environment = SnakeEnvironment(seed=seed, level=level, full_survival=full_survival)
    observation = environment.get_state()
    total_reward = 0.0
    while not environment.done:
        _check_cancelled(cancel)
        if cancel is not None and cancel.wait(step_delay):
            _check_cancelled(cancel)
        action = policy.act(
            observation, environment.action_mask(), explore=False, capture_decision=False
        )
        transition = environment.step(action)
        observation = transition["next_state"]
        total_reward += float(transition["reward"])
    _check_cancelled(cancel)
    result: dict[str, object] = {
        "seed": seed,
        "score": environment.score,
        "food": environment.food_collected,
        "steps": environment.steps,
        "survivalSeconds": round(environment.steps * SIMULATION_DT, 4),
        "enemyEscapes": environment.enemy_escapes,
        "deathCause": environment.last_death,
        "totalReward": round(total_reward, 6),
        "gold": environment.gold_collected,
        "rivalFood": environment.rival_food_collected,
        "dangerExposure": round(environment.danger_exposure / max(1, environment.steps), 6),
        "foodPer100Ticks": round(environment.food_collected * 100 / max(1, environment.steps), 6),
    }
    result["success"] = Curriculum.successful(result, Curriculum.goals(level))
    return result


def _summarize(record: dict[str, object]) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    if record["suite"] != suite_specification():
        raise ValueError("Incompatible champion evaluation suite.")
    rows = record["results"]
    expected = {(level, seed) for level in CHAMPION_LEVELS for seed in CHAMPION_SEEDS}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Champion evaluation must contain every pinned case.")
    seen: set[tuple[int, int]] = set()
    for row in rows:
        _integer(row["level"], "evaluated level", 1, 6)
        _integer(row["seed"], "evaluated seed", 0, 2**32 - 1)
        key = (row["level"], row["seed"])
        if key not in expected or key in seen:
            raise ValueError("Missing, duplicate or unpinned champion evaluation case.")
        seen.add(key)
        _integer(row["score"], "evaluated score", 0)
        steps = _integer(row["steps"], "evaluated survival steps", 1, DIFFICULTIES[key[0]]["step_limit"])
        if (
            isinstance(row["survivalSeconds"], bool)
            or row["survivalSeconds"] != round(steps * SIMULATION_DT, 4)
            or (row["deathCause"] is not None and not isinstance(row["deathCause"], str))
            or (row["deathCause"] in (None, "time_limit") and steps != DIFFICULTIES[key[0]]["step_limit"])
        ):
            raise ValueError("Invalid champion evaluation survival/death outcome.")

    def summary(cases: list[dict[str, object]]) -> dict[str, float]:
        return {
            "meanScore": sum(row["score"] for row in cases) / len(cases),
            "meanSurvivalSeconds": sum(row["survivalSeconds"] for row in cases) / len(cases),
            "deaths": sum(row["deathCause"] not in (None, "time_limit") for row in cases),
        }

    return (
        {level: summary([row for row in rows if row["level"] == level]) for level in CHAMPION_LEVELS},
        summary(rows),
    )


def promotion_gate(baseline: dict[str, object], candidate: dict[str, object]) -> tuple[bool, str]:
    old_levels, old = _summarize(baseline)
    new_levels, new = _summarize(candidate)
    for level in CHAMPION_LEVELS:
        if new_levels[level]["deaths"] > old_levels[level]["deaths"]:
            return False, f"Level {level} deaths regressed; current best retained."
        if new_levels[level]["meanSurvivalSeconds"] < old_levels[level]["meanSurvivalSeconds"]:
            return False, f"Level {level} survival regressed; current best retained."
    if new_levels[1]["meanScore"] < old_levels[1]["meanScore"]:
        return False, "Basic level 1 food Score regressed; current best retained."
    if new["meanScore"] < old["meanScore"]:
        return False, "Overall food Score regressed; current best retained."
    if not (
        new["meanScore"] > old["meanScore"]
        or new["meanSurvivalSeconds"] > old["meanSurvivalSeconds"]
        or new["deaths"] < old["deaths"]
    ):
        return False, "No strict held-out improvement (equal policies are retained, not promoted)."
    return True, "Held-out food/safety improved without a gated regression."


def evaluate_suite(policy: DQNAgent, cancel: threading.Event) -> dict[str, object]:
    frozen = policy.clone()
    started = time.perf_counter()
    results = []
    for level in CHAMPION_LEVELS:
        for seed in CHAMPION_SEEDS:
            row = evaluation_rollout(frozen, seed, level, False, cancel=cancel, step_delay=0.005)
            results.append({**row, "level": level})
    record = {
        "suite": suite_specification(), "policyHash": frozen.parameter_hash(), "results": results,
        "seconds": time.perf_counter() - started,
    }
    levels, overall = _summarize(record)
    record.update({"levels": {str(level): value for level, value in levels.items()}, "overall": overall})
    return record


@dataclass
class LoadedPolicy:
    agent: DQNAgent
    metadata: dict[str, object]
    training_seeds: set[int]
    identity: dict[str, object]


def _read_policy(path: Path, source: str) -> LoadedPolicy:
    label = "Retained champion" if source == "trained" else "Ready 3D weights"
    try:
        agent = DQNAgent(path)
        metadata, arrays = agent.read_checkpoint(path)
        agent.restore_checkpoint(metadata, arrays)
        provenance = metadata["trainingMetadata"]
        if agent.train_steps < 1:
            raise ValueError("Weights have no recorded training updates.")
        if provenance.get("rewardVersion") != REWARD_VERSION:
            raise ValueError("Weights were not trained under the current survival rewards.")
        if provenance.get("environmentVersion") != ENVIRONMENT_VERSION:
            raise ValueError("Weights were not trained on the current arena.")
        seeds: set[int] = set()
        for entry in (provenance, provenance.get("session", {})):
            if not isinstance(entry, dict):
                raise ValueError("Invalid source training provenance.")
            if "trainingSeeds" in entry:
                if not isinstance(entry["trainingSeeds"], list):
                    raise ValueError("Invalid source training seeds.")
                seeds.update(_integer(seed, "source training seed", 0, 2**32 - 1) for seed in entry["trainingSeeds"])
        policy_hash = agent.parameter_hash()
        if source == "trained":
            if seeds.intersection(CHAMPION_SEEDS):
                raise ValueError("Retained champion training seeds overlap its held-out suite.")
            champion = provenance["champion"]
            receipt = champion["receipt"]
            if (
                champion["suite"] != suite_specification()
                or receipt["status"] != "promoted"
                or receipt["candidate"]["policyHash"] != policy_hash
                or receipt["candidate"]["trainingSteps"] != agent.environment_steps
                or receipt["candidate"]["gradientSteps"] != agent.train_steps
                or receipt["candidateEvaluation"]["policyHash"] != policy_hash
                or receipt["baselineEvaluation"]["policyHash"] != receipt["source"]["policyHash"]
                or not promotion_gate(receipt["baselineEvaluation"], receipt["candidateEvaluation"])[0]
            ):
                raise ValueError("Invalid retained champion evaluation receipt.")
        return LoadedPolicy(agent, metadata, seeds, {
            "source": source, "file": str(path.resolve()), "policyHash": policy_hash,
            "fileSHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    except Exception as error:
        raise ValueError(f"{label} could not be loaded; live learner preserved: {error}") from error


class ChampionManager:
    """Bounded 12-case checks; atomic champion.npz, version archives and JSONL receipts."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.path = storage_dir / "champion.npz"
        self.history_path = storage_dir / "champion-evaluations.jsonl"
        with _STORAGE_LOCKS_GUARD:
            self._storage_lock = _STORAGE_LOCKS.setdefault(
                os.path.normcase(str(storage_dir.resolve())), threading.RLock()
            )
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._baseline_cache: dict[str, dict[str, object]] = {}
        self._state = {
            "source": "trained" if self.path.exists() else "ready",
            "status": "idle",
            "message": "Retained best policy is available." if self.path.exists() else "Ready is the initial best policy.",
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._state)

    def _state_update(self, **values: object) -> None:
        with self._lock:
            self._state.update(values)

    def _best(self, ready_path: Path) -> LoadedPolicy:
        source = "trained" if self.path.exists() else "ready"
        self._state_update(source=source)
        return _read_policy(self.path if source == "trained" else ready_path, source)

    def load_best(self, ready_path: Path) -> LoadedPolicy:
        try:
            with self._storage_lock:
                loaded = self._best(ready_path)
            with self._lock:
                if self._state["status"] != "checking":
                    self._state.update(status="idle", message="Best policy loaded for greedy Test; Train continues normal Double DQN.")
            return loaded
        except Exception as error:
            LOGGER.error("Best policy load failed: %s", error)
            self._state_update(status="error", message=str(error))
            raise

    def queue(
        self, agent: DQNAgent, training_seeds: set[int], counters: dict[str, int],
        source: dict[str, object], ready_path: Path,
    ) -> bool:
        with self._lock:
            if self._closed:
                self._state.update(status="error", message="Best-policy helper is closed.")
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            try:
                self._cancel.clear()
                candidate = agent.clone()
                identity = {
                    "policyHash": candidate.parameter_hash(), **counters, "source": copy.deepcopy(source),
                    "trainingSeedCount": len(training_seeds),
                    "trainingSeedsSHA256": hashlib.sha256(json.dumps(sorted(training_seeds)).encode()).hexdigest(),
                }
                self._state.update(status="checking", message="Checking a frozen candidate on 12 held-out food/safety cases; learning continues.")
                self._thread = threading.Thread(
                    target=self._run, args=(candidate, set(training_seeds), identity, Path(ready_path)),
                    name="snake-best-policy", daemon=True,
                )
                self._thread.start()
            except Exception as error:
                LOGGER.exception("Could not start best-policy check")
                self._thread = None
                self._state.update(status="error", message=f"Best-policy check could not start: {error}")
                return False
            return True

    def _remember(self, policy_hash: str, record: dict[str, object]) -> None:
        self._baseline_cache[policy_hash] = copy.deepcopy(record)
        if len(self._baseline_cache) > 4:
            del self._baseline_cache[next(iter(self._baseline_cache))]

    def _record(self, receipt: dict[str, object]) -> None:
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._storage_lock:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _promote(
        self, candidate: DQNAgent, seeds: set[int], baseline: LoadedPolicy,
        ready_path: Path, receipt: dict[str, object],
    ) -> tuple[bool, str]:
        with self._storage_lock:
            _check_cancelled(self._cancel)
            current = self._best(ready_path)
            self._state_update(source=current.identity["source"])
            if current.identity != baseline.identity:
                return False, "Best policy changed during evaluation; current best retained. Check again."
            if self.path.exists():
                directory = self.storage_dir / "champion-history"
                directory.mkdir(parents=True, exist_ok=True)
                archive = directory / f"{uuid.uuid4().hex}-{current.agent.parameter_hash()[:12]}.npz"
                partial = archive.with_suffix(".partial")
                try:
                    with partial.open("xb") as stream:
                        stream.write(self.path.read_bytes())
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(partial, archive)
                finally:
                    if partial.exists():
                        partial.unlink()
            _check_cancelled(self._cancel)
            candidate.save_to(self.path, {
                "label": "retained-food-safety-champion", "savedAt": receipt["checkedAt"],
                "rewardVersion": REWARD_VERSION, "environmentVersion": ENVIRONMENT_VERSION,
                "trainingSeeds": sorted(seeds), "heldOutSeeds": list(CHAMPION_SEEDS),
                "champion": {"suite": suite_specification(), "receipt": {**receipt, "status": "promoted"}},
            })
            self._state_update(source="trained")
            self._remember(candidate.parameter_hash(), receipt["candidateEvaluation"])
            return True, "Promoted held-out food/safety improvement; live learner was not replaced."

    def _run(
        self, candidate: DQNAgent, seeds: set[int], identity: dict[str, object], ready_path: Path,
    ) -> None:
        receipt: dict[str, object] = {
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "suite": suite_specification(), "candidate": identity,
        }
        status = "retained"
        try:
            _check_cancelled(self._cancel)
            if seeds.intersection(CHAMPION_SEEDS):
                raise ValueError("Candidate training seeds overlap the reserved champion suite.")
            if (
                identity["trainingSteps"] != candidate.environment_steps
                or identity["gradientSteps"] != candidate.train_steps
            ):
                raise ValueError("Candidate training/gradient counters do not match its frozen policy.")
            _integer(identity["totalSteps"], "candidate total steps", candidate.environment_steps)
            if candidate.train_steps < 1:
                message = "No optimizer updates in the candidate yet; current best retained."
            else:
                with self._storage_lock:
                    baseline = self._best(ready_path)
                if baseline.training_seeds.intersection(CHAMPION_SEEDS):
                    raise ValueError("Best policy training seeds overlap the reserved champion suite.")
                self._state_update(source=baseline.identity["source"])
                receipt["source"] = baseline.identity
                baseline_hash = baseline.agent.parameter_hash()
                measured = self._baseline_cache.get(baseline_hash)
                if measured is None:
                    measured = evaluate_suite(baseline.agent, self._cancel)
                    if measured["policyHash"] != baseline_hash:
                        raise ValueError("Baseline evaluation policy identity changed.")
                    _summarize(measured)
                    self._remember(baseline_hash, measured)
                receipt["baselineEvaluation"] = copy.deepcopy(measured)
                evaluated = evaluate_suite(candidate, self._cancel)
                if evaluated["policyHash"] != identity["policyHash"]:
                    raise ValueError("Candidate evaluation policy identity changed.")
                receipt["candidateEvaluation"] = evaluated
                promote, message = promotion_gate(measured, evaluated)
                receipt["gate"] = {"promote": promote, "reason": message}
                _check_cancelled(self._cancel)
                if promote:
                    promoted, message = self._promote(candidate, seeds, baseline, ready_path, receipt)
                    status = "promoted" if promoted else "retained"
            receipt.update(status=status, message=message)
        except EvaluationCancelled as error:
            message = str(error)
            receipt.update(status="cancelled", message=message)
        except Exception as error:
            LOGGER.exception("Best-policy evaluation failed")
            status, message = "error", f"Best-policy check failed; current best retained: {error}"
            receipt.update(status=status, message=message)
        try:
            self._record(receipt)
        except Exception as error:
            LOGGER.exception("Best-policy evaluation receipt could not be saved")
            status, message = "error", f"Best-policy receipt could not be saved: {error}"
        self._state_update(status=status, message=message)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancel.set()
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join()
