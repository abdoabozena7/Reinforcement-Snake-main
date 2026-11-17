"""Synthetic gate/persistence fixtures are not evidence of policy improvement."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SNAKE_STORAGE_DIR", str(ROOT / ".runtime-proof" / "champion-test-global"))

import numpy as np

from backend.agent import DQNAgent, Experience, PARAMETERS
from backend.champion import (
    CHAMPION_CHECK_INTERVAL, CHAMPION_LEVELS, CHAMPION_SEEDS, EvaluationCancelled,
    promotion_gate, suite_specification,
)
from backend.config import DIFFICULTIES, ENVIRONMENT_VERSION, REWARD_VERSION, SIMULATION_DT
from backend.session import READY_MODEL_FILE, TrainingSession


def synthetic_evaluation(policy_hash: str, score: int = 3) -> dict[str, object]:
    return {
        "suite": suite_specification(), "policyHash": policy_hash,
        "results": [
            {
                "seed": seed, "level": level, "score": score, "food": score, "gold": 0,
                "steps": 100, "survivalSeconds": 100 * SIMULATION_DT, "deathCause": "obstacle",
            }
            for level in CHAMPION_LEVELS for seed in CHAMPION_SEEDS
        ],
        "fixture": "synthetic gate/persistence outcomes, not a real measured improvement",
    }


def learner_fingerprint(session: TrainingSession) -> str:
    metadata, replay = session._session_metadata()
    digest = hashlib.sha256(json.dumps(
        [metadata, session.agent._metadata(None), session.env.rng.getstate()],
        sort_keys=True,
    ).encode())
    for array in replay.values():
        digest.update(array.tobytes())
    for name in PARAMETERS:
        for array in (
            getattr(session.agent, name), getattr(session.agent, f"target_{name}"),
            session.agent.adam_m[name], session.agent.adam_v[name],
        ):
            digest.update(array.tobytes())
    return digest.hexdigest()


class ChampionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / ".runtime-proof" / f"champion-test-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.sessions: list[TrainingSession] = []
        self.ready_file = self.root / "synthetic-ready.npz"
        fixture = DQNAgent(self.ready_file, seed=111)
        fixture.train_batch([Experience([0.2] * fixture.state_size, 1, 1.0, [0.1] * fixture.state_size, False)])
        fixture.record_training_step()
        fixture.save_to(self.ready_file, {
            "rewardVersion": REWARD_VERSION, "environmentVersion": ENVIRONMENT_VERSION,
            "trainingSeeds": [11111], "fixture": "synthetic training data",
        })
        self.ready_hash = fixture.parameter_hash()
        self.ready_bytes = self.ready_file.read_bytes()
        self.ready_patch = patch("backend.session.READY_MODEL_FILE", self.ready_file)
        self.ready_patch.start()

    def tearDown(self) -> None:
        for session in self.sessions:
            session.close()
        self.ready_patch.stop()
        shutil.rmtree(self.root)

    def session(self, storage: Path | None = None) -> TrainingSession:
        session = TrainingSession(storage_dir=storage or self.root, seed=71, load_saved=False)
        self.sessions.append(session)
        return session

    def candidate(self) -> TrainingSession:
        session = self.session()
        session.load_ready_weights()
        session.min_buffer_for_training = session.batch_size = session.train_every = 1
        session.control({"level": 1, "mode": "train", "paused": False})
        session.tick(2)
        self.assertNotEqual(session.agent.parameter_hash(), self.ready_hash)
        return session

    def finish_check(self, session: TrainingSession, timeout: float = 5) -> dict[str, object]:
        worker = session._champion._thread
        self.assertIsNotNone(worker)
        worker.join(timeout)
        self.assertFalse(worker.is_alive(), "Best-policy worker did not finish")
        return session.get_state(include_network=False)

    def measurements(self, scores: dict[str, int]):
        return lambda policy, cancel: synthetic_evaluation(policy.parameter_hash(), scores[policy.parameter_hash()])

    def promote_fixture(self, session: TrainingSession) -> None:
        scores = {self.ready_hash: 3, session.agent.parameter_hash(): 4}
        with patch("backend.champion.evaluate_suite", side_effect=self.measurements(scores)):
            session.check_best_policy()
            state = self.finish_check(session)
        self.assertEqual(state["bestPolicy"]["status"], "promoted")

    def test_gate_requires_per_level_safety_basic_food_and_overall_food(self) -> None:
        baseline = synthetic_evaluation("a" * 64)
        variants = {}
        deaths = copy.deepcopy(baseline)
        deaths["results"][4].update(deathCause="time_limit", steps=480, survivalSeconds=57.6)
        candidate = copy.deepcopy(deaths)
        candidate["results"][4].update(deathCause="hunter", score=100)
        variants["deaths"] = (deaths, candidate)
        candidate = copy.deepcopy(baseline)
        candidate["results"][8].update(steps=99, survivalSeconds=11.88, score=100)
        candidate["results"][4].update(steps=200, survivalSeconds=24.0)
        variants["survival"] = (baseline, candidate)
        candidate = synthetic_evaluation("b" * 64, score=10)
        candidate["results"][0]["score"] = 0
        for row in candidate["results"][1:4]:
            row["score"] = 3
        variants["level 1"] = (baseline, candidate)
        candidate = copy.deepcopy(baseline)
        candidate["results"][4].update(score=2, steps=101, survivalSeconds=12.12)
        variants["Overall food"] = (baseline, candidate)
        for reason, (old, new) in variants.items():
            with self.subTest(reason=reason):
                allowed, message = promotion_gate(old, new)
                self.assertFalse(allowed)
                self.assertIn(reason.lower(), message.lower())

    def test_gate_retains_equality_but_allows_food_survival_or_death_improvements(self) -> None:
        baseline = synthetic_evaluation("a" * 64)
        self.assertFalse(promotion_gate(baseline, copy.deepcopy(baseline))[0])
        for improvement in ("score", "survival", "deaths"):
            candidate = copy.deepcopy(baseline)
            row = candidate["results"][4]
            if improvement == "score":
                row["score"] += 1
            elif improvement == "survival":
                row.update(steps=101, survivalSeconds=12.12)
            else:
                row.update(deathCause="time_limit", steps=480, survivalSeconds=57.6)
            with self.subTest(improvement=improvement):
                self.assertTrue(promotion_gate(baseline, candidate)[0])
        incomplete = copy.deepcopy(baseline)
        incomplete["results"].pop()
        with self.assertRaises(ValueError):
            promotion_gate(baseline, incomplete)
        incomplete = copy.deepcopy(baseline)
        del incomplete["results"][0]["score"]
        with self.assertRaises(KeyError):
            promotion_gate(baseline, incomplete)

    def test_atomic_promotion_preserves_all_candidate_tensors_and_prior_versions(self) -> None:
        session = self.candidate()
        live = session.agent
        expected = live.clone()
        before = learner_fingerprint(session)
        self.promote_fixture(session)
        self.assertIs(session.agent, live)
        self.assertEqual(learner_fingerprint(session), before)
        retained = DQNAgent(session._champion.path)
        metadata, arrays = retained.read_checkpoint(session._champion.path)
        retained.restore_checkpoint(metadata, arrays)
        self.assertEqual(retained.parameter_hash(), expected.parameter_hash())
        self.assertEqual(retained.environment_steps, expected.environment_steps)
        self.assertEqual(retained.train_steps, expected.train_steps)
        for name in PARAMETERS:
            for array in (name, f"target_{name}"):
                np.testing.assert_array_equal(getattr(retained, array), getattr(expected, array))
            np.testing.assert_array_equal(retained.adam_m[name], expected.adam_m[name])
            np.testing.assert_array_equal(retained.adam_v[name], expected.adam_v[name])
        self.assertFalse(any(name.startswith("replay_") for name in arrays))
        provenance = metadata["trainingMetadata"]
        self.assertNotIn("session", provenance)
        self.assertEqual(provenance["rewardVersion"], REWARD_VERSION)
        self.assertEqual(provenance["environmentVersion"], ENVIRONMENT_VERSION)
        self.assertTrue({71, 11111}.issubset(provenance["trainingSeeds"]))
        self.assertEqual(len(provenance["champion"]["receipt"]["candidateEvaluation"]["results"]), 12)
        previous = session._champion.path.read_bytes()
        previous_hash = session.agent.parameter_hash()
        session.tick()
        scores = {previous_hash: 4, session.agent.parameter_hash(): 5}
        with patch("backend.champion.evaluate_suite", side_effect=self.measurements(scores)):
            session.check_best_policy()
            self.assertEqual(self.finish_check(session)["bestPolicy"]["status"], "promoted")
        archives = list((session.storage_dir / "champion-history").glob("*.npz"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), previous)
        self.assertEqual(self.ready_file.read_bytes(), self.ready_bytes)

    def test_failed_atomic_save_preserves_previous_champion_and_live_learner(self) -> None:
        session = self.candidate()
        self.promote_fixture(session)
        previous = session._champion.path.read_bytes()
        session.tick()
        before = learner_fingerprint(session)
        scores = {session.agent.parameter_hash(): 5}
        replace = os.replace

        def fail_champion(source: Path, destination: Path) -> None:
            if Path(destination) == session._champion.path:
                raise OSError("synthetic atomic replacement failure")
            replace(source, destination)

        with patch("backend.champion.evaluate_suite", side_effect=self.measurements(scores)), patch(
            "backend.agent.os.replace", side_effect=fail_champion
        ), self.assertLogs("backend.champion", level="ERROR"):
            session.check_best_policy()
            state = self.finish_check(session)
        self.assertEqual(state["bestPolicy"]["status"], "error")
        self.assertIn("atomic replacement failure", state["bestPolicy"]["message"])
        self.assertEqual(session._champion.path.read_bytes(), previous)
        self.assertEqual(learner_fingerprint(session), before)
        self.assertFalse(list(session.storage_dir.glob("*.partial")))

    def test_reset_and_restart_keep_champion_and_load_best_even_without_ready(self) -> None:
        session = self.candidate()
        self.promote_fixture(session)
        manager = session._champion
        champion_hash = session.agent.parameter_hash()
        champion_bytes = manager.path.read_bytes()
        counters = (session.training_steps, session.agent.train_steps)
        session.save_manual_checkpoint()
        state = session.reset_training()
        self.assertIs(session._champion, manager)
        self.assertEqual((state["mode"], state["trainingSteps"], state["epsilon"]), ("train", 0, 1.0))
        self.assertNotEqual(session.agent.parameter_hash(), champion_hash)
        self.assertEqual(manager.path.read_bytes(), champion_bytes)
        with patch("backend.session.READY_MODEL_FILE", self.root / "missing-ready.npz"):
            state = session.load_ready_weights()
            self.assertEqual(session.agent.parameter_hash(), champion_hash)
            self.assertEqual((session.training_steps, session.agent.train_steps), counters)
            self.assertEqual((state["mode"], state["epsilon"], state["bestPolicy"]["source"]), ("eval", 0.0, "trained"))
            session.close()
            restarted = self.session()
            state = restarted.load_ready_weights()
            self.assertEqual(restarted.agent.parameter_hash(), champion_hash)
            self.assertEqual(state["bestPolicy"]["source"], "trained")
        self.assertEqual(manager.path.read_bytes(), champion_bytes)

    def test_equality_and_regression_retain_ready_and_cache_its_measurement(self) -> None:
        session = self.candidate()
        before = learner_fingerprint(session)
        with patch("backend.champion.evaluate_suite", side_effect=lambda policy, cancel: synthetic_evaluation(policy.parameter_hash())) as evaluate:
            session.check_best_policy()
            self.assertEqual(self.finish_check(session)["bestPolicy"]["status"], "retained")
            self.assertEqual(evaluate.call_count, 2)
            worse = synthetic_evaluation(session.agent.parameter_hash(), score=100)
            worse["results"][8].update(steps=99, survivalSeconds=11.88)
            evaluate.side_effect = lambda policy, cancel: copy.deepcopy(worse)
            session.check_best_policy()
            state = self.finish_check(session)
            self.assertEqual(evaluate.call_count, 3, "The unchanged baseline should be cached")
        self.assertEqual(state["bestPolicy"]["status"], "retained")
        self.assertIn("survival regressed", state["bestPolicy"]["message"])
        self.assertEqual(learner_fingerprint(session), before)
        self.assertFalse(session._champion.path.exists())
        receipts = [json.loads(line) for line in session._champion.history_path.read_text().splitlines()]
        self.assertEqual([receipt["status"] for receipt in receipts], ["retained", "retained"])

    def test_baseline_identity_is_rechecked_before_promotion(self) -> None:
        session = self.candidate()
        before = learner_fingerprint(session)
        replacement = DQNAgent(self.ready_file, seed=222)
        replacement.train_batch([Experience([0.3] * replacement.state_size, 2, 1.0, [0.1] * replacement.state_size, False)])
        replacement.record_training_step()

        def changed_baseline(policy: DQNAgent, cancel: threading.Event) -> dict[str, object]:
            if policy.parameter_hash() == self.ready_hash:
                return synthetic_evaluation(self.ready_hash)
            replacement.save_to(self.ready_file, {
                "rewardVersion": REWARD_VERSION, "environmentVersion": ENVIRONMENT_VERSION,
                "trainingSeeds": [22222], "fixture": "synthetic concurrent baseline replacement",
            })
            return synthetic_evaluation(policy.parameter_hash(), score=4)

        with patch("backend.champion.evaluate_suite", side_effect=changed_baseline):
            session.check_best_policy()
            state = self.finish_check(session)
        self.assertEqual(state["bestPolicy"]["status"], "retained")
        self.assertIn("changed during evaluation", state["bestPolicy"]["message"])
        self.assertFalse(session._champion.path.exists())
        self.assertEqual(learner_fingerprint(session), before)
        receipt = json.loads(session._champion.history_path.read_text())
        self.assertTrue(receipt["gate"]["promote"])
        self.assertEqual(receipt["status"], "retained")

    def test_check_is_nonblocking_single_worker_and_cancellation_does_not_mutate_learner(self) -> None:
        session = self.candidate()
        entered = threading.Event()

        def blocked(policy: DQNAgent, cancel: threading.Event) -> dict[str, object]:
            with session.lock:
                entered.set()
            cancel.wait(5)
            raise EvaluationCancelled("Synthetic blocked check cancelled; current best retained.")

        with patch("backend.champion.evaluate_suite", side_effect=blocked):
            started = time.perf_counter()
            state = session.check_best_policy()
            self.assertLess(time.perf_counter() - started, 0.5)
            self.assertEqual(state["bestPolicy"]["status"], "checking")
            self.assertTrue(entered.wait(2), "Evaluation must not hold session.lock")
            worker = session._champion._thread
            session.check_best_policy()
            self.assertIs(session._champion._thread, worker)
            steps = session.training_steps
            session.tick(2)
            self.assertEqual(session.training_steps, steps + 2)
            before = learner_fingerprint(session)
            session.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(learner_fingerprint(session), before)
        self.assertFalse(session._champion.path.exists())
        self.assertEqual(session.get_state(False)["bestPolicy"]["status"], "retained")

    def test_error_and_seed_overlap_are_reported_without_evaluation_or_learner_mutation(self) -> None:
        session = self.candidate()
        before = learner_fingerprint(session)
        with patch("backend.champion.evaluate_suite", side_effect=RuntimeError("synthetic rollout error")), self.assertLogs("backend.champion", level="ERROR"):
            session.check_best_policy()
            state = self.finish_check(session)
        self.assertEqual(state["bestPolicy"]["status"], "error")
        self.assertIn("synthetic rollout error", state["bestPolicy"]["message"])
        self.assertEqual(learner_fingerprint(session), before)
        session.training_seeds.add(CHAMPION_SEEDS[0])
        before = learner_fingerprint(session)
        with patch("backend.champion.evaluate_suite") as evaluate, self.assertLogs("backend.champion", level="ERROR"):
            session.check_best_policy()
            state = self.finish_check(session)
            evaluate.assert_not_called()
        self.assertEqual(state["bestPolicy"]["status"], "error")
        self.assertIn("overlap", state["bestPolicy"]["message"])
        self.assertEqual(learner_fingerprint(session), before)
        self.assertFalse(session._champion.path.exists())
        with self.assertRaises(ValueError):
            session.control({"seed": CHAMPION_SEEDS[0]})
        with self.assertRaises(ValueError):
            TrainingSession(storage_dir=self.root / "reserved", seed=CHAMPION_SEEDS[0], load_saved=False)

    def test_worker_start_failure_is_reported_and_close_remains_safe(self) -> None:
        session = self.candidate()
        before = learner_fingerprint(session)
        with patch("backend.champion.threading.Thread.start", side_effect=RuntimeError("synthetic start failure")), self.assertLogs("backend.champion", level="ERROR"):
            state = session.check_best_policy()
        self.assertEqual(state["bestPolicy"]["status"], "error")
        self.assertIn("synthetic start failure", state["bestPolicy"]["message"])
        session.close()
        self.assertEqual(learner_fingerprint(session), before)
        self.assertFalse(session._champion.path.exists())

    def test_corrupt_or_unmeasured_champion_never_silently_falls_back_to_ready(self) -> None:
        session = self.session()
        before = learner_fingerprint(session)
        session.storage_dir.mkdir(parents=True)
        for contents in (b"corrupt retained file", self.ready_bytes):
            session._champion.path.write_bytes(contents)
            with self.subTest(contents=len(contents)), self.assertLogs("backend.champion", level="ERROR"), self.assertRaisesRegex(ValueError, "Retained champion.*live learner preserved"):
                session.load_ready_weights()
            self.assertEqual(session._champion.path.read_bytes(), contents)
            self.assertEqual(learner_fingerprint(session), before)
            self.assertEqual(session.get_state(False)["bestPolicy"]["status"], "error")
        session._champion.path.unlink()
        recovered = session.load_ready_weights()
        self.assertEqual(recovered["bestPolicy"]["source"], "ready")
        self.assertEqual(recovered["bestPolicy"]["status"], "idle")

    def test_loading_and_reset_restart_auto_schedule_from_current_transition_and_gradient_markers(self) -> None:
        session = self.session()
        loaded = session.load_ready_weights()
        self.assertTrue({71, 11111}.issubset(session.training_seeds))
        self.assertEqual(loaded["trainingSteps"], 1)
        self.assertEqual(session._best_check_training_steps, 1)
        self.assertEqual(session._best_check_gradient_steps, 1)
        manager = session._champion
        with patch.object(manager, "queue", return_value=True) as queue:
            session.control({"mode": "train"})
            session.tick()
            queue.assert_not_called()
            # A synthetic due boundary isolates the separate actual-new-gradient requirement.
            session._best_check_training_steps = session.training_steps - CHAMPION_CHECK_INTERVAL
            session.tick()
            queue.assert_not_called()
            session.min_buffer_for_training = session.batch_size = session.train_every = 1
            session.tick()
            queue.assert_called_once()
            self.assertEqual(session._best_check_training_steps, session.training_steps)
            self.assertEqual(session._best_check_gradient_steps, session.agent.train_steps)
            checkpoint = session.save_checkpoint(self.root / "schedule.npz")
            session.reset_training()
            self.assertIs(session._champion, manager)
            self.assertEqual((session._best_check_training_steps, session._best_check_gradient_steps), (0, 0))
            self.assertTrue(session.load_checkpoint(checkpoint), session.log_message)
            self.assertEqual(session._best_check_training_steps, session.training_steps)
            self.assertEqual(session._best_check_gradient_steps, session.agent.train_steps)

    def test_real_unchanged_ready_is_retained_with_every_actual_pinned_case(self) -> None:
        session = self.session()
        asset_hash = hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest()
        with patch("backend.session.READY_MODEL_FILE", READY_MODEL_FILE):
            session.load_ready_weights()
            before = learner_fingerprint(session)
            session.check_best_policy()
            state = self.finish_check(session, timeout=120)
        self.assertEqual(state["bestPolicy"]["status"], "retained")
        self.assertEqual(state["bestPolicy"]["source"], "ready")
        self.assertEqual(learner_fingerprint(session), before)
        self.assertFalse(session._champion.path.exists())
        receipt = json.loads(session._champion.history_path.read_text())
        self.assertFalse(receipt["gate"]["promote"])
        for key in ("baselineEvaluation", "candidateEvaluation"):
            rows = receipt[key]["results"]
            self.assertEqual(len(rows), 12)
            self.assertEqual({(row["level"], row["seed"]) for row in rows}, {
                (level, seed) for level in CHAMPION_LEVELS for seed in CHAMPION_SEEDS
            })
            self.assertFalse(set(CHAMPION_SEEDS).intersection(session.training_seeds))
            for row in rows:
                self.assertLessEqual(row["steps"], DIFFICULTIES[row["level"]]["step_limit"])
        self.assertEqual(hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest(), asset_hash)


if __name__ == "__main__":
    unittest.main()
