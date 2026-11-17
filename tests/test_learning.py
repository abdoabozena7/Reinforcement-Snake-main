from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SNAKE_STORAGE_DIR", str(ROOT / ".runtime-proof" / "learning-test-global"))

from backend.agent import DQNAgent, Experience, PARAMETERS, ReplayBuffer
from backend.config import DIFFICULTIES, ENVIRONMENT_VERSION, MODEL_VERSION
from backend.curriculum import Curriculum
from backend.environment import ACTION_SIZE, OBSERVATION_LABELS, OBSERVATION_SIZE, Point, SnakeEnvironment
from backend.session import READY_MODEL_FILE, TrainingSession

import numpy as np


def learning_fingerprint(session: TrainingSession) -> str:
    digest = hashlib.sha256()
    metadata, arrays = session.buffer.checkpoint()
    for array in arrays.values():
        digest.update(array.tobytes())
    for name in PARAMETERS:
        for array in (
            getattr(session.agent, name),
            getattr(session.agent, f"target_{name}"),
            session.agent.adam_m[name],
            session.agent.adam_v[name],
        ):
            digest.update(array.tobytes())
    digest.update(
        json.dumps(
            {
                "board": session.env.board(),
                "environmentRng": session.env.rng.getstate(),
                "policyRng": session.agent.rng.bit_generator.state,
                "sessionRng": session.rng.bit_generator.state,
                "replay": metadata,
                "epsilon": session.agent.epsilon,
                "trainingSteps": session.training_steps,
                "totalSteps": session.total_steps,
                "gradientSteps": session.agent.train_steps,
                "completed": session.completed_episodes,
                "episode": session.current_episode,
                "actions": session._episode_actions,
                "charts": session.chart_points,
                "trainingSeeds": sorted(session.training_seeds),
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


class LearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / ".runtime-proof" / f"learning-test-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def agent(self, **options: object) -> DQNAgent:
        defaults = {"state_size": 4, "action_size": 4, "hidden_size": 96, "seed": 7}
        defaults.update(options)
        return DQNAgent(self.root / "agent.npz", **defaults)

    def session(self, seed: int = 7) -> TrainingSession:
        return TrainingSession(storage_dir=self.root, seed=seed, load_saved=False)

    def test_masks_greedy_and_exploration_but_not_hazard_safety(self) -> None:
        agent = self.agent(epsilon=1.0)
        for name in PARAMETERS:
            getattr(agent, name).fill(0)
        agent.b3[:] = [100, 4, 3, 2]
        mask = [False, True, False, False]
        self.assertEqual(agent.act([0] * 4, mask, explore=False), 1)
        self.assertEqual({agent.act([0] * 4, mask) for _ in range(50)}, {1})
        self.assertEqual(agent.act([0] * 4, [True] * 4, explore=False), 0)
        session = self.session()
        session.control({"level": 1})
        session.env.snake[0] = Point(0, 0, session.env.snake[0].z)
        self.assertTrue(session.env.action_mask()[3], "A dangerous west turn must remain selectable.")

    def test_greedy_decisions_do_not_consume_action_rng(self) -> None:
        agent = self.agent()
        before = copy.deepcopy(agent.rng.bit_generator.state)
        for _ in range(4):
            agent.act([0.2] * 4, [True] * 4, explore=False)
        self.assertEqual(before, agent.rng.bit_generator.state)

    def test_double_dqn_masked_bootstrap_and_terminal_target(self) -> None:
        agent = self.agent()
        for name in PARAMETERS:
            getattr(agent, name).fill(0)
        agent.b3[:] = [100, 2, 1, 0]
        agent._sync_target_network()
        agent.target_b3[:] = [999, 7, 3, 1]
        metrics = agent.train_batch(
            [
                Experience([0] * 4, 1, 1.0, [0] * 4, False, next_action_mask=[False, True, True, True]),
                Experience([0] * 4, 1, 3.0, [0] * 4, True, next_action_mask=[False] * 4),
            ]
        )
        self.assertAlmostEqual(metrics["meanTarget"], (1 + 0.99 * 7 + 3) / 2, places=5)
        self.assertGreater(metrics["loss"], 0)
        self.assertEqual(agent.train_steps, 1)
        np.testing.assert_array_equal(agent.target_b3, [[999, 7, 3, 1]])

    def test_huber_adam_clips_global_gradient_and_syncs_target(self) -> None:
        agent = self.agent(gradient_clip=0.25, target_sync_interval=1)
        agent.b1.fill(10)
        agent.b2.fill(100)
        original = agent.parameter_hash()
        result = agent.train_batch([Experience([10] * 4, 1, 1e6, [0] * 4, True)])
        self.assertGreater(result["gradientNorm"], 0.25)
        self.assertAlmostEqual(result["clippedGradientNorm"], 0.25)
        self.assertNotEqual(original, agent.parameter_hash())
        self.assertTrue(any(np.any(moment) for moment in agent.adam_m.values()))
        for name in PARAMETERS:
            np.testing.assert_array_equal(getattr(agent, name), getattr(agent, f"target_{name}"))

    def test_replay_ring_and_sampling_rng_round_trip(self) -> None:
        replay = ReplayBuffer(3, state_size=4, action_size=4, seed=12)
        for value in range(5):
            replay.add(Experience([value] * 4, value % 4, float(value), [value + 1] * 4, False))
        self.assertEqual(replay.size(), 3)
        self.assertEqual(set(replay.states[:, 0]), {2.0, 3.0, 4.0})
        metadata, arrays = replay.checkpoint()
        twin = ReplayBuffer.from_checkpoint(metadata, arrays)
        np.testing.assert_array_equal(replay.sample(20).states, twin.sample(20).states)

    def test_checkpoint_exact_weights_optimizer_rng_and_next_update(self) -> None:
        agent = self.agent()
        batch = [Experience([0.2] * 4, 2, 1.0, [0.4] * 4, False)]
        agent.train_batch(batch)
        for _ in range(10):
            agent.record_training_step()
            agent.act([0.1] * 4)
        path = self.root / "exact.npz"
        agent.save_to(path, {"budget": 10})
        restored = self.agent(seed=999)
        self.assertTrue(restored.load_from(path), restored.last_load_error)
        self.assertEqual(agent.parameter_hash(), restored.parameter_hash())
        self.assertEqual(agent.rng.bit_generator.state, restored.rng.bit_generator.state)
        self.assertEqual(agent.epsilon, restored.epsilon)
        for name in PARAMETERS:
            np.testing.assert_array_equal(agent.adam_m[name], restored.adam_m[name])
            np.testing.assert_array_equal(agent.adam_v[name], restored.adam_v[name])
            np.testing.assert_array_equal(getattr(agent, f"target_{name}"), getattr(restored, f"target_{name}"))
        self.assertEqual(agent.act([0.3] * 4), restored.act([0.3] * 4))
        agent.train_batch(batch)
        restored.train_batch(batch)
        self.assertEqual(agent.parameter_hash(), restored.parameter_hash())

    def test_corrupt_incompatible_and_nonfinite_loads_never_partially_mutate(self) -> None:
        agent = self.agent()
        path = self.root / "valid.npz"
        agent.save_to(path)
        with np.load(path, allow_pickle=False) as saved:
            original_arrays = {name: saved[name].copy() for name in saved.files}
        variants = ("version", "shape", "nan", "optimizer", "rng", "float_schema", "legacy")
        for variant in variants:
            with self.subTest(variant=variant):
                arrays = {name: value.copy() for name, value in original_arrays.items()}
                metadata = json.loads(str(arrays["metadata"].item()))
                if variant == "version":
                    metadata["modelVersion"] = "old-six-actions"
                elif variant == "shape":
                    arrays["target_w3"] = arrays["target_w3"][:, :2]
                elif variant == "nan":
                    arrays["w2"][0, 0] = np.nan
                elif variant == "optimizer":
                    arrays["adam_v_b3"][0, 0] = -1
                elif variant == "rng":
                    metadata["rngState"] = {"invalid": True}
                elif variant == "float_schema":
                    metadata["schemaVersion"] = 2.0
                else:
                    arrays.pop("metadata")
                if variant != "legacy":
                    arrays["metadata"] = np.array(json.dumps(metadata))
                bad = self.root / f"{variant}.npz"
                np.savez_compressed(bad, **arrays)
                before = agent.parameter_hash()
                rng = copy.deepcopy(agent.rng.bit_generator.state)
                optimizer = {name: value.copy() for name, value in agent.adam_m.items()}
                self.assertFalse(agent.load_from(bad))
                self.assertEqual(before, agent.parameter_hash())
                self.assertEqual(rng, agent.rng.bit_generator.state)
                for name in PARAMETERS:
                    np.testing.assert_array_equal(optimizer[name], agent.adam_m[name])
        broken = self.root / "broken.npz"
        broken.write_bytes(b"not a zip checkpoint")
        self.assertFalse(agent.load_from(broken))

    def test_atomic_save_preserves_previous_and_refuses_legacy_overwrite(self) -> None:
        agent = self.agent()
        path = self.root / "valid.npz"
        agent.save_to(path)
        original = path.read_bytes()
        agent.w3[0, 0] = np.nan
        with self.assertRaises(ValueError):
            agent.save_to(path)
        self.assertEqual(path.read_bytes(), original)
        agent.w3[0, 0] = 0
        legacy = self.root / "model_weights.npz"
        np.savez(legacy, w1=np.zeros((12, 32)))
        legacy_bytes = legacy.read_bytes()
        with self.assertRaises(ValueError):
            agent.save_to(legacy)
        self.assertEqual(legacy_bytes, legacy.read_bytes())
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_default_pause_intro_and_queued_advance_are_no_ops(self) -> None:
        session = self.session()
        self.assertEqual(session.get_state()["level"], 4)
        self.assertTrue(session.get_state()["paused"])
        baseline = learning_fingerprint(session)
        session.tick(25)
        session.train_steps(256)
        self.assertEqual(baseline, learning_fingerprint(session))
        session.control({"intro": True, "paused": False})
        self.assertTrue(session.paused)
        baseline = learning_fingerprint(session)
        session.train_steps(512)
        session.tick()
        self.assertEqual(baseline, learning_fingerprint(session))
        with self.assertRaises(ValueError):
            session.control({"level": 1})

        class ObservedLock:
            def __init__(self) -> None:
                self.inner = threading.RLock()
                self.queued = threading.Event()

            def __enter__(self) -> ObservedLock:
                if threading.current_thread() is not threading.main_thread():
                    self.queued.set()
                self.inner.acquire()
                return self

            def __exit__(self, *unused: object) -> None:
                self.inner.release()

        lock = ObservedLock()
        session.lock = lock
        with lock:
            worker = threading.Thread(target=lambda: session.tick(16))
            worker.start()
            self.assertTrue(lock.queued.wait(2))
            session.control({"intro": False, "paused": False})
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(baseline, learning_fingerprint(session))
        session.tick()
        self.assertEqual(session.training_steps, 1)

    def test_actual_training_changes_weights_and_decays_per_transition(self) -> None:
        session = self.session()
        session.control({"level": 1, "paused": False})
        before = session.agent.parameter_hash()
        initial = session.agent.epsilon
        result = session.train_steps(512)
        self.assertEqual(result["trainingSteps"], 512)
        self.assertEqual(result["totalSteps"], 512)
        self.assertGreater(result["gradientSteps"], 0)
        self.assertGreater(result["bufferSize"], 0)
        self.assertNotEqual(before, session.agent.parameter_hash())
        self.assertAlmostEqual(session.agent.epsilon, initial * session.agent.epsilon_decay**512)
        self.assertTrue(np.isfinite(result["loss"]))

    def test_explicit_new_episode_changes_run_identity_not_learned_state(self) -> None:
        session = self.session()
        session.control({"level": 1, "paused": False})
        session.tick(3)
        episode = session.current_episode
        policy = session.agent.parameter_hash()
        training_steps = session.training_steps
        buffer_size = session.buffer.size()
        completed = session.completed_episodes
        result = session.control({"newEpisode": True})
        self.assertEqual(result["episode"], episode + 1)
        self.assertEqual(result["board"]["steps"], 0)
        self.assertEqual(session.completed_episodes, completed)
        self.assertEqual(session.training_steps, training_steps)
        self.assertEqual(session.buffer.size(), buffer_size)
        self.assertEqual(session.agent.parameter_hash(), policy)

    def test_evaluation_is_greedy_disjoint_and_live_state_immutable(self) -> None:
        session = self.session()
        session.control({"level": 1, "paused": False})
        session.train_steps(256)
        fingerprint = learning_fingerprint(session)
        attributes = {
            key: id(value) for key, value in session.__dict__.items()
            if key not in ("evaluation",)
        }
        result = session.evaluate(episodes=3)
        evaluation = result["evaluation"]
        self.assertEqual(fingerprint, learning_fingerprint(session))
        self.assertEqual(attributes, {
            key: id(value) for key, value in session.__dict__.items()
            if key not in ("evaluation",)
        })
        self.assertEqual(evaluation["episodes"], 3)
        self.assertEqual(evaluation["epsilon"], 0.0)
        self.assertEqual(evaluation["policyHash"], session.agent.parameter_hash())
        self.assertEqual(evaluation["trainingSteps"], 256)
        self.assertEqual(evaluation["successes"] + evaluation["failures"], 3)
        self.assertEqual(len(evaluation["results"]), 3)
        self.assertEqual(
            evaluation["meanSteps"], sum(result["steps"] for result in evaluation["results"]) / 3
        )
        self.assertFalse(set(evaluation["seeds"]) & session.training_seeds)
        with self.assertRaises(ValueError):
            session.evaluate(seeds=[session.episode_seed])
        with self.assertRaises(ValueError):
            session.evaluate(seeds=[3_000_001_000] * 2)

    def test_eval_playback_never_updates_policy_replay_epsilon_or_rng(self) -> None:
        session = self.session()
        session.control({"level": 1, "mode": "eval", "paused": False})
        policy = session.agent.parameter_hash()
        rng = copy.deepcopy(session.agent.rng.bit_generator.state)
        epsilon = session.agent.epsilon
        result = session.tick(40)
        self.assertEqual(result["totalSteps"], 40)
        self.assertEqual(result["trainingSteps"], 0)
        self.assertEqual(result["bufferSize"], 0)
        self.assertEqual(policy, session.agent.parameter_hash())
        self.assertEqual(rng, session.agent.rng.bit_generator.state)
        self.assertEqual(epsilon, session.agent.epsilon)
        self.assertEqual(result["epsilon"], 0.0)
        self.assertFalse(result["isExploring"])

    def test_ready_policy_uses_original_observations_on_readable_arena_without_mutation(self) -> None:
        session = self.session()
        metadata, _ = session.agent.read_checkpoint(READY_MODEL_FILE)
        self.assertEqual(metadata["inputLabels"], list(OBSERVATION_LABELS))
        self.assertEqual(metadata["architecture"], session.agent.dimensions)
        asset = hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest()
        session.load_ready_weights()
        before = learning_fingerprint(session)
        record = session.evaluate(seeds=[3_000_620_001, 3_000_620_002])["evaluation"]
        self.assertEqual(before, learning_fingerprint(session))
        self.assertEqual(record["environmentVersion"], ENVIRONMENT_VERSION)
        self.assertEqual(record["policyHash"], session.agent.parameter_hash())
        self.assertEqual(record["episodes"], 2)
        self.assertEqual(record["successes"] + record["failures"], 2)
        self.assertEqual(hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest(), asset)

    def test_batch_cap_history_and_exactly_once_terminal_accounting(self) -> None:
        with patch(
            "backend.session.SnakeEnvironment",
            side_effect=lambda **options: SnakeEnvironment(**options, step_limit=1),
        ):
            session = self.session()
            session.control({"level": 1, "paused": False})
            result = session.train_steps(245)
            self.assertEqual(result["trainingSteps"], 245)
            self.assertEqual(result["bufferSize"], 245)
            self.assertEqual(session.completed_episodes, 245)
            self.assertEqual(len(result["chartPoints"]), 240)
            self.assertEqual(len({point["episode"] for point in result["chartPoints"]}), 240)
            self.assertTrue(result["episodeEnded"])
            self.assertEqual(result["board"]["steps"], 1)
            self.assertEqual(result["delayMs"], 1200)
            baseline = learning_fingerprint(session)
            session.get_state()
            session.control({"paused": True})
            session.tick(20)
            self.assertEqual(baseline, learning_fingerprint(session))
            session.control({"paused": False})
            previous_episode = session.current_episode
            next_episode = session.tick()
            self.assertEqual(next_episode["episode"], previous_episode + 1)
            self.assertEqual(next_episode["trainingSteps"], 246)
        session.control({"mode": "eval"})
        before = session.total_steps
        session.tick(2500)
        self.assertEqual(session.total_steps - before, 2048)
        self.assertEqual(session.training_steps, 246)

    def test_physical_contact_replay_and_snapshot_reads_never_repeat_learning(self) -> None:
        session = self.session()
        session.control({"level": 1, "paused": False})
        session.min_buffer_for_training = session.batch_size = session.train_every = 1
        session.agent.epsilon = 0.0
        for name in PARAMETERS:
            getattr(session.agent, name).fill(0)
        session.agent.b3[0, 0] = 100
        head = session.env.snake[0]
        session.env.obstacles = [Point(head.x, head.y, head.z - 1)]
        session.current_state = session.env.get_state()
        policy = session.agent.parameter_hash()
        terminal = session.tick()
        self.assertTrue(terminal["episodeEnded"])
        self.assertEqual(terminal["board"]["contact"]["outcome"], "death")
        self.assertEqual((session.total_steps, session.training_steps, session.agent.train_steps), (1, 1, 1))
        self.assertEqual((session.completed_episodes, session.buffer.size()), (1, 1))
        self.assertNotEqual(session.agent.parameter_hash(), policy)
        session.control({"paused": True})
        frozen = learning_fingerprint(session)
        for _ in range(3):
            self.assertEqual(session.get_state()["board"], terminal["board"])
            self.assertTrue(session.env.step(0)["info"]["terminalAlreadyProcessed"])
            session.tick(8)
            session.train_steps(8)
            self.assertEqual(learning_fingerprint(session), frozen)
        self.assertEqual(sum(event["type"] == "death" for event in session.env.events), 1)

    def test_decision_network_sample_alignment_and_bounded_payload(self) -> None:
        session = self.session()
        session.control({"level": 1, "paused": False})
        session.tick()
        cached = copy.deepcopy(session.agent.last_decision)
        result = session.train_steps(256)
        network = result["network"]
        self.assertEqual(network["dimensions"], [OBSERVATION_SIZE, 96, 96, ACTION_SIZE])
        self.assertEqual(network["lastAction"], result["lastAction"])
        self.assertEqual(network["isExploring"], result["isExploring"])
        self.assertEqual(len(network["qValues"]), ACTION_SIZE)
        self.assertEqual(len(network["actionMask"]), ACTION_SIZE)
        self.assertEqual(len(network["inputLabels"]), len(network["layers"][0]))
        self.assertEqual(len(network["outputLabels"]), len(network["layers"][-1]))
        for index, weights in enumerate(network["weights"]):
            self.assertLessEqual(len(network["layers"][index]), 20)
            self.assertEqual(len(weights), len(network["layers"][index]))
            self.assertTrue(all(len(row) == len(network["layers"][index + 1]) for row in weights))
        self.assertLess(len(json.dumps(network)), 120_000)
        self.assertIsNotNone(cached)
        before = copy.deepcopy(network)
        session.agent.train_batch(session.buffer.sample(8))
        self.assertEqual(before, session.get_state()["network"])
        self.assertIsNone(session.get_state(include_network=False)["network"])

    def test_session_checkpoint_exact_board_replay_rng_and_continuation(self) -> None:
        session = self.session()
        session.control({"level": 1, "paused": False})
        session.train_steps(400)
        session.evaluate(episodes=2)
        checkpoint = self.root / "full-session.npz"
        session.save_checkpoint(checkpoint)
        before = learning_fingerprint(session)
        twin = TrainingSession(storage_dir=self.root / "twin", seed=999, load_saved=False)
        self.assertTrue(twin.load_checkpoint(checkpoint), twin.log_message)
        self.assertEqual(before, learning_fingerprint(twin))
        twin.control({"paused": False})
        session.train_steps(32)
        twin.train_steps(32)
        self.assertEqual(learning_fingerprint(session), learning_fingerprint(twin))

    def test_environment_namespaced_saves_and_reset_preserve_older_same_model_sessions(self) -> None:
        old_directory = self.root / MODEL_VERSION
        old_checkpoint = old_directory / "latest.npz"
        source = self.session(seed=17)
        source.control({"level": 1, "paused": False})
        source.train_steps(8)
        source.save_checkpoint(old_checkpoint)
        with np.load(old_checkpoint, allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        metadata = json.loads(str(arrays["metadata"].item()))
        metadata["trainingMetadata"]["session"]["environmentVersion"] = "survival-3d-v4"
        arrays["metadata"] = np.array(json.dumps(metadata))
        np.savez_compressed(old_checkpoint, **arrays)
        old_bytes = old_checkpoint.read_bytes()
        for name in ("manual.npz", "best.npz"):
            (old_directory / name).write_bytes(old_bytes)
        legacy = self.root / "manual_model_weights.npz"
        legacy.write_bytes(b"untouched legacy weights")
        preserved = {path: path.read_bytes() for path in
                     (old_checkpoint, old_directory / "manual.npz", old_directory / "best.npz", legacy)}
        ready_hash = hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest()

        def assert_preserved() -> None:
            for path, original in preserved.items():
                self.assertTrue(path.is_file(), f"Moved or removed older file: {path.name}")
                self.assertTrue(path.read_bytes() == original, f"Changed older file: {path.name}")
            self.assertEqual(hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest(), ready_hash)

        current = TrainingSession(storage_dir=self.root, seed=31, load_saved=True)
        self.assertEqual(current.training_steps, 0)
        self.assertFalse(current.load_checkpoint(old_checkpoint))
        self.assertIn("survival-3d-v4", current.log_message)
        assert_preserved()
        current.control({"level": 1, "paused": False})
        current.save_every_steps = 1
        current.tick()
        assert_preserved()
        current.save_manual_checkpoint()
        current.save_checkpoint(current.best_model_file)
        assert_preserved()
        self.assertEqual(current.storage_dir, old_directory / ENVIRONMENT_VERSION)
        for path in (current.model_file, current.manual_model_file, current.best_model_file):
            self.assertTrue(path.is_file())

        twin = TrainingSession(storage_dir=self.root / "twin", seed=999, load_saved=False)
        self.assertTrue(twin.load_checkpoint(current.manual_model_file), twin.log_message)
        self.assertEqual(learning_fingerprint(current), learning_fingerprint(twin))
        resumed = TrainingSession(storage_dir=self.root, seed=999, load_saved=True)
        self.assertEqual(resumed.training_steps, current.training_steps)
        self.assertEqual(resumed.agent.parameter_hash(), current.agent.parameter_hash())
        self.assertTrue(resumed.paused)
        assert_preserved()

        current_bytes = current.manual_model_file.read_bytes()
        current.reset_training()
        assert_preserved()
        archived = list((current.storage_dir / "archive").glob("*\\manual.npz"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), current_bytes)
        current.tick()
        current.save_manual_checkpoint()
        assert_preserved()
        self.assertTrue(current.model_file.is_file())
        self.assertTrue(current.manual_model_file.is_file())

    def test_invalid_session_metadata_cannot_replace_valid_policy(self) -> None:
        session = self.session()
        session.control({"level": 1, "paused": False})
        session.train_steps(128)
        valid = self.root / "full.npz"
        session.save_checkpoint(valid)
        with np.load(valid, allow_pickle=False) as saved:
            original_arrays = {name: saved[name].copy() for name in saved.files}
        for variant in ("rng", "chart", "evaluation", "last-decision", "environment"):
            with self.subTest(variant=variant):
                arrays = {name: value.copy() for name, value in original_arrays.items()}
                metadata = json.loads(str(arrays["metadata"].item()))
                saved_session = metadata["trainingMetadata"]["session"]
                if variant == "rng":
                    saved_session["sessionRngState"] = {"not": "an RNG"}
                elif variant == "chart":
                    saved_session["chartPoints"] = [
                        {
                            "episode": 1, "score": "broken", "food": 0,
                            "survivalSeconds": 0.12, "mode": "train", "level": 1,
                            "fullSurvival": False,
                        }
                    ]
                elif variant == "evaluation":
                    saved_session["evaluation"] = {"episodes": 2, "results": "broken"}
                elif variant == "environment":
                    saved_session["environmentVersion"] = "survival-3d-v4"
                else:
                    saved_session["lastDecision"]["qValues"][0] = "NaN"
                arrays["metadata"] = np.array(json.dumps(metadata))
                invalid = self.root / f"bad-session-{variant}.npz"
                np.savez_compressed(invalid, **arrays)
                fingerprint = learning_fingerprint(session)
                self.assertFalse(session.load_checkpoint(invalid))
                self.assertEqual(fingerprint, learning_fingerprint(session))
                if variant == "environment":
                    self.assertIn("survival-3d-v4", session.log_message)
                    self.assertIn(ENVIRONMENT_VERSION, session.log_message)
                    compatible = self.agent(state_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)
                    self.assertTrue(compatible.load_from(invalid), compatible.last_load_error)

    def test_curriculum_requires_real_goals_sample_size_and_boundary(self) -> None:
        session = self.session()
        session.control({"autoCurriculum": True, "paused": False})
        self.assertEqual(session.level, 1)
        session.train_steps(128)
        self.assertEqual(session.level, 1, "Episode count alone cannot promote the learner.")
        goals = Curriculum.goals(1)
        successes = [
            {"food": goals["food"], "steps": goals["survivalSteps"], "enemyEscapes": goals["escapes"]}
            for _ in range(8)
        ]
        record = {"episodes": 8, "level": 1, "fullSurvival": False, "seeds": list(range(100, 108)), "results": successes, "successes": 8}
        small = {**record, "episodes": 2, "seeds": [100, 101], "results": successes[:2], "successes": 2}
        self.assertFalse(session.curriculum.consider(small))
        failed = copy.deepcopy(record)
        for result in failed["results"]:
            result["food"] = 0
        failed["successes"] = 0
        self.assertFalse(session.curriculum.consider(failed))
        self.assertTrue(session.curriculum.consider(record))
        self.assertEqual(session.level, 1)
        self.assertEqual(session.curriculum.pending_level, 2)
        session.control({"newEpisode": True})
        self.assertEqual(session.level, 2)
        self.assertIsNone(session.curriculum.pending_level)

    def test_reset_archives_only_new_format_and_missing_load_has_no_starter(self) -> None:
        session = self.session()
        legacy = self.root / "manual_model_weights.npz"
        legacy.write_bytes(b"legacy model must remain exact")
        before = session.agent.parameter_hash()
        result = session.load_saved_or_best_checkpoint()
        self.assertEqual(before, session.agent.parameter_hash())
        self.assertIn("no starter", result["logMessage"])
        session.save_manual_checkpoint()
        self.assertTrue(session.manual_model_file.exists())
        session.model_file.write_bytes(b"unrecognized older file in the new directory")
        session.reset_training()
        self.assertFalse(session.manual_model_file.exists())
        self.assertEqual(len(list((session.storage_dir / "archive").glob("*\\manual.npz"))), 1)
        self.assertEqual(legacy.read_bytes(), b"legacy model must remain exact")
        self.assertEqual(session.model_file.read_bytes(), b"unrecognized older file in the new directory")
        self.assertFalse(session.paused)
        self.assertEqual(session.tick()["trainingSteps"], 1)

    def test_ready_training_refuses_to_overwrite_source_or_protected_model_with_report(self) -> None:
        from backend import train_ready

        source = self.root / "source.npz"
        protected = self.root / "storage" / "pretrained_3d.npz"
        protected.parent.mkdir()
        source.write_bytes(b"reserved source")
        protected.write_bytes(b"reserved original ready")
        candidate, report = self.root / "candidate.npz", self.root / "report.json"
        before = {path: path.read_bytes() for path in (source, protected)}
        for output, result in ((source, report), (protected, report), (candidate, source),
                               (candidate, candidate), (candidate, protected)):
            with self.subTest(output=output, report=result), patch.object(
                train_ready, "__file__", str(self.root / "backend" / "train_ready.py")
            ), patch("sys.stderr"), self.assertRaises(SystemExit) as failure:
                train_ready.main(["--source", str(source), "--output", str(output),
                                  "--report", str(result), "--steps", "1024"])
            self.assertEqual(failure.exception.code, 2)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)
            self.assertFalse(candidate.exists())

    def test_headless_public_train_and_evaluate_commands(self) -> None:
        checkpoint = self.root / "headless.npz"
        train = subprocess.run(
            [
                sys.executable, "-B", "-m", "backend.train", "--steps", "384",
                "--seed", "31", "--level", "1", "--output", str(checkpoint),
                "--eval-episodes", "2",
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=90, check=False,
        )
        self.assertEqual(train.returncode, 0, train.stderr)
        report = json.loads(train.stdout)
        self.assertEqual(report["transitions"], 384)
        self.assertGreater(report["gradientUpdates"], 0)
        self.assertTrue(report["weightChanged"])
        self.assertTrue(report["seedsDisjoint"])
        self.assertGreater(report["transitionsPerSecond"], 0)
        self.assertTrue(checkpoint.exists())
        evaluate = subprocess.run(
            [
                sys.executable, "-B", "-m", "backend.evaluate", "--checkpoint", str(checkpoint),
                "--seeds", "3000000001,3000000002",
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=90, check=False,
        )
        self.assertEqual(evaluate.returncode, 0, evaluate.stderr)
        result = json.loads(evaluate.stdout)
        self.assertEqual(result["seeds"], [3_000_000_001, 3_000_000_002])
        self.assertTrue(result["policyUnchanged"])
        self.assertTrue(result["seedsDisjoint"])
        self.assertEqual(result["modelVersion"], MODEL_VERSION)
        resume = subprocess.run(
            [
                sys.executable, "-B", "-m", "backend.train", "--steps", "32",
                "--resume", str(checkpoint), "--output", str(checkpoint),
                "--eval-episodes", "2",
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=90, check=False,
        )
        self.assertEqual(resume.returncode, 0, resume.stderr)
        resumed = json.loads(resume.stdout)
        self.assertEqual(resumed["trainingSteps"], 416)
        self.assertEqual(resumed["seed"], 31)


if __name__ == "__main__":
    unittest.main()
