from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient

from backend.agent import DQNAgent
from backend.session import TrainingSession
from backend.config import ENVIRONMENT_VERSION, MODEL_VERSION, REWARD_VERSION
from backend.environment import ACTION_SIZE, Point
from main import create_app


class ArenaAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = Path(__file__).resolve().parents[1] / ".runtime-proof" / f"api-test-{uuid.uuid4().hex}"
        self.storage.mkdir(parents=True)
        self.session = TrainingSession(storage_dir=self.storage, seed=71, load_saved=False)
        self.client = TestClient(create_app(self.session))

    def tearDown(self) -> None:
        self.client.close()
        shutil.rmtree(self.storage)

    def post(self, url: str, data: dict | None = None) -> dict:
        response = self.client.post(url, json=data) if data is not None else self.client.post(url)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def state(self) -> dict:
        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        return response.json()

    def test_initial_page_is_paused_and_reading_state_never_steps(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.headers["cache-control"], "no-cache")
        script = self.client.get("/static/app.js?v=readable-v6")
        self.assertEqual(script.status_code, 200)
        self.assertEqual(script.headers["cache-control"], "no-cache",
                         "Related module exports must not be mixed with a stale cached bundle.")
        self.assertEqual(self.client.get("/api/health").json()["modelVersion"], MODEL_VERSION)
        self.assertEqual(self.client.get("/api/health").json()["rewardVersion"], REWARD_VERSION)
        before, after = self.state(), self.state()
        self.assertTrue(before["paused"])
        self.assertEqual(before["level"], 4)
        self.assertTrue({"hunter", "rival"} <= {enemy["type"] for enemy in before["board"]["enemies"]})
        self.assertEqual(before["board"], after["board"])
        self.assertEqual(self.post("/api/tick")["board"], before["board"])
        self.assertEqual(self.post("/api/train", {"steps": 64})["totalSteps"], before["totalSteps"])

    def test_intro_excludes_all_advancement_until_explicit_unlock(self) -> None:
        self.post("/api/control", {"paused": False})
        self.post("/api/train", {"steps": 32})
        frozen = self.post("/api/control", {"intro": True, "paused": True})
        for endpoint in ("/api/tick", "/api/train"):
            state = self.post(endpoint, {"steps": 128})
            for key in ("board", "totalSteps", "trainingSteps", "bufferSize", "epsilon"):
                self.assertEqual(state[key], frozen[key], key)
        self.post("/api/control", {"intro": False, "paused": False})
        advanced = self.post("/api/train", {"steps": 16})
        self.assertEqual(advanced["totalSteps"] - frozen["totalSteps"], 16)

    def test_fast_training_advances_real_steps_and_updates_network(self) -> None:
        before = self.post("/api/control", {"level": 1, "newEpisode": True, "paused": False})
        after = self.post("/api/train", {"steps": 512})
        self.assertEqual(after["totalSteps"] - before["totalSteps"], 512)
        self.assertGreater(after["trainingSteps"], before["trainingSteps"])
        self.assertGreater(after["bufferSize"], 0)
        self.assertLess(after["epsilon"], before["epsilon"])
        self.assertLess(len(json.dumps(after)), 120_000)
        network = after["network"]
        self.assertEqual(len(network["outputLabels"]), ACTION_SIZE)
        self.assertEqual(len(network["qValues"]), ACTION_SIZE)
        self.assertEqual(len(network["actionMask"]), ACTION_SIZE)
        self.assertEqual(len(network["layers"]), len(network["dimensions"]))
        for layer in network["layers"]:
            self.assertLessEqual(len(layer), 20)
        for index, weights in enumerate(network["weights"]):
            self.assertEqual(len(weights), len(network["layers"][index]))
            self.assertTrue(all(len(row) == len(network["layers"][index + 1]) for row in weights))

    def test_survival_reward_reaches_replay_without_deducting_food_score(self) -> None:
        self.post("/api/control", {"level": 4, "paused": False})
        for _ in range(24):
            state = self.post("/api/tick")
            board = state["board"]
            reward = board["trainingReward"]
            self.assertEqual(state["rewardVersion"], REWARD_VERSION)
            self.assertEqual(reward["version"], REWARD_VERSION)
            self.assertAlmostEqual(sum(reward["components"].values()), reward["total"])
            self.assertAlmostEqual(sum(reward["shaping"].values()), reward["components"]["potential"])
            latest = (self.session.buffer.position - 1) % self.session.buffer.max_size
            self.assertAlmostEqual(float(self.session.buffer.rewards[latest]), reward["total"], places=5)
            food_points = sum(event["value"] for event in board["events"]
                              if event["type"] in {"food", "gold_food"})
            self.assertEqual(state["currentScore"], food_points)

    def test_greedy_playback_does_not_train_or_consume_exploration_schedule(self) -> None:
        before = self.post("/api/control", {"mode": "eval", "paused": False})
        after = self.post("/api/tick")
        for key in ("trainingSteps", "bufferSize", "epsilon"):
            self.assertEqual(after[key], before[key], key)
        self.assertFalse(after["isExploring"])
        self.assertEqual(after["board"]["steps"], before["board"]["steps"] + 1)

    def test_terminal_contact_survives_paused_reads_without_duplicate_accounting(self) -> None:
        self.post("/api/control", {"level": 1, "mode": "eval", "paused": False})
        env = self.session.env
        head = env.snake[0]
        env.obstacles = [Point(head.x + 1, head.y, head.z)]
        for parameter in ("w1", "b1", "w2", "b2", "w3", "b3"):
            getattr(self.session.agent, parameter).fill(0)
        self.session.agent.b3[0, 1] = 100
        self.session.current_state = env.get_state()
        terminal = self.post("/api/tick")
        self.assertTrue(terminal["episodeEnded"])
        self.assertEqual(terminal["environmentVersion"], ENVIRONMENT_VERSION)
        contact = terminal["board"]["contact"]
        self.assertEqual((contact["kind"], contact["outcome"]), ("obstacle", "death"))
        self.assertEqual(terminal["board"]["events"][-1]["contact"], contact)
        paused = self.post("/api/control", {"paused": True})
        rng = env.rng.getstate()
        for state in (self.state(), self.post("/api/tick"), self.post("/api/train", {"steps": 64})):
            self.assertEqual(state["board"], paused["board"])
            for key in ("completedEpisodes", "totalSteps", "trainingSteps", "bufferSize", "chartPoints"):
                self.assertEqual(state[key], terminal[key], key)
            self.assertEqual(state["episodeKey"], terminal["episodeKey"])
            json.dumps(state["board"]["contact"], allow_nan=False)
        self.assertEqual(env.rng.getstate(), rng)
        restarted = self.post("/api/control", {"newEpisode": True})
        self.assertNotEqual(restarted["episodeKey"], terminal["episodeKey"])
        self.assertIsNone(restarted["board"]["contact"])
        self.assertEqual(restarted["board"]["steps"], 0)

    def test_held_out_evaluation_preserves_live_training_state(self) -> None:
        self.post("/api/control", {"level": 1, "paused": False, "newEpisode": True})
        self.post("/api/train", {"steps": 48})
        before = self.post("/api/control", {"paused": True})
        after = self.post("/api/evaluate", {"episodes": 2, "seeds": [100701, 100702]})
        for key in ("board", "totalSteps", "trainingSteps", "bufferSize", "epsilon", "paused"):
            self.assertEqual(after[key], before[key], key)
        self.assertIsNotNone(after["evaluation"])
        self.assertEqual(after["evaluation"]["seeds"], [100701, 100702])

    def test_full_survival_configuration_contains_actual_entities(self) -> None:
        state = self.post("/api/control", {
            "level": 6, "fullSurvival": True, "autoCurriculum": False,
            "newEpisode": True, "paused": True,
        })
        board = state["board"]
        self.assertEqual(state["level"], 6)
        self.assertTrue(board["fullSurvival"])
        types = {enemy["type"] for enemy in board["enemies"]}
        self.assertTrue({"hunter", "patrol", "rival"} <= types)
        self.assertTrue(board["gates"])
        self.assertTrue(board["movingWalls"])
        self.assertIn("ticksUntilShrink", board["zone"])
        self.assertEqual(len(board["abilities"]), 2)

    def test_returned_default_evaluation_seeds_can_be_replayed_through_api(self) -> None:
        first = self.post("/api/evaluate", {"episodes": 1})["evaluation"]
        replay = self.post("/api/evaluate", {"seeds": first["seeds"]})["evaluation"]
        self.assertEqual(replay["results"], first["results"])
        self.assertEqual(replay["policyHash"], first["policyHash"])

    def test_save_and_load_round_trip_does_not_touch_legacy_files(self) -> None:
        legacy = self.storage / "manual_model_weights.npz"
        legacy.write_bytes(b"untouched legacy checkpoint")
        self.post("/api/control", {"level": 1, "paused": False, "newEpisode": True})
        self.post("/api/train", {"steps": 512})
        saved = self.post("/api/save-weights")
        self.post("/api/train", {"steps": 16})
        loaded = self.post("/api/load-weights")
        self.assertEqual(loaded["trainingSteps"], saved["trainingSteps"])
        self.assertEqual(legacy.read_bytes(), b"untouched legacy checkpoint")
        self.assertIn("load", loaded["logMessage"].lower())

    def test_invalid_requests_leave_state_unchanged(self) -> None:
        before = self.state()
        requests = [
            ("/api/tick", {"steps": 0}),
            ("/api/train", {"steps": 2049}),
            ("/api/tick", {"unrecognized": True}),
            ("/api/control", {"level": 7}),
            ("/api/control", {"mode": "scripted"}),
            ("/api/control", {"seed": -1}),
            ("/api/evaluate", {"seeds": [2, 2]}),
            ("/api/evaluate", {"episodes": 33}),
        ]
        for endpoint, body in requests:
            with self.subTest(endpoint=endpoint, body=body):
                self.assertEqual(self.client.post(endpoint, json=body).status_code, 422)
        after = self.state()
        self.assertEqual(after["board"], before["board"])
        self.assertEqual(after["totalSteps"], before["totalSteps"])

    def test_ready_weights_start_greedy_3d_playback_without_overwriting_saves(self) -> None:
        self.post("/api/save-weights")
        saved = self.session.manual_model_file.read_bytes()
        self.post("/api/control", {"fullSurvival": True})
        initial_policy = self.session.agent.parameter_hash()
        loaded = self.post("/api/load-ready-weights")
        self.assertNotEqual(initial_policy, self.session.agent.parameter_hash())
        self.assertEqual(loaded["mode"], "eval")
        self.assertEqual(loaded["epsilon"], 0)
        self.assertFalse(loaded["paused"])
        self.assertTrue(loaded["fullSurvival"])
        self.assertGreater(self.session.agent.train_steps, 0)
        self.assertEqual(loaded["bufferSize"], 0)
        self.assertEqual(self.session.manual_model_file.read_bytes(), saved)
        self.assertIn("Loaded ready 3D weights", loaded["logMessage"])
        policy = self.session.agent.parameter_hash()
        advanced = self.post("/api/tick", {"steps": 16})
        self.assertGreater(advanced["totalSteps"], loaded["totalSteps"])
        self.assertEqual(advanced["trainingSteps"], loaded["trainingSteps"])
        self.assertEqual(policy, self.session.agent.parameter_hash())
        self.assertTrue(any(cell["y"] > 0 for cell in advanced["board"]["snake"]))
        from backend.session import READY_MODEL_FILE
        ready_hash = hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest()
        self.post("/api/reset")
        self.assertEqual(hashlib.sha256(READY_MODEL_FILE.read_bytes()).hexdigest(), ready_hash)

    def test_missing_or_incompatible_ready_weights_report_failure_without_mutation(self) -> None:
        before = self.state()
        policy = self.session.agent.parameter_hash()
        incompatible = self.storage / "incompatible.npz"
        incompatible.write_bytes(b"not a checkpoint")
        old_reward = Path(__file__).resolve().parents[1] / "storage" / "pretrained_3d.npz"
        wrong_arena = self.storage / "wrong-arena.npz"
        fixture = DQNAgent(wrong_arena)
        self.assertTrue(fixture.load_from(old_reward), fixture.last_load_error)
        fixture.save_to(wrong_arena, {"rewardVersion": REWARD_VERSION, "environmentVersion": "obsolete-arena"})
        for path in (self.storage / "missing.npz", incompatible, old_reward, wrong_arena):
            with self.subTest(path=path), patch("backend.session.READY_MODEL_FILE", path):
                response = self.client.post("/api/load-ready-weights")
                self.assertEqual(response.status_code, 400)
                self.assertIn("live learner preserved", response.json()["detail"])
                if path == old_reward:
                    self.assertIn("current survival rewards", response.json()["detail"])
                if path == wrong_arena:
                    self.assertIn("current arena", response.json()["detail"])
                self.assertEqual(policy, self.session.agent.parameter_hash())
                self.assertEqual(self.state()["board"], before["board"])
                self.assertEqual(self.state()["trainingSteps"], before["trainingSteps"])

    def test_reset_during_evaluation_intro_resumes_basic_training_automatically(self) -> None:
        self.post("/api/load-ready-weights")
        trained_policy = self.session.agent.parameter_hash()
        self.post("/api/control", {"intro": True, "paused": True})
        reset = self.post("/api/reset")
        self.assertFalse(reset["paused"])
        self.assertFalse(reset["introActive"])
        self.assertEqual(reset["mode"], "train")
        self.assertEqual(reset["epsilon"], 1)
        self.assertEqual(reset["trainingSteps"], 0)
        self.assertNotEqual(trained_policy, self.session.agent.parameter_hash())
        advanced = self.post("/api/tick", {"steps": 16})
        self.assertEqual(advanced["trainingSteps"], 16)
        self.assertEqual(advanced["bufferSize"], 16)
        self.assertGreater(advanced["totalSteps"], reset["totalSteps"])
        for parameter in ("w1", "b1", "w2", "b2", "w3", "b3"):
            getattr(self.session.agent, parameter).fill(0)
        self.assertEqual(self.post("/api/tick", {"steps": 4})["trainingSteps"], 20)


if __name__ == "__main__":
    unittest.main()
