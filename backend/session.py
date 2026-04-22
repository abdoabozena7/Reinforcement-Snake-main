from __future__ import annotations

import json
from pathlib import Path
import threading

from backend.agent import DQNAgent, Experience, ReplayBuffer
from backend.environment import SnakeEnvironment


BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
STATE_FILE = STORAGE_DIR / "training_state.json"
MODEL_FILE = STORAGE_DIR / "model_weights.npz"


class TrainingSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.batch_size = 64
        self.min_buffer_for_training = 500
        self.save_every = 5
        self.reset_runtime(load_saved=True)

    def reset_runtime(self, load_saved: bool) -> None:
        self.env = SnakeEnvironment()
        self.agent = DQNAgent(model_path=MODEL_FILE)
        self.buffer = ReplayBuffer(10_000)
        self.completed_episodes = 0
        self.current_episode = 0
        self.high_score = 0
        self.score_history: list[int] = []
        self.chart_points: list[dict[str, float | int]] = []
        self.pending_reset = False
        self.log_message = "Booting engine... initializing Python DQN service."

        loaded_model = self.agent.load() if load_saved else False
        if load_saved:
            self._load_state()
        self._start_new_episode()
        if loaded_model:
            self.log_message = f"Resumed training from episode {self.completed_episodes}."

    def _start_new_episode(self) -> None:
        self.current_episode = self.completed_episodes + 1
        self.current_state = self.env.reset()
        self.pending_reset = False

    def _average_score(self) -> float:
        window = self.score_history[-20:]
        if not window:
            return 0.0
        return float(sum(window) / len(window))

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        self.completed_episodes = int(data.get("completedEpisodes", 0))
        self.high_score = int(data.get("highScore", 0))
        self.agent.epsilon = float(data.get("epsilon", 1.0))
        self.score_history = [int(score) for score in data.get("scoreHistory", [])]
        self.chart_points = [
            {
                "episode": int(point["episode"]),
                "score": int(point["score"]),
                "epsilon": float(point["epsilon"]),
                "avgScore": float(point["avgScore"]),
            }
            for point in data.get("chartPoints", [])
            if {"episode", "score", "epsilon", "avgScore"} <= set(point)
        ]

    def _save_state(self) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "completedEpisodes": self.completed_episodes,
            "highScore": self.high_score,
            "epsilon": self.agent.epsilon,
            "scoreHistory": self.score_history,
            "chartPoints": self.chart_points,
        }
        STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.agent.save()

    def _chart_point(self) -> dict[str, float | int]:
        return {
            "episode": self.current_episode,
            "score": self.env.score,
            "epsilon": round(self.agent.epsilon, 3),
            "avgScore": round(self._average_score(), 2),
        }

    def _snapshot(self, include_history: bool = False, chart_point: dict[str, float | int] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "episode": self.current_episode,
            "highScore": self.high_score,
            "epsilon": round(self.agent.epsilon, 3),
            "currentScore": self.env.score,
            "avgScore": round(self._average_score(), 2),
            "bufferSize": self.buffer.size(),
            "logMessage": self.log_message,
            "board": self.env.board(),
            "network": self.agent.network_snapshot(self.current_state),
            "episodeEnded": self.pending_reset,
            "delayMs": 250 if self.pending_reset else 0,
            "chartPoint": chart_point,
        }
        if include_history:
            payload["chartPoints"] = self.chart_points
        return payload

    def get_state(self) -> dict[str, object]:
        with self.lock:
            return self._snapshot(include_history=True)

    def tick(self) -> dict[str, object]:
        with self.lock:
            if self.pending_reset:
                self._start_new_episode()

            action = self.agent.act(self.current_state)
            step_result = self.env.step(action)
            next_state = step_result["next_state"]
            reward = float(step_result["reward"])
            done = bool(step_result["done"])

            self.buffer.add(
                Experience(
                    state=self.current_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

            if self.buffer.size() >= self.min_buffer_for_training:
                batch = self.buffer.sample(self.batch_size)
                self.agent.train_batch(batch)

            self.current_state = next_state
            chart_point = None

            if done:
                self.high_score = max(self.high_score, self.env.score)
                self.score_history.append(self.env.score)
                self.completed_episodes = self.current_episode
                self.agent.decay_epsilon()
                chart_point = self._chart_point()
                self.chart_points.append(chart_point)
                self.pending_reset = True
                self.log_message = f"Episode {self.current_episode} ended with score {self.env.score}."

                if self.completed_episodes % self.save_every == 0:
                    self._save_state()
            else:
                self.log_message = f"Episode {self.current_episode} running. Score {self.env.score}."

            return self._snapshot(chart_point=chart_point)

    def reset_training(self) -> dict[str, object]:
        with self.lock:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            self.agent.clear_memory()
            self.reset_runtime(load_saved=False)
            return self._snapshot(include_history=True)


training_session = TrainingSession()
