from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np


@dataclass
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    def __init__(self, max_size: int = 10_000) -> None:
        self.max_size = max_size
        self.buffer: list[Experience] = []

    def add(self, experience: Experience) -> None:
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        return [random.choice(self.buffer) for _ in range(batch_size)]

    def size(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class DQNAgent:
    input_labels = [
        "Hazard Up",
        "Hazard Down",
        "Hazard Left",
        "Hazard Right",
        "Food Up",
        "Food Down",
        "Food Left",
        "Food Right",
    ]
    output_labels = ["Up", "Down", "Left", "Right"]

    def __init__(
        self,
        model_path: Path,
        state_size: int = 8,
        action_size: int = 4,
        hidden_size: int = 32,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_sync_interval: int = 200,
    ) -> None:
        self.model_path = model_path
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_sync_interval = target_sync_interval
        self.train_steps = 0
        self._init_weights()

    def _init_weights(self) -> None:
        self.w1 = (np.random.randn(self.state_size, self.hidden_size) * 0.1).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.w2 = (np.random.randn(self.hidden_size, self.hidden_size) * 0.1).astype(np.float32)
        self.b2 = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.w3 = (np.random.randn(self.hidden_size, self.action_size) * 0.1).astype(np.float32)
        self.b3 = np.zeros((1, self.action_size), dtype=np.float32)
        self._sync_target_network()
        self.train_steps = 0

    def _sync_target_network(self) -> None:
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b3 = self.b3.copy()

    def load(self) -> bool:
        return self.load_from(self.model_path)

    def load_from(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]
        if {"target_w1", "target_b1", "target_w2", "target_b2", "target_w3", "target_b3"} <= set(data.files):
            self.target_w1 = data["target_w1"]
            self.target_b1 = data["target_b1"]
            self.target_w2 = data["target_w2"]
            self.target_b2 = data["target_b2"]
            self.target_w3 = data["target_w3"]
            self.target_b3 = data["target_b3"]
        else:
            self._sync_target_network()
        self.train_steps = int(data["train_steps"]) if "train_steps" in data.files else 0
        return True

    def save(self) -> None:
        self.save_to(self.model_path)

    def save_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            target_w1=self.target_w1,
            target_b1=self.target_b1,
            target_w2=self.target_w2,
            target_b2=self.target_b2,
            target_w3=self.target_w3,
            target_b3=self.target_b3,
            train_steps=np.array(self.train_steps, dtype=np.int64),
        )

    def clear_memory(self) -> None:
        if self.model_path.exists():
            self.model_path.unlink()
        self._init_weights()
        self.epsilon = 1.0

    def _forward_with(
        self,
        states: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = states @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ w2 + b2
        a2 = np.maximum(z2, 0.0)
        q_values = a2 @ w3 + b3
        return z1, a1, z2, a2, q_values

    def _forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._forward_with(states, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)

    def predict(self, states: np.ndarray) -> np.ndarray:
        _, _, _, _, q_values = self._forward(states.astype(np.float32))
        return q_values

    def predict_target(self, states: np.ndarray) -> np.ndarray:
        _, _, _, _, q_values = self._forward_with(
            states.astype(np.float32),
            self.target_w1,
            self.target_b1,
            self.target_w2,
            self.target_b2,
            self.target_w3,
            self.target_b3,
        )
        return q_values

    def network_snapshot(self, state: list[float]) -> dict[str, object]:
        state_array = np.array([state], dtype=np.float32)
        _, a1, _, a2, q_values = self._forward(state_array)
        return {
            "layers": [
                np.round(state_array[0], 4).tolist(),
                np.round(a1[0], 4).tolist(),
                np.round(a2[0], 4).tolist(),
                np.round(q_values[0], 4).tolist(),
            ],
            "weights": [
                np.round(self.w1, 4).tolist(),
                np.round(self.w2, 4).tolist(),
                np.round(self.w3, 4).tolist(),
            ],
            "inputLabels": self.input_labels,
            "outputLabels": self.output_labels,
        }

    def act(self, state: list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_array = np.array([state], dtype=np.float32)
        q_values = self.predict(state_array)[0]
        return int(np.argmax(q_values))

    def train_batch(self, batch: list[Experience]) -> None:
        states = np.array([item.state for item in batch], dtype=np.float32)
        next_states = np.array([item.next_state for item in batch], dtype=np.float32)

        current_q = self.predict(states)
        next_online_q = self.predict(next_states)
        next_target_q = self.predict_target(next_states)
        targets = current_q.copy()
        best_actions = np.argmax(next_online_q, axis=1)

        for index, item in enumerate(batch):
            best_future = 0.0 if item.done else float(next_target_q[index, best_actions[index]])
            targets[index, item.action] = item.reward + self.gamma * best_future

        z1, a1, z2, a2, predictions = self._forward(states)
        batch_size = max(len(batch), 1)
        grad_output = (2.0 / batch_size) * (predictions - targets)

        grad_w3 = a2.T @ grad_output
        grad_b3 = grad_output.sum(axis=0, keepdims=True)

        grad_a2 = grad_output @ self.w3.T
        grad_z2 = grad_a2 * (z2 > 0)
        grad_w2 = a1.T @ grad_z2
        grad_b2 = grad_z2.sum(axis=0, keepdims=True)

        grad_a1 = grad_z2 @ self.w2.T
        grad_z1 = grad_a1 * (z1 > 0)
        grad_w1 = states.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0, keepdims=True)

        self.w3 -= self.learning_rate * grad_w3.astype(np.float32)
        self.b3 -= self.learning_rate * grad_b3.astype(np.float32)
        self.w2 -= self.learning_rate * grad_w2.astype(np.float32)
        self.b2 -= self.learning_rate * grad_b2.astype(np.float32)
        self.w1 -= self.learning_rate * grad_w1.astype(np.float32)
        self.b1 -= self.learning_rate * grad_b1.astype(np.float32)
        self.train_steps += 1
        if self.train_steps % self.target_sync_interval == 0:
            self._sync_target_network()

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
