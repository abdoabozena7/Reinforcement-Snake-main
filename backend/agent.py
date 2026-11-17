from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
import zipfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from backend.config import MODEL_VERSION
from backend.environment import ACTION_LABELS, ACTION_SIZE, OBSERVATION_LABELS, OBSERVATION_SIZE


PARAMETERS = ("w1", "b1", "w2", "b2", "w3", "b3")
CHECKPOINT_SCHEMA = 2
MAX_CHECKPOINT_BYTES = 192 * 1024 * 1024


def _json_text(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


@dataclass
class Experience:
    state: list[float] | np.ndarray
    action: int
    reward: float
    next_state: list[float] | np.ndarray
    done: bool
    action_mask: list[bool] | np.ndarray | None = None
    next_action_mask: list[bool] | np.ndarray | None = None


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_masks: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)


class ReplayBuffer:
    """Fixed-capacity array ring; sampling never touches the policy's RNG."""

    def __init__(
        self,
        max_size: int = 30_000,
        state_size: int = OBSERVATION_SIZE,
        action_size: int = ACTION_SIZE,
        seed: int = 0,
    ) -> None:
        if not 1 <= int(max_size) <= 100_000:
            raise ValueError("Replay capacity must be between 1 and 100000.")
        self.max_size = int(max_size)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((self.max_size, self.state_size), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(self.max_size, dtype=np.int64)
        self.rewards = np.empty(self.max_size, dtype=np.float32)
        self.dones = np.empty(self.max_size, dtype=np.bool_)
        self.next_masks = np.empty((self.max_size, self.action_size), dtype=np.bool_)
        self.position = 0
        self.count = 0

    def add(self, experience: Experience) -> None:
        if not 0 <= experience.action < self.action_size:
            raise ValueError("Experience action is out of range.")
        index = self.position
        self.states[index] = experience.state
        self.next_states[index] = experience.next_state
        self.actions[index] = experience.action
        self.rewards[index] = experience.reward
        self.dones[index] = experience.done
        self.next_masks[index] = (
            True if experience.next_action_mask is None else experience.next_action_mask
        )
        self.position = (index + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.count == 0:
            raise ValueError("Cannot sample an empty replay buffer.")
        indices = self.rng.integers(0, self.count, size=int(batch_size))
        return ReplayBatch(
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.next_masks[indices],
        )

    def size(self) -> int:
        return self.count

    def clear(self) -> None:
        self.position = self.count = 0

    def checkpoint(self) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        metadata: dict[str, object] = {
            "capacity": self.max_size,
            "size": self.count,
            "position": self.position,
            "stateSize": self.state_size,
            "actionSize": self.action_size,
            "rngState": copy.deepcopy(self.rng.bit_generator.state),
        }
        arrays = {
            f"replay_{name}": getattr(self, name)[: self.count].copy()
            for name in ("states", "actions", "rewards", "next_states", "dones", "next_masks")
        }
        return metadata, arrays

    @classmethod
    def from_checkpoint(
        cls, metadata: dict[str, object], arrays: dict[str, np.ndarray]
    ) -> ReplayBuffer:
        capacity = _integer(metadata.get("capacity"), "replay capacity", 1, 100_000)
        count = _integer(metadata.get("size"), "replay size", 0, capacity)
        position = _integer(metadata.get("position"), "replay position", 0, capacity - 1)
        state_size = _integer(metadata.get("stateSize"), "replay input size", 1, 4096)
        action_size = _integer(metadata.get("actionSize"), "replay action size", 1, 128)
        if capacity * (state_size * 8 + action_size + 13) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Replay allocation exceeds the checkpoint resource limit.")
        if count < capacity and position != count:
            raise ValueError("Invalid replay ring position.")
        specifications = {
            "states": ((count, state_size), np.dtype("float32")),
            "next_states": ((count, state_size), np.dtype("float32")),
            "actions": ((count,), np.dtype("int64")),
            "rewards": ((count,), np.dtype("float32")),
            "dones": ((count,), np.dtype("bool")),
            "next_masks": ((count, action_size), np.dtype("bool")),
        }
        for name, (shape, dtype) in specifications.items():
            value = arrays.get(f"replay_{name}")
            if value is None or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"Incompatible replay array: {name}.")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite replay array: {name}.")
        if count and (
            (arrays["replay_actions"] < 0).any()
            or (arrays["replay_actions"] >= action_size).any()
        ):
            raise ValueError("Invalid replay action.")
        rng = np.random.default_rng()
        rng.bit_generator.state = metadata["rngState"]
        replay = cls(capacity, state_size, action_size)
        replay.rng = rng
        replay.position, replay.count = position, count
        for name in specifications:
            getattr(replay, name)[:count] = arrays[f"replay_{name}"]
        return replay


def _integer(value: object, name: str, minimum: int, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}.")
    return value


class DQNAgent:
    input_labels = list(OBSERVATION_LABELS)
    output_labels = list(ACTION_LABELS)

    def __init__(
        self,
        model_path: Path | str,
        state_size: int = OBSERVATION_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = 96,
        learning_rate: float = 0.0004,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.04,
        epsilon_decay: float = 0.99992,
        target_sync_interval: int = 250,
        seed: int = 0,
        gradient_clip: float = 10.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.state_size, self.action_size = int(state_size), int(action_size)
        self.hidden_size = int(hidden_size)
        self.learning_rate, self.gamma = float(learning_rate), float(gamma)
        self.epsilon, self.epsilon_min = float(epsilon), float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.target_sync_interval = int(target_sync_interval)
        self.gradient_clip = float(gradient_clip)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.input_labels = (
            list(OBSERVATION_LABELS)
            if self.state_size == OBSERVATION_SIZE
            else [f"Observation {index}" for index in range(self.state_size)]
        )
        self.output_labels = (
            list(ACTION_LABELS)
            if self.action_size == ACTION_SIZE
            else [f"Action {index}" for index in range(self.action_size)]
        )
        self.train_steps = 0
        self.environment_steps = 0
        self.last_loss: float | None = None
        self.last_load_error = ""
        self.last_decision: dict[str, object] | None = None
        self._init_weights()

    @property
    def dimensions(self) -> list[int]:
        return [self.state_size, self.hidden_size, self.hidden_size, self.action_size]

    def _init_weights(self) -> None:
        self.w1 = self.rng.normal(
            0.0, math.sqrt(2.0 / self.state_size), (self.state_size, self.hidden_size)
        ).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.w2 = self.rng.normal(
            0.0, math.sqrt(2.0 / self.hidden_size), (self.hidden_size, self.hidden_size)
        ).astype(np.float32)
        self.b2 = np.zeros_like(self.b1)
        self.w3 = self.rng.normal(
            0.0, 0.01, (self.hidden_size, self.action_size)
        ).astype(np.float32)
        self.b3 = np.zeros((1, self.action_size), dtype=np.float32)
        self.adam_m = {name: np.zeros_like(getattr(self, name)) for name in PARAMETERS}
        self.adam_v = {name: np.zeros_like(getattr(self, name)) for name in PARAMETERS}
        self._sync_target_network()

    def _sync_target_network(self) -> None:
        for name in PARAMETERS:
            setattr(self, f"target_{name}", getattr(self, name).copy())

    def _forward(
        self, states: np.ndarray, target: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prefix = "target_" if target else ""
        a1 = np.maximum(
            states @ getattr(self, f"{prefix}w1") + getattr(self, f"{prefix}b1"), 0.0
        )
        a2 = np.maximum(
            a1 @ getattr(self, f"{prefix}w2") + getattr(self, f"{prefix}b2"), 0.0
        )
        q_values = a2 @ getattr(self, f"{prefix}w3") + getattr(self, f"{prefix}b3")
        return a1, a2, q_values

    def predict(self, states: np.ndarray | list[list[float]]) -> np.ndarray:
        return self._forward(np.atleast_2d(np.asarray(states, dtype=np.float32)))[2]

    def predict_target(self, states: np.ndarray | list[list[float]]) -> np.ndarray:
        return self._forward(np.atleast_2d(np.asarray(states, dtype=np.float32)), True)[2]

    def act(
        self,
        state: list[float] | np.ndarray,
        action_mask: list[bool] | np.ndarray | None = None,
        *,
        explore: bool = True,
        capture_decision: bool = True,
    ) -> int:
        mask = (
            np.ones(self.action_size, dtype=np.bool_)
            if action_mask is None
            else np.asarray(action_mask, dtype=np.bool_)
        )
        if mask.shape != (self.action_size,) or not mask.any():
            raise ValueError("A decision needs at least one legal action.")
        state_array = np.asarray(state, dtype=np.float32).reshape(1, self.state_size)
        exploring = bool(explore and self.epsilon > 0.0 and self.rng.random() < self.epsilon)
        forward = self._forward(state_array) if capture_decision or not exploring else None
        if exploring:
            action = int(self.rng.choice(np.flatnonzero(mask)))
        else:
            assert forward is not None
            action = int(np.argmax(np.where(mask, forward[2][0], -np.inf)))
        if capture_decision:
            assert forward is not None
            self.last_decision = self._network_payload(
                state_array[0], forward, mask, action, exploring
            )
        else:
            self.last_decision = None
        return action

    def record_training_step(self) -> None:
        self.environment_steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_epsilon(self) -> None:
        self.record_training_step()

    def train_batch(self, batch: ReplayBatch | list[Experience]) -> dict[str, float]:
        if not isinstance(batch, ReplayBatch):
            batch = ReplayBatch(
                np.asarray([item.state for item in batch], dtype=np.float32),
                np.asarray([item.action for item in batch], dtype=np.int64),
                np.asarray([item.reward for item in batch], dtype=np.float32),
                np.asarray([item.next_state for item in batch], dtype=np.float32),
                np.asarray([item.done for item in batch], dtype=np.bool_),
                np.asarray(
                    [
                        np.ones(self.action_size, dtype=np.bool_)
                        if item.next_action_mask is None
                        else item.next_action_mask
                        for item in batch
                    ],
                    dtype=np.bool_,
                ),
            )
        count = len(batch)
        if count == 0:
            raise ValueError("A training batch cannot be empty.")
        a1, a2, predictions = self._forward(batch.states)
        online_next = self.predict(batch.next_states)
        target_next = self.predict_target(batch.next_states)
        best_actions = np.argmax(np.where(batch.next_masks, online_next, -np.inf), axis=1)
        can_bootstrap = ~batch.dones & batch.next_masks.any(axis=1)
        future = target_next[np.arange(count), best_actions]
        targets = batch.rewards + self.gamma * np.where(can_bootstrap, future, 0.0)
        chosen = predictions[np.arange(count), batch.actions]
        errors = chosen - targets
        absolute = np.abs(errors)
        quadratic = np.minimum(absolute, 1.0)
        losses = 0.5 * quadratic**2 + absolute - quadratic
        grad_output = np.zeros_like(predictions)
        grad_output[np.arange(count), batch.actions] = np.clip(errors, -1.0, 1.0) / count
        grad_z2 = (grad_output @ self.w3.T) * (a2 > 0)
        grad_z1 = (grad_z2 @ self.w2.T) * (a1 > 0)
        gradients = {
            "w3": a2.T @ grad_output,
            "b3": grad_output.sum(axis=0, keepdims=True),
            "w2": a1.T @ grad_z2,
            "b2": grad_z2.sum(axis=0, keepdims=True),
            "w1": batch.states.T @ grad_z1,
            "b1": grad_z1.sum(axis=0, keepdims=True),
        }
        norm = math.sqrt(
            sum(float(np.sum(gradient.astype(np.float64) ** 2)) for gradient in gradients.values())
        )
        if not math.isfinite(norm) or not np.isfinite(losses).all():
            raise ValueError("Non-finite training batch; parameters were not updated.")
        scale = min(1.0, self.gradient_clip / max(norm, 1e-12))
        step = self.train_steps + 1
        correction1, correction2 = 1.0 - 0.9**step, 1.0 - 0.999**step
        for name, gradient in gradients.items():
            gradient *= scale
            moment, variance = self.adam_m[name], self.adam_v[name]
            moment *= 0.9
            moment += 0.1 * gradient
            variance *= 0.999
            variance += 0.001 * gradient**2
            update = (moment / correction1) / (np.sqrt(variance / correction2) + 1e-8)
            getattr(self, name)[:] -= self.learning_rate * update
        self.train_steps = step
        self.last_loss = float(losses.mean())
        if step % self.target_sync_interval == 0:
            self._sync_target_network()
        return {
            "loss": self.last_loss,
            "gradientNorm": norm,
            "clippedGradientNorm": min(norm, self.gradient_clip),
            "meanQ": float(chosen.mean()),
            "meanTarget": float(targets.mean()),
        }

    def _network_payload(
        self,
        state: np.ndarray,
        forward: tuple[np.ndarray, np.ndarray, np.ndarray],
        mask: np.ndarray,
        action: int | None,
        exploring: bool,
    ) -> dict[str, object]:
        indices = [
            np.linspace(0, size - 1, min(size, 20), dtype=np.int64)
            for size in self.dimensions
        ]
        layers = (state, forward[0][0], forward[1][0], forward[2][0])
        return {
            "dimensions": self.dimensions,
            "layers": [
                np.round(layer[index], 5).tolist() for layer, index in zip(layers, indices)
            ],
            "weights": [
                np.round(weight[np.ix_(indices[index], indices[index + 1])], 5).tolist()
                for index, weight in enumerate((self.w1, self.w2, self.w3))
            ],
            "sampleIndices": [index.tolist() for index in indices],
            "inputLabels": [self.input_labels[index] for index in indices[0]],
            "outputLabels": [self.output_labels[index] for index in indices[-1]],
            "qValues": np.round(forward[2][0], 6).tolist(),
            "actionMask": mask.tolist(),
            "lastAction": action,
            "isExploring": exploring,
            "label": "Last decision" if action is not None else "Awaiting first decision",
        }

    def network_snapshot(
        self, state: list[float] | np.ndarray, action_mask: list[bool] | None = None
    ) -> dict[str, object]:
        if self.last_decision is not None:
            return copy.deepcopy(self.last_decision)
        state_array = np.asarray(state, dtype=np.float32).reshape(1, self.state_size)
        mask = np.asarray(
            [True] * self.action_size if action_mask is None else action_mask, dtype=np.bool_
        )
        return self._network_payload(state_array[0], self._forward(state_array), mask, None, False)

    def parameter_hash(self) -> str:
        digest = hashlib.sha256(MODEL_VERSION.encode("utf-8"))
        for name in PARAMETERS:
            digest.update(getattr(self, name).tobytes(order="C"))
        return digest.hexdigest()

    def clone(self) -> DQNAgent:
        return copy.deepcopy(self)

    def _metadata(self, training_metadata: dict[str, object] | None) -> dict[str, object]:
        return {
            "schemaVersion": CHECKPOINT_SCHEMA,
            "modelVersion": MODEL_VERSION,
            "algorithm": "masked-double-dqn",
            "architecture": self.dimensions,
            "activation": "relu",
            "inputLabels": self.input_labels,
            "actionLabels": self.output_labels,
            "optimizer": "adam",
            "learningRate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilonMin": self.epsilon_min,
            "epsilonDecay": self.epsilon_decay,
            "targetSyncInterval": self.target_sync_interval,
            "gradientClip": self.gradient_clip,
            "gradientSteps": self.train_steps,
            "trainingSteps": self.environment_steps,
            "lastLoss": self.last_loss,
            "seed": self.seed,
            "rngState": copy.deepcopy(self.rng.bit_generator.state),
            "trainingMetadata": training_metadata or {},
        }

    def save_to(
        self,
        path: Path | str,
        training_metadata: dict[str, object] | None = None,
        extra_arrays: dict[str, np.ndarray] | None = None,
    ) -> None:
        path = Path(path)
        metadata = self._metadata(training_metadata)
        arrays = {name: getattr(self, name).copy() for name in PARAMETERS}
        arrays.update({f"target_{name}": getattr(self, f"target_{name}").copy() for name in PARAMETERS})
        arrays.update({f"adam_m_{name}": self.adam_m[name].copy() for name in PARAMETERS})
        arrays.update({f"adam_v_{name}": self.adam_v[name].copy() for name in PARAMETERS})
        if extra_arrays:
            if any(not name.startswith("replay_") for name in extra_arrays):
                raise ValueError("Unsupported checkpoint extension.")
            arrays.update(extra_arrays)
        self._validate_checkpoint(metadata, arrays)
        encoded = np.array(_json_text(metadata))
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as previous:
                    old_metadata = json.loads(str(previous["metadata"].item()))
                if old_metadata.get("modelVersion") != MODEL_VERSION:
                    raise ValueError("Incompatible checkpoint.")
            except Exception as error:
                raise ValueError(
                    "Refusing to overwrite an unversioned, incompatible or corrupt file; "
                    "choose a new checkpoint path. Legacy models require retraining."
                ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
        try:
            with partial.open("xb") as stream:
                np.savez_compressed(stream, metadata=encoded, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, path)
        finally:
            if partial.exists():
                partial.unlink()

    def save(self) -> None:
        self.save_to(self.model_path)

    def _validate_checkpoint(
        self, metadata: dict[str, object], arrays: dict[str, np.ndarray]
    ) -> None:
        _integer(metadata.get("schemaVersion"), "checkpoint schema version", CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA)
        architecture = metadata.get("architecture")
        if not isinstance(architecture, list) or any(
            isinstance(size, bool) or not isinstance(size, int) for size in architecture
        ):
            raise ValueError("Invalid checkpoint architecture sizes.")
        if (
            metadata.get("schemaVersion") != CHECKPOINT_SCHEMA
            or metadata.get("modelVersion") != MODEL_VERSION
            or metadata.get("algorithm") != "masked-double-dqn"
            or metadata.get("architecture") != self.dimensions
            or metadata.get("inputLabels") != self.input_labels
            or metadata.get("actionLabels") != self.output_labels
            or metadata.get("optimizer") != "adam"
            or metadata.get("activation") != "relu"
        ):
            raise ValueError(
                f"Incompatible model semantics/architecture for {MODEL_VERSION}. "
                "Older planar and legacy weights need explicit conversion or retraining; files are left untouched."
            )
        for name in PARAMETERS:
            expected_shape = getattr(self, name).shape
            for key in (name, f"target_{name}", f"adam_m_{name}", f"adam_v_{name}"):
                value = arrays.get(key)
                if value is None or value.dtype != np.float32 or value.shape != expected_shape:
                    raise ValueError(f"Incompatible checkpoint array: {key}.")
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite checkpoint array: {key}.")
                if key.startswith("adam_v_") and (value < 0).any():
                    raise ValueError("Negative Adam variance.")
        for key, value in arrays.items():
            if key.startswith("replay_") and (
                value.dtype.kind not in "fib" or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid replay checkpoint array: {key}.")
        for name, low, high in (
            ("learningRate", 1e-8, 0.1),
            ("gamma", 0.0, 1.0),
            ("epsilon", 0.0, 1.0),
            ("epsilonMin", 0.0, 1.0),
            ("epsilonDecay", 0.0, 1.0),
            ("gradientClip", 1e-8, 1e6),
        ):
            value = metadata.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"Invalid checkpoint {name}.")
        _integer(metadata.get("gradientSteps"), "gradient steps", 0)
        _integer(metadata.get("trainingSteps"), "training steps", 0)
        _integer(metadata.get("targetSyncInterval"), "target sync interval", 1, 1_000_000)
        _integer(metadata.get("seed"), "seed", 0, 2**64 - 1)
        loss = metadata.get("lastLoss")
        if loss is not None and (
            isinstance(loss, bool) or not isinstance(loss, (float, int)) or not math.isfinite(loss)
        ):
            raise ValueError("Invalid checkpoint loss.")
        if not isinstance(metadata.get("trainingMetadata"), dict):
            raise ValueError("Invalid training metadata.")
        rng = np.random.default_rng()
        rng.bit_generator.state = metadata["rngState"]
        _json_text(metadata)

    def read_checkpoint(self, path: Path | str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        path = Path(path)
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("Checkpoint is too large.")
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("Expanded checkpoint is too large.")
        with np.load(path, allow_pickle=False) as saved:
            raw_metadata = saved["metadata"]
            if raw_metadata.ndim != 0 or raw_metadata.dtype.kind != "U" or raw_metadata.nbytes > 8_000_000:
                raise ValueError("Invalid checkpoint metadata encoding.")
            metadata = json.loads(str(raw_metadata.item()), parse_constant=_reject_json_constant)
            arrays = {name: saved[name].copy() for name in saved.files if name != "metadata"}
        if not isinstance(metadata, dict):
            raise ValueError("Invalid checkpoint metadata.")
        self._validate_checkpoint(metadata, arrays)
        return metadata, arrays

    def restore_checkpoint(
        self, metadata: dict[str, object], arrays: dict[str, np.ndarray]
    ) -> None:
        self._validate_checkpoint(metadata, arrays)
        for name in PARAMETERS:
            setattr(self, name, arrays[name].copy())
            setattr(self, f"target_{name}", arrays[f"target_{name}"].copy())
        self.adam_m = {name: arrays[f"adam_m_{name}"].copy() for name in PARAMETERS}
        self.adam_v = {name: arrays[f"adam_v_{name}"].copy() for name in PARAMETERS}
        for attribute, key in (
            ("learning_rate", "learningRate"),
            ("gamma", "gamma"),
            ("epsilon", "epsilon"),
            ("epsilon_min", "epsilonMin"),
            ("epsilon_decay", "epsilonDecay"),
            ("gradient_clip", "gradientClip"),
        ):
            setattr(self, attribute, float(metadata[key]))
        self.target_sync_interval = int(metadata["targetSyncInterval"])
        self.train_steps = int(metadata["gradientSteps"])
        self.environment_steps = int(metadata["trainingSteps"])
        self.seed = int(metadata["seed"])
        self.last_loss = metadata["lastLoss"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(metadata["rngState"])
        self.last_decision = None
        self.last_load_error = ""

    def load_from(self, path: Path | str) -> bool:
        try:
            metadata, arrays = self.read_checkpoint(path)
            self.restore_checkpoint(metadata, arrays)
        except Exception as error:
            self.last_load_error = str(error) or type(error).__name__
            return False
        return True

    def load(self) -> bool:
        return self.load_from(self.model_path)

    def clear_memory(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.train_steps = self.environment_steps = 0
        self.epsilon = 1.0
        self.last_loss = None
        self.last_decision = None
        self._init_weights()
