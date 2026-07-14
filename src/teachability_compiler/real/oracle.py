"""Oracle wrapper that trains and measures the real decoder learner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from teachability_compiler.real.model import DecoderConfig, TinyDecoder
from teachability_compiler.real.tasks import all_cluster_names, probe_batch, sample_batch
from teachability_compiler.state import CurriculumAction, LearningState, TransitionObservation


class RealLearnerOracle:
    """Stateful learner oracle backed by a small decoder Transformer and AdamW.

    Library code does not set global torch seeds. Experiments should call ``torch.manual_seed``
    before constructing the oracle; exact bitwise determinism is not guaranteed on GPU, but
    explicit seeds fix model initialization order and all data-order randomness.
    """

    def __init__(
        self,
        config: DecoderConfig,
        cluster_names: Sequence[str],
        device: str,
        base_seed: int,
        steps_per_action: int = 8,
        batch_size: int = 64,
        seq_len: int = 64,
        lr: float = 3e-4,
    ) -> None:
        valid_names = set(all_cluster_names())
        unknown = set(cluster_names) - valid_names
        if unknown:
            raise ValueError(f"unknown clusters: {sorted(unknown)}")
        if not cluster_names:
            raise ValueError("cluster_names must not be empty")
        if steps_per_action <= 0:
            raise ValueError("steps_per_action must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_len > config.max_seq_len:
            raise ValueError("seq_len must be <= config.max_seq_len")
        if lr <= 0.0:
            raise ValueError("lr must be positive")

        self.config = config
        self.cluster_names = tuple(cluster_names)
        self.cluster_to_index = {name: index for index, name in enumerate(self.cluster_names)}
        self.device = torch.device(device)
        self.base_seed = int(base_seed)
        self.steps_per_action = int(steps_per_action)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.lr = float(lr)

        self.model = TinyDecoder(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.step = 0
        self.tokens_seen = 0
        self._exposure_histogram = np.zeros(len(self.cluster_names), dtype=np.float64)
        self._last_update_sketch = np.zeros(4, dtype=np.float64)
        self._last_action_vector = np.zeros(len(self.cluster_names), dtype=np.float64)

    def pretrain(self, n_steps: int, rng_seed: int) -> None:
        """Train on a uniform mixture over all clusters for ``n_steps`` optimizer steps."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        before = self._clone_trainable_parameters()
        action_vector = np.full(len(self.cluster_names), 1.0 / len(self.cluster_names))
        rng = np.random.default_rng(rng_seed)
        grad_norm_mean = self._train_on_distribution(
            action_vector=action_vector,
            rng=rng,
            n_steps=n_steps,
            lr_scale=1.0,
        )
        parameter_delta = float(np.linalg.norm(self._parameter_delta_sketch(before)))

        self._last_action_vector = action_vector.copy()
        self._last_update_sketch = np.array(
            [grad_norm_mean, parameter_delta, self._normalized_action_index(action_vector), 1.0],
            dtype=np.float64,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a CPU deep-copy snapshot of model, optimizer, and counters."""
        return {
            "model_state": _to_cpu_copy(self.model.state_dict()),
            "optimizer_state": _to_cpu_copy(self.optimizer.state_dict()),
            "step": int(self.step),
            "tokens_seen": int(self.tokens_seen),
            "exposure_histogram": self._exposure_histogram.copy(),
            "last_update_sketch": self._last_update_sketch.copy(),
            "last_action_vector": self._last_action_vector.copy(),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a snapshot created by :meth:`snapshot`."""
        self.model.load_state_dict(snapshot["model_state"])
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
        self._move_optimizer_state_to_device()

        self.step = int(snapshot["step"])
        self.tokens_seen = int(snapshot["tokens_seen"])
        self._exposure_histogram = np.asarray(
            snapshot["exposure_histogram"], dtype=np.float64
        ).copy()
        self._last_update_sketch = np.asarray(
            snapshot["last_update_sketch"], dtype=np.float64
        ).copy()
        self._last_action_vector = np.asarray(
            snapshot["last_action_vector"], dtype=np.float64
        ).copy()

        if self._exposure_histogram.shape != (len(self.cluster_names),):
            raise ValueError("snapshot exposure_histogram has wrong shape")
        if self._last_update_sketch.shape != (4,):
            raise ValueError("snapshot last_update_sketch has wrong shape")
        if self._last_action_vector.shape != (len(self.cluster_names),):
            raise ValueError("snapshot last_action_vector has wrong shape")

    def probe_losses(self) -> np.ndarray:
        """Return per-cluster held-out mean cross-entropy losses."""
        losses: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for name in self.cluster_names:
                inputs, targets = probe_batch(name, batch_size=64, seq_len=self.seq_len)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite probe loss for {name}")
                losses.append(float(loss.detach().cpu().item()))
        return np.asarray(losses, dtype=np.float64)

    def encode_state(self) -> LearningState:
        """Encode the current learner checkpoint as a ``LearningState``."""
        probe_losses = self.probe_losses()
        activation_sketch = np.array(
            [
                float(np.mean(probe_losses)),
                float(np.max(probe_losses)),
                float(np.min(probe_losses)),
                float(np.std(probe_losses)),
            ],
            dtype=np.float64,
        )
        optimizer_sketch = np.array(
            [float(self.step) / 1000.0, float(self.optimizer.param_groups[0]["lr"])],
            dtype=np.float64,
        )
        history_embedding = np.array(
            [
                float(np.linalg.norm(self._last_action_vector)),
                float(self.step) / 1000.0,
                float(np.mean(probe_losses)),
                float(np.std(probe_losses)),
            ],
            dtype=np.float64,
        )
        architecture_embedding = np.array(
            [float(self.config.n_layers) / 10.0, float(self.config.d_model) / 1000.0],
            dtype=np.float64,
        )
        return LearningState(
            probe_losses=probe_losses,
            update_sketch=self._last_update_sketch.copy(),
            optimizer_sketch=optimizer_sketch,
            activation_sketch=activation_sketch,
            exposure_histogram=self._exposure_histogram.copy(),
            history_embedding=history_embedding,
            architecture_embedding=architecture_embedding,
            step=int(self.step),
            tokens_seen=int(self.tokens_seen),
        )

    def apply_action(self, action: CurriculumAction, data_seed: int) -> TransitionObservation:
        """Apply a curriculum action and return the measured transition observation."""
        action_vector = self._action_vector(action)
        n_steps = int(action.optimizer_steps)
        if n_steps <= 0:
            raise ValueError("action.optimizer_steps must be positive")

        state_before = self.encode_state()
        before_params = self._clone_trainable_parameters()
        rng = np.random.default_rng(data_seed)
        grad_norm_mean = self._train_on_distribution(
            action_vector=action_vector,
            rng=rng,
            n_steps=n_steps,
            lr_scale=float(action.learning_rate_scale),
        )
        parameter_delta_sketch = self._parameter_delta_sketch(before_params)
        parameter_delta_norm = float(np.linalg.norm(parameter_delta_sketch))

        self._last_action_vector = action_vector.copy()
        self._last_update_sketch = np.array(
            [
                grad_norm_mean,
                parameter_delta_norm,
                self._normalized_action_index(action_vector),
                float(action.learning_rate_scale),
            ],
            dtype=np.float64,
        )

        state_after = self.encode_state()
        probe_delta = state_after.probe_losses - state_before.probe_losses
        activation_delta_sketch = state_after.activation_sketch - state_before.activation_sketch

        return TransitionObservation(
            state_before=state_before,
            action=action,
            state_after=state_after,
            parameter_delta_sketch=parameter_delta_sketch,
            probe_delta=probe_delta.astype(np.float64),
            activation_delta_sketch=activation_delta_sketch.astype(np.float64),
            compute_cost=float(n_steps * self.batch_size * self.seq_len),
            seed_metadata={"seed": int(data_seed), "step": int(state_before.step)},
            simulator_version=None,
        )

    def value(self, state: LearningState) -> float:
        """Planner value: negative mean held-out probe loss."""
        return -float(np.mean(state.probe_losses))

    def _set_optimizer_lr(self, lr_scale: float) -> None:
        if lr_scale <= 0.0:
            raise ValueError("learning_rate_scale must be positive")
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * lr_scale

    def _train_on_distribution(
        self,
        action_vector: np.ndarray,
        rng: np.random.Generator,
        n_steps: int,
        lr_scale: float,
    ) -> float:
        if action_vector.shape != (len(self.cluster_names),):
            raise ValueError("action_vector has wrong shape")
        if not np.isclose(float(np.sum(action_vector)), 1.0):
            raise ValueError("action weights must sum to 1")
        if np.any(action_vector < 0.0):
            raise ValueError("action weights must be non-negative")

        self._set_optimizer_lr(lr_scale)
        self.model.train()
        grad_norms: list[float] = []
        for _ in range(n_steps):
            inputs, targets = self._sample_mixture_batch(action_vector, rng)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            grad_norms.append(self._grad_norm())
            self.optimizer.step()

            self.step += 1
            self.tokens_seen += self.batch_size * self.seq_len
            self._exposure_histogram += action_vector

        return float(np.mean(grad_norms)) if grad_norms else 0.0

    def _sample_mixture_batch(
        self,
        action_vector: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = rng.multinomial(self.batch_size, action_vector)
        input_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for name, count in zip(self.cluster_names, counts, strict=True):
            if count == 0:
                continue
            inputs, targets = sample_batch(name, int(count), self.seq_len, rng)
            input_parts.append(inputs)
            target_parts.append(targets)

        if not input_parts:
            raise RuntimeError("mixture sampling produced no examples")

        inputs = torch.cat(input_parts, dim=0)
        targets = torch.cat(target_parts, dim=0)
        permutation = torch.as_tensor(rng.permutation(inputs.shape[0]), dtype=torch.long)
        inputs = inputs.index_select(0, permutation).to(self.device)
        targets = targets.index_select(0, permutation).to(self.device)
        return inputs, targets

    def _action_vector(self, action: CurriculumAction) -> np.ndarray:
        if len(action.cluster_ids) != len(action.mixture_weights):
            raise ValueError("cluster_ids and mixture_weights must have the same length")
        vector = np.zeros(len(self.cluster_names), dtype=np.float64)
        for cluster_id, weight in zip(action.cluster_ids, action.mixture_weights, strict=True):
            if cluster_id not in self.cluster_to_index:
                raise ValueError(f"unknown action cluster {cluster_id!r}")
            vector[self.cluster_to_index[cluster_id]] += float(weight)

        if vector.shape != (len(self.cluster_names),):
            raise ValueError("action vector shape mismatch")
        if np.any(vector < 0.0):
            raise ValueError("mixture weights must be non-negative")
        if not np.isclose(float(np.sum(vector)), 1.0):
            raise ValueError("mixture weights must sum to 1")
        return vector

    def _normalized_action_index(self, action_vector: np.ndarray) -> float:
        if len(self.cluster_names) == 1:
            return 0.0
        indices = np.arange(len(self.cluster_names), dtype=np.float64)
        indices /= float(len(self.cluster_names) - 1)
        return float(np.dot(action_vector, indices))

    def _grad_norm(self) -> float:
        total_sq = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().float().norm().cpu().item())
            total_sq += norm * norm
        return total_sq**0.5

    def _clone_trainable_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def _parameter_delta_sketch(self, before: Mapping[str, torch.Tensor]) -> np.ndarray:
        norms_sq = np.zeros(8, dtype=np.float64)
        seen: set[str] = set()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name not in before:
                raise ValueError(f"missing parameter {name!r} in snapshot")
            group = self._parameter_group(name)
            diff = parameter.detach().cpu().to(torch.float64) - before[name].to(torch.float64)
            norms_sq[group] += float(torch.sum(diff * diff).item())
            seen.add(name)

        missing = set(before) - seen
        if missing:
            raise ValueError(f"stale parameters in snapshot: {sorted(missing)}")
        return np.sqrt(norms_sq).astype(np.float64)

    def _parameter_group(self, name: str) -> int:
        if name.startswith(("token_embedding", "position_embedding")):
            return 0
        if name.startswith("blocks."):
            parts = name.split(".")
            layer_index = int(parts[1])
            return 1 + min(5, (layer_index * 6) // max(1, self.config.n_layers))
        return 7

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)


def _to_cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_copy(item) for item in value)
    return value
