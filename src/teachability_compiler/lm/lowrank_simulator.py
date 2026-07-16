"""Low-rank developmental transition simulator for LM curriculum actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from teachability_compiler.simulator import TransitionPrediction
from teachability_compiler.state import CurriculumAction, LearningState, TransitionObservation

_PROBE_EMA_ALPHA = 0.25


def _state_vector(state: LearningState) -> np.ndarray:
    as_vector = getattr(state, "as_vector", None)
    if callable(as_vector):
        vec = np.asarray(as_vector(), dtype=np.float64)
    else:
        parts = [
            np.asarray(state.probe_losses, dtype=np.float64).ravel(),
            np.asarray(state.update_sketch, dtype=np.float64).ravel(),
            np.asarray(state.optimizer_sketch, dtype=np.float64).ravel(),
            np.asarray(state.activation_sketch, dtype=np.float64).ravel(),
            np.asarray(state.exposure_histogram, dtype=np.float64).ravel(),
            np.asarray(state.history_embedding, dtype=np.float64).ravel(),
            np.asarray(state.architecture_embedding, dtype=np.float64).ravel(),
            np.asarray([float(state.step), float(state.tokens_seen)], dtype=np.float64),
        ]
        vec = np.concatenate(parts)
    if vec.ndim != 1:
        raise ValueError("LearningState.as_vector() must return a one-dimensional vector")
    if not np.all(np.isfinite(vec)):
        raise ValueError("LearningState vector contains non-finite values")
    return vec


def _bytes_for_state(state: LearningState) -> bytes:
    vec = np.ascontiguousarray(_state_vector(state), dtype=np.float64)
    return vec.tobytes()


def _get_field(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    raise AttributeError(f"Could not find any of fields {list(names)} on {type(obj)!r}")


def _single_action_name(action: str | CurriculumAction) -> str:
    if isinstance(action, str):
        return action
    cluster_ids = [str(x) for x in action.cluster_ids]
    if len(cluster_ids) != 1:
        raise ValueError("LowRankTransitionPredictor predicts single-action transitions only")
    return cluster_ids[0]


def _action_token_budget(action: str | CurriculumAction) -> int:
    if isinstance(action, str):
        return 0
    return int(action.token_budget)


def _orthogonal_rows(rank: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if rank <= dim:
        mat = rng.standard_normal((dim, rank))
        q, _ = np.linalg.qr(mat)
        return q[:, :rank].T.astype(np.float32)
    mat = rng.standard_normal((rank, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-6, None)


class LowRankTransitionPredictor:
    version = "lowrank-v1"

    def __init__(
        self,
        action_names: Sequence[str],
        rank: int = 6,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.action_names = [str(x) for x in action_names]
        if not self.action_names:
            raise ValueError("action_names must not be empty")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action_names must be unique")
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        self.device = torch.device(device)
        self.seed = int(seed)
        self.predict_calls = 0

        self.action_to_idx = {name: i for i, name in enumerate(self.action_names)}
        self.num_actions = len(self.action_names)
        self.probe_dim = self.num_actions

        self._mu_np = np.zeros((self.num_actions, self.probe_dim), dtype=np.float64)
        self._residual_std = np.ones(self.probe_dim, dtype=np.float64)
        self._z_mean: np.ndarray | None = None
        self._z_std: np.ndarray | None = None
        self._z_dim: int | None = None
        self._fitted = False
        self.final_losses: dict[str, float] = {}

        init_dirs = _orthogonal_rows(self.rank, self.probe_dim, self.seed)
        self._directions_param = nn.Parameter(torch.tensor(init_dirs, device=self.device))
        self.action_embedding: nn.Embedding | None = None
        self.alpha_net: nn.Sequential | None = None
        self._mu_t = torch.zeros(self.num_actions, self.probe_dim, device=self.device)

    @property
    def directions(self) -> np.ndarray:
        return self._directions_param.detach().cpu().numpy().astype(np.float64, copy=True)

    @property
    def mu(self) -> np.ndarray:
        return self._mu_np.copy()

    def _build_modules(self, z_dim: int) -> None:
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        init_dirs = _orthogonal_rows(self.rank, self.probe_dim, self.seed)
        self._directions_param = nn.Parameter(torch.tensor(init_dirs, device=self.device))
        self.action_embedding = nn.Embedding(self.num_actions, 8).to(self.device)
        input_dim = int(z_dim) + self.num_actions + 8
        self.alpha_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, self.rank),
        ).to(self.device)

        final = self.alpha_net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _module_parameters(self) -> list[nn.Parameter]:
        if self.action_embedding is None or self.alpha_net is None:
            raise RuntimeError("Predictor modules have not been built")
        return (
            [self._directions_param]
            + list(self.action_embedding.parameters())
            + list(self.alpha_net.parameters())
        )

    def _predict_delta_t(self, z_std: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        if self.action_embedding is None or self.alpha_net is None:
            raise RuntimeError("Predictor has not been fitted")
        one_hot = F.one_hot(action_idx, num_classes=self.num_actions).to(z_std.dtype)
        action_emb = self.action_embedding(action_idx)
        inputs = torch.cat([z_std, one_hot, action_emb], dim=-1)
        alpha = self.alpha_net(inputs)
        residual = alpha @ self._directions_param
        return self._mu_t[action_idx] + residual

    def _observation_arrays(
        self,
        observations: Sequence[TransitionObservation],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bytes]]:
        if not observations:
            raise ValueError("fit() requires at least one observation")

        z_rows: list[np.ndarray] = []
        action_indices: list[int] = []
        deltas: list[np.ndarray] = []
        state_bytes: list[bytes] = []

        for obs in observations:
            state_before = _get_field(obs, ("state_before", "before_state", "previous_state"))
            action = _get_field(obs, ("action", "curriculum_action"))
            probe_delta = np.asarray(
                _get_field(obs, ("probe_delta", "probe_loss_delta")),
                dtype=np.float64,
            )
            if probe_delta.shape != (self.probe_dim,):
                raise ValueError(
                    f"probe_delta has shape {probe_delta.shape}; expected {(self.probe_dim,)}"
                )
            name = _single_action_name(action)
            if name not in self.action_to_idx:
                raise KeyError(f"Unknown action {name!r}")

            z_rows.append(_state_vector(state_before))
            action_indices.append(self.action_to_idx[name])
            deltas.append(probe_delta)
            state_bytes.append(_bytes_for_state(state_before))

        z = np.stack(z_rows).astype(np.float64)
        actions = np.asarray(action_indices, dtype=np.int64)
        y = np.stack(deltas).astype(np.float64)
        return z, actions, y, state_bytes

    def _sample_ranking_pairs(
        self,
        groups: Mapping[bytes, list[int]],
        true_values: np.ndarray,
        epoch: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 1009 * (epoch + 1))
        left: list[int] = []
        right: list[int] = []
        signs: list[float] = []

        for indices in groups.values():
            if len(indices) < 2:
                continue

            max_pairs = 512
            if len(indices) * (len(indices) - 1) // 2 <= max_pairs:
                pairs = [
                    (indices[i], indices[j])
                    for i in range(len(indices))
                    for j in range(i + 1, len(indices))
                ]
            else:
                pairs = []
                attempts = 0
                while len(pairs) < max_pairs and attempts < max_pairs * 8:
                    i, j = rng.choice(indices, size=2, replace=False)
                    pairs.append((int(i), int(j)))
                    attempts += 1

            for i, j in pairs[:max_pairs]:
                diff = float(true_values[i] - true_values[j])
                if diff == 0.0:
                    continue
                left.append(i)
                right.append(j)
                signs.append(1.0 if diff > 0.0 else -1.0)

        return (
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
            np.asarray(signs, dtype=np.float32),
        )

    def _ranking_loss_t(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        pred_left = self._predict_delta_t(z_t[left], a_t[left])
        pred_right = self._predict_delta_t(z_t[right], a_t[right])
        value_left = -pred_left.mean(dim=1)
        value_right = -pred_right.mean(dim=1)
        return F.softplus(-signs * (value_left - value_right)).mean()

    def fit(
        self,
        observations: Sequence[TransitionObservation],
        *,
        epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        ranking_weight: float = 0.1,
    ) -> "LowRankTransitionPredictor":
        epochs = int(epochs)
        batch_size = int(batch_size)
        if epochs < 0:
            raise ValueError("epochs must be nonnegative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if ranking_weight < 0.0:
            raise ValueError("ranking_weight must be nonnegative")

        z, action_idx, y, state_bytes = self._observation_arrays(observations)
        if z.ndim != 2:
            raise ValueError("State matrix must be two-dimensional")
        self._z_dim = int(z.shape[1])
        self._z_mean = z.mean(axis=0)
        self._z_std = z.std(axis=0)
        self._z_std = np.maximum(self._z_std, 1e-6)

        counts = np.bincount(action_idx, minlength=self.num_actions)
        missing = [self.action_names[i] for i, count in enumerate(counts) if count == 0]
        if missing:
            raise ValueError(f"Cannot compute per-action means; missing observations: {missing}")

        mu = np.zeros((self.num_actions, self.probe_dim), dtype=np.float64)
        for action_i in range(self.num_actions):
            mu[action_i] = y[action_idx == action_i].mean(axis=0)
        self._mu_np = mu
        residual = y - mu[action_idx]
        self._residual_std = np.maximum(residual.std(axis=0), 1e-6)

        self._build_modules(self._z_dim)
        self._mu_t = torch.tensor(mu, dtype=torch.float32, device=self.device)

        z_std_np = (z - self._z_mean) / self._z_std
        z_t = torch.tensor(z_std_np, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(action_idx, dtype=torch.long, device=self.device)
        residual_t = torch.tensor(residual, dtype=torch.float32, device=self.device)

        groups: dict[bytes, list[int]] = defaultdict(list)
        for i, key in enumerate(state_bytes):
            groups[key].append(i)

        optimizer = torch.optim.AdamW(
            self._module_parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        rng = np.random.default_rng(self.seed)

        with torch.no_grad():
            initial_pred = self._predict_delta_t(z_t, a_t) - self._mu_t[a_t]
            initial_mse = F.mse_loss(initial_pred, residual_t).item()

        n = int(z_t.shape[0])
        ranking_weight = float(ranking_weight)

        # Joint loss per step: separate ranking-only steps would end every
        # epoch by walking the model away from the MSE optimum (observed as
        # final MSE far above the near-zero init). Each batch therefore
        # combines its MSE term with a slice of that epoch's ranking pairs.
        true_values = -y.mean(axis=1)
        for epoch in range(epochs):
            permutation = rng.permutation(n)
            if ranking_weight > 0.0:
                left_np, right_np, signs_np = self._sample_ranking_pairs(
                    groups,
                    true_values,
                    epoch,
                )
                pair_perm = rng.permutation(left_np.size) if left_np.size else None
            else:
                left_np = right_np = signs_np = np.empty(0, dtype=np.int64)
                pair_perm = None

            n_batches = max(1, (n + batch_size - 1) // batch_size)
            pairs_per_batch = (
                max(1, (left_np.size + n_batches - 1) // n_batches) if left_np.size else 0
            )
            for batch_index, start in enumerate(range(0, n, batch_size)):
                batch_np = permutation[start : start + batch_size]
                batch = torch.tensor(batch_np, dtype=torch.long, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                pred_delta = self._predict_delta_t(z_t[batch], a_t[batch])
                pred_residual = pred_delta - self._mu_t[a_t[batch]]
                loss = F.mse_loss(pred_residual, residual_t[batch])

                if pair_perm is not None and pairs_per_batch:
                    lo = batch_index * pairs_per_batch
                    pair_idx = pair_perm[lo : lo + pairs_per_batch]
                    if pair_idx.size:
                        left = torch.tensor(
                            left_np[pair_idx], dtype=torch.long, device=self.device
                        )
                        right = torch.tensor(
                            right_np[pair_idx], dtype=torch.long, device=self.device
                        )
                        signs = torch.tensor(
                            signs_np[pair_idx], dtype=torch.float32, device=self.device
                        )
                        loss = loss + ranking_weight * self._ranking_loss_t(
                            z_t, a_t, left, right, signs
                        )

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            final_pred = self._predict_delta_t(z_t, a_t) - self._mu_t[a_t]
            final_mse = F.mse_loss(final_pred, residual_t).item()
            true_values = -y.mean(axis=1)
            left_np, right_np, signs_np = self._sample_ranking_pairs(groups, true_values, epochs)
            if left_np.size:
                left = torch.tensor(left_np, dtype=torch.long, device=self.device)
                right = torch.tensor(right_np, dtype=torch.long, device=self.device)
                signs = torch.tensor(signs_np, dtype=torch.float32, device=self.device)
                final_ranking = self._ranking_loss_t(z_t, a_t, left, right, signs).item()
            else:
                final_ranking = 0.0

        self.final_losses = {
            "initial_mse": float(initial_mse),
            "mse": float(final_mse),
            "ranking": float(final_ranking),
            "total": float(final_mse + ranking_weight * final_ranking),
        }
        self._fitted = True
        return self

    def _standardize_state(self, state: LearningState) -> np.ndarray:
        if self._z_mean is None or self._z_std is None or self._z_dim is None:
            raise RuntimeError("Predictor must be fitted before predict()")
        z = _state_vector(state)
        if z.shape != (self._z_dim,):
            raise ValueError(
                f"State vector dimension mismatch: got {z.shape}, expected {self._z_dim}"
            )
        return (z - self._z_mean) / self._z_std

    def _action_index(self, action: str | CurriculumAction) -> tuple[str, int]:
        name = _single_action_name(action)
        if name not in self.action_to_idx:
            raise KeyError(f"Unknown action {name!r}")
        return name, self.action_to_idx[name]

    def predict(
        self,
        state: LearningState,
        action: str | CurriculumAction,
    ) -> TransitionPrediction:
        if not self._fitted:
            raise RuntimeError("Predictor must be fitted before predict()")

        _, action_idx = self._action_index(action)
        z_std = self._standardize_state(state)
        with torch.no_grad():
            z_t = torch.tensor(z_std[None, :], dtype=torch.float32, device=self.device)
            a_t = torch.tensor([action_idx], dtype=torch.long, device=self.device)
            delta = self._predict_delta_t(z_t, a_t)[0].detach().cpu().numpy().astype(np.float64)

        old_probe = np.asarray(state.probe_losses, dtype=np.float64)
        if old_probe.shape != (self.probe_dim,):
            raise ValueError(
                f"state.probe_losses shape {old_probe.shape}; expected {self.probe_dim}"
            )
        new_probe = np.clip(old_probe + delta, 0.0, 20.0)

        update_sketch = np.zeros_like(np.asarray(state.update_sketch, dtype=np.float64))
        if update_sketch.size < 3:
            update_sketch = np.zeros(4, dtype=np.float64)
        update_sketch = update_sketch.copy()
        update_sketch[2] = action_idx / max(1, self.num_actions - 1)

        activation_sketch = np.asarray(
            [
                float(np.mean(new_probe)),
                float(np.max(new_probe)),
                float(np.min(new_probe)),
                float(np.std(new_probe)),
            ],
            dtype=np.float64,
        )

        exposure = np.asarray(state.exposure_histogram, dtype=np.float64).copy()
        if exposure.shape != (self.num_actions,):
            raise ValueError(
                f"state.exposure_histogram shape {exposure.shape}; "
                f"expected {(self.num_actions,)}"
            )
        exposure[action_idx] += 1.0

        token_budget = _action_token_budget(action)
        previous_history = np.asarray(state.history_embedding, dtype=np.float64)
        history = np.zeros_like(previous_history)
        if history.size < 4:
            history = np.zeros(4, dtype=np.float64)
        history = history.copy()
        history[0] = float(np.linalg.norm(_PROBE_EMA_ALPHA * delta))
        history[1] = float(previous_history[1]) if previous_history.size > 1 else 0.0
        history[2] = float(previous_history[2]) if previous_history.size > 2 else 0.0
        history[3] = float(int(state.tokens_seen) + token_budget) / 1e9

        next_state = LearningState(
            probe_losses=new_probe.astype(np.float64),
            update_sketch=update_sketch.astype(np.float64),
            optimizer_sketch=np.asarray(state.optimizer_sketch, dtype=np.float64).copy(),
            activation_sketch=activation_sketch,
            exposure_histogram=exposure,
            history_embedding=history.astype(np.float64),
            architecture_embedding=np.asarray(
                state.architecture_embedding,
                dtype=np.float64,
            ).copy(),
            step=int(state.step) + 1,
            tokens_seen=int(state.tokens_seen) + token_budget,
        )

        self.predict_calls += 1
        return TransitionPrediction(
            next_state_mean=next_state,
            probe_delta_mean=delta.astype(np.float64),
            probe_delta_std=self._residual_std.copy(),
            forgetting_risk=float(np.clip(delta, 0.0, None).sum()),
            novelty=0.0,
            expected_compute=float(token_budget),
        )
