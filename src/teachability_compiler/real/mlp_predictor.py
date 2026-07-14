"""Residual-MLP transition predictor for real transition observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from teachability_compiler.simulator import TransitionPrediction
from teachability_compiler.state import CurriculumAction, LearningState, TransitionObservation


class _ResidualNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        embed_dim: int,
        hidden: tuple[int, ...],
        probe_dim: int,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(n_actions, embed_dim)

        layers: list[nn.Module] = []
        in_dim = state_dim + embed_dim + 1
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.GELU())
            in_dim = width
        layers.append(nn.Linear(in_dim, probe_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        state_features: torch.Tensor,
        action_indices: torch.Tensor,
        stage_features: torch.Tensor,
    ) -> torch.Tensor:
        action_features = self.action_embedding(action_indices)
        features = torch.cat([state_features, action_features, stage_features], dim=1)
        return self.mlp(features)


class ResidualMLPTransitionPredictor:
    """Transition simulator using per-action means plus a learned residual MLP."""

    version = "residual-mlp-v1"

    def __init__(
        self,
        action_names: Sequence[str],
        hidden: tuple[int, ...] = (128, 128),
        embed_dim: int = 16,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.action_names = tuple(action_names)
        if not self.action_names:
            raise ValueError("action_names must be non-empty")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action_names must be unique")

        self.hidden = tuple(int(width) for width in hidden)
        self.embed_dim = int(embed_dim)
        self.device = torch.device(device)
        self.seed = int(seed)
        self._action_index = {name: index for index, name in enumerate(self.action_names)}

        self._net: _ResidualNet | None = None
        self._mu: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._probe_std: np.ndarray | None = None
        self._state_dim: int | None = None
        self._probe_dim: int | None = None
        self._predict_calls = 0
        self.train_losses: list[float] = []

    @property
    def predict_calls(self) -> int:
        """Number of predict calls made by this predictor."""
        return self._predict_calls

    def fit(
        self,
        observations: Sequence[TransitionObservation],
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        ranking_weight: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> "ResidualMLPTransitionPredictor":
        """Fit the residual MLP on transition observations."""
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("fit requires at least one observation")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if ranking_weight < 0.0:
            raise ValueError("ranking_weight must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

        torch.manual_seed(self.seed)

        probe_dim = _one_dimensional(obs_list[0].probe_delta, "probe_delta").shape[0]
        state_dim = _state_vector(obs_list[0].state_before).shape[0]
        n_actions = len(self.action_names)

        deltas: list[np.ndarray] = []
        state_vectors: list[np.ndarray] = []
        action_indices: list[int] = []
        sums = np.zeros((n_actions, probe_dim), dtype=np.float64)
        counts = np.zeros(n_actions, dtype=np.int64)

        for observation in obs_list:
            state_vector = _state_vector(observation.state_before)
            if state_vector.shape[0] != state_dim:
                raise ValueError("state vector dimension mismatch in observations")

            delta = _one_dimensional(observation.probe_delta, "probe_delta")
            if delta.shape[0] != probe_dim:
                raise ValueError("probe_delta dimension mismatch in observations")

            action_name = self._single_action_name(observation.action)
            if action_name not in self._action_index:
                raise ValueError(f"unknown action {action_name!r}")
            action_index = self._action_index[action_name]

            state_vectors.append(state_vector)
            deltas.append(delta)
            action_indices.append(action_index)
            sums[action_index] += delta
            counts[action_index] += 1

        deltas_array = np.stack(deltas, axis=0)
        global_mean = np.mean(deltas_array, axis=0)
        mu = np.zeros((n_actions, probe_dim), dtype=np.float64)
        for index in range(n_actions):
            mu[index] = sums[index] / counts[index] if counts[index] else global_mean
        self._mu = mu

        state_matrix = np.stack(state_vectors, axis=0)
        feature_mean = np.mean(state_matrix, axis=0)
        feature_std = np.std(state_matrix, axis=0)
        feature_std = np.maximum(feature_std, 1e-6)
        standardized_states = (state_matrix - feature_mean) / feature_std
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        self._state_dim = state_dim
        self._probe_dim = probe_dim

        action_array = np.asarray(action_indices, dtype=np.int64)
        residual_targets = deltas_array - mu[action_array]
        stage_features = np.asarray(
            [[np.log1p(float(observation.state_before.step)) / 10.0] for observation in obs_list],
            dtype=np.float64,
        )
        true_delta_values = -np.mean(deltas_array, axis=1)

        device = self.device
        state_t = torch.tensor(standardized_states, dtype=torch.float32, device=device)
        action_t = torch.tensor(action_array, dtype=torch.long, device=device)
        stage_t = torch.tensor(stage_features, dtype=torch.float32, device=device)
        target_t = torch.tensor(residual_targets, dtype=torch.float32, device=device)
        value_t = torch.tensor(true_delta_values, dtype=torch.float32, device=device)
        mu_t = torch.tensor(mu, dtype=torch.float32, device=device)

        net = _ResidualNet(
            state_dim=state_dim,
            n_actions=n_actions,
            embed_dim=self.embed_dim,
            hidden=self.hidden,
            probe_dim=probe_dim,
        ).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        groups: dict[bytes, list[int]] = {}
        for index, observation in enumerate(obs_list):
            key = _state_key(observation.state_before)
            groups.setdefault(key, []).append(index)
        group_arrays = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        self.train_losses = []
        net.train()
        for _ in range(epochs):
            order = torch.randperm(len(group_arrays), generator=generator).tolist()
            epoch_squared_error = 0.0
            epoch_elements = 0
            batch_group_ids: list[int] = []
            batch_rows = 0

            for group_id in order:
                batch_group_ids.append(int(group_id))
                batch_rows += int(group_arrays[group_id].shape[0])
                if batch_rows >= batch_size:
                    loss_sum, element_count = self._train_group_batch(
                        batch_group_ids=batch_group_ids,
                        group_arrays=group_arrays,
                        net=net,
                        optimizer=optimizer,
                        state_t=state_t,
                        action_t=action_t,
                        stage_t=stage_t,
                        target_t=target_t,
                        value_t=value_t,
                        mu_t=mu_t,
                        ranking_weight=ranking_weight,
                    )
                    epoch_squared_error += loss_sum
                    epoch_elements += element_count
                    batch_group_ids = []
                    batch_rows = 0

            if batch_group_ids:
                loss_sum, element_count = self._train_group_batch(
                    batch_group_ids=batch_group_ids,
                    group_arrays=group_arrays,
                    net=net,
                    optimizer=optimizer,
                    state_t=state_t,
                    action_t=action_t,
                    stage_t=stage_t,
                    target_t=target_t,
                    value_t=value_t,
                    mu_t=mu_t,
                    ranking_weight=ranking_weight,
                )
                epoch_squared_error += loss_sum
                epoch_elements += element_count

            self.train_losses.append(epoch_squared_error / max(1, epoch_elements))

        net.eval()
        self._net = net
        with torch.no_grad():
            residual_predictions = net(state_t, action_t, stage_t).cpu().numpy()
        residual_errors = residual_targets - residual_predictions
        self._probe_std = np.std(residual_errors, axis=0)
        return self

    def predict(self, state: LearningState, action: CurriculumAction) -> TransitionPrediction:
        """Predict a transition from state and action."""
        if self._net is None:
            raise RuntimeError("predict called before fit")
        if self._mu is None or self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("predict called before fit")
        if self._probe_std is None or self._state_dim is None or self._probe_dim is None:
            raise RuntimeError("predict called before fit")

        action_name = self._single_action_name(action)
        if action_name not in self._action_index:
            raise ValueError(f"unknown action {action_name!r}")
        action_index = self._action_index[action_name]

        state_vector = _state_vector(state)
        if state_vector.shape[0] != self._state_dim:
            raise ValueError("state vector dimension mismatch")

        old_probes = _one_dimensional(state.probe_losses, "probe_losses")
        if old_probes.shape[0] != self._probe_dim:
            raise ValueError("probe loss dimension mismatch")

        standardized_state = (state_vector - self._feature_mean) / self._feature_std
        stage_feature = np.asarray([[np.log1p(float(state.step)) / 10.0]], dtype=np.float64)

        state_t = torch.tensor(
            standardized_state.reshape(1, -1),
            dtype=torch.float32,
            device=self.device,
        )
        action_t = torch.tensor([action_index], dtype=torch.long, device=self.device)
        stage_t = torch.tensor(stage_feature, dtype=torch.float32, device=self.device)

        self._net.eval()
        with torch.no_grad():
            residual = self._net(state_t, action_t, stage_t).cpu().numpy().reshape(-1)

        probe_delta = self._mu[action_index] + residual
        next_probes = np.clip(old_probes + probe_delta, 0.0, 10.0)
        clipped_delta = next_probes - old_probes
        next_state = _advance_state(state, action, clipped_delta, self.action_names)
        self._predict_calls += 1

        positive_probe_delta = clipped_delta[clipped_delta > 0.0]
        forgetting_risk = float(np.clip(np.sum(positive_probe_delta), 0.0, 1.0))
        expected_compute = float(action.optimizer_steps * action.token_budget)

        return TransitionPrediction(
            next_state_mean=next_state,
            probe_delta_mean=probe_delta,
            probe_delta_std=self._probe_std.copy(),
            forgetting_risk=forgetting_risk,
            novelty=0.0,
            expected_compute=expected_compute,
        )

    def _train_group_batch(
        self,
        *,
        batch_group_ids: list[int],
        group_arrays: list[np.ndarray],
        net: _ResidualNet,
        optimizer: torch.optim.Optimizer,
        state_t: torch.Tensor,
        action_t: torch.Tensor,
        stage_t: torch.Tensor,
        target_t: torch.Tensor,
        value_t: torch.Tensor,
        mu_t: torch.Tensor,
        ranking_weight: float,
    ) -> tuple[float, int]:
        indices = np.concatenate([group_arrays[group_id] for group_id in batch_group_ids])
        index_t = torch.tensor(indices, dtype=torch.long, device=self.device)

        optimizer.zero_grad()
        residual = net(state_t[index_t], action_t[index_t], stage_t[index_t])
        mse = torch.mean((residual - target_t[index_t]) ** 2)
        loss = mse

        if ranking_weight > 0.0:
            ranking_terms: list[torch.Tensor] = []
            for group_id in batch_group_ids:
                group = group_arrays[group_id]
                if group.shape[0] < 2:
                    continue

                group_t = torch.tensor(group, dtype=torch.long, device=self.device)
                group_residual = net(state_t[group_t], action_t[group_t], stage_t[group_t])
                predicted_delta = mu_t[action_t[group_t]] + group_residual
                predicted_values = -torch.mean(predicted_delta, dim=1)
                true_values = value_t[group_t]

                group_size = int(group.shape[0])
                for i in range(group_size):
                    for j in range(i + 1, group_size):
                        true_diff = float(true_values[i] - true_values[j])
                        if true_diff == 0.0:
                            continue
                        sign = 1.0 if true_diff > 0.0 else -1.0
                        margin = (predicted_values[i] - predicted_values[j]) * sign
                        ranking_terms.append(nn.functional.softplus(-margin))

            if ranking_terms:
                loss = loss + ranking_weight * torch.stack(ranking_terms).mean()

        loss.backward()
        optimizer.step()

        element_count = int(indices.shape[0]) * int(target_t.shape[1])
        return float(mse.item()) * element_count, element_count

    @staticmethod
    def _single_action_name(action: CurriculumAction) -> str:
        if len(action.cluster_ids) != 1:
            raise ValueError("ResidualMLPTransitionPredictor expects single-cluster actions")
        return action.cluster_ids[0]


def _advance_state(
    state: LearningState,
    action: CurriculumAction,
    clipped_delta: np.ndarray,
    action_names: tuple[str, ...],
) -> LearningState:
    old_probes = _one_dimensional(state.probe_losses, "probe_losses")
    delta = _one_dimensional(clipped_delta, "clipped_delta")
    if old_probes.shape != delta.shape:
        raise ValueError("probe delta dimension mismatch")

    new_probes = old_probes + delta
    skill_delta = -delta
    positive_skill = float(np.sum(skill_delta[skill_delta > 0.0]))
    negative_skill = float(np.sum(skill_delta[skill_delta < 0.0]))

    if len(action.cluster_ids) != 1:
        raise ValueError("_advance_state expects a single-cluster action")
    action_name = action.cluster_ids[0]
    try:
        action_index = action_names.index(action_name)
    except ValueError as exc:
        raise ValueError(f"unknown action {action_name!r}") from exc

    normalized_action_index = (
        float(action_index) / float(len(action_names) - 1) if len(action_names) > 1 else 0.0
    )
    update_sketch = _fit_to_template(
        [positive_skill, negative_skill, normalized_action_index, action.learning_rate_scale],
        state.update_sketch,
    )

    exposure_histogram = _one_dimensional(state.exposure_histogram, "exposure_histogram").copy()
    if action_index >= exposure_histogram.shape[0]:
        raise ValueError("exposure_histogram is too short for action index")
    exposure_histogram[action_index] += 1.0

    return LearningState(
        probe_losses=new_probes.astype(np.float64, copy=True),
        update_sketch=update_sketch,
        optimizer_sketch=_probe_tiled_sketch(new_probes, state.optimizer_sketch),
        activation_sketch=_probe_tiled_sketch(new_probes, state.activation_sketch),
        exposure_histogram=exposure_histogram,
        history_embedding=_probe_tiled_sketch(new_probes, state.history_embedding),
        architecture_embedding=np.asarray(
            state.architecture_embedding,
            dtype=np.float64,
        ).copy(),
        step=int(state.step) + 1,
        tokens_seen=int(state.tokens_seen) + int(action.token_budget),
    )


def _probe_tiled_sketch(probe_losses: np.ndarray, template: np.ndarray) -> np.ndarray:
    template_array = np.asarray(template, dtype=np.float64)
    if template_array.size == 0:
        return np.zeros(template_array.shape, dtype=np.float64)

    flat_probe = np.asarray(probe_losses, dtype=np.float64).reshape(-1)
    if flat_probe.size == 0:
        raise ValueError("probe_losses must be non-empty")

    repeats = int(np.ceil(template_array.size / flat_probe.size))
    tiled = np.tile(flat_probe, repeats)[: template_array.size]
    return tiled.reshape(template_array.shape).astype(np.float64, copy=True)


def _fit_to_template(values: Sequence[float], template: np.ndarray) -> np.ndarray:
    template_array = np.asarray(template, dtype=np.float64)
    flat_output = np.zeros(template_array.size, dtype=np.float64)
    flat_input = np.asarray(values, dtype=np.float64).reshape(-1)
    limit = min(flat_output.size, flat_input.size)
    flat_output[:limit] = flat_input[:limit]
    return flat_output.reshape(template_array.shape)


def _one_dimensional(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _state_vector(state: LearningState) -> np.ndarray:
    vector = np.asarray(state.as_vector(), dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("state.as_vector() must be one-dimensional")
    return vector


def _state_key(state: LearningState) -> bytes:
    vector = np.ascontiguousarray(_state_vector(state), dtype=np.float64)
    return vector.tobytes()
