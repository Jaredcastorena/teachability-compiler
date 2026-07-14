"""Ridge-regression transition predictor for the synthetic milestone."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .simulator import TransitionPrediction
from .state import CurriculumAction, LearningState, TransitionObservation

FloatVector = npt.NDArray[np.float64]
FloatMatrix = npt.NDArray[np.float64]


class RidgeTransitionPredictor:
    """State-conditioned ridge predictor implementing ``TransitionSimulator``."""

    def __init__(
        self,
        action_names: Sequence[str] | None = None,
        *,
        ridge: float = 1.0e-3,
    ) -> None:
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        names = tuple(action_names) if action_names is not None else ()
        if len(set(names)) != len(names):
            raise ValueError("action_names must be unique")
        self._action_names: tuple[str, ...] = names
        self._action_to_index: dict[str, int] = {
            name: index for index, name in enumerate(self._action_names)
        }
        self._ridge = ridge
        self._coef: FloatMatrix | None = None
        self._residual_std: FloatVector | None = None
        self._feature_mean: FloatVector | None = None
        self._state_dim: int | None = None
        self._probe_dim: int | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return action names known to the predictor."""

        return self._action_names

    def fit(
        self,
        observations: Sequence[TransitionObservation],
    ) -> RidgeTransitionPredictor:
        """Fit ridge regression from ``(state, action)`` features to probe deltas."""

        observation_list = list(observations)
        if not observation_list:
            raise ValueError("at least one observation is required")

        if not self._action_names:
            names = sorted({observation.action.cluster_ids[0] for observation in observation_list})
            self._set_action_names(tuple(names))

        first_state_vector = observation_list[0].state_before.as_vector()
        self._state_dim = int(first_state_vector.shape[0])
        self._probe_dim = int(observation_list[0].state_before.probe_losses.shape[0])

        features: list[FloatVector] = []
        targets: list[FloatVector] = []
        for observation in observation_list:
            self._validate_observation(observation)
            features.append(self._features(observation.state_before, observation.action))
            targets.append(np.asarray(observation.probe_delta, dtype=np.float64))

        x = np.vstack(features).astype(np.float64)
        y = np.vstack(targets).astype(np.float64)
        regularizer = self._ridge * np.eye(x.shape[1], dtype=np.float64)
        regularizer[0, 0] = 0.0

        lhs = x.T @ x + regularizer
        rhs = x.T @ y
        coef: FloatMatrix = np.linalg.solve(lhs, rhs)
        self._coef = coef

        residuals = y - x @ coef
        ddof = 1 if residuals.shape[0] > 1 else 0
        std = np.std(residuals, axis=0, ddof=ddof)
        self._residual_std = np.maximum(std, 1.0e-6).astype(np.float64)
        self._feature_mean = np.mean(x, axis=0).astype(np.float64)
        return self

    def predict(
        self,
        state: LearningState,
        action: CurriculumAction,
    ) -> TransitionPrediction:
        """Predict a distributional summary for one transition."""

        coef = self._coef
        residual_std = self._residual_std
        feature_mean = self._feature_mean
        if coef is None or residual_std is None or feature_mean is None:
            raise RuntimeError("predictor must be fit before predict is called")

        features = self._features(state, action)
        predicted_delta = np.asarray(features @ coef, dtype=np.float64)
        current_losses = np.asarray(state.probe_losses, dtype=np.float64)
        if predicted_delta.shape != current_losses.shape:
            raise ValueError("predicted delta and probe_losses dimensions differ")

        next_losses = np.clip(current_losses + predicted_delta, 0.0, 1.0)
        clipped_delta = next_losses - current_losses
        next_state = self._build_next_state(state, action, clipped_delta)

        novelty = float(np.linalg.norm(features - feature_mean) / np.sqrt(features.size))
        forgetting_risk = float(np.clip(np.sum(np.maximum(clipped_delta, 0.0)), 0.0, 1.0))
        return TransitionPrediction(
            next_state_mean=next_state,
            probe_delta_mean=clipped_delta,
            probe_delta_std=residual_std.copy(),
            forgetting_risk=forgetting_risk,
            novelty=novelty,
            expected_compute=float(action.optimizer_steps * action.token_budget),
        )

    def _set_action_names(self, names: tuple[str, ...]) -> None:
        if len(set(names)) != len(names):
            raise ValueError("action names must be unique")
        self._action_names = names
        self._action_to_index = {name: index for index, name in enumerate(names)}

    def _features(
        self,
        state: LearningState,
        action: CurriculumAction,
    ) -> FloatVector:
        action_index = self._action_index(action)
        state_vector = np.asarray(state.as_vector(), dtype=np.float64)
        if self._state_dim is not None and state_vector.shape != (self._state_dim,):
            raise ValueError("state vector dimension mismatch")

        probe_dim = state.probe_losses.shape[0]
        if self._probe_dim is not None and probe_dim != self._probe_dim:
            raise ValueError("probe dimension mismatch")

        action_one_hot = np.zeros(len(self._action_names), dtype=np.float64)
        action_one_hot[action_index] = 1.0

        skills = np.clip(1.0 - np.asarray(state.probe_losses, dtype=np.float64), 0.0, 1.0)
        interactions = np.outer(action_one_hot, skills).reshape(-1)
        return np.concatenate(
            (
                np.asarray([1.0], dtype=np.float64),
                state_vector,
                action_one_hot,
                interactions,
            )
        ).astype(np.float64)

    def _validate_observation(self, observation: TransitionObservation) -> None:
        if len(observation.action.cluster_ids) != 1:
            raise ValueError("only single-cluster actions are supported")
        self._action_index(observation.action)

        state_dim = self._state_dim
        if state_dim is None:
            raise RuntimeError("state dimension has not been initialized")
        before_vector = observation.state_before.as_vector()
        after_vector = observation.state_after.as_vector()
        if before_vector.shape != (state_dim,) or after_vector.shape != (state_dim,):
            raise ValueError("state vector dimension mismatch in observation")

        probe_dim = self._probe_dim
        if probe_dim is None:
            raise RuntimeError("probe dimension has not been initialized")
        expected_shape = (probe_dim,)
        if observation.probe_delta.shape != expected_shape:
            raise ValueError("probe_delta dimension mismatch")
        if observation.state_before.probe_losses.shape != expected_shape:
            raise ValueError("state_before probe dimension mismatch")
        if observation.state_after.probe_losses.shape != expected_shape:
            raise ValueError("state_after probe dimension mismatch")

    def _build_next_state(
        self,
        state: LearningState,
        action: CurriculumAction,
        probe_delta: FloatVector,
    ) -> LearningState:
        action_index = self._action_index(action)
        action_count = len(self._action_names)

        if state.exposure_histogram.shape != (action_count,):
            raise ValueError("exposure_histogram dimension must match action count")
        if probe_delta.shape != state.probe_losses.shape:
            raise ValueError("probe_delta dimension mismatch")

        next_losses = np.clip(state.probe_losses + probe_delta, 0.0, 1.0).astype(np.float64)
        next_skills = np.clip(1.0 - next_losses, 0.0, 1.0).astype(np.float64)

        exposure = state.exposure_histogram.copy()
        exposure[action_index] += 1.0

        update_sketch = np.zeros_like(state.update_sketch, dtype=np.float64)
        positive_skill_delta = np.maximum(-probe_delta, 0.0)
        negative_skill_delta = np.maximum(probe_delta, 0.0)
        if update_sketch.size >= 1:
            update_sketch[0] = float(np.sum(positive_skill_delta))
        if update_sketch.size >= 2:
            update_sketch[1] = float(np.sum(negative_skill_delta))
        if update_sketch.size >= 3:
            update_sketch[2] = self._normalized_action_index(action_index)
        if update_sketch.size >= 4:
            update_sketch[3] = action.learning_rate_scale

        optimizer_sketch = np.zeros_like(state.optimizer_sketch, dtype=np.float64)
        if optimizer_sketch.size >= 1:
            optimizer_sketch[0] = float(action.optimizer_steps)
        if optimizer_sketch.size >= 2:
            optimizer_sketch[1] = action.token_budget / 1_000.0

        activation_sketch = np.zeros_like(state.activation_sketch, dtype=np.float64)
        activation_values = (
            float(np.mean(next_skills)),
            float(np.max(next_skills)),
            float(np.min(next_skills)),
            float(np.std(next_skills)),
        )
        for index, value in enumerate(activation_values[: activation_sketch.size]):
            activation_sketch[index] = value

        bridge_skill = float(next_skills[4]) if next_skills.size > 4 else 0.0
        replay_skill = float(next_skills[6]) if next_skills.size > 6 else 0.0
        history_embedding = np.zeros_like(state.history_embedding, dtype=np.float64)
        history_values = (
            self._normalized_action_index(action_index),
            (state.step + 1) / 50.0,
            bridge_skill,
            replay_skill,
        )
        for index, value in enumerate(history_values[: history_embedding.size]):
            history_embedding[index] = value

        return LearningState(
            probe_losses=next_losses,
            update_sketch=update_sketch,
            optimizer_sketch=optimizer_sketch,
            activation_sketch=activation_sketch,
            exposure_histogram=exposure,
            history_embedding=history_embedding,
            architecture_embedding=state.architecture_embedding.copy(),
            step=state.step + 1,
            tokens_seen=state.tokens_seen + action.token_budget,
        )

    def _action_index(self, action: CurriculumAction) -> int:
        if len(action.cluster_ids) != 1:
            raise ValueError("only single-cluster actions are supported")
        name = action.cluster_ids[0]
        if name not in self._action_to_index:
            raise ValueError(f"unknown action cluster: {name}")
        return self._action_to_index[name]

    def _normalized_action_index(self, action_index: int) -> float:
        denominator = max(len(self._action_names) - 1, 1)
        return action_index / denominator
