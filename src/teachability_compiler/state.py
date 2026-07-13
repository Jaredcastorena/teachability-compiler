"""Typed data contracts for learner states, curriculum actions, and transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt

FloatVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LearningState:
    """Compressed state used by the transition simulator and planner."""

    probe_losses: FloatVector
    update_sketch: FloatVector
    optimizer_sketch: FloatVector
    activation_sketch: FloatVector
    exposure_histogram: FloatVector
    history_embedding: FloatVector
    architecture_embedding: FloatVector
    step: int
    tokens_seen: int

    def as_vector(self) -> FloatVector:
        """Concatenate continuous state fields into one simulator input vector."""
        return np.concatenate(
            (
                self.probe_losses,
                self.update_sketch,
                self.optimizer_sketch,
                self.activation_sketch,
                self.exposure_histogram,
                self.history_embedding,
                self.architecture_embedding,
                np.asarray([self.step, self.tokens_seen], dtype=np.float64),
            )
        )


@dataclass(frozen=True, slots=True)
class CurriculumAction:
    """One executable curriculum decision."""

    cluster_ids: tuple[str, ...]
    mixture_weights: tuple[float, ...]
    optimizer_steps: int
    token_budget: int
    learning_rate_scale: float = 1.0
    replay_policy: str | None = None
    materialization_seed: int = 0

    def __post_init__(self) -> None:
        if not self.cluster_ids:
            raise ValueError("cluster_ids cannot be empty")
        if len(self.cluster_ids) != len(self.mixture_weights):
            raise ValueError("cluster_ids and mixture_weights must have equal length")
        if any(weight < 0.0 for weight in self.mixture_weights):
            raise ValueError("mixture weights must be non-negative")
        if not np.isclose(sum(self.mixture_weights), 1.0):
            raise ValueError("mixture weights must sum to 1")
        if self.optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.learning_rate_scale <= 0.0:
            raise ValueError("learning_rate_scale must be positive")


@dataclass(frozen=True, slots=True)
class TransitionObservation:
    """Immutable oracle observation for simulator training."""

    state_before: LearningState
    action: CurriculumAction
    state_after: LearningState
    parameter_delta_sketch: FloatVector
    probe_delta: FloatVector
    activation_delta_sketch: FloatVector
    compute_cost: float
    seed_metadata: Mapping[str, int]
    simulator_version: str | None = None
