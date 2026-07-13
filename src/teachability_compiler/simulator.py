"""Protocols for learned transition simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .state import CurriculumAction, LearningState

FloatVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TransitionPrediction:
    """Probabilistic prediction returned by a simulator."""

    next_state_mean: LearningState
    probe_delta_mean: FloatVector
    probe_delta_std: FloatVector
    forgetting_risk: float
    novelty: float
    expected_compute: float


class TransitionSimulator(Protocol):
    """Replaceable interface for target-learning dynamics models."""

    def predict(
        self,
        state: LearningState,
        action: CurriculumAction,
    ) -> TransitionPrediction:
        """Predict a distributional summary of the action's learning effect."""
        ...
