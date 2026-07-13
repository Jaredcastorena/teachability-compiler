"""Metrics for order sensitivity and transition comparison."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatVector = npt.NDArray[np.float64]
State = FloatVector
Transition = Callable[[State], State]
ProbeMap = Callable[[State], FloatVector]
ValueFunction = Callable[[State], float]


def weighted_norm(vector: FloatVector, weight: FloatVector | None = None) -> float:
    """Compute ||v||_W for a diagonal weight matrix W."""
    vector = np.asarray(vector, dtype=np.float64)
    if weight is None:
        return float(np.linalg.norm(vector))
    weight = np.asarray(weight, dtype=np.float64)
    if vector.shape != weight.shape:
        raise ValueError("vector and diagonal weight must have equal shape")
    if np.any(weight < 0.0):
        raise ValueError("weights must be non-negative")
    return float(np.sqrt(np.sum(weight * vector * vector)))


def functional_commutator(
    state: State,
    transition_a: Transition,
    transition_b: Transition,
    probe_map: ProbeMap,
    weight: FloatVector | None = None,
) -> float:
    """Measure ||Phi(T_B(T_A(s))) - Phi(T_A(T_B(s)))||_W."""
    ab = probe_map(transition_b(transition_a(state.copy())))
    ba = probe_map(transition_a(transition_b(state.copy())))
    return weighted_norm(np.asarray(ab) - np.asarray(ba), weight)


def directed_order_advantage(
    state: State,
    transition_a: Transition,
    transition_b: Transition,
    value: ValueFunction,
) -> float:
    """Return V(T_B(T_A(s))) - V(T_A(T_B(s)))."""
    ab = transition_b(transition_a(state.copy()))
    ba = transition_a(transition_b(state.copy()))
    return float(value(ab) - value(ba))
