import numpy as np

from teachability_compiler.metrics import directed_order_advantage, functional_commutator


def test_functional_commutator_detects_order_effect() -> None:
    state = np.asarray([1.0, 1.0])

    def transition_a(x: np.ndarray) -> np.ndarray:
        # Establishes a scale used by B.
        return np.asarray([2.0 * x[0], x[1]])

    def transition_b(x: np.ndarray) -> np.ndarray:
        # Couples the second coordinate to the current first coordinate.
        return np.asarray([x[0], x[1] + x[0]])

    score = functional_commutator(
        state,
        transition_a,
        transition_b,
        probe_map=lambda x: x,
    )

    assert score > 0.0


def test_directed_order_advantage_has_expected_sign() -> None:
    state = np.asarray([1.0, 1.0])

    def transition_a(x: np.ndarray) -> np.ndarray:
        return np.asarray([2.0 * x[0], x[1]])

    def transition_b(x: np.ndarray) -> np.ndarray:
        return np.asarray([x[0], x[1] + x[0]])

    advantage = directed_order_advantage(
        state,
        transition_a,
        transition_b,
        value=lambda x: float(x[1]),
    )

    # A then B lets B use the enlarged first coordinate.
    assert advantage > 0.0
