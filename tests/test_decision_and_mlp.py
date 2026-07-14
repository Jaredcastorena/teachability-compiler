"""Fast CPU tests for decision metrics, persistence, and residual MLP prediction."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from teachability_compiler.real.decision_metrics import decision_metrics, kendall_tau
from teachability_compiler.real.mlp_predictor import ResidualMLPTransitionPredictor
from teachability_compiler.real.persistence import load_transitions, save_transitions
from teachability_compiler.state import CurriculumAction, LearningState, TransitionObservation


def _make_state(rng: np.random.Generator, probes: np.ndarray, step: int) -> LearningState:
    return LearningState(
        probe_losses=np.asarray(probes, dtype=np.float64),
        update_sketch=np.asarray(rng.normal(size=4), dtype=np.float64),
        optimizer_sketch=np.asarray(rng.normal(size=2), dtype=np.float64),
        activation_sketch=np.asarray(rng.normal(size=2), dtype=np.float64),
        exposure_histogram=np.zeros(3, dtype=np.float64),
        history_embedding=np.asarray(rng.normal(size=2), dtype=np.float64),
        architecture_embedding=np.asarray(rng.normal(size=2), dtype=np.float64),
        step=int(step),
        tokens_seen=int(step) * 100,
    )


def _make_action(name: str) -> CurriculumAction:
    return CurriculumAction(
        cluster_ids=(name,),
        mixture_weights=(1.0,),
        optimizer_steps=8,
        token_budget=8 * 64 * 64,
    )


def _synthetic_observations(rng: np.random.Generator) -> list[TransitionObservation]:
    names = ("a", "b", "c")
    actions = {name: _make_action(name) for name in names}
    observations: list[TransitionObservation] = []

    for group_index in range(13):
        base_probe = rng.uniform(0.2, 1.8, size=4)
        state_before = _make_state(rng, base_probe, step=group_index)

        for action_index, name in enumerate(names):
            action_effect = -0.05 * action_index
            state_effect = 0.02 * np.tanh(base_probe - 1.0)
            noise = rng.normal(scale=0.01, size=4)
            delta = action_effect + state_effect + noise
            after_probe = np.clip(base_probe + delta, 0.0, 10.0)
            state_after = _make_state(rng, after_probe, step=group_index + 1)

            observations.append(
                TransitionObservation(
                    state_before=state_before,
                    action=actions[name],
                    state_after=state_after,
                    parameter_delta_sketch=np.asarray(rng.normal(size=2), dtype=np.float64),
                    probe_delta=np.asarray(delta, dtype=np.float64),
                    activation_delta_sketch=np.asarray(rng.normal(size=2), dtype=np.float64),
                    compute_cost=1.0,
                    seed_metadata={"seed": group_index},
                    simulator_version=None,
                )
            )

    return observations


def test_kendall_tau_perfect_and_reversed() -> None:
    increasing = np.array([1.0, 2.0, 3.0, 4.0])
    decreasing = np.array([4.0, 3.0, 2.0, 1.0])
    assert kendall_tau(increasing, increasing) == 1.0
    assert kendall_tau(increasing, decreasing) == -1.0


def test_kendall_tau_hand_case() -> None:
    tau = kendall_tau(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 3.0, 2.0]),
    )
    assert tau == pytest.approx(1.0 / 3.0)


def test_decision_metrics_known_values() -> None:
    true_values = np.arange(20, dtype=np.float64)
    predicted_values = np.arange(20, dtype=np.float64)
    predicted_values[0], predicted_values[19] = predicted_values[19], predicted_values[0]

    metrics = decision_metrics(true_values, predicted_values)

    assert metrics["top1_agreement"] == 0.0
    assert metrics["top3_recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["selected_regret"] == pytest.approx(19.0)
    assert -1.0 <= metrics["kendall_tau"] <= 1.0


def test_decision_metrics_requires_equal_length_at_least_three() -> None:
    with pytest.raises(ValueError):
        decision_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        decision_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_residual_mlp_fit_and_predict() -> None:
    rng = np.random.default_rng(0)
    observations = _synthetic_observations(rng)

    predictor = ResidualMLPTransitionPredictor(
        action_names=("a", "b", "c"),
        device="cpu",
        seed=0,
    )
    predictor.fit(
        observations,
        epochs=80,
        ranking_weight=0.1,
        batch_size=64,
    )

    assert len(predictor.train_losses) == 80
    assert predictor.train_losses[-1] < predictor.train_losses[0]

    prediction = predictor.predict(observations[0].state_before, observations[0].action)
    assert prediction.probe_delta_mean.shape == (4,)
    assert prediction.next_state_mean.probe_losses.shape == (4,)
    assert np.all(np.isfinite(prediction.probe_delta_mean))
    assert np.all(np.isfinite(prediction.next_state_mean.probe_losses))
    assert predictor.predict_calls == 1


def test_residual_mlp_rejects_unknown_action() -> None:
    rng = np.random.default_rng(1)
    observations = _synthetic_observations(rng)
    predictor = ResidualMLPTransitionPredictor(action_names=("a", "b", "c"), seed=0)
    predictor.fit(observations, epochs=5)

    with pytest.raises(ValueError):
        predictor.predict(observations[0].state_before, _make_action("z"))


def test_persistence_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    observations = _synthetic_observations(rng)
    metadata = {"experiment": "test", "probe_suite_hash": "deadbeef00000000"}
    path = tmp_path / "transitions.pkl"

    save_transitions(path, observations, metadata)
    loaded_observations, loaded_metadata = load_transitions([path])

    assert len(loaded_observations) == len(observations)
    assert loaded_metadata == [metadata]
    np.testing.assert_allclose(
        loaded_observations[0].probe_delta,
        observations[0].probe_delta,
    )
