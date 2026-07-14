"""Deterministic milestone tests for the synthetic teachability compiler."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from teachability_compiler.evaluation import (
    evaluate_curriculum,
    run_multi_seed_comparison,
)
from teachability_compiler.metrics import directed_order_advantage, functional_commutator
from teachability_compiler.state import CurriculumAction, LearningState
from teachability_compiler.synthetic import SyntheticEnvironment, VALUE_WEIGHTS

FloatVector = npt.NDArray[np.float64]


def test_prerequisite_ordering_matters() -> None:
    """The advanced prerequisite cluster is valuable only after the basic one."""

    env = SyntheticEnvironment()
    basic = env.action_by_name("prereq_basic")
    advanced = env.action_by_name("prereq_advanced")
    initial_losses = env.initial_state().probe_losses.copy()

    advantage = directed_order_advantage(
        initial_losses,
        _loss_transition(env, basic),
        _loss_transition(env, advanced),
        _loss_value,
    )
    assert advantage > 0.45


def test_destructive_pair_is_measurable() -> None:
    """Training on destructive_x after fragile_y measurably erases fragile skill."""

    env = SyntheticEnvironment()
    destructive = env.action_by_name("destructive_x")
    fragile = env.action_by_name("fragile_y")
    initial_losses = env.initial_state().probe_losses.copy()
    fragile_weight = np.zeros_like(initial_losses)
    fragile_weight[3] = 1.0

    commutator = functional_commutator(
        initial_losses,
        _loss_transition(env, fragile),
        _loss_transition(env, destructive),
        lambda state: state,
        fragile_weight,
    )
    assert commutator > 0.30

    advantage = directed_order_advantage(
        initial_losses,
        _loss_transition(env, destructive),
        _loss_transition(env, fragile),
        _loss_value,
    )
    assert advantage > 0.25


def test_bridge_lower_immediate_reward_but_higher_terminal_value() -> None:
    """The bridge has low immediate reward but unlocks a high terminal value."""

    env = SyntheticEnvironment()
    bridge = env.action_by_name("bridge")
    payload = env.action_by_name("bridge_payload")
    replay = env.action_by_name("replay_foundation")
    fragile = env.action_by_name("fragile_y")

    base_value = env.value(env.initial_state())
    bridge_immediate = _value_after(env, [bridge]) - base_value
    replay_immediate = _value_after(env, [replay]) - base_value
    assert bridge_immediate < replay_immediate

    bridge_terminal = _value_after(env, [bridge, payload])
    high_immediate_terminal = _value_after(env, [replay, fragile])
    wrong_order_terminal = _value_after(env, [payload, bridge])

    assert bridge_terminal > high_immediate_terminal + 1.0
    assert bridge_terminal > wrong_order_terminal + 2.0


def test_replay_sensitivity() -> None:
    """Replay protects the replay skill from step-wise decay."""

    env = SyntheticEnvironment()
    without_replay = [
        env.action_by_name("replay_foundation"),
        env.action_by_name("core_math"),
        env.action_by_name("fragile_y"),
        env.action_by_name("calibration"),
        env.action_by_name("logic_drills"),
    ]
    with_replay = [
        env.action_by_name("replay_foundation"),
        env.action_by_name("core_math"),
        env.action_by_name("fragile_y"),
        env.action_by_name("calibration"),
        env.action_by_name("replay_review"),
    ]

    state_without = _state_after(env, without_replay)
    state_with = _state_after(env, with_replay)

    assert state_with.probe_losses[6] < state_without.probe_losses[6]
    assert env.value(state_with) > env.value(state_without)


def test_mcts_discovers_bridge_more_often_than_greedy() -> None:
    """Decisive test: MCTS finds the bridge more often and scores higher."""

    result = run_multi_seed_comparison(
        5,
        horizon=6,
        oracle_rollouts=70,
        oracle_horizon=6,
        mcts_simulations=120,
    )

    assert (
        result["mcts"]["bridge_discovery_rate"]
        > result["greedy"]["bridge_discovery_rate"]
    )
    assert result["mcts"]["mean_terminal_value"] > result["greedy"]["mean_terminal_value"]


def _loss_transition(
    env: SyntheticEnvironment,
    action: CurriculumAction,
) -> Callable[[FloatVector], FloatVector]:
    def transition(losses: FloatVector) -> FloatVector:
        state = _state_from_losses(env, losses)
        next_state = env.step(state, action, np.random.default_rng(123))
        return next_state.probe_losses.copy()

    return transition


def _loss_value(losses: FloatVector) -> float:
    return -float(np.sum(VALUE_WEIGHTS * losses))


def _value_after(
    env: SyntheticEnvironment,
    actions: Sequence[CurriculumAction],
) -> float:
    return evaluate_curriculum(env, actions, seed=999)


def _state_after(
    env: SyntheticEnvironment,
    actions: Sequence[CurriculumAction],
) -> LearningState:
    rng = np.random.default_rng(999)
    state = env.initial_state()
    for action in actions:
        state = env.step(state, action, rng)
    return state


def _state_from_losses(
    env: SyntheticEnvironment,
    losses: FloatVector,
) -> LearningState:
    initial = env.initial_state()
    return LearningState(
        probe_losses=np.asarray(losses, dtype=np.float64).copy(),
        update_sketch=initial.update_sketch.copy(),
        optimizer_sketch=initial.optimizer_sketch.copy(),
        activation_sketch=initial.activation_sketch.copy(),
        exposure_histogram=initial.exposure_histogram.copy(),
        history_embedding=initial.history_embedding.copy(),
        architecture_embedding=initial.architecture_embedding.copy(),
        step=initial.step,
        tokens_seen=initial.tokens_seen,
    )
