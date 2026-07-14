"""Evaluation helpers for synthetic curriculum policies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .policies import plan_greedy, plan_mcts
from .predictor import RidgeTransitionPredictor
from .state import CurriculumAction
from .synthetic import SyntheticEnvironment


def evaluate_curriculum(
    env: SyntheticEnvironment,
    actions: Sequence[CurriculumAction],
    seed: int,
) -> float:
    """Evaluate a finished curriculum using only the true environment."""

    rng = np.random.default_rng(seed)
    state = env.initial_state()
    for action in actions:
        state = env.step(state, action, rng)
    return env.value(state)


def run_multi_seed_comparison(
    n_seeds: int,
    *,
    horizon: int = 6,
    oracle_rollouts: int = 80,
    oracle_horizon: int = 6,
    mcts_simulations: int = 160,
    seed_offset: int = 0,
) -> dict[str, dict[str, float]]:
    """Compare greedy and MCTS policies over multiple deterministic seeds."""

    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")

    greedy_values: list[float] = []
    mcts_values: list[float] = []
    greedy_discoveries = 0
    mcts_discoveries = 0

    for seed_index in range(n_seeds):
        seed = seed_offset + seed_index
        env = SyntheticEnvironment()
        rng = np.random.default_rng(seed)
        observations = env.generate_oracle_rollouts(
            n_rollouts=oracle_rollouts,
            horizon=oracle_horizon,
            rng=rng,
        )
        predictor = RidgeTransitionPredictor(
            action_names=env.cluster_names,
            ridge=1.0e-3,
        ).fit(observations)

        action_space = env.actions()
        initial_state = env.initial_state()
        greedy_plan = plan_greedy(
            predictor,
            initial_state,
            action_space,
            horizon,
            env.value,
        )
        mcts_plan = plan_mcts(
            predictor,
            initial_state,
            action_space,
            horizon,
            env.value,
            simulations_per_step=mcts_simulations,
        )

        greedy_values.append(evaluate_curriculum(env, greedy_plan, seed + 10_000))
        mcts_values.append(evaluate_curriculum(env, mcts_plan, seed + 20_000))
        greedy_discoveries += int(contains_bridge_sequence(greedy_plan))
        mcts_discoveries += int(contains_bridge_sequence(mcts_plan))

    return {
        "greedy": _summary(greedy_values, greedy_discoveries, n_seeds),
        "mcts": _summary(mcts_values, mcts_discoveries, n_seeds),
    }


def contains_bridge_sequence(actions: Sequence[CurriculumAction]) -> bool:
    """Return whether ``bridge`` is taken before ``bridge_payload``.

    The bridge skill does not decay, so any plan that schedules the payload
    after the bridge exploits the prerequisite; adjacency is not required.
    """

    names = [action.cluster_ids[0] for action in actions]
    if "bridge" not in names or "bridge_payload" not in names:
        return False
    return names.index("bridge") < names.index("bridge_payload")


def _summary(
    values: Sequence[float],
    discoveries: int,
    n_seeds: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    discovery_rate = discoveries / n_seeds
    return {
        "mean": mean,
        "std": std,
        "mean_terminal_value": mean,
        "std_terminal_value": std,
        "discovery_rate": discovery_rate,
        "bridge_discovery_rate": discovery_rate,
        "n_seeds": float(n_seeds),
    }
