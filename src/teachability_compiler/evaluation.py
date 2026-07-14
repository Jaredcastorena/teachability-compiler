"""Evaluation helpers for synthetic curriculum policies."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .policies import plan_greedy, plan_mcts
from .predictor import RidgeTransitionPredictor
from .state import CurriculumAction, LearningState
from .synthetic import SyntheticEnvironment

NO_CHECKPOINT_SENTINEL = "synthetic:no-checkpoint"


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Structured experiment result with mandatory accounting.

    ``policies`` holds per-policy metric summaries. ``provenance`` records
    seed range, config hash, git commit, target checkpoint identifier, and
    simulator version. ``overhead`` reports oracle and search compute so a
    curriculum gain can never be quoted without its total cost.
    """

    policies: dict[str, dict[str, float]] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    overhead: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, policy: str) -> dict[str, float]:
        return self.policies[policy]


def evaluate_curriculum(
    env: SyntheticEnvironment,
    actions: Sequence[CurriculumAction],
    seed: int,
) -> float:
    """Score a finished curriculum with the planner-visible objective.

    Useful for diagnosing environment dynamics; final policy comparisons must
    use :func:`evaluate_curriculum_held_out` instead.
    """

    return env.value(_final_state(env, actions, seed))


def evaluate_curriculum_held_out(
    env: SyntheticEnvironment,
    actions: Sequence[CurriculumAction],
    seed: int,
) -> float:
    """Score a finished curriculum on held-out probes the planner never saw."""

    return env.held_out_value(_final_state(env, actions, seed))


def run_multi_seed_comparison(
    n_seeds: int,
    *,
    horizon: int = 6,
    oracle_rollouts: int = 80,
    oracle_horizon: int = 6,
    mcts_simulations: int = 160,
    seed_offset: int = 0,
) -> ExperimentReport:
    """Compare greedy and MCTS policies over multiple deterministic seeds.

    Planners optimize the planner-visible objective through the learned
    predictor; the reported comparison metrics are computed on the true
    environment, on both the planner-visible and the held-out objectives.
    """

    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")

    config = {
        "n_seeds": n_seeds,
        "horizon": horizon,
        "oracle_rollouts": oracle_rollouts,
        "oracle_horizon": oracle_horizon,
        "mcts_simulations": mcts_simulations,
        "seed_offset": seed_offset,
    }

    greedy_values: list[float] = []
    mcts_values: list[float] = []
    greedy_held_out: list[float] = []
    mcts_held_out: list[float] = []
    greedy_discoveries = 0
    mcts_discoveries = 0
    oracle_transitions = 0
    oracle_compute_cost = 0.0
    greedy_simulator_calls = 0
    mcts_simulator_calls = 0
    simulator_version = "unfitted"

    for seed_index in range(n_seeds):
        seed = seed_offset + seed_index
        env = SyntheticEnvironment()
        rng = np.random.default_rng(seed)
        observations = env.generate_oracle_rollouts(
            n_rollouts=oracle_rollouts,
            horizon=oracle_horizon,
            rng=rng,
        )
        oracle_transitions += len(observations)
        oracle_compute_cost += sum(observation.compute_cost for observation in observations)
        predictor = RidgeTransitionPredictor(
            action_names=env.cluster_names,
            ridge=1.0e-3,
        ).fit(observations)
        simulator_version = predictor.version

        action_space = env.actions()
        initial_state = env.initial_state()

        calls_before = predictor.predict_calls
        greedy_plan = plan_greedy(
            predictor,
            initial_state,
            action_space,
            horizon,
            env.value,
        )
        greedy_simulator_calls += predictor.predict_calls - calls_before

        calls_before = predictor.predict_calls
        mcts_plan = plan_mcts(
            predictor,
            initial_state,
            action_space,
            horizon,
            env.value,
            simulations_per_step=mcts_simulations,
        )
        mcts_simulator_calls += predictor.predict_calls - calls_before

        greedy_values.append(evaluate_curriculum(env, greedy_plan, seed + 10_000))
        mcts_values.append(evaluate_curriculum(env, mcts_plan, seed + 20_000))
        greedy_held_out.append(evaluate_curriculum_held_out(env, greedy_plan, seed + 10_000))
        mcts_held_out.append(evaluate_curriculum_held_out(env, mcts_plan, seed + 20_000))
        greedy_discoveries += int(contains_bridge_sequence(greedy_plan))
        mcts_discoveries += int(contains_bridge_sequence(mcts_plan))

    provenance = {
        "seeds": f"{seed_offset}..{seed_offset + n_seeds - 1}",
        "config_hash": _config_hash(config),
        "git_commit": _git_commit(),
        "target_checkpoint": NO_CHECKPOINT_SENTINEL,
        "simulator_version": simulator_version,
        "environment_version": "synthetic-v1",
    }
    overhead = {
        "oracle_transitions": float(oracle_transitions),
        "oracle_compute_cost": oracle_compute_cost,
        "greedy_simulator_calls": float(greedy_simulator_calls),
        "mcts_simulator_calls": float(mcts_simulator_calls),
        "true_env_evaluation_steps": float(4 * n_seeds * horizon),
    }
    return ExperimentReport(
        policies={
            "greedy": _summary(greedy_values, greedy_held_out, greedy_discoveries, n_seeds),
            "mcts": _summary(mcts_values, mcts_held_out, mcts_discoveries, n_seeds),
        },
        provenance=provenance,
        overhead=overhead,
    )


def contains_bridge_sequence(actions: Sequence[CurriculumAction]) -> bool:
    """Return whether ``bridge`` is taken before ``bridge_payload``.

    The bridge skill does not decay, so any plan that schedules the payload
    after the bridge exploits the prerequisite; adjacency is not required.
    """

    names = [action.cluster_ids[0] for action in actions]
    if "bridge" not in names or "bridge_payload" not in names:
        return False
    return names.index("bridge") < names.index("bridge_payload")


def _final_state(
    env: SyntheticEnvironment,
    actions: Sequence[CurriculumAction],
    seed: int,
) -> LearningState:
    rng = np.random.default_rng(seed)
    state = env.initial_state()
    for action in actions:
        state = env.step(state, action, rng)
    return state


def _summary(
    values: Sequence[float],
    held_out_values: Sequence[float],
    discoveries: int,
    n_seeds: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    held_out_array = np.asarray(held_out_values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    discovery_rate = discoveries / n_seeds
    return {
        "mean": mean,
        "std": std,
        "mean_terminal_value": mean,
        "std_terminal_value": std,
        "mean_held_out_value": float(np.mean(held_out_array)),
        "std_held_out_value": float(np.std(held_out_array, ddof=0)),
        "discovery_rate": discovery_rate,
        "bridge_discovery_rate": discovery_rate,
        "n_seeds": float(n_seeds),
    }


def _config_hash(config: dict[str, int]) -> str:
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except OSError:
        return "unknown"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unknown"
