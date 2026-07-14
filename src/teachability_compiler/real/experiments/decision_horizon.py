"""Measure the decision-valid horizon of a transition predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from teachability_compiler.predictor import RidgeTransitionPredictor
from teachability_compiler.real.decision_metrics import decision_metrics
from teachability_compiler.real.model import DecoderConfig
from teachability_compiler.real.oracle import RealLearnerOracle
from teachability_compiler.real.persistence import probe_suite_hash, save_transitions
from teachability_compiler.real.tasks import VOCAB_SIZE, all_cluster_names
from teachability_compiler.state import CurriculumAction, TransitionObservation

_METRIC_KEYS = ("top1_agreement", "top3_recall", "kendall_tau", "selected_regret")


def main() -> None:
    """Run the decision-horizon measurement."""
    args = _parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_time = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    cluster_names = all_cluster_names()
    n_clusters = len(cluster_names)

    config = DecoderConfig(vocab_size=VOCAB_SIZE)
    oracle = RealLearnerOracle(
        config=config,
        cluster_names=cluster_names,
        device=args.device,
        base_seed=args.seed,
        steps_per_action=args.steps,
    )
    overhead_steps = int(args.pretrain_steps)
    overhead_tokens = int(args.pretrain_steps * oracle.batch_size * oracle.seq_len)

    oracle.pretrain(n_steps=args.pretrain_steps, rng_seed=args.seed + 1)
    checkpoint = oracle.snapshot()

    observations: list[TransitionObservation] = []
    for _ in range(args.train_rollouts):
        oracle.restore(checkpoint)
        sequence = _random_action_sequence(
            rng=rng,
            cluster_names=cluster_names,
            length=args.rollout_horizon,
            steps=args.steps,
            oracle=oracle,
        )
        for action in sequence:
            observation = oracle.apply_action(action, data_seed=_next_seed(rng))
            observations.append(observation)
            overhead_steps += args.steps
            overhead_tokens += args.steps * oracle.batch_size * oracle.seq_len

    predictor = RidgeTransitionPredictor(action_names=cluster_names, ridge=1e-3).fit(observations)
    mean_delta = _fit_action_only_mean_delta(observations, cluster_names)

    candidate_actions = [_single_action(name, args.steps, oracle) for name in cluster_names]
    saved_transitions: list[TransitionObservation] = list(observations)
    all_prefix_results: list[list[dict[str, dict[str, float]]]] = []

    for _ in range(args.prefixes):
        oracle.restore(checkpoint)
        initial_state = oracle.encode_state()
        open_loop_state = initial_state
        baseline_probe = np.asarray(initial_state.probe_losses, dtype=np.float64).copy()
        prefix_actions = _random_action_sequence(
            rng=rng,
            cluster_names=cluster_names,
            length=max(0, args.depths - 1),
            steps=args.steps,
            oracle=oracle,
        )

        depth_results: list[dict[str, dict[str, float]]] = []
        for depth_index in range(args.depths):
            snap_d = oracle.snapshot()

            true_values = np.zeros(n_clusters, dtype=np.float64)
            for candidate_index, candidate in enumerate(candidate_actions):
                oracle.restore(snap_d)
                observation = oracle.apply_action(candidate, data_seed=_next_seed(rng))
                true_values[candidate_index] = float(oracle.value(observation.state_after))
                saved_transitions.append(observation)
                overhead_steps += args.steps
                overhead_tokens += args.steps * oracle.batch_size * oracle.seq_len
            oracle.restore(snap_d)

            ridge_values = np.zeros(n_clusters, dtype=np.float64)
            for candidate_index, candidate in enumerate(candidate_actions):
                prediction = predictor.predict(open_loop_state, candidate)
                probe_losses = prediction.next_state_mean.probe_losses
                ridge_values[candidate_index] = -float(np.mean(probe_losses))

            baseline_values = np.zeros(n_clusters, dtype=np.float64)
            for candidate_index, name in enumerate(cluster_names):
                baseline_values[candidate_index] = -float(
                    np.mean(baseline_probe + mean_delta[name])
                )

            depth_results.append(
                {
                    "ridge": decision_metrics(true_values, ridge_values),
                    "baseline": decision_metrics(true_values, baseline_values),
                }
            )

            if depth_index < len(prefix_actions):
                prefix_action = prefix_actions[depth_index]
                prefix_observation = oracle.apply_action(
                    prefix_action,
                    data_seed=_next_seed(rng),
                )
                saved_transitions.append(prefix_observation)
                overhead_steps += args.steps
                overhead_tokens += args.steps * oracle.batch_size * oracle.seq_len

                open_loop_state = predictor.predict(
                    open_loop_state,
                    prefix_action,
                ).next_state_mean
                baseline_probe = baseline_probe + mean_delta[prefix_action.cluster_ids[0]]

        all_prefix_results.append(depth_results)

    per_depth = _aggregate(all_prefix_results, args.depths, args.prefixes)
    ridge_horizon = _decision_valid_horizon(
        [record["ridge"]["top1_agreement"] for record in per_depth],
        threshold=0.5,
    )
    baseline_horizon = _decision_valid_horizon(
        [record["baseline"]["top1_agreement"] for record in per_depth],
        threshold=0.5,
    )

    git_commit = _git_commit()
    probe_hash = probe_suite_hash()
    wall_seconds = time.perf_counter() - start_time
    output = {
        "per_depth": per_depth,
        "horizons": {
            "ridge": ridge_horizon,
            "baseline": baseline_horizon,
        },
        "train_transition_count": len(observations),
        "provenance": {
            "seed": args.seed,
            "config_hash": _config_hash(args),
            "git_commit": git_commit,
            "target_checkpoint": f"pretrain-{args.pretrain_steps}steps-seed{args.seed}",
            "simulator_version": "ridge-v1",
            "environment_version": "real-v1",
            "probe_suite_hash": probe_hash,
            "args": vars(args),
        },
        "overhead": {
            "total_optimizer_steps": overhead_steps,
            "total_tokens": overhead_tokens,
            "wall_seconds": wall_seconds,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(per_depth, ridge_horizon, baseline_horizon)

    if args.save_transitions is not None:
        save_transitions(
            args.save_transitions,
            saved_transitions,
            {
                "experiment": "decision_horizon",
                "pretrain_steps": args.pretrain_steps,
                "seed": args.seed,
                "probe_suite_hash": probe_hash,
                "git_commit": git_commit,
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--train-rollouts", type=int, default=40)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--prefixes", type=int, default=4)
    parser.add_argument("--depths", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/decision_horizon.json")
    parser.add_argument("--save-transitions", type=str, default=None)
    args = parser.parse_args()

    if args.pretrain_steps < 0:
        raise ValueError("--pretrain-steps must be non-negative")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.train_rollouts <= 0:
        raise ValueError("--train-rollouts must be positive")
    if args.rollout_horizon <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if args.prefixes <= 0:
        raise ValueError("--prefixes must be positive")
    if args.depths <= 0:
        raise ValueError("--depths must be positive")
    return args


def _random_action_sequence(
    rng: np.random.Generator,
    cluster_names: tuple[str, ...],
    length: int,
    steps: int,
    oracle: RealLearnerOracle,
) -> list[CurriculumAction]:
    if length <= 0:
        return []
    sampled = rng.choice(cluster_names, size=length, replace=True)
    return [_single_action(str(name), steps, oracle) for name in sampled]


def _single_action(name: str, steps: int, oracle: RealLearnerOracle) -> CurriculumAction:
    return CurriculumAction(
        cluster_ids=(name,),
        mixture_weights=(1.0,),
        optimizer_steps=steps,
        token_budget=steps * oracle.batch_size * oracle.seq_len,
    )


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _fit_action_only_mean_delta(
    observations: list[TransitionObservation],
    cluster_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    deltas_by_action: dict[str, list[np.ndarray]] = {name: [] for name in cluster_names}
    all_deltas: list[np.ndarray] = []

    for observation in observations:
        if len(observation.action.cluster_ids) != 1:
            raise ValueError("state-blind baseline expects single-cluster actions")
        name = observation.action.cluster_ids[0]
        if name not in deltas_by_action:
            raise ValueError(f"unknown action {name!r}")
        delta = np.asarray(observation.probe_delta, dtype=np.float64)
        deltas_by_action[name].append(delta)
        all_deltas.append(delta)

    if not all_deltas:
        raise ValueError("at least one transition is required")
    global_mean = np.mean(np.stack(all_deltas, axis=0), axis=0)

    result: dict[str, np.ndarray] = {}
    for name, deltas in deltas_by_action.items():
        result[name] = np.mean(np.stack(deltas, axis=0), axis=0) if deltas else global_mean.copy()
    return result


def _aggregate(
    all_prefix_results: list[list[dict[str, dict[str, float]]]],
    depths: int,
    prefixes: int,
) -> list[dict[str, Any]]:
    per_depth: list[dict[str, Any]] = []
    for depth_index in range(depths):
        ridge_values: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
        baseline_values: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
        ridge_per_prefix: list[dict[str, float]] = []
        baseline_per_prefix: list[dict[str, float]] = []

        for prefix_index in range(prefixes):
            ridge_metrics = all_prefix_results[prefix_index][depth_index]["ridge"]
            baseline_metrics = all_prefix_results[prefix_index][depth_index]["baseline"]
            ridge_per_prefix.append(ridge_metrics)
            baseline_per_prefix.append(baseline_metrics)
            for key in _METRIC_KEYS:
                ridge_values[key].append(ridge_metrics[key])
                baseline_values[key].append(baseline_metrics[key])

        per_depth.append(
            {
                "depth": depth_index + 1,
                "ridge": {
                    key: float(np.mean(ridge_values[key]))
                    for key in _METRIC_KEYS
                },
                "baseline": {
                    key: float(np.mean(baseline_values[key]))
                    for key in _METRIC_KEYS
                },
                "ridge_per_prefix": ridge_per_prefix,
                "baseline_per_prefix": baseline_per_prefix,
            }
        )
    return per_depth


def _decision_valid_horizon(top1_by_depth: list[float], threshold: float) -> int:
    horizon = 0
    for depth, top1 in enumerate(top1_by_depth, start=1):
        if top1 >= threshold:
            horizon = depth
        else:
            break
    return horizon


def _config_hash(args: argparse.Namespace) -> str:
    payload = json.dumps(vars(args), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _print_summary(
    per_depth: list[dict[str, Any]],
    ridge_horizon: int,
    baseline_horizon: int,
) -> None:
    print("Decision-horizon per-depth summary")
    print(
        f"{'depth':>5} | {'r_top1':>7} {'r_top3':>7} {'r_tau':>7} {'r_reg':>8} | "
        f"{'b_top1':>7} {'b_top3':>7} {'b_tau':>7} {'b_reg':>8}"
    )
    for record in per_depth:
        ridge = record["ridge"]
        baseline = record["baseline"]
        print(
            f"{int(record['depth']):>5} | "
            f"{ridge['top1_agreement']:>7.3f} "
            f"{ridge['top3_recall']:>7.3f} "
            f"{ridge['kendall_tau']:>7.3f} "
            f"{ridge['selected_regret']:>8.4f} | "
            f"{baseline['top1_agreement']:>7.3f} "
            f"{baseline['top3_recall']:>7.3f} "
            f"{baseline['kendall_tau']:>7.3f} "
            f"{baseline['selected_regret']:>8.4f}"
        )
    print(f"\nridge decision-valid horizon: {ridge_horizon}")
    print(f"state-blind baseline horizon: {baseline_horizon}")


if __name__ == "__main__":
    main()
