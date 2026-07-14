"""Measure the open-loop linearization horizon of a learned transition simulator."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from teachability_compiler.predictor import RidgeTransitionPredictor
from teachability_compiler.real.model import DecoderConfig
from teachability_compiler.real.oracle import RealLearnerOracle
from teachability_compiler.real.persistence import probe_suite_hash, save_transitions
from teachability_compiler.real.tasks import VOCAB_SIZE, all_cluster_names
from teachability_compiler.state import CurriculumAction, TransitionObservation


def main() -> None:
    """Run the linearization-horizon measurement."""
    args = _parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_time = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    cluster_names = all_cluster_names()

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

    transitions: list[TransitionObservation] = []

    observations: list[TransitionObservation] = []
    for _ in range(args.train_rollouts):
        oracle.restore(checkpoint)
        sequence = _random_action_sequence(
            rng, cluster_names, args.rollout_horizon, args.steps, oracle
        )
        for action in sequence:
            observation = oracle.apply_action(action, data_seed=_next_seed(rng))
            observations.append(observation)
            transitions.append(observation)
            overhead_steps += args.steps
            overhead_tokens += args.steps * oracle.batch_size * oracle.seq_len

    predictor = RidgeTransitionPredictor(action_names=cluster_names, ridge=1e-3).fit(observations)
    action_only_deltas = _fit_action_only_mean_delta(observations, cluster_names)

    errors: list[list[float]] = [[] for _ in range(args.max_horizon)]
    baseline_errors: list[list[float]] = [[] for _ in range(args.max_horizon)]
    true_delta_norms: list[list[float]] = [[] for _ in range(args.max_horizon)]
    relative_errors: list[list[float]] = [[] for _ in range(args.max_horizon)]
    baseline_relative_errors: list[list[float]] = [[] for _ in range(args.max_horizon)]

    for _ in range(args.eval_rollouts):
        actions = _random_action_sequence(rng, cluster_names, args.max_horizon, args.steps, oracle)

        oracle.restore(checkpoint)
        initial_state = oracle.encode_state()
        start_probe = initial_state.probe_losses.copy()

        predicted_state = initial_state
        predicted_probes: list[np.ndarray] = []
        for action in actions:
            prediction = predictor.predict(predicted_state, action)
            predicted_state = prediction.next_state_mean
            predicted_probes.append(np.asarray(predicted_state.probe_losses, dtype=np.float64))

        baseline_probe = start_probe.copy()
        baseline_probes: list[np.ndarray] = []
        for action in actions:
            baseline_probe = baseline_probe + action_only_deltas[action.cluster_ids[0]]
            baseline_probes.append(baseline_probe.copy())

        oracle.restore(checkpoint)
        true_probes: list[np.ndarray] = []
        for action in actions:
            observation = oracle.apply_action(action, data_seed=_next_seed(rng))
            true_probes.append(observation.state_after.probe_losses.copy())
            transitions.append(observation)
            overhead_steps += args.steps
            overhead_tokens += args.steps * oracle.batch_size * oracle.seq_len

        for depth in range(args.max_horizon):
            true_probe = true_probes[depth]
            delta_norm = float(np.linalg.norm(true_probe - start_probe))
            model_error = float(np.linalg.norm(predicted_probes[depth] - true_probe))
            baseline_error = float(np.linalg.norm(baseline_probes[depth] - true_probe))

            errors[depth].append(model_error)
            baseline_errors[depth].append(baseline_error)
            true_delta_norms[depth].append(delta_norm)
            relative_errors[depth].append(_safe_relative(model_error, delta_norm))
            baseline_relative_errors[depth].append(_safe_relative(baseline_error, delta_norm))

    noise_medians, overhead_steps, overhead_tokens = _measure_noise_floor(
        oracle=oracle,
        checkpoint=checkpoint,
        rng=rng,
        cluster_names=cluster_names,
        max_horizon=args.max_horizon,
        steps=args.steps,
        overhead_steps=overhead_steps,
        overhead_tokens=overhead_tokens,
        transitions=transitions,
    )

    per_depth, horizon = _summarize_depths(
        errors=errors,
        baseline_errors=baseline_errors,
        true_delta_norms=true_delta_norms,
        relative_errors=relative_errors,
        baseline_relative_errors=baseline_relative_errors,
        noise_medians=noise_medians,
    )
    baseline_horizon = _horizon_from_records(per_depth, error_key="baseline_median_error")

    wall_seconds = time.perf_counter() - start_time
    output = {
        "per_depth": per_depth,
        "linearization_horizon": horizon,
        "state_blind_baseline_horizon": baseline_horizon,
        "train_transition_count": len(observations),
        "provenance": {
            "seed": args.seed,
            "config_hash": _config_hash(args),
            "git_commit": _git_commit(),
            "target_checkpoint": f"pretrain-{args.pretrain_steps}steps-seed{args.seed}",
            "simulator_version": "none:measurement-only",
            "environment_version": "real-v1",
            "probe_suite_hash": probe_suite_hash(),
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
    _print_summary(per_depth, horizon, baseline_horizon)

    if args.save_transitions is not None:
        save_transitions(
            args.save_transitions,
            transitions,
            {
                "experiment": "linearization_horizon",
                "pretrain_steps": args.pretrain_steps,
                "seed": args.seed,
                "probe_suite_hash": probe_suite_hash(),
                "git_commit": _git_commit(),
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rollouts", type=int, default=40)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--eval-rollouts", type=int, default=8)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/linearization_horizon.json")
    parser.add_argument("--save-transitions", type=str, default=None)
    args = parser.parse_args()

    if args.train_rollouts <= 0:
        raise ValueError("--train-rollouts must be positive")
    if args.rollout_horizon <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if args.eval_rollouts <= 0:
        raise ValueError("--eval-rollouts must be positive")
    if args.max_horizon <= 0:
        raise ValueError("--max-horizon must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.pretrain_steps < 0:
        raise ValueError("--pretrain-steps must be non-negative")
    return args


def _random_action_sequence(
    rng: np.random.Generator,
    cluster_names: tuple[str, ...],
    length: int,
    steps: int,
    oracle: RealLearnerOracle,
) -> list[CurriculumAction]:
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


def _measure_noise_floor(
    oracle: RealLearnerOracle,
    checkpoint: dict[str, Any],
    rng: np.random.Generator,
    cluster_names: tuple[str, ...],
    max_horizon: int,
    steps: int,
    overhead_steps: int,
    overhead_tokens: int,
    transitions: list[TransitionObservation],
) -> tuple[list[float], int, int]:
    actions = _random_action_sequence(rng, cluster_names, max_horizon, steps, oracle)
    duplicate_rollouts: list[list[np.ndarray]] = []

    for _ in range(5):
        oracle.restore(checkpoint)
        probes_at_depth: list[np.ndarray] = []
        for action in actions:
            observation = oracle.apply_action(action, data_seed=_next_seed(rng))
            probes_at_depth.append(observation.state_after.probe_losses.copy())
            transitions.append(observation)
            overhead_steps += steps
            overhead_tokens += steps * oracle.batch_size * oracle.seq_len
        duplicate_rollouts.append(probes_at_depth)

    noise_medians: list[float] = []
    for depth in range(max_horizon):
        distances = [
            float(np.linalg.norm(duplicate_rollouts[i][depth] - duplicate_rollouts[j][depth]))
            for i, j in itertools.combinations(range(len(duplicate_rollouts)), 2)
        ]
        noise_medians.append(float(np.median(distances)) if distances else 0.0)

    return noise_medians, overhead_steps, overhead_tokens


def _summarize_depths(
    errors: list[list[float]],
    baseline_errors: list[list[float]],
    true_delta_norms: list[list[float]],
    relative_errors: list[list[float]],
    baseline_relative_errors: list[list[float]],
    noise_medians: list[float],
) -> tuple[list[dict[str, float]], int]:
    records: list[dict[str, float]] = []
    horizon = 0

    for index, depth_errors in enumerate(errors):
        depth = index + 1
        median_error = float(np.median(depth_errors))
        p90_error = float(np.percentile(depth_errors, 90))
        baseline_median_error = float(np.median(baseline_errors[index]))
        baseline_p90_error = float(np.percentile(baseline_errors[index], 90))
        median_delta = float(np.median(true_delta_norms[index]))
        threshold = max(2.0 * noise_medians[index], 0.1 * median_delta)

        if median_error <= threshold:
            horizon = depth

        records.append(
            {
                "depth": float(depth),
                "median_error": median_error,
                "p90_error": p90_error,
                "noise_floor_median": float(noise_medians[index]),
                "median_true_delta_from_start": median_delta,
                "relative_error_median": float(np.median(relative_errors[index])),
                "threshold": threshold,
                "baseline_median_error": baseline_median_error,
                "baseline_p90_error": baseline_p90_error,
                "baseline_relative_error_median": float(
                    np.median(baseline_relative_errors[index])
                ),
            }
        )

    return records, horizon


def _horizon_from_records(records: list[dict[str, float]], error_key: str) -> int:
    horizon = 0
    for record in records:
        if record[error_key] <= record["threshold"]:
            horizon = int(record["depth"])
    return horizon


def _safe_relative(error: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0 if error <= 1e-12 else error / 1e-12
    return error / denominator


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
    per_depth: list[dict[str, float]],
    horizon: int,
    baseline_horizon: int,
) -> None:
    print("Linearization horizon per-depth summary")
    print(
        f"{'k':>3} {'med_err':>10} {'p90_err':>10} {'noise':>10} "
        f"{'rel':>10} {'base_med':>10} {'base_rel':>10} {'threshold':>10}"
    )
    for record in per_depth:
        print(
            f"{int(record['depth']):>3} "
            f"{record['median_error']:>10.6f} "
            f"{record['p90_error']:>10.6f} "
            f"{record['noise_floor_median']:>10.6f} "
            f"{record['relative_error_median']:>10.6f} "
            f"{record['baseline_median_error']:>10.6f} "
            f"{record['baseline_relative_error_median']:>10.6f} "
            f"{record['threshold']:>10.6f}"
        )
    print(f"\nstate-conditioned horizon: {horizon}")
    print(f"state-blind baseline horizon: {baseline_horizon}")


if __name__ == "__main__":
    main()
