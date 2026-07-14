"""Train and evaluate the residual-MLP transition predictor."""

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
from teachability_compiler.real.mlp_predictor import ResidualMLPTransitionPredictor
from teachability_compiler.real.persistence import load_transitions
from teachability_compiler.real.tasks import all_cluster_names
from teachability_compiler.state import TransitionObservation

_METRIC_KEYS = ("top1_agreement", "top3_recall", "kendall_tau", "selected_regret")


def main() -> None:
    """Train residual-MLP, ridge, and state-blind baselines on persisted transitions."""
    args = _parse_args()
    torch.manual_seed(args.seed)
    start_time = time.perf_counter()

    observations, metadata = load_transitions(args.transitions)
    if not observations:
        raise ValueError("no transitions loaded")
    probe_hash = _verify_probe_suite_hash(metadata)

    cluster_names = all_cluster_names()
    rng = np.random.default_rng(args.seed)

    groups = _group_by_state(observations)
    train_observations, holdout_observations, holdout_groups = _split_groups(
        groups=groups,
        holdout_fraction=args.holdout_fraction,
        rng=rng,
    )
    if not train_observations:
        raise ValueError("training split is empty; reduce --holdout-fraction")

    residual_mlp = ResidualMLPTransitionPredictor(
        action_names=cluster_names,
        device=args.device,
        seed=args.seed,
    ).fit(
        train_observations,
        epochs=args.epochs,
        ranking_weight=args.ranking_weight,
    )
    ridge = RidgeTransitionPredictor(action_names=cluster_names, ridge=1e-3).fit(
        train_observations
    )
    mean_delta = _fit_action_only_mean_delta(train_observations, cluster_names)

    mlp_mse = _one_step_mse(
        holdout_observations,
        lambda observation: residual_mlp.predict(
            observation.state_before,
            observation.action,
        ).probe_delta_mean,
    )
    ridge_mse = _one_step_mse(
        holdout_observations,
        lambda observation: ridge.predict(
            observation.state_before,
            observation.action,
        ).probe_delta_mean,
    )
    mean_mse = _one_step_mse(
        holdout_observations,
        lambda observation: mean_delta[observation.action.cluster_ids[0]],
    )

    mlp_decision: list[dict[str, float]] = []
    ridge_decision: list[dict[str, float]] = []
    mean_decision: list[dict[str, float]] = []
    for group in holdout_groups:
        if len(group) < 10:
            continue

        true_values = np.asarray(
            [-float(np.mean(observation.state_after.probe_losses)) for observation in group],
            dtype=np.float64,
        )
        mlp_values = np.asarray(
            [
                -float(
                    np.mean(
                        residual_mlp.predict(
                            observation.state_before,
                            observation.action,
                        ).next_state_mean.probe_losses
                    )
                )
                for observation in group
            ],
            dtype=np.float64,
        )
        ridge_values = np.asarray(
            [
                -float(
                    np.mean(
                        ridge.predict(
                            observation.state_before,
                            observation.action,
                        ).next_state_mean.probe_losses
                    )
                )
                for observation in group
            ],
            dtype=np.float64,
        )
        mean_values = np.asarray(
            [
                -float(
                    np.mean(
                        np.asarray(observation.state_before.probe_losses, dtype=np.float64)
                        + mean_delta[observation.action.cluster_ids[0]]
                    )
                )
                for observation in group
            ],
            dtype=np.float64,
        )

        mlp_decision.append(decision_metrics(true_values, mlp_values))
        ridge_decision.append(decision_metrics(true_values, ridge_values))
        mean_decision.append(decision_metrics(true_values, mean_values))

    git_commit = _git_commit()
    output = {
        "one_step_mse": {
            "residual_mlp": mlp_mse,
            "ridge": ridge_mse,
            "mean_baseline": mean_mse,
        },
        "decision_metrics": {
            "residual_mlp": _mean_metrics(mlp_decision),
            "ridge": _mean_metrics(ridge_decision),
            "mean_baseline": _mean_metrics(mean_decision),
        },
        "counts": {
            "train_transitions": len(train_observations),
            "holdout_transitions": len(holdout_observations),
            "train_groups": len(groups) - len(holdout_groups),
            "holdout_groups": len(holdout_groups),
            "decision_groups": len(mlp_decision),
        },
        "train_losses": residual_mlp.train_losses,
        "provenance": {
            "seed": args.seed,
            "config_hash": _config_hash(args),
            "git_commit": git_commit,
            "probe_suite_hash": probe_hash,
            "simulator_version": residual_mlp.version,
            "environment_version": "real-v1",
            "args": vars(args),
        },
        "overhead": {
            "wall_seconds": time.perf_counter() - start_time,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", nargs="+", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--ranking-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/residual_mlp_eval.json")
    args = parser.parse_args()

    if not 0.0 <= args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be in [0, 1)")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.ranking_weight < 0.0:
        raise ValueError("--ranking-weight must be non-negative")
    return args


def _verify_probe_suite_hash(metadata: list[dict[str, Any]]) -> str:
    hashes: set[str] = set()
    for item in metadata:
        if "probe_suite_hash" not in item:
            raise KeyError("transition metadata missing 'probe_suite_hash'")
        hashes.add(str(item["probe_suite_hash"]))

    if len(hashes) != 1:
        raise ValueError(f"probe_suite_hash mismatch across files: {sorted(hashes)}")
    return next(iter(hashes))


def _group_by_state(
    observations: list[TransitionObservation],
) -> list[list[TransitionObservation]]:
    groups: dict[bytes, list[TransitionObservation]] = {}
    for observation in observations:
        key = np.ascontiguousarray(
            np.asarray(observation.state_before.as_vector(), dtype=np.float64)
        ).tobytes()
        groups.setdefault(key, []).append(observation)
    return list(groups.values())


def _split_groups(
    groups: list[list[TransitionObservation]],
    holdout_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[TransitionObservation], list[TransitionObservation], list[list[TransitionObservation]]]:
    order = rng.permutation(len(groups))
    n_holdout = int(round(holdout_fraction * len(groups)))
    holdout_indices = {int(index) for index in order[:n_holdout]}

    train_observations: list[TransitionObservation] = []
    holdout_observations: list[TransitionObservation] = []
    holdout_groups: list[list[TransitionObservation]] = []

    for index, group in enumerate(groups):
        if index in holdout_indices:
            holdout_observations.extend(group)
            holdout_groups.append(group)
        else:
            train_observations.extend(group)

    return train_observations, holdout_observations, holdout_groups


def _fit_action_only_mean_delta(
    observations: list[TransitionObservation],
    cluster_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    deltas_by_action: dict[str, list[np.ndarray]] = {name: [] for name in cluster_names}
    all_deltas: list[np.ndarray] = []

    for observation in observations:
        if len(observation.action.cluster_ids) != 1:
            raise ValueError("mean baseline expects single-cluster actions")
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


def _one_step_mse(
    observations: list[TransitionObservation],
    predict_delta: Any,
) -> float:
    if not observations:
        return float("nan")

    squared_error = 0.0
    element_count = 0
    for observation in observations:
        true_delta = np.asarray(observation.probe_delta, dtype=np.float64)
        predicted_delta = np.asarray(predict_delta(observation), dtype=np.float64)
        if predicted_delta.shape != true_delta.shape:
            raise ValueError("predicted probe_delta shape mismatch")

        squared_error += float(np.sum((predicted_delta - true_delta) ** 2))
        element_count += int(true_delta.size)

    return squared_error / element_count


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {key: float("nan") for key in _METRIC_KEYS}
    return {key: float(np.mean([item[key] for item in metrics])) for key in _METRIC_KEYS}


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


def _print_summary(results: dict[str, Any]) -> None:
    print("Residual-MLP evaluation")
    print(f"{'model':>14} | {'mse':>10} | {'top1':>7} {'top3':>7} {'tau':>7} {'regret':>8}")
    for model in ("residual_mlp", "ridge", "mean_baseline"):
        mse = results["one_step_mse"][model]
        metrics = results["decision_metrics"][model]
        print(
            f"{model:>14} | {mse:>10.6f} | "
            f"{metrics['top1_agreement']:>7.3f} "
            f"{metrics['top3_recall']:>7.3f} "
            f"{metrics['kendall_tau']:>7.3f} "
            f"{metrics['selected_regret']:>8.4f}"
        )


if __name__ == "__main__":
    main()
