"""Measure the pairwise functional-commutator tail of the real learner."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from teachability_compiler.metrics import weighted_norm
from teachability_compiler.real.model import DecoderConfig
from teachability_compiler.real.oracle import RealLearnerOracle
from teachability_compiler.real.persistence import probe_suite_hash, save_transitions
from teachability_compiler.real.tasks import VOCAB_SIZE, all_cluster_names
from teachability_compiler.state import CurriculumAction, TransitionObservation


def main() -> None:
    """Run the commutator-tail measurement."""
    args = _parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_time = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    cluster_names = all_cluster_names()
    all_pairs = list(itertools.combinations(cluster_names, 2))
    if args.pairs < 0 or args.pairs > len(all_pairs):
        raise ValueError(f"--pairs must be in [0, {len(all_pairs)}]")

    if args.pairs == len(all_pairs):
        sampled_pairs = all_pairs
    else:
        indices = rng.choice(len(all_pairs), size=args.pairs, replace=False)
        sampled_pairs = [all_pairs[int(index)] for index in indices]

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

    noise_values: list[float] = []
    for name in rng.choice(cluster_names, size=10, replace=True):
        action = _single_action(str(name), args.steps, oracle)
        seed_1 = _next_seed(rng)
        seed_2 = _next_seed(rng)

        oracle.restore(checkpoint)
        observation_1 = oracle.apply_action(action, data_seed=seed_1)
        outcome_1 = observation_1.state_after.probe_losses
        transitions.append(observation_1)
        overhead_steps += args.steps
        overhead_tokens += args.steps * oracle.batch_size * oracle.seq_len

        oracle.restore(checkpoint)
        observation_2 = oracle.apply_action(action, data_seed=seed_2)
        outcome_2 = observation_2.state_after.probe_losses
        transitions.append(observation_2)
        overhead_steps += args.steps
        overhead_tokens += args.steps * oracle.batch_size * oracle.seq_len

        noise_values.append(weighted_norm(outcome_1 - outcome_2))

    noise_mean = float(np.mean(noise_values)) if noise_values else 0.0
    noise_p95 = float(np.percentile(noise_values, 95)) if noise_values else 0.0

    pair_records: list[dict[str, Any]] = []
    for a_name, b_name in sampled_pairs:
        commutators: list[float] = []
        advantages: list[float] = []
        action_a = _single_action(a_name, args.steps, oracle)
        action_b = _single_action(b_name, args.steps, oracle)

        for _ in range(args.repeats):
            seed_ab_1 = _next_seed(rng)
            seed_ab_2 = _next_seed(rng)
            seed_ba_1 = _next_seed(rng)
            seed_ba_2 = _next_seed(rng)

            oracle.restore(checkpoint)
            obs_a_first = oracle.apply_action(action_a, data_seed=seed_ab_1)
            obs_ab = oracle.apply_action(action_b, data_seed=seed_ab_2)
            transitions.append(obs_a_first)
            transitions.append(obs_ab)
            state_ab = obs_ab.state_after
            overhead_steps += 2 * args.steps
            overhead_tokens += 2 * args.steps * oracle.batch_size * oracle.seq_len

            oracle.restore(checkpoint)
            obs_b_first = oracle.apply_action(action_b, data_seed=seed_ba_1)
            obs_ba = oracle.apply_action(action_a, data_seed=seed_ba_2)
            transitions.append(obs_b_first)
            transitions.append(obs_ba)
            state_ba = obs_ba.state_after
            overhead_steps += 2 * args.steps
            overhead_tokens += 2 * args.steps * oracle.batch_size * oracle.seq_len

            commutators.append(weighted_norm(state_ab.probe_losses - state_ba.probe_losses))
            advantages.append(oracle.value(state_ab) - oracle.value(state_ba))

            oracle.restore(checkpoint)

        pair_records.append(
            {
                "a": a_name,
                "b": b_name,
                "commutator": commutators,
                "mean": float(np.mean(commutators)) if commutators else 0.0,
                "directed_advantage": advantages,
                "directed_advantage_mean": float(np.mean(advantages)) if advantages else 0.0,
            }
        )

    pair_means = np.asarray([record["mean"] for record in pair_records], dtype=np.float64)
    tail_statistics = _tail_statistics(pair_means)
    repeatability = _repeatability(pair_records)
    count_above_noise = int(np.sum(pair_means > noise_p95)) if pair_means.size else 0

    wall_seconds = time.perf_counter() - start_time
    output = {
        "pairs": pair_records,
        "noise_floor": {
            "values": noise_values,
            "mean": noise_mean,
            "p95": noise_p95,
        },
        "tail_statistics": tail_statistics,
        "repeatability": {
            "pearson_r_repeat1_repeat2": repeatability,
        },
        "count_pairs_above_noise_p95": count_above_noise,
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
    _print_summary(pair_records, tail_statistics, noise_mean, noise_p95, repeatability)

    if args.save_transitions is not None:
        save_transitions(
            args.save_transitions,
            transitions,
            {
                "experiment": "commutator_tail",
                "pretrain_steps": args.pretrain_steps,
                "seed": args.seed,
                "probe_suite_hash": probe_suite_hash(),
                "git_commit": _git_commit(),
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/commutator_tail.json")
    parser.add_argument("--save-transitions", type=str, default=None)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.pretrain_steps < 0:
        raise ValueError("--pretrain-steps must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    return args


def _single_action(name: str, steps: int, oracle: RealLearnerOracle) -> CurriculumAction:
    return CurriculumAction(
        cluster_ids=(name,),
        mixture_weights=(1.0,),
        optimizer_steps=steps,
        token_budget=steps * oracle.batch_size * oracle.seq_len,
    )


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _tail_statistics(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "excess_kurtosis": 0.0,
            "top_decile_share": 0.0,
        }

    centered = values - float(np.mean(values))
    variance = float(np.mean(centered * centered))
    excess_kurtosis = (
        float(np.mean(centered**4) / (variance * variance) - 3.0) if variance > 0.0 else 0.0
    )
    top_count = max(1, int(math.ceil(0.1 * values.size)))
    sorted_values = np.sort(values)
    total_mass = float(np.sum(values))
    top_decile_share = (
        float(np.sum(sorted_values[-top_count:]) / total_mass) if total_mass > 0.0 else 0.0
    )
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "excess_kurtosis": excess_kurtosis,
        "top_decile_share": top_decile_share,
    }


def _repeatability(pair_records: list[dict[str, Any]]) -> float | None:
    first: list[float] = []
    second: list[float] = []
    for record in pair_records:
        commutators = record["commutator"]
        if len(commutators) >= 2:
            first.append(float(commutators[0]))
            second.append(float(commutators[1]))

    if len(first) < 2:
        return None
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if float(np.std(first_array)) == 0.0 or float(np.std(second_array)) == 0.0:
        return None
    return float(np.corrcoef(first_array, second_array)[0, 1])


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
    pair_records: list[dict[str, Any]],
    tail_statistics: dict[str, float],
    noise_mean: float,
    noise_p95: float,
    repeatability: float | None,
) -> None:
    top_records = sorted(pair_records, key=lambda record: float(record["mean"]), reverse=True)[:10]
    print("Top-10 commutator pairs")
    print(f"{'rank':>4}  {'a':<24} {'b':<24} {'mean':>10} {'adv_mean':>10}")
    for rank, record in enumerate(top_records, start=1):
        print(
            f"{rank:>4}  {record['a']:<24} {record['b']:<24} "
            f"{record['mean']:>10.6f} {record['directed_advantage_mean']:>10.6f}"
        )

    print("\nTail statistics")
    for key, value in tail_statistics.items():
        print(f"{key:<20} {value:.6f}")
    print(f"noise_mean          {noise_mean:.6f}")
    print(f"noise_p95           {noise_p95:.6f}")
    repeatability_text = "n/a" if repeatability is None else f"{repeatability:.6f}"
    print(f"repeatability_r     {repeatability_text}")


if __name__ == "__main__":
    main()
