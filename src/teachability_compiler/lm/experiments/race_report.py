"""Post-hoc report for LM curriculum race runs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

THRESHOLD_FACTORS: tuple[float, ...] = (1.10, 1.05, 1.02, 1.01, 1.005, 1.0)


def threshold_labels() -> list[str]:
    return [f"{factor:g}x" for factor in THRESHOLD_FACTORS]


def thresholds_for_target(target_val_bpb: float) -> dict[str, float]:
    return {
        label: float(target_val_bpb) * factor
        for label, factor in zip(threshold_labels(), THRESHOLD_FACTORS, strict=True)
    }


def threshold_crossing_tokens(
    trajectory: list[dict[str, Any]],
    threshold: float,
) -> int | None:
    for point in trajectory:
        val_bpb = point.get("val_bpb")
        if val_bpb is not None and float(val_bpb) <= float(threshold):
            return int(point["tokens"])
    return None


def policy_for_run(data: dict[str, Any]) -> str:
    policy = data.get("policy")
    if policy:
        return str(policy)
    if data.get("kind") == "reference_trajectory":
        return "proportional_shuffle"
    if "target" in data and "trajectory" in data:
        return "proportional_shuffle"
    return "unknown"


def kind_for_run(data: dict[str, Any]) -> str:
    kind = data.get("kind")
    if kind:
        return str(kind)
    if "target" in data and "trajectory" in data:
        return "reference_trajectory"
    return "unknown"


def summarize_run(
    path: str,
    data: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    trajectory = list(data.get("trajectory", []))
    final = trajectory[-1] if trajectory else {}
    overhead = dict(data.get("overhead", {}))

    return {
        "path": path,
        "kind": kind_for_run(data),
        "policy": policy_for_run(data),
        "tokens_to_threshold": {
            label: threshold_crossing_tokens(trajectory, threshold)
            for label, threshold in thresholds.items()
        },
        "final_val_bpb": None if "val_bpb" not in final else float(final["val_bpb"]),
        "final_holdout_ce": None if "holdout_ce" not in final else float(final["holdout_ce"]),
        "probe_overhead_seconds": float(overhead.get("probe_wall_seconds", 0.0)),
    }


def summarize_runs(
    runs: list[tuple[str, dict[str, Any]]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    return [summarize_run(path, data, thresholds) for path, data in runs]


def _median_or_none(values: list[int | float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(statistics.median(numeric))


def compute_compression_ratios(
    per_run: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    reference_runs = [
        run
        for run in per_run
        if run["policy"] == "proportional_shuffle" or run["kind"] == "reference_trajectory"
    ]
    reference_tokens = {
        label: _median_or_none(
            [run["tokens_to_threshold"].get(label) for run in reference_runs],
        )
        for label in thresholds
    }

    policies = sorted({str(run["policy"]) for run in per_run})
    compression_ratios: dict[str, dict[str, float | None]] = {}
    for policy in policies:
        policy_runs = [run for run in per_run if run["policy"] == policy]
        compression_ratios[policy] = {}
        for label in thresholds:
            ref_tokens = reference_tokens[label]
            policy_tokens = _median_or_none(
                [run["tokens_to_threshold"].get(label) for run in policy_runs],
            )
            if ref_tokens is None or policy_tokens is None or policy_tokens <= 0:
                compression_ratios[policy][label] = None
            else:
                compression_ratios[policy][label] = float(ref_tokens / policy_tokens)
    return compression_ratios


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _target_val_bpb_from_runs(runs: list[tuple[str, dict[str, Any]]]) -> float:
    for _, data in runs:
        target = data.get("target")
        if isinstance(target, dict) and "val_bpb" in target:
            return float(target["val_bpb"])

    reference_like = [
        data for _, data in runs if policy_for_run(data) == "proportional_shuffle"
    ]
    for data in reference_like:
        trajectory = list(data.get("trajectory", []))
        if trajectory and "val_bpb" in trajectory[-1]:
            return float(trajectory[-1]["val_bpb"])

    raise ValueError("could not infer target val_bpb from any run")


def print_table(
    compression_ratios: dict[str, dict[str, float | None]],
    thresholds: dict[str, float],
) -> None:
    labels = list(thresholds)
    policy_width = max([len("policy"), *(len(policy) for policy in compression_ratios)])
    header = " ".join(["policy".ljust(policy_width), *(label.rjust(8) for label in labels)])
    print(header)
    print("-" * len(header))
    for policy in sorted(compression_ratios):
        cells = []
        for label in labels:
            value = compression_ratios[policy].get(label)
            cells.append("-".rjust(8) if value is None else f"{value:8.3f}")
        print(" ".join([policy.ljust(policy_width), *cells]))


def build_report(run_paths: list[Path]) -> dict[str, Any]:
    runs = [(str(path), _load_json(path)) for path in run_paths]
    target_val_bpb = _target_val_bpb_from_runs(runs)
    thresholds = thresholds_for_target(target_val_bpb)
    per_run = summarize_runs(runs, thresholds)
    compression_ratios = compute_compression_ratios(per_run, thresholds)
    return {
        "thresholds": thresholds,
        "per_run": per_run,
        "compression_ratios": compression_ratios,
        "provenance": {"git_commit": _git_commit()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="results/lm_race_report.json")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report([Path(path) for path in args.runs])
    print_table(report["compression_ratios"], report["thresholds"])
    _atomic_write_json(Path(args.out), report)


if __name__ == "__main__":
    main()
