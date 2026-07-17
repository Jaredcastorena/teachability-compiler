from __future__ import annotations

import math

import numpy as np
import pytest

from teachability_compiler.lm.experiments.compile_lm import (
    SwitchState,
    damped_worst_probe_choice,
    update_recency,
    weights_for_edu_heavy,
)
from teachability_compiler.lm.experiments.race_report import (
    compute_compression_ratios,
    summarize_run,
    summarize_runs,
    thresholds_for_target,
)


def test_edu_heavy_weights_quality_multipliers() -> None:
    action_names = [
        "semantic_animals_hi",
        "semantic_animals_lo",
        "code_heavy",
        "math_heavy",
        "general",
    ]
    manifest = {
        "actions": {
            name: {"train_tokens": 10}
            for name in action_names
        },
    }

    weights = weights_for_edu_heavy(manifest, action_names)

    assert math.isclose(sum(weights), 1.0)
    # Raw factors are 3, 1, 2, 2, 1 for equal token counts.
    assert weights[0] == pytest.approx(3 / 9)
    assert weights[1] == pytest.approx(1 / 9)
    assert weights[2] == pytest.approx(2 / 9)
    assert weights[3] == pytest.approx(2 / 9)
    assert weights[4] == pytest.approx(1 / 9)


def test_worst_probe_scoring_and_recency_updates() -> None:
    recency = np.zeros(3, dtype=np.float64)

    first = damped_worst_probe_choice([1.0, 2.0, 3.0], recency, penalty=0.5)
    recency = update_recency(recency, first, decay=0.7)
    assert first == 2
    assert recency.tolist() == pytest.approx([0.0, 0.0, 1.0])

    second = damped_worst_probe_choice([1.0, 2.8, 3.0], recency, penalty=0.5)
    recency = update_recency(recency, second, decay=0.7)
    assert second == 1
    assert recency.tolist() == pytest.approx([0.0, 1.0, 0.7])

    third = damped_worst_probe_choice([2.0, 3.0, 2.9], recency, penalty=0.5)
    recency = update_recency(recency, third, decay=0.7)
    assert third == 2
    assert recency.tolist() == pytest.approx([0.0, 0.7, 1.49])


def test_switch_state_fires_exactly_when_ema_below_threshold_after_min_chunks() -> None:
    switch = SwitchState(threshold=0.02, min_chunks=3, alpha=0.5)

    assert switch.update(0.10) is False
    assert switch.switched is False
    assert switch.update(0.04) is False
    assert switch.switched is False

    # EMA is 0.5 * (-0.04) + 0.5 * 0.07 = 0.015, below threshold at chunk 3.
    assert switch.update(-0.04) is True
    assert switch.switched is True
    assert switch.switch_chunk == 3
    assert switch.ema == pytest.approx(0.015)

    # Later calls do not re-fire.
    assert switch.update(1.0) is False
    assert switch.switch_chunk == 3


def test_race_report_threshold_crossing_and_compression_ratios() -> None:
    reference = {
        "kind": "reference_trajectory",
        "policy": "proportional_shuffle",
        "target": {"val_bpb": 1.0},
        "trajectory": [
            {"tokens": 100, "val_bpb": 1.20, "holdout_ce": 2.0},
            {"tokens": 200, "val_bpb": 1.05, "holdout_ce": 1.8},
            {"tokens": 300, "val_bpb": 1.00, "holdout_ce": 1.7},
        ],
        "overhead": {"probe_wall_seconds": 2.0},
    }
    race = {
        "kind": "lm_race",
        "policy": "uniform",
        "target": {"val_bpb": 1.0},
        "trajectory": [
            {"tokens": 50, "val_bpb": 1.30, "holdout_ce": 2.3},
            {"tokens": 100, "val_bpb": 1.08, "holdout_ce": 2.0},
            {"tokens": 250, "val_bpb": 1.01, "holdout_ce": 1.9},
        ],
        "overhead": {"probe_wall_seconds": 3.0},
    }

    # Regression for the denominator-corruption bug: a fixed-mixture baseline
    # produced by the reference DRIVER carries kind == "reference_trajectory"
    # but must never contribute to the reference denominators.
    uniform_as_ref_kind = {
        "kind": "reference_trajectory",
        "policy": "uniform_shuffle",
        "target": {"val_bpb": 1.0},
        "trajectory": [
            {"tokens": 10, "val_bpb": 1.10, "holdout_ce": 2.0},
            {"tokens": 20, "val_bpb": 1.00, "holdout_ce": 1.7},
        ],
        "overhead": {},
    }

    thresholds = thresholds_for_target(1.0)
    reference_summary = summarize_run("reference.json", reference, thresholds)
    reference_tokens = {
        label: reference_summary["tokens_to_threshold"].get(label) for label in thresholds
    }
    per_run = summarize_runs(
        [("uniform.json", race), ("uniform_shuffle.json", uniform_as_ref_kind)],
        thresholds,
    )
    compression_ratios = compute_compression_ratios(per_run, thresholds, reference_tokens)

    uniform = next(run for run in per_run if run["policy"] == "uniform")
    assert uniform["tokens_to_threshold"]["1.1x"] == 100
    assert uniform["tokens_to_threshold"]["1x"] is None
    assert uniform["final_val_bpb"] == pytest.approx(1.01)
    assert uniform["probe_overhead_seconds"] == pytest.approx(3.0)

    # Reference reaches 1.10x at 200 tokens; uniform reaches it at 100 tokens.
    # The uniform_shuffle run (which would drag the median denominator to 105
    # under kind-based inference) must have no effect on this ratio.
    assert compression_ratios["uniform"]["1.1x"] == pytest.approx(2.0)
    # Reference reaches 1.00x, but uniform misses it.
    assert compression_ratios["uniform"]["1x"] is None
    # The mislabeled baseline is reported as its own policy row, denominator
    # still from the true reference (200 / 10 = 20).
    assert compression_ratios["uniform_shuffle"]["1.1x"] == pytest.approx(20.0)
