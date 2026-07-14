"""Fast CPU smoke test for the Phase 2 curriculum compiler experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from teachability_compiler.real import hidden_eval, tasks
from teachability_compiler.real.experiments import compile_curriculum

REQUIRED_KEYS = {
    "policy",
    "reached",
    "actions_executed",
    "race_tokens",
    "overhead_tokens",
    "total_tokens",
    "final_visible_mean",
    "final_hidden_mean",
    "final_visible_losses",
    "final_hidden_losses",
    "target_probe_losses",
    "epsilon",
    "forgetting_auc",
    "trajectory",
    "action_counts",
    "provenance",
    "overhead",
}

TINY_ARGS = [
    "--device",
    "cpu",
    "--seed",
    "0",
    "--steps",
    "2",
    "--seq-len",
    "32",
    "--batch-size",
    "8",
    "--d-model",
    "32",
    "--n-layers",
    "1",
]


def _to_numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_required_keys(data: dict[str, object]) -> None:
    missing = REQUIRED_KEYS.difference(data)
    assert not missing, f"missing required keys: {sorted(missing)}"


def test_compile_smoke(tmp_path: Path) -> None:
    reference_out = tmp_path / "reference.json"
    compile_curriculum.main(
        [
            "--policy",
            "reference",
            "--reference-actions",
            "3",
            "--out",
            str(reference_out),
            *TINY_ARGS,
        ]
    )

    assert reference_out.exists()
    reference_data = json.loads(reference_out.read_text())
    _assert_required_keys(reference_data)
    assert len(reference_data["target_probe_losses"]) == len(
        tasks.all_cluster_names()
    )

    random_out = tmp_path / "random.json"
    compile_curriculum.main(
        [
            "--policy",
            "random",
            "--max-actions",
            "3",
            "--target-file",
            str(reference_out),
            "--out",
            str(random_out),
            *TINY_ARGS,
        ]
    )

    assert random_out.exists()
    random_data = json.loads(random_out.read_text())
    _assert_required_keys(random_data)

    # The hidden channel must differ from the visible probe batches for at
    # least one cluster.
    hidden_differs = False
    for name in tasks.all_cluster_names():
        hidden_inputs, _ = hidden_eval.hidden_probe_batch(
            name, batch_size=8, seq_len=32
        )
        visible_inputs, _ = tasks.probe_batch(name, batch_size=8, seq_len=32)
        if not np.array_equal(_to_numpy(hidden_inputs), _to_numpy(visible_inputs)):
            hidden_differs = True
            break

    assert hidden_differs, "hidden probe batches should differ from visible batches"
