from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from teachability_compiler.lm.lm_oracle import NanochatLearnerOracle
from teachability_compiler.lm.lowrank_simulator import LowRankTransitionPredictor
from teachability_compiler.state import CurriculumAction, LearningState, TransitionObservation


def _write_tokens(path: Path, rng: np.random.Generator, count: int) -> None:
    data = rng.integers(0, 32768, size=count, dtype=np.uint16)
    data.tofile(path)


def _synthetic_tokens_dir(tmp_path: Path) -> tuple[Path, list[str]]:
    rng = np.random.default_rng(123)
    action_names = [f"action_{i:02d}" for i in range(24)]

    for action in action_names:
        _write_tokens(tmp_path / f"train_{action}.bin", rng, 50_000)
        _write_tokens(tmp_path / f"probe_{action}.bin", rng, 20_000)
    _write_tokens(tmp_path / "val.bin", rng, 20_000)
    _write_tokens(tmp_path / "holdout.bin", rng, 20_000)

    manifest = {
        "action_names": action_names,
        "actions": {
            action: {
                "train_docs": 10,
                "train_tokens": 50_000,
                "probe_docs": 4,
                "probe_tokens": 20_000,
            }
            for action in action_names
        },
        "val": {"docs": 4, "tokens": 20_000},
        "holdout": {"docs": 4, "tokens": 20_000},
        "seq_len": 128,
        "tokenizer_sha256_16": "fake-tokenizer",
        "actions_manifest_hash": "fake-actions",
        "seed": 0,
        "git_commit": "fake",
        "created_at": "2025-01-01T00:00:00Z",
        "manifest_hash": "fake-tokens",
    }
    with (tmp_path / "tokens_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
        f.write("\n")
    return tmp_path, action_names


def _write_fake_token_bytes(home: Path) -> None:
    token_dir = home / ".cache" / "nanochat" / "tokenizer"
    token_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(32768, dtype=torch.float32), token_dir / "token_bytes.pt")


def _state(
    probe_losses: np.ndarray,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    exposure: np.ndarray | None = None,
    history: np.ndarray | None = None,
) -> LearningState:
    probe_losses = np.asarray(probe_losses, dtype=np.float64)
    exposure_histogram = (
        np.zeros(24, dtype=np.float64)
        if exposure is None
        else np.asarray(exposure, dtype=np.float64)
    )
    history_embedding = (
        np.zeros(4, dtype=np.float64)
        if history is None
        else np.asarray(history, dtype=np.float64)
    )
    return LearningState(
        probe_losses=probe_losses,
        update_sketch=np.zeros(4, dtype=np.float64),
        optimizer_sketch=np.zeros(2, dtype=np.float64),
        activation_sketch=np.asarray(
            [
                float(np.mean(probe_losses)),
                float(np.max(probe_losses)),
                float(np.min(probe_losses)),
                float(np.std(probe_losses)),
            ],
            dtype=np.float64,
        ),
        exposure_histogram=exposure_histogram,
        history_embedding=history_embedding,
        architecture_embedding=np.asarray([2 / 32, 128 / 2048], dtype=np.float64),
        step=step,
        tokens_seen=tokens_seen,
    )


def _make_observation(
    state_before: LearningState,
    action: CurriculumAction,
    probe_delta: np.ndarray,
) -> TransitionObservation:
    probe_delta = np.asarray(probe_delta, dtype=np.float64)
    state_after = _state(
        np.clip(state_before.probe_losses + probe_delta, 0.0, 20.0),
        step=state_before.step + 1,
        tokens_seen=state_before.tokens_seen + action.token_budget,
        exposure=state_before.exposure_histogram,
        history=state_before.history_embedding,
    )
    values: dict[str, Any] = {
        "state_before": state_before,
        "before_state": state_before,
        "previous_state": state_before,
        "action": action,
        "curriculum_action": action,
        "state_after": state_after,
        "after_state": state_after,
        "next_state": state_after,
        "probe_delta": probe_delta,
        "probe_loss_delta": probe_delta,
        "parameter_delta_sketch": np.zeros(8, dtype=np.float64),
        "param_delta_sketch": np.zeros(8, dtype=np.float64),
        "activation_delta_sketch": state_after.activation_sketch - state_before.activation_sketch,
        "compute_cost": float(action.token_budget),
        "expected_compute": float(action.token_budget),
        "seed_metadata": {"seed": 0},
        "metadata": {"seed": 0},
    }

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for field in dataclasses.fields(TransitionObservation):
        if not field.init:
            continue
        if field.name in values:
            kwargs[field.name] = values[field.name]
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            missing.append(field.name)
    if missing:
        raise TypeError(f"Test helper cannot construct TransitionObservation fields {missing}")
    return TransitionObservation(**kwargs)


def test_lowrank_fit_predict_mse_decreases() -> None:
    rng = np.random.default_rng(5)
    action_names = [f"action_{i:02d}" for i in range(24)]
    base_states = []
    for i in range(3):
        probes = 3.0 + 0.2 * i + rng.normal(0.0, 0.05, size=24)
        history = np.asarray([0.0, 0.1 * i, 0.2 * i, 0.0], dtype=np.float64)
        base_states.append(_state(probes, step=i, tokens_seen=10_000 * i, history=history))

    direction_0 = rng.normal(0.0, 0.03, size=24)
    direction_1 = rng.normal(0.0, 0.03, size=24)
    observations: list[TransitionObservation] = []

    for cycle in range(3):
        for action_idx, action_name in enumerate(action_names):
            state_before = base_states[cycle]
            action = CurriculumAction(
                cluster_ids=(action_name,),
                mixture_weights=(1.0,),
                optimizer_steps=1,
                token_budget=256,
            )
            action_bias = np.zeros(24, dtype=np.float64)
            action_bias[action_idx] = -0.08
            coeff_0 = (action_idx / 23.0) - 0.5
            coeff_1 = float(cycle) - 1.0
            probe_delta = action_bias + coeff_0 * direction_0 + coeff_1 * direction_1
            probe_delta += rng.normal(0.0, 0.003, size=24)
            observations.append(_make_observation(state_before, action, probe_delta))

    predictor = LowRankTransitionPredictor(action_names, rank=4, seed=7)
    predictor.fit(observations, epochs=80, lr=2e-3, batch_size=32, ranking_weight=0.05)

    assert predictor.final_losses["mse"] < predictor.final_losses["initial_mse"]
    pred = predictor.predict(
        base_states[0],
        CurriculumAction(
            cluster_ids=(action_names[0],),
            mixture_weights=(1.0,),
            optimizer_steps=1,
            token_budget=512,
        ),
    )

    assert pred.probe_delta_mean.shape == (24,)
    assert pred.probe_delta_std.shape == (24,)
    assert pred.next_state_mean.probe_losses.shape == (24,)
    assert predictor.directions.shape == (4, 24)
    assert predictor.mu.shape == (24, 24)
    assert np.all(np.isfinite(pred.probe_delta_mean))
    assert np.isfinite(pred.forgetting_risk)
    assert pred.expected_compute == 512


def test_lowrank_predict_requires_fit() -> None:
    action_names = [f"action_{i:02d}" for i in range(24)]
    predictor = LowRankTransitionPredictor(action_names, rank=3, seed=1)
    with pytest.raises(RuntimeError):
        predictor.predict(
            _state(np.full(24, 3.0)),
            CurriculumAction(
                cluster_ids=(action_names[0],),
                mixture_weights=(1.0,),
                optimizer_steps=1,
                token_budget=64,
            ),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="nanochat oracle smoke test requires CUDA",
)
def test_oracle_gpu_apply_action_snapshot_restore_and_hidden_bpb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens_dir, action_names = _synthetic_tokens_dir(tmp_path)
    fake_home = tmp_path / "home"
    _write_fake_token_bytes(fake_home)
    monkeypatch.setenv("HOME", str(fake_home))

    oracle = NanochatLearnerOracle(
        tokens_dir,
        depth=2,
        device="cuda:0",
        seq_len=128,
        device_batch=2,
        grad_accum=1,
        base_seed=11,
        probe_batches=1,
    )
    action = CurriculumAction(
        cluster_ids=(action_names[0],),
        mixture_weights=(1.0,),
        optimizer_steps=1,
        token_budget=256,
    )

    obs = oracle.apply_action(action, data_seed=17)
    assert obs.compute_cost > 0
    assert obs.probe_delta.shape == (24,)
    assert np.all(np.isfinite(obs.probe_delta))

    snap = oracle.snapshot()
    losses_before = oracle.probe_losses()

    oracle.apply_action(
        CurriculumAction(
            cluster_ids=(action_names[1],),
            mixture_weights=(1.0,),
            optimizer_steps=1,
            token_budget=256,
        ),
        data_seed=19,
    )
    oracle.restore(snap)
    losses_after = oracle.probe_losses()

    assert np.allclose(losses_before, losses_after, atol=1e-5)
    bpb = oracle.hidden_val_bpb(max_batches=1)
    assert np.isfinite(bpb)
    assert bpb > 0.0
