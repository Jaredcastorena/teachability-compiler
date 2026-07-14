"""CPU smoke tests for the real learner stack."""

from __future__ import annotations

import numpy as np
import torch

from teachability_compiler.real import (
    VOCAB_SIZE,
    DecoderConfig,
    RealLearnerOracle,
    all_cluster_names,
)
from teachability_compiler.state import CurriculumAction, TransitionObservation


def test_real_oracle_cpu_smoke() -> None:
    """Build a tiny CPU oracle, train briefly, measure a transition, and restore."""
    torch.manual_seed(123)

    cluster_names = all_cluster_names()
    config = DecoderConfig(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )
    oracle = RealLearnerOracle(
        config=config,
        cluster_names=cluster_names,
        device="cpu",
        base_seed=123,
        steps_per_action=1,
        batch_size=8,
        seq_len=32,
        lr=3e-3,
    )

    oracle.pretrain(n_steps=2, rng_seed=124)
    state = oracle.encode_state()

    assert state.probe_losses.shape == (len(cluster_names),)
    assert state.update_sketch.shape == (4,)
    assert state.optimizer_sketch.shape == (2,)
    assert state.activation_sketch.shape == (4,)
    assert state.exposure_histogram.shape == (len(cluster_names),)
    assert state.history_embedding.shape == (4,)
    assert state.architecture_embedding.shape == (2,)
    assert state.as_vector().ndim == 1
    assert np.all(np.isfinite(state.probe_losses))

    snapshot = oracle.snapshot()
    losses_before = oracle.probe_losses()

    action = CurriculumAction(
        cluster_ids=("copy_short",),
        mixture_weights=(1.0,),
        optimizer_steps=1,
        token_budget=oracle.batch_size * oracle.seq_len,
    )
    observation = oracle.apply_action(action, data_seed=125)

    assert isinstance(observation, TransitionObservation)
    assert observation.compute_cost > 0.0
    assert observation.state_after.probe_losses.shape == (len(cluster_names),)
    assert observation.parameter_delta_sketch.shape == (8,)
    assert observation.activation_delta_sketch.shape == (4,)
    assert np.all(np.isfinite(observation.state_after.probe_losses))
    assert not np.allclose(
        observation.state_before.probe_losses, observation.state_after.probe_losses
    )

    oracle.restore(snapshot)
    losses_restored = oracle.probe_losses()
    np.testing.assert_allclose(losses_restored, losses_before, rtol=1e-6, atol=1e-6)
