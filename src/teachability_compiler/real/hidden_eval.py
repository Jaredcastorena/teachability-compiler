"""Hidden evaluation channel for real teachability-compiler experiments.

The hidden channel uses the same public task distributions as the visible probe
suite, but draws batches from a separate deterministic seed stream. It is meant
only for recording held-out measurements and must not be used for planning,
simulator fitting, or termination decisions.

Importing this module must not change ``tasks.probe_batch`` behaviour, and
``persistence.probe_suite_hash()`` must remain byte-for-byte identical.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from teachability_compiler.real import tasks

# Deterministic per-key cache: batches are a pure function of the key.
_HIDDEN_BATCH_CACHE: dict[tuple[str, int, int], tuple[Any, Any]] = {}


def _hidden_seed(name: str) -> int:
    """Derive a 64-bit seed on a stream disjoint from the visible suite."""
    digest = hashlib.sha256(("hidden::" + name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def hidden_probe_batch(
    name: str,
    batch_size: int = 64,
    seq_len: int = 64,
) -> tuple[Any, Any]:
    """Return a deterministic held-out ``(inputs, targets)`` batch for ``name``.

    Same task distribution as ``tasks.probe_batch`` but drawn from a dedicated
    hidden seed stream via ``tasks.sample_batch`` (no private imports).
    """
    key = (name, batch_size, seq_len)
    if key not in _HIDDEN_BATCH_CACHE:
        rng = np.random.default_rng(_hidden_seed(name))
        _HIDDEN_BATCH_CACHE[key] = tasks.sample_batch(name, batch_size, seq_len, rng)
    return _HIDDEN_BATCH_CACHE[key]


def _as_long_tensor(value: Any, device: torch.device | str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.asarray(value), device=device, dtype=torch.long)


def hidden_losses(oracle: Any) -> np.ndarray:
    """Mean cross-entropy of ``oracle.model`` on each cluster's hidden batch.

    Evaluated in eval mode under ``no_grad`` with ``ignore_index=-100`` and
    returned in canonical ``tasks.all_cluster_names()`` order. Fails loudly on a
    non-finite loss. The model's training mode is restored on exit.
    """
    model = oracle.model
    device = oracle.device
    batch_size = oracle.batch_size
    seq_len = oracle.seq_len

    cluster_names = tasks.all_cluster_names()
    losses = np.empty(len(cluster_names), dtype=np.float64)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for index, name in enumerate(cluster_names):
                inputs, targets = hidden_probe_batch(name, batch_size, seq_len)
                input_tensor = _as_long_tensor(inputs, device)
                target_tensor = _as_long_tensor(targets, device)

                logits = model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]

                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target_tensor.reshape(-1),
                    ignore_index=-100,
                )
                loss_value = float(loss.item())
                if not np.isfinite(loss_value):
                    raise ValueError(
                        f"Non-finite hidden loss for cluster {name!r}: {loss_value!r}"
                    )
                losses[index] = loss_value
    finally:
        model.train(was_training)

    return losses


def hidden_value(oracle: Any) -> float:
    """Scalar hidden value: ``-mean(hidden_losses(oracle))``."""
    return -float(np.mean(hidden_losses(oracle)))
