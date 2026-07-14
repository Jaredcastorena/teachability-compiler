"""Persistence helpers and probe-suite fingerprinting for real transitions."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from teachability_compiler.real.tasks import all_cluster_names, probe_batch
from teachability_compiler.state import TransitionObservation


def save_transitions(
    path: str | Path,
    observations: Sequence[TransitionObservation],
    metadata: dict[str, Any],
) -> None:
    """Save transition observations plus metadata as a protocol-5 pickle."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"observations": list(observations), "metadata": metadata}
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)


def load_transitions(
    paths: Sequence[str | Path],
) -> tuple[list[TransitionObservation], list[dict[str, Any]]]:
    """Load and concatenate transition files, returning observations and metadata."""
    observations: list[TransitionObservation] = []
    metadata: list[dict[str, Any]] = []

    for path in paths:
        in_path = Path(path)
        with in_path.open("rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise TypeError(f"{in_path} did not contain a dictionary payload")
        if "observations" not in payload:
            raise KeyError(f"{in_path} is missing required key 'observations'")
        if "metadata" not in payload:
            raise KeyError(f"{in_path} is missing required key 'metadata'")

        file_observations = payload["observations"]
        file_metadata = payload["metadata"]
        if not isinstance(file_metadata, dict):
            raise TypeError(f"{in_path} metadata is not a dictionary")

        observations.extend(file_observations)
        metadata.append(file_metadata)

    return observations, metadata


def probe_suite_hash() -> str:
    """Return a 16-character SHA256 fingerprint of the fixed probe suite."""
    hasher = hashlib.sha256()
    cluster_names = tuple(sorted(all_cluster_names()))
    hasher.update("|".join(cluster_names).encode("utf-8"))

    for name in cluster_names:
        inputs, targets = probe_batch(name, batch_size=64, seq_len=64)
        hasher.update(_flattened_int64_bytes(inputs))
        hasher.update(_flattened_int64_bytes(targets))

    return hasher.hexdigest()[:16]


def _flattened_int64_bytes(values: Any) -> bytes:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(values).reshape(-1), dtype=np.int64)
    return array.tobytes()
