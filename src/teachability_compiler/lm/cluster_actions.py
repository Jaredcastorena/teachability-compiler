"""Build immutable language-corpus curriculum actions from frozen FineWeb shards.

Run with:

    python -m teachability_compiler.lm.cluster_actions --device cuda:1

The module performs no language-model training. It streams the corpus, computes
frozen document features, fits deterministic semantic centers on a seeded sample,
assigns every document to one frozen action or reserved split, and writes an
immutable manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from teachability_compiler.lm.corpus import CorpusSlice, corpus_manifest_hash
from teachability_compiler.lm.embeddings import EduScorer, SentenceEmbedder
from teachability_compiler.lm.features import FEATURE_NAMES, compute_features

N_ACTIONS = 24
STRUCTURAL_ACTIONS: tuple[str, ...] = (
    "code_heavy",
    "math_heavy",
    "list_table_heavy",
    "highly_repetitive",
    "very_long",
    "very_short",
)
# Scaled floor: full runs demand 500 sample docs per semantic cluster; tiny
# --limit-docs smoke runs scale the requirement down with the sample size.
def _min_semantic_cluster_docs(sample_size: int, n_semantic: int) -> int:
    return min(500, max(2, sample_size // (4 * n_semantic)))

CODE_IDX = FEATURE_NAMES.index("code_density")
MATH_IDX = FEATURE_NAMES.index("math_density")
LIST_IDX = FEATURE_NAMES.index("list_table_density")
REPETITIVE_IDX = FEATURE_NAMES.index("repetitiveness")


@dataclass(frozen=True)
class KMeansResult:
    """Output of deterministic numpy k-means."""

    centers: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class FeatureStore:
    """Pass-1 document features and metadata, aligned in corpus order."""

    refs: np.ndarray
    features: np.ndarray
    token_count: np.ndarray
    language_score: np.ndarray


@dataclass(frozen=True)
class FitArtifacts:
    """Frozen fitting artifacts used during full-corpus assignment."""

    centers: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    thresholds: dict[str, float]
    edu_quartiles: np.ndarray
    cluster_edu_medians: np.ndarray
    cluster_sample_counts: np.ndarray


def validate_action_count(n_semantic: int, n_actions: int = N_ACTIONS) -> None:
    """Validate that six structural actions plus semantic hi/lo actions equal 24."""

    actual = len(STRUCTURAL_ACTIONS) + 2 * int(n_semantic)
    if actual != n_actions:
        raise ValueError(
            f"Expected exactly {n_actions} actions, but 6 + 2*n_semantic = {actual}; "
            "use --n-semantic 9 for the default 24-action rung."
        )


def kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    seed: int,
    n_iters: int = 50,
) -> KMeansResult:
    """Deterministic k-means with seeded k-means++ initialization."""

    if vectors.ndim != 2:
        raise ValueError("kmeans expects a rank-2 array")
    if k <= 0:
        raise ValueError("k must be positive")
    if vectors.shape[0] < k:
        raise ValueError(f"k={k} exceeds number of vectors={vectors.shape[0]}")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")
    if not np.isfinite(vectors).all():
        raise ValueError("kmeans input contains NaN or infinity")

    x = np.asarray(vectors, dtype=np.float32)
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus(x, k, rng)

    labels = np.zeros(x.shape[0], dtype=np.int32)
    for _ in range(n_iters):
        labels = nearest_centers(x, centers)
        counts = np.bincount(labels, minlength=k)
        if np.any(counts == 0):
            raise ValueError(
                "Empty semantic cluster encountered during k-means; "
                "try a different seed or k."
            )

        new_centers = np.empty_like(centers)
        for cluster_index in range(k):
            new_centers[cluster_index] = x[labels == cluster_index].mean(axis=0)

        if np.allclose(new_centers, centers, rtol=0.0, atol=1.0e-7):
            centers = new_centers
            break
        centers = new_centers

    labels = nearest_centers(x, centers)
    return KMeansResult(
        centers=centers.astype(np.float32, copy=False),
        labels=labels.astype(np.int32, copy=False),
    )


def nearest_centers(
    vectors: np.ndarray,
    centers: np.ndarray,
    *,
    chunk_size: int = 16384,
) -> np.ndarray:
    """Return nearest-center labels by squared Euclidean distance."""

    if vectors.ndim != 2 or centers.ndim != 2:
        raise ValueError("nearest_centers expects rank-2 arrays")
    if vectors.shape[1] != centers.shape[1]:
        raise ValueError("vectors and centers must have the same dimensionality")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    x = np.asarray(vectors, dtype=np.float32)
    c = np.asarray(centers, dtype=np.float32)
    labels = np.empty(x.shape[0], dtype=np.int32)

    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        diff = x[start:end, None, :] - c[None, :, :]
        distances = np.einsum("nkd,nkd->nk", diff, diff, optimize=True)
        labels[start:end] = np.argmin(distances, axis=1).astype(np.int32)

    return labels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        nargs="+",
        default=[
            "data/fineweb/000_00000.parquet",
            "data/fineweb/001_00000.parquet",
            "data/fineweb/002_00000.parquet",
            "data/fineweb/003_00000.parquet",
        ],
        help="Frozen FineWeb parquet shards.",
    )
    parser.add_argument("--sample-docs", type=int, default=400_000)
    parser.add_argument("--n-semantic", type=int, default=9)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/actions")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.02,
        help="Per-action fraction reserved as cross-domain hidden eval.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.005,
        help="Additional per-action fraction reserved as frozen validation.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=-1,
        help="Process only the first N docs for smoke tests; -1 means all docs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_action_count(args.n_semantic)
    _validate_fractions(args.holdout_fraction, args.val_fraction)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = CorpusSlice(args.shards)
    corpus_hash = corpus_manifest_hash(corpus.shard_paths)
    store = _pass1_features(corpus, out_dir=out_dir, limit_docs=args.limit_docs)

    centers_path = out_dir / "centers.npz"
    cached_artifacts = _load_centers(centers_path, args.n_semantic) if args.limit_docs < 0 else None
    if cached_artifacts is not None:
        print("fit: reusing cached centers.npz (interrupted-run resume)")
        artifacts = cached_artifacts
    else:
        sample_indices = _sample_indices(
            n_docs=store.refs.shape[0],
            sample_docs=args.sample_docs,
            seed=args.seed,
        )
        sample_refs = [tuple(map(int, ref)) for ref in store.refs[sample_indices]]
        sample_rows = corpus.get_docs(sample_refs, columns=("text",))
        sample_texts = [_normalize_text(row["text"]) for row in sample_rows]

        artifacts = _fit_actions(
            store=store,
            sample_indices=sample_indices,
            sample_texts=sample_texts,
            n_semantic=args.n_semantic,
            device=args.device,
            seed=args.seed,
        )
        _write_centers(centers_path, artifacts)

    action_names = _action_names(args.n_semantic)
    original_actions, edu_scores, semantic_clusters = _assign_all_docs(
        corpus=corpus,
        store=store,
        artifacts=artifacts,
        n_semantic=args.n_semantic,
        device=args.device,
    )
    final_actions = _reserve_splits(
        original_actions,
        action_names=action_names,
        holdout_fraction=args.holdout_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    _write_assignments(
        out_dir / "assignments.parquet",
        refs=store.refs,
        actions=final_actions,
        edu_scores=edu_scores,
        semantic_clusters=semantic_clusters,
    )

    action_counts = _counts_by_action(final_actions, store.token_count, action_names)
    reserved_counts = _counts_by_action(final_actions, store.token_count, ("holdout", "val"))
    manifest = _build_manifest(
        args=args,
        action_names=action_names,
        action_counts=action_counts,
        reserved_counts=reserved_counts,
        corpus_hash=corpus_hash,
        artifacts=artifacts,
    )
    _write_manifest(out_dir / "actions_manifest.json", manifest)
    _print_summary(action_counts)


def _kmeans_plus_plus(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    first_index = int(rng.integers(0, x.shape[0]))
    centers[0] = x[first_index]

    closest_distances = _squared_distance_to_one_center(x, centers[0])
    for center_index in range(1, k):
        total_distance = float(np.sum(closest_distances, dtype=np.float64))
        if not np.isfinite(total_distance) or total_distance <= 0.0:
            raise ValueError(
                "Cannot initialize k-means++ centers because sample vectors are degenerate"
            )

        threshold = float(rng.random() * total_distance)
        cumulative = np.cumsum(closest_distances, dtype=np.float64)
        next_index = int(np.searchsorted(cumulative, threshold, side="right"))
        next_index = min(next_index, x.shape[0] - 1)

        centers[center_index] = x[next_index]
        next_distances = _squared_distance_to_one_center(x, centers[center_index])
        closest_distances = np.minimum(closest_distances, next_distances)

    return centers


def _squared_distance_to_one_center(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    diff = x - center[None, :]
    return np.einsum("nd,nd->n", diff, diff, optimize=True)


def _pass1_features(corpus: CorpusSlice, *, out_dir: Path, limit_docs: int) -> FeatureStore:
    # Resume: pass 1 is a pure function of the frozen corpus, so a cached
    # features.npz (from a run interrupted in a later phase) is reusable.
    cache_path = out_dir / "features.npz"
    if cache_path.exists() and limit_docs < 0:
        cached = np.load(cache_path)
        if {"refs", "features", "token_count", "language_score"} <= set(cached.files):
            print(f"pass 1: reusing cached features for {cached['refs'].shape[0]:,} docs")
            return FeatureStore(
                refs=cached["refs"],
                features=cached["features"],
                token_count=cached["token_count"],
                language_score=cached["language_score"],
            )
        print("pass 1: cached features.npz missing keys; recomputing")

    feature_parts: list[np.ndarray] = []
    ref_parts: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []
    language_parts: list[np.ndarray] = []

    processed = 0
    next_progress = 100_000

    for refs, batch in corpus.iter_doc_batches(
        columns=("text", "token_count", "language_score"),
        batch_size=8192,
    ):
        remaining = _remaining_limit(limit_docs, processed)
        if remaining == 0:
            break
        if remaining > 0 and refs.shape[0] > remaining:
            refs = refs[:remaining]
            batch = {key: values[:remaining] for key, values in batch.items()}

        texts = [_normalize_text(value) for value in batch["text"]]
        features = compute_features(texts)
        if not np.isfinite(features).all():
            raise ValueError("NaN or infinite structural features in pass 1")

        ref_parts.append(refs.astype(np.int64, copy=True))
        feature_parts.append(features.astype(np.float32, copy=False))
        token_parts.append(_int_array(batch["token_count"]))
        language_parts.append(_float_array(batch["language_score"]))

        processed += len(texts)
        while processed >= next_progress:
            print(f"pass 1: processed {next_progress:,} docs", flush=True)
            next_progress += 100_000

    if not feature_parts:
        raise ValueError("No documents were read from the corpus")

    store = FeatureStore(
        refs=np.vstack(ref_parts).astype(np.int64, copy=False),
        features=np.vstack(feature_parts).astype(np.float32, copy=False),
        token_count=np.concatenate(token_parts).astype(np.int64, copy=False),
        language_score=np.concatenate(language_parts).astype(np.float32, copy=False),
    )

    np.savez_compressed(
        out_dir / "features.npz",
        refs=store.refs,
        features=store.features,
        token_count=store.token_count,
        language_score=store.language_score,
        feature_names=np.array(FEATURE_NAMES),
    )
    return store


def _sample_indices(n_docs: int, sample_docs: int, seed: int) -> np.ndarray:
    if sample_docs <= 0:
        raise ValueError("--sample-docs must be positive")
    if n_docs <= 0:
        raise ValueError("Cannot sample from an empty corpus")

    sample_size = min(int(sample_docs), int(n_docs))
    rng = np.random.default_rng(seed)
    return rng.choice(n_docs, size=sample_size, replace=False).astype(np.int64)


def _fit_actions(
    *,
    store: FeatureStore,
    sample_indices: np.ndarray,
    sample_texts: Sequence[str],
    n_semantic: int,
    device: str,
    seed: int,
) -> FitArtifacts:
    sample_features = store.features[sample_indices].astype(np.float64)
    sample_tokens = store.token_count[sample_indices].astype(np.float64)

    feature_mean = sample_features.mean(axis=0)
    feature_std = sample_features.std(axis=0)
    feature_std = np.where(feature_std < 1.0e-8, 1.0, feature_std)

    thresholds = _structural_thresholds(sample_features, sample_tokens)

    embedder = SentenceEmbedder(device=device)
    try:
        sample_embeddings = embedder.encode(sample_texts, batch_size=256, max_chars=2000)
    finally:
        embedder.close()

    scorer = EduScorer(device=device)
    try:
        sample_edu_scores = scorer.score(sample_texts, batch_size=128, max_chars=2000)
    finally:
        scorer.close()

    if not np.isfinite(sample_embeddings).all():
        raise ValueError("Sample embeddings contain NaN or infinity")
    if not np.isfinite(sample_edu_scores).all():
        raise ValueError("Sample edu scores contain NaN or infinity")

    kmeans_result = kmeans(sample_embeddings, n_semantic, seed=seed, n_iters=50)
    cluster_counts = np.bincount(kmeans_result.labels, minlength=n_semantic).astype(np.int64)
    min_docs = _min_semantic_cluster_docs(int(cluster_counts.sum()), len(cluster_counts))
    small_clusters = np.flatnonzero(cluster_counts < min_docs)
    if small_clusters.size:
        small = ", ".join(str(int(cluster)) for cluster in small_clusters)
        raise ValueError(
            f"Semantic cluster(s) {small} have fewer than "
            f"{min_docs} sample docs; try a different seed or k."
        )

    cluster_medians = np.empty(n_semantic, dtype=np.float32)
    for cluster_index in range(n_semantic):
        cluster_scores = sample_edu_scores[kmeans_result.labels == cluster_index]
        if cluster_scores.size == 0:
            raise ValueError(
                "Empty semantic cluster encountered when computing medians; "
                "try a different seed or k."
            )
        cluster_medians[cluster_index] = float(np.median(cluster_scores))

    edu_quartiles = np.quantile(sample_edu_scores, [0.25, 0.50, 0.75]).astype(np.float32)

    return FitArtifacts(
        centers=kmeans_result.centers.astype(np.float32, copy=False),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        thresholds=thresholds,
        edu_quartiles=edu_quartiles,
        cluster_edu_medians=cluster_medians,
        cluster_sample_counts=cluster_counts,
    )


def _assign_all_docs(
    *,
    corpus: CorpusSlice,
    store: FeatureStore,
    artifacts: FitArtifacts,
    n_semantic: int,
    device: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    expected_docs = int(store.refs.shape[0])
    if artifacts.centers.shape[0] != n_semantic:
        raise ValueError("Number of semantic centers does not match --n-semantic")

    actions = [""] * expected_docs
    edu_scores = np.full(expected_docs, np.nan, dtype=np.float32)
    semantic_clusters = np.full(expected_docs, -1, dtype=np.int16)

    pending_texts: list[str] = []
    pending_indices: list[int] = []
    embedder: SentenceEmbedder | None = None
    scorer: EduScorer | None = None

    def flush_pending() -> None:
        nonlocal embedder, pending_indices, pending_texts, scorer

        if not pending_texts:
            return
        if embedder is None:
            embedder = SentenceEmbedder(device=device)
        if scorer is None:
            scorer = EduScorer(device=device)

        embeddings = embedder.encode(pending_texts, batch_size=256, max_chars=2000)
        scores = scorer.score(pending_texts, batch_size=128, max_chars=2000)
        labels = nearest_centers(embeddings, artifacts.centers)

        for local_index, doc_index in enumerate(pending_indices):
            cluster_index = int(labels[local_index])
            score = float(scores[local_index])
            median = float(artifacts.cluster_edu_medians[cluster_index])
            quality = "hi" if score >= median else "lo"

            actions[doc_index] = f"semantic_{cluster_index}_{quality}"
            edu_scores[doc_index] = score
            semantic_clusters[doc_index] = cluster_index

        pending_texts = []
        pending_indices = []

    processed = 0
    feature_pos = 0
    next_progress = 100_000

    try:
        for refs, batch in corpus.iter_doc_batches(columns=("text",), batch_size=8192):
            if feature_pos >= expected_docs:
                break

            batch_rows = int(refs.shape[0])
            if feature_pos + batch_rows > expected_docs:
                batch_rows = expected_docs - feature_pos
                refs = refs[:batch_rows]
                batch = {key: values[:batch_rows] for key, values in batch.items()}

            expected_refs = store.refs[feature_pos : feature_pos + batch_rows]
            if not np.array_equal(refs, expected_refs):
                raise RuntimeError("Pass-2 corpus order does not match pass-1 feature refs")

            texts = [_normalize_text(value) for value in batch["text"]]
            features = store.features[feature_pos : feature_pos + batch_rows]
            token_counts = store.token_count[feature_pos : feature_pos + batch_rows]

            for row_offset, text in enumerate(texts):
                doc_index = feature_pos + row_offset
                structural_action = _structural_action(
                    features[row_offset],
                    int(token_counts[row_offset]),
                    artifacts.thresholds,
                )
                if structural_action is None:
                    pending_indices.append(doc_index)
                    pending_texts.append(text)
                    if len(pending_texts) >= 4096:
                        flush_pending()
                else:
                    actions[doc_index] = structural_action

            feature_pos += batch_rows
            processed += batch_rows
            while processed >= next_progress:
                print(f"pass 2: assigned {next_progress:,} docs", flush=True)
                next_progress += 100_000

        flush_pending()
    finally:
        if embedder is not None:
            embedder.close()
        if scorer is not None:
            scorer.close()

    if feature_pos != expected_docs:
        raise RuntimeError(f"Assigned {feature_pos} docs but expected {expected_docs}")
    if any(action == "" for action in actions):
        raise RuntimeError("At least one document was not assigned an action")
    if len(_action_names(n_semantic)) != N_ACTIONS:
        raise RuntimeError("Internal action-name count mismatch")

    return actions, edu_scores, semantic_clusters


def _structural_thresholds(
    sample_features: np.ndarray,
    sample_token_count: np.ndarray,
) -> dict[str, float]:
    code_q95 = _quantile(sample_features[:, CODE_IDX], 0.95)
    non_code_mask = sample_features[:, CODE_IDX] <= code_q95
    math_source = sample_features[non_code_mask, MATH_IDX]
    if math_source.size == 0:
        math_source = sample_features[:, MATH_IDX]

    return {
        "code_density_q95": code_q95,
        "math_density_q95_non_code": _quantile(math_source, 0.95),
        "list_table_density_q95": _quantile(sample_features[:, LIST_IDX], 0.95),
        "repetitiveness_q95": _quantile(sample_features[:, REPETITIVE_IDX], 0.95),
        "token_count_q95": _quantile(sample_token_count, 0.95),
        "token_count_q05": _quantile(sample_token_count, 0.05),
    }


def _structural_action(
    features: np.ndarray,
    token_count: int,
    thresholds: dict[str, float],
) -> str | None:
    if float(features[CODE_IDX]) > thresholds["code_density_q95"]:
        return "code_heavy"
    if float(features[MATH_IDX]) > thresholds["math_density_q95_non_code"]:
        return "math_heavy"
    if float(features[LIST_IDX]) > thresholds["list_table_density_q95"]:
        return "list_table_heavy"
    if float(features[REPETITIVE_IDX]) > thresholds["repetitiveness_q95"]:
        return "highly_repetitive"
    if float(token_count) > thresholds["token_count_q95"]:
        return "very_long"
    if float(token_count) < thresholds["token_count_q05"]:
        return "very_short"
    return None


def _reserve_splits(
    actions: Sequence[str],
    *,
    action_names: Sequence[str],
    holdout_fraction: float,
    val_fraction: float,
    seed: int,
) -> np.ndarray:
    final_actions = np.array(actions, dtype=object)
    rng = np.random.default_rng(seed + 1_000_003)

    for action_name in action_names:
        indices = np.flatnonzero(final_actions == action_name)
        n_holdout = int(np.floor(indices.shape[0] * holdout_fraction))
        n_val = int(np.floor(indices.shape[0] * val_fraction))
        n_reserved = n_holdout + n_val

        if n_reserved == 0:
            continue

        chosen = rng.choice(indices, size=n_reserved, replace=False)
        final_actions[chosen[:n_holdout]] = "holdout"
        final_actions[chosen[n_holdout:]] = "val"

    return final_actions.astype(str)


def _write_assignments(
    path: Path,
    *,
    refs: np.ndarray,
    actions: np.ndarray,
    edu_scores: np.ndarray,
    semantic_clusters: np.ndarray,
) -> None:
    table = pa.table(
        {
            "shard_index": pa.array(refs[:, 0].astype(np.int32)),
            "row_index": pa.array(refs[:, 1].astype(np.int64)),
            "action": pa.array(actions.astype(str)),
            "edu_score": pa.array(edu_scores.astype(np.float32)),
            "semantic_cluster": pa.array(semantic_clusters.astype(np.int16)),
        }
    )
    pq.write_table(table, path, compression="zstd", use_dictionary=["action"])


def _load_centers(path: Path, n_semantic: int) -> FitArtifacts | None:
    """Load fit artifacts cached by an interrupted run; None if absent/invalid."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    required = {
        "centers", "feature_mean", "feature_std", "threshold_names",
        "threshold_values", "edu_quartiles", "cluster_edu_medians",
        "cluster_sample_counts",
    }
    if not required <= set(data.files) or data["centers"].shape[0] != n_semantic:
        print("fit: cached centers.npz incompatible; refitting")
        return None
    thresholds = {
        str(name): float(value)
        for name, value in zip(data["threshold_names"], data["threshold_values"], strict=True)
    }
    return FitArtifacts(
        centers=data["centers"],
        feature_mean=data["feature_mean"],
        feature_std=data["feature_std"],
        thresholds=thresholds,
        edu_quartiles=data["edu_quartiles"],
        cluster_edu_medians=data["cluster_edu_medians"],
        cluster_sample_counts=data["cluster_sample_counts"],
    )


def _write_centers(path: Path, artifacts: FitArtifacts) -> None:
    threshold_names = np.array(list(artifacts.thresholds.keys()))
    threshold_values = np.array(list(artifacts.thresholds.values()), dtype=np.float64)
    np.savez(
        path,
        centers=artifacts.centers,
        feature_mean=artifacts.feature_mean,
        feature_std=artifacts.feature_std,
        threshold_names=threshold_names,
        threshold_values=threshold_values,
        edu_quartiles=artifacts.edu_quartiles,
        cluster_edu_medians=artifacts.cluster_edu_medians,
        cluster_sample_counts=artifacts.cluster_sample_counts,
    )


def _build_manifest(
    *,
    args: argparse.Namespace,
    action_names: Sequence[str],
    action_counts: dict[str, dict[str, int]],
    reserved_counts: dict[str, dict[str, int]],
    corpus_hash: str,
    artifacts: FitArtifacts,
) -> dict[str, object]:
    config = {
        "shards": [str(shard) for shard in args.shards],
        "sample_docs": int(args.sample_docs),
        "n_semantic": int(args.n_semantic),
        "device": str(args.device),
        "seed": int(args.seed),
        "out_dir": str(args.out_dir),
        "holdout_fraction": float(args.holdout_fraction),
        "val_fraction": float(args.val_fraction),
        "limit_docs": int(args.limit_docs),
        "n_actions": N_ACTIONS,
    }

    manifest_base: dict[str, object] = {
        "action_names": list(action_names),
        "actions": _action_definitions(
            n_semantic=args.n_semantic,
            thresholds=artifacts.thresholds,
            cluster_edu_medians=artifacts.cluster_edu_medians,
        ),
        "classifier_model_name": "HuggingFaceFW/fineweb-edu-classifier",
        "config": config,
        "corpus_manifest_hash": corpus_hash,
        "created_at": _created_at_iso(),
        "doc_counts": action_counts,
        "edu_quartile_thresholds": {
            "q25": float(artifacts.edu_quartiles[0]),
            "q50": float(artifacts.edu_quartiles[1]),
            "q75": float(artifacts.edu_quartiles[2]),
        },
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "feature_names": list(FEATURE_NAMES),
        "feature_standardization": {
            "mean": [float(value) for value in artifacts.feature_mean],
            "std": [float(value) for value in artifacts.feature_std],
        },
        "git_commit": _git_commit(),
        "n_actions": N_ACTIONS,
        "reserved_counts": reserved_counts,
        "seed": int(args.seed),
        "semantic_cluster_sample_counts": [
            int(value) for value in artifacts.cluster_sample_counts
        ],
    }

    canonical = json.dumps(
        manifest_base,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    manifest_base["manifest_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return manifest_base


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _action_definitions(
    *,
    n_semantic: int,
    thresholds: dict[str, float],
    cluster_edu_medians: np.ndarray,
) -> list[dict[str, object]]:
    definitions: list[dict[str, object]] = [
        {
            "name": "code_heavy",
            "kind": "structural",
            "first_match_rank": 0,
            "feature": "code_density",
            "operator": ">",
            "threshold": float(thresholds["code_density_q95"]),
            "rule": "first match where code_density exceeds sample q95",
        },
        {
            "name": "math_heavy",
            "kind": "structural",
            "first_match_rank": 1,
            "feature": "math_density",
            "operator": ">",
            "threshold": float(thresholds["math_density_q95_non_code"]),
            "rule": "first non-code-threshold match where math_density exceeds q95",
        },
        {
            "name": "list_table_heavy",
            "kind": "structural",
            "first_match_rank": 2,
            "feature": "list_table_density",
            "operator": ">",
            "threshold": float(thresholds["list_table_density_q95"]),
            "rule": "first match where list_table_density exceeds sample q95",
        },
        {
            "name": "highly_repetitive",
            "kind": "structural",
            "first_match_rank": 3,
            "feature": "repetitiveness",
            "operator": ">",
            "threshold": float(thresholds["repetitiveness_q95"]),
            "rule": "first match where repetitiveness exceeds sample q95",
        },
        {
            "name": "very_long",
            "kind": "structural",
            "first_match_rank": 4,
            "feature": "token_count",
            "operator": ">",
            "threshold": float(thresholds["token_count_q95"]),
            "rule": "first match where token_count exceeds sample q95",
        },
        {
            "name": "very_short",
            "kind": "structural",
            "first_match_rank": 5,
            "feature": "token_count",
            "operator": "<",
            "threshold": float(thresholds["token_count_q05"]),
            "rule": "first match where token_count is below sample q05",
        },
    ]

    for cluster_index in range(n_semantic):
        median = float(cluster_edu_medians[cluster_index])
        definitions.append(
            {
                "name": f"semantic_{cluster_index}_hi",
                "kind": "semantic_quality",
                "semantic_cluster": cluster_index,
                "center_index": cluster_index,
                "quality": "hi",
                "feature": "edu_score",
                "operator": ">=",
                "threshold": median,
                "rule": (
                    "remaining document assigned to nearest embedding center, "
                    "then edu_score >= frozen per-cluster sample median"
                ),
            }
        )
        definitions.append(
            {
                "name": f"semantic_{cluster_index}_lo",
                "kind": "semantic_quality",
                "semantic_cluster": cluster_index,
                "center_index": cluster_index,
                "quality": "lo",
                "feature": "edu_score",
                "operator": "<",
                "threshold": median,
                "rule": (
                    "remaining document assigned to nearest embedding center, "
                    "then edu_score < frozen per-cluster sample median"
                ),
            }
        )

    return definitions


def _counts_by_action(
    actions: np.ndarray,
    token_count: np.ndarray,
    action_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for action_name in action_names:
        mask = actions == action_name
        counts[action_name] = {
            "docs": int(mask.sum()),
            "tokens": int(token_count[mask].sum()),
        }
    return counts


def _print_summary(action_counts: dict[str, dict[str, int]]) -> None:
    total_docs = sum(values["docs"] for values in action_counts.values())
    print("action | docs | ~tokens | share", flush=True)
    for action_name, counts in action_counts.items():
        docs = counts["docs"]
        tokens = counts["tokens"]
        share = 0.0 if total_docs == 0 else docs / total_docs
        print(f"{action_name} | {docs:,} | {tokens:,} | {share:.4f}", flush=True)


def _action_names(n_semantic: int) -> list[str]:
    names = list(STRUCTURAL_ACTIONS)
    for cluster_index in range(n_semantic):
        names.append(f"semantic_{cluster_index}_hi")
        names.append(f"semantic_{cluster_index}_lo")
    return names


def _validate_fractions(holdout_fraction: float, val_fraction: float) -> None:
    if holdout_fraction < 0.0 or val_fraction < 0.0:
        raise ValueError("holdout and validation fractions must be non-negative")
    if holdout_fraction + val_fraction >= 1.0:
        raise ValueError("holdout_fraction + val_fraction must be < 1")


def _remaining_limit(limit_docs: int, processed: int) -> int:
    if limit_docs < 0:
        return -1
    return max(int(limit_docs) - processed, 0)


def _normalize_text(value: object) -> str:
    return "" if value is None else str(value)


def _int_array(values: Sequence[object]) -> np.ndarray:
    return np.array([0 if value is None else int(value) for value in values], dtype=np.int64)


def _float_array(values: Sequence[object]) -> np.ndarray:
    return np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float32,
    )


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute quantile of an empty array")
    return float(np.quantile(values.astype(np.float64), q))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"

    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unknown"
    return commit


def _created_at_iso() -> str:
    configured = os.environ.get("TEACHABILITY_COMPILER_CREATED_AT")
    if configured:
        return configured

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        timestamp = int(source_date_epoch)
        created_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(microsecond=0)
        return created_at.isoformat().replace("+00:00", "Z")

    return "1970-01-01T00:00:00Z"


if __name__ == "__main__":
    main()
