"""Frozen language-corpus action construction for teachability-compiler.

The :mod:`teachability_compiler.lm` subpackage builds immutable curriculum
actions over a frozen FineWeb slice. It provides streaming parquet corpus access,
cheap deterministic document features, frozen embedding/classifier inference
helpers, and the runnable action-clustering pipeline. This rung contains no
language-model training code.
"""

from teachability_compiler.lm.corpus import CorpusSlice, corpus_manifest_hash
from teachability_compiler.lm.embeddings import (
    EduScorer,
    SentenceEmbedder,
    edu_score_texts,
    embed_texts,
)
from teachability_compiler.lm.features import FEATURE_NAMES, compute_features

__all__ = [
    "CorpusSlice",
    "EduScorer",
    "FEATURE_NAMES",
    "SentenceEmbedder",
    "compute_features",
    "corpus_manifest_hash",
    "edu_score_texts",
    "embed_texts",
]
