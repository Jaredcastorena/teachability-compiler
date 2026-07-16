from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from teachability_compiler.lm.cluster_actions import kmeans, validate_action_count
from teachability_compiler.lm.corpus import CorpusSlice
from teachability_compiler.lm.features import FEATURE_NAMES, compute_features


def test_corpus_slice_iteration_and_get_docs_round_trip(tmp_path: Path) -> None:
    shard = tmp_path / "tiny.parquet"
    rows = 200
    texts = [f"document {index}\nwith text" for index in range(rows)]
    ids = [f"id-{index}" for index in range(rows)]
    token_counts = [index + 10 for index in range(rows)]
    language_scores = [0.5 + index / 1000.0 for index in range(rows)]

    table = pa.table(
        {
            "text": texts,
            "id": ids,
            "token_count": token_counts,
            "language_score": language_scores,
        }
    )
    pq.write_table(table, shard, row_group_size=17)

    corpus = CorpusSlice([shard])
    assert corpus.doc_count == rows

    iterated = list(corpus.iter_docs(columns=("text", "id", "token_count")))
    assert len(iterated) == rows
    assert iterated[0][0] == (0, 0)
    assert iterated[-1][0] == (0, rows - 1)
    assert iterated[31][1]["id"] == "id-31"

    refs = [iterated[5][0], iterated[31][0], iterated[0][0], iterated[199][0]]
    fetched = corpus.get_docs(refs, columns=("id", "text", "token_count"))
    assert [row["id"] for row in fetched] == ["id-5", "id-31", "id-0", "id-199"]
    assert fetched[1]["text"] == texts[31]
    assert fetched[3]["token_count"] == token_counts[199]


def test_compute_features_shapes_finiteness_and_ordering() -> None:
    code = """
def add(x, y):
    return x + y;

class Thing:
    pass
"""
    prose = (
        "This is a simple paragraph about gardens and weather. "
        "It has ordinary sentences without programming syntax."
    )
    repeated = "abc " * 2000
    randomish = _random_words(1000)

    features = compute_features([code, prose, repeated, randomish])
    assert features.shape == (4, len(FEATURE_NAMES))
    assert np.isfinite(features).all()

    code_idx = FEATURE_NAMES.index("code_density")
    repetitive_idx = FEATURE_NAMES.index("repetitiveness")
    assert features[0, code_idx] > features[1, code_idx]
    assert features[2, repetitive_idx] > features[3, repetitive_idx]


def test_kmeans_determinism() -> None:
    rng = np.random.default_rng(123)
    vectors = rng.normal(size=(120, 8)).astype(np.float32)

    first = kmeans(vectors, 5, seed=99, n_iters=15)
    second = kmeans(vectors, 5, seed=99, n_iters=15)

    assert np.array_equal(first.labels, second.labels)
    assert np.allclose(first.centers, second.centers)


def test_action_count_validation_error() -> None:
    validate_action_count(9)
    with pytest.raises(ValueError, match="24"):
        validate_action_count(12)


def _random_words(n_words: int) -> str:
    rng = np.random.default_rng(7)
    alphabet = np.array(list(string.ascii_lowercase))
    words: list[str] = []
    for _ in range(n_words):
        letters = rng.choice(alphabet, size=8, replace=True)
        words.append("".join(str(letter) for letter in letters))
    return " ".join(words)
