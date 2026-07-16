"""Cheap deterministic structural document features.

All features are computed from at most the first 8,000 characters of each
document. The repetitiveness feature uses at most the first 4 KiB of that sample.
The implementation is CPU-only and depends only on numpy and the standard
library.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Sequence

import numpy as np

MAX_FEATURE_CHARS = 8000
MAX_LEXICAL_WORDS = 512
MAX_REPETITION_BYTES = 4096

FEATURE_NAMES: tuple[str, ...] = (
    "log_chars",
    "log_token_estimate",
    "code_density",
    "math_density",
    "list_table_density",
    "lexical_diversity",
    "repetitiveness",
    "mean_line_length",
    "uppercase_fraction",
    "digit_fraction",
)

_CODE_LINE_RE = re.compile(
    r"(^\s{4,}\S)|(\t)|([{};]\s*$)|"
    r"(\b(def|class|import|from|function|var|let|const)\b)|(=>)"
)
_LIST_TABLE_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\|)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_DIGIT_OPERATOR_RE = re.compile(
    r"(?:\d\s*[=+\-*/^]\s*\d|\d\s*[=+\-*/^]|[=+\-*/^]\s*\d)"
)
_LATEX_RE = re.compile(r"(?:\\frac|\\begin|\\end|\$[^$\n]{1,200}\$)")
_MATH_CHARS = set("=+-*/^_{}\\()")


def compute_features(texts: list[str]) -> np.ndarray:
    """Compute structural features for ``texts``.

    Parameters
    ----------
    texts:
        Documents to featurize. Every document is truncated to the first 8,000
        characters before feature extraction.

    Returns
    -------
    np.ndarray
        Finite ``float32`` array with shape ``[len(texts), len(FEATURE_NAMES)]``.
    """

    features = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float32)
    for row_index, text in enumerate(texts):
        features[row_index] = _compute_one("" if text is None else str(text))

    if not np.isfinite(features).all():
        raise ValueError("NaN or infinite document feature encountered")
    return features


def _compute_one(text: str) -> np.ndarray:
    sample = text[:MAX_FEATURE_CHARS]
    char_count = len(sample)
    token_estimate = char_count / 4.0

    lines = sample.splitlines()
    line_count = len(lines)
    line_denominator = max(line_count, 1)

    code_density = _line_match_fraction(lines, _CODE_LINE_RE, line_denominator)
    math_density = _math_density(sample)
    list_table_density = _line_match_fraction(lines, _LIST_TABLE_LINE_RE, line_denominator)
    lexical_diversity = _lexical_diversity(sample)
    repetitiveness = _repetitiveness(sample)
    mean_line_length = float(char_count / line_denominator) if char_count else 0.0
    uppercase_fraction = _char_fraction(sample, str.isupper)
    digit_fraction = _char_fraction(sample, str.isdigit)

    return np.array(
        [
            np.log1p(float(char_count)),
            np.log1p(float(token_estimate)),
            code_density,
            math_density,
            list_table_density,
            lexical_diversity,
            repetitiveness,
            mean_line_length,
            uppercase_fraction,
            digit_fraction,
        ],
        dtype=np.float32,
    )


def _line_match_fraction(
    lines: Sequence[str],
    pattern: re.Pattern[str],
    denominator: int,
) -> float:
    if not lines:
        return 0.0
    matches = sum(1 for line in lines if pattern.search(line) is not None)
    return float(matches / denominator)


def _math_density(sample: str) -> float:
    if not sample:
        return 0.0

    math_char_count = sum(1 for char in sample if char in _MATH_CHARS)
    digit_operator_count = sum(
        len(match.group(0)) for match in _DIGIT_OPERATOR_RE.finditer(sample)
    )
    latex_count = sum(len(match.group(0)) for match in _LATEX_RE.finditer(sample))
    return float(min(1.0, (math_char_count + digit_operator_count + latex_count) / len(sample)))


def _lexical_diversity(sample: str) -> float:
    words = [match.group(0).lower() for match in _WORD_RE.finditer(sample)]
    if not words:
        return 0.0

    first_words = words[:MAX_LEXICAL_WORDS]
    return float(len(set(first_words)) / len(first_words))


def _repetitiveness(sample: str) -> float:
    data = sample.encode("utf-8", errors="ignore")[:MAX_REPETITION_BYTES]
    if not data:
        return 0.0

    compressed = zlib.compress(data)
    return float(1.0 - (len(compressed) / len(data)))


def _char_fraction(sample: str, predicate: Callable[[str], bool]) -> float:
    if not sample:
        return 0.0

    count = sum(1 for char in sample if predicate(char))
    return float(count / len(sample))
