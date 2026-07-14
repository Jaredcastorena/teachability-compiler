"""Deterministic formal-language clusters for the real decoder learner.

The corpus uses a shared printable-ASCII character vocabulary plus PAD/BOS/EOS ids. Each
cluster is intentionally small and structured, so curriculum interactions are interpretable:

* copy_short/copy_long teach copying after a separator; long copy stresses context length.
* reverse_short/reverse_long teach positional manipulation; long reverse is the harder version.
* sort_digits teaches permutation-invariant digit ordering.
* add_1digit/add_2digit form a prerequisite ladder; 2-digit addition builds on 1-digit sums.
* sub_1digit uses similar symbols to addition but different arithmetic, a mild destructive
  analogue.
* mod_arith teaches a distinct arithmetic operation over small integers.
* compare_numbers teaches relational numeric decisions with yes/no outputs.
* bracket_match/bracket_depth teach stack-like structure; matching supports depth estimation.
* letter_shift teaches a prompt-conditioned Caesar +1 mapping.
* letter_shift_conflict uses the same prompt format for Caesar +2, a deliberately destructive
  pair.
* pattern_repeat/pattern_alternate teach sequence continuation with related but distinct rules.
* count_chars teaches lookup plus counting, with digit answers.
* field_lookup teaches named-field retrieval, a prerequisite for concat_fields.
* concat_fields combines field lookup with copying and concatenation, a bridge analogue.
* mixed_review samples easy variants across families, acting as replay.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

PRINTABLE_CHARS: tuple[str, ...] = tuple(chr(code) for code in range(32, 127))
CHAR_TO_ID: dict[str, int] = {char: index + 3 for index, char in enumerate(PRINTABLE_CHARS)}
ID_TO_CHAR: dict[int, str] = {index + 3: char for index, char in enumerate(PRINTABLE_CHARS)}
VOCAB_SIZE = len(PRINTABLE_CHARS) + 3

_CLUSTER_NAMES: tuple[str, ...] = (
    "copy_short",
    "copy_long",
    "reverse_short",
    "reverse_long",
    "sort_digits",
    "add_1digit",
    "add_2digit",
    "sub_1digit",
    "mod_arith",
    "compare_numbers",
    "bracket_match",
    "bracket_depth",
    "letter_shift",
    "letter_shift_conflict",
    "pattern_repeat",
    "pattern_alternate",
    "count_chars",
    "concat_fields",
    "field_lookup",
    "mixed_review",
)

PROBE_CLUSTERS: tuple[str, ...] = _CLUSTER_NAMES

CLUSTER_DESCRIPTIONS: dict[str, str] = {
    "copy_short": "Copy a short string after a separator; prerequisite for field copying.",
    "copy_long": "Copy a longer string after a separator; harder length generalization.",
    "reverse_short": "Reverse a short string; teaches position-dependent transduction.",
    "reverse_long": "Reverse a longer string; harder positional manipulation.",
    "sort_digits": "Sort digit strings; teaches order statistics over symbols.",
    "add_1digit": "Single-digit addition; prerequisite analogue for multi-digit addition.",
    "add_2digit": "Two-digit addition; builds on add_1digit and carry-like behavior.",
    "sub_1digit": "Single-digit subtraction; mildly interferes with addition habits.",
    "mod_arith": "Small-number modular arithmetic; separate arithmetic skill.",
    "compare_numbers": "Numeric comparison with yes/no answers.",
    "bracket_match": "Balanced-bracket recognition; stack-like structural skill.",
    "bracket_depth": "Maximum nesting-depth prediction; interacts with bracket matching.",
    "letter_shift": "Caesar +1 mapping under the prompt format 'shift:x->y'.",
    "letter_shift_conflict": "Caesar +2 using the same prompt format; destructive with +1 shift.",
    "pattern_repeat": "Continue a repeated motif; sequence extrapolation.",
    "pattern_alternate": "Continue an alternating letter/digit/case pattern.",
    "count_chars": "Count occurrences of a queried character; lookup plus counting.",
    "concat_fields": "Read x/y fields and concatenate them; bridge from lookup and copy.",
    "field_lookup": "Retrieve a named field; prerequisite for concat_fields.",
    "mixed_review": "Replay-like mixture of easy variants from other clusters.",
}


def all_cluster_names() -> tuple[str, ...]:
    """Return the exactly 20 named curriculum clusters."""
    return _CLUSTER_NAMES


def sample_batch(
    name: str,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a deterministic-random batch for next-token prediction.

    Args:
        name: Cluster name from :func:`all_cluster_names`.
        batch_size: Number of examples.
        seq_len: Returned sequence length.
        rng: Explicit NumPy generator controlling all data randomness.

    Returns:
        ``inputs`` and ``targets`` tensors of shape ``[batch_size, seq_len]``. Targets are
        shifted next-token ids with ``-100`` in padded positions for PyTorch loss masking.
    """
    if name not in _CLUSTER_NAMES:
        raise ValueError(f"unknown cluster {name!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    encoded = [_encode_example(_make_example(name, rng), seq_len) for _ in range(batch_size)]
    inputs = torch.stack([item[0] for item in encoded], dim=0).long()
    targets = torch.stack([item[1] for item in encoded], dim=0).long()
    return inputs, targets


def probe_batch(
    name: str,
    rng: np.random.Generator | None = None,
    batch_size: int = 64,
    seq_len: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed held-out probe batch for a cluster.

    The ``rng`` argument is accepted for API symmetry, but probes intentionally use a
    cluster-specific fixed seed so repeated evaluations measure the same held-out examples.
    """
    del rng
    probe_rng = np.random.default_rng(_stable_seed(f"probe:{name}"))
    return sample_batch(name, batch_size=batch_size, seq_len=seq_len, rng=probe_rng)


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _encode_example(text: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        token_ids = [BOS_ID, *(CHAR_TO_ID[char] for char in text), EOS_ID]
    except KeyError as exc:
        raise ValueError(f"example contains character outside printable ASCII: {exc}") from exc

    input_ids = token_ids[:-1][:seq_len]
    target_ids = token_ids[1:][:seq_len]
    pad_len = seq_len - len(input_ids)

    if pad_len > 0:
        input_ids = [*input_ids, *([PAD_ID] * pad_len)]
        target_ids = [*target_ids, *([-100] * pad_len)]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def _make_example(name: str, rng: np.random.Generator) -> str:
    if name == "copy_short":
        text = _random_letters(rng, _randint(rng, 3, 8))
        return f"copy:{text}|{text}"
    if name == "copy_long":
        text = _random_letters(rng, _randint(rng, 9, 17))
        return f"copy:{text}|{text}"
    if name == "reverse_short":
        text = _random_letters(rng, _randint(rng, 3, 8))
        return f"rev:{text}|{text[::-1]}"
    if name == "reverse_long":
        text = _random_letters(rng, _randint(rng, 9, 17))
        return f"rev:{text}|{text[::-1]}"
    if name == "sort_digits":
        digits = _random_digits(rng, _randint(rng, 4, 10))
        return f"sort:{digits}|{''.join(sorted(digits))}"
    if name == "add_1digit":
        a = _randint(rng, 0, 10)
        b = _randint(rng, 0, 10)
        return f"{a}+{b}={a + b}"
    if name == "add_2digit":
        a = _randint(rng, 10, 100)
        b = _randint(rng, 10, 100)
        return f"{a}+{b}={a + b}"
    if name == "sub_1digit":
        a = _randint(rng, 0, 10)
        b = _randint(rng, 0, a + 1)
        return f"{a}-{b}={a - b}"
    if name == "mod_arith":
        a = _randint(rng, 0, 31)
        b = _randint(rng, 1, 10)
        return f"{a}%{b}={a % b}"
    if name == "compare_numbers":
        a = _randint(rng, 0, 100)
        b = _randint(rng, 0, 100)
        answer = "yes" if a < b else "no"
        return f"{a}<{b}?{answer}"
    if name == "bracket_match":
        text, is_balanced = _bracket_match_instance(rng)
        answer = "yes" if is_balanced else "no"
        return f"br:{text}?{answer}"
    if name == "bracket_depth":
        depth = _randint(rng, 1, 7)
        text = _balanced_brackets(rng, depth)
        return f"depth:{text}={depth}"
    if name == "letter_shift":
        text = _random_letters(rng, _randint(rng, 3, 9), alphabet="abcdefxyz")
        return f"shift:{text}->{_shift_letters(text, 1)}"
    if name == "letter_shift_conflict":
        text = _random_letters(rng, _randint(rng, 3, 9), alphabet="abcdefxyz")
        return f"shift:{text}->{_shift_letters(text, 2)}"
    if name == "pattern_repeat":
        return _pattern_repeat_instance(rng)
    if name == "pattern_alternate":
        return _pattern_alternate_instance(rng)
    if name == "count_chars":
        chars = "abc"
        text = _random_letters(rng, _randint(rng, 5, 10), alphabet=chars)
        query = _choice(rng, chars)
        return f"count:{query} in {text}={text.count(query)}"
    if name == "concat_fields":
        x_value = _random_letters(rng, _randint(rng, 2, 5), alphabet="abcdxyz")
        y_value = _random_letters(rng, _randint(rng, 2, 5), alphabet="abcdxyz")
        return f"x={x_value};y={y_value};xy=?{x_value + y_value}"
    if name == "field_lookup":
        x_value = _random_letters(rng, _randint(rng, 2, 5), alphabet="abcdxyz")
        y_value = _random_letters(rng, _randint(rng, 2, 5), alphabet="abcdxyz")
        key = _choice(rng, ("x", "y"))
        value = x_value if key == "x" else y_value
        return f"x={x_value};y={y_value};{key}=?{value}"
    if name == "mixed_review":
        easy_names = (
            "copy_short",
            "reverse_short",
            "sort_digits",
            "add_1digit",
            "sub_1digit",
            "compare_numbers",
            "bracket_match",
            "letter_shift",
            "pattern_repeat",
            "count_chars",
            "field_lookup",
        )
        return _make_example(_choice(rng, easy_names), rng)

    raise ValueError(f"unknown cluster {name!r}")


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high))


def _choice(rng: np.random.Generator, values: Sequence[str]) -> str:
    return values[_randint(rng, 0, len(values))]


def _random_letters(
    rng: np.random.Generator,
    length: int,
    alphabet: str = "abcdefghi",
) -> str:
    return "".join(_choice(rng, alphabet) for _ in range(length))


def _random_digits(rng: np.random.Generator, length: int) -> str:
    return "".join(_choice(rng, "0123456789") for _ in range(length))


def _shift_letters(text: str, amount: int) -> str:
    base = ord("a")
    return "".join(chr(base + ((ord(char) - base + amount) % 26)) for char in text)


def _balanced_brackets(rng: np.random.Generator, depth: int) -> str:
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    opens: list[str] = []
    closes: list[str] = []
    for _ in range(depth):
        open_char, close_char = pairs[_randint(rng, 0, len(pairs))]
        opens.append(open_char)
        closes.append(close_char)
    return "".join([*opens, *reversed(closes)])


def _bracket_match_instance(rng: np.random.Generator) -> tuple[str, bool]:
    depth = _randint(rng, 1, 5)
    balanced = _balanced_brackets(rng, depth)
    if _randint(rng, 0, 2) == 0:
        return balanced, True

    invalid_variants = (
        balanced[:-1],
        balanced[1:],
        f"{balanced})",
        f"({balanced}",
        balanced[:-1] + "(",
    )
    return _choice(rng, invalid_variants), False


def _pattern_repeat_instance(rng: np.random.Generator) -> str:
    motif_len = _randint(rng, 2, 4)
    motif = "".join(_choice(rng, "abxy01") for _ in range(motif_len))
    prefix_len = _randint(rng, 5, 10)
    full = (motif * 8)[: prefix_len + 4]
    return f"repeat:{full[:prefix_len]}->{full[prefix_len:prefix_len + 4]}"


def _pattern_alternate_instance(rng: np.random.Generator) -> str:
    letter = _choice(rng, "abcxyz")
    digit = _choice(rng, "012345")
    cycle = (letter, digit, letter.upper(), digit)
    prefix_len = _randint(rng, 6, 11)
    full = "".join(cycle[index % len(cycle)] for index in range(prefix_len + 4))
    return f"alt:{full[:prefix_len]}->{full[prefix_len:prefix_len + 4]}"
