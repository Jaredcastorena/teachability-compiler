#!/usr/bin/env python3
"""Deterministic algebra checks for the selective overlap-record proposal.

Run with Python 3 (standard library only):
    python verify_math.py

The default output is verification_results.json beside this script. These
finite checks supplement the written proofs. They do not establish neural
learning performance, generalization, or an implementation speed advantage.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import struct


def relu(value):
    return max(value, 0)


def crelu(value):
    return relu(value), relu(-value)


def correction(a, b):
    return relu(a) + relu(b) - relu(a + b)


def merge(left, right):
    sl, rl = left
    sr, rr = right
    return sl + sr, rl + rr + correction(sl, sr)


@lru_cache(maxsize=None)
def binary_tree_shapes(leaves):
    if leaves == 1:
        return (None,)
    result = []
    for split in range(1, leaves):
        for left in binary_tree_shapes(split):
            for right in binary_tree_shapes(leaves - split):
                result.append((left, right))
    return tuple(result)


def evaluate_tree(shape, values, offset=0):
    if shape is None:
        return (values[offset], 0), offset + 1
    left, middle = evaluate_tree(shape[0], values, offset)
    right, end = evaluate_tree(shape[1], values, middle)
    return merge(left, right), end


def exact_rank(rows):
    """Row rank by Gaussian elimination over the rational numbers."""
    matrix = [[Fraction(v) for v in row] for row in rows]
    if not matrix:
        return 0
    row_count, columns = len(matrix), len(matrix[0])
    rank = 0
    for column in range(columns):
        pivot = next(
            (i for i in range(rank, row_count) if matrix[i][column]), None
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [v / scale for v in matrix[rank]]
        for row in range(row_count):
            if row != rank and matrix[row][column]:
                scale = matrix[row][column]
                matrix[row] = [
                    a - scale * b
                    for a, b in zip(matrix[row], matrix[rank])
                ]
        rank += 1
        if rank == row_count:
            break
    return rank


def check_two_inputs():
    count = 0
    encoded_targets = {}
    for a, b in product(range(-8, 9), repeat=2):
        total, contrast = a + b, a - b
        r = correction(a, b)
        target = relu(a) + relu(b)
        assert 2 * r == abs(a) + abs(b) - abs(total)
        assert 2 * r == relu(abs(contrast) - abs(total))
        assert r == (0 if a * b >= 0 else min(abs(a), abs(b)))
        assert target == relu(total) + r
        assert (target, relu(-a) + relu(-b)) == (
            relu(total) + r, relu(-total) + r
        )
        code = relu(total), r
        previous = encoded_targets.setdefault(code, target)
        assert previous == target
        count += 1
    return {
        "domain": "A,B are integers from -8 through 8",
        "input_pairs": count,
        "distinct_retained_h_and_correction_pairs": len(encoded_targets),
        "checked": [
            "absolute-value identity",
            "common/contrast identity",
            "mixed-sign gate and minimum magnitude",
            "target reconstruction",
            "positive/negative mass identity",
            "target constancy on retained-record fibers in this finite domain",
        ],
        "passed": True,
    }


def check_arbitrary_coefficients():
    count = 0
    for alpha, beta, a, b in product(range(-3, 4), repeat=4):
        target = alpha * relu(a) + beta * relu(b)
        total = alpha * a + beta * b
        residual = target - relu(total)
        assert 2 * residual == (
            alpha * abs(a) + beta * abs(b) - abs(total)
        )
        leaf_a = alpha * relu(a) - relu(alpha * a)
        leaf_b = beta * relu(b) - relu(beta * b)
        assert leaf_a == (0 if alpha >= 0 else alpha * abs(a))
        assert leaf_b == (0 if beta >= 0 else beta * abs(b))
        combined = merge((alpha * a, leaf_a), (beta * b, leaf_b))
        assert combined == (total, residual)
        assert relu(combined[0]) + combined[1] == target
        if alpha >= 0 and beta >= 0:
            assert residual >= 0
        count += 1
    return {
        "domain": "alpha,beta,A,B are integers from -3 through 3",
        "coefficient_input_quadruples": count,
        "checked": [
            "general signed coefficient correction",
            "negative-coefficient leaf correction",
            "merge with signed leaf records",
        ],
        "passed": True,
    }


def check_positive_position_weights():
    count = 0
    weights = [(lower, upper) for lower in range(1, 5)
               for upper in range(lower + 1, 5)]
    for lower, upper in weights:
        for a, b in product(range(-3, 4), repeat=2):
            u, v = lower * a, upper * b
            target = lower * relu(a) + upper * relu(b)
            assert target == relu(u + v) + correction(u, v)
            count += 1
    return {
        "weights": weights,
        "input_domain": "A,B are integers from -3 through 3",
        "weighted_input_pairs": count,
        "convention": "The second position receives the larger positive weight",
        "passed": True,
    }


def check_original_four_feature_teacher():
    count = 0
    for u1, u2, u3, u4 in product(range(-3, 4), repeat=4):
        twice_target = -relu(u1) + relu(u2) - relu(u3) + relu(u4)
        twice_collapsed = relu(u2 + u4) - relu(u1 + u3)
        twice_record = correction(u2, u4) - correction(u1, u3)
        assert twice_target == twice_collapsed + twice_record
        count += 1
    return {
        "teacher": "f=(-ReLU(u1)+ReLU(u2)-ReLU(u3)+ReLU(u4))/2",
        "pairing": {"positive_output_pair": [2, 4],
                    "negative_output_pair": [1, 3]},
        "domain": "u1,u2,u3,u4 are integers from -3 through 3",
        "input_quadruples": count,
        "arithmetic": "The equality for twice f is checked with integers",
        "scope": "Four intrinsic teacher coordinates; this check does not train an eight-dimensional-input student",
        "passed": True,
    }


def check_multiple_merges():
    shapes = binary_tree_shapes(5)
    assert len(shapes) == 14
    stack_count = 0
    for values in product(range(-2, 3), repeat=5):
        positive = sum(relu(v) for v in values)
        negative = sum(relu(-v) for v in values)
        expected = sum(values), min(positive, negative)
        for shape in shapes:
            state, end = evaluate_tree(shape, values)
            assert end == len(values)
            assert state == expected
            total, residual = state
            assert relu(total) + residual == positive
            assert relu(-total) + residual == negative
        stack_count += 1
    return {
        "domain": "Five integer leaves, each from -2 through 2",
        "stacks": stack_count,
        "ordered_full_binary_groupings_per_stack": len(shapes),
        "tree_evaluations": stack_count * len(shapes),
        "checked": [
            "all five-leaf binary groupings yield the same (S,R)",
            "telescoping target reconstruction",
            "sum CReLU(leaves) equals CReLU(S)+(R,R)",
        ],
        "passed": True,
    }


def check_counterexamples():
    contrast_examples = [(1, -4), (2, -3)]
    contrast_codes = [(relu(a + b), a - b) for a, b in contrast_examples]
    contrast_targets = [relu(a) + relu(b) for a, b in contrast_examples]
    assert contrast_codes[0] == contrast_codes[1]
    assert contrast_targets[0] != contrast_targets[1]

    crelu_examples = [(1, -1), (0, 0)]
    crelu_codes = [crelu(a + b) for a, b in crelu_examples]
    crelu_targets = [relu(a) + relu(b) for a, b in crelu_examples]
    assert crelu_codes[0] == crelu_codes[1]
    assert crelu_targets[0] != crelu_targets[1]

    triples = [(1, 1, 1), (-1, -1, -1)]
    pair_records = [(x * y, x * z, y * z) for x, y, z in triples]
    triple_targets = [x * y * z for x, y, z in triples]
    assert pair_records[0] == pair_records[1]
    assert triple_targets[0] != triple_targets[1]
    return {
        "clipped_sum_plus_signed_contrast": {
            "inputs": contrast_examples, "retained": contrast_codes,
            "targets": contrast_targets,
        },
        "crelu_after_the_sum": {
            "inputs": crelu_examples, "retained": crelu_codes,
            "targets": crelu_targets,
        },
        "pairwise_products_do_not_determine_triple_parity": {
            "inputs": triples, "pairwise_records": pair_records,
            "triple_targets": triple_targets,
            "scope": "An illustrative record ambiguity, not a claim about every atlas transition map",
        },
        "passed": True,
    }


def check_minimal_linear_measurement():
    base = [[1, 1, 0, 0], [0, 0, 1, 0]]
    need = [[1, -1, 0, 0]]
    base_rank = exact_rank(base)
    augmented_rank = exact_rank(base + need)
    assert base_rank == 2 and augmented_rank == 3
    count = 0
    for a, b, c, d in product(range(-2, 3), repeat=4):
        retained = (a + b, c)
        selected = a - b
        positive, negative = crelu(selected)
        decoded_measurement = positive - negative
        assert decoded_measurement == selected
        assert Fraction(retained[0] + decoded_measurement, 2) == a
        assert Fraction(retained[0] - decoded_measurement, 2) == b
        count += 1
    # The nuisance fourth coordinate is deliberately not encoded.
    witness_a, witness_b = (0, 0, 0, 0), (0, 0, 0, 1)
    encode = lambda x: ((x[0] + x[1], x[2]), crelu(x[0] - x[1]))
    assert encode(witness_a) == encode(witness_b)
    return {
        "P": base,
        "V_and_chosen_U": need,
        "rank_P": base_rank,
        "rank_stacked_P_V": augmented_rank,
        "minimum_added_linear_measurements": augmented_rank - base_rank,
        "domain": "Four integer coordinates, each from -2 through 2",
        "inputs": count,
        "unrecovered_nuisance_coordinate_witness": [witness_a, witness_b],
        "scope": "The base map is linear and unrectified; the rank criterion alone is not sufficient for a rectified base",
        "passed": True,
    }


def float32(value):
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def check_finite_precision_examples():
    a32, b32 = 2 ** 24, 2 ** 24 + 1
    assert a32 != b32 and float32(a32) == float32(b32)
    a64, b64 = 2 ** 53, 2 ** 53 + 1
    assert a64 != b64 and float(a64) == float(b64)
    a, b = -1e20, 1.0
    absolute_formula = (abs(a) + abs(b) - abs(a + b)) / 2
    stable_piecewise = 0.0 if (a >= 0) == (b >= 0) else min(abs(a), abs(b))
    assert absolute_formula == 0.0 and stable_piecewise == 1.0
    return {
        "float32_collision": {"distinct_integers": [a32, b32],
                              "same_stored_value": float32(a32)},
        "float64_collision": {"distinct_integers": [a64, b64],
                              "same_stored_value": float(a64)},
        "subtractive_roundoff": {
            "inputs": [a, b],
            "mathematical_correction": 1,
            "absolute_value_formula_in_float64": absolute_formula,
            "sign_gated_minimum_in_float64": stable_piecewise,
        },
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().with_name("verification_results.json"),
    )
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "status": "passed",
        "arithmetic": "Integers and exact Fractions, except explicitly labeled floating-point counterexamples",
        "scope": "Finite verification supports the algebraic derivations; no neural training, efficiency, or generalization claim is tested",
        "checks": {
            "two_input_identities": check_two_inputs(),
            "arbitrary_signed_coefficients": check_arbitrary_coefficients(),
            "positive_position_weights": check_positive_position_weights(),
            "original_four_feature_teacher": check_original_four_feature_teacher(),
            "multiple_merge_telescope": check_multiple_merges(),
            "insufficient_records": check_counterexamples(),
            "selected_linear_measurement": check_minimal_linear_measurement(),
            "finite_precision": check_finite_precision_examples(),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "output": str(args.output),
                      "check_groups": len(result["checks"])}, indent=2))


if __name__ == "__main__":
    main()
