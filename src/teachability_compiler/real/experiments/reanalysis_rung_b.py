"""Rung B reanalyses from saved transition pickles (no new oracle runs).

1. Field rank: SVD of coordinate-resolved commutator vectors per checkpoint.
2. Fingerprint-commutator correlation: does the fingerprint dot product
   predict the commutator magnitude?
3. Stage-breaking test: residual-MLP one-step error on developmentally
   divergent race trajectories vs the uniform reference at matched steps.

Commutator transitions exist only for the 10k and 30k checkpoints (the
300/3k runs predate --save-transitions), so analyses 1 and 2 cover those
two stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from teachability_compiler.real.mlp_predictor import ResidualMLPTransitionPredictor
from teachability_compiler.real.persistence import load_transitions, probe_suite_hash
from teachability_compiler.real.tasks import all_cluster_names
from teachability_compiler.state import TransitionObservation

N_NOISE_TRANSITIONS = 20
LEGS_PER_REPEAT = 4


def _reconstruct_pairs(
    observations: list[TransitionObservation],
    repeats: int,
) -> list[dict[str, Any]]:
    """Rebuild (A, B, C-vector per repeat) records from the saved leg order.

    commutator_tail.py appends: 20 noise transitions, then per pair per
    repeat the legs (A-first, AB, B-first, BA). The AB/BA chain structure is
    verified via state linkage before anything is trusted.
    """

    legs = observations[N_NOISE_TRANSITIONS:]
    per_pair = repeats * LEGS_PER_REPEAT
    if len(legs) % per_pair != 0:
        raise ValueError(f"leg count {len(legs)} not divisible by {per_pair}")

    records: list[dict[str, Any]] = []
    for start in range(0, len(legs), per_pair):
        chunk = legs[start : start + per_pair]
        a_name = chunk[0].action.cluster_ids[0]
        b_name = chunk[1].action.cluster_ids[0]
        vectors: list[np.ndarray] = []
        for r in range(repeats):
            a_first, ab, b_first, ba = chunk[r * 4 : r * 4 + 4]
            if (a_first.action.cluster_ids[0], ab.action.cluster_ids[0]) != (a_name, b_name):
                raise ValueError(f"leg order mismatch at pair ({a_name}, {b_name})")
            if (b_first.action.cluster_ids[0], ba.action.cluster_ids[0]) != (b_name, a_name):
                raise ValueError(f"reverse leg mismatch at pair ({a_name}, {b_name})")
            if not np.allclose(
                ab.state_before.probe_losses, a_first.state_after.probe_losses
            ) or not np.allclose(ba.state_before.probe_losses, b_first.state_after.probe_losses):
                raise ValueError(f"broken state linkage at pair ({a_name}, {b_name})")
            vectors.append(
                np.asarray(ab.state_after.probe_losses, dtype=np.float64)
                - np.asarray(ba.state_after.probe_losses, dtype=np.float64)
            )
        records.append(
            {
                "a": a_name,
                "b": b_name,
                "vectors": vectors,
                "mean_vector": np.mean(np.stack(vectors), axis=0),
                "first_leg_deltas": {
                    a_name: [np.asarray(chunk[0].probe_delta, dtype=np.float64)],
                    b_name: [np.asarray(chunk[2].probe_delta, dtype=np.float64)],
                },
            }
        )
    return records


def _field_rank(records: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    matrix = np.stack([r["mean_vector"] for r in records])
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    energy = singular**2 / float(np.sum(singular**2))
    cluster_names = all_cluster_names()
    components = []
    for i in range(top_k):
        loadings = vt[i]
        order = np.argsort(-np.abs(loadings))[:4]
        components.append(
            {
                "energy_fraction": float(energy[i]),
                "top_probes": [
                    {"cluster": cluster_names[j], "loading": float(loadings[j])} for j in order
                ],
            }
        )
    return {
        "n_pairs": len(records),
        "singular_values": [float(s) for s in singular],
        "energy_fractions": [float(e) for e in energy],
        "top3_energy": float(np.sum(energy[:3])),
        "components": components,
        "_vt": vt,
    }


def _principal_angles(vt_a: np.ndarray, vt_b: np.ndarray, k: int = 3) -> list[float]:
    qa, qb = vt_a[:k].T, vt_b[:k].T
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))) for s in sv]


def _fingerprint_correlation(records: list[dict[str, Any]]) -> dict[str, Any]:
    # Fingerprints: mean one-step probe delta per action over all first-leg
    # transitions at this checkpoint (each cluster appears as a first leg in
    # many pairs).
    deltas: dict[str, list[np.ndarray]] = {}
    for r in records:
        for name, ds in r["first_leg_deltas"].items():
            deltas.setdefault(name, []).extend(ds)
    fingerprints = {n: np.mean(np.stack(ds), axis=0) for n, ds in deltas.items()}

    dots, abs_dots, cosines, magnitudes = [], [], [], []
    for r in records:
        fa, fb = fingerprints[r["a"]], fingerprints[r["b"]]
        dot = float(np.dot(fa, fb))
        denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
        dots.append(dot)
        abs_dots.append(abs(dot))
        cosines.append(dot / denom if denom > 0 else 0.0)
        magnitudes.append(float(np.linalg.norm(r["mean_vector"])))

    def pearson(x: list[float]) -> float:
        return float(np.corrcoef(np.asarray(x), np.asarray(magnitudes))[0, 1])

    def spearman(x: list[float]) -> float:
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(magnitudes)).astype(np.float64)
        return float(np.corrcoef(rx, ry)[0, 1])

    return {
        "n_pairs": len(records),
        "pearson_dot": pearson(dots),
        "pearson_abs_dot": pearson(abs_dots),
        "pearson_cosine": pearson(cosines),
        "spearman_abs_dot": spearman(abs_dots),
    }


def _mse_by_bucket(
    predictor: ResidualMLPTransitionPredictor,
    observations: list[TransitionObservation],
    bucket_edges: list[int],
    action_filter: str | None = None,
) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for obs in observations:
        if action_filter is not None and obs.action.cluster_ids[0] != action_filter:
            continue
        step = int(obs.state_before.step)
        label = None
        for lo, hi in zip(bucket_edges, bucket_edges[1:], strict=False):
            if lo <= step < hi:
                label = f"{lo}-{hi}"
                break
        if label is None:
            continue
        pred = predictor.predict(obs.state_before, obs.action).probe_delta_mean
        err = float(np.mean((pred - np.asarray(obs.probe_delta, dtype=np.float64)) ** 2))
        buckets.setdefault(label, []).append(err)
    return {
        label: {"mse": float(np.mean(v)), "n": len(v)} for label, v in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--out", type=str, default="results/rung_b_reanalysis.json")
    args = parser.parse_args()

    output: dict[str, Any] = {}

    # --- Analyses 1 & 2 --------------------------------------------------
    field: dict[str, Any] = {}
    finger: dict[str, Any] = {}
    vts: dict[str, np.ndarray] = {}
    for stage, path in [("10k", "results/transitions/ct_10k.pkl"),
                        ("30k", "results/transitions/ct_30k.pkl")]:
        observations, metadata = load_transitions([path])
        records = _reconstruct_pairs(observations, repeats=2)
        rank = _field_rank(records)
        vts[stage] = rank.pop("_vt")
        field[stage] = rank
        finger[stage] = _fingerprint_correlation(records)
        del metadata
    field["cross_stage_principal_angles_top3_deg"] = _principal_angles(vts["10k"], vts["30k"])
    field["cross_stage_component_cosines"] = [
        float(abs(np.dot(vts["10k"][i], vts["30k"][i]))) for i in range(3)
    ]
    output["field_rank"] = field
    output["fingerprint_correlation"] = finger

    # Analyses 1-2 are cheap; persist them before the expensive MLP refit so
    # a slow or failed fit cannot lose them.
    partial_path = Path(args.out)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.with_suffix(".partial.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )

    # --- Analysis 3: stage-breaking --------------------------------------
    pool_paths = [
        "results/transitions/dh_300.pkl", "results/transitions/dh_3000.pkl",
        "results/transitions/dh_10k.pkl", "results/transitions/dh_30k.pkl",
        "results/transitions/ct_10k.pkl", "results/transitions/ct_30k.pkl",
        "results/transitions/lh_10k.pkl", "results/transitions/lh_30k.pkl",
    ]
    pool, pool_meta = load_transitions(pool_paths)
    hashes = {m["probe_suite_hash"] for m in pool_meta}
    if hashes != {probe_suite_hash()}:
        raise ValueError(f"probe suite drift across pools: {hashes}")
    predictor = ResidualMLPTransitionPredictor(
        all_cluster_names(), device=args.device, seed=0
    ).fit(pool, epochs=args.mlp_epochs, ranking_weight=0.1)

    reference, _ = load_transitions(["results/transitions/race_ref_s0.pkl"])
    divergent, _ = load_transitions([
        "results/transitions/race_comp_s10.pkl", "results/transitions/race_comp_s11.pkl",
        "results/transitions/race_compeps_s10.pkl", "results/transitions/race_compeps_s11.pkl",
    ])
    edges = [0, 1600, 3200, 4800, 6400, 8000, 9600]
    stage_breaking = {
        "uniform_reference": _mse_by_bucket(predictor, reference, edges),
        "divergent_races": _mse_by_bucket(predictor, divergent, edges),
        "uniform_reference_mixed_review_only": _mse_by_bucket(
            predictor, reference, edges, action_filter="mixed_review"
        ),
        "divergent_races_mixed_review_only": _mse_by_bucket(
            predictor, divergent, edges, action_filter="mixed_review"
        ),
        "note": (
            "Divergent pools are dominated by mixed_review transitions; only the"
            " action-matched (mixed_review-only) comparison isolates state"
            " divergence from action composition."
        ),
    }
    output["stage_breaking"] = stage_breaking

    output["provenance"] = {
        "config_hash": hashlib.sha256(
            json.dumps(vars(args), sort_keys=True).encode()
        ).hexdigest()[:16],
        "git_commit": _git_commit(),
        "probe_suite_hash": probe_suite_hash(),
        "simulator_version": "residual-mlp-v1",
        "environment_version": "real-v1",
        "note": "ct transitions exist only for 10k/30k; 300/3k predate --save-transitions",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(output)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _print_summary(output: dict[str, Any]) -> None:
    for stage in ("10k", "30k"):
        rank = output["field_rank"][stage]
        print(f"[field rank {stage}] top-3 energy {rank['top3_energy']:.3f}; "
              f"energies {[round(e, 3) for e in rank['energy_fractions'][:5]]}")
        for i, comp in enumerate(rank["components"][:3]):
            probes = ", ".join(
                f"{p['cluster']}({p['loading']:+.2f})" for p in comp["top_probes"][:3]
            )
            print(f"  PC{i + 1} ({comp['energy_fraction']:.1%}): {probes}")
        fc = output["fingerprint_correlation"][stage]
        print(f"[fingerprint {stage}] pearson dot={fc['pearson_dot']:+.3f} "
              f"|dot|={fc['pearson_abs_dot']:+.3f} cosine={fc['pearson_cosine']:+.3f} "
              f"spearman|dot|={fc['spearman_abs_dot']:+.3f}")
    print(f"[cross-stage] principal angles (deg): "
          f"{[round(a, 1) for a in output['field_rank']['cross_stage_principal_angles_top3_deg']]}; "
          f"component cosines: "
          f"{[round(c, 3) for c in output['field_rank']['cross_stage_component_cosines']]}")
    sb = output["stage_breaking"]
    print("[stage-breaking] bucket | uniform MSE (n) | divergent MSE (n) | matched-action ratio")
    for label in sb["uniform_reference"]:
        u = sb["uniform_reference"][label]
        d = sb["divergent_races"].get(label)
        um = sb["uniform_reference_mixed_review_only"].get(label)
        dm = sb["divergent_races_mixed_review_only"].get(label)
        ratio = (dm["mse"] / um["mse"]) if (um and dm and um["mse"] > 0) else None
        div_text = "-" if d is None else f"{d['mse']:.4f} ({d['n']:>5})"
        ratio_text = "-" if ratio is None else f"{ratio:.2f}x"
        print(f"  {label:>10} | {u['mse']:.4f} ({u['n']:>4}) | {div_text} | {ratio_text}")


if __name__ == "__main__":
    main()
