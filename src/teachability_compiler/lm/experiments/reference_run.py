"""Proportional-shuffle reference trajectory for the language-model rung.

Trains the nanochat learner on the natural corpus mixture (each action sampled
in proportion to its train-token count) for a fixed token budget, recording the
hidden channel (validation bits-per-byte + holdout cross-entropy) densely along
the way. The final checkpoint defines the target state s* that curriculum
policies must reach with fewer tokens.

Run from the nanochat venv (has nanochat on the path):

    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -m teachability_compiler.lm.experiments.reference_run \
        --tokens-dir data/tokens --token-budget 1_000_000_000 --device cuda:0

Resumable: pass --resume to restart from the latest checkpoint after a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from teachability_compiler.lm.lm_oracle import NanochatLearnerOracle
from teachability_compiler.state import CurriculumAction


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_hash(args: argparse.Namespace) -> str:
    payload = json.dumps(vars(args), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _proportional_action(oracle: NanochatLearnerOracle, steps: int, tokens_per_step: int
                         ) -> CurriculumAction:
    """All 24 actions, weighted by their train-token share (natural mixture)."""
    names = list(oracle.action_names)
    weights = np.asarray(
        [float(oracle._train_tokens[name].shape[0]) for name in names], dtype=np.float64
    )
    weights = weights / weights.sum()
    return CurriculumAction(
        cluster_ids=tuple(names),
        mixture_weights=tuple(float(w) for w in weights),
        optimizer_steps=steps,
        token_budget=steps * tokens_per_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-dir", default="data/tokens")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-budget", type=int, default=1_000_000_000)
    parser.add_argument("--steps-per-chunk", type=int, default=200)
    parser.add_argument("--device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/lm_reference_seed0.json")
    parser.add_argument("--ckpt-dir", default="data/checkpoints/reference")
    parser.add_argument("--ckpt-every-chunks", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"

    oracle = NanochatLearnerOracle(
        tokens_dir=args.tokens_dir,
        depth=args.depth,
        device=args.device,
        seq_len=args.seq_len,
        device_batch=args.device_batch,
        grad_accum=args.grad_accum,
        base_seed=args.seed,
    )
    tokens_per_step = args.device_batch * args.grad_accum * args.seq_len

    trajectory: list[dict[str, Any]] = []
    chunk_index = 0
    if args.resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        oracle.restore(blob["snapshot"])
        trajectory = blob["trajectory"]
        chunk_index = int(blob["chunk_index"])
        print(f"resumed at chunk {chunk_index}, tokens_seen {oracle.tokens_seen:,}")

    action = _proportional_action(oracle, args.steps_per_chunk, tokens_per_step)
    provenance = {
        "seed": args.seed,
        "config_hash": _config_hash(args),
        "git_commit": _git_commit(),
        "tokens_manifest_hash": oracle.manifest.get("manifest_hash"),
        "actions_manifest_hash": oracle.manifest.get("actions_manifest_hash"),
        "environment_version": "lm-v1",
        "depth": args.depth,
    }

    wall_start = time.time()
    if not trajectory:
        # Record the untrained baseline point.
        trajectory.append({
            "chunk": 0, "tokens": 0, "step": 0,
            "val_bpb": oracle.hidden_val_bpb(), "holdout_ce": oracle.hidden_holdout_ce(),
            "probe_mean": float(np.mean(oracle.probe_losses())),
        })

    while oracle.tokens_seen < args.token_budget:
        chunk_index += 1
        oracle.apply_action(action, data_seed=args.seed * 1_000_003 + chunk_index)
        val_bpb = oracle.hidden_val_bpb()
        holdout_ce = oracle.hidden_holdout_ce()
        probe_mean = float(np.mean(oracle.probe_losses()))
        trajectory.append({
            "chunk": chunk_index, "tokens": int(oracle.tokens_seen), "step": int(oracle.step),
            "val_bpb": val_bpb, "holdout_ce": holdout_ce, "probe_mean": probe_mean,
        })
        _atomic_write_json(out_path, {
            "kind": "reference_trajectory", "policy": "proportional_shuffle",
            "token_budget": args.token_budget, "trajectory": trajectory,
            "provenance": provenance,
            "wall_seconds": time.time() - wall_start,
        })
        if chunk_index % args.ckpt_every_chunks == 0:
            torch.save(
                {"snapshot": oracle.snapshot(), "trajectory": trajectory,
                 "chunk_index": chunk_index},
                ckpt_path,
            )
        print(f"[reference] chunk {chunk_index} tokens {oracle.tokens_seen/1e6:.0f}M "
              f"val_bpb {val_bpb:.4f} holdout_ce {holdout_ce:.4f} "
              f"({(time.time()-wall_start)/60:.0f} min)", flush=True)

    final = trajectory[-1]
    torch.save(
        {"snapshot": oracle.snapshot(), "trajectory": trajectory, "chunk_index": chunk_index},
        ckpt_dir / "target.pt",
    )
    _atomic_write_json(out_path, {
        "kind": "reference_trajectory", "policy": "proportional_shuffle",
        "token_budget": args.token_budget, "trajectory": trajectory,
        "target": {
            "tokens": final["tokens"], "val_bpb": final["val_bpb"],
            "holdout_ce": final["holdout_ce"], "probe_mean": final["probe_mean"],
            "target_probe_losses": oracle.probe_losses().tolist(),
        },
        "provenance": provenance,
        "wall_seconds": time.time() - wall_start,
    })
    print(f"\nTARGET s*: val_bpb {final['val_bpb']:.4f} holdout_ce {final['holdout_ce']:.4f} "
          f"at {final['tokens']/1e9:.2f}B tokens in {(time.time()-wall_start)/3600:.1f}h")


if __name__ == "__main__":
    main()
