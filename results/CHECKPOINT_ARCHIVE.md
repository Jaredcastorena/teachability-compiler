# Checkpoint archive record

Git is the primary record: all trajectory JSONs, manifests, and provenance
hashes live in this repo. Model checkpoints are too large for GitHub
(3.1 GB > 2 GB LFS cap), so weights are archived out-of-band and pinned
here by hash.

## Archived weights

| file | sha256 | location | status |
|---|---|---|---|
| reference s* (`target.pt`, 1B-token proportional shuffle, seed 0) | `64dcae57374bdec8ae18faa97359c6e250e991bfb603b9270a382b0ad2d2bb98` | hf.co/Jared1728/teachability-compiler-checkpoints `reference_s_star/target.pt` (LFS sha256 verified identical) + local copy | ARCHIVED |

## Retention policy

- Per-run `latest.pt` files are crash insurance only: deleted when their
  run completes (uniform, edu_heavy, reference already cleaned).
- Race harness keeps exactly one atomic checkpoint per active run.
- Candidates for archive on completion: final models of winning policies.
- Offsite archive live: private Hugging Face repo
  (`Jared1728/teachability-compiler-checkpoints`); upload-one, verify
  hash, then local copy becomes evictable.
