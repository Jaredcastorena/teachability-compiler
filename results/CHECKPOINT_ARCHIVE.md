# Checkpoint archive record

Git is the primary record: all trajectory JSONs, manifests, and provenance
hashes live in this repo. Model checkpoints are too large for GitHub
(3.1 GB > 2 GB LFS cap), so weights are archived out-of-band and pinned
here by hash.

## Archived weights

All seven final checkpoints from the 600M-token seed-10 race (plus the
1B-token seed-0 reference) are archived offsite at
hf.co/Jared1728/teachability-compiler-checkpoints. Every upload was
verified byte-for-byte against its local sha256 via HF's LFS-recorded
hash before the local copy was deleted. No checkpoints remain on disk.

| file | sha256 | HF path |
|---|---|---|
| reference s* (1B tok, seed 0) | `64dcae57374bdec8ae18faa97359c6e250e991bfb603b9270a382b0ad2d2bb98` | `reference_s_star/target.pt` |
| uniform (seed 10) | `2fc9c89cecd505c08d9904061be028fde271fe49019e9d7fd206b3d904897bfd` | `race_uniform_seed10/target.pt` |
| edu_heavy (seed 10) | `68a5c2007fad9a3c0ada24efec65145bc4d561323340f89bea700b94624effb6` | `race_edu_heavy_seed10/target.pt` |
| worst_probe (seed 10) | `736764ef794e254c6db6fd461ba960a41d01397e61610293ac2a05e1673d33a2` | `race_worst_probe_seed10/latest.pt` |
| compiler (seed 10) | `706e48ec3ee97833efbe2c40c102462f245a02458fd92b9f6e891fb0ddf16d11` | `race_compiler_seed10/latest.pt` |
| staged (seed 10) | `5b85de63039fedfdb6fb1364cf62fdeae6af7e24f6c42dcbb5041d2fdd2bb16f` | `race_staged_seed10/latest.pt` |
| proportional (seed 10) | `c71193ecb0b43024e7cabf72eb1c767b1d5ed70f8b3742bad1bd1d91b0e4f0f2` | `race_proportional_seed10/latest.pt` |
| proportional (seed 10, final) | `2e978393433c6c3ba76d58e1fa1e7f2f70b9dbf48fc5d2779ccbe919e5393663` | `race_proportional_seed10/target.pt` |

## Retention policy

- Git is the primary record for everything except weights: trajectory
  JSONs, manifests, and provenance hashes all live in this repo.
- Every checkpoint is uploaded to the private HF repo, hash-verified,
  then deleted locally (`data/checkpoints/` is empty by default).
- To re-run local evals or resume a policy, re-download the relevant
  file from the HF repo above; verify its sha256 against this table
  before trusting it.
