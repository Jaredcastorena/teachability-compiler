# Rung A + B Report

Seeds 10/11/12 (races), seed 0 (reference + measurement checkpoints). All race
numbers are hidden-channel judged. Policy accounting = race tokens + bootstrap;
instrument (commutator-panel) tokens are reported separately. Probe suite
verified identical across every artifact via `probe_suite_hash`.

## Rung A — staged policy v0

Staged = compiler-greedy opening (residual MLP, ε=0.1 exploration) → switch on
predicted-gain EMA < 0.002 (min 100 actions) → damped loss_greedy coverage
(recency penalty 0.3, decay 0.7). Switches fired at actions 106 / 100 / 190.

### Tokens to hidden-mean threshold (policy accounting)

| threshold | random (s10/11/12) | ε-compiler (s10/11/12) | staged (s10/11/12) |
|---|---|---|---|
| 2.0 | 5.5 / 6.3 / 4.7M | 2.6 / 2.4 / 2.6M | **2.6 / 2.4 / 2.6M** |
| 1.6 | 11.0 / 13.1 / 11.5M | 15.7 / 17.0 / 14.9M | **13.9 / 11.3 / 12.6M** |
| 1.45 | 17.0 / 19.1 / 17.3M | 24.6 / 24.1 / 21.0M | 24.6 / 20.7 / 21.0M |

Final hidden means: staged 1.356 / 1.330 / 1.277 vs random 1.335 / 1.276 / 1.357.

- Registered prediction (13–16M to 1.45): **missed** (20.7–24.6M), but the
  window captured the 1.6 crossing (11.3–13.9M), where staged now ties or beats
  random — which the ε-compiler never did.
- Staged dominates to ~1.6 (2×+ at 2.0), then random's implicit interleaving
  retains the deep tail. Damped coverage fixed the loss_greedy oscillation floor
  (plain loss_greedy stalled at 1.46–1.50 final) but is not a tail policy.
  The tail residue is the Phase 3 search claim.

### Switch-trigger comparison (post hoc)

- Pragmatic (predicted-gain EMA): actions 106 / 100 / 190.
- Velocity pulse (realized-gain EMA < 10% of peak): ~32 on all seeds — too
  early; the opening still delivered through ~100.
- Commutator panel (3 pairs / 150 actions, ~4 % token overhead): seed 12,
  still in its monoculture opening at the first panel event, measured 5.04;
  seeds already in coverage measured 0.19–0.35. Interference is high during
  the opening and collapses once coverage diversifies — directionally correct
  as a trigger, granularity-limited at this cadence, and cleanly causal as an
  instrument.

## Rung B — three reanalyses from saved transitions

Commutator transitions exist only for 10k/30k (300/3k predate persistence),
so analyses 1–2 cover those stages.

### 1. Field rank (SVD of coordinate-resolved commutator vectors, 190 pairs)

| stage | top-3 energy | PC1 | PC2 |
|---|---|---|---|
| 10k | **84.6 %** (PC1 69.2 %) | arithmetic (mod_arith, add_2digit, add_1digit) | syntax/shift (bracket_depth, letter_shift[_conflict]) |
| 30k | **81.5 %** (PC1 54.2 %) | arithmetic (same) | syntax (bracket_depth, count_chars, bracket_match) |

Registered call ">70 % in top-3": **confirmed**. Identity rotation 10k→30k:
**not observed** — principal angles 12°/23°/49°, component cosines
0.86/0.77/0.63. The field crystallizes by mid-training; rotation (per the
turnover Jaccard 0.027) is an early-training phenomenon. Untestable for
300→3k (transitions unsaved).

### 2. Fingerprint→commutator correlation

| stage | Pearson |Δ_A·Δ_B| vs |C| | Spearman | cosine |
|---|---|---|---|
| 10k | **+0.481** | +0.724 | −0.187 |
| 30k | **+0.343** | +0.729 | −0.149 |

Registered 0.3–0.6: **confirmed** at both stages. The signal is carried by
fingerprint magnitudes (big movers collide); direction-normalized alignment is
uninformative-to-inverse. Informative prior, irreducible second-order remainder.

### 3. Stage-breaking test (E_stage gate)

Residual MLP (trained on uniform-pretraining pools, stage = log step count)
scored on race transitions vs uniform-reference transitions at matched steps.
Action-matched ratio = divergent MSE / uniform MSE on mixed_review transitions
only (controls for the races' action composition).

| step bucket | matched-action MSE ratio |
|---|---|
| 0–1600 | 1.20× |
| 1600–3200 | 6.76× |
| 3200–4800 | 8.71× |
| 4800–6400 | 13.10× |
| 6400–8000 | 15.98× |
| 8000–9600 | **22.73×** |

Chronological stage is **measurably breaking**, monotonically in divergence
duration: at matched step counts the simulator is up to 22.7× worse on
developmentally divergent states. τ-as-step-count is dead outside the earliest
window. Per the pre-registered design caution, the next step is τ v1 =
(tokens consumed, EMA of probe-delta magnitudes) — not a learned encoder —
with this same diagnostic as its acceptance test; a learned E_stage is
justified only if τ v1 leaves this ratio on the table.

## Overhead this rung

Races: 5 runs × 39.3M race tokens + 4.7M overhead each (staged/eps incl.
panel), replication 2 × 39.3M. Reanalysis: zero new oracle transitions
(one MLP refit, 100 epochs, cuda:0).
