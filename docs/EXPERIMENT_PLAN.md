# Experiment Plan

## Phase 0 — Demonstrate noncommutativity

### Goal

Show that order effects are measurable and predictable in a controlled learner.

### Setup

- tiny MLP or transformer;
- synthetic tasks with known prerequisites and contradictions;
- 20–50 data clusters;
- multiple initialization seeds;
- one to ten optimizer steps per action.

### Measurements

For every sampled pair \((A,B)\):

\[
C_\Phi(A,B\mid s)
=
\|\Phi(T_B(T_A(s)))-\Phi(T_A(T_B(s)))\|_W.
\]

Record the directed value difference

\[
M_{AB}(s)=V(T_B(T_A(s)))-V(T_A(T_B(s))).
\]

### Success condition

- repeatable order-sensitive pairs exist;
- functional commutators are not explained solely by noise;
- a simple predictor beats a constant or random baseline at ranking high-commutator pairs.

---

## Phase 1 — Learn one-step transition effects

### Goal

Predict compressed state changes from current state and action.

### Baselines

- action-only average effect;
- semantic embedding regression;
- per-example loss;
- gradient norm;
- gradient alignment to evaluation probes;
- nearest-neighbor transition lookup.

### Metrics

- mean squared error of probe deltas;
- rank correlation of action value;
- sign accuracy for beneficial vs harmful anchor changes;
- uncertainty calibration;
- top-k action recall;
- out-of-state generalization.

### Critical ablation

Compare

\[
F(a)
\quad\text{vs}\quad
F(s,a).
\]

The state-conditioned model must win materially or the central hypothesis is unsupported.

---

## Phase 2 — Greedy receding-horizon compiler

### Goal

Use the simulator to select one action at a time, periodically correcting it with real execution.

### Baselines

- random shuffled corpus;
- static quality filter;
- loss-proportional sampling;
- gradient-norm selection;
- semantic diversity selection;
- static influence ranking;
- oracle immediate-gain greedy policy.

### Metrics

- tokens to reach target validation loss;
- tokens to reach a capability threshold;
- final capability at fixed tokens;
- forgetting area under curve;
- overhead-adjusted wall-clock/FLOPs;
- simulator error over training time.

This phase tests whether a transition model is useful before adding tree search.

---

## Phase 3 — MCTS for delayed value

### Goal

Construct datasets where greedy immediate gain is provably or empirically suboptimal.

Task patterns:

- prerequisite: \(A\) unlocks \(B\);
- bridge: \(A\rightarrow X\rightarrow B\) succeeds while direct paths fail;
- replay: \(A\rightarrow B\rightarrow A\) outperforms either order;
- interleaving: mixed exposure prevents destructive specialization;
- correction: new trusted data should overwrite an old false rule;
- ambiguity separation: apparently conflicting examples become compatible after a disambiguating feature is learned.

### Comparison

- greedy simulator policy;
- beam search;
- dynamic programming where tractable;
- vanilla MCTS;
- policy/value-guided PUCT;
- robust MCTS.

### Success condition

MCTS must discover curricula with superior terminal performance despite including steps with weak or negative immediate reward.

---

## Phase 4 — Teachability compression on a tiny language model

### Setup

- target: 10M–100M decoder;
- corpus: 10M–100M raw tokens, initially clustered into 20–200 actions;
- fixed tokenizer and architecture;
- full-data reference runs across at least five seeds;
- hidden evaluation suite spanning language modeling, syntax, factual recall, arithmetic, and code-like patterns.

### Primary metric

For tolerance \(\epsilon\), estimate

\[
B^*(\epsilon)
=
\min_\pi c(\pi)
\quad\text{s.t.}\quad
D_\Phi(s_T^\pi,s_T^D)\le\epsilon.
\]

Report

\[
\operatorname{CR}(\epsilon)=\frac{c(D)}{B^*(\epsilon)}.
\]

### Required reporting

- mean and worst-seed performance;
- total search + oracle + target compute;
- factual retention separately from general capability;
- simulator calibration by training stage;
- exact amount of repeated data in the compiled curriculum;
- transfer to unseen seeds and at least one nearby model size.

---

## Phase 5 — Robustness game

Optimize either

\[
\max_\pi\min_\xi J(\pi;\xi)
\]

or

\[
\max_\pi\operatorname{CVaR}_\alpha[J(\pi;\xi)].
\]

Compare curricula optimized for:

- mean return;
- lower-tail return;
- architecture family robustness;
- simulator ensemble robustness.

A robust curriculum should have smaller seed variance and fewer catastrophic failures, even if its best-seed score is lower.

---

## Phase 6 — Scale and transfer

Questions:

1. Does a curriculum learned for 20M parameters help at 50M or 100M?
2. Does ordering transfer while exact chunk selection does not?
3. Are commutator-heavy pairs stable across scale?
4. Does conditioning on architecture descriptors improve transfer?
5. How often must the real target correct the simulator?

Do not claim general pretraining savings until overhead-adjusted gains survive this phase.

---

## Ablation matrix

Remove one component at a time:

- no learner state, action-only predictor;
- no optimizer sketch;
- no recent-history embedding;
- no activation sketch;
- no uncertainty ensemble;
- no real-model correction;
- no MCTS, greedy only;
- no adversary/risk term;
- no replay/interleaving actions;
- semantic clusters instead of effect clusters;
- parameter-space commutator instead of functional commutator;
- terminal reward only vs trajectory matching.

---

## Negative-result criteria

The project should be considered unsuccessful in its current form if:

- simulator ranking does not beat cheap static heuristics;
- search overhead exceeds saved target compute at small scale with no scaling trend;
- MCTS gains disappear under new seeds;
- curricula exploit probes but fail hidden evaluations;
- state-conditioned prediction offers no gain over action-only scoring;
- transfer across nearby target checkpoints is too poor for practical replanning intervals.

These outcomes should trigger simplification or reframing rather than adding complexity blindly.
