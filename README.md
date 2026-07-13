# Teachability Compiler

> Compile a raw corpus into a minimal **ordered teaching program** by predicting state-dependent learning effects and searching over curricula.

**Status:** private research foundation / pre-prototype.

## Core idea

Training data is not an unordered set of independent contributions. A data unit acts on the current learner state, and those actions generally do not commute:

\[
T_B(T_A(s)) \neq T_A(T_B(s)).
\]

Therefore, the problem is not merely to select a high-quality subset

\[
S^* = \arg\max_{S\subseteq D} U(S),
\]

but to find a compact, ordered, possibly repeated and interleaved curriculum

\[
\pi^*=(a_0,a_1,\ldots,a_{T-1})
\]

that drives a target model toward the useful learned state produced by a much larger corpus.

Teachability Compiler combines:

1. a compressed representation of the target model's learning state;
2. a small learned simulator of training transitions;
3. explicit measurement of constructive and destructive interference;
4. hierarchical Monte Carlo Tree Search over curriculum actions;
5. robust best-case / worst-case evaluation across seeds and simulator uncertainty;
6. periodic correction using real short-horizon target-model training branches.

The desired output is not simply a smaller dataset. It is a **model-specific executable curriculum**.

## One-line formulation

Given raw data \(D\), target learner \(M\), optimizer \(\mathcal A\), token budget \(B\), and evaluation map \(\Phi\), find an ordered teaching program \(\pi\) such that

\[
\pi^* = \arg\min_{\pi:\,\operatorname{cost}(\pi)\le B}
D_\Phi\!\left(s_T^{\pi},s_T^{D}\right)
+\lambda\,\operatorname{cost}(\pi)
+\mu\,\operatorname{risk}(\pi),
\]

where

\[
s_{t+1}^{\pi}=T(s_t^{\pi},a_t),
\qquad a_t\in\mathcal A_D,
\]

and \(\mathcal A_D\) contains data clusters, mixtures, repetitions, bridges, and replay actions derived from the corpus.

## Why this differs from data filtering

A static filter asks:

> Is this chunk intrinsically good?

Teachability Compiler asks:

> Given this model's current state and learning history, what transition would this chunk or batch cause, what future transitions would it unlock, and is that trajectory worth its cost?

A chunk can therefore be:

- useful early and redundant later;
- harmful immediately but prerequisite to a later gain;
- individually weak but strongly complementary with another chunk;
- high quality yet unnecessary because its learning effect is already covered;
- superficially redundant while creating a novel internal learning direction;
- valuable only when interleaved or replayed.

## Repository map

- [`docs/RESEARCH_SPEC.md`](docs/RESEARCH_SPEC.md) — full mathematical formulation and research hypotheses.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — proposed system components and data flow.
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) — staged experiments, baselines, metrics, and ablations.
- [`AGENTS.md`](AGENTS.md) — implementation brief for coding/research agents.
- [`src/teachability_compiler/`](src/teachability_compiler/) — typed foundation interfaces.
- [`tests/`](tests/) — initial mathematical sanity tests.

## Initial scope

The first prototype should deliberately be small:

- target model: 10M–100M parameter decoder;
- corpus: 20–100 learning-effect clusters;
- transition action: 1–20 optimizer steps on a cluster or mixture;
- simulator: small MLP / transformer predicting a distribution over compressed state deltas;
- search: hierarchical PUCT/MCTS;
- oracle: short real training branches from periodic target checkpoints.

The first success criterion is not state-of-the-art language modeling. It is evidence that searched curricula reach a fixed capability target with fewer tokens than random shuffling, static filtering, or greedy selection—and that the gain persists across seeds.

## Central hypotheses

1. **Learning effects are state-dependent.** A chunk has no single permanent utility score.
2. **Order effects are measurable.** A minority of action pairs have large functional commutators and dominate curriculum sensitivity.
3. **Transition space is lower-dimensional than text space.** Many different chunks induce nearly equivalent useful learning transitions.
4. **A small simulator can predict enough of that transition space to guide search.**
5. **Hierarchical search can exploit prerequisites, bridges, replay, spacing, and interleaving that greedy ranking cannot.**
6. **Robust curriculum optimization can compress teachability without merely overfitting one seed or one proxy model.**

## Non-goals

This project does not assume that all factual information can be removed from training data. Unique atomic facts must remain, be distilled into another carrier, or move to retrieval. The primary target is redundant representation-building and capability-teaching compute.

## Working project name

**Teachability Compiler** describes the intended system: raw corpus in, compact executable teaching program out.
