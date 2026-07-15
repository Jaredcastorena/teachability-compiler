# Teachability Compiler

> Compile a raw corpus into a minimal **ordered teaching program** by predicting state-dependent learning effects and searching over curricula.

**Status:** controlled real-learner prototype. The measurement phase and the first staged-policy compiler rung are complete; delayed-value search is the current research gate.

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

The desired output is not simply a smaller dataset. It is a **model-specific executable curriculum**.

## Current empirical status

The project has moved beyond the synthetic proof of concept to a 9.92M-parameter decoder with 20 formal-task clusters, fixed visible and hidden evaluation channels, real optimizer-preserving learner branches, and full provenance and token accounting.

### What is established

- **Learning effects are state-dependent.** A nonlinear residual transition model materially beats an action-only model on held-out transition prediction: one-step MSE `0.47` vs `1.21`, top-3 recall `0.63` vs `0.37`, and Kendall-τ `0.52` vs `0.40`.
- **Order effects are real and repeatable.** Functional commutators remain far above measured noise and recover stable within-stage structure.
- **Interference is stage-windowed rather than permanently sparse.** Top-decile commutator share rises `18.5% → 28.1% → 35.7%` and then recedes to `31.2%`; the strongest concentration appears near the active learning frontier.
- **The commutator field is strongly low-rank at 10k and 30k.** Its top three singular directions explain `84.6%` and `81.5%` of field energy, with persistent arithmetic and syntax axes.
- **One-step greedy control is insufficient.** A pure compiler correctly identifies the immediate `mixed_review` winner, then collapses into monoculture, plateaus, and forgets.
- **Exploration repairs pool collapse but not delayed-value myopia.** An ε-greedy compiler avoids catastrophic collapse but remains behind random interleaving late in training.
- **A staged policy produces real threshold compression.** Compiler-guided opening followed by damped deficit coverage reaches hidden mean `2.0` in `2.4–2.6M` tokens versus random's `4.7–6.3M`, and is competitive around hidden mean `1.6`.
- **The deep tail remains unresolved.** At hidden mean `1.45`, random reaches the threshold in `17.0–19.1M` tokens while the staged policy needs `20.7–24.6M`. The remaining problem is useful delayed ordering and interleaving, not merely coverage.
- **Chronological step count is not a sufficient developmental coordinate.** On curriculum-divergent states at matched step counts, residual-MLP error rises monotonically to `22.73×` the uniform-trajectory control.

The current evidence supports a three-regime picture:

\[
\text{diffuse early plasticity}
\rightarrow
\text{concentrated learning frontier}
\rightarrow
\text{weaker but persistent late structure}.
\]

The compiler can exploit the opening regime and detect when a fixed greedy policy has become stale. It does not yet know the best fine-grained sequence for the late regime.

See [`results/RUNG_AB_REPORT.md`](results/RUNG_AB_REPORT.md) for the consolidated Rung A+B tables and analysis.

## Current research verdict

| Hypothesis | Verdict |
|---|---|
| Learning effects depend on learner state | **Confirmed** |
| Functional order effects are measurable | **Confirmed** |
| A minority of pairs always dominates | **Revised:** concentration is stage-windowed and peaks near the learning frontier |
| Transition geometry is lower-dimensional than raw text space | **Supported:** the mature commutator field is strongly low-rank |
| A small simulator can guide curriculum choice | **Partially confirmed:** useful locally and early with the right nonlinear model class |
| Chronological stage is an adequate simulator coordinate | **Rejected off-trajectory** |
| Greedy immediate gain is a sufficient teaching policy | **Rejected** |
| A staged controller can compress teaching | **Partially confirmed:** strong early gain, parity near the middle, no deep-tail win yet |
| Multi-step search can beat random interleaving on the real learner | **Open** |
| A globally minimal ordered teaching program has been found | **Not yet** |

## Implemented system

The repository currently includes:

1. a controlled synthetic learner with known prerequisites, bridges, destructive pairs, replay effects, and an MCTS-vs-greedy demonstration;
2. a 9.92M-parameter real decoder learner over 20 formal-task clusters;
3. optimizer-state-preserving snapshots and real short-horizon oracle branches;
4. fixed visible probes and a structurally isolated hidden evaluation channel;
5. functional commutator measurement with replica noise floors and persistence checks;
6. decision-validity metrics for top-1, top-3, ranking correlation, and regret;
7. a pooled residual MLP transition predictor with action means, state-conditioned residuals, stage features, and ranking loss;
8. random, mixed-review, loss-greedy, pure compiler, ε-greedy compiler, and staged compiler policies;
9. switch-trigger instrumentation and in-race commutator panels;
10. low-rank field, transition-fingerprint, and stage-breaking reanalyses;
11. full result JSONs, probe-suite hashes, provenance, and oracle-token accounting.

## Current research gate

The next system must solve the problem exposed by the staged races:

\[
\arg\max_a r(s,a) \neq \arg\max_a Q(s,a).
\]

The local simulator can identify high-value immediate teaching actions, but the late regime requires delayed value, useful interleaving, and actions whose benefit may only appear after subsequent training.

Current priorities are:

1. replace raw chronological stage with a developmental coordinate grounded in learner movement, beginning with `(tokens consumed, EMA of probe-delta magnitude)`;
2. preserve real-execution correction while adding short-horizon beam/MCTS search;
3. use low-rank commutator structure and action fingerprints as search priors rather than treating all action pairs equally;
4. test whether planned late-stage sequences beat random interleaving at equal oracle and learner cost;
5. distinguish aggregate threshold compression from attainment of the full reference probe vector.

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

- [`docs/RESEARCH_SPEC.md`](docs/RESEARCH_SPEC.md) — mathematical formulation and research hypotheses.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — proposed system components and data flow.
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) — staged experiments, baselines, metrics, and ablations.
- [`AGENTS.md`](AGENTS.md) — implementation brief and frozen research contracts.
- [`src/teachability_compiler/`](src/teachability_compiler/) — typed simulator, planner, real learner, and experiment implementations.
- [`tests/`](tests/) — synthetic, persistence, simulator, decision-validity, and compiler smoke tests.
- [`results/RUNG_AB_REPORT.md`](results/RUNG_AB_REPORT.md) — current consolidated result narrative.
- [`results/rung_b_reanalysis.json`](results/rung_b_reanalysis.json) — low-rank field, fingerprint, and stage-breaking measurements.
- [`results/`](results/) — committed race, commutator, horizon, and evaluation artifacts.

## Current experimental scope

- target model: 9.92M-parameter decoder;
- action vocabulary: 20 formal-task clusters and mixtures;
- transition action: 8 optimizer steps on one cluster or mixture;
- simulator: residual MLP predicting compressed probe-state transitions;
- controller: one-step closed-loop policies plus a staged compiler-to-coverage controller;
- evaluation: fixed visible probes, isolated hidden probes, multi-seed threshold races;
- search target: receding-horizon beam/MCTS with real oracle correction.

The first complete success criterion remains:

> Reach the full fixed capability target with fewer total learner and oracle tokens than random shuffling, static filtering, or greedy selection, with the gain persisting across seeds.

Current progress is narrower but positive: the compiler demonstrates replicated early threshold compression and a useful state-dependent regime switch, while random interleaving still wins the deepest measured tail.

## Non-goals

This project does not assume that all factual information can be removed from training data. Unique atomic facts must remain, be distilled into another carrier, or move to retrieval. The primary target is redundant representation-building and capability-teaching compute.

## Working project name

**Teachability Compiler** describes the intended system: raw corpus in, compact executable teaching program out.
