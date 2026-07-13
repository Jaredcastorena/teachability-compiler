# Agent Implementation Brief

## Mission

Build **Teachability Compiler**, a research system that compiles a raw corpus into a compact ordered curriculum by modeling state-dependent, noncommutative learning transitions and planning with hierarchical MCTS.

## Non-negotiable conceptual constraints

1. Do not assign a permanent context-free utility score to a data chunk.
2. Treat utility as conditional: \(U(a\mid s,h)\).
3. Preserve optimizer state and recent history in transition experiments.
4. Prefer functional/behavioral change measurements over raw parameter distance.
5. Distinguish destructive forgetting from justified correction using trusted probes.
6. Search over repetition, replay, interleaving, and bridge actions—not only permutations of unique examples.
7. Report total overhead, including oracle branches and search compute.
8. Never evaluate only on the probes visible to the planner.
9. Require multiple target seeds before accepting a curriculum gain.
10. Keep the first implementation small and falsifiable.

## Canonical equations

True transition:

\[
x_{t+1}=\mathcal T(x_t,a_t,\xi_t).
\]

Learned simulator:

\[
F_\psi(s_t,a_t)\rightarrow p(\Delta s_t,y_t\mid s_t,a_t).
\]

Functional commutator:

\[
C_\Phi(A,B\mid s)
=
\|\Phi(T_B(T_A(s)))-\Phi(T_A(T_B(s)))\|_W.
\]

Robust planning objective:

\[
\pi^*=\arg\max_\pi\operatorname{CVaR}_\alpha[J(\pi;\xi)].
\]

Budgeted compression objective:

\[
\pi_B^*
=
\arg\min_{\pi:c(\pi)\le B}
D_\Phi(s_T^\pi,s_T^D).
\]

## First implementation milestone

Implement a synthetic environment with:

- a small learner;
- 20 named data clusters;
- at least one prerequisite pair;
- one destructive pair;
- one bridge sequence;
- one replay-sensitive sequence;
- oracle transition generation;
- a state-conditioned transition predictor;
- greedy and MCTS curriculum policies;
- multi-seed evaluation.

The milestone is complete only when a test demonstrates that a sequence with lower immediate reward can produce higher terminal value and MCTS discovers it more often than greedy selection.

## Engineering principles

- Typed interfaces and deterministic tests.
- Configuration-driven experiments.
- Immutable transition records.
- Every result records seed, config hash, git commit, target checkpoint hash, and simulator version.
- No hidden fallback from real transitions to simulated transitions.
- Fail loudly when dimensions, probe versions, or state encoders mismatch.
- Keep algorithms replaceable behind protocols.
