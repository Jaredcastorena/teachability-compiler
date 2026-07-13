# Teachability Compiler — Research Specification

## 1. Thesis

A training corpus is not a bag of fixed educational values. Each data exposure is a state-dependent operator acting on a learner whose parameters, optimizer state, representations, and recent history were created by previous exposures.

The same data units can therefore produce different outcomes under different orderings:

\[
T_B\circ T_A \neq T_A\circ T_B.
\]

The research objective is to learn a cheap model of those transitions and use sequential search to construct a smaller ordered curriculum that preserves the useful teachability of a much larger corpus.

This is **training-trajectory compression**, not merely dataset pruning.

---

## 2. Learner dynamics

Let the complete learner state be

\[
x_t=(\theta_t,o_t,h_t),
\]

where:

- \(\theta_t\) are model parameters;
- \(o_t\) is optimizer state, including momentum and adaptive moments;
- \(h_t\) is relevant training history not captured by \((\theta_t,o_t)\), such as recent data exposures or replay statistics.

A curriculum action \(a_t\) can be one chunk, a batch, a cluster, a domain mixture, a replay operation, or a short training recipe. The true transition is

\[
x_{t+1}=\mathcal T(x_t,a_t,\xi_t),
\]

where \(\xi_t\) contains stochasticity such as dropout, batch realization, numerical nondeterminism, augmentation, and sampling noise.

For SGD on a batch \(B_t\),

\[
\theta_{t+1}=\theta_t-\eta_t g_{B_t}(\theta_t),
\qquad
g_{B_t}(\theta)=\nabla_\theta L(B_t;\theta).
\]

For Adam-like optimizers, \(\mathcal T\) must also update first and second moments, making order dependence stronger than the parameter-only equation suggests.

---

## 3. Why order matters

For two training units \(A\) and \(B\), one SGD step on each gives

\[
\theta_{AB}
=
\theta-\eta g_A(\theta)
-\eta g_B\!\left(\theta-\eta g_A(\theta)\right),
\]

and

\[
\theta_{BA}
=
\theta-\eta g_B(\theta)
-\eta g_A\!\left(\theta-\eta g_B(\theta)\right).
\]

Using a first-order Taylor expansion of the second gradient,

\[
\theta_{AB}
\approx
\theta-\eta(g_A+g_B)+\eta^2H_Bg_A,
\]

\[
\theta_{BA}
\approx
\theta-\eta(g_B+g_A)+\eta^2H_Ag_B,
\]

so

\[
\theta_{AB}-\theta_{BA}
\approx
\eta^2(H_Bg_A-H_Ag_B).
\]

Even when the first-order sum is identical, curvature causes the learning operators to fail to commute.

### 3.1 Parameter-space commutator

Define

\[
[\mathcal T_A,\mathcal T_B](x)
=
\mathcal T_B(\mathcal T_A(x))
-
\mathcal T_A(\mathcal T_B(x)).
\]

A raw parameter norm is

\[
C_{\theta}(A,B\mid x)
=
\left\|
\theta_{AB}-\theta_{BA}
\right\|_2.
\]

This is easy to define but can be misleading because networks possess parameter symmetries and reparameterizations.

### 3.2 Functional commutator

Let \(\Phi(x)\in\mathbb R^d\) be a behavioral probe map containing validation losses, logits, capability scores, calibration measures, and selected activation statistics. Define

\[
C_{\Phi}(A,B\mid x)
=
\left\|
\Phi(\mathcal T_B(\mathcal T_A(x)))
-
\Phi(\mathcal T_A(\mathcal T_B(x)))
\right\|_W,
\]

with weighted norm

\[
\|v\|_W=\sqrt{v^\top Wv}.
\]

This measures whether order changes what the model does rather than merely where it sits in parameter coordinates.

### 3.3 Directed order advantage

For scalar value function \(V\), define

\[
M_{AB}(x)
=
V(\mathcal T_B(\mathcal T_A(x)))
-
V(\mathcal T_A(\mathcal T_B(x))).
\]

If \(M_{AB}(x)>0\), then \(A\rightarrow B\) is preferred to \(B\rightarrow A\) at state \(x\). Because the sign can change during training, prerequisite relations are state-dependent rather than permanent edges.

---

## 4. Interference as a measurable learning signal

For an old probe example \(z_j\) and a new training unit \(a\), let

\[
\Delta\theta_a=\theta'-\theta.
\]

The old loss changes approximately as

\[
\Delta L_j
=
L_j(\theta+\Delta\theta_a)-L_j(\theta)
\approx
g_j^\top\Delta\theta_a
+
\frac{1}{2}\Delta\theta_a^\top H_j\Delta\theta_a.
\]

The first term captures local gradient alignment. The second captures curvature and state reorganization.

Define an interference vector over an anchor set \(\mathcal P=\{z_1,\dots,z_m\}\):

\[
i(a\mid x)
=
[\Delta L_1,\ldots,\Delta L_m]^\top.
\]

Negative components indicate improvement on the corresponding anchor loss; positive components indicate damage.

A new action can therefore be:

- **constructive:** improves old probes and the target objective;
- **destructive but corrective:** harms obsolete behavior while improving trusted truth;
- **destructive and unjustified:** catastrophic forgetting or representation collision;
- **orthogonal/novel:** introduces a useful direction with little old-probe movement;
- **redundant:** reproduces a transition already covered by prior actions.

---

## 5. Compressed learning state

The planner cannot carry the full target model in every search node. Define a compressed state encoder

\[
s_t=E(x_t)\in\mathbb R^k,
\qquad k\ll |\theta|.
\]

A candidate state representation may concatenate:

\[
s_t=
\begin{bmatrix}
\ell_t^{\text{probe}}\\
P_\theta\theta_t\\
P_g\bar g_t\\
P_a A_t\\
q(o_t)\\
e_t^{\text{domain}}\\
r_t^{\text{history}}
\end{bmatrix},
\]

where:

- \(\ell_t^{\text{probe}}\): losses and scores on persistent capability anchors;
- \(P_\theta\theta_t\): optional random/learned parameter sketch;
- \(P_g\bar g_t\): projected gradient or update-subspace summary;
- \(P_a A_t\): projected activation/covariance statistics;
- \(q(o_t)\): compressed optimizer moments;
- \(e_t^{\text{domain}}\): exposure counts and recency by domain/effect cluster;
- \(r_t^{\text{history}}\): recurrent summary of the recent curriculum.

The state is intentionally behavioral and dynamical rather than purely semantic.

Because no finite sketch is guaranteed to be Markov, the planner should treat this as a partially observed process and include history or a belief distribution.

---

## 6. Learning-effect signature

For action \(a\) at compressed state \(s\), define a target signature

\[
y(a\mid s)=
\begin{bmatrix}
\Delta s\\
\Delta \ell^{\text{probe}}\\
P_\Delta\Delta\theta\\
\Delta A^{\text{sketch}}\\
F^{\text{forget}}\\
N^{\text{novelty}}\\
R^{\text{retention}}\\
C^{\text{compute}}
\end{bmatrix}.
\]

The critical object is not a permanent score attached to data. It is the conditional transition distribution

\[
p(y,s'\mid s,a).
\]

Two semantically different chunks may have near-identical signatures and be interchangeable for teaching. Two similar chunks may have different signatures and both be necessary.

---

## 7. Transition simulator

Train a small model \(F_\psi\) to predict the next compressed state or transition signature:

\[
F_\psi(s_t,a_t)
\rightarrow
p_\psi(\Delta s_t,y_t\mid s_t,a_t).
\]

A probabilistic objective is preferred:

\[
\mathcal L_{\text{sim}}(\psi)
=
-\sum_n
\log p_\psi(y_n,\Delta s_n\mid s_n,a_n).
\]

The simulator should output epistemic or ensemble uncertainty

\[
\sigma_\psi(s,a),
\]

because MCTS must know when predicted transitions are unreliable.

### 7.1 Oracle transition collection

At selected target checkpoints:

1. sample candidate actions;
2. fork short target-model branches, preferably with copy-on-write states or temporary LoRA adapters;
3. run \(K\) real optimizer steps;
4. compute \(s'\), probe deltas, update sketches, and interference signatures;
5. store tuples \((s,a,y,s')\);
6. retrain or update \(F_\psi\).

The simulator is therefore repeatedly corrected by the real learner rather than trusted as a permanently valid proxy.

### 7.2 Active oracle querying

Prioritize real probes with high expected information gain, high search visitation, high predicted value, or high uncertainty. One acquisition score is

\[
A(s,a)
=
\alpha\,\sigma_\psi(s,a)
+\beta\,N_{\text{search}}(s,a)
+\gamma\,|\widehat Q(s,a)|
+\delta\,\operatorname{novelty}(s,a).
\]

---

## 8. Curriculum action space

Direct search over individual examples is factorial and impossible. Actions must be hierarchical.

### 8.1 Action hierarchy

1. **Domain action:** code, mathematics, prose, science, dialogue, etc.
2. **Learning-effect cluster:** examples grouped by predicted transition signature.
3. **Recipe action:** mixture proportions, step count, learning rate multiplier, or contrastive pairing.
4. **Example realization:** concrete chunks sampled within the chosen recipe.

An action may be represented as

\[
a_t=(c_t,w_t,k_t,\eta_t,r_t),
\]

where:

- \(c_t\): one or more clusters;
- \(w_t\): mixture weights;
- \(k_t\): number of optimizer steps or token budget;
- \(\eta_t\): optional learning-rate control;
- \(r_t\): replay/interleaving policy.

### 8.2 Special actions

The planner must be able to express:

- teach \(A\), then \(B\);
- interleave \(A\) and \(B\);
- teach bridge data \(X\) between them;
- replay \(A\) after \(B\);
- delay a useful action until prerequisites exist;
- allocate a new adapter/expert when interference is too high;
- stop when marginal teachability falls below cost.

---

## 9. Planning objective

Let curriculum \(\pi=(a_0,\dots,a_{T-1})\). The state trajectory is

\[
s_{t+1}=F_\psi(s_t,a_t).
\]

A general return is

\[
J(\pi)
=
\mathbb E\left[
R_T(s_T)
+
\sum_{t=0}^{T-1}
\left(
r(s_t,a_t,s_{t+1})
-\lambda_c c(a_t)
-\lambda_f f_t
-\lambda_u u_t
\right)
\right],
\]

where:

- \(R_T\): terminal capability or trajectory-match value;
- \(c(a_t)\): token/FLOP/wall-clock cost;
- \(f_t\): forgetting penalty;
- \(u_t\): instability or uncertainty penalty.

### 9.1 Full-corpus trajectory matching

Let \(s_t^D\) be reference states from a conventional full-data run. A compressed curriculum can minimize

\[
J_{\text{match}}(\pi)
=
\sum_{t\in\mathcal K}
\left\|
\Phi(s_t^\pi)-\Phi(s_t^D)
\right\|_W^2
+
\lambda_c\operatorname{cost}(\pi).
\]

Terminal-only matching is cheaper but may miss fragile or qualitatively different routes to the same benchmark score.

### 9.2 Capability-first optimization

A more ambitious objective does not imitate the full-data trajectory. It searches for a better one:

\[
\pi^*
=
\arg\max_\pi
\left[
V(s_T^\pi)-\lambda_c\operatorname{cost}(\pi)
\right].
\]

This allows the compiler to remove harmful redundancies in the original training stream rather than merely reproduce them.

---

## 10. Robust teacher–adversary game

A curriculum may succeed only under a lucky seed or an inaccurate simulator. Introduce nuisance/adversarial variables \(\xi\):

- initialization seed;
- batch realization;
- dropout and numerical noise;
- candidate model scale;
- simulator ensemble member;
- probe weighting;
- plausible contradictory data;
- transition-model error.

The robust objective is

\[
\pi^*
=
\arg\max_\pi
\min_{\xi\in\Xi}
J(\pi;\xi).
\]

Because strict minimax can be dominated by pathological cases, a practical alternative is lower-tail optimization:

\[
\pi^*
=
\arg\max_\pi
\operatorname{CVaR}_\alpha[J(\pi;\xi)].
\]

The teacher selects curriculum actions. The adversary selects or samples difficult plausible training conditions. The resulting curriculum should be robust rather than merely high-mean.

---

## 11. Hierarchical Monte Carlo Tree Search

A search node contains a belief over compressed learner state and remaining data budget. An edge is a curriculum action.

For node \(s\), select action \(a\) using a risk- and uncertainty-aware PUCT rule:

\[
a^*
=
\arg\max_a
\left[
Q(s,a)
+c_{\text{puct}}P(a\mid s)
\frac{\sqrt{N(s)}}{1+N(s,a)}
+\kappa\,I(s,a)
-\rho\,\operatorname{Risk}(s,a)
\right],
\]

where:

- \(Q(s,a)\): backed-up long-horizon value;
- \(P(a\mid s)\): teacher-policy prior;
- \(N\): visit counts;
- \(I(s,a)\): information value or simulator uncertainty bonus;
- \(\operatorname{Risk}\): predicted forgetting, instability, or lower-tail loss.

### 11.1 Progressive widening

When action space is large, allow the number of expanded actions to grow as

\[
|A_{\text{expanded}}(s)|
\le kN(s)^\alpha,
\qquad 0<\alpha<1.
\]

### 11.2 Transpositions and approximate state merging

Different curriculum prefixes may reach behaviorally similar states. Merge or share statistics when

\[
\|s_i-s_j\|_W<\epsilon
\]

and their uncertainty sets overlap. This converts the search tree toward a graph and avoids repeated simulation.

### 11.3 Commuting-block reduction

Estimate pairwise functional commutators. If a set of actions approximately commutes,

\[
C_\Phi(a_i,a_j\mid s)<\epsilon_c,
\]

collapse their permutations into one equivalence class. Search effort is then concentrated on strongly order-sensitive actions.

### 11.4 Macro-actions

Learn common successful subsequences as reusable options:

\[
\omega=(a_t,\ldots,a_{t+k}).
\]

Examples may include "foundation → contrast → replay" or "bridge → specialization → consolidation."

---

## 12. Dataset compression objective

Let \(D\) be the full corpus and \(\Pi(D)\) the set of executable curricula derived from it. Under budget \(B\):

\[
\pi_B^*
=
\arg\min_{\pi\in\Pi(D),\,c(\pi)\le B}
D_\Phi(s_T^\pi,s_T^D).
\]

Alternatively, find the minimum budget needed to reach tolerance \(\epsilon\):

\[
B^*(\epsilon)
=
\min_\pi c(\pi)
\quad\text{s.t.}\quad
D_\Phi(s_T^\pi,s_T^D)\le\epsilon.
\]

The compression ratio is

\[
\operatorname{CR}(\epsilon)
=
\frac{c(D)}{B^*(\epsilon)}.
\]

The retained object may contain repeated examples. Compression concerns total teaching cost and trajectory, not uniqueness of stored records.

---

## 13. Coverage in transition space

Let \(z_i(s)\) be the predicted transition embedding of candidate chunk or cluster \(i\). Select actions that cover useful transition directions while avoiding redundancy.

One set objective is

\[
U(S\mid s)
=
\operatorname{Coverage}(S\mid s)
-\lambda_r\operatorname{Redundancy}(S\mid s)
-\lambda_h\operatorname{Harm}(S\mid s)
+\lambda_y\operatorname{Synergy}(S\mid s).
\]

A kernelized coverage form is

\[
\operatorname{Coverage}(S)
=
\sum_{j\in D}
\max_{i\in S}K(z_i,z_j).
\]

Pairwise interaction can be approximated by

\[
U(B\mid s)
=
\sum_{i\in B}u_i(s)
+
\sum_{i<j}I_{ij}(s),
\]

but the simulator and MCTS are intended to capture higher-order and delayed effects that pairwise scoring misses.

---

## 14. Information limits

Teachability compression cannot preserve information that is wholly removed from every storage channel.

Separate corpus content into:

1. **generalizable structure:** grammar, procedures, abstractions, algorithms, causal patterns, and reusable representations;
2. **atomic payload:** unique names, dates, rare events, identifiers, and isolated facts.

Generalizable structure may be heavily compressible because many examples induce overlapping transitions. Atomic payload must be:

- retained in the curriculum;
- distilled into a denser carrier;
- placed in external retrieval memory;
- or explicitly accepted as lost.

The project should report capability preservation and factual coverage separately.

---

## 15. Main failure modes

### Simulator exploitation

MCTS may discover curricula that fool the transition model but fail on the real learner. Mitigations:

- ensembles and uncertainty penalties;
- frequent real-branch verification;
- adversarial simulator members;
- conservative value bounds;
- search-depth limits between corrections.

### State aliasing

Two compressed states may look identical but respond differently. Mitigations:

- recurrent history encoding;
- richer optimizer sketches;
- contrastive state-identifiability training;
- uncertainty inflation for out-of-distribution states.

### Proxy-scale mismatch

Learning order at 20M parameters may not transfer to 1B. Mitigations:

- condition simulator on architecture/scale descriptors;
- train across a family of target sizes;
- periodically query the actual target scale;
- search for curricula robust across student variants.

### Search explosion

Mitigations:

- hierarchical actions;
- progressive widening;
- approximate commuting classes;
- transition-effect clustering;
- learned policy/value priors;
- macro-actions and transposition tables.

### Overfitting the probe map

The compiler may optimize anchor metrics while damaging unmeasured behavior. Mitigations:

- broad hidden evaluation sets;
- rotating anchors;
- random logit/function probes;
- representation and calibration measures;
- adversarial holdout tasks.

### Short-horizon bias

Some actions are prerequisites with delayed value. Mitigations:

- long-horizon MCTS backup;
- successor-feature prediction;
- bridge/replay actions;
- terminal capability rewards;
- explicit delayed-synergy tests.

---

## 16. Testable predictions

1. Pairwise functional commutator magnitudes will be heavy-tailed: most action pairs nearly commute, while a minority dominate order sensitivity.
2. Transition-effect clustering will identify substantial redundancy not visible through embedding similarity alone.
3. Static per-example utility will decay in predictive value as the target state moves away from the scoring checkpoint.
4. MCTS will outperform greedy selection most strongly on datasets containing prerequisites, contradictions, bridges, and replay-sensitive tasks.
5. A robust curriculum optimized across seeds will sacrifice some mean simulator value but achieve better real-model transfer.
6. A state-conditioned curriculum will transfer poorly to sufficiently different architectures unless architecture descriptors or multi-student training are included.
7. The best compact curriculum will contain deliberate repetition and interleaving rather than a unique one-pass subset.

---

## 17. Minimal scientific claim

The first publishable claim should be narrow:

> On controlled small-language-model tasks with known prerequisite and interference structure, a periodically corrected transition simulator plus hierarchical MCTS finds ordered curricula that reach a fixed held-out capability target with fewer training tokens than random shuffle, static data selection, and greedy influence selection, across multiple seeds.

No broader claim should be made until the method survives scale, domain, and architecture transfer tests.
