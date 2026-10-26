# Formal verification

This directory contains the new Selective Overlap formalization and an earlier,
separate coalition-game proof. The research proposal and its original source
snapshot are in [data/selective_overlap/](../data/selective_overlap/).

## Selective Overlap Records

The two new proof modules contain **64 public theorems**: 17 in Preservation and
47 in Relu. They were added for this repository upload; the original proposal
PDF and ZIP did not contain Lean files.

| File | Checked scope |
| --- | --- |
| [SelectiveOverlap/Preservation.lean](SelectiveOverlap/Preservation.lean) | Target sufficiency on observation fibers; decoder existence on the actual observation range; composition; side-record preservation; linear kernel containment and linear range decoders; agreement and source-explanation conditions; a compatible-but-insufficient counterexample |
| [SelectiveOverlap/Relu.lean](SelectiveOverlap/Relu.lean) | Exact real-valued ReLU/CReLU algebra; absolute-value, contrast, and sign-gate corrections; position weights and signed coefficients; original four-feature teacher repair; finite binary-tree telescopes; positive/negative mass identities; merge laws; explicit target-changing and order-loss collisions |
| [SelectiveOverlap.lean](SelectiveOverlap.lean) | Public entry point importing both modules |
| [AxiomAudit.lean](AxiomAudit.lean) | Axiom-dependency queries for all 64 public theorems |
| [axiom_audit.txt](axiom_audit.txt) | Recorded dependency audit |
| [verification_status.json](verification_status.json) | Compiler/dependency versions, source hashes, theorem inventory, and verification results |

### Preservation contract

The function-level theorem uses the complete retained record:

$$
E(x)=E(x') \Longrightarrow T(x)=T(x').
$$

It is equivalent to a decoder on the actual range of E. Using that domain
handles empty types without silently assuming an inhabited target type.
The abstract existence proof uses classical choice; it does not provide a
computational decoder.

For linear maps the checked statements relate target preservation to
kernel containment and construct a linear decoder on the observed submodule:

$$
\ker M \subseteq \ker C.
$$

The linear theorems work over modules over a ring. They do not assert an
extension of the decoder to the entire ambient record space under those
general assumptions. The counterexample separately proves overlap agreement,
coverage of all compatible records, and failure to preserve the target.

### Rectification and finite piles

The real-valued correction is defined by

$$
r(a,b)=\operatorname{ReLU}(a)+\operatorname{ReLU}(b)
       -\operatorname{ReLU}(a+b).
$$

Its absolute-value, contrast, and mixed-sign minimum formulas are proved,
along with reconstruction and nonnegativity. An inductive ordered binary
tree proves the finite-pile telescope. The aggregate merge is associative
and commutative in exact real arithmetic; an explicit witness proves that
it does not preserve the original order.

The four-feature teacher identity assumes access to the teacher features.
It is an algebraic oracle identity, not a theorem that a student learns
those features.

### Explicit boundary of formal coverage

The following proposal material is **not** certified by these Lean modules:

- Minimum residual rank, orthogonal residual bases, and pseudoinverse formulas.
- Reconstruction conditioning and quantitative error bounds.
- Full chart-transition cocycles and Hodge decomposition.
- The Gaussian ReLU kernel and conditional-risk derivations.
- Grid-rotation inverses and finite-alphabet packing.
- Floating-point or automatic-differentiation behavior.
- Any learning, efficiency, generalization, or neural benchmark claim.

Those topics remain documented in the proposal and, where applicable,
its [deterministic Python checks](../data/selective_overlap/source/verify_math.py).
The formalized real-arithmetic identities do not imply bitwise reversibility.

## Pinned build

The project pins Lean **4.35.0** and mathlib **v4.35.0**. The committed
[lake-manifest.json](lake-manifest.json) fixes the exact dependency revisions,
including mathlib commit 56a005855ea17e0b0958f73ec801ae140b0a1679.
Use the Lean toolchain specified in [lean-toolchain](lean-toolchain).

From this directory:

~~~bash
lake exe cache get Mathlib.Basic.Real.Basic Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum Mathlib.Tactic.Ring Mathlib.Algebra.Module.Submodule.Range
lake build
lake env lean AxiomAudit.lean
~~~

The targeted cache command retrieves the compiled imports needed by these
modules. Downloaded dependencies and build outputs belong under the ignored
.lake directory.

On 26 October 2026, the combined project build and axiom audit completed
successfully. The audit found only Lean's standard axioms **propext**,
**Classical.choice**, and **Quot.sound** among the theorem dependencies.
It found no sorryAx. No custom axioms or admitted proofs were added.

## Earlier coalition-game proof

[ExactLocalCoalition.lean](ExactLocalCoalition.lean) predates this proposal
and remains unchanged. It concerns integer coefficient identities for a finite
expert-coalition game. It is not part of the Selective Overlap proof coverage.
It also compiles with the pinned toolchain:

~~~bash
lake env lean ExactLocalCoalition.lean
~~~

The repository's [license notice](../LICENSE-NOTICE.md) continues to apply.
