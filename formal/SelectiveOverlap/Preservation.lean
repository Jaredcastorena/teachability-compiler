module

public import Mathlib.Algebra.Module.Submodule.Range

/-!
# Target preservation and compatible local records

This file checks a deliberately limited part of the Selective Overlap proposal:

* target constancy on record fibers is equivalent to a decoder on the actual range;
* explicit factorizations compose, and retained side records preserve sufficiency;
* for linear maps, target sufficiency is equivalent to kernel containment and to
  a linear decoder on the observation map's range;
* agreement of source-generated records and coverage of compatible records are
  separate from target sufficiency;
* a concrete integer example is compatible and covers every compatible record,
  but still loses its declared target.

The general decoder has domain `Set.range record`. There is no assumption that
the target type is inhabited, so an extension to every possible record value is
not claimed. Existence proofs using classical choice are not executable decoder
algorithms. Rank formulas, pseudoinverses, conditioning, finite-precision
arithmetic, learned maps, atlas cocycles, and Hodge decomposition are outside the
formalized scope of this file.
-/

@[expose] public section

namespace SelectiveOverlap

universe u v w r s t

section Functions

variable {X : Type u} {E : Type v} {Y : Type w} {F : Type r}

/-- Equal complete records must imply equal declared target values. -/
def PreservesTarget (record : X → E) (target : X → Y) : Prop :=
  ∀ x x', record x = record x' → target x = target x'

/-- A source's record, regarded as an element of the actual observation range. -/
def rangeRecord (record : X → E) (x : X) : Set.range record :=
  ⟨record x, ⟨x, rfl⟩⟩

/--
Fiber sufficiency gives a decoder on the actual range, even for empty types.
No extension to the full ambient record space is asserted.
-/
theorem preservesTarget_iff_exists_range_decoder
    (record : X → E) (target : X → Y) :
    PreservesTarget record target ↔
      ∃ decoder : Set.range record → Y,
        ∀ x, decoder (rangeRecord record x) = target x := by
  constructor
  · intro h
    classical
    refine ⟨fun e => target (Classical.choose e.property), ?_⟩
    intro x
    exact h _ x (Classical.choose_spec (rangeRecord record x).property)
  · rintro ⟨decoder, hdecoder⟩ x x' hrecord
    have hrange : rangeRecord record x = rangeRecord record x' :=
      Subtype.ext hrecord
    calc
      target x = decoder (rangeRecord record x) := (hdecoder x).symm
      _ = decoder (rangeRecord record x') := congrArg decoder hrange
      _ = target x' := hdecoder x'

/-- An explicitly supplied total decoder is sufficient, when one is available. -/
theorem preservesTarget_of_factorization
    (record : X → E) (target : X → Y) (decoder : E → Y)
    (h : ∀ x, decoder (record x) = target x) :
    PreservesTarget record target := by
  intro x x' hrecord
  calc
    target x = decoder (record x) := (h x).symm
    _ = decoder (record x') := congrArg decoder hrecord
    _ = target x' := h x'

/-- Explicit target factorizations compose in their stated order. -/
theorem target_factorizations_comp
    (record : X → E) (next : E → F) (target : X → Y)
    (firstDecoder : E → Y) (secondDecoder : F → Y)
    (hfirst : ∀ x, firstDecoder (record x) = target x)
    (hsecond : ∀ e, secondDecoder (next e) = firstDecoder e) :
    ∀ x, secondDecoder (next (record x)) = target x := by
  intro x
  exact (hsecond (record x)).trans (hfirst x)

/-- A next stage is sufficient when it preserves the target already defined on its input. -/
theorem preservesTarget_comp
    (record : X → E) (next : E → F)
    (target : X → Y) (intermediateTarget : E → Y)
    (hfirst : ∀ x, intermediateTarget (record x) = target x)
    (hnext : PreservesTarget next intermediateTarget) :
    PreservesTarget (fun x => next (record x)) target := by
  intro x x' hrecord
  calc
    target x = intermediateTarget (record x) := (hfirst x).symm
    _ = intermediateTarget (record x') := hnext _ _ hrecord
    _ = target x' := hfirst x'

/-- Appending a side record cannot destroy the sufficiency of the retained first record. -/
theorem preservesTarget_with_side_record
    (record : X → E) (side : X → F) (target : X → Y)
    (h : PreservesTarget record target) :
    PreservesTarget (fun x => (record x, side x)) target := by
  intro x x' hrecords
  exact h x x' (congrArg Prod.fst hrecords)

/-- A postprocessing step cannot recover a target distinction already lost by its input. -/
theorem preservesTarget_before_postprocessing
    (record : X → E) (postprocess : E → F) (target : X → Y)
    (h : PreservesTarget (fun x => postprocess (record x)) target) :
    PreservesTarget record target := by
  intro x x' hrecord
  exact h x x' (congrArg postprocess hrecord)

/-- A target-changing collision refutes the preservation contract. -/
theorem not_preservesTarget_of_collision
    (record : X → E) (target : X → Y) (x x' : X)
    (hrecord : record x = record x') (htarget : target x ≠ target x') :
    ¬ PreservesTarget record target := by
  intro h
  exact htarget (h x x' hrecord)

end Functions

section Linear

variable {𝕜 : Type u} [Ring 𝕜]
variable {V : Type v} [AddCommGroup V] [Module 𝕜 V]
variable {E : Type w} [AddCommGroup E] [Module 𝕜 E]
variable {Y : Type r} [AddCommGroup Y] [Module 𝕜 Y]
variable {O : Type s} [AddCommGroup O] [Module 𝕜 O]

/-- Linear target observability is exactly the inclusion of observation kernels. -/
theorem ker_le_iff_preservesTarget (M : V →ₗ[𝕜] E) (C : V →ₗ[𝕜] Y) :
    M.ker ≤ C.ker ↔ PreservesTarget M C := by
  constructor
  · intro h x x' hrecord
    have hM : x - x' ∈ M.ker := by
      change M (x - x') = 0
      rw [map_sub, hrecord, sub_self]
    have hC := h hM
    change C (x - x') = 0 at hC
    rw [map_sub] at hC
    exact sub_eq_zero.mp hC
  · intro h x hx
    change C x = 0
    have hMx : M x = 0 := hx
    have hzero : M x = M 0 := by simpa only [map_zero] using hMx
    simpa only [map_zero] using h x 0 hzero

/--
Kernel containment admits a linear decoder on the observed submodule. This
does not require finite dimension, an inner product, or an ambient extension.
-/
theorem exists_linear_range_decoder_of_ker_le
    (M : V →ₗ[𝕜] E) (C : V →ₗ[𝕜] Y) (hker : M.ker ≤ C.ker) :
    ∃ decoder : M.range →ₗ[𝕜] Y,
      ∀ x, decoder (M.rangeRestrict x) = C x := by
  classical
  let pick : M.range → V :=
    fun e => Classical.choose (LinearMap.mem_range.mp e.property)
  have hpick : ∀ e : M.range, M (pick e) = (e : E) := by
    intro e
    exact Classical.choose_spec (LinearMap.mem_range.mp e.property)
  have hfiber := (ker_le_iff_preservesTarget M C).mp hker
  let decoder : M.range →ₗ[𝕜] Y :=
    { toFun := fun e => C (pick e)
      map_add' := by
        intro a b
        change C (pick (a + b)) = C (pick a) + C (pick b)
        calc
          C (pick (a + b)) = C (pick a + pick b) := by
            apply hfiber
            calc
              M (pick (a + b)) = ((a + b : M.range) : E) := hpick _
              _ = (a : E) + (b : E) := rfl
              _ = M (pick a) + M (pick b) := by rw [hpick, hpick]
              _ = M (pick a + pick b) := (map_add M _ _).symm
          _ = C (pick a) + C (pick b) := map_add C _ _
      map_smul' := by
        intro c a
        change C (pick (c • a)) = c • C (pick a)
        calc
          C (pick (c • a)) = C (c • pick a) := by
            apply hfiber
            calc
              M (pick (c • a)) = ((c • a : M.range) : E) := hpick _
              _ = c • (a : E) := rfl
              _ = c • M (pick a) := by rw [hpick]
              _ = M (c • pick a) := (map_smul M _ _).symm
          _ = c • C (pick a) := map_smul C _ _ }
  refine ⟨decoder, ?_⟩
  intro x
  change C (pick (M.rangeRestrict x)) = C x
  apply hfiber
  exact hpick (M.rangeRestrict x)

/-- The linear range factorization is equivalent to kernel containment. -/
theorem ker_le_iff_exists_linear_range_decoder
    (M : V →ₗ[𝕜] E) (C : V →ₗ[𝕜] Y) :
    M.ker ≤ C.ker ↔
      ∃ decoder : M.range →ₗ[𝕜] Y,
        ∀ x, decoder (M.rangeRestrict x) = C x := by
  constructor
  · exact exists_linear_range_decoder_of_ker_le M C
  · rintro ⟨decoder, hdecoder⟩
    apply (ker_le_iff_preservesTarget M C).mpr
    intro x x' hrecord
    have hrange : M.rangeRestrict x = M.rangeRestrict x' :=
      Subtype.ext hrecord
    calc
      C x = decoder (M.rangeRestrict x) := (hdecoder x).symm
      _ = decoder (M.rangeRestrict x') := congrArg decoder hrange
      _ = C x' := hdecoder x'

/--
Agreement means source-generated records lie in the mismatch kernel. It is
not a statement that the source map preserves a separate target.
-/
theorem range_le_ker_iff_agreement_zero
    (M : V →ₗ[𝕜] E) (δ : E →ₗ[𝕜] O) :
    M.range ≤ δ.ker ↔ δ.comp M = 0 := by
  constructor
  · intro h
    apply LinearMap.ext
    intro x
    change δ (M x) = 0
    exact h (LinearMap.mem_range_self M x)
  · intro h e he
    obtain ⟨x, hx⟩ := LinearMap.mem_range.mp he
    change δ e = 0
    rw [← hx]
    have hvalue := congrArg (fun f : V →ₗ[𝕜] O => f x) h
    exact hvalue

/-- Equality of image and mismatch kernel supplies a source for every compatible record. -/
theorem compatible_record_has_source
    (M : V →ₗ[𝕜] E) (δ : E →ₗ[𝕜] O)
    (hexact : M.range = δ.ker) (e : E) (he : δ e = 0) :
    ∃ x, M x = e := by
  apply LinearMap.mem_range.mp
  rw [hexact]
  exact he

/--
Coverage supplies existence; target kernel containment supplies independence
of the source chosen for the compatible record. Neither condition replaces
the other.
-/
theorem compatible_record_has_unique_target
    (M : V →ₗ[𝕜] E) (δ : E →ₗ[𝕜] O) (C : V →ₗ[𝕜] Y)
    (hexact : M.range = δ.ker) (htarget : M.ker ≤ C.ker)
    (e : E) (he : δ e = 0) :
    ∃ value : Y,
      (∃ x, M x = e ∧ C x = value) ∧
      ∀ x, M x = e → C x = value := by
  obtain ⟨x, hx⟩ := compatible_record_has_source M δ hexact e he
  refine ⟨C x, ⟨x, hx, rfl⟩, ?_⟩
  intro x' hx'
  exact (ker_le_iff_preservesTarget M C).mp htarget x' x (hx'.trans hx.symm)

end Linear

namespace Counterexample

/-- Both saved views retain only the middle coordinate. -/
def diagonalRecords (x : ℤ × ℤ × ℤ) : ℤ × ℤ := (x.2.1, x.2.1)

/-- The overlap test asks whether the two saved middle coordinates agree. -/
def diagonalMismatch (e : ℤ × ℤ) : ℤ := e.1 - e.2

/-- The declared target depends on the two coordinates omitted from the records. -/
def omittedContrast (x : ℤ × ℤ × ℤ) : ℤ := x.1 - x.2.2

theorem diagonalRecords_agree (x : ℤ × ℤ × ℤ) :
    diagonalMismatch (diagonalRecords x) = 0 := by
  simp only [diagonalMismatch, diagonalRecords, sub_self]

/-- Every pair passing the agreement test actually arises from a source. -/
theorem diagonalRecords_cover_compatible :
    Set.range diagonalRecords = {e | diagonalMismatch e = 0} := by
  apply Set.ext
  intro e
  constructor
  · rintro ⟨x, rfl⟩
    exact diagonalRecords_agree x
  · intro he
    change e.1 - e.2 = 0 at he
    have heq : e.1 = e.2 := sub_eq_zero.mp he
    refine ⟨(0, e.1, 0), ?_⟩
    apply Prod.ext
    · rfl
    · exact heq

/-- Compatibility and complete coverage of compatible records still lose this target. -/
theorem diagonalRecords_lose_target :
    ¬ PreservesTarget diagonalRecords omittedContrast := by
  apply not_preservesTarget_of_collision diagonalRecords omittedContrast
    (1, 0, 0) (0, 0, 0)
  · rfl
  · intro h
    have hbad : (1 : ℤ) = 0 := by
      simpa only [omittedContrast, sub_zero] using h
    exact (by decide : (1 : ℤ) ≠ 0) hbad

/-- Consequently, even a decoder restricted to the actual record range cannot recover it. -/
theorem diagonalRecords_have_no_target_decoder :
    ¬ ∃ decoder : Set.range diagonalRecords → ℤ,
      ∀ x, decoder (rangeRecord diagonalRecords x) = omittedContrast x := by
  intro h
  exact diagonalRecords_lose_target
    ((preservesTarget_iff_exists_range_decoder diagonalRecords omittedContrast).mpr h)

end Counterexample

end SelectiveOverlap
