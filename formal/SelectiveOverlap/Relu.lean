module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.Ring

/-!
# Selective overlap corrections for real-valued ReLU computations

This is a new formalization prepared for the repository upload. It was not
part of the original Selective Overlap proposal bundle.

The statements concern exact real arithmetic. They do not assert that a
student learns the required features, that a finite-precision implementation
is bitwise reversible, or that the aggregate retains arbitrary input order.
The final counterexample makes the order limitation explicit.
-/

@[expose] public section

noncomputable section

namespace SelectiveOverlap

/-- Rectification of one real-valued measurement. -/
def relu (x : ℝ) : ℝ := max x 0

/-- The positive and negative channels of the supplied measurement. -/
def crelu (x : ℝ) : ℝ × ℝ := (relu x, relu (-x))

@[simp] theorem relu_zero : relu 0 = 0 := by
  simp [relu]

theorem relu_of_nonneg {x : ℝ} (hx : 0 ≤ x) : relu x = x :=
  max_eq_left hx

theorem relu_of_nonpos {x : ℝ} (hx : x ≤ 0) : relu x = 0 :=
  max_eq_right hx

theorem relu_nonneg (x : ℝ) : 0 ≤ relu x :=
  le_max_right x 0

theorem self_le_relu (x : ℝ) : x ≤ relu x :=
  le_max_left x 0

/-- CReLU recovers its supplied signed measurement, not a prior projection. -/
theorem relu_sub_neg (x : ℝ) : relu x - relu (-x) = x := by
  by_cases hx : 0 ≤ x
  · rw [relu_of_nonneg hx, relu_of_nonpos (neg_nonpos.mpr hx)]
    ring
  · have hx' : x ≤ 0 := by linarith
    rw [relu_of_nonpos hx', relu_of_nonneg (neg_nonneg.mpr hx')]
    ring

theorem crelu_signed_inverse (x : ℝ) : (crelu x).1 - (crelu x).2 = x :=
  relu_sub_neg x

theorem relu_add_neg (x : ℝ) : relu x + relu (-x) = |x| := by
  by_cases hx : 0 ≤ x
  · rw [relu_of_nonneg hx, relu_of_nonpos (neg_nonpos.mpr hx),
      abs_of_nonneg hx]
    ring
  · have hx' : x ≤ 0 := by linarith
    rw [relu_of_nonpos hx', relu_of_nonneg (neg_nonneg.mpr hx'),
      abs_of_nonpos hx']
    ring

theorem two_mul_relu (x : ℝ) : 2 * relu x = x + |x| := by
  by_cases hx : 0 ≤ x
  · rw [relu_of_nonneg hx, abs_of_nonneg hx]
    ring
  · have hx' : x ≤ 0 := by linarith
    rw [relu_of_nonpos hx', abs_of_nonpos hx']
    ring

theorem relu_eq_half (x : ℝ) : relu x = (x + |x|) / 2 := by
  linarith [two_mul_relu x]

/-- The amount needed to correct rectification after a pairwise sum. -/
def correction (a b : ℝ) : ℝ := relu a + relu b - relu (a + b)

theorem correction_reconstruct (a b : ℝ) :
    relu a + relu b = relu (a + b) + correction a b := by
  unfold correction
  ring

theorem correction_nonneg (a b : ℝ) : 0 ≤ correction a b := by
  have hs : a + b ≤ relu a + relu b :=
    add_le_add (self_le_relu a) (self_le_relu b)
  have hn : 0 ≤ relu a + relu b :=
    add_nonneg (relu_nonneg a) (relu_nonneg b)
  have hc : relu (a + b) ≤ relu a + relu b := max_le hs hn
  exact sub_nonneg.mpr hc

theorem correction_abs (a b : ℝ) :
    correction a b = (|a| + |b| - |a + b|) / 2 := by
  unfold correction
  rw [relu_eq_half a, relu_eq_half b, relu_eq_half (a + b)]
  ring

theorem correction_comm (a b : ℝ) : correction a b = correction b a := by
  unfold correction
  rw [add_comm a b]
  ring

@[simp] theorem correction_zero_left (a : ℝ) : correction 0 a = 0 := by
  simp [correction]

@[simp] theorem correction_zero_right (a : ℝ) : correction a 0 = 0 := by
  simp [correction]

theorem correction_of_nonneg {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    correction a b = 0 := by
  unfold correction
  rw [relu_of_nonneg ha, relu_of_nonneg hb,
    relu_of_nonneg (add_nonneg ha hb)]
  ring

theorem correction_of_nonpos {a b : ℝ} (ha : a ≤ 0) (hb : b ≤ 0) :
    correction a b = 0 := by
  unfold correction
  rw [relu_of_nonpos ha, relu_of_nonpos hb,
    relu_of_nonpos (add_nonpos ha hb)]
  ring

theorem correction_of_nonneg_nonpos {a b : ℝ}
    (ha : 0 ≤ a) (hb : b ≤ 0) :
    correction a b = min a (-b) := by
  unfold correction
  rw [relu_of_nonneg ha, relu_of_nonpos hb]
  by_cases hs : 0 ≤ a + b
  · have hba : -b ≤ a := by linarith
    rw [relu_of_nonneg hs, min_eq_right hba]
    ring
  · have hs' : a + b ≤ 0 := by linarith
    have hab : a ≤ -b := by linarith
    rw [relu_of_nonpos hs', min_eq_left hab]
    ring

theorem correction_of_nonpos_nonneg {a b : ℝ}
    (ha : a ≤ 0) (hb : 0 ≤ b) :
    correction a b = min (-a) b := by
  rw [correction_comm a b, correction_of_nonneg_nonpos hb ha, min_comm]

/-- A gate bit alone is insufficient: the mixed-sign branch retains a magnitude. -/
theorem correction_gate (a b : ℝ) :
    correction a b = if a * b < 0 then min |a| |b| else 0 := by
  by_cases ha : 0 ≤ a
  · by_cases hb : 0 ≤ b
    · have hab : ¬ a * b < 0 := not_lt.mpr (mul_nonneg ha hb)
      rw [ite_eq_right hab, correction_of_nonneg ha hb]
    · have hb' : b ≤ 0 := by linarith
      by_cases hab : a * b < 0
      · rw [ite_eq_left hab, correction_of_nonneg_nonpos ha hb',
          abs_of_nonneg ha, abs_of_nonpos hb']
      · have hp : a * b = 0 :=
          le_antisymm (mul_nonpos_of_nonneg_of_nonpos ha hb')
            (le_of_not_gt hab)
        have ha0 : a = 0 := by
          rcases mul_eq_zero.mp hp with ha0 | hb0
          · exact ha0
          · exact False.elim (hb (by simp [hb0]))
        rw [ite_eq_right hab, ha0, correction_zero_left]
  · have ha' : a ≤ 0 := by linarith
    by_cases hb : 0 ≤ b
    · by_cases hab : a * b < 0
      · rw [ite_eq_left hab, correction_of_nonpos_nonneg ha' hb,
          abs_of_nonpos ha', abs_of_nonneg hb]
      · have hp : a * b = 0 :=
          le_antisymm (mul_nonpos_of_nonpos_of_nonneg ha' hb)
            (le_of_not_gt hab)
        have hb0 : b = 0 := by
          rcases mul_eq_zero.mp hp with ha0 | hb0
          · exact False.elim (ha (by simp [ha0]))
          · exact hb0
        rw [ite_eq_right hab, hb0, correction_zero_right]
    · have hb' : b ≤ 0 := by linarith
      have hab : ¬ a * b < 0 :=
        not_lt.mpr (mul_nonneg_of_nonpos_of_nonpos ha' hb')
      rw [ite_eq_right hab, correction_of_nonpos ha' hb']

/-- The common and contrast magnitudes determine the sum of magnitudes. -/
theorem abs_common_contrast (a b : ℝ) :
    |a| + |b| = max |a + b| |a - b| := by
  by_cases ha : 0 ≤ a
  · by_cases hb : 0 ≤ b
    · have hs : 0 ≤ a + b := add_nonneg ha hb
      have hd : |a - b| ≤ a + b := by
        apply abs_le.mpr
        constructor <;> linarith
      rw [abs_of_nonneg ha, abs_of_nonneg hb, abs_of_nonneg hs,
        max_eq_left hd]
    · have hb' : b ≤ 0 := by linarith
      have hd : 0 ≤ a - b := by linarith
      have hs : |a + b| ≤ a - b := by
        apply abs_le.mpr
        constructor <;> linarith
      rw [abs_of_nonneg ha, abs_of_nonpos hb', abs_of_nonneg hd,
        max_eq_right hs]; ring
  · have ha' : a ≤ 0 := by linarith
    by_cases hb : 0 ≤ b
    · have hd : a - b ≤ 0 := by linarith
      have hs : |a + b| ≤ -(a - b) := by
        apply abs_le.mpr
        constructor <;> linarith
      rw [abs_of_nonpos ha', abs_of_nonneg hb, abs_of_nonpos hd,
        max_eq_right hs]; ring
    · have hb' : b ≤ 0 := by linarith
      have hs : a + b ≤ 0 := add_nonpos ha' hb'
      have hd : |a - b| ≤ -(a + b) := by
        apply abs_le.mpr
        constructor <;> linarith
      rw [abs_of_nonpos ha', abs_of_nonpos hb', abs_of_nonpos hs,
        max_eq_left hd]; ring

private theorem max_sub_first (x y : ℝ) :
    max x y - x = max (y - x) 0 := by
  by_cases h : x ≤ y
  · rw [max_eq_right h, max_eq_left (sub_nonneg.mpr h)]
  · have h' : y ≤ x := by linarith
    rw [max_eq_left h', max_eq_right (sub_nonpos.mpr h')]
    ring

theorem correction_contrast (a b : ℝ) :
    correction a b = relu (|a - b| - |a + b|) / 2 := by
  change correction a b = max (|a - b| - |a + b|) 0 / 2
  rw [correction_abs, abs_common_contrast, max_sub_first]

/-- Positive position weights can be absorbed before the correction. -/
theorem relu_mul_nonneg (c x : ℝ) (hc : 0 ≤ c) :
    relu (c * x) = c * relu x := by
  by_cases hx : 0 ≤ x
  · rw [relu_of_nonneg (mul_nonneg hc hx), relu_of_nonneg hx]
  · have hx' : x ≤ 0 := by linarith
    rw [relu_of_nonpos (mul_nonpos_of_nonneg_of_nonpos hc hx'),
      relu_of_nonpos hx']
    ring

theorem positive_weighted_repair (α β a b : ℝ)
    (hα : 0 ≤ α) (hβ : 0 ≤ β) :
    α * relu a + β * relu b =
      relu (α * a + β * b) + correction (α * a) (β * b) := by
  rw [← relu_mul_nonneg α a hα, ← relu_mul_nonneg β b hβ]
  exact correction_reconstruct (α * a) (β * b)

/-- Arbitrary signed output coefficients need a possibly signed correction. -/
def signedCorrection (α β a b : ℝ) : ℝ :=
  α * relu a + β * relu b - relu (α * a + β * b)

theorem signed_correction_abs (α β a b : ℝ) :
    signedCorrection α β a b =
      (α * |a| + β * |b| - |α * a + β * b|) / 2 := by
  unfold signedCorrection
  rw [relu_eq_half a, relu_eq_half b, relu_eq_half (α * a + β * b)]
  ring

theorem signed_correction_reconstruct (α β a b : ℝ) :
    α * relu a + β * relu b =
      relu (α * a + β * b) + signedCorrection α β a b := by
  unfold signedCorrection
  ring

/-- Initial residual required when a leaf has a negative output coefficient. -/
def leafCorrection (α a : ℝ) : ℝ := α * relu a - relu (α * a)

theorem leaf_correction_of_nonneg (α a : ℝ) (hα : 0 ≤ α) :
    leafCorrection α a = 0 := by
  unfold leafCorrection
  rw [relu_mul_nonneg α a hα]
  ring

theorem leaf_correction_of_nonpos (α a : ℝ) (hα : α ≤ 0) :
    leafCorrection α a = α * |a| := by
  unfold leafCorrection
  by_cases ha : 0 ≤ a
  · rw [relu_of_nonneg ha,
      relu_of_nonpos (mul_nonpos_of_nonpos_of_nonneg hα ha),
      abs_of_nonneg ha]
    ring
  · have ha' : a ≤ 0 := by linarith
    rw [relu_of_nonpos ha',
      relu_of_nonneg (mul_nonneg_of_nonpos_of_nonpos hα ha'),
      abs_of_nonpos ha']
    ring

/-- The intrinsic-coordinate teacher; this definition supplies its features. -/
def fourTeacher (u₁ u₂ u₃ u₄ : ℝ) : ℝ :=
  (-relu u₁ + relu u₂ - relu u₃ + relu u₄) / 2

def collapsedTeacher (u₁ u₂ u₃ u₄ : ℝ) : ℝ :=
  (relu (u₂ + u₄) - relu (u₁ + u₃)) / 2

/-- An oracle feature identity, not a claim that a student learns the features. -/
theorem four_teacher_repair (u₁ u₂ u₃ u₄ : ℝ) :
    fourTeacher u₁ u₂ u₃ u₄ =
      collapsedTeacher u₁ u₂ u₃ u₄ +
        (correction u₂ u₄ - correction u₁ u₃) / 2 := by
  unfold fourTeacher collapsedTeacher correction
  ring

/-- A signed sum and its accumulated target correction. -/
def merge (left right : ℝ × ℝ) : ℝ × ℝ :=
  (left.1 + right.1,
   left.2 + right.2 + correction left.1 right.1)

theorem correction_cocycle (a b c : ℝ) :
    correction a b + correction (a + b) c =
      correction b c + correction a (b + c) := by
  unfold correction
  rw [add_assoc a b c]
  ring

theorem merge_assoc (x y z : ℝ × ℝ) :
    merge (merge x y) z = merge x (merge y z) := by
  apply Prod.ext
  · simp only [merge]
    ring
  · simp only [merge, correction]
    rw [add_assoc x.1 y.1 z.1]
    ring

theorem merge_comm (x y : ℝ × ℝ) : merge x y = merge y x := by
  apply Prod.ext
  · simp only [merge]
    ring
  · simp only [merge]
    rw [correction_comm x.1 y.1]
    ring

@[simp] theorem merge_zero_left (x : ℝ × ℝ) : merge (0, 0) x = x := by
  apply Prod.ext <;> simp [merge]

@[simp] theorem merge_zero_right (x : ℝ × ℝ) : merge x (0, 0) = x := by
  apply Prod.ext <;> simp [merge]

theorem merge_reconstruct (x y : ℝ × ℝ) :
    relu (merge x y).1 + (merge x y).2 =
      (relu x.1 + x.2) + (relu y.1 + y.2) := by
  dsimp [merge, correction]
  ring

theorem merge_signed_leaves (α β a b : ℝ) :
    relu (merge (α * a, leafCorrection α a)
      (β * b, leafCorrection β b)).1 +
      (merge (α * a, leafCorrection α a)
        (β * b, leafCorrection β b)).2 =
      α * relu a + β * relu b := by
  rw [merge_reconstruct]
  dsimp [leafCorrection]
  ring

/-- A finite ordered binary tree. Its source order is not discarded by the type. -/
inductive OrderedTree where
  | leaf (value : ℝ)
  | node (left right : OrderedTree)

namespace OrderedTree

def signedSum : OrderedTree → ℝ
  | .leaf value => value
  | .node left right => signedSum left + signedSum right

def positiveMass : OrderedTree → ℝ
  | .leaf value => relu value
  | .node left right => positiveMass left + positiveMass right

def negativeMass : OrderedTree → ℝ
  | .leaf value => relu (-value)
  | .node left right => negativeMass left + negativeMass right

/-- Sum of the correction at every internal node. -/
def totalCorrection : OrderedTree → ℝ
  | .leaf _ => 0
  | .node left right =>
      totalCorrection left + totalCorrection right +
        correction (signedSum left) (signedSum right)

def summary (tree : OrderedTree) : ℝ × ℝ :=
  (signedSum tree, totalCorrection tree)

theorem summary_node (left right : OrderedTree) :
    summary (.node left right) = merge (summary left) (summary right) := rfl

/-- Exact telescope for every finite ordered binary merge tree. -/
theorem positive_telescope (tree : OrderedTree) :
    positiveMass tree = relu (signedSum tree) + totalCorrection tree := by
  induction tree with
  | leaf value => simp [positiveMass, signedSum, totalCorrection]
  | node left right ihl ihr =>
      simp only [positiveMass, signedSum, totalCorrection]
      rw [ihl, ihr]
      unfold correction
      ring

theorem signed_sum_eq_mass_difference (tree : OrderedTree) :
    signedSum tree = positiveMass tree - negativeMass tree := by
  induction tree with
  | leaf value =>
      exact (relu_sub_neg value).symm
  | node left right ihl ihr =>
      simp only [signedSum, positiveMass, negativeMass]
      rw [ihl, ihr]
      ring

theorem negative_telescope (tree : OrderedTree) :
    negativeMass tree = relu (-signedSum tree) + totalCorrection tree := by
  have hp := positive_telescope tree
  have hs := signed_sum_eq_mass_difference tree
  have hr := relu_sub_neg (signedSum tree)
  linarith

/-- Componentwise summation of leaf CReLU channels. -/
theorem crelu_mass_identity (tree : OrderedTree) :
    (positiveMass tree, negativeMass tree) =
      ((crelu (signedSum tree)).1 + totalCorrection tree,
       (crelu (signedSum tree)).2 + totalCorrection tree) := by
  apply Prod.ext
  · exact positive_telescope tree
  · exact negative_telescope tree

theorem total_correction_eq_min_masses (tree : OrderedTree) :
    totalCorrection tree = min (positiveMass tree) (negativeMass tree) := by
  rw [positive_telescope tree, negative_telescope tree]
  by_cases hs : 0 ≤ signedSum tree
  · rw [relu_of_nonneg hs,
      relu_of_nonpos (neg_nonpos.mpr hs)]
    simp only [zero_add]
    rw [min_eq_right (by linarith :
      totalCorrection tree ≤ signedSum tree + totalCorrection tree)]
  · have hs' : signedSum tree ≤ 0 := by linarith
    rw [relu_of_nonpos hs',
      relu_of_nonneg (neg_nonneg.mpr hs')]
    simp only [zero_add]
    rw [min_eq_left (by linarith :
      totalCorrection tree ≤ -signedSum tree + totalCorrection tree)]

theorem total_correction_nonneg (tree : OrderedTree) :
    0 ≤ totalCorrection tree := by
  induction tree with
  | leaf value => simp [totalCorrection]
  | node left right ihl ihr =>
      exact add_nonneg (add_nonneg ihl ihr)
        (correction_nonneg (signedSum left) (signedSum right))

end OrderedTree

/-- CReLU after a destructive sum does not recover the missing contrast. -/
theorem crelu_after_sum_counterexample :
    crelu ((1 : ℝ) + (-1)) = crelu (0 + 0) ∧
      relu (1 : ℝ) + relu (-1) ≠ relu 0 + relu 0 := by
  norm_num [crelu, relu, max_def]

/-- A rectified sum and the signed contrast still do not determine the target. -/
theorem clipped_sum_contrast_counterexample :
    (relu ((1 : ℝ) + (-4)), (1 : ℝ) - (-4)) =
      (relu ((2 : ℝ) + (-3)), (2 : ℝ) - (-3)) ∧
    relu (1 : ℝ) + relu (-4) ≠ relu 2 + relu (-3) := by
  norm_num [relu, max_def]

/-- The aggregate is commutative even for two different source orders. -/
theorem merge_order_counterexample :
    merge ((1 : ℝ), 0) (-1, 0) = (0, 1) ∧
    merge ((-1 : ℝ), 0) (1, 0) = (0, 1) ∧
    ((1 : ℝ), (-1 : ℝ)) ≠ ((-1 : ℝ), (1 : ℝ)) := by
  norm_num [merge, correction, relu, max_def]

end SelectiveOverlap
