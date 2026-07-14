"""Synthetic milestone environment for curriculum-planning experiments.

The environment is an analytic skill-vector learner. ``probe_losses`` is the
primary state view: skill ``i`` is represented as ``1.0 - probe_losses[i]``.
Auxiliary sketches have fixed dimensions:

* ``update_sketch``: four values summarizing the most recent update.
* ``optimizer_sketch``: two values for optimizer-step and token-budget scale.
* ``activation_sketch``: mean, max, min, and standard deviation of skills.
* ``exposure_histogram``: one count per cluster, always length 20.
* ``history_embedding``: latest action, normalized step, bridge skill, replay skill.
* ``architecture_embedding``: two constant metadata features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from .simulator import TransitionPrediction
from .state import CurriculumAction, LearningState, TransitionObservation

FloatVector = npt.NDArray[np.float64]

SKILL_COUNT: Final[int] = 8
TOKEN_BUDGET_PER_ACTION: Final[int] = 1_000
UPDATE_SKETCH_DIM: Final[int] = 4
OPTIMIZER_SKETCH_DIM: Final[int] = 2
ACTIVATION_SKETCH_DIM: Final[int] = 4
HISTORY_EMBEDDING_DIM: Final[int] = 4
ARCHITECTURE_EMBEDDING_DIM: Final[int] = 2
REPLAY_SKILL_INDEX: Final[int] = 6
REPLAY_DECAY: Final[float] = 0.18

VALUE_WEIGHTS: Final[FloatVector] = np.asarray(
    [1.0, 1.8, 0.8, 0.9, 0.05, 4.0, 1.2, 0.6],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class ClusterDefinition:
    """Analytic definition of one synthetic data cluster."""

    name: str
    skill_index: int
    base_gain: float
    prerequisites: tuple[tuple[int, float, float], ...] = ()
    decay_targets: tuple[tuple[int, float], ...] = ()
    replay_action: bool = False


class SyntheticEnvironment:
    """State-dependent noncommutative synthetic learner environment."""

    def __init__(self) -> None:
        self._clusters: tuple[ClusterDefinition, ...] = (
            ClusterDefinition("prereq_basic", 0, 0.55),
            ClusterDefinition("prereq_advanced", 1, 0.75, ((0, 0.50, 0.05),)),
            ClusterDefinition("destructive_x", 2, 0.62, (), ((3, 0.45),)),
            ClusterDefinition("fragile_y", 3, 0.65),
            ClusterDefinition("bridge", 4, 0.82),
            ClusterDefinition("bridge_payload", 5, 0.92, ((4, 0.70, 0.02),)),
            ClusterDefinition("replay_foundation", 6, 0.65, (), (), True),
            ClusterDefinition("replay_review", 6, 0.42, (), (), True),
            ClusterDefinition("core_math", 0, 0.28),
            ClusterDefinition("language_patterns", 1, 0.22),
            ClusterDefinition("logic_drills", 2, 0.34),
            ClusterDefinition("memory_palace", 3, 0.25),
            ClusterDefinition("synthesis", 5, 0.32, ((2, 0.60, 0.04),)),
            ClusterDefinition("calibration", 7, 0.45),
            ClusterDefinition("examples", 0, 0.24),
            ClusterDefinition("proofs", 1, 0.30, ((0, 0.50, 0.15),)),
            ClusterDefinition("contrastive", 2, 0.27),
            ClusterDefinition("safety", 7, 0.35),
            ClusterDefinition("compression", 7, 0.25),
            ClusterDefinition("review_mix", 6, 0.18, (), (), True),
        )
        self._cluster_index: dict[str, int] = {
            cluster.name: index for index, cluster in enumerate(self._clusters)
        }

    @property
    def cluster_names(self) -> tuple[str, ...]:
        """Return cluster names in canonical action order."""

        return tuple(cluster.name for cluster in self._clusters)

    @property
    def skill_count(self) -> int:
        """Return the number of analytic skills."""

        return SKILL_COUNT

    @property
    def cluster_count(self) -> int:
        """Return the number of named data clusters."""

        return len(self._clusters)

    @property
    def token_budget_per_action(self) -> int:
        """Return the fixed token budget used by milestone actions."""

        return TOKEN_BUDGET_PER_ACTION

    def actions(self) -> list[CurriculumAction]:
        """Return all single-cluster actions in canonical order."""

        return [self.action_by_name(name) for name in self.cluster_names]

    def action_by_name(self, name: str) -> CurriculumAction:
        """Construct the canonical single-cluster action for ``name``."""

        if name not in self._cluster_index:
            raise ValueError(f"unknown cluster: {name}")
        return CurriculumAction(
            cluster_ids=(name,),
            mixture_weights=(1.0,),
            optimizer_steps=1,
            token_budget=TOKEN_BUDGET_PER_ACTION,
        )

    def initial_state(self) -> LearningState:
        """Return the all-unlearned initial learner state."""

        skills: FloatVector = np.zeros(SKILL_COUNT, dtype=np.float64)
        exposure: FloatVector = np.zeros(self.cluster_count, dtype=np.float64)
        return self._state_from_skills(
            skills=skills,
            step=0,
            tokens_seen=0,
            exposure_histogram=exposure,
            action_index=-1,
            skill_delta=np.zeros(SKILL_COUNT, dtype=np.float64),
            learning_rate_scale=1.0,
        )

    def value(self, state: LearningState) -> float:
        """Return negative weighted probe loss; higher values are better."""

        self._validate_state(state)
        if state.probe_losses.shape != VALUE_WEIGHTS.shape:
            raise ValueError("probe loss and value-weight dimensions do not match")
        return -float(np.sum(VALUE_WEIGHTS * state.probe_losses))

    def step(
        self,
        state: LearningState,
        action: CurriculumAction,
        rng: np.random.Generator,
    ) -> LearningState:
        """Apply the true environment transition.

        The current milestone dynamics are deterministic; the generator is still
        explicit so future stochastic extensions cannot introduce hidden global
        randomness.
        """

        _ = rng
        return self._deterministic_step(state, action)

    def predict(
        self,
        state: LearningState,
        action: CurriculumAction,
    ) -> TransitionPrediction:
        """Expose the true transition as a deterministic simulator."""

        next_state = self._deterministic_step(state, action)
        probe_delta = next_state.probe_losses - state.probe_losses
        forgetting = float(np.clip(np.sum(np.maximum(probe_delta, 0.0)), 0.0, 1.0))
        return TransitionPrediction(
            next_state_mean=next_state,
            probe_delta_mean=probe_delta,
            probe_delta_std=np.full(SKILL_COUNT, 1.0e-9, dtype=np.float64),
            forgetting_risk=forgetting,
            novelty=0.0,
            expected_compute=float(action.optimizer_steps * action.token_budget),
        )

    def oracle_rollout(
        self,
        actions: Sequence[CurriculumAction],
        seed: int,
    ) -> list[TransitionObservation]:
        """Run a true-environment rollout and return oracle observations."""

        rng = np.random.default_rng(seed)
        state = self.initial_state()
        observations: list[TransitionObservation] = []
        for step_index, action in enumerate(actions):
            next_state = self.step(state, action, rng)
            observations.append(
                self._observation(
                    state,
                    action,
                    next_state,
                    {
                        "seed": seed,
                        "rollout": 0,
                        "step": step_index,
                        "action_index": self._action_index(action),
                    },
                )
            )
            state = next_state
        return observations

    def generate_oracle_rollouts(
        self,
        *,
        n_rollouts: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> list[TransitionObservation]:
        """Generate oracle transition observations from templates and random rollouts."""

        if n_rollouts < 0:
            raise ValueError("n_rollouts must be non-negative")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        observations: list[TransitionObservation] = []
        for template_index, template in enumerate(self._template_sequences(horizon)):
            state = self.initial_state()
            for step_index, name in enumerate(template[:horizon]):
                action = self.action_by_name(name)
                next_state = self.step(state, action, rng)
                observations.append(
                    self._observation(
                        state,
                        action,
                        next_state,
                        {
                            "seed": 0,
                            "rollout": -template_index - 1,
                            "step": step_index,
                            "action_index": self._action_index(action),
                        },
                    )
                )
                state = next_state

        action_space = self.actions()
        for rollout_index in range(n_rollouts):
            state = self.initial_state()
            for step_index in range(horizon):
                action_index = int(rng.integers(0, len(action_space)))
                action = action_space[action_index]
                next_state = self.step(state, action, rng)
                observations.append(
                    self._observation(
                        state,
                        action,
                        next_state,
                        {
                            "seed": rollout_index,
                            "rollout": rollout_index,
                            "step": step_index,
                            "action_index": action_index,
                        },
                    )
                )
                state = next_state

        return observations

    def generate_oracle_transitions(
        self,
        *,
        n_rollouts: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> list[TransitionObservation]:
        """Alias for ``generate_oracle_rollouts``."""

        return self.generate_oracle_rollouts(
            n_rollouts=n_rollouts,
            horizon=horizon,
            rng=rng,
        )

    def _deterministic_step(
        self,
        state: LearningState,
        action: CurriculumAction,
    ) -> LearningState:
        self._validate_state(state)
        action_index = self._action_index(action)
        definition = self._clusters[action_index]
        skills_before = self._skills_from_state(state)
        skills_after = skills_before.copy()

        if not definition.replay_action:
            skills_after[REPLAY_SKILL_INDEX] *= 1.0 - REPLAY_DECAY

        gate = self._prerequisite_gate(skills_after, definition)
        target = definition.skill_index
        gain = (
            action.learning_rate_scale
            * definition.base_gain
            * gate
            * (1.0 - skills_after[target])
        )
        skills_after[target] = np.clip(skills_after[target] + gain, 0.0, 1.0)

        for decay_index, retain_fraction in definition.decay_targets:
            skills_after[decay_index] *= retain_fraction

        skills_after = np.clip(skills_after, 0.0, 1.0)
        exposure = state.exposure_histogram.copy()
        exposure[action_index] += 1.0
        skill_delta = skills_after - skills_before

        return self._state_from_skills(
            skills=skills_after,
            step=state.step + 1,
            tokens_seen=state.tokens_seen + action.token_budget,
            exposure_histogram=exposure,
            action_index=action_index,
            skill_delta=skill_delta,
            learning_rate_scale=action.learning_rate_scale,
        )

    def _prerequisite_gate(
        self,
        skills: FloatVector,
        definition: ClusterDefinition,
    ) -> float:
        gate = 1.0
        for skill_index, threshold, ineffective_gate in definition.prerequisites:
            if threshold <= 0.0:
                raise ValueError("prerequisite threshold must be positive")
            readiness = float(np.clip(skills[skill_index] / threshold, 0.0, 1.0))
            local_gate = ineffective_gate + (1.0 - ineffective_gate) * readiness
            gate = min(gate, local_gate)
        return gate

    def _observation(
        self,
        state_before: LearningState,
        action: CurriculumAction,
        state_after: LearningState,
        seed_metadata: dict[str, int],
    ) -> TransitionObservation:
        probe_delta = state_after.probe_losses - state_before.probe_losses
        skills_before = self._skills_from_state(state_before)
        skills_after = self._skills_from_state(state_after)
        parameter_delta = skills_after - skills_before
        activation_delta = state_after.activation_sketch - state_before.activation_sketch
        return TransitionObservation(
            state_before=state_before,
            action=action,
            state_after=state_after,
            parameter_delta_sketch=parameter_delta,
            probe_delta=probe_delta,
            activation_delta_sketch=activation_delta,
            compute_cost=float(action.optimizer_steps * action.token_budget),
            seed_metadata=seed_metadata,
            simulator_version="synthetic-v1",
        )

    def _template_sequences(self, horizon: int) -> list[tuple[str, ...]]:
        one_step: list[tuple[str, ...]] = [(name,) for name in self.cluster_names]
        templates: list[tuple[str, ...]] = [
            ("prereq_basic", "prereq_advanced", "proofs"),
            ("prereq_advanced", "prereq_basic", "prereq_advanced"),
            ("fragile_y", "destructive_x"),
            ("destructive_x", "fragile_y"),
            ("bridge", "bridge_payload", "replay_foundation", "prereq_basic"),
            ("bridge_payload", "bridge", "bridge_payload"),
            ("replay_foundation", "core_math", "logic_drills", "calibration"),
            ("replay_foundation", "core_math", "logic_drills", "replay_review"),
            ("logic_drills", "synthesis"),
            ("bridge", "bridge_payload", "fragile_y", "prereq_basic", "prereq_advanced"),
        ]
        extension = ("core_math", "language_patterns", "calibration", "examples")
        padded: list[tuple[str, ...]] = []
        for sequence in one_step + templates:
            values = list(sequence)
            extension_index = 0
            while len(values) < horizon:
                values.append(extension[extension_index % len(extension)])
                extension_index += 1
            padded.append(tuple(values))
        return padded

    def _state_from_skills(
        self,
        *,
        skills: FloatVector,
        step: int,
        tokens_seen: int,
        exposure_histogram: FloatVector,
        action_index: int,
        skill_delta: FloatVector,
        learning_rate_scale: float,
    ) -> LearningState:
        if skills.shape != (SKILL_COUNT,):
            raise ValueError("skills must have shape (8,)")
        if skill_delta.shape != (SKILL_COUNT,):
            raise ValueError("skill_delta must have shape (8,)")
        if exposure_histogram.shape != (self.cluster_count,):
            raise ValueError("exposure_histogram must have one entry per cluster")

        clipped_skills = np.clip(np.asarray(skills, dtype=np.float64), 0.0, 1.0)
        losses = 1.0 - clipped_skills
        action_norm = self._normalized_action_index(action_index)

        positive_delta = np.maximum(skill_delta, 0.0)
        negative_delta = np.maximum(-skill_delta, 0.0)

        update_sketch = np.asarray(
            [
                float(np.sum(positive_delta)),
                float(np.sum(negative_delta)),
                action_norm,
                learning_rate_scale,
            ],
            dtype=np.float64,
        )
        optimizer_sketch = np.asarray(
            [1.0 if step > 0 else 0.0, TOKEN_BUDGET_PER_ACTION / 1_000.0],
            dtype=np.float64,
        )
        activation_sketch = np.asarray(
            [
                float(np.mean(clipped_skills)),
                float(np.max(clipped_skills)),
                float(np.min(clipped_skills)),
                float(np.std(clipped_skills)),
            ],
            dtype=np.float64,
        )
        history_embedding = np.asarray(
            [
                action_norm,
                step / 50.0,
                clipped_skills[4],
                clipped_skills[REPLAY_SKILL_INDEX],
            ],
            dtype=np.float64,
        )
        architecture_embedding = np.asarray(
            [SKILL_COUNT / 10.0, self.cluster_count / 20.0],
            dtype=np.float64,
        )

        return LearningState(
            probe_losses=losses.astype(np.float64),
            update_sketch=update_sketch,
            optimizer_sketch=optimizer_sketch,
            activation_sketch=activation_sketch,
            exposure_histogram=exposure_histogram.astype(np.float64),
            history_embedding=history_embedding,
            architecture_embedding=architecture_embedding,
            step=step,
            tokens_seen=tokens_seen,
        )

    def _skills_from_state(self, state: LearningState) -> FloatVector:
        self._validate_state(state)
        return np.clip(1.0 - state.probe_losses, 0.0, 1.0).astype(np.float64)

    def _validate_state(self, state: LearningState) -> None:
        if state.probe_losses.shape != (SKILL_COUNT,):
            raise ValueError("probe_losses must have shape (8,)")
        if state.update_sketch.shape != (UPDATE_SKETCH_DIM,):
            raise ValueError("update_sketch must have shape (4,)")
        if state.optimizer_sketch.shape != (OPTIMIZER_SKETCH_DIM,):
            raise ValueError("optimizer_sketch must have shape (2,)")
        if state.activation_sketch.shape != (ACTIVATION_SKETCH_DIM,):
            raise ValueError("activation_sketch must have shape (4,)")
        if state.exposure_histogram.shape != (self.cluster_count,):
            raise ValueError("exposure_histogram must have shape (20,)")
        if state.history_embedding.shape != (HISTORY_EMBEDDING_DIM,):
            raise ValueError("history_embedding must have shape (4,)")
        if state.architecture_embedding.shape != (ARCHITECTURE_EMBEDDING_DIM,):
            raise ValueError("architecture_embedding must have shape (2,)")
        if state.step < 0:
            raise ValueError("step must be non-negative")
        if state.tokens_seen < 0:
            raise ValueError("tokens_seen must be non-negative")

    def _action_index(self, action: CurriculumAction) -> int:
        if len(action.cluster_ids) != 1 or action.mixture_weights != (1.0,):
            raise ValueError("synthetic milestone supports single-cluster actions only")
        name = action.cluster_ids[0]
        if name not in self._cluster_index:
            raise ValueError(f"unknown cluster: {name}")
        return self._cluster_index[name]

    def _normalized_action_index(self, action_index: int) -> float:
        if action_index < 0:
            return -1.0
        denominator = max(self.cluster_count - 1, 1)
        return action_index / denominator
