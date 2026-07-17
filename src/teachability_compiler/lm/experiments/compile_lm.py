"""Language-model curriculum race driver.

This module runs fixed-token-budget curriculum policies against NanochatLearnerOracle and
records the hidden validation channel only in the trajectory recorder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

THRESHOLD_FACTORS: tuple[float, ...] = (1.10, 1.05, 1.02, 1.01, 1.005, 1.0)
SIMULATOR_VERSION = "lowrank-v1"
ENVIRONMENT_VERSION = "lm-v1"


def weights_for_edu_heavy(manifest: dict[str, Any], action_names: list[str]) -> tuple[float, ...]:
    """Construct a crude FineWeb-Edu-style quality-tilted mixture.

    Base weights are proportional to each action's train token count. The 9
    ``semantic_*_hi`` actions receive a 3x multiplier, while ``code_heavy`` and
    ``math_heavy`` receive a 2x multiplier, then all weights are normalized.
    """

    actions = manifest.get("actions", {})
    raw_weights: list[float] = []

    for name in action_names:
        if name not in actions or "train_tokens" not in actions[name]:
            raise KeyError(f"manifest action {name!r} is missing train_tokens")
        train_tokens = float(actions[name]["train_tokens"])
        if train_tokens < 0:
            raise ValueError(f"manifest action {name!r} has negative train_tokens")

        multiplier = 1.0
        if name.startswith("semantic_") and name.endswith("_hi"):
            multiplier = 3.0
        elif name in {"code_heavy", "math_heavy"}:
            multiplier = 2.0

        raw_weights.append(train_tokens * multiplier)

    total = float(sum(raw_weights))
    if total <= 0.0:
        raise ValueError("edu_heavy weights have zero total mass")
    return tuple(float(weight / total) for weight in raw_weights)


def damped_worst_probe_choice(
    probe_losses: np.ndarray | list[float],
    recency: np.ndarray | list[float],
    penalty: float,
) -> int:
    """Return argmax_i probe_loss_i - penalty * recency_i."""

    losses = np.asarray(probe_losses, dtype=np.float64)
    recency_arr = np.asarray(recency, dtype=np.float64)
    if losses.ndim != 1 or recency_arr.ndim != 1:
        raise ValueError("probe_losses and recency must be one-dimensional")
    if losses.shape != recency_arr.shape:
        raise ValueError("probe_losses and recency must have the same shape")
    if losses.size == 0:
        raise ValueError("cannot choose from an empty action set")

    scores = losses - float(penalty) * recency_arr
    return int(np.argmax(scores))


def update_recency(
    recency: np.ndarray | list[float],
    chosen_index: int,
    decay: float,
) -> np.ndarray:
    """Decay all recency masses and add one unit to the chosen action."""

    recency_arr = np.asarray(recency, dtype=np.float64).copy()
    if recency_arr.ndim != 1:
        raise ValueError("recency must be one-dimensional")
    if not 0 <= chosen_index < recency_arr.size:
        raise IndexError("chosen_index is outside the recency vector")
    recency_arr *= float(decay)
    recency_arr[int(chosen_index)] += 1.0
    return recency_arr


@dataclass
class SwitchState:
    """EMA-based staged-policy switch state.

    ``update`` returns True only on the call that newly fires the permanent switch.
    """

    threshold: float
    min_chunks: int
    alpha: float
    ema: float | None = None
    chunks_seen: int = 0
    switched: bool = False
    switch_chunk: int | None = None

    def update(self, improvement: float, chunk: int | None = None) -> bool:
        if self.switched:
            return False

        if chunk is None:
            self.chunks_seen += 1
            check_chunk = self.chunks_seen
        else:
            check_chunk = int(chunk)
            self.chunks_seen = max(self.chunks_seen + 1, check_chunk)

        improvement_value = float(improvement)
        if self.ema is None:
            self.ema = improvement_value
        else:
            self.ema = float(self.alpha) * improvement_value + (1.0 - float(self.alpha)) * self.ema

        if check_chunk >= int(self.min_chunks) and self.ema < float(self.threshold):
            self.switched = True
            self.switch_chunk = check_chunk
            return True
        return False


def tokens_to_threshold(trajectory: list[dict[str, Any]], threshold: float) -> int | None:
    """First token count whose hidden validation BPB is at or below threshold."""

    for point in trajectory:
        val_bpb = point.get("val_bpb")
        if val_bpb is not None and float(val_bpb) <= float(threshold):
            return int(point["tokens"])
    return None


def _threshold_labels() -> list[str]:
    return [f"{factor:g}x" for factor in THRESHOLD_FACTORS]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _config_hash(args: argparse.Namespace) -> str:
    payload = json.dumps(vars(args), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _curriculum_action(
    cluster_ids: tuple[str, ...],
    mixture_weights: tuple[float, ...],
    optimizer_steps: int,
    token_budget: int,
) -> Any:
    from teachability_compiler.state import CurriculumAction

    return CurriculumAction(
        cluster_ids=cluster_ids,
        mixture_weights=mixture_weights,
        optimizer_steps=int(optimizer_steps),
        token_budget=int(token_budget),
    )


def _single_action(name: str, optimizer_steps: int, tokens_per_chunk: int) -> Any:
    return _curriculum_action((name,), (1.0,), optimizer_steps, tokens_per_chunk)


def _mixture_action(
    action_names: list[str],
    weights: tuple[float, ...],
    optimizer_steps: int,
    tokens_per_chunk: int,
) -> Any:
    if len(weights) != len(action_names):
        raise ValueError("mixture weight count does not match action count")
    total = float(sum(weights))
    if not np.isclose(total, 1.0):
        raise ValueError(f"mixture weights must sum to 1, got {total}")
    return _curriculum_action(
        tuple(action_names),
        tuple(float(w) for w in weights),
        optimizer_steps,
        tokens_per_chunk,
    )


def _state_probe_losses(state: Any) -> np.ndarray:
    losses = getattr(state, "probe_losses", None)
    if losses is None and isinstance(state, dict):
        losses = state.get("probe_losses")
    if losses is None:
        raise AttributeError("LearningState-like object does not expose probe_losses")
    return np.asarray(losses, dtype=np.float64)


def _probe_mean_from_losses(losses: np.ndarray | list[float]) -> float:
    return float(np.mean(np.asarray(losses, dtype=np.float64)))


def _initial_overhead() -> dict[str, float | int]:
    return {
        "probe_calls": 0,
        "probe_wall_seconds": 0.0,
        "simulator_fit_wall_seconds": 0.0,
        "simulator_predict_calls": 0,
        "wall_seconds": 0.0,
    }


def _metered_probe_losses(oracle: Any, overhead: dict[str, float | int]) -> np.ndarray:
    start = time.perf_counter()
    losses = np.asarray(oracle.probe_losses(), dtype=np.float64)
    overhead["probe_calls"] = int(overhead["probe_calls"]) + 1
    overhead["probe_wall_seconds"] = float(overhead["probe_wall_seconds"]) + (
        time.perf_counter() - start
    )
    return losses


def _metered_encode_state(oracle: Any, overhead: dict[str, float | int]) -> Any:
    start = time.perf_counter()
    state = oracle.encode_state()
    overhead["probe_calls"] = int(overhead["probe_calls"]) + 1
    overhead["probe_wall_seconds"] = float(overhead["probe_wall_seconds"]) + (
        time.perf_counter() - start
    )
    return state


def _fit_simulator(
    action_names: list[str],
    observations: list[Any],
    args: argparse.Namespace,
    epochs: int,
    overhead: dict[str, float | int],
) -> Any:
    if not observations:
        return None

    from teachability_compiler.lm.lowrank_simulator import LowRankTransitionPredictor

    start = time.perf_counter()
    simulator = LowRankTransitionPredictor(
        action_names,
        rank=int(args.sim_rank),
        device="cpu",
        seed=int(args.seed),
    )
    simulator.fit(
        observations,
        epochs=int(epochs),
        lr=1e-3,
        batch_size=256,
        weight_decay=1e-4,
        ranking_weight=0.1,
    )
    overhead["simulator_fit_wall_seconds"] = float(overhead["simulator_fit_wall_seconds"]) + (
        time.perf_counter() - start
    )
    return simulator


def _simulator_choice(
    simulator: Any,
    state: Any,
    action_names: list[str],
    args: argparse.Namespace,
    overhead: dict[str, float | int],
    rng: np.random.Generator,
    tokens_per_chunk: int,
) -> int:
    if rng.random() < float(args.explore_epsilon):
        return int(rng.integers(len(action_names)))

    best_index = 0
    best_value = -float("inf")
    for index, name in enumerate(action_names):
        action = _single_action(name, int(args.steps_per_chunk), tokens_per_chunk)
        prediction = simulator.predict(state, action)
        overhead["simulator_predict_calls"] = int(overhead["simulator_predict_calls"]) + 1
        predicted_losses = _state_probe_losses(prediction.next_state_mean)
        value = -_probe_mean_from_losses(predicted_losses)
        if value > best_value:
            best_index = index
            best_value = value
    return int(best_index)


def _record_trajectory_point(
    oracle: Any,
    trajectory: list[dict[str, Any]],
    chunk_index: int,
    cached_probe_mean: float | None,
    overhead: dict[str, float | int],
) -> dict[str, Any]:
    """Record hidden metrics after the policy has chosen and applied an action."""

    val_bpb = float(oracle.hidden_val_bpb(max_batches=8))
    holdout_ce = float(oracle.hidden_holdout_ce(max_batches=8))
    if cached_probe_mean is None:
        losses = _metered_probe_losses(oracle, overhead)
        probe_mean = _probe_mean_from_losses(losses)
    else:
        probe_mean = float(cached_probe_mean)

    point = {
        "chunk": int(chunk_index),
        "tokens": int(oracle.tokens_seen),
        "step": int(oracle.step),
        "val_bpb": val_bpb,
        "holdout_ce": holdout_ce,
        "probe_mean": probe_mean,
    }
    trajectory.append(point)
    return point


def _checkpoint_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "latest.pt"


def _load_checkpoint(ckpt_dir: Path, device: str) -> dict[str, Any]:
    import torch

    path = _checkpoint_path(ckpt_dir)
    if not path.exists():
        raise FileNotFoundError(f"resume requested but checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_checkpoint(
    ckpt_dir: Path,
    snapshot: Any,
    trajectory: list[dict[str, Any]],
    chunk_index: int,
    policy_state: dict[str, Any],
    transition_pool: list[Any] | None = None,
) -> None:
    import torch

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot": snapshot,
        "trajectory": trajectory,
        "chunk_index": int(chunk_index),
        "policy_state": policy_state,
        # Compiler/staged: the simulator's training evidence must survive a
        # crash, otherwise a resumed run is a different policy.
        "transition_pool": list(transition_pool or []),
    }
    # Keep exactly one checkpoint, written atomically: per-chunk archives
    # are ~3.1 GB each and filled the disk in production.
    target = _checkpoint_path(ckpt_dir)
    tmp = target.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def _initial_policy_state(num_actions: int) -> dict[str, Any]:
    return {
        "recency": [0.0] * num_actions,
        "gain_ema": None,
        "switch_flag": False,
        "switch_chunk": None,
        "switch_chunks_seen": 0,
        "bootstrap_observations": 0,
        "prev_probe_mean": None,
        "action_sequence": [],
        "actions_executed": {},
        "sim_initial_fit_done": False,
        "sim_last_fit_chunk": None,
        "overhead": _initial_overhead(),
    }


def _increment_action(policy_state: dict[str, Any], chosen_name: str) -> None:
    sequence = policy_state.setdefault("action_sequence", [])
    counts = policy_state.setdefault("actions_executed", {})
    sequence.append(chosen_name)
    counts[chosen_name] = int(counts.get(chosen_name, 0)) + 1


def _normalise_policy_state(policy_state: dict[str, Any], num_actions: int) -> dict[str, Any]:
    state = _initial_policy_state(num_actions)
    state.update(policy_state)
    recency = list(state.get("recency", []))
    if len(recency) != num_actions:
        recency = [0.0] * num_actions
    state["recency"] = [float(value) for value in recency]
    state["action_sequence"] = list(state.get("action_sequence", []))
    state["actions_executed"] = {
        str(key): int(value) for key, value in dict(state.get("actions_executed", {})).items()
    }

    overhead = _initial_overhead()
    overhead.update(dict(state.get("overhead", {})))
    overhead["probe_calls"] = int(overhead["probe_calls"])
    overhead["simulator_predict_calls"] = int(overhead["simulator_predict_calls"])
    state["overhead"] = overhead
    return state


def _build_output(
    args: argparse.Namespace,
    trajectory: list[dict[str, Any]],
    policy_state: dict[str, Any],
    target_data: dict[str, Any],
    oracle_manifest: dict[str, Any],
    overhead: dict[str, float | int],
    config_hash: str,
    git_commit: str,
    resumed: bool,
    notes: list[str],
) -> dict[str, Any]:
    target_provenance = target_data.get("provenance", {})
    simulator_version = SIMULATOR_VERSION if args.policy in {"compiler", "staged"} else "none"
    return {
        "kind": "lm_race",
        "policy": args.policy,
        "seed": int(args.seed),
        "token_budget": int(args.token_budget),
        "steps_per_chunk": int(args.steps_per_chunk),
        "trajectory": trajectory,
        "actions_executed": dict(policy_state.get("actions_executed", {})),
        "action_sequence": list(policy_state.get("action_sequence", [])),
        "switch_chunk": policy_state.get("switch_chunk"),
        "resumed": bool(resumed),
        "target": target_data["target"],
        "provenance": {
            "seed": int(args.seed),
            "config_hash": config_hash,
            "git_commit": git_commit,
            "tokens_manifest_hash": oracle_manifest.get("manifest_hash"),
            "actions_manifest_hash": oracle_manifest.get("actions_manifest_hash"),
            "target_config_hash": target_provenance.get("config_hash"),
            "simulator_version": simulator_version,
            "environment_version": ENVIRONMENT_VERSION,
        },
        "overhead": {
            "probe_calls": int(overhead["probe_calls"]),
            "probe_wall_seconds": float(overhead["probe_wall_seconds"]),
            "simulator_fit_wall_seconds": float(overhead["simulator_fit_wall_seconds"]),
            "simulator_predict_calls": int(overhead["simulator_predict_calls"]),
            "wall_seconds": float(overhead["wall_seconds"]),
        },
        "notes": notes,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.token_budget <= 0:
        raise ValueError("--token-budget must be positive")
    if args.steps_per_chunk <= 0:
        raise ValueError("--steps-per-chunk must be positive")
    if args.record_every_chunks <= 0:
        raise ValueError("--record-every-chunks must be positive")
    if args.ckpt_every_chunks <= 0:
        raise ValueError("--ckpt-every-chunks must be positive")
    if args.bootstrap_chunks < 0:
        raise ValueError("--bootstrap-chunks must be non-negative")
    if args.refit_every_chunks <= 0:
        raise ValueError("--refit-every-chunks must be positive")
    if not 0.0 <= args.explore_epsilon <= 1.0:
        raise ValueError("--explore-epsilon must be in [0, 1]")
    if not 0.0 < args.gain_ema_alpha <= 1.0:
        raise ValueError("--gain-ema-alpha must be in (0, 1]")


def run_race(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)

    import torch
    from teachability_compiler.lm.lm_oracle import NanochatLearnerOracle

    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    target_path = Path(args.target_file)
    if not target_path.exists():
        raise FileNotFoundError(f"target file does not exist: {target_path}")
    target_data = _load_json(target_path)
    if "target" not in target_data:
        raise KeyError(f"target file {target_path} is missing a 'target' block")

    oracle = NanochatLearnerOracle(
        args.tokens_dir,
        depth=int(args.depth),
        device=args.device,
        seq_len=int(args.seq_len),
        device_batch=int(args.device_batch),
        grad_accum=int(args.grad_accum),
        base_seed=int(args.seed),
    )

    target_provenance = target_data.get("provenance", {})
    target_manifest_hash = target_provenance.get("tokens_manifest_hash")
    oracle_manifest_hash = oracle.manifest.get("manifest_hash")
    if target_manifest_hash != oracle_manifest_hash:
        raise RuntimeError(
            "target provenance tokens_manifest_hash does not match oracle manifest_hash: "
            f"{target_manifest_hash!r} != {oracle_manifest_hash!r}"
        )

    action_names = list(oracle.action_names)
    num_actions = len(action_names)
    if num_actions == 0:
        raise RuntimeError("oracle exposes no actions")

    tokens_per_chunk = int(args.steps_per_chunk) * int(args.device_batch)
    tokens_per_chunk *= int(args.grad_accum) * int(args.seq_len)

    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out)
    config_hash = _config_hash(args)
    git_commit = _git_commit()
    notes: list[str] = []

    restored_pool: list[Any] = []
    if args.resume:
        checkpoint = _load_checkpoint(ckpt_dir, args.device)
        oracle.restore(checkpoint["snapshot"])
        trajectory = list(checkpoint.get("trajectory", []))
        chunk_index = int(checkpoint.get("chunk_index", len(trajectory)))
        policy_state = _normalise_policy_state(
            dict(checkpoint.get("policy_state", {})),
            num_actions,
        )
        restored_pool = list(checkpoint.get("transition_pool", []))
        notes.append("Resumed from checkpoint.")
        if args.policy in {"compiler", "staged"}:
            notes.append(
                f"Resume restored the transition pool ({len(restored_pool)} observations); "
                "the simulator refits on the full pre+post-resume evidence."
            )
        resumed = True
    else:
        trajectory = []
        chunk_index = 0
        policy_state = _initial_policy_state(num_actions)
        resumed = False

    recency = np.asarray(policy_state["recency"], dtype=np.float64)
    bootstrap_count = int(policy_state.get("bootstrap_observations", 0))
    prev_probe_mean = policy_state.get("prev_probe_mean")
    if prev_probe_mean is not None:
        prev_probe_mean = float(prev_probe_mean)

    switch_state = SwitchState(
        threshold=float(args.switch_threshold),
        min_chunks=int(args.switch_min_chunks),
        alpha=float(args.gain_ema_alpha),
        ema=policy_state.get("gain_ema"),
        chunks_seen=int(policy_state.get("switch_chunks_seen", 0)),
        switched=bool(policy_state.get("switch_flag", False)),
        switch_chunk=policy_state.get("switch_chunk"),
    )

    overhead = dict(policy_state.get("overhead", _initial_overhead()))
    wall_offset = float(overhead.get("wall_seconds", 0.0))
    wall_start = time.perf_counter()

    transition_pool: list[Any] = list(restored_pool)
    simulator: Any = None
    sim_initial_fit_done = bool(policy_state.get("sim_initial_fit_done", False))
    sim_last_fit_chunk = policy_state.get("sim_last_fit_chunk")
    if sim_last_fit_chunk is not None:
        sim_last_fit_chunk = int(sim_last_fit_chunk)

    if args.policy == "uniform":
        uniform_weights = tuple(1.0 / num_actions for _ in action_names)
        fixed_action = _mixture_action(
            action_names,
            uniform_weights,
            int(args.steps_per_chunk),
            tokens_per_chunk,
        )
    elif args.policy == "edu_heavy":
        edu_weights = weights_for_edu_heavy(oracle.manifest, action_names)
        fixed_action = _mixture_action(
            action_names,
            edu_weights,
            int(args.steps_per_chunk),
            tokens_per_chunk,
        )
    else:
        fixed_action = None

    def current_overhead() -> dict[str, float | int]:
        snapshot = dict(overhead)
        snapshot["wall_seconds"] = wall_offset + (time.perf_counter() - wall_start)
        return snapshot

    def sync_policy_state() -> None:
        policy_state["recency"] = [float(value) for value in recency]
        policy_state["gain_ema"] = switch_state.ema
        policy_state["switch_flag"] = bool(switch_state.switched)
        policy_state["switch_chunk"] = switch_state.switch_chunk
        policy_state["switch_chunks_seen"] = int(switch_state.chunks_seen)
        policy_state["bootstrap_observations"] = int(bootstrap_count)
        policy_state["prev_probe_mean"] = prev_probe_mean
        policy_state["sim_initial_fit_done"] = bool(sim_initial_fit_done)
        policy_state["sim_last_fit_chunk"] = sim_last_fit_chunk
        policy_state["overhead"] = current_overhead()

    def write_output() -> dict[str, Any]:
        sync_policy_state()
        payload = _build_output(
            args,
            trajectory,
            policy_state,
            target_data,
            oracle.manifest,
            current_overhead(),
            config_hash,
            git_commit,
            resumed,
            notes,
        )
        _atomic_write_json(out_path, payload)
        return payload

    last_chunk_probe_mean: float | None = None
    last_recorded_chunk = int(trajectory[-1]["chunk"]) if trajectory else -1

    while int(oracle.tokens_seen) < int(args.token_budget):
        chunk_start_time = time.perf_counter()
        visible_probe_mean: float | None = None
        chosen_name = "mixture"
        was_bootstrap_chunk = False

        if args.policy in {"uniform", "edu_heavy"}:
            action = fixed_action
            if action is None:
                raise RuntimeError("fixed-action policy did not construct its action")

        elif args.policy == "worst_probe":
            losses = _metered_probe_losses(oracle, overhead)
            visible_probe_mean = _probe_mean_from_losses(losses)
            chosen_index = damped_worst_probe_choice(losses, recency, args.recency_penalty)
            recency = update_recency(recency, chosen_index, args.recency_decay)
            chosen_name = action_names[chosen_index]
            action = _single_action(chosen_name, int(args.steps_per_chunk), tokens_per_chunk)

        elif args.policy in {"compiler", "staged"}:
            in_switched_stage = args.policy == "staged" and switch_state.switched

            if in_switched_stage:
                losses = _metered_probe_losses(oracle, overhead)
                visible_probe_mean = _probe_mean_from_losses(losses)
                chosen_index = damped_worst_probe_choice(losses, recency, args.recency_penalty)
                recency = update_recency(recency, chosen_index, args.recency_decay)
                chosen_name = action_names[chosen_index]
                action = _single_action(chosen_name, int(args.steps_per_chunk), tokens_per_chunk)

            elif bootstrap_count < int(args.bootstrap_chunks):
                was_bootstrap_chunk = True
                # Coverage-first bootstrap: walk a seeded permutation of the
                # action set (cycling) so the simulator's per-action anchors
                # see as many distinct actions as the bootstrap budget allows.
                perm = np.random.default_rng(args.seed).permutation(num_actions)
                chosen_index = int(perm[bootstrap_count % num_actions])
                chosen_name = action_names[chosen_index]
                action = _single_action(chosen_name, int(args.steps_per_chunk), tokens_per_chunk)

            else:
                if transition_pool and simulator is None:
                    epochs = args.sim_epochs_initial
                    if sim_initial_fit_done:
                        epochs = args.sim_epochs_refit
                    simulator = _fit_simulator(
                        action_names,
                        transition_pool,
                        args,
                        epochs,
                        overhead,
                    )
                    sim_initial_fit_done = True
                    sim_last_fit_chunk = chunk_index
                elif (
                    transition_pool
                    and simulator is not None
                    and sim_last_fit_chunk is not None
                    and chunk_index - sim_last_fit_chunk >= int(args.refit_every_chunks)
                ):
                    simulator = _fit_simulator(
                        action_names,
                        transition_pool,
                        args,
                        args.sim_epochs_refit,
                        overhead,
                    )
                    sim_last_fit_chunk = chunk_index

                if simulator is None:
                    chosen_index = int(rng.integers(num_actions))
                    chosen_name = action_names[chosen_index]
                    action = _single_action(
                        chosen_name, int(args.steps_per_chunk), tokens_per_chunk
                    )
                else:
                    state = _metered_encode_state(oracle, overhead)
                    state_losses = _state_probe_losses(state)
                    visible_probe_mean = _probe_mean_from_losses(state_losses)

                    if args.policy == "staged":
                        if prev_probe_mean is not None:
                            improvement = float(prev_probe_mean) - visible_probe_mean
                            switch_state.update(improvement, chunk=chunk_index)
                        prev_probe_mean = visible_probe_mean

                    if args.policy == "staged" and switch_state.switched:
                        chosen_index = damped_worst_probe_choice(
                            state_losses,
                            recency,
                            args.recency_penalty,
                        )
                        recency = update_recency(recency, chosen_index, args.recency_decay)
                    else:
                        chosen_index = _simulator_choice(
                            simulator,
                            state,
                            action_names,
                            args,
                            overhead,
                            rng,
                            tokens_per_chunk,
                        )

                    chosen_name = action_names[chosen_index]
                    action = _single_action(
                        chosen_name, int(args.steps_per_chunk), tokens_per_chunk
                    )
        else:
            raise ValueError(f"unknown policy: {args.policy}")

        observation = oracle.apply_action(
            action,
            data_seed=int(args.seed) * 1_000_003 + int(chunk_index),
        )

        if args.policy in {"compiler", "staged"}:
            if was_bootstrap_chunk:
                bootstrap_count += 1
            if args.policy == "compiler" or not switch_state.switched:
                transition_pool.append(observation)

        _increment_action(policy_state, chosen_name)
        chunk_index += 1
        last_chunk_probe_mean = visible_probe_mean

        record_point = None
        if chunk_index % int(args.record_every_chunks) == 0:
            record_point = _record_trajectory_point(
                oracle,
                trajectory,
                chunk_index,
                visible_probe_mean,
                overhead,
            )
            last_recorded_chunk = chunk_index
            if visible_probe_mean is None:
                visible_probe_mean = float(record_point["probe_mean"])
            write_output()

        sync_policy_state()
        if chunk_index % int(args.ckpt_every_chunks) == 0:
            _save_checkpoint(
                ckpt_dir,
                oracle.snapshot(),
                trajectory,
                chunk_index,
                dict(policy_state),
                transition_pool,
            )

        visible_text = "-" if visible_probe_mean is None else f"{visible_probe_mean:.4f}"
        tokens_m = float(oracle.tokens_seen) / 1_000_000.0
        wall_minutes = current_overhead()["wall_seconds"] / 60.0
        chunk_seconds = time.perf_counter() - chunk_start_time
        print(
            f"chunk={chunk_index} tokens={tokens_m:.1f}M action={chosen_name} "
            f"probe_mean={visible_text} wall_min={wall_minutes:.2f} "
            f"chunk_sec={chunk_seconds:.1f}",
            flush=True,
        )

    if last_recorded_chunk != chunk_index:
        _record_trajectory_point(
            oracle,
            trajectory,
            chunk_index,
            last_chunk_probe_mean,
            overhead,
        )
        last_recorded_chunk = chunk_index

    output = write_output()
    sync_policy_state()
    _save_checkpoint(
        ckpt_dir, oracle.snapshot(), trajectory, chunk_index, dict(policy_state),
        transition_pool,
    )

    if trajectory:
        final_point = trajectory[-1]
        print(
            "final "
            f"val_bpb={float(final_point['val_bpb']):.6f} "
            f"holdout_ce={float(final_point['holdout_ce']):.6f}",
            flush=True,
        )
        target_val_bpb = float(target_data["target"]["val_bpb"])
        print("tokens-to-threshold:", flush=True)
        for factor, label in zip(THRESHOLD_FACTORS, _threshold_labels(), strict=True):
            threshold = target_val_bpb * factor
            tokens = tokens_to_threshold(trajectory, threshold)
            token_text = "-" if tokens is None else str(tokens)
            print(f"  {label} ({threshold:.6f}): {token_text}", flush=True)

    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        required=True,
        choices=["uniform", "edu_heavy", "worst_probe", "compiler", "staged"],
    )
    parser.add_argument("--tokens-dir", default="data/tokens")
    parser.add_argument("--target-file", default="results/lm_reference_seed0.json")
    parser.add_argument("--token-budget", type=int, default=600_000_000)
    parser.add_argument("--steps-per-chunk", type=int, default=50)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--record-every-chunks", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--ckpt-every-chunks", type=int, default=16)
    parser.add_argument("--resume", action="store_true")

    # Default covers every action once so the first simulator fit has a real
    # anchor per action (24 chunks ~= 39M tokens, paid inside the budget).
    parser.add_argument("--bootstrap-chunks", type=int, default=24)
    parser.add_argument("--refit-every-chunks", type=int, default=8)
    parser.add_argument("--explore-epsilon", type=float, default=0.1)
    parser.add_argument("--sim-rank", type=int, default=6)
    parser.add_argument("--sim-epochs-initial", type=int, default=300)
    parser.add_argument("--sim-epochs-refit", type=int, default=100)

    parser.add_argument("--switch-threshold", type=float, default=0.02)
    parser.add_argument("--switch-min-chunks", type=int, default=24)
    parser.add_argument("--gain-ema-alpha", type=float, default=0.2)
    parser.add_argument("--recency-penalty", type=float, default=0.3)
    parser.add_argument("--recency-decay", type=float, default=0.7)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = f"results/lm_race_{args.policy}_seed{args.seed}.json"
    if args.ckpt_dir is None:
        args.ckpt_dir = f"data/checkpoints/race_{args.policy}_seed{args.seed}"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_race(args)


if __name__ == "__main__":
    main()
