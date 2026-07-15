"""Phase 2 curriculum compiler experiment.

Greedy receding-horizon curriculum compilation with real-execution correction,
raced against baseline policies to a fixed capability target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from teachability_compiler.real import hidden_eval, tasks
from teachability_compiler.real.mlp_predictor import ResidualMLPTransitionPredictor
from teachability_compiler.real.model import DecoderConfig
from teachability_compiler.real.oracle import RealLearnerOracle
from teachability_compiler.real.persistence import probe_suite_hash, save_transitions
from teachability_compiler.state import CurriculumAction

POLICIES = (
    "reference",
    "random",
    "mixed_review",
    "loss_greedy",
    "compiler",
    "staged",
    "oracle_greedy",
)

SIMULATOR_VERSION = "residual-mlp-v1"
ENVIRONMENT_VERSION = "real-v1"

# Highest mid-training commutator pairs from prior measurement.
COMMUTATOR_PANEL_PAIRS = (
    ("mod_arith", "bracket_match"),
    ("add_2digit", "bracket_match"),
    ("add_1digit", "bracket_depth"),
    ("compare_numbers", "bracket_depth"),
    ("add_1digit", "pattern_alternate"),
)


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def _config_hash(args: argparse.Namespace) -> str:
    payload = json.dumps(vars(args), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --------------------------------------------------------------------------- #
# Trajectory recording (the ONLY place the hidden channel is ever touched)
# --------------------------------------------------------------------------- #
def _record_trajectory_point(
    oracle: RealLearnerOracle,
    trajectory: list[dict[str, Any]],
    action_index: int,
    tokens: int,
    chosen_cluster: str | None,
    gain_ema: float | None = None,
    realized_gain_ema: float | None = None,
) -> np.ndarray:
    """Append a trajectory point and return the visible loss vector.

    Hidden losses are RECORDED ONLY here and never returned to callers for
    decision-making.
    """
    visible_losses = np.asarray(oracle.probe_losses(), dtype=np.float64)
    if not np.all(np.isfinite(visible_losses)):
        raise ValueError("Non-finite visible probe loss while recording trajectory")

    hidden_losses = hidden_eval.hidden_losses(oracle)

    trajectory.append(
        {
            "action_index": int(action_index),
            "tokens": int(tokens),
            "visible_mean": float(np.mean(visible_losses)),
            "hidden_mean": float(np.mean(hidden_losses)),
            "visible_losses": visible_losses.tolist(),
            "hidden_losses": hidden_losses.tolist(),
            "chosen_cluster": chosen_cluster,
            "gain_ema": (float(gain_ema) if gain_ema is not None else None),
            "realized_gain_ema": (
                float(realized_gain_ema) if realized_gain_ema is not None else None
            ),
        }
    )
    return visible_losses


def _forgetting_auc(trajectory: list[dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0

    losses = np.asarray(
        [point["visible_losses"] for point in trajectory], dtype=np.float64
    )
    running_min = np.minimum.accumulate(losses, axis=0)
    excess = np.maximum(0.0, losses - running_min)
    return float(excess.sum() / len(trajectory))


def _target_reached(
    visible_losses: np.ndarray,
    target_losses: np.ndarray,
    epsilon: float,
) -> bool:
    delta = visible_losses - target_losses
    if not np.all(np.isfinite(delta)):
        raise ValueError("Non-finite loss in target termination check")
    return float(np.max(delta)) <= epsilon


# --------------------------------------------------------------------------- #
# Argument parsing / validation
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Phase 2 curriculum compiler race.")
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--target-file", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--reference-actions", type=int, default=1200)
    parser.add_argument("--bootstrap-rollouts", type=int, default=12)
    parser.add_argument("--bootstrap-horizon", type=int, default=4)
    parser.add_argument("--refit-every", type=int, default=16)
    parser.add_argument("--explore-epsilon", type=float, default=0.0)
    parser.add_argument("--mlp-epochs-initial", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--save-transitions", type=str, default=None)

    # Staged policy + switch-trigger instrumentation knobs.
    parser.add_argument("--switch-threshold", type=float, default=0.002)
    parser.add_argument("--switch-min-actions", type=int, default=100)
    parser.add_argument("--gain-ema-alpha", type=float, default=0.2)
    parser.add_argument("--recency-penalty", type=float, default=0.3)
    parser.add_argument("--recency-decay", type=float, default=0.7)
    parser.add_argument("--probe-pairs", type=int, default=3)
    parser.add_argument("--probe-every", type=int, default=150)

    # Apparatus knobs (kept as CLI args so the smoke test can run a tiny model).
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.policy != "reference" and args.target_file is None:
        parser.error("--target-file is required for all policies except 'reference'")

    positive_fields = (
        "steps",
        "batch_size",
        "seq_len",
        "d_model",
        "n_layers",
        "refit_every",
        "mlp_epochs_initial",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")

    nonnegative_fields = (
        "max_actions",
        "reference_actions",
        "pretrain_steps",
        "bootstrap_rollouts",
        "bootstrap_horizon",
    )
    for field in nonnegative_fields:
        if getattr(args, field) < 0:
            parser.error(f"--{field.replace('_', '-')} must be non-negative")

    if args.epsilon < 0.0:
        parser.error("--epsilon must be non-negative")

    if not 0.0 <= args.explore_epsilon <= 1.0:
        parser.error("--explore-epsilon must be in [0, 1]")

    if args.switch_threshold < 0.0:
        parser.error("--switch-threshold must be non-negative")

    if not 0.0 < args.gain_ema_alpha <= 1.0:
        parser.error("--gain-ema-alpha must be in (0, 1]")

    if args.recency_penalty < 0.0:
        parser.error("--recency-penalty must be non-negative")

    if not 0.0 <= args.recency_decay < 1.0:
        parser.error("--recency-decay must be in [0, 1)")

    if args.probe_pairs < 0:
        parser.error("--probe-pairs must be non-negative")

    if args.probe_every <= 0:
        parser.error("--probe-every must be positive")


def _make_action(cluster_name: str, steps: int, token_budget: int) -> CurriculumAction:
    return CurriculumAction(
        cluster_ids=(cluster_name,),
        mixture_weights=(1.0,),
        optimizer_steps=steps,
        token_budget=token_budget,
    )


def _load_target(
    target_file: str,
    n_clusters: int,
    current_probe_suite_hash: str,
) -> tuple[np.ndarray, str | None]:
    target_data = json.loads(Path(target_file).read_text())
    target_probe_suite_hash = target_data.get("provenance", {}).get("probe_suite_hash")
    if target_probe_suite_hash != current_probe_suite_hash:
        raise ValueError(
            "probe_suite_hash mismatch (apparatus drift): "
            f"target={target_probe_suite_hash!r} current={current_probe_suite_hash!r}"
        )

    target = np.asarray(target_data["target_probe_losses"], dtype=np.float64)
    if target.shape != (n_clusters,):
        raise ValueError(f"target vector length {target.shape} != ({n_clusters},)")
    if not np.all(np.isfinite(target)):
        raise ValueError("target vector contains non-finite losses")

    target_config_hash = target_data.get("provenance", {}).get("config_hash")
    return target, target_config_hash


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    panel_rng = np.random.default_rng(args.seed + 1_000_003)

    cluster_names = tasks.all_cluster_names()
    n_clusters = len(cluster_names)
    mixed_review_id = cluster_names.index("mixed_review")

    steps = int(args.steps)
    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    tokens_per_action = steps * batch_size * seq_len

    current_probe_suite_hash = probe_suite_hash()
    target: np.ndarray | None = None
    target_file_config_hash: str | None = None
    if args.policy != "reference":
        target, target_file_config_hash = _load_target(
            args.target_file,
            n_clusters,
            current_probe_suite_hash,
        )

    config_hash = _config_hash(args)
    git_commit = _git_commit()

    config = DecoderConfig(
        vocab_size=tasks.VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    oracle = RealLearnerOracle(
        config,
        cluster_names,
        device=args.device,
        base_seed=args.seed,
        steps_per_action=steps,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    if args.pretrain_steps > 0:
        oracle.pretrain(args.pretrain_steps, rng_seed=args.seed)
        target_checkpoint = f"pretrain-{args.pretrain_steps}"
    else:
        target_checkpoint = "scratch"

    def next_seed() -> int:
        return int(rng.integers(0, 2**31 - 1))

    def next_panel_seed() -> int:
        return int(panel_rng.integers(0, 2**31 - 1))

    def action_for(cluster_index: int) -> CurriculumAction:
        return _make_action(cluster_names[cluster_index], steps, tokens_per_action)

    trajectory: list[dict[str, Any]] = []
    action_counts = {name: 0 for name in cluster_names}
    race_observations: list[Any] = []

    race_tokens = 0
    overhead_tokens = 0
    race_optimizer_steps = 0
    exploration_optimizer_steps = 0
    bootstrap_optimizer_steps = 0
    refit_wall_seconds = 0.0

    predictor: ResidualMLPTransitionPredictor | None = None
    transition_pool: list[Any] = []

    # Switch-trigger instrumentation state (compiler + staged).
    alpha = float(args.gain_ema_alpha)
    gain_ema: float | None = None
    realized_gain_ema: float | None = None
    prev_visible_mean: float | None = None
    switched = False
    switch_action_index: int | None = None
    recency = np.zeros(n_clusters, dtype=np.float64)
    commutator_panel: list[dict[str, Any]] = []
    panel_pairs = COMMUTATOR_PANEL_PAIRS[: args.probe_pairs]

    def _coverage_select(visible_losses: np.ndarray) -> int:
        # Damped loss_greedy: decay recency first, then score visible losses.
        recency[:] *= args.recency_decay
        scores = visible_losses - args.recency_penalty * recency
        return sorted(
            range(n_clusters),
            key=lambda index: (-scores[index], cluster_names[index]),
        )[0]

    wall_start = time.perf_counter()

    # Compiler/staged: predictor + transition pool + bootstrap.
    if args.policy in {"compiler", "staged"}:
        predictor = ResidualMLPTransitionPredictor(
            cluster_names,
            device=args.device,
            seed=args.seed,
        )
        for _ in range(args.bootstrap_rollouts):
            snapshot = oracle.snapshot()
            try:
                for _ in range(args.bootstrap_horizon):
                    cluster_id = int(rng.integers(0, n_clusters))
                    observation = oracle.apply_action(
                        action_for(cluster_id), next_seed()
                    )
                    transition_pool.append(observation)
                    bootstrap_optimizer_steps += steps
                    overhead_tokens += tokens_per_action
            finally:
                oracle.restore(snapshot)

        if not transition_pool:
            raise ValueError(
                f"{args.policy} policy requires at least one bootstrap transition"
            )

        refit_start = time.perf_counter()
        predictor.fit(
            transition_pool,
            epochs=args.mlp_epochs_initial,
            ranking_weight=0.1,
        )
        refit_wall_seconds += time.perf_counter() - refit_start

    # Record initial (baseline) state.
    _record_trajectory_point(
        oracle,
        trajectory,
        action_index=0,
        tokens=0,
        chosen_cluster=None,
    )

    max_actions = (
        args.reference_actions if args.policy == "reference" else args.max_actions
    )
    reached = args.policy == "reference"
    executed = 0
    last_cluster: str | None = None

    while executed < max_actions:
        visible_for_policy: np.ndarray | None = None

        # Termination check for non-reference policies.
        if args.policy != "reference":
            if target is None:
                raise RuntimeError("target must be loaded for non-reference policies")
            visible_for_policy = np.asarray(oracle.probe_losses(), dtype=np.float64)

            # Realized-gain EMA (learning-velocity signal) for compiler + staged.
            if args.policy in {"compiler", "staged"}:
                current_visible_mean = float(np.mean(visible_for_policy))
                if prev_visible_mean is not None:
                    realized = prev_visible_mean - current_visible_mean
                    if realized_gain_ema is None:
                        realized_gain_ema = realized
                    else:
                        realized_gain_ema = (
                            alpha * realized + (1.0 - alpha) * realized_gain_ema
                        )
                prev_visible_mean = current_visible_mean

            if _target_reached(visible_for_policy, target, args.epsilon):
                reached = True
                break

        # --- action selection ------------------------------------------------
        if args.policy in {"reference", "random"}:
            chosen_id = int(rng.integers(0, n_clusters))
        elif args.policy == "mixed_review":
            chosen_id = mixed_review_id
        elif args.policy == "loss_greedy":
            if visible_for_policy is None:
                visible_for_policy = np.asarray(
                    oracle.probe_losses(), dtype=np.float64
                )
            chosen_id = sorted(
                range(n_clusters),
                key=lambda index: (-visible_for_policy[index], cluster_names[index]),
            )[0]
        elif args.policy in {"compiler", "staged"}:
            action_idx = executed + 1
            coverage_phase = args.policy == "staged" and switched

            if (
                args.policy == "staged"
                and not switched
                and gain_ema is not None
                and action_idx >= args.switch_min_actions
                and gain_ema < args.switch_threshold
            ):
                switched = True
                switch_action_index = action_idx
                print(
                    f"[staged] switch at action {action_idx} "
                    f"(gain_ema={gain_ema})"
                )
                coverage_phase = True

            if coverage_phase:
                if visible_for_policy is None:
                    raise RuntimeError("visible losses required for staged coverage")
                chosen_id = _coverage_select(visible_for_policy)
            elif rng.random() < args.explore_epsilon:
                # Exploration keeps the online transition pool diverse; without
                # it the refits only ever see the incumbent argmax action's data.
                chosen_id = int(rng.integers(0, n_clusters))
            else:
                if predictor is None:
                    raise RuntimeError("compiler predictor was not initialized")
                state = oracle.encode_state()
                current_state_mean = float(
                    np.mean(np.asarray(state.probe_losses, dtype=np.float64))
                )
                best_score = -float("inf")
                chosen_id = 0
                for candidate_id in range(n_clusters):
                    prediction = predictor.predict(state, action_for(candidate_id))
                    predicted_losses = np.asarray(
                        prediction.next_state_mean.probe_losses,
                        dtype=np.float64,
                    )
                    if not np.all(np.isfinite(predicted_losses)):
                        raise ValueError(
                            "Non-finite simulator prediction for "
                            f"{cluster_names[candidate_id]!r}"
                        )
                    score = -float(np.mean(predicted_losses))
                    if score > best_score:
                        best_score = score
                        chosen_id = candidate_id

                predicted_best_gain = current_state_mean - (-best_score)
                if gain_ema is None:
                    gain_ema = predicted_best_gain
                else:
                    gain_ema = (
                        alpha * predicted_best_gain + (1.0 - alpha) * gain_ema
                    )

                # Staged pragmatic switch to coverage.
                if (
                    args.policy == "staged"
                    and action_idx >= args.switch_min_actions
                    and gain_ema < args.switch_threshold
                ):
                    switched = True
                    switch_action_index = action_idx
                    print(
                        f"[staged] switch at action {action_idx} "
                        f"(gain_ema={gain_ema})"
                    )
                    if visible_for_policy is None:
                        raise RuntimeError("visible losses required for staged coverage")
                    chosen_id = _coverage_select(visible_for_policy)
        elif args.policy == "oracle_greedy":
            snapshot = oracle.snapshot()
            best_value = -float("inf")
            chosen_id = 0
            try:
                for candidate_id in range(n_clusters):
                    oracle.restore(snapshot)
                    oracle.apply_action(action_for(candidate_id), next_seed())
                    value = float(oracle.value(oracle.encode_state()))
                    if not np.isfinite(value):
                        raise ValueError(
                            "Non-finite oracle-greedy value for "
                            f"{cluster_names[candidate_id]!r}"
                        )
                    exploration_optimizer_steps += steps
                    overhead_tokens += tokens_per_action
                    if value > best_value:
                        best_value = value
                        chosen_id = candidate_id
            finally:
                oracle.restore(snapshot)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"Unknown policy {args.policy!r}")

        # --- execution on the real oracle -----------------------------------
        observation = oracle.apply_action(action_for(chosen_id), next_seed())
        race_observations.append(observation)

        opening_phase = args.policy == "compiler" or (
            args.policy == "staged" and not switched
        )
        if opening_phase:
            transition_pool.append(observation)

        executed += 1
        race_optimizer_steps += steps
        race_tokens += tokens_per_action
        last_cluster = cluster_names[chosen_id]
        action_counts[last_cluster] += 1

        # Coverage recency bookkeeping (staged only, after the switch).
        if args.policy == "staged" and switched:
            recency[chosen_id] += 1.0

        # Simulator refit from scratch on ALL accumulated transitions (opening).
        if opening_phase and executed % args.refit_every == 0:
            if predictor is None:
                raise RuntimeError("compiler predictor was not initialized")
            refit_start = time.perf_counter()
            predictor.fit(transition_pool, epochs=80, ranking_weight=0.1)
            refit_wall_seconds += time.perf_counter() - refit_start

        # Record every 8 actions.
        if executed % 8 == 0:
            log_gain = None if (args.policy == "staged" and switched) else gain_ema
            _record_trajectory_point(
                oracle,
                trajectory,
                action_index=executed,
                tokens=race_tokens,
                chosen_cluster=last_cluster,
                gain_ema=log_gain,
                realized_gain_ema=realized_gain_ema,
            )

        # Compact progress line every 40 actions (reuses last recorded point).
        if executed % 40 == 0:
            point = trajectory[-1]
            print(
                f"[{args.policy}] action={executed} "
                f"visible_mean={point['visible_mean']:.4f} "
                f"hidden_mean={point['hidden_mean']:.4f} "
                f"tokens={race_tokens}"
            )

        # In-race commutator panel (compiler + staged), overhead-only.
        if (
            args.policy in {"compiler", "staged"}
            and args.probe_pairs > 0
            and executed % args.probe_every == 0
        ):
            panel_snapshot = oracle.snapshot()
            try:
                for pair_a, pair_b in panel_pairs:
                    a_id = cluster_names.index(pair_a)
                    b_id = cluster_names.index(pair_b)

                    oracle.restore(panel_snapshot)
                    oracle.apply_action(action_for(a_id), next_panel_seed())
                    oracle.apply_action(action_for(b_id), next_panel_seed())
                    ab_losses = np.asarray(oracle.probe_losses(), dtype=np.float64)

                    oracle.restore(panel_snapshot)
                    oracle.apply_action(action_for(b_id), next_panel_seed())
                    oracle.apply_action(action_for(a_id), next_panel_seed())
                    ba_losses = np.asarray(oracle.probe_losses(), dtype=np.float64)

                    if not np.all(np.isfinite(ab_losses)) or not np.all(
                        np.isfinite(ba_losses)
                    ):
                        raise ValueError(
                            "Non-finite visible probe loss in commutator panel"
                        )

                    commutator = float(np.linalg.norm(ab_losses - ba_losses))
                    exploration_optimizer_steps += 4 * steps
                    overhead_tokens += 4 * tokens_per_action
                    commutator_panel.append(
                        {
                            "action_index": int(executed),
                            "pair": [pair_a, pair_b],
                            "commutator": commutator,
                        }
                    )
            finally:
                oracle.restore(panel_snapshot)

    # Final trajectory point (at termination), avoiding duplicates.
    if not trajectory or trajectory[-1]["action_index"] != executed:
        log_gain = None if (args.policy == "staged" and switched) else gain_ema
        _record_trajectory_point(
            oracle,
            trajectory,
            action_index=executed,
            tokens=race_tokens,
            chosen_cluster=last_cluster,
            gain_ema=log_gain,
            realized_gain_ema=realized_gain_ema,
        )

    final_point = trajectory[-1]
    final_visible_losses = final_point["visible_losses"]
    final_hidden_losses = final_point["hidden_losses"]
    final_visible_mean = float(final_point["visible_mean"])
    final_hidden_mean = float(final_point["hidden_mean"])

    if args.policy != "reference":
        if target is None:
            raise RuntimeError("target must be loaded for non-reference policies")
        reached = _target_reached(
            np.asarray(final_visible_losses, dtype=np.float64),
            target,
            args.epsilon,
        )
        target_probe_losses = target.tolist()
    else:
        target_probe_losses = final_visible_losses

    total_tokens = race_tokens + overhead_tokens
    total_optimizer_steps = (
        race_optimizer_steps
        + exploration_optimizer_steps
        + bootstrap_optimizer_steps
    )
    wall_seconds = time.perf_counter() - wall_start

    result: dict[str, Any] = {
        "policy": args.policy,
        "reached": bool(reached),
        "actions_executed": int(executed),
        "race_tokens": int(race_tokens),
        "overhead_tokens": int(overhead_tokens),
        "total_tokens": int(total_tokens),
        "final_visible_mean": final_visible_mean,
        "final_hidden_mean": final_hidden_mean,
        "final_visible_losses": final_visible_losses,
        "final_hidden_losses": final_hidden_losses,
        "target_probe_losses": target_probe_losses,
        "epsilon": float(args.epsilon),
        "switch_action_index": switch_action_index,
        "switch_threshold": float(args.switch_threshold),
        "commutator_panel": commutator_panel,
        "forgetting_auc": _forgetting_auc(trajectory),
        "trajectory": trajectory,
        "action_counts": action_counts,
        "provenance": {
            "seed": int(args.seed),
            "config_hash": config_hash,
            "git_commit": git_commit,
            "target_checkpoint": target_checkpoint,
            "simulator_version": (
                SIMULATOR_VERSION
                if args.policy in {"compiler", "staged"}
                else "none"
            ),
            "environment_version": ENVIRONMENT_VERSION,
            "probe_suite_hash": current_probe_suite_hash,
            "target_file": args.target_file,
            "target_file_config_hash": target_file_config_hash,
        },
        "overhead": {
            "total_optimizer_steps": int(total_optimizer_steps),
            "race_optimizer_steps": int(race_optimizer_steps),
            "exploration_optimizer_steps": int(exploration_optimizer_steps),
            "bootstrap_optimizer_steps": int(bootstrap_optimizer_steps),
            "refit_wall_seconds": float(refit_wall_seconds),
            "wall_seconds": float(wall_seconds),
        },
    }

    out_path = Path(
        args.out or f"results/compile_{args.policy}_seed{args.seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    if args.save_transitions:
        observations_to_save = (
            transition_pool if args.policy == "compiler" else race_observations
        )
        save_transitions(
            args.save_transitions,
            observations_to_save,
            {
                "policy": args.policy,
                "seed": int(args.seed),
                "config_hash": config_hash,
                "probe_suite_hash": current_probe_suite_hash,
            },
        )

    print(
        f"[{args.policy}] DONE reached={reached} actions={executed} "
        f"visible_mean={final_visible_mean:.4f} hidden_mean={final_hidden_mean:.4f} "
        f"race_tokens={race_tokens} overhead_tokens={overhead_tokens} "
        f"forgetting_auc={result['forgetting_auc']:.4f} -> {out_path}"
    )

    return result


if __name__ == "__main__":
    main()
