"""Nanochat-backed learner oracle for curriculum-action transitions.

The hidden_val_bpb() and hidden_holdout_ce() methods are the hidden evaluation channel.
Policies, simulators, curriculum search, model selection, and training loops must never call
those methods. They are only for final hidden validation/reporting.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import nanochat.flash_attention as fa
import numpy as np
import torch
from nanochat.gpt import GPT, GPTConfig

from teachability_compiler.state import CurriculumAction, LearningState, TransitionObservation

fa.USE_FA3 = False

_VOCAB_SIZE = 32768
_PROBE_EMA_ALPHA = 0.25
_RECENCY_DECAY = 0.9
_LAYER_RE = re.compile(r"(?:^|\.)(?:h|blocks|block|layers)\.(\d+)(?:\.|$)")

_MALLOC_TRIM_CHECKED = False
_MALLOC_TRIM: Any | None = None
_MALLOC_TRIM_LIBC: Any | None = None


def _return_freed_memory_to_os() -> None:
    """Return CPU heap memory freed this chunk back to the operating system.

    Leak fix. Every chunk ``apply_action`` builds a full fp32 CPU parameter
    snapshot (``_cpu_param_snapshot``, ~1.5 GB), and on checkpoint chunks
    ``snapshot()`` deep-copies the model + optimizer state (~4.7 GB). Python
    frees these promptly by reference counting, but glibc keeps the pages in its
    malloc arenas instead of returning them to the kernel, so resident memory
    ratchets up tens-to-hundreds of MB per chunk and is never released,
    eventually exhausting RAM + swap. A cyclic GC pass (to break any grad_fn
    reference cycles) followed by malloc_trim(0) hands the freed pages back so
    RSS tracks the live working set. No-op where glibc/malloc_trim is
    unavailable (e.g. macOS/Windows).
    """
    global _MALLOC_TRIM_CHECKED, _MALLOC_TRIM, _MALLOC_TRIM_LIBC

    gc.collect()
    if not _MALLOC_TRIM_CHECKED:
        _MALLOC_TRIM_CHECKED = True
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            trim = libc.malloc_trim
        except (OSError, AttributeError):
            _MALLOC_TRIM_LIBC = None
            _MALLOC_TRIM = None
        else:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            _MALLOC_TRIM_LIBC = libc
            _MALLOC_TRIM = trim

    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required token manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return manifest


def _action_names_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    names = manifest.get("action_names")
    if isinstance(names, list):
        out = [str(x) for x in names]
    else:
        actions = manifest.get("actions")
        if isinstance(actions, dict):
            out = [str(x) for x in actions.keys()]
        elif isinstance(actions, list):
            out = [
                str(item["name"] if isinstance(item, dict) and "name" in item else item)
                for item in actions
            ]
        else:
            out = []
    out = [x for x in out if x not in {"holdout", "val"}]
    if len(out) != 24:
        raise ValueError(f"Expected exactly 24 action names in tokens manifest, found {len(out)}")
    if len(set(out)) != len(out):
        raise ValueError("Action names in tokens manifest are not unique")
    return out


def _stable_seed(base_seed: int, label: str) -> int:
    payload = f"{int(base_seed)}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _device_autocast(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _loss_from_model_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 0:
            raise ValueError("Expected scalar loss tensor from nanochat GPT")
        return output
    if isinstance(output, (tuple, list)):
        for item in reversed(output):
            if torch.is_tensor(item) and item.ndim == 0:
                return item
    if hasattr(output, "loss") and torch.is_tensor(output.loss):
        if output.loss.ndim != 0:
            raise ValueError("Expected scalar loss tensor in model output.loss")
        return output.loss
    raise ValueError(f"Could not extract scalar loss from model output type {type(output)!r}")


def _to_cpu_deep(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {copy.deepcopy(k): _to_cpu_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_deep(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_deep(v) for v in obj)
    return copy.deepcopy(obj)


def _to_device_deep(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device_deep(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device_deep(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device_deep(v, device) for v in obj)
    return obj


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for key, value in list(state.items()):
                state[key] = _to_device_deep(value, device)


def _construct_transition_observation(values: dict[str, Any]) -> TransitionObservation:
    aliases = {
        "state_before": values["state_before"],
        "before_state": values["state_before"],
        "previous_state": values["state_before"],
        "state_after": values["state_after"],
        "after_state": values["state_after"],
        "next_state": values["state_after"],
        "action": values["action"],
        "curriculum_action": values["action"],
        "probe_delta": values["probe_delta"],
        "probe_loss_delta": values["probe_delta"],
        "parameter_delta_sketch": values["parameter_delta_sketch"],
        "param_delta_sketch": values["parameter_delta_sketch"],
        "activation_delta_sketch": values["activation_delta_sketch"],
        "compute_cost": values["compute_cost"],
        "expected_compute": values["compute_cost"],
        "seed_metadata": values["seed_metadata"],
        "metadata": values["seed_metadata"],
    }

    if dataclasses.is_dataclass(TransitionObservation):
        kwargs: dict[str, Any] = {}
        missing: list[str] = []
        for field in dataclasses.fields(TransitionObservation):
            if not field.init:
                continue
            if field.name in aliases:
                kwargs[field.name] = aliases[field.name]
            elif (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                missing.append(field.name)
        if missing:
            raise TypeError(f"Cannot construct TransitionObservation; unknown fields: {missing}")
        return TransitionObservation(**kwargs)

    return TransitionObservation(**values)


class NanochatLearnerOracle:
    def __init__(
        self,
        tokens_dir: str | Path,
        depth: int = 12,
        device: str = "cuda:0",
        seq_len: int = 1024,
        device_batch: int = 4,
        grad_accum: int = 8,
        base_seed: int = 0,
        probe_batches: int = 2,
        matrix_lr: float = 0.02,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        scalar_lr: float = 0.5,
        weight_decay: float = 0.0,
    ) -> None:
        fa.USE_FA3 = False

        self.tokens_dir = Path(tokens_dir)
        self.manifest = _load_manifest(self.tokens_dir / "tokens_manifest.json")
        self._action_names = _action_names_from_manifest(self.manifest)

        manifest_seq_len = self.manifest.get("seq_len")
        if manifest_seq_len is not None and int(manifest_seq_len) != int(seq_len):
            raise ValueError(
                f"seq_len mismatch: manifest={manifest_seq_len}, requested={seq_len}"
            )

        self.depth = int(depth)
        self.seq_len = int(seq_len)
        self.device_batch = int(device_batch)
        self.grad_accum = int(grad_accum)
        self.base_seed = int(base_seed)
        self.probe_batches = int(probe_batches)
        self.matrix_lr = float(matrix_lr)
        self.device = torch.device(device)

        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.seq_len <= 0 or self.device_batch <= 0 or self.grad_accum <= 0:
            raise ValueError("seq_len, device_batch, and grad_accum must be positive")
        if self.probe_batches <= 0:
            raise ValueError("probe_batches must be positive")

        self._action_to_idx = {name: i for i, name in enumerate(self._action_names)}
        self._train_tokens = self._open_action_memmaps("train")
        self._probe_tokens = self._open_action_memmaps("probe")
        self._val_tokens = self._open_memmap(self.tokens_dir / "val.bin")
        self._holdout_tokens = self._open_memmap(self.tokens_dir / "holdout.bin")

        token_bytes_path = Path(os.path.expanduser("~/.cache/nanochat/tokenizer/token_bytes.pt"))
        if not token_bytes_path.exists():
            raise FileNotFoundError(f"Required token_bytes tensor is missing: {token_bytes_path}")
        token_bytes = torch.load(token_bytes_path, map_location="cpu")
        if int(token_bytes.numel()) != _VOCAB_SIZE:
            raise ValueError(f"token_bytes has {token_bytes.numel()} entries, expected {_VOCAB_SIZE}")
        self._token_bytes_np = token_bytes.to(torch.float64).cpu().numpy()

        torch.manual_seed(self.base_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.base_seed)

        self.model_dim = ((self.depth * 64 + 127) // 128) * 128
        n_head = self.model_dim // 128
        config = GPTConfig(
            sequence_len=self.seq_len,
            vocab_size=_VOCAB_SIZE,
            n_layer=self.depth,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=self.model_dim,
        )
        self.model = GPT(config).to(self.device)
        self.optimizer = self.model.setup_optimizer(
            unembedding_lr=float(unembedding_lr),
            embedding_lr=float(embedding_lr),
            matrix_lr=float(matrix_lr),
            weight_decay=float(weight_decay),
            scalar_lr=float(scalar_lr),
        )

        self.tokens_seen = 0
        self.step = 0
        self._exposure = np.zeros(len(self._action_names), dtype=np.float64)
        self._recency = np.zeros(len(self._action_names), dtype=np.float64)
        self._probe_delta_ema = np.zeros(len(self._action_names), dtype=np.float64)
        self._last_update_sketch = np.zeros(4, dtype=np.float64)

        self._probe_eval_batches = {
            name: self._make_fixed_batches(
                self._probe_tokens[name],
                _stable_seed(self.base_seed, f"probe:{name}"),
                self.probe_batches,
            )
            for name in self._action_names
        }
        hidden_batches = max(8, self.probe_batches)
        self._val_eval_batches = self._make_fixed_batches(
            self._val_tokens,
            _stable_seed(self.base_seed, "hidden:val"),
            hidden_batches,
        )
        self._holdout_eval_batches = self._make_fixed_batches(
            self._holdout_tokens,
            _stable_seed(self.base_seed, "hidden:holdout"),
            hidden_batches,
        )

        self.model.train()

    @property
    def action_names(self) -> list[str]:
        return list(self._action_names)

    @property
    def probe_delta_ema(self) -> np.ndarray:
        return self._probe_delta_ema.copy()

    def _open_action_memmaps(self, prefix: str) -> dict[str, np.memmap]:
        return {
            name: self._open_memmap(self.tokens_dir / f"{prefix}_{name}.bin")
            for name in self._action_names
        }

    def _open_memmap(self, path: Path) -> np.memmap:
        if not path.exists():
            raise FileNotFoundError(f"Required token file is missing: {path}")
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        if int(mm.shape[0]) < self.seq_len + 1:
            raise ValueError(f"Token file {path} has fewer than seq_len+1 tokens")
        if int(mm.max()) >= _VOCAB_SIZE:
            raise ValueError(
                f"Token file {path} contains token id outside vocab size {_VOCAB_SIZE}"
            )
        return mm

    def _pin_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda" and torch.cuda.is_available():
            return tensor.pin_memory()
        return tensor

    def _make_fixed_batches(
        self,
        tokens: np.memmap,
        seed: int,
        batches: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(seed)
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_start = int(tokens.shape[0]) - self.seq_len - 1
        if max_start < 0:
            raise ValueError("Token memmap is too short for fixed evaluation windows")
        for _ in range(int(batches)):
            starts = rng.integers(0, max_start + 1, size=self.device_batch)
            windows = np.stack(
                [
                    np.asarray(
                        tokens[int(start) : int(start) + self.seq_len + 1],
                        dtype=np.int64,
                    )
                    for start in starts
                ]
            )
            idx = torch.from_numpy(windows[:, :-1].copy()).long()
            targets = torch.from_numpy(windows[:, 1:].copy()).long()
            out.append((self._pin_if_needed(idx), self._pin_if_needed(targets)))
        return out

    def _to_device_batch(
        self,
        idx_cpu: torch.Tensor,
        targets_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        return (
            idx_cpu.to(self.device, non_blocking=non_blocking),
            targets_cpu.to(self.device, non_blocking=non_blocking),
        )

    def _forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with _device_autocast(self.device):
            return _loss_from_model_output(self.model(idx, targets))

    def _validate_action_name(self, action: str) -> int:
        if action not in self._action_to_idx:
            raise KeyError(f"Unknown action {action!r}")
        return self._action_to_idx[action]

    def sample_batch(
        self,
        action: str,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_action_name(action)
        tokens = self._train_tokens[action]
        max_start = int(tokens.shape[0]) - self.seq_len - 1
        if max_start < 0:
            raise ValueError(f"Training token file for {action!r} is too short")
        starts = rng.integers(0, max_start + 1, size=self.device_batch)
        windows = np.stack(
            [
                np.asarray(tokens[int(start) : int(start) + self.seq_len + 1], dtype=np.int64)
                for start in starts
            ]
        )
        idx_cpu = torch.from_numpy(windows[:, :-1].copy()).long()
        targets_cpu = torch.from_numpy(windows[:, 1:].copy()).long()
        return self._to_device_batch(idx_cpu, targets_cpu)

    def _curriculum_components(
        self,
        action: CurriculumAction,
    ) -> tuple[list[str], np.ndarray, float]:
        names = [str(x) for x in action.cluster_ids]
        if not names:
            raise ValueError("CurriculumAction.cluster_ids must not be empty")
        for name in names:
            self._validate_action_name(name)

        weights = np.asarray(action.mixture_weights, dtype=np.float64)
        if weights.shape != (len(names),):
            raise ValueError("CurriculumAction mixture_weights must match cluster_ids")
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("CurriculumAction mixture_weights must be nonnegative and nonzero")
        weights = weights / float(weights.sum())

        normalized_index = 0.0
        denom = max(1, len(self._action_names) - 1)
        for name, weight in zip(names, weights, strict=True):
            normalized_index += float(weight) * (self._action_to_idx[name] / denom)
        return names, weights, normalized_index

    def _cpu_param_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().float().clone()
            for name, param in self.model.named_parameters()
        }

    def _param_group_index(self, name: str) -> int:
        lower = name.lower()
        if (
            any(key in lower for key in ("wte", "embedding", "embed"))
            and "unembedding" not in lower
            and "lm_head" not in lower
        ):
            return 0

        match = _LAYER_RE.search(lower)
        if match is not None:
            layer_idx = int(match.group(1))
            bucket = min(5, (layer_idx * 6) // max(1, self.depth))
            return 1 + bucket

        return 7

    def _param_delta_sketch(self, before: Mapping[str, torch.Tensor]) -> np.ndarray:
        group_sq = np.zeros(8, dtype=np.float64)
        for name, param in self.model.named_parameters():
            if name not in before:
                raise KeyError(f"Parameter {name!r} missing from pre-action snapshot")
            after = param.detach().cpu().float()
            diff = after - before[name]
            group_sq[self._param_group_index(name)] += float(torch.sum(diff * diff).item())
        return np.sqrt(group_sq)

    def _grad_norm(self) -> float:
        total = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total += float(torch.sum(grad * grad).item())
        return math.sqrt(total)

    def apply_action(self, action: CurriculumAction, data_seed: int) -> TransitionObservation:
        names, weights, normalized_action_index = self._curriculum_components(action)
        optimizer_steps = int(action.optimizer_steps)
        if optimizer_steps < 0:
            raise ValueError("CurriculumAction.optimizer_steps must be nonnegative")

        rng = np.random.default_rng(int(data_seed))
        state_before = self.encode_state()
        probe_before = state_before.probe_losses.copy()
        activation_before = state_before.activation_sketch.copy()
        params_before = self._cpu_param_snapshot()

        tokens_per_step = self.device_batch * self.grad_accum * self.seq_len
        last_grad_norm = 0.0
        self.model.train()

        for _ in range(optimizer_steps):
            self.optimizer.zero_grad(set_to_none=True)
            for micro_idx in range(self.grad_accum):
                sampled_name = str(rng.choice(names, p=weights))
                idx, targets = self.sample_batch(sampled_name, rng)
                loss = self._forward_loss(idx, targets) / float(self.grad_accum)
                loss.backward()
                if micro_idx == self.grad_accum - 1:
                    last_grad_norm = self._grad_norm()
                # Leak guard: free each micro-batch's loss (and its now-consumed
                # autograd graph) plus its input tensors before the next iteration
                # allocates new ones, so no graph is pinned across micro-batches.
                del loss, idx, targets
            self.optimizer.step()

            self.step += 1
            self.tokens_seen += tokens_per_step
            self._exposure += weights
            self._recency *= _RECENCY_DECAY
            for name, weight in zip(names, weights, strict=True):
                self._recency[self._action_to_idx[name]] += float(weight)

        param_delta = self._param_delta_sketch(params_before)
        param_delta_norm = float(np.linalg.norm(param_delta))
        # Leak fix: release the ~1.5 GB fp32 CPU parameter snapshot the moment the
        # delta sketch is computed, so it is not pinned across the post-action
        # probes and observation construction below.
        del params_before

        probe_after = self.probe_losses()
        probe_delta = probe_after - probe_before

        self._probe_delta_ema = (
            (1.0 - _PROBE_EMA_ALPHA) * self._probe_delta_ema + _PROBE_EMA_ALPHA * probe_delta
        )
        self._last_update_sketch = np.asarray(
            [last_grad_norm, param_delta_norm, normalized_action_index, 1.0],
            dtype=np.float64,
        )

        state_after = self._state_from_probe_losses(probe_after)
        activation_delta = state_after.activation_sketch - activation_before
        compute_cost = int(optimizer_steps * tokens_per_step)

        observation = _construct_transition_observation(
            {
                "state_before": state_before,
                "action": action,
                "state_after": state_after,
                "probe_delta": probe_delta.astype(np.float64),
                "parameter_delta_sketch": param_delta.astype(np.float64),
                "activation_delta_sketch": activation_delta.astype(np.float64),
                "compute_cost": compute_cost,
                "seed_metadata": {"seed": int(data_seed), "step": int(state_before.step)},
            }
        )
        # Leak fix: the per-chunk fp32 snapshot above (and the periodic ~4.7 GB
        # checkpoint copy created by snapshot()) are freed by reference counting,
        # but glibc keeps those arenas resident. Hand the freed pages back to the
        # OS here -- the path every policy runs each chunk -- so RSS tracks the
        # live working set instead of the allocator high-water mark.
        _return_freed_memory_to_os()
        return observation

    def probe_losses(self) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        losses = np.zeros(len(self._action_names), dtype=np.float64)
        with torch.no_grad():
            for action_idx, name in enumerate(self._action_names):
                action_losses: list[float] = []
                for idx_cpu, targets_cpu in self._probe_eval_batches[name]:
                    idx, targets = self._to_device_batch(idx_cpu, targets_cpu)
                    loss = self._forward_loss(idx, targets)
                    action_losses.append(float(loss.detach().cpu().item()))
                losses[action_idx] = float(np.mean(action_losses))
        if was_training:
            self.model.train()
        return losses

    def _optimizer_sq_summary(self) -> float:
        values: list[float] = []
        for state in self.optimizer.state.values():
            if not isinstance(state, dict):
                continue
            exp_avg_sq = state.get("exp_avg_sq")
            if torch.is_tensor(exp_avg_sq):
                values.append(
                    float(torch.sqrt(exp_avg_sq.detach().float()).mean().cpu().item())
                )
        if not values:
            return 0.0
        return float(np.mean(values))

    def _history_embedding(self) -> np.ndarray:
        ema_norm = float(np.linalg.norm(self._probe_delta_ema))
        recency_total = float(self._recency.sum())
        if recency_total > 0.0:
            probs = self._recency / recency_total
            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            entropy /= math.log(len(self._action_names))
        else:
            entropy = 0.0
        return np.asarray(
            [
                ema_norm,
                entropy,
                float(np.mean(self._recency)),
                float(self.tokens_seen) / 1e9,
            ],
            dtype=np.float64,
        )

    def _state_from_probe_losses(self, probe_losses: np.ndarray) -> LearningState:
        probe_losses = np.asarray(probe_losses, dtype=np.float64)
        if probe_losses.shape != (len(self._action_names),):
            raise ValueError(f"Expected probe_losses shape {(len(self._action_names),)}")

        activation = np.asarray(
            [
                float(np.mean(probe_losses)),
                float(np.max(probe_losses)),
                float(np.min(probe_losses)),
                float(np.std(probe_losses)),
            ],
            dtype=np.float64,
        )
        return LearningState(
            probe_losses=probe_losses.copy(),
            update_sketch=self._last_update_sketch.copy(),
            optimizer_sketch=np.asarray(
                [self._optimizer_sq_summary(), self.matrix_lr],
                dtype=np.float64,
            ),
            activation_sketch=activation,
            exposure_histogram=self._exposure.copy(),
            history_embedding=self._history_embedding(),
            architecture_embedding=np.asarray(
                [self.depth / 32.0, self.model_dim / 2048.0],
                dtype=np.float64,
            ),
            step=int(self.step),
            tokens_seen=int(self.tokens_seen),
        )

    def encode_state(self) -> LearningState:
        return self._state_from_probe_losses(self.probe_losses())

    def snapshot(self) -> dict[str, Any]:
        return {
            "model_state": _to_cpu_deep(self.model.state_dict()),
            "optimizer_state": _to_cpu_deep(self.optimizer.state_dict()),
            "tokens_seen": int(self.tokens_seen),
            "step": int(self.step),
            "exposure": self._exposure.copy(),
            "recency": self._recency.copy(),
            "probe_delta_ema": self._probe_delta_ema.copy(),
            "last_update_sketch": self._last_update_sketch.copy(),
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        model_state = _to_device_deep(snap["model_state"], self.device)
        self.model.load_state_dict(model_state, strict=True)

        self.optimizer.load_state_dict(copy.deepcopy(snap["optimizer_state"]))
        _move_optimizer_state_to_device(self.optimizer, self.device)

        self.tokens_seen = int(snap["tokens_seen"])
        self.step = int(snap["step"])
        self._exposure = np.asarray(snap["exposure"], dtype=np.float64).copy()
        self._recency = np.asarray(snap["recency"], dtype=np.float64).copy()
        self._probe_delta_ema = np.asarray(snap["probe_delta_ema"], dtype=np.float64).copy()
        self._last_update_sketch = np.asarray(
            snap["last_update_sketch"], dtype=np.float64
        ).copy()
        self.model.train()

    def _hidden_mean_ce(
        self,
        batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        max_batches: int,
    ) -> float:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for idx_cpu, targets_cpu in batches[:max_batches]:
                idx, targets = self._to_device_batch(idx_cpu, targets_cpu)
                loss = self._forward_loss(idx, targets)
                losses.append(float(loss.detach().cpu().item()))
        if was_training:
            self.model.train()
        return float(np.mean(losses))

    def hidden_val_bpb(self, max_batches: int = 8) -> float:
        """HIDDEN CHANNEL: policies and simulators must never call this method."""
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        was_training = self.model.training
        self.model.eval()
        total_ce_nats = 0.0
        total_bytes = 0.0

        with torch.no_grad():
            for idx_cpu, targets_cpu in self._val_eval_batches[:max_batches]:
                idx, targets = self._to_device_batch(idx_cpu, targets_cpu)
                loss = self._forward_loss(idx, targets)
                token_count = int(targets_cpu.numel())
                total_ce_nats += float(loss.detach().cpu().item()) * token_count
                target_np = targets_cpu.numpy()
                total_bytes += float(self._token_bytes_np[target_np].sum())

        if was_training:
            self.model.train()
        if total_bytes <= 0.0:
            raise ValueError("Hidden validation batch has zero byte count")
        return float(total_ce_nats / (math.log(2.0) * total_bytes))

    def hidden_holdout_ce(self, max_batches: int = 8) -> float:
        """HIDDEN CHANNEL: policies and simulators must never call this method."""
        return self._hidden_mean_ce(self._holdout_eval_batches, max_batches)

    @staticmethod
    def value(state: LearningState) -> float:
        return -float(np.mean(np.asarray(state.probe_losses, dtype=np.float64)))
