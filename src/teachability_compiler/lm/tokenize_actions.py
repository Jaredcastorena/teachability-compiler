"""Tokenize curriculum-action document pools into raw uint16 token streams."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nanochat.tokenizer import get_tokenizer

from teachability_compiler.lm.corpus import CorpusSlice

_RESERVED_ACTIONS = {"holdout", "val"}
_TOKEN_CHUNK_SIZE = 4 * 1024 * 1024
_DOC_BATCH_SIZE = 512
_VOCAB_SIZE = 32768


def _sha256_16(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required tokenizer file is missing: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _canonical_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest file is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _extract_action_names(
    actions_manifest: dict[str, Any],
    present_actions: set[str],
) -> list[str]:
    candidates: list[str] = []

    raw_action_names = actions_manifest.get("action_names")
    if isinstance(raw_action_names, list):
        candidates = [str(x) for x in raw_action_names]

    if not candidates:
        raw_actions = actions_manifest.get("actions")
        if isinstance(raw_actions, dict):
            candidates = [str(x) for x in raw_actions.keys()]
        elif isinstance(raw_actions, list):
            names: list[str] = []
            for item in raw_actions:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    for key in ("name", "action", "action_name"):
                        if key in item:
                            names.append(str(item[key]))
                            break
            candidates = names

    candidates = [x for x in candidates if x not in _RESERVED_ACTIONS]
    if len(candidates) == 24 and set(candidates).issubset(present_actions):
        if len(set(candidates)) != 24:
            raise ValueError("Action names in manifest are not unique")
        return candidates

    names = sorted(present_actions - _RESERVED_ACTIONS)
    if len(names) != 24:
        raise ValueError(
            "Expected exactly 24 curriculum actions excluding holdout/val; "
            f"found {len(names)}: {names}"
        )
    return names


def _refs_by_action(assignments: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    required = {"shard_index", "row_index", "action"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignments parquet is missing required columns: {sorted(missing)}")

    refs: dict[str, list[tuple[int, int]]] = {}
    for action, frame in assignments.groupby("action", sort=False):
        shard_indices = frame["shard_index"].astype(int).to_numpy()
        row_indices = frame["row_index"].astype(int).to_numpy()
        refs[str(action)] = list(zip(shard_indices.tolist(), row_indices.tolist(), strict=True))
    return refs


def _build_corpus(shards: Sequence[Path]) -> CorpusSlice:
    shard_strings = [str(path) for path in shards]
    try:
        return CorpusSlice(shard_strings)
    except TypeError:
        try:
            return CorpusSlice(shards=shard_strings)
        except TypeError:
            return CorpusSlice(shard_paths=shard_strings)


def _text_from_doc(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "content", "document"):
            if key in doc:
                return str(doc[key])
    if isinstance(doc, pd.Series):
        for key in ("text", "content", "document"):
            if key in doc:
                return str(doc[key])
    if isinstance(doc, (tuple, list)):
        for item in reversed(doc):
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                try:
                    return _text_from_doc(item)
                except ValueError:
                    pass
    text_attr = getattr(doc, "text", None)
    if text_attr is not None:
        return str(text_attr)
    raise ValueError(f"Could not extract text from CorpusSlice document of type {type(doc)!r}")


def _coerce_docs(result: Any, refs: Sequence[tuple[int, int]]) -> list[str]:
    if isinstance(result, pd.DataFrame):
        docs = [_text_from_doc(row) for _, row in result.iterrows()]
    elif isinstance(result, dict):
        docs = []
        for ref in refs:
            if ref in result:
                docs.append(_text_from_doc(result[ref]))
            elif tuple(map(int, ref)) in result:
                docs.append(_text_from_doc(result[tuple(map(int, ref))]))
            else:
                raise KeyError(f"CorpusSlice result dictionary is missing ref {ref}")
    else:
        docs = [_text_from_doc(doc) for doc in list(result)]

    if len(docs) != len(refs):
        raise ValueError(f"CorpusSlice returned {len(docs)} docs for {len(refs)} refs")
    return docs


def _get_docs_batch(corpus: CorpusSlice, refs: Sequence[tuple[int, int]]) -> list[str]:
    rows = corpus.get_docs(list(refs), columns=("text",))
    return _coerce_docs(rows, refs)


def _iter_docs(corpus: CorpusSlice, refs: Sequence[tuple[int, int]]) -> Iterator[str]:
    for start in range(0, len(refs), _DOC_BATCH_SIZE):
        batch_refs = refs[start : start + _DOC_BATCH_SIZE]
        yield from _get_docs_batch(corpus, batch_refs)


class _TokenWriter:
    def __init__(self, path: Path, chunk_tokens: int = _TOKEN_CHUNK_SIZE) -> None:
        self.path = path
        self.chunk_tokens = int(chunk_tokens)
        self.token_count = 0
        self._pending_count = 0
        self._pending: list[np.ndarray] = []
        self._handle = path.open("wb")

    def write_ids(self, ids: Sequence[int]) -> None:
        arr64 = np.asarray(ids, dtype=np.int64)
        if arr64.size == 0:
            return
        min_id = int(arr64.min())
        max_id = int(arr64.max())
        if min_id < 0 or max_id >= _VOCAB_SIZE:
            raise ValueError(
                f"Token ids must be in [0, {_VOCAB_SIZE}); got min={min_id}, max={max_id}"
            )
        arr16 = arr64.astype(np.uint16, copy=False)
        self._pending.append(arr16)
        self._pending_count += int(arr16.size)
        self.token_count += int(arr16.size)
        if self._pending_count >= self.chunk_tokens:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        if len(self._pending) == 1:
            out = self._pending[0]
        else:
            out = np.concatenate(self._pending)
        out.tofile(self._handle)
        self._pending.clear()
        self._pending_count = 0

    def close(self) -> None:
        self.flush()
        self._handle.close()

    def __enter__(self) -> "_TokenWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _tokenize_refs(
    *,
    out_path: Path,
    corpus: CorpusSlice,
    refs: Sequence[tuple[int, int]],
    tokenizer: Any,
) -> dict[str, int]:
    doc_count = 0
    with _TokenWriter(out_path) as writer:
        for doc in _iter_docs(corpus, refs):
            ids = tokenizer.encode(doc, prepend="<|bos|>")
            writer.write_ids(ids)
            doc_count += 1
        token_count = writer.token_count
    return {"docs": doc_count, "tokens": token_count}


def _split_action_refs(
    refs: Sequence[tuple[int, int]],
    *,
    probe_docs_per_action: int,
    limit_docs_per_action: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    shuffled = list(refs)
    rng.shuffle(shuffled)
    if limit_docs_per_action >= 0:
        shuffled = shuffled[:limit_docs_per_action]
    probe_refs = shuffled[:probe_docs_per_action]
    train_refs = shuffled[probe_docs_per_action:]
    return train_refs, probe_refs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", default="data/actions/assignments.parquet")
    parser.add_argument("--manifest", default="data/actions/actions_manifest.json")
    parser.add_argument(
        "--shards",
        nargs="+",
        default=[
            "data/fineweb/000_00000.parquet",
            "data/fineweb/001_00000.parquet",
            "data/fineweb/002_00000.parquet",
            "data/fineweb/003_00000.parquet",
        ],
    )
    parser.add_argument("--out-dir", default="data/tokens")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--probe-docs-per-action", type=int, default=256)
    parser.add_argument("--limit-docs-per-action", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.seq_len) <= 0:
        raise ValueError("--seq-len must be positive")
    if int(args.probe_docs_per_action) < 0:
        raise ValueError("--probe-docs-per-action must be nonnegative")
    if int(args.limit_docs_per_action) < -1:
        raise ValueError("--limit-docs-per-action must be -1 or nonnegative")

    assignments_path = Path(args.assignments)
    manifest_path = Path(args.manifest)
    shards = [Path(x) for x in args.shards]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments parquet is missing: {assignments_path}")
    for shard in shards:
        if not shard.exists():
            raise FileNotFoundError(f"Corpus shard is missing: {shard}")

    actions_manifest = _load_json(manifest_path)
    actions_manifest_hash = actions_manifest.get("manifest_hash")
    if not isinstance(actions_manifest_hash, str) or not actions_manifest_hash:
        raise ValueError(f"{manifest_path} is missing required manifest_hash")

    assignments = pd.read_parquet(assignments_path)
    refs = _refs_by_action(assignments)
    present_actions = set(refs)
    for reserved in ("val", "holdout"):
        if reserved not in refs:
            raise ValueError(f"Assignments parquet is missing required action {reserved!r}")

    action_names = _extract_action_names(actions_manifest, present_actions)
    expected_labels = set(action_names) | _RESERVED_ACTIONS
    extra_labels = sorted(present_actions - expected_labels)
    if extra_labels:
        raise ValueError(f"Assignments parquet contains unknown action labels: {extra_labels}")

    assert _VOCAB_SIZE < 65536, "uint16 token output requires vocab < 65536"

    tokenizer_path = Path(os.path.expanduser("~/.cache/nanochat/tokenizer/tokenizer.pkl"))
    tokenizer_sha = _sha256_16(tokenizer_path)
    tokenizer = get_tokenizer()
    corpus = _build_corpus(shards)
    rng = np.random.default_rng(int(args.seed))

    actions_stats: dict[str, dict[str, int]] = {}
    print(
        f"{'action':<32} {'train_docs':>10} {'train_tokens':>14} "
        f"{'probe_docs':>10} {'probe_tokens':>14}"
    )
    for action in action_names:
        if action not in refs:
            raise ValueError(f"Assignments parquet is missing action {action!r}")

        train_refs, probe_refs = _split_action_refs(
            refs[action],
            probe_docs_per_action=int(args.probe_docs_per_action),
            limit_docs_per_action=int(args.limit_docs_per_action),
            rng=rng,
        )
        train_stats = _tokenize_refs(
            out_path=out_dir / f"train_{action}.bin",
            corpus=corpus,
            refs=train_refs,
            tokenizer=tokenizer,
        )
        probe_stats = _tokenize_refs(
            out_path=out_dir / f"probe_{action}.bin",
            corpus=corpus,
            refs=probe_refs,
            tokenizer=tokenizer,
        )
        actions_stats[action] = {
            "train_docs": train_stats["docs"],
            "train_tokens": train_stats["tokens"],
            "probe_docs": probe_stats["docs"],
            "probe_tokens": probe_stats["tokens"],
        }
        print(
            f"{action:<32} {train_stats['docs']:>10d} {train_stats['tokens']:>14d} "
            f"{probe_stats['docs']:>10d} {probe_stats['tokens']:>14d}"
        )

    val_stats = _tokenize_refs(
        out_path=out_dir / "val.bin",
        corpus=corpus,
        refs=refs["val"],
        tokenizer=tokenizer,
    )
    holdout_stats = _tokenize_refs(
        out_path=out_dir / "holdout.bin",
        corpus=corpus,
        refs=refs["holdout"],
        tokenizer=tokenizer,
    )

    manifest_payload: dict[str, Any] = {
        "action_names": action_names,
        "actions": actions_stats,
        "val": {"docs": val_stats["docs"], "tokens": val_stats["tokens"]},
        "holdout": {"docs": holdout_stats["docs"], "tokens": holdout_stats["tokens"]},
        "seq_len": int(args.seq_len),
        "tokenizer_sha256_16": tokenizer_sha,
        "actions_manifest_hash": actions_manifest_hash,
        "seed": int(args.seed),
        "git_commit": _git_commit(),
        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    }
    manifest_payload["manifest_hash"] = _canonical_hash(manifest_payload)

    with (out_dir / "tokens_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
