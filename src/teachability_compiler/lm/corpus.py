"""Streaming access to frozen FineWeb parquet shards."""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DocRef = tuple[int, int]
RowDict = dict[str, object]


class CorpusSlice:
    """A frozen parquet corpus addressed by stable ``(shard_index, row_index)`` refs.

    ``shard_index`` is the position of the parquet path in ``shard_paths``.
    ``row_index`` is the zero-based row number within that shard. The pair is
    intended to be a permanent document identifier for manifests and assignment
    files.
    """

    def __init__(self, shard_paths: Sequence[str | Path]) -> None:
        """Verify shard paths and cache parquet row-count metadata."""

        if not shard_paths:
            raise ValueError("CorpusSlice requires at least one parquet shard")

        self._shard_paths = tuple(Path(path) for path in shard_paths)
        self._row_counts: list[int] = []
        self._row_group_offsets: list[tuple[int, ...]] = []

        for path in self._shard_paths:
            if not path.exists():
                raise FileNotFoundError(f"FineWeb shard does not exist: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"FineWeb shard is not a file: {path}")

            parquet_file = pq.ParquetFile(path)
            metadata = parquet_file.metadata
            self._row_counts.append(int(metadata.num_rows))

            offsets = [0]
            total_rows = 0
            for row_group_index in range(metadata.num_row_groups):
                total_rows += int(metadata.row_group(row_group_index).num_rows)
                offsets.append(total_rows)

            if total_rows != metadata.num_rows:
                raise ValueError(f"Parquet metadata row-count mismatch in shard: {path}")
            self._row_group_offsets.append(tuple(offsets))

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        """Shard paths in stable shard-index order."""

        return self._shard_paths

    @property
    def doc_count(self) -> int:
        """Total number of documents across all shards."""

        return int(sum(self._row_counts))

    def iter_doc_batches(
        self,
        columns: Sequence[str] = ("text", "id", "token_count"),
        *,
        batch_size: int = 8192,
    ) -> Iterator[tuple[np.ndarray, dict[str, list[object]]]]:
        """Yield batches of document refs and requested columns.

        The returned refs array has shape ``[batch_rows, 2]`` with columns
        ``shard_index`` and ``row_index``. Iteration is pyarrow-batched and never
        loads all shard text into memory.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        requested_columns = list(columns)
        for shard_index, path in enumerate(self._shard_paths):
            parquet_file = pq.ParquetFile(path)
            row_index = 0

            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=requested_columns,
            ):
                batch_rows = int(batch.num_rows)
                refs = np.empty((batch_rows, 2), dtype=np.int64)
                refs[:, 0] = shard_index
                refs[:, 1] = np.arange(row_index, row_index + batch_rows, dtype=np.int64)

                names = list(batch.schema.names)
                data = {
                    name: batch.column(column_index).to_pylist()
                    for column_index, name in enumerate(names)
                }

                yield refs, data
                row_index += batch_rows

    def iter_docs(
        self,
        columns: Sequence[str] = ("text", "id", "token_count"),
    ) -> Iterator[tuple[DocRef, RowDict]]:
        """Yield ``(doc_ref, row_dict)`` for each document in corpus order."""

        requested_columns = tuple(columns)
        for refs, batch in self.iter_doc_batches(requested_columns):
            for row_offset in range(int(refs.shape[0])):
                doc_ref = (int(refs[row_offset, 0]), int(refs[row_offset, 1]))
                row = {column: batch[column][row_offset] for column in requested_columns}
                yield doc_ref, row

    def get_docs(self, refs: Sequence[DocRef], columns: Sequence[str]) -> list[RowDict]:
        """Fetch arbitrary document refs, preserving input order.

        Refs are grouped by shard and parquet row group. The method reads only
        requested columns and only row groups containing requested rows; for
        dense requests this may still read most of a shard column-slice, which is
        acceptable for the offline corpus-action pipeline.
        """

        requested_columns = list(columns)
        normalized_refs = [(int(shard), int(row)) for shard, row in refs]
        output: list[RowDict] = [{} for _ in normalized_refs]

        refs_by_shard: dict[int, list[tuple[int, int]]] = {}
        for output_index, (shard_index, row_index) in enumerate(normalized_refs):
            if shard_index < 0 or shard_index >= len(self._shard_paths):
                raise IndexError(f"Invalid shard_index in doc ref: {shard_index}")
            if row_index < 0 or row_index >= self._row_counts[shard_index]:
                raise IndexError(
                    f"Invalid row_index in doc ref {(shard_index, row_index)}; "
                    f"shard has {self._row_counts[shard_index]} rows"
                )
            refs_by_shard.setdefault(shard_index, []).append((output_index, row_index))

        for shard_index, shard_refs in refs_by_shard.items():
            offsets = self._row_group_offsets[shard_index]
            refs_by_row_group: dict[int, list[tuple[int, int]]] = {}

            for output_index, row_index in shard_refs:
                row_group_index = bisect_right(offsets, row_index) - 1
                local_row_index = row_index - offsets[row_group_index]
                refs_by_row_group.setdefault(row_group_index, []).append(
                    (output_index, local_row_index)
                )

            parquet_file = pq.ParquetFile(self._shard_paths[shard_index])
            for row_group_index, local_refs in refs_by_row_group.items():
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=requested_columns,
                )
                column_values = {
                    column: table[column].to_pylist() for column in requested_columns
                }

                for output_index, local_row_index in local_refs:
                    output[output_index] = {
                        column: column_values[column][local_row_index]
                        for column in requested_columns
                    }

        return output


def corpus_manifest_hash(shard_paths: Sequence[str | Path]) -> str:
    """Return ``sha256[:16]`` over sorted ``MANIFEST.sha256`` contents.

    The checksum manifest is expected next to the first shard, matching the
    frozen ``data/fineweb/MANIFEST.sha256`` layout. Missing manifests are fatal
    because the corpus identity must be immutable.
    """

    paths = tuple(Path(path) for path in shard_paths)
    manifest_path = (
        paths[0].parent / "MANIFEST.sha256"
        if paths
        else Path("data/fineweb/MANIFEST.sha256")
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"FineWeb checksum manifest is missing: {manifest_path}")

    lines = manifest_path.read_bytes().splitlines()
    payload = b"\n".join(sorted(lines))
    if lines:
        payload += b"\n"
    return hashlib.sha256(payload).hexdigest()[:16]
