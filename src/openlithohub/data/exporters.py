"""Sharded exporters for foundation-model pretraining workflows.

OpenLithoHub's adapters expose samples one at a time via ``__getitem__``.
Pretraining vision / diffusion models on millions of small samples is
I/O-bound when those samples are individual ``.pt`` or PNG files; the
ML-community standard at scale is sharded archives (WebDataset's ``.tar``
shards or Apache Parquet shards), which let the dataloader stream large
contiguous reads instead of doing one ``stat`` per sample.

This module ships writers for both formats:

- ``WebdatasetExporter``  — writes ``shard-NNNNN.tar`` archives whose
  records follow the ``<key>.<ext>`` convention readable by
  ``webdataset.WebDataset(...)``. Built on stdlib ``tarfile`` so the
  ``webdataset`` package is *not* a write-side dependency.
- ``ParquetExporter``     — writes ``shard-NNNNN.parquet`` files with
  one row per sample. Built on ``pyarrow`` (already a transitive dep of
  ``datasets`` / ``huggingface_hub``).

Per-sample record (both formats):

- ``key``     — stable string ``"<dataset_tag>-<index_zero_padded>"``.
                Re-derivable from index alone, so re-exports produce the
                exact same shard layout (issue #12 determinism question).
- ``design``  — ``np.float32`` array, serialised as ``.npy`` bytes.
- ``mask``    — ``np.float32`` array, serialised as ``.npy`` bytes;
                omitted (WebDataset) or null (Parquet) when the source
                sample has no mask.
- ``meta``    — JSON object with the adapter's per-sample metadata,
                Croissant-compatible at the dataset level via
                ``DatasetAdapter.to_croissant``.

Sharding strategy:

- Shard count comes from ``--shards N`` (default: ``len(adapter)`` rows
  per shard, i.e. one shard). Sample ``i`` lands in shard ``i % N``;
  this round-robin keeps shard sizes within 1 of each other and is
  stable across re-exports.
- ``--shard-size SIZE`` (e.g. ``"1GB"``, ``"500MB"``) is an alternative
  knob that derives ``N`` from the post-write size of the first batch.
  Mutually exclusive with ``--shards``.

Out of scope (deliberate, per issue #12 non-goals):

- Hugging Face Hub upload. Local export only; the user runs
  ``huggingface-cli upload`` themselves on the produced shards. We may
  revisit ``--hf-repo --push`` in a follow-up.
- ``tar.gz`` compression. Lossy on float arrays; not the WebDataset
  community default.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?B)\s*$", re.IGNORECASE)
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "PB": 1024**5}


def parse_size(spec: str) -> int:
    """Parse a human-readable size spec like ``'1GB'`` into bytes.

    Accepts ``B`` / ``KB`` / ``MB`` / ``GB`` / ``TB`` / ``PB`` suffixes
    (case-insensitive). Uses 1024-based units to match how WebDataset
    documentation specifies shard sizes.
    """
    match = _SIZE_RE.match(spec)
    if not match:
        raise ValueError(
            f"Cannot parse size {spec!r}; expected formats like '1GB', '500MB', '1024B'."
        )
    value, unit = match.groups()
    return int(float(value) * _SIZE_UNITS[unit.upper()])


def _tensor_to_npy_bytes(tensor: torch.Tensor) -> bytes:
    """Serialise a tensor as ``.npy`` bytes (the WebDataset / numpy convention)."""
    arr = tensor.detach().cpu().numpy()
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _meta_to_json(metadata: dict[str, Any]) -> bytes:
    """JSON-serialise per-sample metadata, coercing numpy/torch scalars."""

    def default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot JSON-serialise {type(obj).__name__}: {obj!r}")

    return json.dumps(metadata, default=default, sort_keys=True).encode("utf-8")


def _stable_key(dataset_tag: str, index: int, total: int) -> str:
    """``<tag>-NNNNN`` zero-padded to the index width that fits ``total``."""
    width = max(5, len(str(max(total - 1, 0))))
    return f"{dataset_tag}-{index:0{width}d}"


@dataclass(frozen=True)
class _SampleBlob:
    """Pre-serialised sample bytes ready to be written into any shard format."""

    key: str
    design_npy: bytes
    mask_npy: bytes | None
    meta_json: bytes
    nominal_bytes: int  # for shard-size accounting

    @classmethod
    def from_sample(cls, key: str, sample: LithoSample) -> _SampleBlob:
        design = _tensor_to_npy_bytes(sample.design)
        mask = _tensor_to_npy_bytes(sample.mask) if sample.mask is not None else None
        meta = _meta_to_json(sample.metadata)
        size = len(design) + len(meta) + (len(mask) if mask is not None else 0)
        return cls(key=key, design_npy=design, mask_npy=mask, meta_json=meta, nominal_bytes=size)


def _iter_blobs(adapter: DatasetAdapter, dataset_tag: str) -> Iterator[_SampleBlob]:
    total = len(adapter)
    for i in range(total):
        sample = adapter[i]
        yield _SampleBlob.from_sample(_stable_key(dataset_tag, i, total), sample)


def _resolve_shard_count(
    adapter: DatasetAdapter,
    dataset_tag: str,
    shards: int | None,
    shard_size_bytes: int | None,
) -> int:
    """Resolve the final shard count from the user's --shards / --shard-size knob."""
    total = len(adapter)
    if shards is not None and shard_size_bytes is not None:
        raise ValueError("--shards and --shard-size are mutually exclusive")
    if shards is not None:
        if shards < 1:
            raise ValueError(f"--shards must be >= 1, got {shards}")
        return min(shards, max(total, 1))
    if shard_size_bytes is not None:
        # Probe the first sample to estimate per-record bytes, then derive N.
        if total == 0:
            return 1
        first = next(_iter_blobs(adapter, dataset_tag))
        per_sample = max(first.nominal_bytes, 1)
        target_records_per_shard = max(shard_size_bytes // per_sample, 1)
        return max(1, (total + target_records_per_shard - 1) // target_records_per_shard)
    return 1


class WebdatasetExporter:
    """Write a dataset as ``shard-NNNNN.tar`` archives in WebDataset format.

    Args:
        adapter: any ``DatasetAdapter`` with ``len()`` support (streaming
            adapters are rejected — random access is required for
            deterministic shard assignment).
        output_dir: directory to write shards into. Created if missing.
        dataset_tag: short string used as the per-record key prefix.
            Defaults to ``adapter.croissant_name()`` lowercased.
        shards: explicit shard count. ``None`` (default) plus no
            ``shard_size_bytes`` means a single shard.
        shard_size_bytes: target size per shard (post-tar overhead is
            small but not zero, so the actual size is approximate).
            Mutually exclusive with ``shards``.

    The writer also emits ``croissant.json`` alongside the shards
    (dataset-level Croissant metadata from
    ``adapter.to_croissant()``) so consumers can find the license and
    citation without reading any shard.
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        output_dir: str | Path,
        *,
        dataset_tag: str | None = None,
        shards: int | None = None,
        shard_size_bytes: int | None = None,
    ) -> None:
        if not adapter.supports_random_access:
            raise TypeError(
                "WebdatasetExporter requires a random-access adapter; got a "
                "streaming adapter (supports_random_access=False)."
            )
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.dataset_tag = (dataset_tag or adapter.croissant_name()).lower().replace(" ", "-")
        self.shards = _resolve_shard_count(adapter, self.dataset_tag, shards, shard_size_bytes)

    def export(self) -> list[Path]:
        """Write all shards. Returns the list of shard paths in shard-index order."""
        from contextlib import ExitStack

        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [self.output_dir / f"shard-{i:05d}.tar" for i in range(self.shards)]
        with ExitStack() as stack:
            # Open all shard tarfiles up front so we can dispatch records round-robin.
            shard_writers = [stack.enter_context(tarfile.open(p, mode="w")) for p in shard_paths]
            for i, blob in enumerate(_iter_blobs(self.adapter, self.dataset_tag)):
                _add_blob_to_tar(shard_writers[i % self.shards], blob)

        # Drop dataset-level Croissant metadata next to the shards.
        croissant_path = self.output_dir / "croissant.json"
        croissant_path.write_text(
            json.dumps(self.adapter.to_croissant(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return shard_paths


def _add_blob_to_tar(writer: tarfile.TarFile, blob: _SampleBlob) -> None:
    _add_member(writer, f"{blob.key}.design.npy", blob.design_npy)
    if blob.mask_npy is not None:
        _add_member(writer, f"{blob.key}.mask.npy", blob.mask_npy)
    _add_member(writer, f"{blob.key}.meta.json", blob.meta_json)


def _add_member(writer: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0  # deterministic: re-exports produce byte-identical shards
    writer.addfile(info, io.BytesIO(payload))


class ParquetExporter:
    """Write a dataset as ``shard-NNNNN.parquet`` files.

    One row per sample with columns ``key``, ``design`` (npy bytes),
    ``mask`` (nullable npy bytes), ``meta`` (JSON string). Snappy
    compression by default — well-supported across pandas / pyarrow /
    duckdb / polars / Hugging Face ``datasets``.

    See :class:`WebdatasetExporter` for shared semantics
    (deterministic keys, round-robin sharding, Croissant emission).
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        output_dir: str | Path,
        *,
        dataset_tag: str | None = None,
        shards: int | None = None,
        shard_size_bytes: int | None = None,
        compression: str = "snappy",
    ) -> None:
        if not adapter.supports_random_access:
            raise TypeError(
                "ParquetExporter requires a random-access adapter; got a "
                "streaming adapter (supports_random_access=False)."
            )
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.dataset_tag = (dataset_tag or adapter.croissant_name()).lower().replace(" ", "-")
        self.shards = _resolve_shard_count(adapter, self.dataset_tag, shards, shard_size_bytes)
        self.compression = compression

    def export(self) -> list[Path]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [self.output_dir / f"shard-{i:05d}.parquet" for i in range(self.shards)]
        # Buffer rows per-shard before writing; per-shard size is bounded by
        # len(adapter)//shards, so this stays well under any reasonable RAM
        # budget for the foundation-model use case (issue #12).
        buckets: list[list[_SampleBlob]] = [[] for _ in range(self.shards)]
        for i, blob in enumerate(_iter_blobs(self.adapter, self.dataset_tag)):
            buckets[i % self.shards].append(blob)

        for path, rows in zip(shard_paths, buckets, strict=True):
            keys = [r.key for r in rows]
            designs = [r.design_npy for r in rows]
            masks = [r.mask_npy for r in rows]
            metas = [r.meta_json.decode("utf-8") for r in rows]
            table = pa.table(
                {
                    "key": pa.array(keys, type=pa.string()),
                    "design": pa.array(designs, type=pa.binary()),
                    "mask": pa.array(masks, type=pa.binary()),
                    "meta": pa.array(metas, type=pa.string()),
                }
            )
            pq.write_table(table, path, compression=self.compression)  # type: ignore[no-untyped-call,unused-ignore]

        croissant_path = self.output_dir / "croissant.json"
        croissant_path.write_text(
            json.dumps(self.adapter.to_croissant(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return shard_paths
