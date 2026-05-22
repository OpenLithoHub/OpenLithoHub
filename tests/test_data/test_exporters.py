"""Tests for ``openlithohub.data.exporters`` (WebDataset + Parquet writers)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample
from openlithohub.data.exporters import (
    ParquetExporter,
    WebdatasetExporter,
    parse_size,
)


class _MemAdapter(DatasetAdapter):
    """In-memory adapter for round-trip tests; bypasses any file I/O."""

    def __init__(self, n: int, *, with_mask: bool = True, name: str = "memds") -> None:
        self._n = n
        self._with_mask = with_mask
        self._name = name

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> LithoSample:
        if not 0 <= index < self._n:
            raise IndexError(index)
        rng = np.random.default_rng(seed=index)
        design = torch.from_numpy(rng.random((4, 4), dtype=np.float32))
        mask = torch.from_numpy(rng.random((4, 4), dtype=np.float32)) if self._with_mask else None
        return LithoSample(
            design=design,
            mask=mask,
            metadata={"cell_name": f"cell_{index}", "index": index, "license": "CC0"},
        )

    def download(self, root: str) -> None:
        return None

    def croissant_name(self) -> str:
        return self._name


class _StreamingAdapter(_MemAdapter):
    @property
    def supports_random_access(self) -> bool:
        return False


# ---------- parse_size ----------


class TestParseSize:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("1B", 1),
            ("1KB", 1024),
            ("1MB", 1024**2),
            ("1GB", 1024**3),
            ("0.5GB", 1024**3 // 2),
            ("500MB", 500 * 1024**2),
            ("  2gb ", 2 * 1024**3),
        ],
    )
    def test_parses(self, spec: str, expected: int) -> None:
        assert parse_size(spec) == expected

    @pytest.mark.parametrize("spec", ["", "GB", "1XB", "abc", "1 1GB"])
    def test_rejects(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_size(spec)


# ---------- WebdatasetExporter ----------


class TestWebdatasetExporter:
    def test_writes_single_shard_by_default(self, tmp_path: Path) -> None:
        adapter = _MemAdapter(n=3)
        paths = WebdatasetExporter(adapter, tmp_path).export()
        assert paths == [tmp_path / "shard-00000.tar"]
        assert (tmp_path / "croissant.json").exists()

    def test_round_trip_records_match_source(self, tmp_path: Path) -> None:
        adapter = _MemAdapter(n=5)
        WebdatasetExporter(adapter, tmp_path, shards=2).export()
        # Re-read every record across both shards.
        rows: dict[str, dict[str, bytes | str]] = {}
        for p in sorted(tmp_path.glob("shard-*.tar")):
            with tarfile.open(p) as tf:
                for member in tf.getmembers():
                    key, _, ext = member.name.partition(".")
                    f = tf.extractfile(member)
                    assert f is not None
                    rows.setdefault(key, {})[ext] = f.read()

        assert len(rows) == 5
        for i in range(5):
            sample = adapter[i]
            key = next(k for k in rows if k.endswith(f"-{i:05d}"))
            design = np.load(io.BytesIO(rows[key]["design.npy"]))  # type: ignore[arg-type]
            mask = np.load(io.BytesIO(rows[key]["mask.npy"]))  # type: ignore[arg-type]
            meta = json.loads(rows[key]["meta.json"])
            np.testing.assert_array_equal(design, sample.design.numpy())
            assert sample.mask is not None
            np.testing.assert_array_equal(mask, sample.mask.numpy())
            assert meta["index"] == i
            assert meta["cell_name"] == f"cell_{i}"

    def test_omits_mask_when_source_has_none(self, tmp_path: Path) -> None:
        adapter = _MemAdapter(n=2, with_mask=False)
        WebdatasetExporter(adapter, tmp_path).export()
        with tarfile.open(tmp_path / "shard-00000.tar") as tf:
            names = [m.name for m in tf.getmembers()]
        assert not any(n.endswith(".mask.npy") for n in names)
        assert sum(1 for n in names if n.endswith(".design.npy")) == 2

    def test_round_robin_shard_assignment(self, tmp_path: Path) -> None:
        adapter = _MemAdapter(n=6)
        WebdatasetExporter(adapter, tmp_path, shards=3).export()
        per_shard: list[set[int]] = []
        for p in sorted(tmp_path.glob("shard-*.tar")):
            with tarfile.open(p) as tf:
                indices = set()
                for m in tf.getmembers():
                    if m.name.endswith(".design.npy"):
                        idx_str = m.name.rsplit("-", 1)[-1].split(".", 1)[0]
                        indices.add(int(idx_str))
                per_shard.append(indices)
        assert per_shard == [{0, 3}, {1, 4}, {2, 5}]

    def test_deterministic_byte_identical_re_export(self, tmp_path: Path) -> None:
        adapter = _MemAdapter(n=4)
        a = tmp_path / "a"
        b = tmp_path / "b"
        WebdatasetExporter(adapter, a, shards=2).export()
        WebdatasetExporter(adapter, b, shards=2).export()
        for name in ("shard-00000.tar", "shard-00001.tar"):
            assert (a / name).read_bytes() == (b / name).read_bytes()

    def test_streaming_adapter_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="random-access"):
            WebdatasetExporter(_StreamingAdapter(n=3), tmp_path)

    def test_invalid_shards_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            WebdatasetExporter(_MemAdapter(n=3), tmp_path, shards=0)

    def test_shards_capped_to_total(self, tmp_path: Path) -> None:
        # Asking for more shards than samples produces one shard per sample.
        paths = WebdatasetExporter(_MemAdapter(n=2), tmp_path, shards=10).export()
        assert len(paths) == 2

    def test_shard_size_derives_count(self, tmp_path: Path) -> None:
        # Pick a tiny size so each sample lands in its own shard.
        paths = WebdatasetExporter(_MemAdapter(n=4), tmp_path, shard_size_bytes=64).export()
        assert len(paths) == 4

    def test_shards_and_shard_size_are_mutually_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            WebdatasetExporter(_MemAdapter(n=2), tmp_path, shards=1, shard_size_bytes=1024)

    def test_emits_croissant_metadata(self, tmp_path: Path) -> None:
        adapter = _MemAdapter(n=2, name="MyDS")
        WebdatasetExporter(adapter, tmp_path).export()
        meta = json.loads((tmp_path / "croissant.json").read_text())
        assert meta["name"] == "MyDS"


# ---------- ParquetExporter ----------


class TestParquetExporter:
    def test_writes_single_shard_by_default(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        adapter = _MemAdapter(n=3)
        paths = ParquetExporter(adapter, tmp_path).export()
        assert paths == [tmp_path / "shard-00000.parquet"]
        assert (tmp_path / "croissant.json").exists()

    def test_round_trip_records_match_source(self, tmp_path: Path) -> None:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        adapter = _MemAdapter(n=5)
        ParquetExporter(adapter, tmp_path, shards=2).export()

        rows: dict[str, dict[str, object]] = {}
        for p in sorted(tmp_path.glob("shard-*.parquet")):
            table = pq.read_table(p)
            for row in table.to_pylist():
                rows[row["key"]] = row
        assert len(rows) == 5

        for i in range(5):
            sample = adapter[i]
            key = next(k for k in rows if k.endswith(f"-{i:05d}"))
            design = np.load(io.BytesIO(rows[key]["design"]))  # type: ignore[arg-type]
            mask = np.load(io.BytesIO(rows[key]["mask"]))  # type: ignore[arg-type]
            meta = json.loads(rows[key]["meta"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(design, sample.design.numpy())
            assert sample.mask is not None
            np.testing.assert_array_equal(mask, sample.mask.numpy())
            assert meta["index"] == i

        # silence unused-import lint
        del pa

    def test_mask_column_nullable_when_source_has_none(self, tmp_path: Path) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        adapter = _MemAdapter(n=2, with_mask=False)
        ParquetExporter(adapter, tmp_path).export()
        table = pq.read_table(tmp_path / "shard-00000.parquet")
        for row in table.to_pylist():
            assert row["mask"] is None

    def test_streaming_adapter_rejected(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        with pytest.raises(TypeError, match="random-access"):
            ParquetExporter(_StreamingAdapter(n=3), tmp_path)
