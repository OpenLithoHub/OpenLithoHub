"""Tests for LithoBench and LithoSim data adapters."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from openlithohub.data import LithoBenchDataset, LithoSample, LithoSimDataset
from openlithohub.data.base import natural_sort_key
from openlithohub.data.transforms import align_resolution, normalize_to_binary

# ==== natural_sort_key tests ====


class TestNaturalSortKey:
    def test_pure_numeric_strings(self):
        assert sorted(["10", "2", "1"], key=natural_sort_key) == ["1", "2", "10"]

    def test_mixed_alphanumeric(self):
        ids = ["sample_10", "sample_2", "sample_1", "sample_100"]
        assert sorted(ids, key=natural_sort_key) == [
            "sample_1",
            "sample_2",
            "sample_10",
            "sample_100",
        ]

    def test_zero_padded_unaffected(self):
        ids = ["sample_0010", "sample_0002", "sample_0001"]
        assert sorted(ids, key=natural_sort_key) == [
            "sample_0001",
            "sample_0002",
            "sample_0010",
        ]


# ==== LithoSample tests ====


class TestLithoSample:
    def test_creation_full(self, sample_design, sample_mask):
        sample = LithoSample(design=sample_design, mask=sample_mask, metadata={"node": "45nm"})
        assert sample.design.shape == (64, 64)
        assert sample.mask.shape == (64, 64)
        assert sample.resist is None
        assert sample.metadata["node"] == "45nm"

    def test_creation_design_only(self, sample_design):
        sample = LithoSample(design=sample_design)
        assert sample.mask is None
        assert sample.resist is None
        assert sample.metadata == {}


# ==== LithoBench Subdirectory Layout tests ====


class TestLithoBenchSubdirectory:
    @pytest.fixture
    def subdir_dataset(self, tmp_path):
        """Create a subdirectory-layout LithoBench dataset."""
        design_dir = tmp_path / "design"
        mask_dir = tmp_path / "mask"
        resist_dir = tmp_path / "resist"
        design_dir.mkdir()
        mask_dir.mkdir()
        resist_dir.mkdir()

        for i in range(3):
            name = f"sample_{i:04d}.npy"
            np.save(design_dir / name, np.random.rand(128, 128).astype(np.float32))
            np.save(mask_dir / name, np.random.rand(128, 128).astype(np.float32))
            if i < 2:
                np.save(resist_dir / name, np.random.rand(128, 128).astype(np.float32))

        metadata = {
            "sample_0000": {"process_node": "45nm", "pitch": 90},
            "sample_0001": {"process_node": "45nm", "pitch": 64},
        }
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return tmp_path

    def test_length(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        assert len(ds) == 3

    def test_getitem_full_sample(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        sample = ds[0]
        assert isinstance(sample, LithoSample)
        assert sample.design.shape == (128, 128)
        assert sample.design.dtype == torch.float32
        assert sample.mask is not None
        assert sample.mask.shape == (128, 128)
        assert sample.resist is not None

    def test_getitem_missing_resist(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        sample = ds[2]
        assert sample.design.shape == (128, 128)
        assert sample.mask is not None
        assert sample.resist is None

    def test_metadata_populated(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        sample = ds[0]
        assert sample.metadata["dataset"] == "lithobench"
        assert sample.metadata["sample_id"] == "sample_0000"
        assert sample.metadata["process_node"] == "45nm"
        assert sample.metadata["pitch"] == 90

    def test_metadata_defaults_without_json(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        sample = ds[2]
        assert sample.metadata["dataset"] == "lithobench"
        assert sample.metadata["pixel_nm"] == 1.0
        assert "process_node" not in sample.metadata

    def test_index_out_of_range(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        with pytest.raises(IndexError):
            ds[99]
        with pytest.raises(IndexError):
            ds[-1]

    def test_iteration(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        samples = list(ds)
        assert len(samples) == 3
        assert all(isinstance(s, LithoSample) for s in samples)

    def test_sample_ids_property(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset)
        ids = ds.sample_ids
        assert ids == ["sample_0000", "sample_0001", "sample_0002"]

    def test_custom_pixel_nm(self, subdir_dataset):
        ds = LithoBenchDataset(root=subdir_dataset, pixel_nm=2.0)
        sample = ds[0]
        assert sample.metadata["pixel_nm"] == 2.0

    def test_natural_sort_ordering(self, tmp_path):
        """sample_2 must come before sample_10 (regression: lexical sort would invert)."""
        design_dir = tmp_path / "design"
        design_dir.mkdir()
        for i in [1, 2, 10, 100]:
            np.save(design_dir / f"sample_{i}.npy", np.zeros((4, 4), dtype=np.float32))
        ds = LithoBenchDataset(root=tmp_path)
        assert ds.sample_ids == ["sample_1", "sample_2", "sample_10", "sample_100"]


class TestLithoBenchFlat:
    @pytest.fixture
    def flat_dataset(self, tmp_path):
        """Create a flat-layout LithoBench dataset."""
        for i in range(2):
            sid = f"sample_{i:04d}"
            np.save(tmp_path / f"{sid}_design.npy", np.ones((64, 64), dtype=np.float32))
            np.save(tmp_path / f"{sid}_mask.npy", np.ones((64, 64), dtype=np.float32))
        return tmp_path

    def test_flat_layout_detection(self, flat_dataset):
        ds = LithoBenchDataset(root=flat_dataset)
        assert ds._layout == "flat"
        assert len(ds) == 2

    def test_flat_getitem(self, flat_dataset):
        ds = LithoBenchDataset(root=flat_dataset)
        sample = ds[0]
        assert sample.design.shape == (64, 64)
        assert sample.mask is not None
        assert sample.resist is None


class TestLithoBenchSplit:
    @pytest.fixture
    def split_dataset(self, tmp_path):
        train_dir = tmp_path / "train" / "design"
        test_dir = tmp_path / "test" / "design"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        for i in range(5):
            np.save(train_dir / f"s{i}.npy", np.zeros((32, 32), dtype=np.float32))
        for i in range(2):
            np.save(test_dir / f"s{i}.npy", np.zeros((32, 32), dtype=np.float32))
        return tmp_path

    def test_split_selection(self, split_dataset):
        train_ds = LithoBenchDataset(root=split_dataset, split="train")
        test_ds = LithoBenchDataset(root=split_dataset, split="test")
        assert len(train_ds) == 5
        assert len(test_ds) == 2


class TestLithoBenchErrors:
    def test_nonexistent_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LithoBenchDataset(root=tmp_path / "nonexistent")

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = LithoBenchDataset(root=empty)
        assert len(ds) == 0

    def test_download_not_implemented(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = LithoBenchDataset(root=empty)
        with pytest.raises(ValueError, match="Unknown LithoBench artifact"):
            ds.download(str(tmp_path / "out"), artifact="bogus.tar.gz")

    def test_download_requires_gdown(self, tmp_path):
        import sys

        empty = tmp_path / "empty"
        empty.mkdir()
        ds = LithoBenchDataset(root=empty)
        with (
            patch.dict(sys.modules, {"gdown": None}),
            pytest.raises(ImportError, match="gdown"),
        ):
            ds.download(str(tmp_path / "out"))

    def test_download_verifies_sha256_and_rejects_mismatch(self, tmp_path):
        """A gdown response with wrong bytes must surface as IntegrityError."""
        import sys

        from openlithohub._utils.integrity import IntegrityError

        empty = tmp_path / "empty"
        empty.mkdir()
        ds = LithoBenchDataset(root=empty)

        out = tmp_path / "out"

        def fake_download(url, dst, quiet):
            # Write a 1-byte file — guaranteed to fail size check.
            Path(dst).write_bytes(b"x")

        fake_gdown = MagicMock()
        fake_gdown.download.side_effect = fake_download

        with (
            patch.dict(sys.modules, {"gdown": fake_gdown}),
            pytest.raises(IntegrityError, match="Size mismatch"),
        ):
            ds.download(str(out))

    def test_download_extracts_when_hash_matches(self, tmp_path, monkeypatch):
        """Happy path: matching SHA-256 triggers tar extraction."""
        import io
        import sys
        import tarfile as tf

        from openlithohub.data import lithobench as lb_mod

        # Build a tiny tar in memory and override the pin to match its hash.
        buf = io.BytesIO()
        with tf.open(fileobj=buf, mode="w:gz") as tar:
            payload = b"hello"
            ti = tf.TarInfo(name="hello.txt")
            ti.size = len(payload)
            tar.addfile(ti, io.BytesIO(payload))
        tar_bytes = buf.getvalue()

        import hashlib

        from openlithohub._utils.integrity import KnownGoodHash

        pin = KnownGoodHash(
            sha256=hashlib.sha256(tar_bytes).hexdigest(),
            size_bytes=len(tar_bytes),
            source="test fixture",
        )

        monkeypatch.setitem(lb_mod.KNOWN_GOOD_SHA256, "lithomodels.tar.gz", pin)

        def fake_download(url, dst, quiet):
            Path(dst).write_bytes(tar_bytes)

        fake_gdown = MagicMock()
        fake_gdown.download.side_effect = fake_download

        empty = tmp_path / "empty"
        empty.mkdir()
        ds = LithoBenchDataset(root=empty)

        out = tmp_path / "out"
        with patch.dict(sys.modules, {"gdown": fake_gdown}):
            ds.download(str(out))

        assert (out / "hello.txt").read_bytes() == b"hello"

    def test_download_skips_when_already_present(self, tmp_path, monkeypatch):
        """An existing-and-valid tarball must not be re-downloaded."""
        import io
        import sys
        import tarfile as tf

        from openlithohub.data import lithobench as lb_mod

        buf = io.BytesIO()
        with tf.open(fileobj=buf, mode="w:gz") as tar:
            payload = b"already-here"
            ti = tf.TarInfo(name="x.txt")
            ti.size = len(payload)
            tar.addfile(ti, io.BytesIO(payload))
        tar_bytes = buf.getvalue()

        import hashlib

        from openlithohub._utils.integrity import KnownGoodHash

        pin = KnownGoodHash(
            sha256=hashlib.sha256(tar_bytes).hexdigest(),
            size_bytes=len(tar_bytes),
            source="test fixture",
        )
        monkeypatch.setitem(lb_mod.KNOWN_GOOD_SHA256, "lithomodels.tar.gz", pin)

        out = tmp_path / "out"
        out.mkdir()
        (out / "lithomodels.tar.gz").write_bytes(tar_bytes)

        fake_gdown = MagicMock()
        fake_gdown.download.side_effect = AssertionError("must not be called")

        empty = tmp_path / "empty"
        empty.mkdir()
        ds = LithoBenchDataset(root=empty)

        with patch.dict(sys.modules, {"gdown": fake_gdown}):
            ds.download(str(out))
        fake_gdown.download.assert_not_called()


# ==== LithoSim tests (mocked HuggingFace) ====


class TestLithoSimDataset:
    @pytest.fixture
    def mock_hf_dataset(self):
        """Create a mock HuggingFace dataset."""
        from PIL import Image

        img_design = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img_mask = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))

        rows = [
            {
                "design": img_design,
                "mask": img_mask,
                "resist": None,
                "process_node": "28nm",
                "pitch_nm": 56,
                "sample_id": "litho_0000",
            },
            {
                "design": img_design,
                "mask": img_mask,
                "resist": img_mask,
                "process_node": "28nm",
                "pitch_nm": 48,
                "sample_id": "litho_0001",
            },
        ]

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=2)
        mock_ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
        mock_ds.column_names = ["design", "mask", "resist", "process_node", "pitch_nm", "sample_id"]
        return mock_ds

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_length(self, mock_ensure, mock_hf_dataset):
        ds = LithoSimDataset(split="test")
        ds._ds = mock_hf_dataset
        assert len(ds) == 2

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_getitem(self, mock_ensure, mock_hf_dataset):
        ds = LithoSimDataset(split="test")
        ds._ds = mock_hf_dataset
        ds._len = 2

        sample = ds[0]
        assert isinstance(sample, LithoSample)
        assert sample.design.shape == (256, 256)
        assert sample.design.dtype == torch.float32
        assert sample.design.max() <= 1.0
        assert sample.mask is not None
        assert sample.resist is None

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_metadata(self, mock_ensure, mock_hf_dataset):
        ds = LithoSimDataset(split="test")
        ds._ds = mock_hf_dataset
        ds._len = 2

        sample = ds[1]
        assert sample.metadata["dataset"] == "lithosim"
        assert sample.metadata["process_node"] == "28nm"
        assert sample.metadata["pitch_nm"] == 48
        assert sample.metadata["sample_id"] == "litho_0001"

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_resist_present(self, mock_ensure, mock_hf_dataset):
        ds = LithoSimDataset(split="test")
        ds._ds = mock_hf_dataset
        ds._len = 2

        sample = ds[1]
        assert sample.resist is not None
        assert sample.resist.shape == (256, 256)

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_index_out_of_range(self, mock_ensure, mock_hf_dataset):
        ds = LithoSimDataset(split="test")
        ds._ds = mock_hf_dataset
        ds._len = 2

        with pytest.raises(IndexError):
            ds[5]

    def test_import_error_without_datasets(self):
        with (
            patch.dict("sys.modules", {"datasets": None}),
            pytest.raises(ImportError, match="datasets"),
        ):
            LithoSimDataset(split="test")

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_columns_property(self, mock_ensure, mock_hf_dataset):
        ds = LithoSimDataset(split="test")
        ds._ds = mock_hf_dataset
        assert "design" in ds.columns
        assert "mask" in ds.columns

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_gated_repo_remediation(self, mock_ensure):
        """A 401/gated load surfaces a RuntimeError with login instructions."""
        import sys

        ds = LithoSimDataset(split="test")

        class FakeGatedError(Exception):
            pass

        FakeGatedError.__name__ = "GatedRepoError"

        fake_datasets = MagicMock()
        fake_datasets.load_dataset.side_effect = FakeGatedError("401 Client Error: Unauthorized")
        with (
            patch.dict(sys.modules, {"datasets": fake_datasets}),
            pytest.raises(RuntimeError, match="huggingface-cli login") as exc_info,
        ):
            ds._load_dataset()
        assert "request access" in str(exc_info.value)

    def test_default_revision_is_pinned(self):
        """The constructor default for `revision` must not be None (irreproducible)."""
        from openlithohub.data import lithosim as lithosim_mod

        assert lithosim_mod._DEFAULT_REVISION is not None
        # Verify the constructor wires the default through.
        with patch("openlithohub.data.lithosim._ensure_datasets_available"):
            ds = LithoSimDataset(split="test")
            assert ds.revision == lithosim_mod._DEFAULT_REVISION


class TestSupportsRandomAccess:
    """Adapters declare whether `len()` / `ds[i]` are well-defined.

    Streaming adapters answer False so callers can branch between batched
    evaluation and online consumption without catching TypeError.
    """

    def test_lithobench_supports_random_access(self, tmp_path):
        design_dir = tmp_path / "design"
        design_dir.mkdir()
        np.save(design_dir / "s_0.npy", np.zeros((4, 4), dtype=np.float32))
        ds = LithoBenchDataset(root=tmp_path)
        assert ds.supports_random_access is True

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_lithosim_batched_supports_random_access(self, mock_ensure):
        ds = LithoSimDataset(split="test", streaming=False)
        assert ds.supports_random_access is True

    @patch("openlithohub.data.lithosim._ensure_datasets_available")
    def test_lithosim_streaming_does_not_support_random_access(self, mock_ensure):
        ds = LithoSimDataset(split="test", streaming=True)
        assert ds.supports_random_access is False


class TestLithoSimAuthErrorDetection:
    def test_detects_status_401(self):
        from openlithohub.data.lithosim import _is_auth_error

        exc = Exception("boom")
        exc.response = MagicMock(status_code=401)
        assert _is_auth_error(exc)

    def test_detects_status_403(self):
        from openlithohub.data.lithosim import _is_auth_error

        exc = Exception("boom")
        exc.response = MagicMock(status_code=403)
        assert _is_auth_error(exc)

    def test_detects_message_substring(self):
        from openlithohub.data.lithosim import _is_auth_error

        assert _is_auth_error(Exception("dataset is gated, request access"))
        assert _is_auth_error(Exception("HTTP 401 Unauthorized"))

    def test_passes_through_unrelated_errors(self):
        from openlithohub.data.lithosim import _is_auth_error

        assert not _is_auth_error(ValueError("bad split name"))
        assert not _is_auth_error(FileNotFoundError("missing"))


class TestLithoSimTensorConversion:
    def test_numpy_array_conversion(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        tensor = LithoSimDataset._to_tensor(arr)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (64, 64)

    def test_uint8_normalization(self):
        arr = np.full((32, 32), 255, dtype=np.uint8)
        tensor = LithoSimDataset._to_tensor(arr)
        assert tensor.max() == 1.0

    def test_uint16_normalization(self):
        # SEM/aerial-image rows in industrial litho datasets are commonly
        # uint16; without the dedicated branch the values would land in
        # [0, 65535] and silently break the [0, 1] resist threshold.
        arr = np.full((32, 32), 65535, dtype=np.uint16)
        tensor = LithoSimDataset._to_tensor(arr)
        assert tensor.dtype == torch.float32
        assert tensor.max() == 1.0
        assert tensor.min() == 1.0

    def test_uint16_midrange(self):
        arr = np.full((4, 4), 32768, dtype=np.uint16)
        tensor = LithoSimDataset._to_tensor(arr)
        assert tensor.dtype == torch.float32
        assert abs(tensor[0, 0].item() - 32768 / 65535) < 1e-6

    def test_unsupported_integer_dtype_raises(self):
        arr = np.zeros((4, 4), dtype=np.int32)
        with pytest.raises(TypeError, match="Unsupported integer dtype"):
            LithoSimDataset._to_tensor(arr)

    def test_pil_image_conversion(self):
        from PIL import Image

        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        tensor = LithoSimDataset._to_tensor(img)
        assert tensor.shape == (64, 64)
        assert tensor.dtype == torch.float32

    def test_bytes_dict_conversion(self):
        import io

        from PIL import Image

        img = Image.fromarray(np.zeros((32, 32), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        value = {"bytes": buf.getvalue()}

        tensor = LithoSimDataset._to_tensor(value)
        assert tensor.shape == (32, 32)

    def test_unsupported_type_raises(self):
        with pytest.raises((TypeError, ImportError)):
            LithoSimDataset._to_tensor("invalid")


# ==== Transform tests ====


class TestAlignResolution:
    def test_identity(self):
        t = torch.rand(64, 64)
        result = align_resolution(t, source_pixel_nm=1.0, target_pixel_nm=1.0)
        assert torch.allclose(result, t)

    def test_upscale_2x(self):
        t = torch.rand(32, 32)
        result = align_resolution(t, source_pixel_nm=2.0, target_pixel_nm=1.0)
        assert result.shape == (64, 64)

    def test_downscale_2x(self):
        t = torch.rand(64, 64)
        result = align_resolution(t, source_pixel_nm=1.0, target_pixel_nm=2.0)
        assert result.shape == (32, 32)

    def test_3d_input(self):
        t = torch.rand(1, 64, 64)
        result = align_resolution(t, source_pixel_nm=2.0, target_pixel_nm=1.0)
        assert result.shape == (1, 128, 128)

    def test_invalid_pixel_size(self):
        t = torch.rand(32, 32)
        with pytest.raises(ValueError):
            align_resolution(t, source_pixel_nm=0, target_pixel_nm=1.0)
        with pytest.raises(ValueError):
            align_resolution(t, source_pixel_nm=1.0, target_pixel_nm=-1.0)

    def test_invalid_ndim(self):
        t = torch.rand(2, 3, 64, 64)
        with pytest.raises(ValueError):
            align_resolution(t, source_pixel_nm=1.0, target_pixel_nm=2.0)

    def test_nearest_mode(self):
        t = torch.rand(32, 32)
        result = align_resolution(t, source_pixel_nm=2.0, target_pixel_nm=1.0, mode="nearest")
        assert result.shape == (64, 64)


class TestNormalizeToBinary:
    def test_basic(self):
        t = torch.tensor([0.3, 0.5, 0.7])
        result = normalize_to_binary(t)
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.equal(result, expected)

    def test_custom_threshold(self):
        t = torch.tensor([0.3, 0.5, 0.7])
        result = normalize_to_binary(t, threshold=0.3)
        expected = torch.tensor([0.0, 1.0, 1.0])
        assert torch.equal(result, expected)
