"""Tests for openlithohub.models.hub."""

from pathlib import Path

import pytest

from openlithohub.models.hub import ModelHub


class TestModelHub:
    def test_cache_dir_creation(self, tmp_path: Path) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        assert hub.cache_dir.exists()

    def test_list_cached_empty(self, tmp_path: Path) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        assert hub.list_cached() == []

    def test_clear_cache_empty(self, tmp_path: Path) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        hub.clear_cache()
        assert hub.cache_dir.exists()

    def test_clear_cache_specific_model(self, tmp_path: Path) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        model_dir = hub.cache_dir / "org--model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").write_bytes(b"fake")
        hub.clear_cache("org/model")
        assert not model_dir.exists()

    def test_download_cached_returns_existing(self, tmp_path: Path) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        cached = hub.cache_dir / "org--model" / "model.pt"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"weights")
        path = hub.download_weights("org/model", filename="model.pt")
        assert path == cached

    def test_checksum(self, tmp_path: Path) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello")
        checksum = hub.get_checksum(test_file)
        assert len(checksum) == 64  # SHA256 hex digest
        assert checksum == hub.get_checksum(test_file)  # deterministic

    @pytest.mark.parametrize(
        "filename",
        ["../../etc/passwd", "/etc/passwd", "..\\..\\windows\\system32", "subdir/weights.pt"],
    )
    def test_filename_path_traversal_rejected(self, tmp_path: Path, filename: str) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        with pytest.raises(ValueError):
            hub.download_weights("org/model", filename=filename)

    def test_model_id_path_traversal_rejected(self, tmp_path: Path) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        # Slashes get rewritten to `--` for HF-style ids before sanitisation,
        # so direct path traversal must come via raw `..` in the segment.
        with pytest.raises(ValueError):
            hub.download_weights("..", filename="model.pt")

    @pytest.mark.parametrize(
        "model_id",
        [
            "../foo",  # traversal hidden in multi-segment id (the boundary bug)
            "foo/..",
            "owner/../etc",
            "/abs/path",
            "owner\\..\\repo",
            "owner/repo/extra",  # HF ids are exactly owner/repo
            "owner//repo",  # empty middle segment
            ".",
            "owner/.",
            "with\x00null",
        ],
    )
    def test_model_id_multi_segment_traversal_rejected(self, tmp_path: Path, model_id: str) -> None:
        hub = ModelHub(cache_dir=tmp_path / "models")
        with pytest.raises(ValueError):
            hub.download_weights(model_id, filename="model.pt")
