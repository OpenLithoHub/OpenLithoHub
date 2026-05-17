"""Tests for openlithohub.models.hub."""

from pathlib import Path

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
