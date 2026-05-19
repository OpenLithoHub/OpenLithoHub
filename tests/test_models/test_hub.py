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

    @pytest.mark.parametrize(
        "model_id",
        ["..", "../foo", "owner/../etc", "/abs/path", "with\x00null"],
    )
    def test_clear_cache_traversal_rejected(self, tmp_path: Path, model_id: str) -> None:
        # `clear_cache` must apply the same per-segment validation as
        # `download_weights`; otherwise a caller-controlled `..` would
        # let `shutil.rmtree` escape the cache directory.
        hub = ModelHub(cache_dir=tmp_path / "models")
        sentinel = tmp_path / "sibling"
        sentinel.mkdir()
        with pytest.raises(ValueError):
            hub.clear_cache(model_id)
        assert sentinel.exists()
        assert hub.cache_dir.exists()

    def test_list_cached_round_trips_through_clear_cache(self, tmp_path: Path) -> None:
        # Whatever shape `list_cached` returns must be a valid input to
        # `clear_cache` so the pair is composable.
        hub = ModelHub(cache_dir=tmp_path / "models")
        (hub.cache_dir / "owner--repo").mkdir()
        (hub.cache_dir / "url--abc123").mkdir()
        cached = hub.list_cached()
        assert sorted(cached) == ["owner/repo", "url--abc123"]
        for entry in cached:
            hub.clear_cache(entry)
        assert hub.list_cached() == []

    def test_url_download_writes_under_url_segment_and_round_trips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Lock the URL-cache layout: download_weights(url, sha256=...) MUST
        # write under cache_dir/url--<sha256(url)[:32]>/<filename>, and the
        # resulting on-disk segment must be exactly what list_cached returns
        # (which clear_cache then accepts). This is the write side of the
        # contract that originally drifted from list_cached/clear_cache.
        import hashlib

        url = "https://example.com/weights.bin"
        payload = b"fake-weights"
        digest = hashlib.sha256(payload).hexdigest()
        expected_segment = "url--" + hashlib.sha256(url.encode()).hexdigest()[:32]

        def fake_download_url(self: ModelHub, u: str, target: Path, sha256: str) -> Path:
            # The real _download_url does host vetting + pinned TLS; for the
            # contract test we just need to confirm download_weights routes
            # through here with the right target path.
            assert u == url
            assert sha256 == digest
            assert target.parent.name == expected_segment
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            return target

        monkeypatch.setattr(ModelHub, "_download_url", fake_download_url)
        hub = ModelHub(cache_dir=tmp_path / "models")
        out = hub.download_weights(url, filename="weights.bin", sha256=digest)

        assert out == hub.cache_dir / expected_segment / "weights.bin"
        assert out.read_bytes() == payload
        assert hub.list_cached() == [expected_segment]
        hub.clear_cache(expected_segment)
        assert hub.list_cached() == []
