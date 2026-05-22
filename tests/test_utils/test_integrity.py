"""Tests for openlithohub._utils.integrity."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pytest

from openlithohub._utils.integrity import (
    IntegrityError,
    KnownGoodHash,
    sha256_of_file,
    verify_manifest,
    verify_sha256,
    warn_unverified_data_root,
    write_manifest,
)


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class TestKnownGoodHash:
    def test_valid_construction(self) -> None:
        h = KnownGoodHash(sha256="a" * 64, size_bytes=100, source="test")
        assert h.size_bytes == 100

    def test_uppercase_hex_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid SHA-256"):
            KnownGoodHash(sha256="A" * 64, size_bytes=1)

    def test_short_hash_rejected(self) -> None:
        with pytest.raises(ValueError, match="64 lowercase hex"):
            KnownGoodHash(sha256="a" * 63, size_bytes=1)

    def test_non_hex_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid SHA-256"):
            KnownGoodHash(sha256="g" * 64, size_bytes=1)

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            KnownGoodHash(sha256="a" * 64, size_bytes=-1)


class TestSha256OfFile:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        payload = b"hello, lithography\n" * 1000
        p = tmp_path / "f.bin"
        p.write_bytes(payload)
        assert sha256_of_file(p) == _digest(payload)

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert sha256_of_file(p) == _digest(b"")

    def test_streams_files_larger_than_chunk(self, tmp_path: Path) -> None:
        # Force ≥2 chunks (1 MiB + 1) so multi-chunk path is exercised.
        payload = b"x" * ((1 << 20) + 1)
        p = tmp_path / "big.bin"
        p.write_bytes(payload)
        assert sha256_of_file(p) == _digest(payload)


class TestVerifySha256:
    def test_match_passes_silently(self, tmp_path: Path) -> None:
        payload = b"correct"
        p = tmp_path / "f.bin"
        p.write_bytes(payload)
        verify_sha256(p, KnownGoodHash(sha256=_digest(payload), size_bytes=len(payload)))

    def test_size_mismatch_caught_first(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"short")
        with pytest.raises(IntegrityError, match="Size mismatch"):
            verify_sha256(
                p,
                KnownGoodHash(sha256=_digest(b"short"), size_bytes=999),
            )

    def test_hash_mismatch_reported(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"actual")
        with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
            verify_sha256(
                p,
                KnownGoodHash(sha256="b" * 64, size_bytes=len(b"actual")),
            )

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IntegrityError, match="not found"):
            verify_sha256(
                tmp_path / "nope.bin",
                KnownGoodHash(sha256="a" * 64, size_bytes=1),
            )

    def test_error_includes_pin_source(self, tmp_path: Path) -> None:
        """The pin's `source` field must surface in the error message so
        future maintainers can audit where the hash came from."""
        p = tmp_path / "f.bin"
        p.write_bytes(b"actual")
        with pytest.raises(IntegrityError) as exc_info:
            verify_sha256(
                p,
                KnownGoodHash(
                    sha256="b" * 64,
                    size_bytes=len(b"actual"),
                    source="acquisition_log.md 2026-05-22",
                ),
            )
        assert "acquisition_log.md 2026-05-22" in str(exc_info.value)


class TestLithoBenchKnownGoodTable:
    """The LithoBench adapter pins at least one known-good artifact hash.

    Per playbook §6, adapters that may grow auto-fetchers must register
    SHA-256s for the bytes they will receive so a regression cannot land
    a corrupted artifact.
    """

    def test_lithobench_models_pinned(self) -> None:
        from openlithohub.data.lithobench import KNOWN_GOOD_SHA256

        assert "lithomodels.tar.gz" in KNOWN_GOOD_SHA256
        pin = KNOWN_GOOD_SHA256["lithomodels.tar.gz"]
        assert pin.size_bytes > 0
        assert pin.source  # provenance must be non-empty


class TestManifest:
    def test_write_then_verify_round_trip(self, tmp_path: Path) -> None:
        (tmp_path / "a.npy").write_bytes(b"alpha")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.npy").write_bytes(b"beta-bytes")
        manifest = write_manifest(tmp_path)
        assert manifest.name == "MANIFEST.SHA256"
        verify_manifest(tmp_path)  # passes silently

    def test_verify_detects_tampering(self, tmp_path: Path) -> None:
        (tmp_path / "a.bin").write_bytes(b"original")
        write_manifest(tmp_path)
        # Tamper after manifest write.
        (tmp_path / "a.bin").write_bytes(b"changed")
        with pytest.raises(IntegrityError, match="Manifest mismatch"):
            verify_manifest(tmp_path)

    def test_verify_no_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IntegrityError, match="No MANIFEST.SHA256"):
            verify_manifest(tmp_path)


class TestWarnUnverifiedDataRoot:
    def test_warns_when_no_manifest(self, tmp_path: Path) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_unverified_data_root(tmp_path, "lithobench")
        assert any("MANIFEST.SHA256" in str(w.message) for w in caught)

    def test_silent_with_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "MANIFEST.SHA256").write_text("# empty manifest\n")
        # Use a fresh path to avoid the dedup cache from earlier tests.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_unverified_data_root(tmp_path, "freepdk45")
        # Manifest present → no warning emitted.
        assert not any("MANIFEST.SHA256" in str(w.message) for w in caught)
