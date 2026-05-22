"""Content-hash verification for externally-fetched dataset bytes.

Per playbook §6, every adapter that downloads bytes from a mutable URL
(GitHub ``main`` branch, HF default branch, Google Drive ID, etc.) must
verify the bytes against a known-good SHA-256 before exposing them to
users. Mismatches refuse to load with an actionable error rather than
silently corrupting downstream metrics.

This module is the verification primitive. Adapters register their
known-good hashes via a per-class ``KNOWN_GOOD_SHA256`` mapping and call
:func:`verify_sha256` on the downloaded artifact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Read in 1 MiB chunks — large enough that hash throughput dominates
# Python overhead, small enough to avoid spiking RSS on multi-GB tars.
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class KnownGoodHash:
    """A SHA-256 / size pair captured at acquisition time.

    Attributes:
        sha256: 64-char lowercase hex digest of the file contents.
        size_bytes: Expected file size, kept alongside the hash because
            size mismatches are detectable in O(1) without rehashing the
            whole file (useful as a fast-fail in resume scenarios).
        source: Free-form string describing where the hash came from
            (e.g. ``"acquisition_log.md 2026-05-22"``). Surfaced in error
            messages so a future maintainer can audit the pin's origin.
    """

    sha256: str
    size_bytes: int
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(
                f"Invalid SHA-256: expected 64 lowercase hex chars, got {self.sha256!r}"
            )
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {self.size_bytes}")


class IntegrityError(RuntimeError):
    """Raised when a downloaded file fails its known-good hash check."""


def sha256_of_file(path: str | Path) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected: KnownGoodHash) -> None:
    """Verify ``path`` matches ``expected`` or raise :class:`IntegrityError`.

    Checks size first (cheap, catches truncated downloads in O(1)) before
    streaming the bytes through SHA-256. The error message names both the
    expected and actual digests so callers can either update the pin
    deliberately (with audit trail) or re-download.
    """
    p = Path(path)
    if not p.exists():
        raise IntegrityError(f"File not found for integrity check: {p}")
    actual_size = p.stat().st_size
    if actual_size != expected.size_bytes:
        raise IntegrityError(
            f"Size mismatch for {p}: expected {expected.size_bytes} bytes "
            f"(per {expected.source or 'pinned hash'}), got {actual_size}. "
            "The download may be truncated or the upstream artifact changed."
        )
    actual = sha256_of_file(p)
    if actual != expected.sha256:
        raise IntegrityError(
            f"SHA-256 mismatch for {p}:\n"
            f"  expected: {expected.sha256}\n"
            f"  actual:   {actual}\n"
            f"  pinned source: {expected.source or '(unspecified)'}\n"
            "If the upstream artifact has legitimately advanced, update "
            "the KNOWN_GOOD_SHA256 entry deliberately and record the new "
            "pin's provenance."
        )
