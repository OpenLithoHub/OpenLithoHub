"""SOTA tracking and leaderboard management.

The store is a single JSON file shared across CLI invocations and the Spaces
app. Read-modify-write therefore needs:
- A POSIX advisory lock (`fcntl.flock`) on a sidecar `.lock` file so two
  concurrent submitters serialize.
- An atomic rename (`tempfile` + `os.replace`) so the file is never
  partially written.

Submission IDs are stored under the public ``submission_id`` field of
``BenchmarkResult`` so they round-trip through `model_validate`.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openlithohub.leaderboard.schema import BenchmarkResult

_DEFAULT_LEADERBOARD_DIR = Path.home() / ".openlithohub"
_LEADERBOARD_FILENAME = "leaderboard.json"

# Submission IDs prefix `model_name` truncated to this many characters,
# then 8 hex chars of randomness. The truncation is for table readability
# (schema.py allows 120-char names) — full name is on the entry itself.
_ID_NAME_PREFIX_LEN = 20

# Bump when the on-disk shape changes (new required fields, renamed fields,
# changed enum values). Files without this key are treated as v1 (legacy).
# Add a migration in `_migrate_entries` when bumping.
#
# v2 (2026-05): added wafer-level fields (``epe_wafer_*``, ``l2_error_*``) to
# ``BenchmarkResult``; ranking key moved from ``epe_mean_nm`` (mask-level,
# Identity scores 0) to ``l2_error_pixels`` with ``pvband_mean_nm`` as the
# secondary key. Older v1 entries still load — the new fields are optional
# and entries without them sort to the bottom of the table.
LEADERBOARD_SCHEMA_VERSION = 2


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-platform exclusive lock on a sidecar file.

    POSIX uses ``fcntl.flock``; Windows uses ``msvcrt.locking``. The lock file
    is opened in append mode so a fresh holder does not truncate the file
    while a prior holder still has it open.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        with open(lock_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return
    except ImportError:
        pass

    # Windows: use msvcrt.locking on a single byte at offset 0.
    import msvcrt  # type: ignore[import-not-found,unused-ignore]

    with open(lock_path, "a+b") as f:
        f.seek(0)
        # Block until the lock is acquired; retry on transient EAGAIN.
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined,unused-ignore]
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            f.seek(0)
            with contextlib.suppress(OSError):
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined,unused-ignore]


class LeaderboardStore:
    """JSON file-backed leaderboard data store."""

    def __init__(self, path: Path | None = None) -> None:
        if path is not None:
            self._path = Path(path)
        else:
            env_path = os.environ.get("OPENLITHOHUB_LEADERBOARD_PATH")
            if env_path:
                self._path = Path(env_path)
            else:
                self._path = _DEFAULT_LEADERBOARD_DIR / _LEADERBOARD_FILENAME

    @property
    def path(self) -> Path:
        return self._path

    @property
    def _lock_path(self) -> Path:
        return self._path.with_suffix(self._path.suffix + ".lock")

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        text = self._path.read_text(encoding="utf-8")
        data = json.loads(text)
        # Pre-schema-versioned files were written as a top-level JSON list.
        # Treat that as version 0 so the migration path is a single
        # well-defined funnel and a corrupted file is a clear error rather
        # than an `AttributeError` deep inside `.get`.
        if isinstance(data, list):
            return _migrate_entries(data, from_version=0)
        if not isinstance(data, dict):
            raise ValueError(
                f"Leaderboard file at {self._path} is neither a JSON object nor a list "
                f"(got {type(data).__name__}); refusing to load."
            )
        version = int(data.get("schema_version", 1))
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError(
                f"Leaderboard file at {self._path} has a non-list 'entries' "
                f"(got {type(entries).__name__}); refusing to load."
            )
        return _migrate_entries(entries, from_version=version)

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"schema_version": LEADERBOARD_SCHEMA_VERSION, "entries": entries},
            indent=2,
            default=str,
        )
        # tempfile in the same directory so os.replace is atomic across the
        # whole sequence (cross-device rename would otherwise be a copy).
        fd, tmp_name = tempfile.mkstemp(
            prefix=self._path.name + ".", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_name, self._path)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise

    def submit(self, result: BenchmarkResult) -> str:
        with _file_lock(self._lock_path):
            entries = self._read_entries()
            submission_id = _generate_id(result.model_name)
            entry = result.model_dump(mode="json")
            entry["submission_id"] = submission_id
            entries.append(entry)
            self._write_entries(entries)
        return submission_id

    def query(
        self,
        dataset: str | None = None,
        process_node: str | None = None,
    ) -> list[BenchmarkResult]:
        entries = self._read_entries()
        results: list[BenchmarkResult] = []
        for entry in entries:
            r = BenchmarkResult.model_validate(entry)
            if dataset and r.dataset != dataset:
                continue
            if process_node and r.process_node.value != process_node:
                continue
            results.append(r)
        # Rank by the canonical Neural-ILT printability scalar
        # (``l2_error_pixels``), then PV-band mean as a tiebreaker.
        # Entries written before v2 (no wafer fields) sort to the bottom.
        # ``epe_mean_nm`` is mask-level — kept on the entry for sanity but
        # NEVER used as the primary key, because Identity models score 0.
        results.sort(key=_ranking_key)
        return results


def _ranking_key(r: BenchmarkResult) -> tuple[float, float, float]:
    """Sort key: (l2_error_pixels, pvband_mean_nm, epe_wafer_mean_nm).

    Missing values become ``+inf`` so legacy entries (or partial submissions)
    rank below complete ones rather than appearing at the top by accident.
    """
    inf = float("inf")
    l2 = r.l2_error_pixels if r.l2_error_pixels is not None else inf
    pvb = r.pvband_mean_nm if r.pvband_mean_nm is not None else inf
    epe_w = r.epe_wafer_mean_nm if r.epe_wafer_mean_nm is not None else inf
    return (l2, pvb, epe_w)


def _generate_id(model_name: str) -> str:
    safe_name = model_name.replace(" ", "-").lower()[:_ID_NAME_PREFIX_LEN]
    return f"{safe_name}-{secrets.token_hex(4)}"


def _migrate_entries(entries: list[dict[str, Any]], *, from_version: int) -> list[dict[str, Any]]:
    """Migrate leaderboard entries from an older schema version.

    v0 (legacy bare list) → v1: a no-op; the only difference was the
    surrounding envelope. v1 → v2 added optional wafer-level fields
    (``epe_wafer_*``, ``l2_error_*``); the new fields default to ``None``,
    so the migration leaves entries untouched and Pydantic fills the
    missing keys with ``None`` at validate time.
    """
    if from_version > LEADERBOARD_SCHEMA_VERSION:
        raise ValueError(
            f"Leaderboard file was written by a newer schema "
            f"(v{from_version} > v{LEADERBOARD_SCHEMA_VERSION}). "
            "Upgrade openlithohub or move the file aside."
        )
    return entries


def _get_store() -> LeaderboardStore:
    """Resolve the default leaderboard store fresh each call.

    Previously this memoized a single ``LeaderboardStore`` at module level,
    which baked in whatever ``OPENLITHOHUB_LEADERBOARD_PATH`` happened to be
    set the first time. That made the env var effectively unchangeable
    across a process lifetime — fine in production, but it broke test
    isolation and any caller that legitimately rebinds the env mid-run.
    The default ``LeaderboardStore()`` constructor is cheap (no I/O until
    submit), so re-instantiating is free.
    """
    return LeaderboardStore()


def submit_result(result: BenchmarkResult, *, store: LeaderboardStore | None = None) -> str:
    """Submit a benchmark result to the leaderboard.

    Args:
        result: Validated BenchmarkResult entry.
        store: Optional explicit store (for testing). Uses default if None.

    Returns:
        Submission ID for tracking.
    """
    s = store or _get_store()
    return s.submit(result)


def get_leaderboard(
    dataset: str | None = None,
    process_node: str | None = None,
    *,
    store: LeaderboardStore | None = None,
) -> list[BenchmarkResult]:
    """Retrieve current leaderboard entries with optional filtering.

    Args:
        dataset: Filter by dataset name.
        process_node: Filter by process node.
        store: Optional explicit store (for testing). Uses default if None.

    Returns:
        Sorted list of BenchmarkResult entries (by L2 ascending, then
        PV-band mean, then wafer-EPE mean — see ``_ranking_key``).
    """
    s = store or _get_store()
    return s.query(dataset=dataset, process_node=process_node)
