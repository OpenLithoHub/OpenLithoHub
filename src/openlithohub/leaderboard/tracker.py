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
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openlithohub.leaderboard.schema import BenchmarkResult

_DEFAULT_LEADERBOARD_DIR = Path.home() / ".openlithohub"
_LEADERBOARD_FILENAME = "leaderboard.json"

# Bump when the on-disk shape changes (new required fields, renamed fields,
# changed enum values). Files without this key are treated as v1 (legacy).
# Add a migration in `_migrate_entries` when bumping.
LEADERBOARD_SCHEMA_VERSION = 1


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
        entries: list[dict[str, Any]] = data.get("entries", [])
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
        results.sort(key=lambda r: r.epe_mean_nm)
        return results


def _generate_id(model_name: str) -> str:
    ts_hex = f"{int(time.time() * 1000):x}"[-8:]
    safe_name = model_name.replace(" ", "-").lower()[:20]
    return f"{safe_name}-{ts_hex}"


def _migrate_entries(entries: list[dict[str, Any]], *, from_version: int) -> list[dict[str, Any]]:
    """Migrate leaderboard entries from an older schema version.

    Currently a no-op: only schema v1 exists. When bumping
    LEADERBOARD_SCHEMA_VERSION, add a stepwise migration here so older
    on-disk files keep loading.
    """
    if from_version > LEADERBOARD_SCHEMA_VERSION:
        raise ValueError(
            f"Leaderboard file was written by a newer schema "
            f"(v{from_version} > v{LEADERBOARD_SCHEMA_VERSION}). "
            "Upgrade openlithohub or move the file aside."
        )
    return entries


_default_store: LeaderboardStore | None = None


def _get_store() -> LeaderboardStore:
    global _default_store  # noqa: PLW0603
    if _default_store is None:
        _default_store = LeaderboardStore()
    return _default_store


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
        Sorted list of BenchmarkResult entries (by EPE ascending).
    """
    s = store or _get_store()
    return s.query(dataset=dataset, process_node=process_node)
