"""PR-E durable job authority — the single JobStore boundary.

Ownership split (PR-E §4), replacing the PR-A ``OrderedDict + queue.Queue``
storage while keeping every public /v1 semantic frozen:

* :class:`JobStore` (this module) — durable job metadata, queue ordering,
  transactional claims, retention/eviction authority.
* :class:`~openlithohub.server.runtime.ServerRuntime` — lifecycle,
  admission, worker and upload-reservation ownership.
* ``workflow/execution.py`` — dense/streaming execution authority.
* filesystem — durable input/output bytes under
  ``OPENLITHOHUB_STATE_DIR/jobs/<job-id>/``.

Two backends implement one protocol:

* :class:`InMemoryJobStore` — the historical lightweight mode
  (``durable = false``, jobs lost on restart).
* :class:`SQLiteJobStore` — stdlib ``sqlite3`` WAL store whose **rows are
  the queue** (``durable = true``). The in-memory wake signal is only an
  optimization: a committed QUEUED row is discovered by polling even if
  every wake signal is lost (PR-E E6).

Crash/restart semantics (PR-E §12) live in :meth:`SQLiteJobStore.recover`:

* stale ``RUNNING`` (process died mid-execution) → ``FAILED`` with an
  explicit restart-interruption error; **no automatic retry**;
* ``QUEUED`` with intact durable input stays queued and runs after
  restart; with missing input it is cancelled with an explicit error —
  never silently dropped;
* terminal jobs stay terminal; a committed ``SUCCEEDED`` and its artifact
  survive restart.

Single-process ownership is enforced fail-fast: the SQLite store holds an
exclusive advisory lock on its state directory, so a second process that
tries to operate the same store dies at startup instead of corrupting the
truth while capabilities still claim single-process authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import shutil
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from openlithohub.server.schemas import JobStatus, transition_job_status

logger = logging.getLogger(__name__)

PARAMS_SCHEMA = "openlithohub.job-params.v1"
"""Versioned deterministic params envelope (PR-E §10). Never pickle."""

SQLITE_SCHEMA_VERSION = 1

_RESTART_INTERRUPT_ERROR = (
    "server restarted while job execution was in progress; outcome was not committed"
)


def _dumps_summary(summary: dict[str, Any] | None) -> str | None:
    return json.dumps(summary, sort_keys=True) if summary is not None else None


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def unix_now() -> float:
    return time.time()


def serialize_params(params: dict[str, Any]) -> str:
    """Deterministic versioned JSON envelope for persisted execution params."""
    return json.dumps({"schema": PARAMS_SCHEMA, "params": params}, sort_keys=True)


def parse_params(raw: str | None) -> dict[str, Any]:
    """Parse persisted params, failing closed on anything unexpected.

    Raises :class:`ValueError` for malformed JSON, an unknown schema
    version or non-dict params — the caller converts that into an honest
    job failure, never into silent misexecution.
    """
    if raw is None:
        raise ValueError("job has no persisted parameters")
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"persisted parameters are not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != PARAMS_SCHEMA:
        raise ValueError(
            f"persisted parameters carry unknown schema {envelope.get('schema')!r}; "
            f"expected {PARAMS_SCHEMA!r}"
        )
    params = envelope.get("params")
    if not isinstance(params, dict):
        raise ValueError("persisted parameters envelope has no params object")
    return params


@dataclass
class JobRecord:
    """Internal job row. Private fields (paths, artifact digests, leases)
    never cross the public boundary — the runtime projects them through
    ``JobStatusResponse``."""

    job_id: str
    status: JobStatus
    created_utc: str = field(default_factory=utc_now)
    started_utc: str | None = None
    completed_utc: str | None = None
    input_path: str | None = None
    output_path: str | None = None
    input_format: str | None = None
    params_json: str | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifact_sha256: str | None = None
    artifact_bytes: int | None = None
    # Retention bookkeeping (unix epoch seconds; monotonic clocks are
    # meaningless across restarts, so the durable backend stores wall time).
    last_access_unix: float = field(default_factory=unix_now)
    created_unix: float = field(default_factory=unix_now)
    completed_unix: float | None = None


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Digest + byte size of a file, streamed (artifact publication)."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class JobStore(Protocol):
    """Storage authority for job metadata + queue ordering (PR-E §4).

    Implementations: :class:`InMemoryJobStore`, :class:`SQLiteJobStore`.
    """

    def create_queued(self, record: JobRecord) -> None: ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def count_queued(self) -> int: ...

    def count_running(self) -> int: ...

    def count_tracked(self) -> int: ...

    def peek_next_queued(self) -> str | None: ...

    def claim_next(self, lease_owner: str) -> JobRecord | None: ...

    def transition(
        self,
        job_id: str,
        new_status: JobStatus,
        *,
        error: str | None = None,
        summary: dict[str, Any] | None = None,
        output_path: str | None = None,
        artifact_sha256: str | None = None,
        artifact_bytes: int | None = None,
    ) -> JobRecord: ...

    def touch_artifact(self, job_id: str) -> None: ...

    def delete(self, job_id: str) -> JobRecord: ...

    def evict_terminal(
        self, *, ttl_seconds: float, history_cap: int, skip_job_ids: frozenset[str]
    ) -> list[JobRecord]: ...

    def recover(self) -> dict[str, list[str]]: ...

    def collect_garbage(self) -> list[str]: ...

    def snapshot_counts(self) -> dict[str, int]: ...

    def close(self) -> None: ...


class InMemoryJobStore:
    """The historical process-local store, extracted verbatim in behavior.

    Non-durable by definition: restart loses jobs. Queue ordering is the
    insertion order of QUEUED records.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()

    def create_queued(self, record: JobRecord) -> None:
        record.status = JobStatus.QUEUED
        with self._lock:
            self._jobs[record.job_id] = record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def count_queued(self) -> int:
        with self._lock:
            return sum(1 for r in self._jobs.values() if r.status is JobStatus.QUEUED)

    def count_running(self) -> int:
        with self._lock:
            return sum(1 for r in self._jobs.values() if r.status is JobStatus.RUNNING)

    def count_tracked(self) -> int:
        with self._lock:
            return len(self._jobs)

    def peek_next_queued(self) -> str | None:
        with self._lock:
            for record in self._jobs.values():
                if record.status is JobStatus.QUEUED:
                    return record.job_id
            return None

    def claim_next(self, lease_owner: str) -> JobRecord | None:
        """Transactional claim: the first QUEUED record leaves the queue in
        one critical section (QUEUED -> RUNNING)."""
        with self._lock:
            for _job_id, record in self._jobs.items():
                if record.status is JobStatus.QUEUED:
                    record.status = transition_job_status(record.status, JobStatus.RUNNING)
                    record.started_utc = utc_now()
                    return record
            return None

    def transition(
        self,
        job_id: str,
        new_status: JobStatus,
        *,
        error: str | None = None,
        summary: dict[str, Any] | None = None,
        output_path: str | None = None,
        artifact_sha256: str | None = None,
        artifact_bytes: int | None = None,
    ) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]  # KeyError -> UnknownJobError upstream
            record.status = transition_job_status(record.status, new_status)
            if new_status is JobStatus.RUNNING and record.started_utc is None:
                record.started_utc = utc_now()
            if new_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                record.completed_utc = utc_now()
                record.completed_unix = unix_now()
            if error is not None:
                record.error = error
            if summary is not None:
                record.summary = summary
            if output_path is not None:
                record.output_path = output_path
            if artifact_sha256 is not None:
                record.artifact_sha256 = artifact_sha256
            if artifact_bytes is not None:
                record.artifact_bytes = artifact_bytes
            return record

    def touch_artifact(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.last_access_unix = unix_now()

    def delete(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.pop(job_id)
            return record

    def evict_terminal(
        self, *, ttl_seconds: float, history_cap: int, skip_job_ids: frozenset[str]
    ) -> list[JobRecord]:
        terminal = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
        now = unix_now()
        with self._lock:
            doomed: list[JobRecord] = []
            for job_id in list(self._jobs.keys()):
                record = self._jobs[job_id]
                if record.status not in terminal or job_id in skip_job_ids:
                    continue
                reference = max(
                    record.completed_unix or 0.0,
                    record.last_access_unix or 0.0,
                    record.created_unix,
                )
                if len(self._jobs) > history_cap or now - reference > ttl_seconds:
                    self._jobs.pop(job_id, None)
                    doomed.append(record)
            return doomed

    def recover(self) -> dict[str, list[str]]:
        # Nothing durable to recover; a fresh process starts empty.
        return {"failed_stale_running": [], "cancelled_missing_input": []}

    def collect_garbage(self) -> list[str]:
        return []

    def snapshot_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "jobs_queued": sum(1 for r in self._jobs.values() if r.status is JobStatus.QUEUED),
                "jobs_running": sum(
                    1 for r in self._jobs.values() if r.status is JobStatus.RUNNING
                ),
                "jobs_tracked": len(self._jobs),
            }

    def close(self) -> None:
        pass


JOB_COLUMNS = (
    "job_id, status, created_utc, started_utc, completed_utc, input_path, "
    "output_path, input_format, params_json, summary_json, error, "
    "artifact_sha256, artifact_bytes, last_access_unix, created_unix, completed_unix"
)
"""Compile-time column list. The only fragment ever combined into SQL;
every value is bound through parameters."""

_JOB_INSERT = (
    "INSERT INTO jobs ("
    "job_id, status, created_utc, started_utc, completed_utc, input_path, "
    "output_path, input_format, params_json, summary_json, error, "
    "artifact_sha256, artifact_bytes, last_access_unix, created_unix, completed_unix"
    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
_JOB_SELECT = (
    "SELECT job_id, status, created_utc, started_utc, completed_utc, input_path, "
    "output_path, input_format, params_json, summary_json, error, "
    "artifact_sha256, artifact_bytes, last_access_unix, created_unix, completed_unix "
    "FROM jobs"
)
_JOB_SELECT_BY_ID = (
    "SELECT job_id, status, created_utc, started_utc, completed_utc, input_path, "
    "output_path, input_format, params_json, summary_json, error, "
    "artifact_sha256, artifact_bytes, last_access_unix, created_unix, completed_unix "
    "FROM jobs WHERE job_id = ?"
)


class SQLiteJobStore:
    """Durable store: committed QUEUED rows ARE the queue (PR-E §7).

    Engineering boundary (PR-E §20): stdlib sqlite3, WAL, busy_timeout,
    parameterized SQL, transactional claims, a schema-version table that
    fails closed on unknown newer databases, and an exclusive advisory
    lock on the state directory so a second process cannot operate the
    same store while capabilities claim single-process ownership.
    """

    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.jobs_dir = self.state_dir / "jobs"
        self.tmp_dir = self.state_dir / "tmp"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "jobs.sqlite3"

        # Fail-fast single-process ownership guard: an exclusive advisory
        # lock is released by the OS on process death, so a crashed owner
        # never blocks a legitimate restart (E-matrix) while a LIVE second
        # process fails loudly instead of racing the same rows.
        self._lock_path = self.state_dir / "store.lock"
        self._lock_file = open(self._lock_path, "w")  # noqa: SIM115 — owned for lifetime
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_file.close()
            raise RuntimeError(
                f"state directory {self.state_dir} is owned by another live "
                "openlithohub process (OPENLITHOHUB_JOB_BACKEND=sqlite is "
                "single-process; start the second process with its own "
                "OPENLITHOHUB_STATE_DIR)"
            ) from exc

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        # sqlite3 connections are not safe for concurrent multi-thread use
        # (segfaults, not exceptions): one lock serializes EVERY access.
        self._conn_lock = threading.Lock()
        self._closed = False
        self._init_schema()

    # ---- schema ---------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("job store is closed")

    def _init_schema(self) -> None:
        with self._conn_lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SQLITE_SCHEMA_VERSION),),
                )
            elif int(row[0]) > SQLITE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"job store {self.db_path} was written by a NEWER schema "
                    f"(version {row[0]} > {SQLITE_SCHEMA_VERSION}); refusing to "
                    "start — upgrade OpenLithoHub or use a fresh state directory"
                )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    started_utc TEXT,
                    completed_utc TEXT,
                    input_path TEXT,
                    output_path TEXT,
                    input_format TEXT,
                    params_json TEXT,
                    summary_json TEXT,
                    error TEXT,
                    artifact_sha256 TEXT,
                    artifact_bytes INTEGER,
                    last_access_unix REAL NOT NULL,
                    created_unix REAL NOT NULL,
                    completed_unix REAL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_order ON jobs (created_unix)"
            )

    # ---- row mapping ------------------------------------------------------

    @staticmethod
    def _to_record(row: tuple[Any, ...]) -> JobRecord:
        (
            job_id,
            status,
            created_utc,
            started_utc,
            completed_utc,
            input_path,
            output_path,
            input_format,
            params_json,
            summary_json,
            error,
            artifact_sha256,
            artifact_bytes,
            last_access_unix,
            created_unix,
            completed_unix,
        ) = row
        summary = None
        if summary_json is not None:
            try:
                summary = json.loads(summary_json)
            except json.JSONDecodeError:
                logger.warning("job %s has unreadable summary_json; reporting null", job_id)
        return JobRecord(
            job_id=job_id,
            status=JobStatus(status),
            created_utc=created_utc,
            started_utc=started_utc,
            completed_utc=completed_utc,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            params_json=params_json,
            summary=summary,
            error=error,
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
            last_access_unix=last_access_unix,
            created_unix=created_unix,
            completed_unix=completed_unix,
        )

    @staticmethod
    def _to_row(record: JobRecord) -> tuple[Any, ...]:
        summary_json = (
            json.dumps(record.summary, sort_keys=True) if record.summary is not None else None
        )
        return (
            record.job_id,
            record.status.value,
            record.created_utc,
            record.started_utc,
            record.completed_utc,
            record.input_path,
            record.output_path,
            record.input_format,
            record.params_json,
            summary_json,
            record.error,
            record.artifact_sha256,
            record.artifact_bytes,
            record.last_access_unix,
            record.created_unix,
            record.completed_unix,
        )

    # ---- JobStore protocol ------------------------------------------------

    def create_queued(self, record: JobRecord) -> None:
        record.status = JobStatus.QUEUED
        record.completed_utc = None
        record.completed_unix = None
        self._ensure_open()
        with self._conn_lock, self._conn:
            self._conn.execute(_JOB_INSERT, self._to_row(record))

    def get(self, job_id: str) -> JobRecord | None:
        self._ensure_open()
        with self._conn_lock:
            row = self._conn.execute(_JOB_SELECT_BY_ID, (job_id,)).fetchone()
            return self._to_record(row) if row is not None else None

    def count_queued(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (JobStatus.QUEUED.value,)
            ).fetchone()[0]
        )

    def count_running(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (JobStatus.RUNNING.value,)
            ).fetchone()[0]
        )

    def count_tracked(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])

    def peek_next_queued(self) -> str | None:
        self._ensure_open()
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT job_id FROM jobs WHERE status = ? ORDER BY created_unix ASC LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            return row[0] if row is not None else None

    def claim_next(self, lease_owner: str) -> JobRecord | None:
        """Transactionally claim the oldest QUEUED job (QUEUED -> RUNNING).

        One UPDATE with a status guard makes the claim atomic even though
        PR-E remains single-process.
        """
        self._ensure_open()
        with self._conn_lock, self._conn:
            row = self._conn.execute(
                "SELECT job_id FROM jobs WHERE status = ? ORDER BY created_unix ASC LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            job_id = row[0]
            now_utc = utc_now()
            cursor = self._conn.execute(
                "UPDATE jobs SET status = ?, started_utc = ? WHERE job_id = ? AND status = ?",
                (JobStatus.RUNNING.value, now_utc, job_id, JobStatus.QUEUED.value),
            )
            if cursor.rowcount != 1:  # pragma: no cover — guarded single writer
                return None
            claimed = self._conn.execute(_JOB_SELECT_BY_ID, (job_id,)).fetchone()
            return self._to_record(claimed) if claimed is not None else None

    def transition(
        self,
        job_id: str,
        new_status: JobStatus,
        *,
        error: str | None = None,
        summary: dict[str, Any] | None = None,
        output_path: str | None = None,
        artifact_sha256: str | None = None,
        artifact_bytes: int | None = None,
    ) -> JobRecord:
        self._ensure_open()
        with self._conn_lock, self._conn:
            row = self._conn.execute(_JOB_SELECT_BY_ID, (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            record = self._to_record(row)
            record.status = transition_job_status(record.status, new_status)
            if new_status is JobStatus.RUNNING and record.started_utc is None:
                record.started_utc = utc_now()
            terminal = new_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)
            if terminal:
                record.completed_utc = utc_now()
                record.completed_unix = unix_now()
            if error is not None:
                record.error = error
            if summary is not None:
                record.summary = summary
            if output_path is not None:
                record.output_path = output_path
            if artifact_sha256 is not None:
                record.artifact_sha256 = artifact_sha256
            if artifact_bytes is not None:
                record.artifact_bytes = artifact_bytes
            self._conn.execute(
                """UPDATE jobs SET status = ?, started_utc = ?, completed_utc = ?,
                   completed_unix = ?, error = ?, summary_json = ?, output_path = ?,
                   artifact_sha256 = ?, artifact_bytes = ? WHERE job_id = ?""",
                (
                    record.status.value,
                    record.started_utc,
                    record.completed_utc,
                    record.completed_unix,
                    record.error,
                    json.dumps(record.summary, sort_keys=True)
                    if record.summary is not None
                    else None,
                    record.output_path,
                    record.artifact_sha256,
                    record.artifact_bytes,
                    job_id,
                ),
            )
            return record

    def touch_artifact(self, job_id: str) -> None:
        self._ensure_open()
        with self._conn_lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET last_access_unix = ? WHERE job_id = ?",
                (unix_now(), job_id),
            )

    def delete(self, job_id: str) -> JobRecord:
        self._ensure_open()
        with self._conn_lock, self._conn:
            row = self._conn.execute(_JOB_SELECT_BY_ID, (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return self._to_record(row)

    def evict_terminal(
        self, *, ttl_seconds: float, history_cap: int, skip_job_ids: frozenset[str]
    ) -> list[JobRecord]:
        terminal = tuple(
            s.value for s in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)
        )
        now = unix_now()
        self._ensure_open()
        with self._conn_lock, self._conn:
            doomed: list[JobRecord] = []
            tracked = self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = self._conn.execute(
                _JOB_SELECT + " WHERE status IN (?,?,?) ORDER BY created_unix ASC",
                terminal,
            ).fetchall()
            for row in rows:
                record = self._to_record(row)
                if record.job_id in skip_job_ids:
                    continue
                reference = max(
                    record.completed_unix or 0.0,
                    record.last_access_unix or 0.0,
                    record.created_unix,
                )
                if tracked > history_cap or now - reference > ttl_seconds:
                    self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (record.job_id,))
                    tracked -= 1
                    doomed.append(record)
            return doomed

    def _jobs_rows(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT job_id FROM jobs").fetchall()]

    def recover(self) -> dict[str, list[str]]:
        """Crash/restart recovery (PR-E §12). Run once at runtime start,
        before the worker accepts work."""
        self._ensure_open()
        with self._conn_lock, self._conn:
            failed: list[str] = []
            cancelled: list[str] = []
            now_utc = utc_now()
            now_unix = unix_now()
            stale_running = self._conn.execute(
                "SELECT job_id FROM jobs WHERE status = ?", (JobStatus.RUNNING.value,)
            ).fetchall()
            for (job_id,) in stale_running:
                self._conn.execute(
                    """UPDATE jobs SET status = ?, completed_utc = ?, completed_unix = ?,
                       error = ? WHERE job_id = ?""",
                    (
                        JobStatus.FAILED.value,
                        now_utc,
                        now_unix,
                        _RESTART_INTERRUPT_ERROR,
                        job_id,
                    ),
                )
                failed.append(job_id)
            queued = self._conn.execute(
                _JOB_SELECT + " WHERE status = ?",
                (JobStatus.QUEUED.value,),
            ).fetchall()
            for row in queued:
                record = self._to_record(row)
                if not record.input_path or not Path(record.input_path).is_file():
                    self._conn.execute(
                        """UPDATE jobs SET status = ?, completed_utc = ?, completed_unix = ?,
                           error = ? WHERE job_id = ?""",
                        (
                            JobStatus.CANCELLED.value,
                            now_utc,
                            now_unix,
                            "durable input is missing after restart; job cannot run",
                            record.job_id,
                        ),
                    )
                    cancelled.append(record.job_id)
            if failed or cancelled:
                logger.warning(
                    "job store recovery: stale RUNNING -> FAILED %s; "
                    "QUEUED with missing input -> CANCELLED %s",
                    failed,
                    cancelled,
                )
            return {"failed_stale_running": failed, "cancelled_missing_input": cancelled}

    def collect_garbage(self) -> list[str]:
        """Remove orphaned per-job directories and stale tmp files from an
        interrupted upload (PR-E E5). Returns removed paths for logging."""
        self._ensure_open()
        with self._conn_lock:
            known = {r[0] for r in self._conn.execute("SELECT job_id FROM jobs").fetchall()}

        def _known() -> set[str]:
            return known

        removed: list[str] = []
        if self.jobs_dir.exists():
            for entry in self.jobs_dir.iterdir():
                if entry.is_dir() and entry.name not in _known():
                    shutil.rmtree(entry, ignore_errors=True)
                    removed.append(str(entry))
        if self.tmp_dir.exists():
            for entry in self.tmp_dir.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink()
                    removed.append(str(entry))
                except OSError:
                    pass
        # Partial artifacts from a crash inside the publication window are
        # never valid output (the SUCCEEDED commit renames them first).
        for tmp_artifact in self.jobs_dir.glob("*/artifact*.tmp*"):
            try:
                tmp_artifact.unlink()
                removed.append(str(tmp_artifact))
            except OSError:
                pass
        return removed

    def snapshot_counts(self) -> dict[str, int]:
        self._ensure_open()
        with self._conn_lock:
            return {
                "jobs_queued": int(
                    self._conn.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status = ?",
                        (JobStatus.QUEUED.value,),
                    ).fetchone()[0]
                ),
                "jobs_running": int(
                    self._conn.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status = ?",
                        (JobStatus.RUNNING.value,),
                    ).fetchone()[0]
                ),
                "jobs_tracked": int(self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]),
            }

    def close(self) -> None:
        with self._conn_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            finally:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()


def build_store(backend: str, state_dir: str | Path | None) -> Any:
    """The one factory the runtime may use to obtain a JobStore."""
    if backend == "in-memory":
        return InMemoryJobStore()
    if backend == "sqlite":
        if not state_dir:
            raise ValueError(
                "OPENLITHOHUB_JOB_BACKEND=sqlite requires OPENLITHOHUB_STATE_DIR "
                "to point at a persistent directory"
            )
        return SQLiteJobStore(state_dir)
    raise ValueError(f"unsupported job backend {backend!r}")
