"""App-owned server runtime — lifecycle, admission and worker ownership
(repair-plan P0.1/P0.2, PR-A, PR-D, PR-E).

Authority model:

* App CONSTRUCTION starts no threads. The FastAPI lifespan creates one
  :class:`ServerRuntime` per app instance, starts its single job worker
  on entry and terminates it on exit. ``create_app()`` is free of
  lifecycle side effects.
* PR-E: job metadata + queue ordering + retention live in a
  :class:`~openlithohub.server.job_store.JobStore` — ``in-memory`` keeps
  the historical lightweight mode, ``sqlite`` makes committed jobs
  restart-safe. The runtime keeps lifecycle, admission, the single job
  worker and the pre-upload reservation counter. The in-memory wake
  signal is only an optimization: the worker polls the store, so a lost
  wake can never lose a committed job (PR-E E6).
* Job-queue capacity is guarded by explicit transactional
  :class:`QueueReservation` objects reserved BEFORE a potentially large
  upload is ingested; in durable mode
  ``store.count_queued() + active reservations < job_queue_depth`` is
  enforced under the runtime lock. A reservation ends in exactly one
  terminal state — COMMITTED (its ``commit()`` created the durable
  QUEUED job) or RELEASED.
* Shutdown is a state machine (NEW -> RUNNING -> DRAINING -> STOPPED)
  with the strong PR-A terminal invariant: STOPPED ==> no owned worker
  thread alive AND no admitted execution in flight. Forced stops retain
  ownership truthfully until a reaper completes the transition.
* New-work gates live at the runtime authority boundary, atomically
  ordered against RUNNING -> DRAINING through ``_lifecycle_lock``.
* State introspection goes through :meth:`ServerRuntime.snapshot` — the
  PR-D observability authority; in sqlite mode job counters derive from
  the store, admission/reservation counters stay runtime-owned.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import queue  # noqa: F401 — referenced in docs; retained for reservation errors
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openlithohub.server.config import ServerConfig
from openlithohub.server.job_store import (
    JobRecord,
    JobStore,
    build_store,
    parse_params,
)
from openlithohub.server.observability import emit_event
from openlithohub.server.observability import registry as _metrics
from openlithohub.server.schemas import (
    API_SCHEMA_VERSION,
    JobStatus,
    project_optimize_metadata,
)

logger = logging.getLogger(__name__)

# How often the idle worker wakes to notice shutdown / newly committed
# durable work (the poll that makes lost wake signals harmless).
_WORKER_POLL_SECONDS = 0.25

_RESTART_INTERRUPT_ERROR = (
    "server restarted while job execution was in progress; outcome was not committed"
)


class RuntimeState(str, Enum):
    """Lifecycle state machine (repair-plan P1.2)."""

    NEW = "new"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"


class JobQueueFullError(RuntimeError):
    """Raised when the bounded job queue has no free slot."""


class UnknownJobError(KeyError):
    """Raised when a job id is not in the store."""


class JobArtifactUnavailableError(RuntimeError):
    """Raised when a job has no downloadable artifact (not succeeded)."""


class JobStillRunningError(RuntimeError):
    """Raised when a deletion targets a running job."""


class AdmissionDeniedError(RuntimeError):
    """Raised when the global optimize admission capacity is exhausted."""


class RuntimeNotAcceptingWorkError(RuntimeError):
    """Raised when the runtime is no longer accepting new work (draining
    or stopped). HTTP layer maps this to 503 — never 400/429."""


def cleanup_paths(doomed: list[str]) -> None:
    """Remove files/directories, ignoring errors. Callers run this OUTSIDE
    store/runtime locks (slow rmtree must not block job state)."""
    for d in doomed:
        if not d:
            continue
        path = Path(d)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                path.unlink()


class _ReservationState(Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    RELEASED = "released"


class QueueReservation:
    """A transactional reservation of one bounded job-queue slot.

    Contract (repair-plan P0.2, preserved by PR-E):

    * exactly one terminal state, reached exactly once: COMMITTED via
      :meth:`commit` (which creates the durable QUEUED job), or RELEASED;
    * double commit / double release raise ``RuntimeError``;
    * the reserved counter can never go negative;
    * :meth:`ServerRuntime.reserve_job_slot` releases any reservation
      left ACTIVE when its ``with`` block exits.
    """

    def __init__(self, runtime: ServerRuntime) -> None:
        self._runtime = runtime
        self._lock = threading.Lock()
        self._state = _ReservationState.ACTIVE

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    def commit(self, record: JobRecord) -> str:
        """Create the QUEUED job through the store, consuming this
        reservation exactly once. Returns the job id. On failure the
        reservation stays ACTIVE (the context manager releases it)."""
        with self._lock:
            if self._state is _ReservationState.COMMITTED:
                raise RuntimeError("queue reservation already committed")
            if self._state is _ReservationState.RELEASED:
                raise RuntimeError("queue reservation already released")
            self._runtime._enqueue_reserved(record)
            self._state = _ReservationState.COMMITTED
            return record.job_id

    def release(self) -> None:
        """Explicitly give the slot back. Raises if already terminal."""
        with self._lock:
            if self._state is _ReservationState.COMMITTED:
                raise RuntimeError("cannot release a committed queue reservation")
            if self._state is _ReservationState.RELEASED:
                raise RuntimeError("queue reservation already released")
            self._state = _ReservationState.RELEASED
        self._runtime._release_reserved_slot()

    def _release_if_active(self) -> None:
        with self._lock:
            if self._state is not _ReservationState.ACTIVE:
                return
            self._state = _ReservationState.RELEASED
        self._runtime._release_reserved_slot()


class ServerRuntime:
    """Owns admission control, the job worker thread and the JobStore
    handle for one app instance.

    Created and attached as ``app.state.runtime`` by the FastAPI lifespan.
    Constructing a runtime starts nothing — call :meth:`start` exactly
    once, then :meth:`stop` exactly once.
    """

    def __init__(
        self,
        config: ServerConfig,
        *,
        optimize_runner: Callable[..., dict[str, Any]],
    ) -> None:
        self.config = config
        self._optimize_runner = optimize_runner

        # PR-E storage authority: one JobStore per runtime.
        self._store: JobStore = build_store(config.job_backend, config.state_dir)
        self._job_counter = itertools.count(1)
        self._instance = uuid.uuid4().hex[:12]
        # Wake-up optimization only (PR-E §7); correctness comes from the
        # worker's store poll.
        self._wake = threading.Condition()

        # Artifact download leases (PR-E §14): process-local, deliberately
        # NOT durable — an active HTTP response cannot survive process
        # death. Refcount per job id; eviction must skip these.
        self._artifact_leases: dict[str, int] = {}
        self._artifact_lease_lock = threading.Lock()

        # Admission control: one bounded semaphore admits at most
        # max_concurrent_optimize concurrent optimizes; the in-use counter
        # mirrors it so snapshot() never needs private semaphore internals.
        self._admission = threading.BoundedSemaphore(config.max_concurrent_optimize)
        self._admission_meta_lock = threading.Lock()
        self._admission_in_use = 0

        # Pre-upload reservation counter (guarded by _job_lock alongside
        # store capacity checks).
        self._job_lock = threading.Lock()
        self._slots_reserved = 0

        # Lifecycle state machine.
        self._lifecycle_lock = threading.Lock()
        self._state = RuntimeState.NEW
        self._shutdown = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._stop_forced = False
        self._worker_exited = threading.Event()
        self._reaper_thread: threading.Thread | None = None

    # ---- job id / staging helpers ---------------------------------------

    def new_job_id(self) -> str:
        return f"job-{next(self._job_counter)}-{uuid.uuid4().hex[:8]}"

    def staging_path(self, job_id: str, suffix: str) -> Path:
        """Where a durable-mode in-progress upload is written: inside the
        state directory so the final ``os.replace`` onto the job input is
        same-filesystem atomic (PR-E §11). In-memory mode keeps staging in
        the request scratch dir (endpoint-provided) exactly as before."""
        from openlithohub.server.job_store import SQLiteJobStore

        if isinstance(self._store, SQLiteJobStore):
            self._store.tmp_dir.mkdir(parents=True, exist_ok=True)
            return self._store.tmp_dir / f"{job_id}.upload{suffix}"
        raise ValueError("staging_path is only used by the sqlite backend")

    def finalize_durable_input(self, job_id: str, staging: Path, suffix: str) -> str:
        """Publish an uploaded input atomically into the durable job dir
        (PR-E §11): write tmp -> fsync -> rename -> then create_queued."""
        from openlithohub.server.job_store import SQLiteJobStore

        store = self._store
        if not isinstance(store, SQLiteJobStore):
            raise RuntimeError("finalize_durable_input requires the sqlite backend")
        job_dir = store.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        final = job_dir / f"input{suffix}"
        _fsync_file(staging)
        os.replace(staging, final)
        return str(final)

    def durable_job_dir(self, job_id: str) -> Path:
        from openlithohub.server.job_store import SQLiteJobStore

        store = self._store
        if not isinstance(store, SQLiteJobStore):
            raise RuntimeError("durable_job_dir requires the sqlite backend")
        return store.jobs_dir / job_id

    def job_cleanup_dir(self, record: JobRecord) -> str:
        """Filesystem footprint of one job, for post-delete cleanup."""
        from openlithohub.server.job_store import SQLiteJobStore

        if isinstance(self._store, SQLiteJobStore):
            return str(self._store.jobs_dir / record.job_id)
        if record.input_path:
            return str(Path(record.input_path).parent)
        if record.output_path:
            return str(Path(record.output_path).parent)
        return ""

    # ---- lifecycle ----------------------------------------------------

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def stop_forced(self) -> bool:
        """True when a stop had to retain ownership past its grace period."""
        return self._stop_forced

    @property
    def accepting_jobs(self) -> bool:
        """True only while RUNNING with a live worker. DRAINING/STOPPED
        runtimes reject new submissions (repair-plan P1.2)."""
        return self._state is RuntimeState.RUNNING and self.worker_alive

    @property
    def worker_alive(self) -> bool:
        thread = self._worker_thread
        return thread is not None and thread.is_alive()

    @property
    def executing(self) -> bool:
        """True while an admitted optimize (worker job or synchronous run)
        is still in flight under this runtime's admission authority."""
        with self._admission_meta_lock:
            return self._admission_in_use > 0

    def start(self) -> None:
        """Start the single job worker. Illegal from any state but NEW.

        PR-E: crash/restart recovery runs before the worker accepts work —
        stale RUNNING fails closed, QUEUED jobs with intact durable input
        stay queued, and interrupted-upload garbage is collected.
        """
        with self._lifecycle_lock:
            if self._state is not RuntimeState.NEW:
                raise RuntimeError(
                    f"cannot start ServerRuntime in state {self._state.value!r}; "
                    "construct a new runtime instead"
                )
            recovery = self._store.recover()
            garbage = self._store.collect_garbage()
            if recovery["failed_stale_running"] or recovery["cancelled_missing_input"] or garbage:
                logger.warning(
                    "job store recovery at start: stale RUNNING -> FAILED %s; "
                    "QUEUED missing input -> CANCELLED %s; collected %d orphan path(s)",
                    recovery["failed_stale_running"],
                    recovery["cancelled_missing_input"],
                    len(garbage),
                )
            self._shutdown.clear()
            self._stop_forced = False
            self._state = RuntimeState.RUNNING
            logger.info(
                "server runtime starting: job_backend=%s durable=%s state_dir=%s "
                "(single worker process), max_concurrent_optimize=%d, job_queue_depth=%d",
                self.config.job_backend,
                self.config.job_backend == "sqlite",
                self.config.state_dir or "-",
                self.config.max_concurrent_optimize,
                self.config.job_queue_depth,
            )
            thread = threading.Thread(target=self._worker_loop, name="olh-job-worker", daemon=True)
            self._worker_thread = thread
            thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Graceful shutdown state machine (PR-A, preserved verbatim)."""
        grace = self.config.shutdown_grace_seconds if timeout is None else timeout
        reentry = False
        with self._lifecycle_lock:
            if self._state is RuntimeState.STOPPED:
                return
            if self._state is RuntimeState.DRAINING:
                reentry = True
            else:
                self._state = RuntimeState.DRAINING
        self._shutdown.set()
        with self._wake:
            self._wake.notify_all()

        if not reentry:
            self._drain_queued_jobs()

        deadline = time.monotonic() + grace
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=grace)
        while self.executing and time.monotonic() < deadline:
            time.sleep(0.02)

        if self._owns_no_execution():
            with self._lifecycle_lock:
                self._state = RuntimeState.STOPPED
            logger.info("server runtime stopped (forced=%s)", self._stop_forced)
            self._store.close()
            return

        self._stop_forced = True
        logger.error(
            "runtime stop: owned execution still alive after %.1fs grace "
            "(worker_alive=%s, admitted_in_flight=%s); retaining ownership "
            "in DRAINING until it exits — STOPPED is not claimed",
            grace,
            self.worker_alive,
            self.snapshot()["admission_in_use"],
        )
        self._arm_stop_reaper()

    def _owns_no_execution(self) -> bool:
        return not self.worker_alive and not self.executing

    def _arm_stop_reaper(self) -> None:
        with self._lifecycle_lock:
            if self._reaper_thread is not None:
                return
            self._reaper_thread = threading.Thread(
                target=self._reap_until_stopped, name="olh-worker-reaper", daemon=True
            )
            self._reaper_thread.start()

    def _reap_until_stopped(self) -> None:
        self._worker_exited.wait()
        while self.executing:
            time.sleep(0.05)
        with self._lifecycle_lock:
            if self._state is RuntimeState.DRAINING and self._owns_no_execution():
                self._state = RuntimeState.STOPPED
        logger.info("forced stop: last owned execution exited; runtime now STOPPED")
        self._store.close()

    def _drain_queued_jobs(self) -> None:
        """Cancel queued jobs: the worker must never execute work that was
        accepted before shutdown but never started."""
        while True:
            job_id = self._store.peek_next_queued()
            if job_id is None:
                break
            doomed: list[str] = []
            try:
                self._store.transition(
                    job_id,
                    JobStatus.CANCELLED,
                    error="server shutdown while queued",
                )
                record = self._store.get(job_id)
                if record is not None:
                    doomed.append(self.job_cleanup_dir(record))
            except (KeyError, RuntimeError):
                # Already transitioned/removed concurrently.
                pass
            cleanup_paths(doomed)

    # ---- public introspection (PR-D observability authority) ------------

    def snapshot(self) -> dict[str, Any]:
        """Atomic, public runtime counters. Job counters derive from the
        JobStore (PR-E §16); admission/reservations from the runtime."""
        try:
            counts = self._store.snapshot_counts()
        except Exception:  # noqa: BLE001 — a stopped runtime's store is closed;
            # admission/lifecycle counters remain authoritative (PR-A tests
            # introspect snapshots after stop()).
            counts = {"jobs_queued": 0, "jobs_running": 0, "jobs_tracked": 0}
        with self._admission_meta_lock:
            admission_in_use = self._admission_in_use
        with self._job_lock:
            reserved = self._slots_reserved
        return {
            "state": self._state.value,
            "accepting_requests": self.accepting_jobs,
            "worker_alive": self.worker_alive,
            "admission_capacity": self.config.max_concurrent_optimize,
            "admission_in_use": admission_in_use,
            "job_queue_capacity": self.config.job_queue_depth,
            "job_queue_size": counts["jobs_queued"],
            "job_queue_reserved": reserved,
            "jobs_running": counts["jobs_running"],
            "jobs_tracked": counts["jobs_tracked"],
            "job_backend": self.config.job_backend,
            "stop_forced": self._stop_forced,
        }

    # ---- admission control ---------------------------------------------

    def try_acquire_admission(self) -> bool:
        if self._admission.acquire(blocking=False):
            with self._admission_meta_lock:
                self._admission_in_use += 1
            return True
        return False

    def release_admission(self) -> None:
        with self._admission_meta_lock:
            if self._admission_in_use <= 0:
                raise RuntimeError("admission counter underflow: unbalanced release")
            self._admission_in_use -= 1
        self._admission.release()

    def wait_for_admission(self, poll_seconds: float = _WORKER_POLL_SECONDS) -> bool:
        while True:
            with self._lifecycle_lock:
                if self._state is not RuntimeState.RUNNING:
                    return False
                if self.try_acquire_admission():
                    return True
            time.sleep(poll_seconds)

    def run_admitted(self, **params: Any) -> dict[str, Any]:
        """Run one optimize under the admission semaphore (PR-A gates)."""
        with self._lifecycle_lock:
            if self._state is not RuntimeState.RUNNING:
                raise RuntimeNotAcceptingWorkError(
                    f"runtime is {self._state.value}; not accepting new work"
                )
            if not self.try_acquire_admission():
                raise AdmissionDeniedError("server at maximum concurrent optimizations")
        try:
            return self._optimize_runner(**params)
        finally:
            self.release_admission()

    # ---- queue slot reservations (repair-plan P0.2, PR-E durable capacity)

    @contextmanager
    def reserve_job_slot(self) -> Iterator[QueueReservation]:
        """Reserve one queue slot for the duration of the block. The slot
        is reserved BEFORE the caller ingests an upload body; any exit
        path that did not ``commit()`` releases the reservation."""
        reservation = self.try_reserve_job_slot()
        try:
            yield reservation
        finally:
            reservation._release_if_active()

    def try_reserve_job_slot(self) -> QueueReservation:
        """Reserve a slot or raise :class:`JobQueueFullError`; fail closed
        with :class:`RuntimeNotAcceptingWorkError` unless RUNNING.

        PR-E durable capacity: durable QUEUED rows count against depth, so
        ``store.count_queued() + active reservations < job_queue_depth``
        is enforced atomically under the runtime locks — persisted work
        and in-flight uploads can never overbook the queue (PR-E E4)."""
        with self._lifecycle_lock:
            if self._state is not RuntimeState.RUNNING:
                raise RuntimeNotAcceptingWorkError(
                    f"runtime is {self._state.value}; not accepting new jobs"
                )
            with self._job_lock:
                durable_queued = self._store.count_queued()
                if durable_queued + self._slots_reserved >= self.config.job_queue_depth:
                    raise JobQueueFullError(
                        f"job queue is full ({self.config.job_queue_depth} slots)"
                    )
                self._slots_reserved += 1
        return QueueReservation(self)

    def _release_reserved_slot(self) -> None:
        with self._job_lock:
            if self._slots_reserved <= 0:
                raise RuntimeError("reservation underflow: release without reservation")
            self._slots_reserved -= 1

    def _enqueue_reserved(self, record: JobRecord) -> None:
        """Create the committed QUEUED job through the store on behalf of
        an ACTIVE reservation, then wake the worker.

        Authority gate (PR-A): the RUNNING check happens under
        ``_lifecycle_lock`` — the same lock under which ``stop()``
        performs RUNNING -> DRAINING — so a commit either lands before the
        drain or is rejected once draining has begun. PR-E order: the
        store INSERT is the queue; the wake signal is an optimization, so
        a lost wake cannot lose committed work (E6)."""
        with self._lifecycle_lock:
            if self._state is not RuntimeState.RUNNING:
                raise RuntimeNotAcceptingWorkError(
                    f"runtime is {self._state.value}; not accepting new jobs"
                )
            with self._job_lock:
                if self._slots_reserved <= 0:  # defensive: broken protocol
                    raise RuntimeError(
                        "job queue capacity invariant violated: enqueue without reservation"
                    )
                self._store.create_queued(record)
                self._slots_reserved -= 1
        with self._wake:
            self._wake.notify_all()

    # ---- job store facades (public /v1 semantics, PR-D frozen) ----------

    def _get_or_raise(self, job_id: str) -> JobRecord:
        record = self._store.get(job_id)
        if record is None:
            raise UnknownJobError(job_id)
        return record

    def get_job_snapshot(self, job_id: str) -> dict[str, Any]:
        """Public snapshot of one job; reads also drive time-based
        eviction of terminal jobs (P1.5)."""
        doomed: list[str] = []
        self._evict_terminal()
        record = self._get_or_raise(job_id)
        snapshot = {
            "api_schema_version": API_SCHEMA_VERSION,
            "job_id": job_id,
            "status": JobStatus(record.status).value,
            "created_utc": record.created_utc,
            "started_utc": record.started_utc,
            "completed_utc": record.completed_utc,
            # Public OptimizeMetadata projection — private runner keys
            # (output_path, storage paths, internal ledger) never leave.
            "summary": _safe_project(record.summary),
            "error": record.error,
        }
        cleanup_paths(doomed)
        return snapshot

    def acquire_artifact_lease(self, job_id: str) -> str:
        """Open a download lease: retention must not delete the artifact
        while an HTTP response is streaming it (PR-E §14). The lease is
        registered BEFORE the retention sweep so it protects the artifact
        from the very sweep the acquire triggers."""
        record = self._get_or_raise(job_id)
        if record.status is not JobStatus.SUCCEEDED or not record.output_path:
            raise JobArtifactUnavailableError(job_id)
        if not Path(record.output_path).is_file():
            logger.error(
                "storage integrity fault: SUCCEEDED job %s artifact is missing at %s",
                job_id,
                record.output_path,
            )
            raise JobArtifactUnavailableError(job_id)
        with self._artifact_lease_lock:
            self._artifact_leases[job_id] = self._artifact_leases.get(job_id, 0) + 1
        self._evict_terminal()
        return record.output_path

    def release_artifact_lease(self, job_id: str) -> None:
        with self._artifact_lease_lock:
            count = self._artifact_leases.get(job_id, 0) - 1
            if count > 0:
                self._artifact_leases[job_id] = count
            else:
                self._artifact_leases.pop(job_id, None)

    def get_job_artifact_path(self, job_id: str) -> str:
        """Output path of a succeeded job; refreshes its TTL so janitorial
        eviction cannot delete the backing file mid-download."""
        self._store.touch_artifact(job_id)
        return self.acquire_artifact_lease(job_id)

    def delete_job(self, job_id: str) -> str:
        """Delete a terminal/queued job; returns its cleanup path for
        removal by the caller (outside locks). Cancels queued jobs so the
        worker drops them; running jobs are refused."""
        record = self._get_or_raise(job_id)
        if record.status is JobStatus.RUNNING:
            raise JobStillRunningError(job_id)
        if record.status is JobStatus.QUEUED:
            record = self._store.transition(
                job_id, JobStatus.CANCELLED, error="deleted while queued"
            )
        record = self._store.delete(job_id)
        return self.job_cleanup_dir(record)

    def _evict_terminal(self) -> list[str]:
        """TTL/history eviction on durable records (PR-E §15). Filesystem
        deletion happens outside the store; actively leased artifacts are
        skipped. A failed cleanup never resurrects the row — the files are
        retried on the next sweep."""
        with self._artifact_lease_lock:
            skip = frozenset(self._artifact_leases)
        try:
            doomed_records = self._store.evict_terminal(
                ttl_seconds=self.config.job_ttl_seconds,
                history_cap=self.config.job_history_cap,
                skip_job_ids=skip,
            )
        except Exception:  # noqa: BLE001 — eviction is janitorial
            logger.exception("job retention sweep failed; will retry on next pass")
            return []
        cleanup_dirs: list[str] = []
        for record in doomed_records:
            cleanup_dir = self.job_cleanup_dir(record)
            if cleanup_dir:
                cleanup_dirs.append(cleanup_dir)
            elif record.output_path:
                cleanup_dirs.append(record.output_path)
            elif record.input_path:
                cleanup_dirs.append(record.input_path)
        return cleanup_dirs

    # ---- the single job worker -------------------------------------------

    def _worker_loop(self) -> None:
        """Single fixed worker draining the store's durable queue.

        Flow (PR-E §7): wake-or-poll -> peek oldest QUEUED -> win
        admission -> claim (QUEUED -> RUNNING) -> execute -> terminal
        transition with atomic artifact publication. A dequeued-but-
        never-admitted job stays QUEUED and cancels legally at shutdown.
        """
        while not self._shutdown.is_set():
            if getattr(self._store, "_closed", False):
                # The runtime's store was closed (test-abandoned runtime or
                # reaper completion) — stop claiming immediately.
                break
            job_id = self._store.peek_next_queued()
            if job_id is None:
                with self._wake:
                    self._wake.wait(timeout=_WORKER_POLL_SECONDS)
                continue

            # Admission before claim: the job stays QUEUED while waiting.
            if not self.wait_for_admission():
                record = self._store.get(job_id)
                if record is not None and record.status is JobStatus.QUEUED:
                    try:
                        self._store.transition(
                            job_id,
                            JobStatus.CANCELLED,
                            error="server shutdown while waiting for admission",
                        )
                        emit_event("job_cancelled", job_id=job_id)
                        _metrics.job_terminal("cancelled")
                    except (KeyError, RuntimeError):
                        pass
                continue

            try:
                claimed = self._store.claim_next(self._instance)
            except Exception:  # noqa: BLE001 — never kill the worker loop
                logger.exception("job claim failed; retrying after poll interval")
                continue
            if claimed is None or claimed.job_id != job_id:
                # Someone else resolved it between peek and claim.
                self.release_admission()
                continue
            emit_event("job_started", job_id=claimed.job_id)

            self._execute_job(claimed)
        # Worker-exit handshake (PR-A red-team blocker A).
        self._worker_exited.set()

    def _execute_job(self, claimed: JobRecord) -> None:
        """Execute one claimed job: reconstruct finite typed params, run
        the shared spine, publish the artifact atomically, then commit the
        terminal state (PR-E §10/§11)."""
        job_id = claimed.job_id
        try:
            try:
                params = parse_params(claimed.params_json)
            except ValueError as exc:
                # Fail closed on malformed persisted parameters.
                self._fail_job(job_id, f"persisted parameters failed validation: {exc}")
                return

            # Durable output publication window (E7): the runner writes to
            # a temp path (never the final artifact name); the artifact is
            # renamed to its final durable path immediately before the
            # SUCCEEDED commit, never before. A partial/unrenamed artifact
            # can therefore never be served as successful output.
            durable = self.config.job_backend == "sqlite"
            output_final = params.get("output_path")
            params = dict(params)
            params["input_path"] = Path(str(params.get("input_path")))
            if durable and output_final:
                # Publication window (E7): the runner writes to a TEMP name
                # in the same directory that keeps the real suffix (klayout
                # infers the format from it); the artifact is renamed to
                # its final durable path only right before the SUCCEEDED
                # commit.
                final = Path(str(output_final))
                params["output_path"] = final.with_name(f"{final.stem}.tmp{final.suffix}")
            else:
                params["output_path"] = Path(str(output_final))

            try:
                summary = self._optimize_runner(**params, job_id=job_id)
            except Exception as e:  # noqa: BLE001 - surfaced via job status
                self._fail_job(job_id, str(e))
                if durable and output_final:
                    tmp = Path(str(output_final))
                    _discard(tmp.with_name(f"{tmp.stem}.tmp{tmp.suffix}"))
                emit_event("job_failed", job_id=job_id, error_type=type(e).__name__)
                _metrics.job_terminal("failed")
                return

            produced = summary.get("output_path")
            artifact_path: str | None = None
            sha: str | None = None
            size: int | None = None
            if durable and output_final:
                if not produced or not Path(produced).is_file():
                    self._fail_job(
                        job_id,
                        "runner produced no artifact at the expected temp path",
                    )
                    _metrics.job_terminal("failed")
                    emit_event("job_failed", job_id=job_id, error_type="MissingArtifact")
                    return
                final = Path(str(output_final))
                produced_path = Path(produced)
                if produced_path != final:
                    _fsync_file(produced_path)
                    os.replace(produced_path, final)  # atomic publication
                sha, size = _digest_file(final)
                artifact_path = str(final)
                summary = dict(summary)
                summary["output_path"] = artifact_path
            elif produced:
                artifact_path = str(produced)
                if Path(artifact_path).is_file():
                    sha, size = _digest_file(Path(artifact_path))

            self._store.transition(
                job_id,
                JobStatus.SUCCEEDED,
                summary=summary,
                output_path=artifact_path,
                artifact_sha256=sha,
                artifact_bytes=size,
            )
            emit_event(
                "job_succeeded",
                job_id=job_id,
                execution_mode=summary.get("execution_mode"),
                execution_reason=summary.get("execution_reason"),
                tile_count=summary.get("tiles"),
            )
            _metrics.job_terminal("succeeded")
        finally:
            self.release_admission()

    def _fail_job(self, job_id: str, error: str) -> None:
        try:
            self._store.transition(job_id, JobStatus.FAILED, error=error)
        except (KeyError, RuntimeError):
            logger.warning("could not record failure for %s: %s", job_id, error)


def _fsync_file(path: Path) -> None:
    try:
        with open(path, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _digest_file(path: Path) -> tuple[str, int]:
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _safe_project(summary: Any) -> Any:
    """Defend the public snapshot against contract-violating runner
    summaries (a runner bug is logged, never leaked and never a 500)."""
    try:
        return project_optimize_metadata(summary)
    except ValidationError:
        logger.warning(
            "job summary violated the OptimizeMetadata contract; "
            "public snapshot reports summary=null"
        )
        return None
