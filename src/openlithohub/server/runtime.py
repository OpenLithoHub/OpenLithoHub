"""App-owned server runtime — the single lifecycle owner (repair-plan P0.1,
P0.2, P1.2, P1.4).

Authority model established by PR-A:

* App CONSTRUCTION starts no threads. The FastAPI lifespan creates one
  :class:`ServerRuntime` per app instance, starts its single job worker
  on entry and terminates it on exit. ``create_app()`` is free of
  lifecycle side effects.
* Job-queue capacity is guarded by explicit transactional
  :class:`QueueReservation` objects. A reservation ends in exactly one
  terminal state — COMMITTED (its ``commit()`` enqueued the job and
  atomically consumed the reserved slot) or RELEASED — and the context
  manager releases any reservation left ACTIVE by an exception or a
  forgotten commit. There is no second, manual reserve/release
  protocol.
* Shutdown is a state machine (NEW -> RUNNING -> DRAINING -> STOPPED)
  with a strong terminal invariant:

      STOPPED  ==>  no owned worker thread is alive AND no admitted
                    execution is still in flight under this runtime.

  New submissions are rejected while draining, queued jobs are
  cancelled per documented policy, and owned work (the worker plus any
  admitted synchronous run) is awaited up to the configured grace
  period. Python threads cannot be force-killed, so work that outlives
  the grace period keeps ownership truthful: the runtime stays
  DRAINING with ``stop_forced`` set and retains its worker reference —
  it does NOT claim STOPPED. A reaper handshake (``_worker_exited``)
  completes DRAINING -> STOPPED once the last owned execution exits.
* New-work gates live at the runtime authority boundary, atomically
  ordered against the RUNNING -> DRAINING transition through
  ``_lifecycle_lock`` (lock ordering: lifecycle_lock -> job_lock /
  admission_meta_lock; never the reverse): reservation creation,
  ``QueueReservation.commit`` enqueue, and ``run_admitted`` admission
  all validate state and acquire/enqueue in one critical section, so a
  commit either lands before shutdown (and is drained/cancelled by it)
  or is rejected — never enqueued into a stopped runtime.
* State introspection goes through :meth:`ServerRuntime.snapshot` —
  public counters only; no private semaphore internals in API code.
"""

from __future__ import annotations

import itertools
import logging
import queue
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

from openlithohub.server.config import ServerConfig

logger = logging.getLogger(__name__)

# How often the idle worker wakes to notice shutdown, and how long an
# admission waiter sleeps between probes. Short enough that a graceful
# shutdown never waits seconds for an idle loop.
_WORKER_POLL_SECONDS = 0.25


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


def cleanup_scratch_dirs(doomed: list[str]) -> None:
    """Remove scratch directories, ignoring empties and errors. Callers
    run this OUTSIDE the job lock (slow rmtree must not block job state)."""
    for d in doomed:
        if d:
            shutil.rmtree(d, ignore_errors=True)


class _ReservationState(Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    RELEASED = "released"


class QueueReservation:
    """A transactional reservation of one bounded job-queue slot.

    Contract (repair-plan P0.2):

    * exactly one terminal state, reached exactly once: COMMITTED via
      :meth:`commit`, or RELEASED via :meth:`release`;
    * ``commit()`` atomically enqueues the job and consumes the
      reservation — queue-full after reservation is impossible by
      invariant (the capacity check and the counter increment happen
      under one lock, and the enqueue happens under the same lock);
    * double commit / double release raise ``RuntimeError``;
    * the reserved counter can never go negative — it is only ever
      decremented by a matching ACTIVE reservation;
    * :meth:`ServerRuntime.reserve_job_slot` releases any reservation
      left ACTIVE when its ``with`` block exits, so exceptions and
      forgotten commits cannot leak capacity.
    """

    def __init__(self, runtime: ServerRuntime) -> None:
        self._runtime = runtime
        self._lock = threading.Lock()
        self._state = _ReservationState.ACTIVE

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    def commit(self, record: dict[str, Any], params: dict[str, Any]) -> str:
        """Enqueue the job, consuming this reservation exactly once.
        Returns the new job id. On failure the reservation stays ACTIVE
        (the context manager will release it)."""
        with self._lock:
            if self._state is _ReservationState.COMMITTED:
                raise RuntimeError("queue reservation already committed")
            if self._state is _ReservationState.RELEASED:
                raise RuntimeError("queue reservation already released")
            job_id = self._runtime._enqueue_reserved(record, params)
            self._state = _ReservationState.COMMITTED
            return job_id

    def release(self) -> None:
        """Explicitly give the slot back. Raises if the reservation was
        already committed or released."""
        with self._lock:
            if self._state is _ReservationState.COMMITTED:
                raise RuntimeError("cannot release a committed queue reservation")
            if self._state is _ReservationState.RELEASED:
                raise RuntimeError("queue reservation already released")
            self._state = _ReservationState.RELEASED
        self._runtime._release_reserved_slot()

    def _release_if_active(self) -> None:
        """Context-manager exit path: release unless already terminal.
        Idempotent — never raises on double exit."""
        with self._lock:
            if self._state is not _ReservationState.ACTIVE:
                return
            self._state = _ReservationState.RELEASED
        self._runtime._release_reserved_slot()


class ServerRuntime:
    """Owns admission control, the job store/queue, and the single job
    worker thread for one app instance.

    Created and attached as ``app.state.runtime`` by the FastAPI
    lifespan. Constructing a runtime starts nothing — call :meth:`start`
    exactly once, then :meth:`stop` exactly once. A stopped runtime
    cannot be restarted; build a new one (that is what a lifespan entry
    does).
    """

    def __init__(
        self,
        config: ServerConfig,
        *,
        optimize_runner: Callable[..., dict[str, Any]],
    ) -> None:
        self.config = config
        self._optimize_runner = optimize_runner

        # Admission control: one bounded semaphore admits at most
        # max_concurrent_optimize concurrent optimizes; the in-use counter
        # mirrors it so snapshot() never needs private semaphore internals.
        self._admission = threading.BoundedSemaphore(config.max_concurrent_optimize)
        self._admission_meta_lock = threading.Lock()
        self._admission_in_use = 0

        # Job store + bounded queue + reservation counter, all guarded by
        # one lock so capacity decisions are atomic.
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._job_lock = threading.Lock()
        self._job_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(
            maxsize=config.job_queue_depth
        )
        self._slots_reserved = 0
        self._job_counter = itertools.count(1)

        # Lifecycle state machine.
        self._lifecycle_lock = threading.Lock()
        self._state = RuntimeState.NEW
        self._shutdown = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._stop_forced = False
        # Worker-exit handshake: the worker loop signals this in its
        # finally block; a reaper armed by a forced stop completes
        # DRAINING -> STOPPED only after it is set and no admitted run
        # is still in flight.
        self._worker_exited = threading.Event()
        self._reaper_thread: threading.Thread | None = None

    # ---- lifecycle ----------------------------------------------------

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def accepting_jobs(self) -> bool:
        """True only while RUNNING with a live worker. DRAINING/STOPPED
        runtimes reject new submissions (repair-plan P1.2)."""
        return self._state is RuntimeState.RUNNING and self.worker_alive

    @property
    def worker_alive(self) -> bool:
        """True while the owned job worker thread is still alive. A
        forced stop retains the worker reference, so this stays truthful
        after ``stop()`` returns."""
        thread = self._worker_thread
        return thread is not None and thread.is_alive()

    @property
    def executing(self) -> bool:
        """True while an admitted optimize (worker job or synchronous
        run) is still in flight under this runtime's admission authority."""
        with self._admission_meta_lock:
            return self._admission_in_use > 0

    def start(self) -> None:
        """Start the single job worker. Illegal from any state but NEW."""
        with self._lifecycle_lock:
            if self._state is not RuntimeState.NEW:
                raise RuntimeError(
                    f"cannot start ServerRuntime in state {self._state.value!r}; "
                    "construct a new runtime instead"
                )
            self._shutdown.clear()
            self._stop_forced = False
            self._state = RuntimeState.RUNNING
            # Truthful single-process contract (repair-plan §4 Phase 1):
            # logged at startup, surfaced via /v1/capabilities.
            logger.info(
                "server runtime starting: job_backend=%s (process-local; "
                "restart loses jobs; requires a single worker process), "
                "max_concurrent_optimize=%d, job_queue_depth=%d",
                self.config.job_backend,
                self.config.max_concurrent_optimize,
                self.config.job_queue_depth,
            )
            thread = threading.Thread(target=self._worker_loop, name="olh-job-worker", daemon=True)
            self._worker_thread = thread
            thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Graceful shutdown state machine (repair-plan P1.2).

        1. DRAINING — readiness goes false; the runtime gates (reservation
           creation, commit enqueue, sync admission) reject new work
           atomically against this transition.
        2. Stop dequeueing; queued jobs are cancelled (documented policy)
           and their scratch removed.
        3. Await owned work — the worker thread plus any admitted
           in-flight run — up to the grace period.
        4. STOPPED only when no owned worker is alive and nothing is
           still executing. If the grace period expires with work still
           running, ownership is RETAINED: state stays DRAINING,
           ``stop_forced`` records it, the worker reference is kept, and
           a reaper completes the STOPPED transition when the last owned
           execution exits. Calling ``stop()`` again while draining
           awaits the worker for its own grace budget instead of raising
           (lifespan teardown must not fail).
        """
        grace = self.config.shutdown_grace_seconds if timeout is None else timeout
        reentry = False
        with self._lifecycle_lock:
            if self._state is RuntimeState.STOPPED:
                return
            if self._state is RuntimeState.DRAINING:
                reentry = True
            else:
                self._state = RuntimeState.DRAINING
        # Past this transition, no gate can admit new work: enqueue and
        # admission validate state under _lifecycle_lock, so any commit
        # that saw RUNNING finished enqueueing strictly before the drain
        # below — it will be seen and cancelled, never stranded.
        self._shutdown.set()

        if not reentry:
            self._drain_queued_jobs()

        deadline = time.monotonic() + grace
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=grace)
        # In-flight admitted work gets the remaining grace budget.
        while self.executing and time.monotonic() < deadline:
            time.sleep(0.02)

        if self._owns_no_execution():
            with self._lifecycle_lock:
                self._state = RuntimeState.STOPPED
            logger.info("server runtime stopped (forced=%s)", self._stop_forced)
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
        """True when this runtime holds no live execution authority."""
        return not self.worker_alive and not self.executing

    def _arm_stop_reaper(self) -> None:
        """Arm the once-only reaper that finishes a forced stop."""
        with self._lifecycle_lock:
            if self._reaper_thread is not None:
                return
            self._reaper_thread = threading.Thread(
                target=self._reap_until_stopped, name="olh-worker-reaper", daemon=True
            )
            self._reaper_thread.start()

    def _reap_until_stopped(self) -> None:
        """Complete DRAINING -> STOPPED after the last owned execution
        exits (worker-exit handshake + in-flight admission drain)."""
        self._worker_exited.wait()
        while self.executing:
            time.sleep(0.05)
        with self._lifecycle_lock:
            if self._state is RuntimeState.DRAINING and self._owns_no_execution():
                self._state = RuntimeState.STOPPED
        logger.info("forced stop: last owned execution exited; runtime now STOPPED")

    def _drain_queued_jobs(self) -> None:
        """Cancel queued jobs: the worker must never execute work that
        was accepted before shutdown but never started."""
        while True:
            try:
                job_id, _params = self._job_queue.get_nowait()
            except queue.Empty:
                break
            doomed = ""
            with self._job_lock:
                record = self._jobs.get(job_id)
                if record is not None and record["status"] == "queued":
                    record["status"] = "cancelled"
                    record["error"] = "server shutdown while queued"
                    doomed = str(record.get("scratch_dir") or "")
            cleanup_scratch_dirs([doomed])
            self._job_queue.task_done()

    # ---- public introspection (repair-plan P1.4) -----------------------

    def snapshot(self) -> dict[str, Any]:
        """Atomic, public runtime counters for readiness/metrics. No
        private semaphore internals; no underscore keys."""
        with self._job_lock:
            jobs_running = sum(1 for r in self._jobs.values() if r["status"] == "running")
            queue_size = self._job_queue.qsize()
            reserved = self._slots_reserved
            jobs_tracked = len(self._jobs)
        with self._admission_meta_lock:
            admission_in_use = self._admission_in_use
        return {
            "state": self._state.value,
            "accepting_requests": self.accepting_jobs,
            "worker_alive": self.worker_alive,
            "admission_capacity": self.config.max_concurrent_optimize,
            "admission_in_use": admission_in_use,
            "job_queue_capacity": self.config.job_queue_depth,
            "job_queue_size": queue_size,
            "job_queue_reserved": reserved,
            "jobs_running": jobs_running,
            "jobs_tracked": jobs_tracked,
            "job_backend": self.config.job_backend,
            "stop_forced": self._stop_forced,
        }

    # ---- admission control ---------------------------------------------

    def try_acquire_admission(self) -> bool:
        """Non-blocking admission probe. Returns False when exhausted."""
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
        """Wait (in bounded sleeps) for admission capacity. Returns False
        once shutdown has been requested — queued jobs are patient with
        transient contention but never outlive shutdown."""
        while not self._shutdown.is_set():
            if self.try_acquire_admission():
                return True
            time.sleep(poll_seconds)
        return False

    def run_admitted(self, **params: Any) -> dict[str, Any]:
        """Run one optimize under the admission semaphore.

        Authority gate (PR-A red-team blocker B): the RUNNING check and
        the admission acquisition happen atomically under
        ``_lifecycle_lock``, totally ordered against the RUNNING ->
        DRAINING transition. Winning before shutdown makes this
        legitimate in-flight work that ``stop()`` grace-controls; losing
        means the runtime is draining/stopped and this raises
        :class:`RuntimeNotAcceptingWorkError` (HTTP 503)."""
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

    # ---- queue slot reservations (repair-plan P0.2) ---------------------

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
        with :class:`RuntimeNotAcceptingWorkError` unless RUNNING. The
        state check and the capacity check/counter increment are atomic
        (lifecycle lock ordering over the job lock), so concurrent
        reservers can never oversubscribe capacity. This early gate does
        NOT replace the commit-time recheck — a reservation may span a
        long upload across a shutdown."""
        with self._lifecycle_lock:
            if self._state is not RuntimeState.RUNNING:
                raise RuntimeNotAcceptingWorkError(
                    f"runtime is {self._state.value}; not accepting new jobs"
                )
            with self._job_lock:
                if self._job_queue.qsize() + self._slots_reserved >= self.config.job_queue_depth:
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

    def _enqueue_reserved(self, record: dict[str, Any], params: dict[str, Any]) -> str:
        """Enqueue on behalf of an ACTIVE reservation. Called at most once
        per reservation (QueueReservation.commit guards the transition).

        Authority gate (PR-A red-team blocker B): the RUNNING check and
        the enqueue happen under ``_lifecycle_lock`` — the same lock under
        which ``stop()`` performs RUNNING -> DRAINING — so the two are
        totally ordered. A commit either lands before the transition (the
        shutdown drain then sees and cancels the job) or is rejected once
        draining has begun; a job can never be enqueued into a stopped
        runtime. Lock ordering is lifecycle_lock -> job_lock everywhere;
        no path takes them in reverse."""
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
                job_id = f"job-{next(self._job_counter)}-{uuid.uuid4().hex[:8]}"
                try:
                    self._job_queue.put_nowait((job_id, params))
                except queue.Full as exc:
                    # Only reachable if capacity was stolen outside the
                    # reservation protocol. Consume nothing; the caller's
                    # reservation stays ACTIVE and releases on exit.
                    raise RuntimeError(
                        "job queue capacity invariant violated: a reserved slot failed to enqueue"
                    ) from exc
                record["job_id"] = job_id
                record["_created_monotonic"] = time.monotonic()
                self._jobs[job_id] = record
                doomed = self._evict_terminal_locked(time.monotonic())
                # The reservation is consumed: real queue capacity now owns
                # the slot. Exactly one decrement per committed reservation.
                self._slots_reserved -= 1
        cleanup_scratch_dirs(doomed)
        return job_id

    # ---- job store ------------------------------------------------------

    def get_job_snapshot(self, job_id: str) -> dict[str, Any]:
        """Public snapshot of one job; reads also drive time-based
        eviction of terminal jobs (P1.5)."""
        doomed: list[str] = []
        with self._job_lock:
            doomed = self._evict_terminal_locked(time.monotonic())
            record = self._jobs.get(job_id)
            if record is None:
                raise UnknownJobError(job_id)
            snapshot = {
                "api_schema_version": "1",
                "job_id": job_id,
                "status": record["status"],
                "created_utc": record["created_utc"],
                "summary": record["summary"],
                "error": record["error"],
            }
        cleanup_scratch_dirs(doomed)
        return snapshot

    def get_job_artifact_path(self, job_id: str) -> str:
        """Output path of a succeeded job; refreshes its TTL so janitorial
        eviction cannot delete the backing file mid-download."""
        doomed: list[str] = []
        with self._job_lock:
            doomed = self._evict_terminal_locked(time.monotonic())
            record = self._jobs.get(job_id)
            if record is None:
                raise UnknownJobError(job_id)
            if record["status"] != "succeeded" or not record.get("output_path"):
                raise JobArtifactUnavailableError(job_id)
            output_path = str(record["output_path"])
            record["_last_access_monotonic"] = time.monotonic()
        cleanup_scratch_dirs(doomed)
        return output_path

    def delete_job(self, job_id: str) -> str:
        """Delete a terminal/queued job record; returns its scratch dir
        for removal by the caller (outside the lock). Cancels queued jobs
        so the worker drops them on dequeue; running jobs are refused."""
        with self._job_lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise UnknownJobError(job_id)
            if record["status"] == "running":
                raise JobStillRunningError(job_id)
            if record["status"] == "queued":
                record["status"] = "cancelled"
            self._jobs.pop(job_id, None)
            return str(record.get("scratch_dir") or "")

    def _evict_terminal_locked(self, now: float) -> list[str]:
        """Evict terminal jobs beyond the history cap or TTL. CALLER HOLDS
        _job_lock — bookkeeping only; scratch removal happens after the
        lock is released."""
        doomed: list[str] = []
        for job_id in list(self._jobs.keys()):
            record = self._jobs[job_id]
            if record["status"] not in ("succeeded", "failed", "cancelled"):
                continue
            # TTL begins at completion (or last access for succeeded).
            reference = max(
                record.get("_completed_monotonic", 0),
                record.get("_last_access_monotonic", 0),
                record.get("_created_monotonic", 0),
            )
            age = now - reference
            if len(self._jobs) > self.config.job_history_cap or age > self.config.job_ttl_seconds:
                self._jobs.pop(job_id, None)
                doomed.append(str(record.get("scratch_dir") or ""))
        return doomed

    # ---- the single job worker -------------------------------------------

    def _worker_loop(self) -> None:
        """Single fixed worker draining the bounded job queue. Exits
        within one poll interval of shutdown."""
        while not self._shutdown.is_set():
            try:
                job_id, params = self._job_queue.get(timeout=_WORKER_POLL_SECONDS)
            except queue.Empty:
                continue
            doomed_scratch: list[str] = []
            skip = False
            with self._job_lock:
                record = self._jobs.get(job_id)
                if record is None or record["status"] in ("cancelled", "uploading"):
                    doomed_scratch.append(str(record.get("scratch_dir") or "") if record else "")
                    self._job_queue.task_done()
                    skip = True
                else:
                    record["status"] = "running"
            cleanup_scratch_dirs(doomed_scratch)
            if skip:
                continue
            # The worker HOLDS the admission slot for the whole run:
            # queued jobs wait patiently for capacity instead of racing
            # the synchronous endpoint.
            if not self.wait_for_admission():
                with self._job_lock:
                    record = self._jobs.get(job_id)
                    if record is not None:
                        record["status"] = "cancelled"
                        record["error"] = "server shutdown while waiting for admission"
                self._job_queue.task_done()
                continue
            try:
                summary = self._optimize_runner(**params)
                with self._job_lock:
                    record = self._jobs.get(job_id)
                    doomed: list[str] = []
                    if record is not None:
                        record["status"] = "succeeded"
                        record["summary"] = summary
                        record["output_path"] = str(summary["output_path"])
                        record["_completed_monotonic"] = time.monotonic()
                        record["_last_access_monotonic"] = time.monotonic()
                        doomed = self._evict_terminal_locked(time.monotonic())
                cleanup_scratch_dirs(doomed)
            except Exception as e:  # noqa: BLE001 - surfaced via job status
                failed_scratch = ""
                with self._job_lock:
                    record = self._jobs.get(job_id)
                    if record is not None:
                        record["status"] = "failed"
                        record["error"] = str(e)
                        record["_completed_monotonic"] = time.monotonic()
                        failed_scratch = str(record.get("scratch_dir") or "")
                if failed_scratch:
                    shutil.rmtree(failed_scratch, ignore_errors=True)
            finally:
                self.release_admission()
                self._job_queue.task_done()
        # Worker-exit handshake (PR-A red-team blocker A): the reaper of a
        # forced stop waits for this before completing DRAINING -> STOPPED.
        self._worker_exited.set()
