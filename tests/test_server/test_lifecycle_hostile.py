"""PR-A hostile lifecycle and reservation tests (server lifecycle authority).

Acceptance contract from the repair plan:

    App construction creates no worker.
    App lifespan owns exactly one worker.
    Shutdown terminates owned worker state cleanly.
    Queue reservation is single-owner and exception-safe.
    In-memory async jobs reject unsupported multi-process topology.

These tests deliberately poke runtime internals where no public API
exists; the ban on private introspection applies to API/production code,
not to the hostile suite that polices it.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlithohub.server import create_app  # noqa: E402
from openlithohub.server.config import ServerConfig  # noqa: E402
from openlithohub.server.runtime import (  # noqa: E402
    AdmissionDeniedError,
    JobQueueFullError,
    RuntimeNotAcceptingWorkError,
    RuntimeState,
    ServerRuntime,
    UnknownJobError,
)

WORKER_THREAD_NAME = "olh-job-worker"


def _worker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == WORKER_THREAD_NAME]


def _npy_buf() -> io.BytesIO:
    buf = io.BytesIO()
    np.save(buf, np.zeros((32, 32), dtype=np.float32))
    buf.seek(0)
    return buf


def _post_job(client: TestClient, name: str = "in.npy") -> Any:
    return client.post(
        "/v1/jobs/optimize",
        files={"layout": (name, _npy_buf(), "application/octet-stream")},
        data={"model": "dummy-identity", "node": "45nm"},
    )


def _post_sync(client: TestClient) -> Any:
    return client.post(
        "/v1/optimize",
        files={"layout": ("in.npy", _npy_buf(), "application/octet-stream")},
        data={"model": "dummy-identity", "node": "45nm"},
    )


def _job_record(scratch_dir: str = "") -> dict[str, object]:
    return {
        "status": "queued",
        "created_utc": "now",
        "scratch_dir": scratch_dir,
        "summary": None,
        "error": None,
        "output_path": None,
    }


def _bare_runtime(**overrides: object) -> ServerRuntime:
    """A runtime with a trivial runner and no HTTP machinery. Never
    started unless the test does it explicitly."""
    config = ServerConfig(**overrides)  # type: ignore[arg-type]
    return ServerRuntime(
        config, optimize_runner=lambda **params: {"output_path": str(params.get("output_path", ""))}
    )


# ---- lifecycle ownership -------------------------------------------------


def test_construction_starts_no_worker_and_owns_nothing() -> None:
    """create_app() only -> zero NEW job-worker threads, no runtime."""
    before = threading.active_count()
    workers_before = len(_worker_threads())
    app = create_app(ServerConfig())
    assert getattr(app.state, "runtime", None) is None
    assert len(_worker_threads()) == workers_before
    assert threading.active_count() == before


def test_lifespan_owns_exactly_one_worker() -> None:
    """enter lifespan -> exactly one owned worker; exit lifespan -> none."""
    before = len(_worker_threads())
    app = create_app(ServerConfig())
    assert len(_worker_threads()) == before
    with TestClient(app) as client:
        assert len(_worker_threads()) - before == 1
        runtime = client.app.state.runtime
        assert runtime is not None
        assert runtime.state is RuntimeState.RUNNING
        assert client.get("/v1/health").status_code == 200
    assert len(_worker_threads()) == before
    assert runtime.state is RuntimeState.STOPPED


def test_repeat_create_start_stop_20x_no_thread_accumulation() -> None:
    # Constant worker population across every cycle, zero at the end —
    # process-global counts stay delta-based so unrelated process state
    # cannot fail this test.
    workers_seen: int | None = None
    for _ in range(20):
        app = create_app(ServerConfig())
        with TestClient(app):
            n = len(_worker_threads())
            workers_seen = n if workers_seen is None else workers_seen
            assert n == workers_seen
    assert len(_worker_threads()) == 0


def test_independent_configs_coexist_in_one_process() -> None:
    """P1.5: multiple independent app configs in one Python process."""
    before = len(_worker_threads())
    app_a = create_app(ServerConfig(job_queue_depth=1, max_concurrent_optimize=1))
    app_b = create_app(ServerConfig(job_queue_depth=8, max_concurrent_optimize=3))
    with TestClient(app_a) as ca, TestClient(app_b) as cb:
        ra, rb = ca.app.state.runtime, cb.app.state.runtime
        assert ra is not rb
        assert ra.config.job_queue_depth == 1 and ra.config.max_concurrent_optimize == 1
        assert rb.config.job_queue_depth == 8 and rb.config.max_concurrent_optimize == 3
        assert len(_worker_threads()) - before == 2
    assert len(_worker_threads()) == before


def test_endpoints_503_before_lifespan() -> None:
    """Without the lifespan there is no runtime: work-accepting endpoints
    must 503, not silently boot one (construction != startup)."""
    app = create_app(ServerConfig())
    client = TestClient(app)  # no `with` — lifespan NOT entered
    assert client.get("/v1/ready").status_code == 503
    assert client.get("/v1/jobs/job-1-abcd").status_code == 503
    assert _post_sync(client).status_code == 503
    assert _post_job(client).status_code == 503
    # No worker may have been started by any of these requests.
    assert not any(t.is_alive() for t in _worker_threads())


def test_start_twice_raises_and_stop_is_idempotent() -> None:
    runtime = _bare_runtime()
    runtime.start()
    with pytest.raises(RuntimeError, match="cannot start"):
        runtime.start()
    runtime.stop()
    assert runtime.state is RuntimeState.STOPPED
    runtime.stop()  # idempotent second stop
    with pytest.raises(RuntimeError, match="cannot start"):
        runtime.start()  # stopped runtimes are never resurrected


def test_shutdown_rejects_new_work() -> None:
    """P1.2: after shutdown begins, submissions are rejected, not queued."""
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.stop()  # early explicit stop; lifespan's stop is idempotent
        assert runtime.state is RuntimeState.STOPPED
        assert _post_job(client).status_code == 503
        assert _post_sync(client).status_code == 503
        assert client.get("/v1/ready").status_code == 503


def test_shutdown_drains_queue_and_reports_forced_join_honestly() -> None:
    """P1.2 hostile: shutdown cancels queued jobs per documented policy;
    an in-flight job outliving the grace period is reported as stop_forced
    instead of pretending ownership ended; it still completes honestly
    once its work returns."""
    app = create_app(ServerConfig(shutdown_grace_seconds=0.5))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        release = threading.Event()
        running = threading.Event()

        def blocking_runner(**params: object) -> dict[str, object]:
            running.set()
            release.wait(timeout=10)
            return {"output_path": str(params["output_path"])}

        runtime._optimize_runner = blocking_runner

        created_a = _post_job(client, "a.npy")
        assert created_a.status_code == 202
        job_a = created_a.json()["job_id"]
        assert running.wait(timeout=5), "worker never picked up job A"

        created_b = _post_job(client, "b.npy")
        assert created_b.status_code == 202
        job_b = created_b.json()["job_id"]

        # Stop while A runs (blocked) and B sits queued.
        runtime.stop()
        # Forced: ownership is RETAINED — the runtime does NOT claim
        # STOPPED while its worker is still alive (blocker A invariant:
        # STOPPED ==> no owned worker alive).
        assert runtime.state is RuntimeState.DRAINING
        assert runtime.snapshot()["stop_forced"] is True
        assert runtime.worker_alive is True
        assert runtime.snapshot()["accepting_requests"] is False
        # B: cancelled by the documented drain policy, never executed.
        assert runtime.get_job_snapshot(job_b)["status"] == "cancelled"
        assert len(_worker_threads()) == 1

        release.set()
        # The reaper handshake completes DRAINING -> STOPPED once the
        # last owned execution truly exits.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if runtime.state is RuntimeState.STOPPED:
                break
            time.sleep(0.05)
        assert runtime.state is RuntimeState.STOPPED
        assert runtime.worker_alive is False
        assert runtime.get_job_snapshot(job_a)["status"] == "succeeded"
    assert _worker_threads() == []


def test_worker_crash_marks_job_failed() -> None:
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = client.app.state.runtime

        def exploding_runner(**params: object) -> dict[str, object]:
            raise ValueError("runner exploded")

        runtime._optimize_runner = exploding_runner
        created = _post_job(client)
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        deadline = time.monotonic() + 5
        status = None
        while time.monotonic() < deadline:
            status = runtime.get_job_snapshot(job_id)["status"]
            if status in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert status == "failed"
        assert "exploded" in str(runtime.get_job_snapshot(job_id)["error"])


# ---- public snapshot -----------------------------------------------------


def test_snapshot_is_public_and_never_goes_negative() -> None:
    app = create_app(ServerConfig(max_concurrent_optimize=2, job_queue_depth=4))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        snap = runtime.snapshot()
        expected_keys = {
            "state",
            "accepting_requests",
            "worker_alive",
            "admission_capacity",
            "admission_in_use",
            "job_queue_capacity",
            "job_queue_size",
            "job_queue_reserved",
            "jobs_running",
            "jobs_tracked",
            "job_backend",
            "stop_forced",
        }
        assert expected_keys <= set(snap)
        assert all(not k.startswith("_") for k in snap)
        assert snap["accepting_requests"] is True
        assert snap["worker_alive"] is True
        assert snap["admission_capacity"] == 2
        assert snap["admission_in_use"] == 0
        assert snap["jobs_running"] == 0

        # Admission counter tracks acquisitions and refuses underflow.
        assert runtime.try_acquire_admission() is True
        assert runtime.snapshot()["admission_in_use"] == 1
        runtime.release_admission()
        assert runtime.snapshot()["admission_in_use"] == 0
        with pytest.raises(RuntimeError, match="underflow"):
            runtime.release_admission()


def test_wait_for_admission_bails_out_on_shutdown() -> None:
    runtime = _bare_runtime(max_concurrent_optimize=1)
    runtime.start()
    assert runtime.try_acquire_admission() is True  # exhaust the one slot
    runtime.release_admission()
    runtime.stop()  # clean stop: no owned execution left
    assert runtime.wait_for_admission(poll_seconds=0.01) is False


# ---- queue reservation contract (P0.2) -----------------------------------


class TestQueueReservationContract:
    def test_commit_consumes_reservation_exactly_once(self) -> None:
        runtime = _bare_runtime(job_queue_depth=2)
        runtime.start()
        with runtime.reserve_job_slot() as res:
            assert runtime.snapshot()["job_queue_reserved"] == 1
            job_id = res.commit(_job_record(), {"x": 1})
            assert job_id
            # Consumed: the queued item now owns real capacity. The live
            # worker may drain it instantly, so assert the store, not the
            # transient queue depth.
            assert runtime.snapshot()["job_queue_reserved"] == 0
            assert runtime.snapshot()["jobs_tracked"] == 1
        # Double commit / release after commit both raise.
        with pytest.raises(RuntimeError, match="already committed"):
            res.commit(_job_record(), {})
        with pytest.raises(RuntimeError, match="committed"):
            res.release()
        runtime.stop()

    def test_exception_before_commit_releases_slot(self) -> None:
        class BoomError(Exception):
            pass

        runtime = _bare_runtime(job_queue_depth=2)
        runtime.start()
        with pytest.raises(BoomError), runtime.reserve_job_slot():
            assert runtime.snapshot()["job_queue_reserved"] == 1
            raise BoomError
        assert runtime.snapshot()["job_queue_reserved"] == 0
        assert runtime.snapshot()["job_queue_size"] == 0
        # The slot is reusable afterwards (the live worker may drain the
        # committed job instantly — assert the store, not queue depth).
        with runtime.reserve_job_slot() as res2:
            res2.commit(_job_record(), {})
        assert runtime.snapshot()["jobs_tracked"] == 1
        runtime.stop()

    def test_forgotten_commit_is_released_on_exit(self) -> None:
        runtime = _bare_runtime()
        runtime.start()
        with runtime.reserve_job_slot() as res:
            pass  # never committed
        assert runtime.snapshot()["job_queue_reserved"] == 0
        assert runtime.snapshot()["job_queue_size"] == 0
        with pytest.raises(RuntimeError, match="already released"):
            res.release()
        runtime.stop()

    def test_double_release_raises(self) -> None:
        runtime = _bare_runtime()
        runtime.start()
        res = runtime.try_reserve_job_slot()
        assert res.state == "active"
        res.release()
        assert res.state == "released"
        with pytest.raises(RuntimeError, match="already released"):
            res.release()
        assert runtime.snapshot()["job_queue_reserved"] == 0
        runtime.stop()

    def test_release_after_commit_raises(self) -> None:
        runtime = _bare_runtime(job_queue_depth=2)
        runtime.start()
        with runtime.reserve_job_slot() as res:
            res.commit(_job_record(), {})
        with pytest.raises(RuntimeError, match="committed"):
            res.release()
        runtime.stop()

    def test_capacity_never_oversubscribes_under_concurrency(self) -> None:
        depth = 4
        runtime = _bare_runtime(job_queue_depth=depth)
        runtime.start()
        acquired: list[object] = []
        rejected: list[JobQueueFullError] = []
        barrier = threading.Barrier(16)

        def reserver() -> None:
            barrier.wait()
            try:
                acquired.append(runtime.try_reserve_job_slot())
            except JobQueueFullError as exc:
                rejected.append(exc)

        threads = [threading.Thread(target=reserver) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(acquired) == depth
        assert len(rejected) == 16 - depth
        snap = runtime.snapshot()
        assert snap["job_queue_reserved"] == depth
        # The invariant: qsize + reserved <= capacity, always.
        assert snap["job_queue_size"] + snap["job_queue_reserved"] <= depth
        runtime.stop()

    def test_invariant_holds_under_concurrent_hammer(self) -> None:
        depth = 3
        runtime = _bare_runtime(job_queue_depth=depth)
        runtime.start()
        stop_flag = threading.Event()
        violations: list[tuple[int, int]] = []

        def hammer() -> None:
            while not stop_flag.is_set():
                try:
                    res = runtime.try_reserve_job_slot()
                except JobQueueFullError:
                    continue
                with runtime._job_lock:
                    q, r = runtime._job_queue.qsize(), runtime._slots_reserved
                if q + r > depth:
                    violations.append((q, r))
                # Half the threads commit, half release — deterministic.
                if threading.get_ident() % 2 == 0:
                    res.commit(_job_record(), {})
                else:
                    res.release()

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop_flag.set()
        for t in threads:
            t.join()
        assert violations == []
        snap = runtime.snapshot()
        assert snap["job_queue_reserved"] == 0
        assert snap["job_queue_size"] <= depth
        runtime.stop()

    def test_commit_after_external_capacity_theft_fails_cleanly(self) -> None:
        """Even if the invariant is violated from outside the protocol,
        commit must fail without leaking the reservation. The worker is
        occupied first so it cannot drain the hostile puts."""
        started = threading.Event()
        release = threading.Event()

        def blocking_runner(**params: object) -> dict[str, object]:
            started.set()
            release.wait(timeout=10)
            return {"output_path": ""}

        runtime = ServerRuntime(
            ServerConfig(job_queue_depth=2, shutdown_grace_seconds=5.0),
            optimize_runner=blocking_runner,
        )
        runtime.start()
        res = None
        try:
            # Occupy the single worker so it cannot drain the hostile items.
            with runtime.reserve_job_slot() as first:
                first.commit(_job_record(), {})
            assert started.wait(timeout=5)
            res = runtime.try_reserve_job_slot()
            # Hostile direct puts exceed capacity behind the worker's back.
            runtime._job_queue.put(("external-1", {}))
            runtime._job_queue.put(("external-2", {}))
            with pytest.raises(RuntimeError, match="invariant"):
                res.commit(_job_record(), {})
            # The reservation stays ACTIVE and still releases cleanly.
            assert res.state == "active"
            assert runtime.snapshot()["job_queue_reserved"] == 1
        finally:
            if res is not None:
                res.release()
            release.set()
        assert runtime.snapshot()["job_queue_reserved"] == 0
        runtime.stop()  # drains the hostile items (unknown ids are dropped)
        assert runtime.snapshot()["job_queue_reserved"] == 0

    def test_unknown_job_raises_public_error(self) -> None:
        runtime = _bare_runtime()
        with pytest.raises(UnknownJobError):
            runtime.get_job_snapshot("nope")
        with pytest.raises(UnknownJobError):
            runtime.get_job_artifact_path("nope")
        with pytest.raises(UnknownJobError):
            runtime.delete_job("nope")


# ---- blocker A: forced-stop ownership exclusion --------------------------


def test_forced_timeout_excludes_second_execution_authority() -> None:
    """Deterministic ownership-exclusion test: while a forced stop leaves
    worker A alive (holding the single admission slot), no second
    lifespan may start a second execution authority in this process.
    Only after worker A exits and runtime A reaches STOPPED may a new
    lifespan start."""
    app = create_app(ServerConfig(max_concurrent_optimize=1, shutdown_grace_seconds=0.3))
    release = threading.Event()
    running = threading.Event()

    def blocking_runner(**params: object) -> dict[str, object]:
        running.set()
        release.wait(timeout=10)
        return {"output_path": str(params["output_path"])}

    with TestClient(app) as client:
        runtime_a = client.app.state.runtime
        assert runtime_a is not None
        workers_with_a = len(_worker_threads())
        runtime_a._optimize_runner = blocking_runner
        created = _post_job(client)
        assert created.status_code == 202
        assert running.wait(timeout=5), "worker never picked up the job"

        runtime_a.stop()  # grace 0.3s < blocked job -> forced
        assert runtime_a.state is RuntimeState.DRAINING
        assert runtime_a.snapshot()["stop_forced"] is True
        assert runtime_a.worker_alive is True
        # The detached worker still owns the ONLY admission slot.
        assert runtime_a.snapshot()["admission_in_use"] == 1
        # Forced stop retains the runtime reference on the app state so a
        # future lifespan can see the live ownership.
        assert app.state.runtime is runtime_a
        # Runtime A's worker is still alive (delta-based count).
        assert len(_worker_threads()) == workers_with_a

    # A second lifespan must be refused while worker A is alive.
    workers_before_refusal = len(_worker_threads())
    with pytest.raises(RuntimeError, match="second execution authority"), TestClient(app):
        pass
    assert len(_worker_threads()) == workers_before_refusal, "no second worker may exist"
    assert app.state.runtime is runtime_a, "ownership not stolen"

    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if runtime_a.state is RuntimeState.STOPPED:
            break
        time.sleep(0.05)
    assert runtime_a.state is RuntimeState.STOPPED
    assert runtime_a.worker_alive is False
    assert len(_worker_threads()) == workers_with_a - 1

    # Only now may the next lifespan start.
    with TestClient(app) as client2:
        runtime_b = client2.app.state.runtime
        assert runtime_b is not runtime_a
        assert runtime_b.state is RuntimeState.RUNNING
        assert len(_worker_threads()) == workers_with_a
    assert len(_worker_threads()) == workers_with_a - 1


# ---- blocker B: shutdown vs commit / admission ordering ------------------


def test_reserve_fails_closed_after_stop() -> None:
    runtime = _bare_runtime()
    runtime.start()
    runtime.stop()
    with pytest.raises(RuntimeNotAcceptingWorkError):
        runtime.try_reserve_job_slot()


def test_commit_before_shutdown_is_cancelled_not_lost() -> None:
    """Legal interleaving 1: commit wins, shutdown drains the queue and
    cancels the queued job — never strands it. The worker is occupied
    first so the queued job cannot complete before shutdown, and the
    forced-stop path is exercised end to end."""
    started = threading.Event()
    release = threading.Event()

    def blocking_runner(**params: object) -> dict[str, object]:
        started.set()
        release.wait(timeout=10)
        return {"output_path": ""}

    runtime = ServerRuntime(
        ServerConfig(job_queue_depth=2, shutdown_grace_seconds=0.4),
        optimize_runner=blocking_runner,
    )
    runtime.start()
    with runtime.reserve_job_slot() as res1:
        job_running = res1.commit(_job_record(), {})
    assert started.wait(timeout=5), "worker never picked up the first job"
    with runtime.reserve_job_slot() as res2:
        job_queued = res2.commit(_job_record(), {})

    runtime.stop()  # grace expires with the blocked job -> forced
    # The queued job: seen by the drain and cancelled, never executed.
    assert runtime.get_job_snapshot(job_queued)["status"] == "cancelled"
    assert runtime.state is RuntimeState.DRAINING
    snap = runtime.snapshot()
    assert snap["job_queue_size"] == 0
    assert snap["job_queue_reserved"] == 0

    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if runtime.state is RuntimeState.STOPPED:
            break
        time.sleep(0.05)
    assert runtime.state is RuntimeState.STOPPED
    assert runtime.get_job_snapshot(job_running)["status"] == "succeeded"


def test_reservation_commit_after_complete_stop_rejected() -> None:
    """Legal interleaving 2: shutdown wins, the pending reservation's
    commit is rejected at the runtime authority gate — no job enters the
    queue, the reservation releases exactly once."""
    runtime = _bare_runtime(job_queue_depth=2)
    runtime.start()
    res = runtime.try_reserve_job_slot()
    runtime.stop()
    assert runtime.state is RuntimeState.STOPPED
    with pytest.raises(RuntimeNotAcceptingWorkError):
        res.commit(_job_record(), {})
    assert runtime.snapshot()["job_queue_size"] == 0
    # The failed commit leaves the reservation ACTIVE; it releases once.
    res.release()
    assert runtime.snapshot()["job_queue_reserved"] == 0
    with pytest.raises(RuntimeError, match="already released"):
        res.release()


def test_commit_race_with_shutdown_never_strands_a_job() -> None:
    """Concurrent interleaving hammer: commits racing a shutdown may only
    ever (a) land before the drain and be cancelled/executed, or (b) be
    rejected with RuntimeNotAcceptingWorkError. STOPPED + queued job is
    forbidden, and capacity must not leak."""
    runtime = _bare_runtime(job_queue_depth=8, shutdown_grace_seconds=5.0)
    runtime.start()
    stop_now = threading.Event()
    committed: list[str] = []

    def committer() -> None:
        while not stop_now.is_set():
            try:
                with runtime.reserve_job_slot() as res:
                    committed.append(res.commit(_job_record(), {}))
            except RuntimeNotAcceptingWorkError:
                return
            except JobQueueFullError:
                continue

    threads = [threading.Thread(target=committer) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    runtime.stop()
    stop_now.set()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert runtime.state is RuntimeState.STOPPED
    snap = runtime.snapshot()
    assert snap["job_queue_size"] == 0, "no job may remain queued after STOPPED"
    assert snap["job_queue_reserved"] == 0
    with runtime._job_lock:
        statuses = [r["status"] for r in runtime._jobs.values()]
    # History-cap eviction may have removed terminal records; whatever is
    # still tracked must be terminal, nothing may remain queued, and the
    # committer must have gotten real work through before shutdown.
    assert len(committed) > 0
    assert len(statuses) <= len(committed)
    assert all(s in ("cancelled", "succeeded", "failed") for s in statuses)


def test_run_admitted_after_stop_rejected() -> None:
    runtime = ServerRuntime(
        ServerConfig(max_concurrent_optimize=1),
        optimize_runner=lambda **params: {"output_path": ""},
    )
    runtime.start()
    runtime.stop()
    calls: list[int] = []

    def counting_runner(**params: object) -> dict[str, object]:
        calls.append(1)
        return {"output_path": ""}

    runtime2 = ServerRuntime(runtime.config, optimize_runner=counting_runner)
    runtime2.start()
    runtime2.stop()
    with pytest.raises(RuntimeNotAcceptingWorkError):
        runtime2.run_admitted(k=1)
    assert calls == []
    assert runtime2.snapshot()["admission_in_use"] == 0


def test_sync_admission_wins_before_drain_is_grace_controlled() -> None:
    """Legal interleaving for the sync path: run_admitted that wins before
    DRAINING is legitimate in-flight work — shutdown grace-controls it,
    keeps ownership (DRAINING + stop_forced), and only reaches STOPPED
    once the run truly finishes."""
    release = threading.Event()
    started = threading.Event()

    def blocking_runner(**params: object) -> dict[str, object]:
        started.set()
        release.wait(timeout=10)
        return {"output_path": ""}

    runtime = ServerRuntime(
        ServerConfig(max_concurrent_optimize=1, shutdown_grace_seconds=0.4),
        optimize_runner=blocking_runner,
    )
    runtime.start()
    result: dict[str, object] = {}

    def work() -> None:
        result["summary"] = runtime.run_admitted(k=1)

    worker = threading.Thread(target=work)
    worker.start()
    assert started.wait(timeout=5)
    assert runtime.snapshot()["admission_in_use"] == 1

    runtime.stop()
    # The sync run outlived the 0.4s grace: ownership retained, honest state.
    assert runtime.state is RuntimeState.DRAINING
    assert runtime.snapshot()["stop_forced"] is True
    assert runtime.snapshot()["admission_in_use"] == 1

    release.set()
    worker.join(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if runtime.state is RuntimeState.STOPPED:
            break
        time.sleep(0.05)
    assert runtime.state is RuntimeState.STOPPED
    assert runtime.snapshot()["admission_in_use"] == 0
    assert "summary" in result


def test_sync_admission_race_with_shutdown_never_leaks() -> None:
    """Concurrent sync-admission hammer vs stop(): every run either
    completes (started before DRAINING) or is rejected; admission never
    leaks and no execution can start after shutdown won."""
    runtime = ServerRuntime(
        ServerConfig(max_concurrent_optimize=2, shutdown_grace_seconds=5.0),
        optimize_runner=lambda **params: {"output_path": ""},
    )
    runtime.start()
    stop_now = threading.Event()
    rejected: list[RuntimeNotAcceptingWorkError] = []

    def hammer() -> None:
        while not stop_now.is_set():
            try:
                runtime.run_admitted(k=1)
            except RuntimeNotAcceptingWorkError as exc:
                rejected.append(exc)
                return
            except AdmissionDeniedError:  # noqa: BLE001 - capacity retry
                continue

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    runtime.stop()
    stop_now.set()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert runtime.state is RuntimeState.STOPPED
    assert runtime.snapshot()["admission_in_use"] == 0
    assert len(rejected) >= 1, "post-shutdown attempts must be rejected"


# ---- tail A: worker admission vs shutdown ordering -----------------------


def test_worker_admission_never_starts_after_draining_wins() -> None:
    """Deterministic gate: a dequeued job parked in wait_for_admission
    must be cancelled — never executed — once DRAINING has won, even
    though it began waiting while the shutdown event was still clear.
    The lifecycle state under the lock is the sole admission authority.

    PR-D job-state semantics: a dequeued-but-never-admitted job stays
    QUEUED (RUNNING begins only at admitted execution), so shutdown
    cancels it via the legal QUEUED -> CANCELLED edge."""
    runner_calls: list[int] = []

    def counting_runner(**params: object) -> dict[str, object]:
        runner_calls.append(1)
        return {"output_path": ""}

    runtime = ServerRuntime(
        ServerConfig(max_concurrent_optimize=1, shutdown_grace_seconds=5.0),
        optimize_runner=counting_runner,
    )
    runtime.start()
    # Execution A occupies the single admission slot (not via the runner).
    assert runtime.try_acquire_admission() is True
    # Job B: committed, dequeued by the worker, then parked waiting for
    # admission (PR-D: it stays QUEUED until admission is won).
    with runtime.reserve_job_slot() as res:
        job_b = res.commit(_job_record(), {})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snap = runtime.get_job_snapshot(job_b)
        if snap["status"] == "queued" and snap["started_utc"] is None:
            break
        time.sleep(0.02)
    assert runtime.get_job_snapshot(job_b)["status"] == "queued", (
        "worker transitioned job B out of QUEUED before winning admission"
    )

    # Shutdown wins while B is waiting for admission.
    runtime.stop(timeout=0.3)  # slot A still held -> forced, ownership retained

    # B must have been cancelled by the admission gate, never executed.
    assert runner_calls == []
    assert runtime.get_job_snapshot(job_b)["status"] == "cancelled"
    assert "shutdown" in str(runtime.get_job_snapshot(job_b)["error"])

    # Releasing A lets the reaper finish; nothing leaks.
    runtime.release_admission()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if runtime.state is RuntimeState.STOPPED:
            break
        time.sleep(0.02)
    assert runtime.state is RuntimeState.STOPPED
    assert runtime.snapshot()["admission_in_use"] == 0
    assert runner_calls == []


def test_worker_admission_gate_hammer_vs_shutdown() -> None:
    """Race hammer: admission waiters racing a shutdown may only win
    before DRAINING; after stop() returns, every waiter is refused and
    admission never leaks."""
    runtime = ServerRuntime(
        ServerConfig(max_concurrent_optimize=2, shutdown_grace_seconds=5.0),
        optimize_runner=lambda **params: {"output_path": ""},
    )
    runtime.start()
    stop_now = threading.Event()
    admitted_count = 0
    count_lock = threading.Lock()

    def hammer() -> None:
        nonlocal admitted_count
        while not stop_now.is_set():
            if runtime.wait_for_admission(poll_seconds=0.001):
                with count_lock:
                    admitted_count += 1
                runtime.release_admission()

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    runtime.stop()  # clean: hammer admissions drain once DRAINING blocks new ones
    stop_now.set()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert runtime.state is RuntimeState.STOPPED
    assert runtime.snapshot()["admission_in_use"] == 0
    assert admitted_count > 0, "hammer never exercised the admission path"
    # The admission authority stays closed for good.
    assert runtime.wait_for_admission(poll_seconds=0.001) is False
