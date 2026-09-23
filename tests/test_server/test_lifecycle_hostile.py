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
    JobQueueFullError,
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
    """create_app() only -> zero job-worker threads, no runtime."""
    before = threading.active_count()
    app = create_app(ServerConfig())
    assert getattr(app.state, "runtime", None) is None
    assert _worker_threads() == []
    assert threading.active_count() == before


def test_lifespan_owns_exactly_one_worker() -> None:
    """enter lifespan -> exactly one worker; exit lifespan -> zero."""
    app = create_app(ServerConfig())
    assert _worker_threads() == []
    with TestClient(app) as client:
        assert len(_worker_threads()) == 1
        runtime = client.app.state.runtime
        assert runtime is not None
        assert runtime.state is RuntimeState.RUNNING
        assert client.get("/v1/health").status_code == 200
    assert _worker_threads() == []
    assert runtime.state is RuntimeState.STOPPED


def test_repeat_create_start_stop_20x_no_thread_accumulation() -> None:
    baseline = threading.active_count()
    for _ in range(20):
        app = create_app(ServerConfig())
        with TestClient(app):
            assert len(_worker_threads()) == 1
    assert _worker_threads() == []
    assert threading.active_count() <= baseline + 2


def test_independent_configs_coexist_in_one_process() -> None:
    """P1.5: multiple independent app configs in one Python process."""
    app_a = create_app(ServerConfig(job_queue_depth=1, max_concurrent_optimize=1))
    app_b = create_app(ServerConfig(job_queue_depth=8, max_concurrent_optimize=3))
    with TestClient(app_a) as ca, TestClient(app_b) as cb:
        ra, rb = ca.app.state.runtime, cb.app.state.runtime
        assert ra is not rb
        assert ra.config.job_queue_depth == 1 and ra.config.max_concurrent_optimize == 1
        assert rb.config.job_queue_depth == 8 and rb.config.max_concurrent_optimize == 3
        assert len(_worker_threads()) == 2
    assert _worker_threads() == []


def test_endpoints_503_before_lifespan() -> None:
    """Without the lifespan there is no runtime: work-accepting endpoints
    must 503, not silently boot one (construction != startup)."""
    app = create_app(ServerConfig())
    client = TestClient(app)  # no `with` — lifespan NOT entered
    assert client.get("/v1/ready").status_code == 503
    assert client.get("/v1/jobs/job-1-abcd").status_code == 503
    assert _post_sync(client).status_code == 503
    assert _post_job(client).status_code == 503
    assert _worker_threads() == []


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
        assert runtime.state is RuntimeState.STOPPED
        # B: cancelled by the documented drain policy, never executed.
        assert runtime.get_job_snapshot(job_b)["status"] == "cancelled"
        # A outlived the 0.5s grace: the runtime reports the forced stop
        # honestly while the worker thread is still alive.
        assert runtime.snapshot()["stop_forced"] is True
        assert len(_worker_threads()) == 1

        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if runtime.get_job_snapshot(job_a)["status"] == "succeeded":
                break
            time.sleep(0.05)
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
    assert runtime.try_acquire_admission() is True  # exhaust the one slot
    runtime.stop()  # NEW -> STOPPED, sets the shutdown event
    assert runtime.wait_for_admission(poll_seconds=0.01) is False


# ---- queue reservation contract (P0.2) -----------------------------------


class TestQueueReservationContract:
    def test_commit_consumes_reservation_exactly_once(self) -> None:
        runtime = _bare_runtime(job_queue_depth=2)
        with runtime.reserve_job_slot() as res:
            assert runtime.snapshot()["job_queue_reserved"] == 1
            job_id = res.commit(_job_record(), {"x": 1})
            assert job_id
            # Consumed: the queued item now owns real capacity.
            assert runtime.snapshot()["job_queue_reserved"] == 0
            assert runtime.snapshot()["job_queue_size"] == 1
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
        with pytest.raises(BoomError), runtime.reserve_job_slot():
            assert runtime.snapshot()["job_queue_reserved"] == 1
            raise BoomError
        assert runtime.snapshot()["job_queue_reserved"] == 0
        assert runtime.snapshot()["job_queue_size"] == 0
        # The slot is reusable afterwards.
        with runtime.reserve_job_slot() as res2:
            res2.commit(_job_record(), {})
        assert runtime.snapshot()["job_queue_size"] == 1
        runtime.stop()

    def test_forgotten_commit_is_released_on_exit(self) -> None:
        runtime = _bare_runtime()
        with runtime.reserve_job_slot() as res:
            pass  # never committed
        assert runtime.snapshot()["job_queue_reserved"] == 0
        assert runtime.snapshot()["job_queue_size"] == 0
        with pytest.raises(RuntimeError, match="already released"):
            res.release()

    def test_double_release_raises(self) -> None:
        runtime = _bare_runtime()
        res = runtime.try_reserve_job_slot()
        assert res.state == "active"
        res.release()
        assert res.state == "released"
        with pytest.raises(RuntimeError, match="already released"):
            res.release()
        assert runtime.snapshot()["job_queue_reserved"] == 0

    def test_release_after_commit_raises(self) -> None:
        runtime = _bare_runtime(job_queue_depth=2)
        with runtime.reserve_job_slot() as res:
            res.commit(_job_record(), {})
        with pytest.raises(RuntimeError, match="committed"):
            res.release()
        runtime.stop()

    def test_capacity_never_oversubscribes_under_concurrency(self) -> None:
        depth = 4
        runtime = _bare_runtime(job_queue_depth=depth)
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

    def test_invariant_holds_under_concurrent_hammer(self) -> None:
        depth = 3
        runtime = _bare_runtime(job_queue_depth=depth)
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
        commit must fail without leaking the reservation."""
        runtime = _bare_runtime(job_queue_depth=1)
        res = runtime.try_reserve_job_slot()
        try:
            runtime._job_queue.put(("external", {}))  # hostile direct put
            with pytest.raises(RuntimeError, match="invariant"):
                res.commit(_job_record(), {})
            # The reservation stays ACTIVE and still releases cleanly.
            assert res.state == "active"
            assert runtime.snapshot()["job_queue_reserved"] == 1
        finally:
            res.release()
            runtime._job_queue.get_nowait()  # remove the hostile item
        assert runtime.snapshot()["job_queue_reserved"] == 0

    def test_unknown_job_raises_public_error(self) -> None:
        runtime = _bare_runtime()
        with pytest.raises(UnknownJobError):
            runtime.get_job_snapshot("nope")
        with pytest.raises(UnknownJobError):
            runtime.get_job_artifact_path("nope")
        with pytest.raises(UnknownJobError):
            runtime.delete_job("nope")
