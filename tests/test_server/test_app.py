"""Tests for the FastAPI HTTP engine."""

from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlithohub.server import create_app  # noqa: E402
from openlithohub.server.config import ServerConfig  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    # The lifespan owns the runtime: entering the client starts the job
    # worker, exiting it stops the worker (PR-A lifecycle authority).
    with TestClient(create_app(ServerConfig())) as c:
        yield c


def test_health(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_includes_dummy(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert "dummy-identity" in models


def test_optimize_with_npy_layout_round_trips(client: TestClient, tmp_path) -> None:
    layout = np.zeros((64, 64), dtype=np.float32)
    layout[16:48, 16:48] = 1.0
    layout_path = tmp_path / "input.npy"
    np.save(layout_path, layout)

    with layout_path.open("rb") as fh:
        response = client.post(
            "/v1/optimize",
            files={"layout": ("input.npy", fh, "application/octet-stream")},
            data={
                "model": "dummy-identity",
                "node": "3nm-euv",
                "pixel_nm": "1.0",
                "tile_size": "64",
                "writer": "vsb",
            },
        )

    assert response.status_code == 200, response.text
    assert "X-OLH-Tiles" in response.headers
    assert response.headers["X-OLH-Shape"] == "64x64"
    assert len(response.content) > 0


def test_optimize_with_pt_layout_falls_back_to_torch_when_klayout_missing(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    layout = torch.zeros((32, 32), dtype=torch.float32)
    layout[8:24, 8:24] = 1.0
    layout_path = tmp_path / "input.pt"
    torch.save(layout, str(layout_path))

    import openlithohub.workflow.export as export_mod

    def _raise(*_args, **_kwargs):
        raise ImportError("klayout not available in test env")

    monkeypatch.setattr(export_mod, "export_oasis", _raise)

    with layout_path.open("rb") as fh:
        response = client.post(
            "/v1/optimize",
            files={"layout": ("input.pt", fh, "application/octet-stream")},
            data={
                "model": "dummy-identity",
                "tile_size": "32",
                "node": "3nm-euv",
                "pixel_nm": "1.0",
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers["X-OLH-Export-Format"] == "torch"
    out_buf = io.BytesIO(response.content)
    restored = torch.load(out_buf, weights_only=True)
    assert restored.shape == (32, 32)


def test_optimize_unknown_model_returns_404(client: TestClient, tmp_path) -> None:
    layout_path = tmp_path / "input.npy"
    np.save(layout_path, np.zeros((16, 16), dtype=np.float32))

    with layout_path.open("rb") as fh:
        response = client.post(
            "/v1/optimize",
            files={"layout": ("input.npy", fh, "application/octet-stream")},
            data={"model": "not-a-real-model", "tile_size": "16"},
        )

    assert response.status_code == 404


def test_get_or_load_model_returns_per_key_lock() -> None:
    """Issue #37 regression: cached model and its serialisation lock must
    be returned together so concurrent requests can serialise predict()."""
    import threading

    from openlithohub.server.app import _MODEL_CACHE, _MODEL_LOCKS, _get_or_load_model

    saved_cache = dict(_MODEL_CACHE)
    saved_locks = dict(_MODEL_LOCKS)
    _MODEL_CACHE.clear()
    _MODEL_LOCKS.clear()
    try:
        model_a, lock_a, _key_a = _get_or_load_model("dummy-identity", {})
        model_b, lock_b, _key_b = _get_or_load_model("dummy-identity", {})
        assert model_a is model_b
        assert lock_a is lock_b
        assert isinstance(lock_a, type(threading.Lock()))
    finally:
        _MODEL_CACHE.clear()
        _MODEL_LOCKS.clear()
        _MODEL_CACHE.update(saved_cache)
        _MODEL_LOCKS.update(saved_locks)


def test_get_or_load_model_concurrent_requests_load_once() -> None:
    """Two threads asking for the same model must share one instance —
    no double-load, no race that produces two distinct models."""
    import threading

    from openlithohub.server.app import _MODEL_CACHE, _MODEL_LOCKS, _get_or_load_model

    saved_cache = dict(_MODEL_CACHE)
    saved_locks = dict(_MODEL_LOCKS)
    _MODEL_CACHE.clear()
    _MODEL_LOCKS.clear()
    try:
        results: list[tuple[object, object, object]] = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait()
            results.append(_get_or_load_model("dummy-identity", {}))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(m) for m, _, _ in results}) == 1
        assert len({id(lock) for _, lock, _ in results}) == 1
    finally:
        _MODEL_CACHE.clear()
        _MODEL_LOCKS.clear()
        _MODEL_CACHE.update(saved_cache)
        _MODEL_LOCKS.update(saved_locks)


def test_version(client: TestClient) -> None:
    response = client.get("/v1/version")
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "v1"
    assert body["package"] == "openlithohub"
    assert body["version"]
    assert isinstance(body["git_commit"], str) and body["git_commit"]
    assert body["torch"]


def test_capabilities(client: TestClient) -> None:
    response = client.get("/v1/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert "dummy-identity" in body["models"]
    assert "hopkins" in body["simulator_backends"]
    assert isinstance(body["gpu"]["available"], bool)
    assert body["streaming_pipeline"] is True


def test_eviction_parks_inflight_model_instead_of_teardown(monkeypatch) -> None:
    """P1.1 hostile: a model acquired (refcounted) before a concurrent
    insert evicts it must be PARKED, never torn down while in flight."""

    from openlithohub.server import app as app_mod

    class _Fake:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.torn_down = False

        def setup(self) -> None:
            pass

        def teardown(self) -> None:
            self.torn_down = True

    saved = (
        dict(app_mod._MODEL_CACHE),
        dict(app_mod._MODEL_LOCKS),
        dict(app_mod._MODEL_REFCOUNTS),
        dict(app_mod._PENDING_TEARDOWN),
        app_mod._MODEL_CACHE_CAP,
    )
    app_mod._MODEL_CACHE.clear()
    app_mod._MODEL_LOCKS.clear()
    app_mod._MODEL_REFCOUNTS.clear()
    app_mod._PENDING_TEARDOWN.clear()
    monkeypatch.setattr(app_mod, "_MODEL_CACHE_CAP", 1)
    made: list[_Fake] = []

    def fake_get(name: str, **kwargs: object) -> _Fake:
        m = _Fake(f"{name}-{kwargs.get('slot', 0)}")
        made.append(m)
        return m

    monkeypatch.setattr(app_mod, "_MODEL_CACHE_CAP", 1)
    import openlithohub.models.registry as registry_mod

    monkeypatch.setattr(registry_mod, "register_builtin_models", lambda: None)

    class _Reg:
        @staticmethod
        def get(name: str, **kwargs: object) -> _Fake:
            return fake_get(name, **kwargs)

    monkeypatch.setattr(app_mod, "registry", _Reg, raising=False)
    # _get_or_load_model imports registry locally; patch the registry module attr
    import sys

    real_registry = registry_mod.registry
    monkeypatch.setattr(registry_mod, "registry", _Reg)
    try:
        # Acquire A and hold the reference (in flight).
        model_a, lock_a, key_a = app_mod._get_or_load_model("dummy-identity", {"slot": "a"})
        # Another load forces eviction of A while A is still in flight.
        model_b, lock_b, key_b = app_mod._get_or_load_model("dummy-identity", {"slot": "b"})
        assert model_a.torn_down is False, "in-flight model was torn down (P1.1 race)"
        assert key_a in app_mod._PENDING_TEARDOWN, "in-flight model must be parked"
        # Release A: only now may the deferred teardown run.
        app_mod._release_model(key_a)
        assert model_a.torn_down is True
        assert key_a not in app_mod._PENDING_TEARDOWN
        # B was never evicted: releasing it leaves it resident in cache.
        app_mod._release_model(key_b)
        assert model_b.torn_down is False
        assert key_b in app_mod._MODEL_CACHE
    finally:
        (
            app_mod._MODEL_CACHE.clear(),
            app_mod._MODEL_LOCKS.clear(),
            app_mod._MODEL_REFCOUNTS.clear(),
            app_mod._PENDING_TEARDOWN.clear(),
        )
        (
            app_mod._MODEL_CACHE.update(saved[0]),
            app_mod._MODEL_LOCKS.update(saved[1]),
            app_mod._MODEL_REFCOUNTS.update(saved[2]),
            app_mod._PENDING_TEARDOWN.update(saved[3]),
        )
        app_mod._MODEL_CACHE_CAP = saved[4]
        monkeypatch.setattr(registry_mod, "registry", real_registry)
        assert sys.modules  # no-op keeps imports referenced


def test_sidecar_state_bounded_under_churn(monkeypatch) -> None:
    """P1.3: distinct-key churn must not leak lock/refcount sidecar state."""
    from openlithohub.server import app as app_mod

    class _Fake:
        def setup(self) -> None:
            pass

        def teardown(self) -> None:
            pass

    saved = (
        dict(app_mod._MODEL_CACHE),
        dict(app_mod._MODEL_LOCKS),
        dict(app_mod._MODEL_REFCOUNTS),
        dict(app_mod._PENDING_TEARDOWN),
        app_mod._MODEL_CACHE_CAP,
    )
    app_mod._MODEL_CACHE.clear()
    app_mod._MODEL_LOCKS.clear()
    app_mod._MODEL_REFCOUNTS.clear()
    app_mod._PENDING_TEARDOWN.clear()
    monkeypatch.setattr(app_mod, "_MODEL_CACHE_CAP", 2)
    import openlithohub.models.registry as registry_mod

    monkeypatch.setattr(registry_mod, "register_builtin_models", lambda: None)

    class _Reg:
        @staticmethod
        def get(name: str, **kwargs: object) -> _Fake:
            return _Fake()

    real_registry = registry_mod.registry
    monkeypatch.setattr(registry_mod, "registry", _Reg)
    try:
        for i in range(200):
            _, _, key = app_mod._get_or_load_model("dummy-identity", {"slot": i})
            app_mod._release_model(key)
        assert len(app_mod._MODEL_CACHE) <= 2
        assert len(app_mod._MODEL_LOCKS) <= 2
        assert len(app_mod._MODEL_REFCOUNTS) == 0
        assert len(app_mod._PENDING_TEARDOWN) == 0
    finally:
        (
            app_mod._MODEL_CACHE.clear(),
            app_mod._MODEL_LOCKS.clear(),
            app_mod._MODEL_REFCOUNTS.clear(),
            app_mod._PENDING_TEARDOWN.clear(),
        )
        (
            app_mod._MODEL_CACHE.update(saved[0]),
            app_mod._MODEL_LOCKS.update(saved[1]),
            app_mod._MODEL_REFCOUNTS.update(saved[2]),
            app_mod._PENDING_TEARDOWN.update(saved[3]),
        )
        app_mod._MODEL_CACHE_CAP = saved[4]
        monkeypatch.setattr(registry_mod, "registry", real_registry)


def test_job_optimize_lifecycle() -> None:
    """POST /v1/jobs/optimize → poll → download artifact (P1.15)."""
    import time as time_mod

    client_cm = TestClient(create_app(ServerConfig()))
    with client_cm as client:
        layout = np.zeros((64, 64), dtype=np.float32)
        layout[16:48, 16:48] = 1.0
        buf = io.BytesIO()
        np.save(buf, layout)
        buf.seek(0)
        created = client.post(
            "/v1/jobs/optimize",
            files={"layout": ("in.npy", buf, "application/octet-stream")},
            data={"model": "dummy-identity", "node": "45nm"},
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["job_id"]
        for _ in range(100):
            status = client.get(f"/v1/jobs/{job_id}").json()
            if status["status"] in ("succeeded", "failed"):
                break
            time_mod.sleep(0.1)
        assert status["status"] == "succeeded", status
        artifact = client.get(f"/v1/jobs/{job_id}/artifact")
        assert artifact.status_code == 200
        assert len(artifact.content) > 0
        deleted = client.delete(f"/v1/jobs/{job_id}")
        assert deleted.status_code == 200
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_ready_endpoint(client: TestClient) -> None:
    response = client.get("/v1/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["checks"]["models_registered"] is True


def test_admission_returns_429_when_exhausted(monkeypatch) -> None:
    """When the runtime's admission capacity is exhausted, /v1/optimize
    answers 429. The semaphore is swapped on the lifespan-owned runtime —
    no module-global admission state exists anymore (PR-A)."""

    class _LockedSemaphore:
        def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
            return False

        def release(self) -> None:
            pass

    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = app.state.runtime
        assert runtime is not None
        monkeypatch.setattr(runtime, "_admission", _LockedSemaphore())
        buf = io.BytesIO()
        np.save(buf, np.zeros((32, 32), dtype=np.float32))
        buf.seek(0)
        response = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", buf, "application/octet-stream")},
            data={"model": "dummy-identity", "node": "45nm"},
        )
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "5"


def test_api_key_rejects_without_header() -> None:
    with TestClient(create_app(ServerConfig(api_key="sekrit"))) as fresh:
        assert fresh.get("/v1/models").status_code == 401
        assert fresh.get("/v1/health").status_code == 200  # liveness stays open
    keyed = TestClient(create_app(ServerConfig(api_key="sekrit")), headers={"X-API-Key": "sekrit"})
    with keyed as ok:
        assert ok.get("/v1/models").status_code == 200


def test_job_queue_full_before_upload_returns_429_without_copy() -> None:
    """P1.4: the queue slot is reserved BEFORE the upload is ingested.
    One occupied slot (depth 1, held via the public reservation protocol)
    must 429 the next submission without consuming its upload."""
    app = create_app(ServerConfig(job_queue_depth=1))
    with TestClient(app) as client:
        runtime = app.state.runtime
        res = runtime.try_reserve_job_slot()  # consume the only slot
        try:
            buf = io.BytesIO()
            np.save(buf, np.zeros((32, 32), dtype=np.float32))
            buf.seek(0)
            response = client.post(
                "/v1/jobs/optimize",
                files={"layout": ("in.npy", buf, "application/octet-stream")},
                data={"model": "dummy-identity", "node": "45nm"},
            )
            assert response.status_code == 429
            assert response.headers.get("Retry-After") == "10"
            assert runtime.snapshot()["jobs_tracked"] == 0
        finally:
            res.release()
        assert runtime.snapshot()["job_queue_reserved"] == 0


def test_job_reservation_releases_on_scratch_error(monkeypatch) -> None:
    """P1.1: if scratch creation fails after slot reservation, the slot
    must be released — the queue must not permanently show full."""
    from openlithohub.server import app as app_mod

    app = create_app(ServerConfig())
    with TestClient(app, raise_server_exceptions=False) as client:
        runtime = app.state.runtime
        monkeypatch.setattr(
            app_mod,
            "scratch_root",
            lambda: (_ for _ in ()).throw(OSError("disk full")),
        )
        buf = io.BytesIO()
        np.save(buf, np.zeros((32, 32), dtype=np.float32))
        buf.seek(0)
        response = client.post(
            "/v1/jobs/optimize",
            files={"layout": ("in.npy", buf, "application/octet-stream")},
            data={"model": "dummy-identity", "node": "45nm"},
        )
        # 500 from unhandled OSError, but the reservation was released by
        # the transactional reservation context manager.
        assert response.status_code == 500
        assert runtime.snapshot()["job_queue_reserved"] == 0, "reservation leaked"


def test_job_upload_too_large_releases_reservation() -> None:
    """P1.1: 413 path must release the reservation (config-bounded)."""
    app = create_app(ServerConfig(max_upload_bytes=10))
    with TestClient(app, raise_server_exceptions=False) as client:
        runtime = app.state.runtime
        response = client.post(
            "/v1/jobs/optimize",
            files={"layout": ("big.bin", io.BytesIO(b"x" * 100), "application/octet-stream")},
            data={"model": "dummy-identity"},
        )
        assert response.status_code == 413
        assert runtime.snapshot()["job_queue_reserved"] == 0, "reservation leaked on 413"


def _seed_job(runtime, job_id: str, status: str) -> None:
    import time as time_mod

    with runtime._job_lock:
        runtime._jobs[job_id] = {
            "status": status,
            "error": None,
            "summary": None,
            "output_path": None,
            "scratch_dir": "",
            "created_utc": "now",
            "job_id": job_id,
            "_created_monotonic": time_mod.monotonic(),
        }


def test_get_job_artifact_returns_409_for_failed_job() -> None:
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = app.state.runtime
        _seed_job(runtime, "test-failed", "failed")
        runtime._jobs["test-failed"]["error"] = "boom"
        response = client.get("/v1/jobs/test-failed/artifact")
        assert response.status_code == 409
        with runtime._job_lock:
            runtime._jobs.clear()


def test_job_delete_running_returns_409() -> None:
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = app.state.runtime
        _seed_job(runtime, "test-running", "running")
        response = client.delete("/v1/jobs/test-running")
        assert response.status_code == 409
        with runtime._job_lock:
            runtime._jobs.clear()
