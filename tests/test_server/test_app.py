"""Tests for the FastAPI HTTP engine."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlithohub.server import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


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
    import io
    import time as time_mod

    import numpy as np

    client = TestClient(create_app())
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
    """When the admission semaphore is exhausted, /v1/optimize answers 429."""
    from openlithohub.server import app as app_mod

    class _LockedSemaphore:
        def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr(app_mod, "_ADMIT", _LockedSemaphore())
    import io

    import numpy as np

    client = TestClient(create_app())
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


def test_api_key_rejects_without_header(monkeypatch) -> None:
    from openlithohub.server import app as app_mod

    monkeypatch.setattr(app_mod, "API_KEY", "sekrit")
    fresh = TestClient(create_app())
    assert fresh.get("/v1/models").status_code == 401
    assert fresh.get("/v1/health").status_code == 200  # liveness stays open
    ok = TestClient(create_app(), headers={"X-API-Key": "sekrit"})
    assert ok.get("/v1/models").status_code == 200


def test_worker_start_is_idempotent() -> None:
    """P1.1: repeated create_app() must not spawn additional workers."""
    import threading

    before = threading.active_count()
    apps = [create_app() for _ in range(5)]
    assert all(apps)
    # bounded slack: other tests may hold transient threads, but the
    # worker pool itself must not grow by create_app() count.
    after = threading.active_count()
    assert after - before <= 1


def test_job_queue_full_before_upload_returns_429_without_copy(monkeypatch) -> None:
    """P1.4: the queue slot is reserved BEFORE the upload is ingested."""
    from openlithohub.server import app as app_mod

    monkeypatch.setattr(app_mod, "_JOB_QUEUE_DEPTH", 1)
    monkeypatch.setattr(app_mod, "_JOB_QUEUE", __import__("queue").Queue(maxsize=1))
    import io

    import numpy as np

    client = TestClient(create_app())
    buf = io.BytesIO()
    np.save(buf, np.zeros((32, 32), dtype=np.float32))
    buf.seek(0)
    # fill the queue with a real queued record so the reservation fails
    with JobReserveGuard(app_mod):
        response = client.post(
            "/v1/jobs/optimize",
            files={"layout": ("in.npy", buf, "application/octet-stream")},
            data={"model": "dummy-identity", "node": "45nm"},
        )
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "10"


class JobReserveGuard:
    """Fill the job queue for the duration of the request."""

    def __init__(self, app_mod) -> None:
        self.app_mod = app_mod

    def __enter__(self):
        self.app_mod._JOB_QUEUE.put(("sentinel", {}))
        return self

    def __exit__(self, *exc):
        import contextlib

        with contextlib.suppress(Exception):
            self.app_mod._JOB_QUEUE.get_nowait()


def test_job_reservation_releases_on_scratch_error(monkeypatch) -> None:
    """P1.1: if scratch creation fails after slot reservation, the slot
    must be released — the queue must not permanently show full."""
    from openlithohub.server import app as app_mod

    monkeypatch.setattr(
        app_mod,
        "make_scratch_dir",
        lambda prefix: (_ for _ in ()).throw(OSError("disk full")),
    )
    client = TestClient(create_app(), raise_server_exceptions=False)
    buf = io.BytesIO()
    np.save(buf, np.zeros((32, 32), dtype=np.float32))
    buf.seek(0)
    response = client.post(
        "/v1/jobs/optimize",
        files={"layout": ("in.npy", buf, "application/octet-stream")},
        data={"model": "dummy-identity", "node": "45nm"},
    )
    # 500 from unhandled OSError, but the reservation was released by the
    # try/except BaseException in the endpoint.
    assert response.status_code == 500
    assert app_mod._JOB_QUEUE_RESERVED == 0, "reservation leaked"


def test_job_upload_too_large_releases_reservation(monkeypatch) -> None:
    """P1.1: 413 path must release the reservation."""
    from openlithohub.server import app as app_mod

    monkeypatch.setattr(app_mod, "_MAX_UPLOAD_BYTES", 10)
    initial = app_mod._JOB_QUEUE_RESERVED
    client = TestClient(create_app(), raise_server_exceptions=False)
    buf = io.BytesIO(b"x" * 100)
    response = client.post(
        "/v1/jobs/optimize",
        files={"layout": ("big.bin", buf, "application/octet-stream")},
        data={"model": "dummy-identity"},
    )
    assert response.status_code == 413
    assert initial == app_mod._JOB_QUEUE_RESERVED, "reservation leaked on 413"


def test_get_job_artifact_returns_409_for_failed_job() -> None:

    from openlithohub.server import app as app_mod

    client = TestClient(create_app())
    with app_mod._JOB_LOCK:
        app_mod._JOBS["test-failed"] = {
            "status": "failed",
            "error": "boom",
            "summary": None,
            "output_path": None,
            "scratch_dir": "",
            "created_utc": "now",
            "job_id": "test-failed",
            "_created_monotonic": __import__("time").monotonic(),
        }
    response = client.get("/v1/jobs/test-failed/artifact")
    assert response.status_code == 409
    with app_mod._JOB_LOCK:
        app_mod._JOBS.clear()


def test_job_delete_running_returns_409() -> None:

    from openlithohub.server import app as app_mod

    try:
        with app_mod._JOB_LOCK:
            app_mod._JOBS["test-running"] = {
                "status": "running",
                "error": None,
                "summary": None,
                "output_path": None,
                "scratch_dir": "",
                "created_utc": "now",
                "job_id": "test-running",
            }
        client = TestClient(create_app())
        response = client.delete("/v1/jobs/test-running")
        assert response.status_code == 409
    finally:
        with app_mod._JOB_LOCK:
            app_mod._JOBS.clear()
