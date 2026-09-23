"""PR-F client behaviour tests, run against the real ASGI app through
``httpx.ASGITransport`` — the same HTTP semantics as production, no
sockets. Covers: typed metadata, sync optimize with header projection,
programmable error codes, single-attempt POST policy, job lifecycle with
polling + atomic artifact download, and poll-after-restart on the sqlite
backend."""

from __future__ import annotations

import hashlib
import io
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402


class AsgiSyncTransport(httpx.BaseTransport):
    """Sync transport driving the ASGI app on a fresh event loop per
    request — production HTTP semantics for the sync client, no sockets."""

    def __init__(self, app: FastAPI) -> None:
        import asyncio

        self._asyncio = asyncio
        self._inner = httpx.ASGITransport(app=app)

    async def _drive(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        response = await self._inner.handle_async_request(request)
        await response.aread()
        return response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._asyncio.run(self._drive(request))
        # Fully materialized: a bytes-backed Response is a sync byte stream.
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
        )


from openlithohub.client import (  # noqa: E402
    ErrorCode,
    JobStatus,
    JobStatusResponse,
    OpenLithoHubApiError,
    OpenLithoHubClient,
    OptimizeResult,
)
from openlithohub.server import create_app  # noqa: E402
from openlithohub.server.config import ServerConfig  # noqa: E402
from openlithohub.server.runtime import ServerRuntime  # noqa: E402
from openlithohub.server.schemas import API_SCHEMA_VERSION  # noqa: E402


def _app(config: ServerConfig | None = None) -> FastAPI:
    """App with its lifespan authority started manually (ASGITransport does
    not run lifespans); mirrors create_app's lifespan exactly."""
    app = create_app(config if config is not None else ServerConfig())
    config = config if config is not None else ServerConfig()
    runtime = ServerRuntime(config, optimize_runner=app_module_optimize_runner(app))
    app.state.runtime = runtime
    runtime.start()
    return app


def app_module_optimize_runner(app: FastAPI):  # noqa: ANN201 — trivial alias
    from openlithohub.server.app import _run_optimize

    return _run_optimize


def _close_app(app: FastAPI) -> None:
    runtime: ServerRuntime | None = getattr(app.state, "runtime", None)
    if runtime is not None:
        runtime.stop()
        if not (runtime.worker_alive or runtime.executing):
            app.state.runtime = None


@pytest.fixture
def client() -> Iterator[OpenLithoHubClient]:
    app = _app()
    try:
        with OpenLithoHubClient("http://testserver", transport=AsgiSyncTransport(app)) as c:
            yield c
    finally:
        _close_app(app)


def _npy(side: int = 64, keep: tuple[int, int, int, int] = (16, 16, 48, 48)) -> io.BytesIO:
    arr = np.zeros((side, side), dtype=np.float32)
    y0, x0, y1, x1 = keep
    arr[y0:y1, x0:x1] = 1.0
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return buf


def _write_npy(path: Path, side: int = 64) -> None:
    arr = np.zeros((side, side), dtype=np.float32)
    arr[side // 4 : side // 2, side // 4 : side // 2] = 1.0
    np.save(path, arr)


# ---- typed metadata ---------------------------------------------------------


def test_metadata_endpoints_are_typed(client: OpenLithoHubClient) -> None:
    assert client.health().status == "ok"
    assert client.ready().ready is True
    version = client.version()
    assert version.api == "v1" and version.package == "openlithohub"
    capabilities = client.capabilities()
    assert capabilities.api_schema_version == API_SCHEMA_VERSION
    assert capabilities.streaming.engine is True
    assert "dummy-identity" in capabilities.models
    assert "dummy-identity" in client.models()
    metrics = client.metrics()
    assert metrics.scope == "process"
    assert "requests" in metrics.model_dump()


# ---- sync optimize: streaming + header projection ------------------------------


def test_sync_optimize_streams_to_destination(client: OpenLithoHubClient, tmp_path: Path) -> None:
    layout = tmp_path / "in.npy"
    _write_npy(layout)
    destination = tmp_path / "out.oas"

    result = client.optimize(
        layout,
        model="dummy-identity",
        pixel_nm=1.0,
        tile_size=32,
        writer="vsb",
        execution_mode="auto",
        destination=destination,
    )
    assert isinstance(result, OptimizeResult)
    assert destination.is_file() and destination.stat().st_size > 0
    assert not destination.with_name(destination.name + ".part").exists()
    assert result.execution_mode == "dense"
    assert result.execution_reason == "DENSE_SMALL_LAYOUT"
    assert result.input_backend == "dense-raster"
    assert result.shape == (64, 64)
    assert result.request_id


def test_streaming_upload_handles_multi_chunk_files(
    client: OpenLithoHubClient, tmp_path: Path
) -> None:
    """A file far larger than one HTTP chunk flows through the streamed
    multipart path end to end (no full-body buffering)."""
    layout = tmp_path / "big.npy"
    side = 2048  # 2048² fp32 = 16 MB
    with open(layout, "wb") as fh:
        np.save(fh, np.zeros((side, side), dtype=np.float32))
    destination = tmp_path / "big-out.npy"

    result = client.optimize(
        layout,
        model="dummy-identity",
        pixel_nm=1.0,
        tile_size=2048,
        writer="vsb",
        execution_mode="auto",
        destination=destination,
    )
    assert destination.is_file()
    assert result.tiles >= 1


# ---- programmable error codes (PR-D vocabulary) ---------------------------------


def test_error_mapping_keys_on_code_not_text(client: OpenLithoHubClient, tmp_path: Path) -> None:
    layout = tmp_path / "in.npy"
    _write_npy(layout)
    with pytest.raises(OpenLithoHubApiError) as excinfo:
        client.optimize(layout, model="not-a-real-model", destination=tmp_path / "o.oas")
    error = excinfo.value
    assert error.code == ErrorCode.UNKNOWN_MODEL.value
    assert error.status_code == 404
    assert error.request_id
    assert error.message  # human detail available, never the primary key


def test_streaming_unsupported_error_is_programmatic(
    client: OpenLithoHubClient, tmp_path: Path
) -> None:
    layout = tmp_path / "in.npy"
    _write_npy(layout)
    with pytest.raises(OpenLithoHubApiError) as excinfo:
        client.optimize(
            layout,
            model="dummy-identity",
            pixel_nm=1.0,
            writer="mbmw",
            execution_mode="streaming",
            destination=tmp_path / "o.oas",
        )
    assert excinfo.value.code == ErrorCode.STREAMING_UNSUPPORTED.value
    assert excinfo.value.status_code == 400


def test_admission_full_surfaces_as_programmatic_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Locked:
        def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
            return False

        def release(self) -> None:
            pass

    app = _app()
    try:
        runtime = app.state.runtime
        assert runtime is not None
        monkeypatch.setattr(runtime, "_admission", _Locked())
        with OpenLithoHubClient("http://testserver", transport=AsgiSyncTransport(app)) as client:
            layout = tmp_path / "in.npy"
            _write_npy(layout)
            with pytest.raises(OpenLithoHubApiError) as excinfo:
                client.optimize(layout, model="dummy-identity", destination=tmp_path / "o.oas")
        assert excinfo.value.code == ErrorCode.ADMISSION_FULL.value
        assert excinfo.value.status_code == 429
    finally:
        _close_app(app)


def test_connection_errors_are_distinct(tmp_path: Path) -> None:
    from openlithohub.client import OpenLithoHubConnectionError

    with (
        OpenLithoHubClient("http://127.0.0.1:1") as client,  # nothing listens
        pytest.raises(OpenLithoHubConnectionError),
    ):
        client.health()


# ---- POST single-attempt policy ------------------------------------------------


class _CountingTransport(httpx.BaseTransport):
    """Fails every request with a 500 envelope and counts attempts."""

    def __init__(self) -> None:
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(
            500,
            json={
                "api_schema_version": "1",
                "detail": "internal server error",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "internal server error",
                    "request_id": "req-fixed",
                },
            },
        )


def test_requests_are_never_retried(tmp_path: Path) -> None:
    transport = _CountingTransport()
    with OpenLithoHubClient("http://testserver", transport=transport) as client:
        layout = tmp_path / "in.npy"
        _write_npy(layout)
        with pytest.raises(OpenLithoHubApiError) as excinfo:
            client.optimize(layout, model="dummy-identity", destination=tmp_path / "o.oas")
    assert excinfo.value.code == "INTERNAL_ERROR"
    assert excinfo.value.request_id == "req-fixed"
    assert transport.calls == 1, "POST must not be retried"


# ---- async job lifecycle: polling + atomic download ------------------------------


def test_job_lifecycle_poll_and_atomic_download(client: OpenLithoHubClient, tmp_path: Path) -> None:
    layout = tmp_path / "in.npy"
    _write_npy(layout)
    job_id = client.create_job(
        layout, model="dummy-identity", pixel_nm=1.0, tile_size=32, writer="vsb"
    )
    assert job_id

    polls: list[JobStatusResponse] = []
    snapshot = client.wait_for_job(job_id, timeout=60, poll_interval=0.05, on_poll=polls.append)
    assert snapshot.status == JobStatus.SUCCEEDED
    assert snapshot.summary is not None
    assert snapshot.summary.tiles >= 1
    assert snapshot.completed_utc is not None
    for earlier in polls:
        assert earlier.status in ("queued", "running")

    destination = tmp_path / "artifact.oas"
    served = client.download_artifact(job_id, destination)
    assert served == destination
    assert destination.is_file()
    assert not destination.with_name(destination.name + ".part").exists()
    first_digest = hashlib.sha256(destination.read_bytes()).hexdigest()

    # overwrite guard
    with pytest.raises(FileExistsError):
        client.download_artifact(job_id, destination)
    client.download_artifact(job_id, destination, overwrite=True)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == first_digest, (
        "download must be stable"
    )

    assert client.delete_job(job_id) == job_id
    with pytest.raises(OpenLithoHubApiError) as excinfo:
        client.get_job(job_id)
    assert excinfo.value.code == ErrorCode.UNKNOWN_JOB.value


def test_artifact_of_running_job_is_programmatic_409(
    client: OpenLithoHubClient, tmp_path: Path
) -> None:
    from openlithohub.server.job_store import JobRecord

    runtime = client._client._transport._inner.app.state.runtime  # type: ignore[attr-defined]
    assert runtime is not None
    runtime._store.create_queued(JobRecord(job_id="job-ui", status=JobStatus.QUEUED))
    runtime._store.transition("job-ui", JobStatus.RUNNING)
    with pytest.raises(OpenLithoHubApiError) as excinfo:
        client.download_artifact("job-ui", tmp_path / "never.oas")
    assert excinfo.value.code == ErrorCode.JOB_ARTIFACT_UNAVAILABLE.value
    assert excinfo.value.status_code == 409
    assert not (tmp_path / "never.oas").exists()


def test_wait_for_job_timeout_is_explicit(tmp_path: Path) -> None:
    from openlithohub.client import OpenLithoHubJobTimeoutError

    app = _app()
    try:
        runtime = app.state.runtime
        assert runtime is not None
        from openlithohub.server.job_store import JobRecord, serialize_params

        runtime._store.create_queued(
            JobRecord(
                job_id="job-slow",
                status=JobStatus.QUEUED,
                params_json=serialize_params({"output_path": ""}),
            )
        )
        # Freeze the worker claim so the job never leaves QUEUED.
        runtime._store.peek_next_queued = lambda: None  # type: ignore[method-assign]
        with (
            OpenLithoHubClient("http://testserver", transport=AsgiSyncTransport(app)) as client,
            pytest.raises(OpenLithoHubJobTimeoutError) as excinfo,
        ):
            client.wait_for_job("job-slow", timeout=0.5, poll_interval=0.05)
        assert excinfo.value.job_id == "job-slow"
    finally:
        _close_app(app)


# ---- poll after server restart (sqlite durability through the SDK) ---------------


def test_job_polling_and_artifact_survive_server_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ServerConfig(job_backend="sqlite", state_dir=str(tmp_path / "state"))

    app_a = _app(config)
    with OpenLithoHubClient("http://testserver", transport=AsgiSyncTransport(app_a)) as client_a:
        layout = tmp_path / "in.npy"
        _write_npy(layout)
        job_id = client_a.create_job(
            layout, model="dummy-identity", pixel_nm=1.0, tile_size=32, writer="vsb"
        )
        snapshot = client_a.wait_for_job(job_id, timeout=60, poll_interval=0.05)
        assert snapshot.status == JobStatus.SUCCEEDED
    _close_app(app_a)

    # "Restart": a brand-new app instance on the same durable state dir.
    app_b = _app(config)
    try:
        with OpenLithoHubClient(
            "http://testserver", transport=AsgiSyncTransport(app_b)
        ) as client_b:
            after = client_b.get_job(job_id)
            assert after.status == JobStatus.SUCCEEDED
            assert after.completed_utc is not None
            destination = tmp_path / "after-restart.oas"
            client_b.download_artifact(job_id, destination)
            assert destination.stat().st_size > 0
    finally:
        _close_app(app_b)


# ---- api key header ----------------------------------------------------------------


def test_api_key_header_is_sent(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    class _EchoTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            seen["api_key"] = request.headers.get("X-API-Key", "")
            seen["request_id"] = request.headers.get("X-Request-ID", "")
            return httpx.Response(200, json={"status": "ok"})

    with OpenLithoHubClient(
        "http://testserver", api_key="secret-key", transport=_EchoTransport()
    ) as client:
        assert client.health().status == "ok"
    assert seen["api_key"] == "secret-key"
    assert len(seen["request_id"]) == 32


# ---- concurrent downloads do not cross streams ---------------------------------------


def test_parallel_downloads_write_distinct_files(
    client: OpenLithoHubClient, tmp_path: Path
) -> None:
    layout = tmp_path / "in.npy"
    _write_npy(layout)
    job_ids = [
        client.create_job(layout, model="dummy-identity", pixel_nm=1.0, tile_size=32, writer="vsb")
        for _ in range(2)
    ]
    for job_id in job_ids:
        client.wait_for_job(job_id, timeout=60, poll_interval=0.05)

    results: dict[str, int] = {}

    def download(job_id: str) -> None:
        destination = tmp_path / f"art-{job_id}.oas"
        client.download_artifact(job_id, destination)
        results[job_id] = destination.stat().st_size

    threads = [threading.Thread(target=download, args=(job_id,)) for job_id in job_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert len(results) == 2
    assert all(size > 0 for size in results.values())
    time.sleep(0)  # placate linters; joins above are the synchronization
