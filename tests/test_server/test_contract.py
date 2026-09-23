"""PR-D hostile contract tests: the frozen API surface.

Every major error case is pinned by (HTTP status, error.code) — never by
English text. Request-ID correlation spans response header, error envelope
and structured logs. The OpenAPI snapshot is a drift gate.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlithohub.server import create_app  # noqa: E402
from openlithohub.server.config import ServerConfig  # noqa: E402
from openlithohub.server.errors import runtime_error_code  # noqa: E402
from openlithohub.server.observability import registry  # noqa: E402
from openlithohub.server.schemas import (  # noqa: E402
    API_SCHEMA_VERSION,
    JOB_TRANSITIONS,
    ErrorCode,
    IllegalJobTransitionError,
    JobStatus,
    OptimizeMetadata,
    transition_job_status,
)
from openlithohub.workflow.execution import StreamingUnsupportedError  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_metrics() -> Iterator[None]:
    registry.reset()
    yield
    registry.reset()


def _npy(side: int = 64) -> io.BytesIO:
    arr = np.zeros((side, side), dtype=np.float32)
    arr[side // 4 : side // 2, side // 4 : side // 2] = 1.0
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return buf


def _post(client: TestClient, data: dict[str, str], **kw: object) -> object:
    return client.post(
        "/v1/optimize",
        files={"layout": ("input.npy", _npy(), "application/octet-stream")},
        data={"model": "dummy-identity", "pixel_nm": "1.0", "tile_size": "32", **data},
    )


# ---- job state machine firewall (roadmap §16) -----------------------------


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (JobStatus.SUCCEEDED, JobStatus.RUNNING),
        (JobStatus.SUCCEEDED, JobStatus.QUEUED),
        (JobStatus.FAILED, JobStatus.QUEUED),
        (JobStatus.FAILED, JobStatus.RUNNING),
        (JobStatus.CANCELLED, JobStatus.SUCCEEDED),
        (JobStatus.CANCELLED, JobStatus.RUNNING),
        (JobStatus.RUNNING, JobStatus.QUEUED),
        (JobStatus.RUNNING, JobStatus.CANCELLED),
    ],
)
def test_illegal_job_transitions_rejected(current: JobStatus, new: JobStatus) -> None:
    with pytest.raises(IllegalJobTransitionError):
        transition_job_status(current, new)


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (JobStatus.QUEUED, JobStatus.RUNNING),
        (JobStatus.QUEUED, JobStatus.CANCELLED),
        (JobStatus.RUNNING, JobStatus.SUCCEEDED),
        (JobStatus.RUNNING, JobStatus.FAILED),
    ],
)
def test_legal_job_transitions_accepted(current: JobStatus, new: JobStatus) -> None:
    assert transition_job_status(current, new) is new


def test_transition_table_is_exactly_the_documented_set() -> None:
    legal = {
        (JobStatus.QUEUED, JobStatus.RUNNING),
        (JobStatus.QUEUED, JobStatus.CANCELLED),
        (JobStatus.RUNNING, JobStatus.SUCCEEDED),
        (JobStatus.RUNNING, JobStatus.FAILED),
    }
    have = {(a, b) for a, outs in JOB_TRANSITIONS.items() for b in outs}
    assert have == legal
    # RUNNING -> CANCELLED is deliberately absent (PR-D §5): the runtime
    # does not own cancellation of an executing model.
    assert (JobStatus.RUNNING, JobStatus.CANCELLED) not in have


# ---- error-code stability (roadmap §16) ------------------------------------


def _error_code(body: dict) -> str:
    return body["error"]["code"]


def test_unknown_model_is_404_unknown_model() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        response = _post(client, {"model": "not-a-real-model"})
    assert response.status_code == 404
    body = response.json()
    assert _error_code(body) == ErrorCode.UNKNOWN_MODEL.value
    assert body["api_schema_version"] == API_SCHEMA_VERSION
    assert body["detail"]  # legacy compatibility detail preserved


def test_unknown_job_is_404_unknown_job() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        response = client.get("/v1/jobs/job-does-not-exist")
    assert response.status_code == 404
    assert _error_code(response.json()) == ErrorCode.UNKNOWN_JOB.value


def test_artifact_unavailable_is_409_job_artifact_unavailable() -> None:
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = app.state.runtime
        assert runtime is not None
        from openlithohub.server.job_store import JobRecord

        runtime._store.create_queued(JobRecord(job_id="j1", status=JobStatus.QUEUED))
        runtime._store.transition("j1", JobStatus.RUNNING)
        runtime._store.transition("j1", JobStatus.FAILED, error="boom")
        response = client.get("/v1/jobs/j1/artifact")
    assert response.status_code == 409
    assert _error_code(response.json()) == ErrorCode.JOB_ARTIFACT_UNAVAILABLE.value


def test_delete_running_job_is_409_job_running() -> None:
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = app.state.runtime
        assert runtime is not None
        from openlithohub.server.job_store import JobRecord

        runtime._store.create_queued(JobRecord(job_id="j2", status=JobStatus.QUEUED))
        runtime._store.transition("j2", JobStatus.RUNNING)
        response = client.delete("/v1/jobs/j2")
    assert response.status_code == 409
    assert _error_code(response.json()) == ErrorCode.JOB_RUNNING.value


def test_upload_too_large_is_413_upload_too_large() -> None:
    app = create_app(ServerConfig(max_upload_bytes=10))
    with TestClient(app) as client:
        response = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", io.BytesIO(b"x" * 100), "application/octet-stream")},
            data={"model": "dummy-identity"},
        )
    assert response.status_code == 413
    assert _error_code(response.json()) == ErrorCode.UPLOAD_TOO_LARGE.value


def test_queue_full_is_429_queue_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic: the worker's claim path is frozen so the single
    durable QUEUED row stays queued and the second creation must hit the
    capacity gate (durable rows count toward depth, PR-E E4)."""
    app = create_app(ServerConfig(job_queue_depth=1))
    with TestClient(app) as client:
        runtime = app.state.runtime
        assert runtime is not None
        monkeypatch.setattr(runtime._store, "peek_next_queued", lambda: None)
        first = client.post(
            "/v1/jobs/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={"model": "dummy-identity", "node": "45nm"},
        )
        assert first.status_code == 202, first.text
        second = client.post(
            "/v1/jobs/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={"model": "dummy-identity", "node": "45nm"},
        )
    assert second.status_code == 429
    assert _error_code(second.json()) == ErrorCode.QUEUE_FULL.value


def test_streaming_unsupported_is_400_streaming_unsupported() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        response = _post(client, {"writer": "mbmw", "execution_mode": "streaming"})
    assert response.status_code == 400
    assert _error_code(response.json()) == ErrorCode.STREAMING_UNSUPPORTED.value


def test_invalid_execution_mode_is_400_invalid_request() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        response = _post(client, {"execution_mode": "turbo"})
    assert response.status_code == 400
    assert _error_code(response.json()) == ErrorCode.INVALID_REQUEST.value


def test_missing_form_field_is_400_invalid_request() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        response = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={"pixel_nm": "1.0"},  # model missing -> request validation error
        )
    assert response.status_code == 400
    assert _error_code(response.json()) == ErrorCode.INVALID_REQUEST.value


def test_unknown_node_is_400_invalid_request() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        response = _post(client, {"node": "not-a-node"})
    assert response.status_code == 400
    assert _error_code(response.json()) == ErrorCode.INVALID_REQUEST.value


def test_admission_full_is_429_admission_full(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Locked:
        def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
            return False

        def release(self) -> None:
            pass

    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = app.state.runtime
        assert runtime is not None
        monkeypatch.setattr(runtime, "_admission", _Locked())
        response = _post(client, {})
    assert response.status_code == 429
    assert _error_code(response.json()) == ErrorCode.ADMISSION_FULL.value


def test_internal_error_is_500_internal_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openlithohub.server.app as app_mod

    def _poison(*_a: object, **_k: object) -> object:
        raise RuntimeError("secret internal detail /tmp/hidden/path")

    monkeypatch.setattr(app_mod, "_get_or_load_model", _poison)
    with TestClient(create_app(ServerConfig())) as client:
        response = _post(client, {})
    assert response.status_code == 500
    body = response.json()
    assert _error_code(body) == ErrorCode.INTERNAL_ERROR.value
    assert body["detail"] == "internal server error"
    assert "secret" not in response.text and "/tmp/hidden" not in response.text


def test_runtime_error_code_mapping_is_central() -> None:
    from openlithohub.server.runtime import (
        AdmissionDeniedError,
        JobArtifactUnavailableError,
        JobQueueFullError,
        JobStillRunningError,
        RuntimeNotAcceptingWorkError,
        UnknownJobError,
    )

    assert runtime_error_code(JobQueueFullError("x")) == (429, ErrorCode.QUEUE_FULL)
    assert runtime_error_code(AdmissionDeniedError("x")) == (429, ErrorCode.ADMISSION_FULL)
    assert runtime_error_code(RuntimeNotAcceptingWorkError("x")) == (
        503,
        ErrorCode.SERVER_NOT_ACCEPTING,
    )
    assert runtime_error_code(UnknownJobError("x")) == (404, ErrorCode.UNKNOWN_JOB)
    assert runtime_error_code(JobArtifactUnavailableError("x")) == (
        409,
        ErrorCode.JOB_ARTIFACT_UNAVAILABLE,
    )
    assert runtime_error_code(JobStillRunningError("x")) == (409, ErrorCode.JOB_RUNNING)
    assert runtime_error_code(StreamingUnsupportedError("r", "m")) == (
        400,
        ErrorCode.STREAMING_UNSUPPORTED,
    )
    assert runtime_error_code(ValueError("x")) == (400, ErrorCode.INVALID_REQUEST)
    assert runtime_error_code(KeyError("model 'x'")) == (404, ErrorCode.UNKNOWN_MODEL)
    assert runtime_error_code(KeyError("process node 'y'")) == (400, ErrorCode.INVALID_REQUEST)
    assert runtime_error_code(ImportError("x")) == (
        503,
        ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
    )
    assert runtime_error_code(Exception("x")) == (500, ErrorCode.INTERNAL_ERROR)


# ---- request-ID correlation (roadmap §16) ----------------------------------


def test_request_id_correlates_response_error_and_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger="openlithohub.server.access"),
        TestClient(create_app(ServerConfig())) as client,
    ):
        response = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={"model": "not-a-real-model"},
            headers={"X-Request-ID": "corr-identity-123"},
        )
    assert response.status_code == 404
    body = response.json()
    assert response.headers["X-Request-ID"] == "corr-identity-123"
    assert body["error"]["request_id"] == "corr-identity-123"
    access_records = [
        record for record in caplog.records if record.name == "openlithohub.server.access"
    ]
    assert access_records, "structured access log line missing"
    logged = json.loads(access_records[-1].message)
    assert logged["request_id"] == "corr-identity-123"


def test_invalid_request_id_is_replaced_by_bounded_id() -> None:
    hostile = "x" * 200 + "\ninject"
    with TestClient(create_app(ServerConfig())) as client:
        response = client.get("/v1/jobs/nope", headers={"X-Request-ID": hostile})
    assert response.status_code == 404
    rid = response.headers["X-Request-ID"]
    assert rid != hostile
    assert len(rid) <= 64
    assert response.json()["error"]["request_id"] == rid


def test_sanitize_request_id_table() -> None:
    from openlithohub.server.observability import sanitize_request_id

    assert sanitize_request_id("abc-DEF_123.4") == "abc-DEF_123.4"
    assert sanitize_request_id(None) != sanitize_request_id(None)  # generated
    for bad in ("a" * 65, "with space", "semi;colon", "unicode-é", "tab\tchar"):
        got = sanitize_request_id(bad)
        assert got != bad and len(got) <= 64


# ---- typed contract: schema drift must fail ---------------------------------


def test_optimize_metadata_forbids_field_drift() -> None:
    base = {
        "shape": [4, 4],
        "tiles": 1,
        "halo_px": 0,
        "writer": "vsb",
        "export_format": "oasis",
        "execution_mode": "dense",
        "execution_reason": "DENSE_SMALL_LAYOUT",
        "input_backend": "dense-raster",
        "output_backend": "dense-oasis",
        "threshold": 0.5,
    }
    OptimizeMetadata.model_validate(base)
    drifted = dict(base, totally_new_field=1)
    with pytest.raises(Exception, match="totally_new_field"):
        OptimizeMetadata.model_validate(drifted)


def test_public_job_snapshot_has_no_private_fields() -> None:
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        runtime = app.state.runtime
        assert runtime is not None
        from openlithohub.server.job_store import JobRecord

        runtime._store.create_queued(JobRecord(job_id="j3", status=JobStatus.QUEUED))
        runtime._store.transition("j3", JobStatus.RUNNING)
        runtime._store.transition(
            "j3",
            JobStatus.SUCCEEDED,
            summary={
                "shape": [8, 8],
                "tiles": 1,
                "halo_px": 0,
                "writer": "vsb",
                "export_format": "oasis",
                "execution_mode": "dense",
                "execution_reason": "DENSE_SMALL_LAYOUT",
                "input_backend": "dense-raster",
                "output_backend": "dense-oasis",
                "threshold": 0.5,
                "output_path": "/private/scratch/hidden.oas",
                "scratch_dir": "/private/scratch",
            },
            output_path="/private/scratch/hidden.oas",
        )
        body = client.get("/v1/jobs/j3").json()
    text = json.dumps(body)
    assert "hidden" not in text and "scratch" not in text
    assert body["status"] == "succeeded"
    assert body["started_utc"] is not None and body["completed_utc"] is not None
    assert body["summary"]["execution_reason"] == "DENSE_SMALL_LAYOUT"
    assert body["summary"]["threshold"] == 0.5
    assert "output_path" not in (body["summary"] or {})


# ---- sync optimize headers (roadmap §8) -------------------------------------


_KLAYOUT = True
try:
    import klayout.db  # noqa: F401
except ImportError:
    _KLAYOUT = False


@pytest.mark.skipif(not _KLAYOUT, reason="streaming header case needs klayout.db")
def test_sync_optimize_headers_exact() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        response = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={
                "model": "dummy-identity",
                "pixel_nm": "1.0",
                "tile_size": "32",
                "writer": "vsb",
                "execution_mode": "streaming",
            },
        )
    assert response.status_code == 200, response.text
    expected = {
        "X-Request-ID": None,  # presence only
        "X-OLH-Tiles": None,
        "X-OLH-Halo-Px": None,
        "X-OLH-Export-Format": None,
        "X-OLH-Shape": None,
        "X-OLH-Execution-Mode": "streaming",
        "X-OLH-Execution-Reason": "STREAMING_SUPPORTED",
        "X-OLH-Input-Backend": "memmap-raster",
        "X-OLH-Output-Backend": "streaming-manhattan-oasis",
    }
    for header, value in expected.items():
        assert header in response.headers, f"missing contract header {header}"
        if value is not None:
            assert response.headers[header] == value, header


# ---- OpenAPI snapshot drift gate (roadmap §16) -------------------------------


def test_openapi_snapshot_matches_generated() -> None:
    import scripts.check_openapi_snapshot as checker

    rendered = checker.normalize(checker.generate_openapi())
    assert checker.SNAPSHOT.exists()
    assert rendered == checker.SNAPSHOT.read_text(), (
        "OpenAPI drift: regenerate docs/api/openapi-v1.json in this PR"
    )


def test_openapi_drift_is_detectable() -> None:
    """Negative control: mutating one documented field must be detectable,
    so the drift gate is not vacuously green."""
    import scripts.check_openapi_snapshot as checker

    mutated = json.loads(checker.SNAPSHOT.read_text())
    components = mutated["components"]["schemas"]
    name = next(iter(components))
    components[name]["properties"]["__prc_drift_probe__"] = {"type": "string"}
    assert checker.normalize(mutated) != checker.SNAPSHOT.read_text()


# ---- metrics ownership + execution-reason telemetry (roadmap §14/§16) -------


def test_metrics_endpoint_derives_from_runtime_snapshot() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        body = client.get("/v1/metrics").json()
        runtime = client.app.state.runtime  # type: ignore[attr-defined]
        assert body["runtime"] == runtime.snapshot()
        assert body["api_schema_version"] == API_SCHEMA_VERSION
        assert body["scope"] == "process"
        for section in ("requests", "optimize", "jobs"):
            assert section in body


def test_execution_reason_telemetry_exact_values() -> None:
    with TestClient(create_app(ServerConfig())) as client:
        dense_explicit = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={
                "model": "dummy-identity",
                "pixel_nm": "1.0",
                "tile_size": "32",
                "execution_mode": "dense",
            },
        )
        auto = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={
                "model": "dummy-identity",
                "pixel_nm": "1.0",
                "tile_size": "32",
                "execution_mode": "auto",
            },
        )
    assert dense_explicit.status_code == 200 and auto.status_code == 200
    assert dense_explicit.headers["X-OLH-Execution-Reason"] == "DENSE_REQUESTED_EXPLICITLY"
    assert auto.headers["X-OLH-Execution-Reason"] == "DENSE_SMALL_LAYOUT"
    modes = registry.render()["optimize"]["optimize_by_execution_mode"]
    assert modes.get("dense") == 2
    reasons = registry.render()["optimize"]["optimize_by_execution_reason"]
    assert reasons.get("DENSE_REQUESTED_EXPLICITLY") == 1
    assert reasons.get("DENSE_SMALL_LAYOUT") == 1


@pytest.mark.skipif(not _KLAYOUT, reason="auto->streaming policy case needs klayout.db")
def test_memory_policy_forced_streaming_is_observable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENLITHOHUB_MAX_DENSE_BYTES", "1")
    with TestClient(create_app(ServerConfig())) as client:
        response = client.post(
            "/v1/optimize",
            files={"layout": ("in.npy", _npy(), "application/octet-stream")},
            data={
                "model": "dummy-identity",
                "pixel_nm": "1.0",
                "tile_size": "32",
                "writer": "vsb",
                "execution_mode": "auto",
            },
        )
    assert response.status_code == 200, response.text
    assert response.headers["X-OLH-Execution-Reason"] == "STREAMING_REQUIRED_BY_MEMORY_POLICY"
    reasons = registry.render()["optimize"]["optimize_by_execution_reason"]
    assert reasons.get("STREAMING_REQUIRED_BY_MEMORY_POLICY") == 1
