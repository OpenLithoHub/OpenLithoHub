"""PR-E E11/E12: backend parity + OpenAPI invariance.

The async API contract must be identical across ``in-memory`` and
``sqlite`` backends (modulo the truthful capability values), and the
storage migration must not alter the OpenAPI v1 document.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlithohub.server import create_app  # noqa: E402
from openlithohub.server.config import ServerConfig  # noqa: E402
from openlithohub.server.schemas import API_SCHEMA_VERSION  # noqa: E402


def _sqlite_config(state_dir) -> ServerConfig:
    return ServerConfig(job_backend="sqlite", state_dir=str(state_dir))


@pytest.fixture(params=["in-memory", "sqlite"])
def backend_client(request, tmp_path) -> Iterator[tuple[TestClient, str]]:
    config = _sqlite_config(tmp_path / "state") if request.param == "sqlite" else ServerConfig()
    with TestClient(create_app(config)) as client:
        yield client, request.param


def _npy() -> io.BytesIO:
    arr = np.zeros((48, 48), dtype=np.float32)
    arr[12:24, 12:36] = 1.0
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return buf


def _post_job(client: TestClient, **extra: str):
    return client.post(
        "/v1/jobs/optimize",
        files={"layout": ("in.npy", _npy(), "application/octet-stream")},
        data={"model": "dummy-identity", "node": "45nm", **extra},
    )


def test_e11_job_lifecycle_parity(backend_client, tmp_path) -> None:
    """Create -> poll -> artifact -> delete produces byte-identical public
    JSON shapes and status codes on both backends."""
    client, _backend = backend_client
    created = _post_job(client)
    assert created.status_code == 202, created.text
    body = created.json()
    # Public create contract is identical (shape + fields + version).
    assert set(body.keys()) == {"api_schema_version", "job_id", "status", "poll"}
    assert body["api_schema_version"] == API_SCHEMA_VERSION
    assert body["status"] == "queued"
    job_id = body["job_id"]

    snapshot = None
    for _ in range(200):
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        snapshot = response.json()
        if snapshot["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert snapshot["status"] == "succeeded", snapshot
    # Public job snapshot contract is identical on both backends.
    assert set(snapshot.keys()) == {
        "api_schema_version",
        "job_id",
        "status",
        "created_utc",
        "started_utc",
        "completed_utc",
        "summary",
        "error",
    }
    assert snapshot["summary"]["execution_reason"] == "DENSE_SMALL_LAYOUT"
    assert "output_path" not in (snapshot["summary"] or {})

    artifact = client.get(f"/v1/jobs/{job_id}/artifact")
    assert artifact.status_code == 200
    assert len(artifact.content) > 0

    deleted = client.delete(f"/v1/jobs/{job_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": job_id}
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_e11_error_contract_parity(backend_client) -> None:
    client, _backend = backend_client
    unknown_job = client.get("/v1/jobs/never-existed")
    assert unknown_job.status_code == 404
    body = unknown_job.json()
    assert body["error"]["code"] == "UNKNOWN_JOB"
    assert body["api_schema_version"] == API_SCHEMA_VERSION

    artifact = client.get("/v1/jobs/never-existed/artifact")
    assert artifact.status_code == 404
    assert artifact.json()["error"]["code"] == "UNKNOWN_JOB"


def test_e11_capability_truthfulness(backend_client) -> None:
    client, backend = backend_client
    jobs = client.get("/v1/capabilities").json()["jobs"]
    assert jobs == {
        "backend": backend,
        "durable": backend == "sqlite",
        "restart_loses_jobs": backend != "sqlite",
        "single_process_only": True,
    }


def test_e12_openapi_identical_across_backends(tmp_path) -> None:
    in_memory = create_app(ServerConfig()).openapi()
    sqlite = create_app(_sqlite_config(tmp_path / "state")).openapi()
    assert in_memory == sqlite, "storage backend leaked into the OpenAPI schema"


def test_e12_snapshot_unchanged_by_pr_e() -> None:
    import scripts.check_openapi_snapshot as checker

    assert checker.SNAPSHOT.exists()
    assert checker.normalize(checker.generate_openapi()) == checker.SNAPSHOT.read_text()
