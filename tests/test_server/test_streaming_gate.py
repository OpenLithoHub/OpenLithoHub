"""PR-C server gate: the HTTP surface exposes the shared execution planner.

Covers: ``execution_mode`` form field on the sync endpoint and the job
API, execution decision headers, fail-closed 400 for unsupported large
requests, and the truthful ``/v1/capabilities`` streaming matrix.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlithohub.server import create_app  # noqa: E402
from openlithohub.server.config import ServerConfig  # noqa: E402
from openlithohub.workflow.execution import (  # noqa: E402
    STREAMING_INPUT_SUPPORT,
    klayout_available,
    streaming_capability_matrix,
)

_KLAYOUT = klayout_available()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(ServerConfig())) as c:
        yield c


def _npy_bytes(side: int = 64, keep: tuple[int, int, int, int] = (16, 16, 48, 48)) -> bytes:
    arr = np.zeros((side, side), dtype=np.float32)
    y0, x0, y1, x1 = keep
    arr[y0:y1, x0:x1] = 1.0
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _post_optimize(client: TestClient, data: dict[str, str]) -> object:
    with io.BytesIO(_npy_bytes()) as fh:
        return client.post(
            "/v1/optimize",
            files={"layout": ("input.npy", fh, "application/octet-stream")},
            data={"model": "dummy-identity", "pixel_nm": "1.0", "tile_size": "32", **data},
        )


def test_sync_endpoint_defaults_to_dense_and_reports_decision(client: TestClient) -> None:
    response = _post_optimize(client, {"writer": "vsb", "execution_mode": "auto"})
    assert response.status_code == 200, response.text
    assert response.headers["X-OLH-Execution-Mode"] == "dense"
    assert response.headers["X-OLH-Execution-Reason"] == "DENSE_SMALL_LAYOUT"


@pytest.mark.skipif(not _KLAYOUT, reason="streaming Manhattan output needs klayout.db")
def test_sync_endpoint_explicit_streaming_succeeds(client: TestClient, tmp_path: Path) -> None:
    response = _post_optimize(client, {"writer": "vsb", "execution_mode": "streaming"})
    assert response.status_code == 200, response.text
    assert response.headers["X-OLH-Execution-Mode"] == "streaming"
    assert response.headers["X-OLH-Execution-Reason"] == "STREAMING_SUPPORTED"
    assert len(response.content) > 0


def test_sync_endpoint_unsupported_streaming_is_400(client: TestClient) -> None:
    response = _post_optimize(client, {"writer": "mbmw", "execution_mode": "streaming"})
    assert response.status_code == 400
    assert "STREAMING_UNSUPPORTED_OUTPUT" in response.text


def test_sync_endpoint_pt_input_stays_dense_under_default_policy(
    client: TestClient,
) -> None:
    """A .pt upload is inherently dense input: under the default policy the
    planner keeps it dense (the tiny-policy fail-closed case is pinned at
    the unit layer, where the budget is injectable)."""
    import torch

    buf = io.BytesIO()
    torch.save(torch.zeros((64, 64), dtype=torch.float32), buf)
    buf.seek(0)
    response = client.post(
        "/v1/optimize",
        files={"layout": ("input.pt", buf, "application/octet-stream")},
        data={
            "model": "dummy-identity",
            "pixel_nm": "1.0",
            "tile_size": "32",
            "writer": "vsb",
            "execution_mode": "auto",
        },
    )
    # The policy lives in the executor's environment; force it server-side
    # via the env the app was constructed with is not possible here, so a
    # normal-size job stays dense. A tiny policy is exercised at the unit
    # layer; here we assert the honest default still works end to end.
    assert response.status_code == 200
    assert response.headers["X-OLH-Execution-Mode"] == "dense"


def test_sync_endpoint_invalid_execution_mode_is_400(client: TestClient) -> None:
    response = _post_optimize(client, {"execution_mode": "turbo"})
    assert response.status_code == 400


def test_capabilities_reports_truthful_streaming_matrix(client: TestClient) -> None:
    response = client.get("/v1/capabilities")
    assert response.status_code == 200
    body = response.json()
    matrix = body["streaming"]
    assert matrix == streaming_capability_matrix()
    assert matrix["engine"] is True
    assert matrix["product_execution"] is True
    assert matrix["inputs"] == dict(STREAMING_INPUT_SUPPORT)
    assert matrix["outputs"]["manhattan_oasis"] == klayout_available()
    assert matrix["outputs"]["curvilinear_oasis"] is False
    assert matrix["memory_policy"]["fail_closed"] is True
    # Legacy coarse flag remains for v1 clients.
    assert body["streaming_pipeline"] is True


@pytest.mark.skipif(not _KLAYOUT, reason="streaming Manhattan output needs klayout.db")
def test_job_api_reports_execution_decision_in_summary(client: TestClient) -> None:
    with io.BytesIO(_npy_bytes()) as fh:
        created = client.post(
            "/v1/jobs/optimize",
            files={"layout": ("input.npy", fh, "application/octet-stream")},
            data={
                "model": "dummy-identity",
                "pixel_nm": "1.0",
                "tile_size": "32",
                "writer": "vsb",
                "execution_mode": "streaming",
            },
        )
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    for _ in range(200):
        snap = client.get(f"/v1/jobs/{job_id}").json()
        if snap["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert snap["status"] == "succeeded", snap
    summary = snap["summary"]
    assert summary["execution_mode"] == "streaming"
    assert summary["execution_reason"] == "STREAMING_SUPPORTED"
    assert summary["output_backend"] == "streaming-manhattan-oasis"
