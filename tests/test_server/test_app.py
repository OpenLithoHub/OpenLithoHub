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

    _MODEL_CACHE.clear()
    _MODEL_LOCKS.clear()
    model_a, lock_a = _get_or_load_model("dummy-identity", {})
    model_b, lock_b = _get_or_load_model("dummy-identity", {})
    assert model_a is model_b
    assert lock_a is lock_b
    assert isinstance(lock_a, type(threading.Lock()))


def test_get_or_load_model_concurrent_requests_load_once() -> None:
    """Two threads asking for the same model must share one instance —
    no double-load, no race that produces two distinct models."""
    import threading

    from openlithohub.server.app import _MODEL_CACHE, _MODEL_LOCKS, _get_or_load_model

    _MODEL_CACHE.clear()
    _MODEL_LOCKS.clear()
    results: list[tuple[object, object]] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        results.append(_get_or_load_model("dummy-identity", {}))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(m) for m, _ in results}) == 1
    assert len({id(lock) for _, lock in results}) == 1
