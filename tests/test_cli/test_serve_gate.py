"""PR-A: `openlithohub serve --workers > 1` must be rejected while the
async job backend is process-local (repair-plan P0.3) — an invalid
topology must fail fast, not serve 404s for jobs created in another
worker process."""

from __future__ import annotations

import sys
import types

import pytest
from typer.testing import CliRunner

from openlithohub.cli.app import app

runner = CliRunner()


@pytest.fixture
def fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Stub out uvicorn so no server actually boots; record calls."""
    calls: list[dict[str, object]] = []

    def _fake_run(*args: object, **kwargs: object) -> None:
        calls.append({"args": args, **kwargs})

    stub = types.ModuleType("uvicorn")
    stub.run = _fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", stub)
    return calls


def test_serve_multi_worker_rejected_with_in_memory_backend(
    fake_uvicorn: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENLITHOHUB_JOB_BACKEND", raising=False)
    result = runner.invoke(app, ["serve", "--port", "0", "--workers", "2"])
    assert result.exit_code == 1
    assert "workers" in result.output
    assert fake_uvicorn == [], "uvicorn must not boot an invalid topology"


def test_serve_single_worker_boots(
    fake_uvicorn: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENLITHOHUB_JOB_BACKEND", raising=False)
    result = runner.invoke(app, ["serve", "--port", "0", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert len(fake_uvicorn) == 1
    assert fake_uvicorn[0]["workers"] == 1
    assert fake_uvicorn[0]["factory"] is True
    assert fake_uvicorn[0]["args"] == ("openlithohub.server.app:create_app",)


def test_serve_rejects_unknown_job_backend(
    fake_uvicorn: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENLITHOHUB_JOB_BACKEND", "not-a-backend")
    result = runner.invoke(app, ["serve", "--port", "0", "--workers", "1"])
    assert result.exit_code == 1
    assert "job backend" in result.output
    assert fake_uvicorn == []
