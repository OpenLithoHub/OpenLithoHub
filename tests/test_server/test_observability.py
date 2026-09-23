"""PR-D observability unit tests: bounded cardinality, frozen event
vocabulary, and the metric-dimension firewall (roadmap §11/§13)."""

from __future__ import annotations

import json
import logging

import pytest

from openlithohub.server import observability as obs
from openlithohub.server.observability import MetricsRegistry, emit_event


@pytest.fixture(autouse=True)
def _clean() -> object:
    obs.registry.reset()
    yield
    obs.registry.reset()


def test_cardinality_hammer_unique_identifiers_do_not_create_series() -> None:
    """Five thousand unique request/job IDs must not create five thousand
    metric series (PR-D §13): identifiers are not metric dimensions."""
    registry = MetricsRegistry()
    for i in range(5000):
        registry.request_completed(200 + (i % 3), 1.0)
        registry.optimize_completed(
            execution_mode="dense",
            execution_reason=f"REASON_{i}",  # hostile: unbounded source
            input_backend="dense-raster",
            output_backend="dense-oasis",
            tiles=1,
        )
        registry.job_created()
    rendered = registry.render()
    by_reason = rendered["optimize"]["optimize_by_execution_reason"]
    assert len(by_reason) <= 33, "dimension cardinality cap breached"
    assert by_reason.get("_other"), "overflow must be bucketed, never dropped"
    assert rendered["requests"]["requests_total"] == 5000
    assert rendered["optimize"]["optimize_succeeded_total"] == 5000
    assert rendered["jobs"]["jobs_created_total"] == 5000


def test_bounded_dimensions_keep_known_values_exact() -> None:
    registry = MetricsRegistry()
    for mode in ("auto", "dense", "streaming"):
        registry.optimize_completed(
            execution_mode=mode,
            execution_reason="DENSE_SMALL_LAYOUT",
            input_backend="dense-raster",
            output_backend="dense-oasis",
            tiles=2,
        )
    rendered = registry.render()
    assert rendered["optimize"]["optimize_by_execution_mode"] == {
        "auto": 1,
        "dense": 1,
        "streaming": 1,
    }
    assert rendered["optimize"]["tiles_processed_total"] == 6


def test_request_gauges_and_duration_contract() -> None:
    registry = MetricsRegistry()
    for _ in range(5):
        registry.request_started()
    registry.request_completed(200, 5.0)
    registry.request_completed(503, 2.5)
    registry.request_completed(404, 1.0)
    rendered = registry.render()
    requests = rendered["requests"]
    assert requests["requests_total"] == 3
    assert requests["requests_in_flight"] == 2
    assert requests["request_duration_ms_count"] == 3
    assert requests["request_duration_ms_sum"] == 8.5
    assert requests["requests_by_status_class"] == {"2xx": 1, "5xx": 1, "4xx": 1}


def test_in_flight_gauge_never_goes_negative() -> None:
    registry = MetricsRegistry()
    registry.request_completed(200, 1.0)  # misbalanced caller
    assert registry.render()["requests"]["requests_in_flight"] == 0


def test_event_vocabulary_is_frozen() -> None:
    expected = {
        "http_request_completed",
        "optimize_started",
        "optimize_completed",
        "optimize_failed",
        "job_created",
        "job_started",
        "job_succeeded",
        "job_failed",
        "job_cancelled",
        "runtime_started",
        "runtime_draining",
        "runtime_stopped",
    }
    assert set(obs.EVENTS) == expected


def test_emit_event_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="vocabulary is frozen"):
        emit_event("totally_ad_hoc_event", request_id="r1")


def test_emit_event_payload_shape_and_log_correlation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="openlithohub.server.events"):
        emit_event(
            "optimize_completed",
            request_id="req-7",
            job_id=None,
            execution_mode="streaming",
            execution_reason="STREAMING_REQUIRED_BY_MEMORY_POLICY",
            input_backend="memmap-raster",
            output_backend="streaming-memmap-raster",
            tile_count=9,
            duration_ms=12.5,
        )
    records = [r for r in caplog.records if r.name == "openlithohub.server.events"]
    assert records, "event was not logged"
    payload = json.loads(records[-1].message)
    assert payload["event"] == "optimize_completed"
    assert payload["request_id"] == "req-7"
    assert payload["job_id"] is None
    assert payload["execution_reason"] == "STREAMING_REQUIRED_BY_MEMORY_POLICY"
    assert payload["tile_count"] == 9
    assert payload["api_schema_version"] == "1"
    assert "utc" in payload


def test_job_terminal_counters() -> None:
    registry = MetricsRegistry()
    registry.job_terminal("succeeded")
    registry.job_terminal("succeeded")
    registry.job_terminal("failed")
    registry.job_terminal("cancelled")
    rendered = registry.render()["jobs"]
    assert rendered == {
        "jobs_created_total": 0,
        "jobs_succeeded_total": 2,
        "jobs_failed_total": 1,
        "jobs_cancelled_total": 1,
    }
