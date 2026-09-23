"""PR-D process-local observability — bounded metrics + structured events.

Provider-neutral by contract: this module freezes *semantics* (metric
names, dimensions, event vocabulary, cardinality rules) without pulling in
Prometheus/OpenTelemetry. A future provider adapter reads
:class:`MetricsRegistry` state or hooks :func:`emit_event`; it must not
need to change call sites.

Cardinality firewall (PR-D §13)
    Metric series are keyed ONLY on bounded dimensions — HTTP status
    class, execution mode, execution reason, input/output backend. The
    registry buckets any dimension value beyond ``_DIMENSION_CAP``
    distinct keys into ``_OTHER`` so a hammer of unique request/job IDs,
    filenames or paths can never create unbounded series. Raw identifiers
    (request_id, job_id) belong in *events* (logs), never in metric keys.

Scope (PR-D §12)
    Everything here is process-local and non-durable by definition: it
    describes one uvicorn worker, resets on restart, and is authoritative
    for nothing beyond this process. ``/v1/metrics`` renders it; runtime
    queue/admission state still comes from ``ServerRuntime.snapshot()`` —
    this module never duplicates that source of truth.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from openlithohub.server.schemas import API_SCHEMA_VERSION

EVENT_LOGGER = logging.getLogger("openlithohub.server.events")

# Structured event vocabulary (PR-D §11). Kept as a frozen tuple so a test
# can assert that every emitted event name belongs to the contract.
EVENTS: tuple[str, ...] = (
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
)

_DIMENSION_CAP = 32
_OTHER = "_other"

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def sanitize_request_id(raw: str | None) -> str:
    """Valid client IDs are preserved; anything else becomes a bounded
    generated ID (log-injection / cardinality firewall, PR-D §4)."""
    if raw is not None and _REQUEST_ID_RE.fullmatch(raw):
        return raw
    import uuid

    return uuid.uuid4().hex


def _utc_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


class _BoundedDimension:
    """A counter map over one dimension with a hard cardinality cap."""

    def __init__(self, cap: int = _DIMENSION_CAP) -> None:
        self._cap = cap
        self._counts: dict[str, int] = {}
        self._overflow = 0

    def add(self, key: str, amount: int = 1) -> None:
        if key in self._counts:
            self._counts[key] += amount
        elif len(self._counts) < self._cap:
            self._counts[key] = amount
        else:
            self._overflow += amount

    def render(self) -> dict[str, int]:
        out = dict(self._counts)
        if self._overflow:
            out[_OTHER] = out.get(_OTHER, 0) + self._overflow
        return out

    def __len__(self) -> int:
        return len(self._counts)


class MetricsRegistry:
    """Thread-safe, bounded, process-local counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_in_flight = 0
        self.request_duration_ms_count = 0
        self.request_duration_ms_sum = 0.0
        self.requests_by_status_class = _BoundedDimension()

        self.optimize_started_total = 0
        self.optimize_succeeded_total = 0
        self.optimize_failed_total = 0
        self.optimize_in_flight = 0
        self.optimize_by_execution_mode = _BoundedDimension()
        self.optimize_by_execution_reason = _BoundedDimension()
        self.tiles_processed_total = 0
        self.forward_pixels_total = 0
        self.screened_pixels_total = 0

        self.jobs_created_total = 0
        self.jobs_succeeded_total = 0
        self.jobs_failed_total = 0
        self.jobs_cancelled_total = 0

    # -- requests --------------------------------------------------------

    def request_started(self) -> None:
        with self._lock:
            self.requests_in_flight += 1

    def request_completed(self, status_code: int, duration_ms: float) -> None:
        with self._lock:
            # Clamp at zero: a defensive floor keeps a misbalanced caller
            # from ever publishing a negative gauge.
            self.requests_in_flight = max(0, self.requests_in_flight - 1)
            self.requests_total += 1
            self.request_duration_ms_count += 1
            self.request_duration_ms_sum += duration_ms
            self.requests_by_status_class.add(_status_class(status_code))

    # -- optimize ---------------------------------------------------------

    def optimize_started(self) -> None:
        with self._lock:
            self.optimize_started_total += 1
            self.optimize_in_flight += 1

    def optimize_completed(
        self,
        *,
        execution_mode: str,
        execution_reason: str,
        input_backend: str,
        output_backend: str,
        tiles: int,
        forward_pixels: int = 0,
        screened_pixels: int = 0,
        outcome: str = "succeeded",
    ) -> None:
        """Record one optimize outcome. Only bounded dimensions are keyed;
        everything identifier-shaped must go through :func:`emit_event`."""
        with self._lock:
            self.optimize_in_flight = max(0, self.optimize_in_flight - 1)
            if outcome == "succeeded":
                self.optimize_succeeded_total += 1
            else:
                self.optimize_failed_total += 1
            self.optimize_by_execution_mode.add(execution_mode)
            self.optimize_by_execution_reason.add(execution_reason)
            self.tiles_processed_total += tiles
            self.forward_pixels_total += forward_pixels
            self.screened_pixels_total += screened_pixels

    # -- jobs ---------------------------------------------------------------

    def job_created(self) -> None:
        with self._lock:
            self.jobs_created_total += 1

    def job_terminal(self, status: str) -> None:
        with self._lock:
            if status == "succeeded":
                self.jobs_succeeded_total += 1
            elif status == "failed":
                self.jobs_failed_total += 1
            elif status == "cancelled":
                self.jobs_cancelled_total += 1

    # -- rendering -----------------------------------------------------------

    def render(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": {
                    "requests_total": self.requests_total,
                    "requests_in_flight": self.requests_in_flight,
                    "requests_by_status_class": self.requests_by_status_class.render(),
                    "request_duration_ms_count": self.request_duration_ms_count,
                    "request_duration_ms_sum": round(self.request_duration_ms_sum, 3),
                },
                "optimize": {
                    "optimize_started_total": self.optimize_started_total,
                    "optimize_succeeded_total": self.optimize_succeeded_total,
                    "optimize_failed_total": self.optimize_failed_total,
                    "optimize_in_flight": self.optimize_in_flight,
                    "optimize_by_execution_mode": self.optimize_by_execution_mode.render(),
                    "optimize_by_execution_reason": self.optimize_by_execution_reason.render(),
                    "tiles_processed_total": self.tiles_processed_total,
                    "forward_pixels_total": self.forward_pixels_total,
                    "screened_pixels_total": self.screened_pixels_total,
                },
                "jobs": {
                    "jobs_created_total": self.jobs_created_total,
                    "jobs_succeeded_total": self.jobs_succeeded_total,
                    "jobs_failed_total": self.jobs_failed_total,
                    "jobs_cancelled_total": self.jobs_cancelled_total,
                },
            }

    def reset(self) -> None:
        """Test-only: restore pristine counters (process-local scope)."""
        with self._lock:
            self.__init__()  # type: ignore[misc]


registry = MetricsRegistry()
"""Process-local singleton. Non-durable by definition; never ship this
across processes or persist it."""


def emit_event(event: str, **fields: Any) -> None:
    """Emit one structured event on the events log channel.

    Events carry identifiers (request_id/job_id) at *log* cardinality —
    never as metric labels. Callers must pass only bounded, non-secret
    fields (no API keys, no uploaded content, no filesystem paths).
    """
    if event not in EVENTS:
        raise ValueError(f"unknown observability event {event!r}; vocabulary is frozen")
    payload = {"event": event, "utc": _utc_now(), "api_schema_version": API_SCHEMA_VERSION}
    payload.update(fields)
    EVENT_LOGGER.info("%s", json.dumps(payload, default=str))
