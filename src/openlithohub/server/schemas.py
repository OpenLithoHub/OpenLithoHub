"""PR-D public API contract — typed response schemas, error codes, and the
job-state machine.

This module is the machine-contract authority for the HTTP surface:

* ``API_SCHEMA_VERSION`` is the single schema-version constant; handlers
  and runtime code must never hardcode a literal ``"1"``.
* :class:`ErrorCode` is the finite, machine-readable error vocabulary.
  Clients key on ``error.code``, never on English ``detail`` text.
* :class:`JobStatus` owns the public job-state vocabulary and
  :func:`transition_job_status` is the single checked transition helper —
  illegal transitions (terminal resurrection, skipped states) raise
  :class:`IllegalJobTransitionError` instead of corrupting state silently.
* The Pydantic models below freeze the response bodies. ``extra="forbid"``
  on the payload models makes accidental field drift fail in tests rather
  than leak into the OpenAPI surface.

Compatibility policy: see ``docs/server-api-contract.md``. Removing or
renaming a field here is a REVIEW REQUIRED change and must update the
OpenAPI snapshot (``docs/api/openapi-v1.json``) in the same PR.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

API_SCHEMA_VERSION = "1"
"""Single schema-version authority for every public response body and the
OpenAPI document. Do not scatter literal versions through handlers."""


class ErrorCode(str, Enum):
    """Finite public error vocabulary (PR-D §3). Clients key on this."""

    INVALID_REQUEST = "INVALID_REQUEST"
    UNKNOWN_MODEL = "UNKNOWN_MODEL"
    UNKNOWN_JOB = "UNKNOWN_JOB"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    QUEUE_FULL = "QUEUE_FULL"
    ADMISSION_FULL = "ADMISSION_FULL"
    SERVER_NOT_ACCEPTING = "SERVER_NOT_ACCEPTING"
    JOB_ARTIFACT_UNAVAILABLE = "JOB_ARTIFACT_UNAVAILABLE"
    JOB_RUNNING = "JOB_RUNNING"
    STREAMING_UNSUPPORTED = "STREAMING_UNSUPPORTED"
    MISSING_OPTIONAL_DEPENDENCY = "MISSING_OPTIONAL_DEPENDENCY"
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class JobStatus(str, Enum):
    """Public job-state vocabulary (PR-D §5).

    Values are the wire format. ``RUNNING -> CANCELLED`` is deliberately
    absent: the PR-A runtime drains already-admitted work and does not own
    cancellation of an executing model, so the contract must not claim it.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


JOB_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}
"""The complete legal-transition table. Terminal states have no outgoing
edges; a cancelled job can never succeed, a failed job can never requeue."""


class IllegalJobTransitionError(RuntimeError):
    """Raised by :func:`transition_job_status` for an illegal transition."""


def transition_job_status(current: JobStatus, new: JobStatus) -> JobStatus:
    """The single checked job-status transition helper.

    Returns ``new`` when legal; raises :class:`IllegalJobTransitionError`
    otherwise (fail loudly internally rather than resurrecting a terminal
    job). Values that are plain strings (legacy records) are coerced.
    """
    current = JobStatus(current)
    new = JobStatus(new)
    if new not in JOB_TRANSITIONS[current]:
        raise IllegalJobTransitionError(
            f"illegal job transition {current.value!r} -> {new.value!r}"
        )
    return new


# ---- response schemas ------------------------------------------------------


class _ContractModel(BaseModel):
    """Base for response payloads: unknown fields are a contract error, not
    a round-trip opportunity."""

    model_config = ConfigDict(extra="forbid")


class ApiError(_ContractModel):
    """Machine-readable error payload. ``code`` is the stable key; the
    human-readable ``detail`` next to it may change wording at any time."""

    code: ErrorCode
    message: str
    request_id: str | None = Field(
        default=None, description="Correlated X-Request-ID of the failed request."
    )


class ErrorResponse(_ContractModel):
    """Error envelope. The top-level ``detail`` is kept verbatim for
    FastAPI/backward compatibility during this stabilization PR; clients
    should key on ``error.code``."""

    api_schema_version: str = API_SCHEMA_VERSION
    detail: Any = None
    error: ApiError


class HealthResponse(_ContractModel):
    status: Literal["ok"]


class ReadyChecks(_ContractModel):
    models_registered: bool
    scratch_writable: bool
    accepting_requests: bool


class ReadyResponse(_ContractModel):
    ready: bool
    checks: ReadyChecks
    metrics: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


class VersionResponse(_ContractModel):
    api: str
    package: str
    version: str
    git_commit: str
    build: dict[str, str]
    torch: str


class MemoryPolicyContract(_ContractModel):
    env: str
    default_bytes: int
    unlimited_marker: int
    fail_closed: bool


class StreamingMatrixContract(_ContractModel):
    engine: bool
    product_execution: bool
    inputs: dict[str, bool]
    outputs: dict[str, bool]
    scheduling_authority: str
    memory_policy: MemoryPolicyContract
    conditions: dict[str, str]


class JobsCapability(_ContractModel):
    backend: str
    durable: bool
    restart_loses_jobs: bool
    single_process_only: bool


class CapabilitiesResponse(_ContractModel):
    api_schema_version: str
    capability_schema_version: str
    models: list[str]
    simulator_backends: list[str]
    gpu: dict[str, Any]
    streaming_pipeline: bool
    streaming: StreamingMatrixContract
    input_formats: list[str]
    export_formats: list[str]
    jobs: JobsCapability
    proof_verification: dict[str, Any]
    build_provenance: dict[str, Any]


class WorkCounters(_ContractModel):
    """Stable, bounded subset of the streaming work-accounting ledger.

    These are the only counters the public contract commits to; the full
    internal ledger may grow freely.
    """

    forward_pixels: int | None = None
    read_pixels: int | None = None
    screened_pixels: int | None = None


class OptimizeMetadata(_ContractModel):
    """Typed public projection of one optimize run (PR-C vocabulary)."""

    shape: list[int]
    tiles: int
    halo_px: int
    writer: str
    export_format: str
    execution_mode: str
    execution_reason: str
    input_backend: str
    output_backend: str
    threshold: float
    estimated_dense_bytes: int | None = None
    max_dense_bytes: int | None = None
    work: WorkCounters | None = None


_METADATA_FIELDS = frozenset(OptimizeMetadata.model_fields.keys())


def project_optimize_metadata(summary: dict[str, Any] | None) -> OptimizeMetadata | None:
    """Project an internal runner summary onto the public contract.

    Drops private fields (``output_path`` and anything the runner adds
    later) instead of dumping arbitrary dictionaries into the API; unknown
    internal keys can never leak because the model forbids extras.
    """
    if summary is None:
        return None
    payload = {k: summary[k] for k in _METADATA_FIELDS if k in summary}
    return OptimizeMetadata.model_validate(payload)


class JobCreateResponse(_ContractModel):
    api_schema_version: str = API_SCHEMA_VERSION
    job_id: str
    status: JobStatus
    poll: str


class JobStatusResponse(_ContractModel):
    api_schema_version: str = API_SCHEMA_VERSION
    job_id: str
    status: JobStatus
    created_utc: str
    started_utc: str | None = None
    completed_utc: str | None = None
    summary: OptimizeMetadata | None = None
    error: str | None = None


class JobDeleteResponse(_ContractModel):
    deleted: str


class MetricsResponse(_ContractModel):
    api_schema_version: str = API_SCHEMA_VERSION
    scope: Literal["process"] = "process"
    runtime: dict[str, Any] = Field(default_factory=dict)
    requests: dict[str, Any] = Field(default_factory=dict)
    optimize: dict[str, Any] = Field(default_factory=dict)
    jobs: dict[str, Any] = Field(default_factory=dict)
