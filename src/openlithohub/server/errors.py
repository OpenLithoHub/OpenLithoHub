"""PR-D centralized error mapping — one place owns exception →
(HTTP status, machine-readable code) translation.

The HTTP handlers raise :class:`ApiHTTPException` (or let a domain
exception propagate into the app-level mapper) so every error response
carries the stable :class:`~openlithohub.server.schemas.ErrorCode`
envelope while keeping the legacy top-level ``detail`` verbatim.

Tracebacks and internal paths never reach clients: messages surfaced in
envelopes are the exception's user-facing string only, and unexpected
exceptions collapse to ``INTERNAL_ERROR`` with a fixed message.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from openlithohub.server.schemas import API_SCHEMA_VERSION, ErrorCode


class ApiHTTPException(HTTPException):
    """HTTPException carrying the machine-readable error code.

    The app-level exception handler renders this as the PR-D envelope::

        {"api_schema_version": "1",
         "detail": <message>,
         "error": {"code": ..., "message": ..., "request_id": ...}}
    """

    def __init__(
        self,
        status_code: int,
        code: ErrorCode | str,
        message: str | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        code = ErrorCode(code)
        super().__init__(status_code=status_code, detail=message or code.value, headers=headers)
        self.code = code
        self.error_message = message or code.value


def error_envelope(
    *, code: ErrorCode | str, message: str, detail: Any = None, request_id: str | None = None
) -> dict[str, Any]:
    """Build the transitional error body (legacy ``detail`` preserved)."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "detail": detail if detail is not None else message,
        "error": {
            "code": ErrorCode(code).value,
            "message": message,
            "request_id": request_id,
        },
    }


def runtime_error_code(exc: Exception) -> tuple[int, ErrorCode]:
    """Map a domain/runtime exception to its (status, code) contract pair.

    Unknown exceptions are NOT mapped here — unexpected exceptions become
    ``500 INTERNAL_ERROR`` at the middleware boundary, never a guessed
    code with leaked internals.
    """
    from openlithohub.server.runtime import (
        AdmissionDeniedError,
        JobArtifactUnavailableError,
        JobQueueFullError,
        JobStillRunningError,
        RuntimeNotAcceptingWorkError,
        UnknownJobError,
    )
    from openlithohub.workflow.execution import StreamingUnsupportedError

    if isinstance(exc, JobQueueFullError):
        return 429, ErrorCode.QUEUE_FULL
    if isinstance(exc, AdmissionDeniedError):
        return 429, ErrorCode.ADMISSION_FULL
    if isinstance(exc, RuntimeNotAcceptingWorkError):
        return 503, ErrorCode.SERVER_NOT_ACCEPTING
    if isinstance(exc, UnknownJobError):
        return 404, ErrorCode.UNKNOWN_JOB
    if isinstance(exc, JobArtifactUnavailableError):
        return 409, ErrorCode.JOB_ARTIFACT_UNAVAILABLE
    if isinstance(exc, JobStillRunningError):
        return 409, ErrorCode.JOB_RUNNING
    if isinstance(exc, StreamingUnsupportedError):
        return 400, ErrorCode.STREAMING_UNSUPPORTED
    if isinstance(exc, FileNotFoundError):
        return 400, ErrorCode.INVALID_REQUEST
    if isinstance(exc, ImportError):
        return 503, ErrorCode.MISSING_OPTIONAL_DEPENDENCY
    if isinstance(exc, KeyError):
        # Ambiguous by nature; disambiguate by message. Unknown model names
        # and unknown process nodes are the only KeyErrors the runner
        # legitimately raises.
        message = str(exc).strip("'\"").lower()
        if "process node" in message:
            return 400, ErrorCode.INVALID_REQUEST
        return 404, ErrorCode.UNKNOWN_MODEL
    if isinstance(exc, ValueError):
        return 400, ErrorCode.INVALID_REQUEST
    return 500, ErrorCode.INTERNAL_ERROR
