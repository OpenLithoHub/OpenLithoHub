"""PR-F — the OpenLithoHub Python client: a thin, typed, sync SDK.

Design contract (PR-F):

* **One contract source.** The client re-uses ``openlithohub.server.schemas``
  — the very models behind the server's ``response_model`` declarations and
  the OpenAPI v1 snapshot — as its typed vocabulary. It never generates a
  second schema and never re-implements server logic. This import is
  hygiene-safe: ``openlithohub.server.schemas`` needs only pydantic (a core
  dependency), never the ``[server]`` extra — enforced by a hostile
  subprocess test that blocks ``fastapi``/``uvicorn``/``starlette``.
* **Thin.** The client owns exactly five things: HTTP transport, typed
  error mapping, job polling, atomic artifact download, and streamed
  multipart upload / download so large layouts never fully transit RAM.
* **Fail honest.** POST/PUT/DELETE are never retried. Every non-2xx
  response raises :class:`OpenLithoHubApiError` keyed on the machine
  ``error.code`` (PR-D vocabulary) with the correlated ``request_id``.
* **Sync only this round.** The server's long-task surface is already the
  async job API; the sync client does short requests plus local polling.
  An ``AsyncOpenLithoHubClient`` can be added purely additively later.

Requires the ``[client]`` extra (``httpx``)::

    pip install openlithohub[client]
"""

from __future__ import annotations

from openlithohub.client._client import (
    OpenLithoHubApiError,
    OpenLithoHubClient,
    OpenLithoHubConnectionError,
    OpenLithoHubError,
    OpenLithoHubJobTimeoutError,
    OptimizeResult,
)
from openlithohub.server.schemas import (
    API_SCHEMA_VERSION,
    CapabilitiesResponse,
    ErrorCode,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    MetricsResponse,
    OptimizeMetadata,
    ReadyResponse,
    VersionResponse,
)

__all__ = [
    "API_SCHEMA_VERSION",
    "CapabilitiesResponse",
    "ErrorCode",
    "HealthResponse",
    "JobStatus",
    "JobStatusResponse",
    "MetricsResponse",
    "OptimizeMetadata",
    "OptimizeResult",
    "OpenLithoHubApiError",
    "OpenLithoHubClient",
    "OpenLithoHubConnectionError",
    "OpenLithoHubError",
    "OpenLithoHubJobTimeoutError",
    "ReadyResponse",
    "VersionResponse",
]
