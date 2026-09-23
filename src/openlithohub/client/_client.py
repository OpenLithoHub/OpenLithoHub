"""The sync OpenLithoHub HTTP client (PR-F). See the package docstring for
the design contract: one schema source, thin responsibilities, no retries,
streamed transfer, typed errors keyed on the PR-D error codes."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from openlithohub.server.schemas import (
    API_SCHEMA_VERSION,
    CapabilitiesResponse,
    ErrorCode,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    MetricsResponse,
    ReadyResponse,
    VersionResponse,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

_UPLOAD_CHUNK = 1024 * 1024
_DOWNLOAD_CHUNK = 1024 * 1024

TerminalStatus = Literal["succeeded", "failed", "cancelled"]

_POLL_CALLBACK = Callable[["JobStatusResponse"], None]


class OpenLithoHubError(Exception):
    """Base class for every client-side failure."""


class OpenLithoHubConnectionError(OpenLithoHubError):
    """The server could not be reached (DNS, connect, timeout, reset).

    Never raised after a request reached the server — protocol-level
    failures use :class:`OpenLithoHubApiError`."""


class OpenLithoHubApiError(OpenLithoHubError):
    """A structured server error (PR-D envelope).

    Key on :attr:`code` (the machine-readable PR-D ``error.code``
    vocabulary) — never on :attr:`message` text.
    """

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        request_id: str | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(f"[{code}] {message} (HTTP {status_code})")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.detail = detail


class OpenLithoHubJobTimeoutError(OpenLithoHubError):
    """``wait_for_job`` exceeded its deadline; the job itself keeps
    running server-side and stays pollable."""

    def __init__(self, job_id: str, timeout: float) -> None:
        super().__init__(
            f"job {job_id!r} did not reach a terminal state within {timeout:.0f}s; "
            "it is still running server-side and can be polled"
        )
        self.job_id = job_id
        self.timeout = timeout


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of a synchronous optimize: the downloaded artifact plus the
    PR-C execution metadata projected from the contractual headers."""

    destination: Path
    request_id: str
    shape: tuple[int, int]
    tiles: int
    halo_px: int
    export_format: str
    execution_mode: str
    execution_reason: str
    input_backend: str
    output_backend: str

    @classmethod
    def from_headers(cls, destination: Path, request_id: str, headers: Any) -> OptimizeResult:
        required = (
            "X-OLH-Tiles",
            "X-OLH-Halo-Px",
            "X-OLH-Export-Format",
            "X-OLH-Shape",
            "X-OLH-Execution-Mode",
            "X-OLH-Execution-Reason",
            "X-OLH-Input-Backend",
            "X-OLH-Output-Backend",
        )
        missing = [name for name in required if name not in headers]
        if missing:
            raise OpenLithoHubError(
                "server response is missing contractual optimize headers: " + ", ".join(missing)
            )
        shape_raw = headers["X-OLH-Shape"].split("x")
        return cls(
            destination=destination,
            request_id=request_id,
            shape=(int(shape_raw[0]), int(shape_raw[1])),
            tiles=int(headers["X-OLH-Tiles"]),
            halo_px=int(headers["X-OLH-Halo-Px"]),
            export_format=headers["X-OLH-Export-Format"],
            execution_mode=headers["X-OLH-Execution-Mode"],
            execution_reason=headers["X-OLH-Execution-Reason"],
            input_backend=headers["X-OLH-Input-Backend"],
            output_backend=headers["X-OLH-Output-Backend"],
        )


class OpenLithoHubClient:
    """Thin synchronous client for one OpenLithoHub server.

    Args:
        base_url: Server root, e.g. ``http://localhost:8000``.
        api_key: Value for the ``X-API-Key`` header when the server
            requires one (``OPENLITHOHUB_API_KEY``).
        timeout: Per-request timeout in seconds (all phases).
        poll_interval: Seconds between job-status polls in
            :meth:`wait_for_job`.
        transport: Optional ``httpx.BaseTransport`` injection for tests
            (e.g. ``httpx.ASGITransport(app=...)``). Production callers
            leave this unset.

    Every request carries a generated ``X-Request-ID`` (PR-D §4
    correlation); failed requests surface it on the raised
    :class:`OpenLithoHubApiError`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        transport: Any = None,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover — [client] extra absent
            raise ImportError(
                "the OpenLithoHub client requires httpx. "
                "Install with: pip install openlithohub[client]"
            ) from exc
        self._httpx = httpx
        self._poll_interval = max(0.05, float(poll_interval))
        headers = {"X-Request-ID": uuid.uuid4().hex, "Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenLithoHubClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- request core ------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        ok_status: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """One request, one attempt — never retried (PR-F policy).

        Returns the httpx.Response when the status matches ``ok_status``
        (default: any 2xx); otherwise raises :class:`OpenLithoHubApiError`
        parsed from the PR-D error envelope.
        """
        self._client.headers["X-Request-ID"] = uuid.uuid4().hex
        try:
            response = self._client.request(method, path, **kwargs)
        except self._httpx.HTTPError as exc:
            raise OpenLithoHubConnectionError(
                f"{method} {path} failed before the server answered: {exc}"
            ) from exc
        expected = ok_status if ok_status is not None else 0
        if expected and response.status_code == expected:
            return response
        if 200 <= response.status_code < 300:
            return response
        raise self._api_error(response)

    def _api_error(self, response: Any) -> OpenLithoHubApiError:
        request_id: str | None = None
        code: str = "INTERNAL_ERROR" if response.status_code >= 500 else "INVALID_REQUEST"
        message = response.text or "request failed"
        detail: Any = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            detail = body.get("detail")
            error = body.get("error")
            if isinstance(error, dict):
                code = str(error.get("code", code))
                message = str(error.get("message", message))
                request_id = error.get("request_id")
            else:
                message = str(body.get("detail", message))
        request_id = request_id or response.headers.get("X-Request-ID")
        return OpenLithoHubApiError(
            status_code=response.status_code,
            code=code,
            message=message,
            request_id=request_id,
            detail=detail,
        )

    def _json(self, method: str, path: str, model: Any, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        return model.model_validate(response.json())

    def _json_model(self, method: str, path: str, model: type[ModelT], **kwargs: Any) -> ModelT:
        response = self._request(method, path, **kwargs)
        return model.model_validate(response.json())

    # ---- metadata endpoints -------------------------------------------------

    def health(self) -> HealthResponse:
        return self._json_model("GET", "/v1/health", HealthResponse, ok_status=200)

    def ready(self) -> ReadyResponse:
        return self._json_model("GET", "/v1/ready", ReadyResponse, ok_status=200)

    def version(self) -> VersionResponse:
        return self._json_model("GET", "/v1/version", VersionResponse, ok_status=200)

    def capabilities(self) -> CapabilitiesResponse:
        return self._json_model("GET", "/v1/capabilities", CapabilitiesResponse, ok_status=200)

    def metrics(self) -> MetricsResponse:
        return self._json_model("GET", "/v1/metrics", MetricsResponse, ok_status=200)

    def models(self) -> list[str]:
        response = self._request("GET", "/v1/models", ok_status=200)
        body = response.json()
        models = body.get("models")
        if not isinstance(models, list):
            raise OpenLithoHubError("GET /v1/models returned no models list")
        return [str(name) for name in models]

    # ---- synchronous optimize ------------------------------------------------

    def optimize(
        self,
        layout: str | Path,
        *,
        model: str,
        node: str = "3nm-euv",
        pixel_nm: float | None = None,
        tile_size: int = 2048,
        writer: str = "mbmw",
        layer: str | None = None,
        pretrained: bool = False,
        min_area_nm2: float = 0.0,
        execution_mode: str = "auto",
        destination: str | Path,
    ) -> OptimizeResult:
        """Run one synchronous optimize and stream the artifact to disk.

        The layout file is streamed from disk (multipart) and the response
        is streamed to ``destination`` in chunks — neither side ever holds
        the full bytes in RAM. ``destination`` is written via a temp file
        and an atomic rename. Requests are never retried; a
        :class:`OpenLithoHubApiError` (keyed on ``error.code`` — e.g.
        ``ADMISSION_FULL`` when the server is saturated, or
        ``STREAMING_UNSUPPORTED`` for an unsupported large job) is raised
        for every non-2xx response.
        """
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination_path.with_name(destination_path.name + ".part")
        layout_path = Path(layout)
        data: dict[str, str] = {
            "model": model,
            "node": node,
            "tile_size": str(tile_size),
            "writer": writer,
            "pretrained": "true" if pretrained else "false",
            "min_area_nm2": str(min_area_nm2),
            "execution_mode": execution_mode,
        }
        if pixel_nm is not None:
            data["pixel_nm"] = str(pixel_nm)
        if layer is not None:
            data["layer"] = layer

        self._client.headers["X-Request-ID"] = uuid.uuid4().hex
        try:
            with (
                layout_path.open("rb") as handle,
                self._client.stream(
                    "POST",
                    "/v1/optimize",
                    files={"layout": (layout_path.name, handle, "application/octet-stream")},
                    data=data,
                ) as response,
            ):
                if response.status_code != 200:
                    response.read()
                    raise self._api_error(response)
                request_id = response.headers.get("X-Request-ID")
                result = OptimizeResult.from_headers(
                    destination_path, request_id or "", response.headers
                )
                with tmp_destination.open("wb") as out:
                    for chunk in response.iter_bytes(_DOWNLOAD_CHUNK):
                        out.write(chunk)
            os.replace(tmp_destination, destination_path)
        except self._httpx.HTTPError as exc:
            tmp_destination.unlink(missing_ok=True)
            raise OpenLithoHubConnectionError(
                f"optimize request failed before the server answered: {exc}"
            ) from exc
        except BaseException:
            tmp_destination.unlink(missing_ok=True)
            raise
        return result

    # ---- async job API ---------------------------------------------------------

    def create_job(
        self,
        layout: str | Path,
        *,
        model: str,
        node: str = "3nm-euv",
        pixel_nm: float | None = None,
        tile_size: int = 2048,
        writer: str = "mbmw",
        layer: str | None = None,
        pretrained: bool = False,
        min_area_nm2: float = 0.0,
        execution_mode: str = "auto",
    ) -> str:
        """Submit one async optimize job; returns the ``job_id``.

        The upload is streamed from disk. A ``202`` response implies the
        durable contract of the running backend (``HTTP 202 -> committed
        job``); poll with :meth:`get_job` / :meth:`wait_for_job`.
        """
        layout_path = Path(layout)
        data: dict[str, str] = {
            "model": model,
            "node": node,
            "tile_size": str(tile_size),
            "writer": writer,
            "pretrained": "true" if pretrained else "false",
            "min_area_nm2": str(min_area_nm2),
            "execution_mode": execution_mode,
        }
        if pixel_nm is not None:
            data["pixel_nm"] = str(pixel_nm)
        if layer is not None:
            data["layer"] = layer
        with layout_path.open("rb") as handle:
            response = self._request(
                "POST",
                "/v1/jobs/optimize",
                ok_status=202,
                files={"layout": (layout_path.name, handle, "application/octet-stream")},
                data=data,
            )
        body = response.json()
        job_id = body.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise OpenLithoHubError("POST /v1/jobs/optimize returned no job_id")
        return job_id

    def get_job(self, job_id: str) -> JobStatusResponse:
        return self._json_model("GET", f"/v1/jobs/{job_id}", JobStatusResponse, ok_status=200)

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float | None = 3600.0,
        poll_interval: float | None = None,
        on_poll: _POLL_CALLBACK | None = None,
    ) -> JobStatusResponse:
        """Poll until the job reaches a terminal state.

        The snapshot is returned for ALL terminal states — check
        :attr:`JobStatusResponse.status` (and ``.error``) rather than
        assuming success. A job that was mid-flight during a server crash
        surfaces as ``failed`` with the server's restart-interruption
        error (PR-E recovery semantics); polling simply continues to work
        against the restarted server.

        Raises :class:`OpenLithoHubJobTimeoutError` when ``timeout``
        (non-``None``, default 3600.0) elapses first (the job keeps
        running server-side).
        """
        interval = self._poll_interval if poll_interval is None else max(0.05, poll_interval)
        effective_timeout = 3600.0 if timeout is None else timeout
        deadline: float | None = None if timeout is None else time.monotonic() + effective_timeout
        while True:
            snapshot = self.get_job(job_id)
            if snapshot.status in ("succeeded", "failed", "cancelled"):
                return snapshot
            if on_poll is not None:
                on_poll(snapshot)
            if deadline is not None and time.monotonic() >= deadline:
                raise OpenLithoHubJobTimeoutError(job_id, effective_timeout)
            time.sleep(interval)

    def download_artifact(
        self,
        job_id: str,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Stream a succeeded job's artifact to ``destination``.

        Written via a temp file + atomic rename (never a partial file at
        the final path, never fully resident in RAM). Requires
        ``overwrite=True`` when the destination already exists.
        """
        destination_path = Path(destination)
        if destination_path.exists() and not overwrite:
            raise FileExistsError(
                f"{destination_path} already exists; pass overwrite=True to replace it"
            )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination_path.with_name(destination_path.name + ".part")
        try:
            with self._client.stream("GET", f"/v1/jobs/{job_id}/artifact") as response:
                if response.status_code != 200:
                    response.read()
                    raise self._api_error(response)
                with tmp_destination.open("wb") as out:
                    for chunk in response.iter_bytes(_DOWNLOAD_CHUNK):
                        out.write(chunk)
            os.replace(tmp_destination, destination_path)
        except self._httpx.HTTPError as exc:
            tmp_destination.unlink(missing_ok=True)
            raise OpenLithoHubConnectionError(
                f"artifact download for {job_id!r} failed: {exc}"
            ) from exc
        except BaseException:
            tmp_destination.unlink(missing_ok=True)
            raise
        return destination_path

    def delete_job(self, job_id: str) -> str:
        response = self._request("DELETE", f"/v1/jobs/{job_id}", ok_status=200)
        return str(response.json().get("deleted", job_id))


# Literal alias used by type checkers for terminal states.
TERMINAL_STATUSES: tuple[TerminalStatus, ...] = ("succeeded", "failed", "cancelled")

__all__ = [
    "API_SCHEMA_VERSION",
    "ErrorCode",
    "JobStatus",
    "JobStatusResponse",
    "OptimizeResult",
    "OpenLithoHubApiError",
    "OpenLithoHubClient",
    "OpenLithoHubConnectionError",
    "OpenLithoHubError",
    "OpenLithoHubJobTimeoutError",
    "TERMINAL_STATUSES",
]
