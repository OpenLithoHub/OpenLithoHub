"""Typed server configuration (repair-plan P1.5).

All server environment parsing lives here and happens at
``create_app()`` / ``ServerConfig.from_env()`` call time — never at
import time — so one Python process can build several independent app
configurations (tests, embedded use) without them leaking into each
other.

Scratch-directory selection stays in :mod:`openlithohub.server.app`
(``OLH_SCRATCH_DIR`` at call time) until the durable-job backend lands;
adding it here now would create a second source of truth.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# The only job backend that exists today. Its state is process-local, so
# the service must run with a single Uvicorn worker process (repair-plan
# P0.3): a job created in one worker is invisible to the others.
SUPPORTED_JOB_BACKENDS: tuple[str, ...] = ("in-memory",)

_DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _int_from_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_from_env(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class ServerConfig:
    """Validated configuration for one server runtime / app instance.

    Construct directly (tests, embedded use) or via :meth:`from_env`.
    All bounds are enforced here so a bad environment fails fast at app
    construction instead of misbehaving mid-request.
    """

    max_concurrent_optimize: int = 2
    job_queue_depth: int = 8
    job_history_cap: int = 100
    job_ttl_seconds: float = 3600.0
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES
    shutdown_grace_seconds: float = 5.0
    api_key: str = ""
    job_backend: str = "in-memory"

    def __post_init__(self) -> None:
        if self.max_concurrent_optimize < 1:
            raise ValueError("max_concurrent_optimize must be >= 1")
        if self.job_queue_depth < 1:
            raise ValueError("job_queue_depth must be >= 1")
        if self.job_history_cap < 1:
            raise ValueError("job_history_cap must be >= 1")
        if self.job_ttl_seconds < 0:
            raise ValueError("job_ttl_seconds must be >= 0")
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be > 0")
        if self.shutdown_grace_seconds < 0:
            raise ValueError("shutdown_grace_seconds must be >= 0")
        if self.job_backend not in SUPPORTED_JOB_BACKENDS:
            supported = ", ".join(repr(b) for b in SUPPORTED_JOB_BACKENDS)
            raise ValueError(
                f"unsupported job backend {self.job_backend!r}; supported: {supported}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServerConfig:
        """Parse and validate the server environment. Parsed at app
        construction time, never at import time (P1.5)."""
        env = os.environ if env is None else env
        return cls(
            max_concurrent_optimize=_int_from_env(env, "OPENLITHOHUB_MAX_CONCURRENT_OPTIMIZE", 2),
            job_queue_depth=_int_from_env(env, "OPENLITHOHUB_JOB_QUEUE_DEPTH", 8),
            job_history_cap=_int_from_env(env, "OPENLITHOHUB_JOB_HISTORY_CAP", 100),
            job_ttl_seconds=_float_from_env(env, "OPENLITHOHUB_JOB_TTL_SECONDS", 3600.0),
            max_upload_bytes=_int_from_env(
                env, "OPENLITHOHUB_MAX_UPLOAD_BYTES", _DEFAULT_MAX_UPLOAD_BYTES
            ),
            shutdown_grace_seconds=_float_from_env(env, "OPENLITHOHUB_SHUTDOWN_GRACE_SECONDS", 5.0),
            api_key=env.get("OPENLITHOHUB_API_KEY") or "",
            job_backend=env.get("OPENLITHOHUB_JOB_BACKEND") or "in-memory",
        )
