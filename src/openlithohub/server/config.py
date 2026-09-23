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

# PR-E: "in-memory" keeps the historical lightweight mode (durable=False,
# restart loses jobs); "sqlite" persists committed jobs + artifacts under
# OPENLITHOHUB_STATE_DIR (durable=True). BOTH remain single-process: a job
# created in one Uvicorn worker is invisible to the others, and the SQLite
# store takes an exclusive per-directory ownership lock, so the service
# must still run with exactly one worker process (repair-plan P0.3).
SUPPORTED_JOB_BACKENDS: tuple[str, ...] = ("in-memory", "sqlite")

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
    state_dir: str | None = None
    """Durable async-job state root (OPENLITHOHUB_STATE_DIR); required for
    the sqlite backend and ignored by in-memory."""

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
        if self.job_backend == "sqlite" and not self.state_dir:
            raise ValueError(
                "OPENLITHOHUB_JOB_BACKEND=sqlite requires OPENLITHOHUB_STATE_DIR "
                "to point at a persistent directory; durability without "
                "persistent storage is not claimed"
            )
        if self.job_backend == "in-memory" and self.state_dir is not None:
            raise ValueError(
                "state_dir only applies to the sqlite job backend; the "
                "in-memory backend is honest about being non-durable"
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
            state_dir=env.get("OPENLITHOHUB_STATE_DIR") or None,
        )
