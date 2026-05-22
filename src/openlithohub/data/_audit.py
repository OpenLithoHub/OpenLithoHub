"""Per-call audit log for ``DatasetAdapter.download`` invocations.

When ``OPENLITHOHUB_AUDIT_DIR`` is set to a writable directory, every
``DatasetAdapter.download`` call appends one JSONL record to
``<OPENLITHOHUB_AUDIT_DIR>/<adapter>.jsonl`` summarising the attempt:

- ``timestamp`` — ISO-8601 UTC, second resolution
- ``adapter`` — fully-qualified class name
- ``root`` — destination directory passed to ``download``
- ``args`` / ``kwargs`` — extra positional/keyword payload (best-effort,
  stringified — large objects are truncated)
- ``outcome`` — ``"success"``, ``"error"``
- ``error_class`` / ``error_message`` — present only on ``error``
- ``elapsed_ms`` — wall-clock call duration
- ``size_bytes`` — best-effort total size of files under ``root`` after
  the call returned (``None`` on error or missing directory)

The schema mirrors ``acquisition_log.md`` rows in spirit (timestamp,
status, sizes) but trades the prose narrative for a strict
machine-readable shape so external CI / docs jobs can ingest it.

When the env var is unset, the wrapper is a no-op — adapters remain
free to call ``download`` without any I/O side-effect.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_AUDIT_ENV_VAR = "OPENLITHOHUB_AUDIT_DIR"
_MAX_REPR_LEN = 200


def _audit_dir() -> Path | None:
    raw = os.environ.get(_AUDIT_ENV_VAR)
    if not raw:
        return None
    p = Path(raw)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return p


def _safe_repr(value: Any) -> str:
    try:
        s = repr(value)
    except Exception:  # noqa: BLE001 — best-effort
        s = f"<unrepr {type(value).__name__}>"
    if len(s) > _MAX_REPR_LEN:
        s = s[: _MAX_REPR_LEN - 3] + "..."
    return s


def _dir_size(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return None


def wrap_download(adapter_cls: type) -> None:
    """Patch ``adapter_cls.download`` to emit an audit record per call.

    The wrapper is a no-op when ``OPENLITHOHUB_AUDIT_DIR`` is unset, so
    production callers pay only an env-var read per ``download`` call.
    A double-wrap is detected via the ``__openlithohub_audited`` marker
    so subclass chains don't stack records.
    """
    original = adapter_cls.__dict__.get("download")
    if original is None:
        return  # subclass inherits download from a parent — already wrapped there
    if getattr(original, "__openlithohub_audited", False):
        return

    def wrapped(self: Any, root: str, *args: Any, **kwargs: Any) -> Any:
        audit_dir = _audit_dir()
        if audit_dir is None:
            return original(self, root, *args, **kwargs)

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "adapter": f"{adapter_cls.__module__}.{adapter_cls.__qualname__}",
            "root": str(root),
            "args": [_safe_repr(a) for a in args],
            "kwargs": {k: _safe_repr(v) for k, v in kwargs.items()},
        }
        start = time.monotonic()
        try:
            result = original(self, root, *args, **kwargs)
        except BaseException as exc:
            record["outcome"] = "error"
            record["error_class"] = type(exc).__name__
            record["error_message"] = _safe_repr(str(exc))
            record["elapsed_ms"] = int((time.monotonic() - start) * 1000)
            record["size_bytes"] = _dir_size(Path(root))
            _append_record(audit_dir, adapter_cls, record)
            raise
        else:
            record["outcome"] = "success"
            record["elapsed_ms"] = int((time.monotonic() - start) * 1000)
            record["size_bytes"] = _dir_size(Path(root))
            _append_record(audit_dir, adapter_cls, record)
            return result

    wrapped.__openlithohub_audited = True  # type: ignore[attr-defined]
    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    adapter_cls.download = wrapped  # type: ignore[attr-defined]


def _append_record(audit_dir: Path, adapter_cls: type, record: dict[str, Any]) -> None:
    fname = f"{adapter_cls.__name__}.jsonl"
    target = audit_dir / fname
    try:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str))
            fh.write("\n")
    except OSError:
        # Auditing must never break the underlying download — swallow
        # filesystem errors silently. The caller learns about real
        # problems via the original return value / exception.
        return


def install_audit_hook(adapter_cls: type) -> Callable[[type], type]:
    """Hook used by :class:`DatasetAdapter.__init_subclass__`.

    Exposed as a small named entry point so the inheritance hook in
    ``base.py`` doesn't accumulate logic. Apply once per concrete
    subclass that defines ``download``.
    """
    wrap_download(adapter_cls)
    return adapter_cls
