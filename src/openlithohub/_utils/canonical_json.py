"""Canonical artifact bytes (GPU Authority Repair v3, §24).

One canonical strict-JSON writer for every artifact the formal
measurement harnesses emit.  The byte contract is platform-independent:

* UTF-8
* LF line endings only — never ``os.linesep`` (the Windows CRLF drift
  observed during Scale qualification cannot be produced by this writer)
* one stable trailing newline
* strict JSON (``NaN``/``Infinity`` rejected at write time)
* sorted keys, stable indentation

Every authority artifact written through this helper is byte-stable
across platforms and regeneration runs (§25).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["canonical_json_bytes", "write_canonical_json"]


def canonical_json_bytes(payload: Any) -> bytes:
    """Canonical strict-JSON bytes: UTF-8, LF only, stable trailing
    newline, sorted keys, ``allow_nan=False``."""
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return text.encode("utf-8")


def write_canonical_json(path: str | Path, payload: Any) -> None:
    """Write canonical artifact bytes (byte-based — never text mode, so a
    Windows host cannot introduce CRLF)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(payload))
