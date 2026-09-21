#!/usr/bin/env python3
"""Write the release build-identity module (audit P0.9/P2.2).

The publish workflow (and package-smoke) call this BEFORE building the
wheel so the released bytes carry their own build identity without any
YAML here-doc quoting.  The generated module is valid importable Python
and is byte-compiled-checked in CI.

Usage:
    python scripts/write_build_info.py \
        --commit $GITHUB_SHA \
        --run-id $GITHUB_RUN_ID \
        --version $VERSION \
        --out src/openlithohub/_build.py
"""

from __future__ import annotations

import argparse
import py_compile
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE = """# Generated file — DO NOT EDIT.
# Written by scripts/write_build_info.py at release/build time.

BUILD_COMMIT = {commit!r}
BUILD_TIMESTAMP = {timestamp!r}
BUILD_RUN_ID = {run_id!r}
BUILD_VERSION = {version!r}
"""


def write_build_info(
    *,
    commit: str,
    run_id: str,
    version: str,
    out: Path,
    timestamp: str | None = None,
) -> Path:
    """Generate the _build module and byte-compile-check it."""
    if not commit:
        raise ValueError("commit is required for a build identity")
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        TEMPLATE.format(commit=commit, timestamp=ts, run_id=run_id, version=version),
        encoding="utf-8",
    )
    py_compile.compile(str(out), doraise=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--out", type=Path, default=Path("src/openlithohub/_build.py"))
    args = ap.parse_args()
    path = write_build_info(
        commit=args.commit,
        run_id=args.run_id,
        version=args.version,
        out=args.out,
        timestamp=args.timestamp,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
