#!/usr/bin/env python3
"""Write the release build-identity module (audit P0.9/P2.2).

The publish workflow (and package-smoke) call this BEFORE building the
wheel so the released bytes carry their own build identity without any
YAML here-doc quoting.  The generated module is valid importable Python
and is byte-compiled-checked in CI.

P0.12: only deterministic fields (BUILD_COMMIT, BUILD_VERSION) go into
the wheel bytes.  BUILD_TIMESTAMP and BUILD_RUN_ID belong in external
provenance (GitHub attestation / PyPI provenance / OCI labels).

Usage:
    python scripts/write_build_info.py \
        --commit $GITHUB_SHA \
        --version $VERSION \
        --out src/openlithohub/_build.py
"""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

TEMPLATE = """# Generated file — DO NOT EDIT.
# Written by scripts/write_build_info.py at release/build time.
# P0.12: deterministic fields only — same commit + version = same wheel bytes.

BUILD_COMMIT = {commit!r}
BUILD_VERSION = {version!r}
"""


def write_build_info(*, commit: str, version: str, out: Path) -> Path:
    """Generate the _build module and byte-compile-check it."""
    if not commit:
        raise ValueError("commit is required for a build identity")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        TEMPLATE.format(commit=commit, version=version),
        encoding="utf-8",
    )
    py_compile.compile(str(out), doraise=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--out", type=Path, default=Path("src/openlithohub/_build.py"))
    args = ap.parse_args()
    path = write_build_info(commit=args.commit, version=args.version, out=args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
