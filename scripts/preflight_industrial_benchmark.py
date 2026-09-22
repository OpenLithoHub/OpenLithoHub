#!/usr/bin/env python3
"""Pre-rerun preflight for the Industrial Benchmark (audit convergence A1).

This is a trivial wrapper around the harness's ``--preflight-only`` flag.
It exists so operators have a memorable entry point; the REAL preparation
pipeline (clean tree, runtime code identity, model canonicalization,
fixture hashes, environment lock, run identity) lives in
``run_industrial_benchmark.py`` and is shared by ``--preflight-only``,
``--print-run-identity`` and the formal measurement.

Output on success: ``PRE-RERUN PREFLIGHT: PASS`` + ``RUN_IDENTITY: <hex>``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "benchmarks" / "industrial" / "run_industrial_benchmark.py"


def main() -> int:
    # Forward all arguments to the harness with --preflight-only prepended.
    cmd = [
        sys.executable,
        str(HARNESS),
        "--preflight-only",
        *sys.argv[1:],
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))  # noqa: S603
    if result.returncode == 0:
        print("PRE-RERUN PREFLIGHT: PASS")
    else:
        print("PRE-RERUN PREFLIGHT: FAIL", file=sys.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
