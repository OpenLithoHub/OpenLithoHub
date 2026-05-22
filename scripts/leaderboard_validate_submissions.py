"""Validate community leaderboard submission YAMLs against the schema.

Reads ``_pr_submissions/submissions/**/*.y*ml`` (the layout produced by the
auto-leaderboard workflow's contents-API fetch step), validates each one
against :class:`openlithohub.leaderboard.schema.BenchmarkResult`, and writes
the validated dump to ``_validated.json`` for the next workflow step.

Lives in ``scripts/`` so ruff / mypy / pytest can cover it — earlier this
logic lived inline in a workflow heredoc and was untestable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from openlithohub.leaderboard.schema import BenchmarkResult
from openlithohub.leaderboard.tracker import _require_forward_simulation


def main(submissions_root: Path, output: Path) -> int:
    subs = sorted(submissions_root.rglob("*.y*ml"))
    if not subs:
        print(f"::error::No submission files found under {submissions_root}/.")
        return 1

    out: list[tuple[str, dict[str, object]]] = []
    failures: list[tuple[Path, str]] = []
    # Validate every file before deciding to fail. A submitter pushing a
    # batch PR with N broken submissions used to fix-and-rebase N times
    # (workflow exited on the first error); collecting all errors lets
    # them rev once. Output is only written when every file validated.
    for p in subs:
        try:
            data = yaml.safe_load(p.read_text())
            result = BenchmarkResult.model_validate(data)
            # Forward-sim gate: reject mask-only submissions that would
            # let an Identity model trivially score 0 on mask-level EPE.
            _require_forward_simulation(result)
        except Exception as e:  # noqa: BLE001 — surface as workflow error
            print(f"::error file={p}::Schema validation failed: {e}")
            failures.append((p, str(e)))
            continue
        out.append((str(p), result.model_dump(mode="json")))
        print(f"OK: {p} -> {result.model_name} ({result.dataset}, {result.process_node.value})")

    if failures:
        print(
            f"::error::{len(failures)} of {len(subs)} submission file(s) failed validation; "
            "see annotations above."
        )
        return 1

    output.write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    submissions_root = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_pr_submissions/submissions")
    )
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("_validated.json")
    sys.exit(main(submissions_root, output))
