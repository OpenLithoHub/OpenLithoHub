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


def main(submissions_root: Path, output: Path) -> int:
    subs = sorted(submissions_root.rglob("*.y*ml"))
    if not subs:
        print(f"::error::No submission files found under {submissions_root}/.")
        return 1

    out: list[tuple[str, dict[str, object]]] = []
    for p in subs:
        data = yaml.safe_load(p.read_text())
        try:
            result = BenchmarkResult.model_validate(data)
        except Exception as e:  # noqa: BLE001 — surface as workflow error
            print(f"::error file={p}::Schema validation failed: {e}")
            return 1
        out.append((str(p), result.model_dump(mode="json")))
        print(f"OK: {p} -> {result.model_name} ({result.dataset}, {result.process_node.value})")

    output.write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    submissions_root = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_pr_submissions/submissions")
    )
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("_validated.json")
    sys.exit(main(submissions_root, output))
