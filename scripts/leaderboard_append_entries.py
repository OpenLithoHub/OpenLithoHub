"""Append validated leaderboard entries to the canonical store.

Reads the JSON file produced by :mod:`leaderboard_validate_submissions`
(default ``_validated.json``), re-instantiates each ``BenchmarkResult``, and
inserts them into ``leaderboard/leaderboard.json`` via
:func:`openlithohub.leaderboard.tracker.submit_result`.

Lives in ``scripts/`` so ruff / mypy / pytest can cover it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openlithohub.leaderboard.schema import BenchmarkResult
from openlithohub.leaderboard.tracker import LeaderboardStore, submit_result


def main(validated: Path, store_path: Path) -> int:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = LeaderboardStore(store_path)
    for path, payload in json.loads(validated.read_text()):
        result = BenchmarkResult.model_validate(payload)
        sid = submit_result(result, store=store)
        print(f"  inserted {sid} <- {path}")
    return 0


if __name__ == "__main__":
    validated = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_validated.json")
    store_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("leaderboard/leaderboard.json")
    sys.exit(main(validated, store_path))
