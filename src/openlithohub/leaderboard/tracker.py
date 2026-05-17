"""SOTA tracking and leaderboard management."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from openlithohub.leaderboard.schema import BenchmarkResult

_DEFAULT_LEADERBOARD_DIR = Path.home() / ".openlithohub"
_LEADERBOARD_FILENAME = "leaderboard.json"


class LeaderboardStore:
    """JSON file-backed leaderboard data store."""

    def __init__(self, path: Path | None = None) -> None:
        if path is not None:
            self._path = Path(path)
        else:
            env_path = os.environ.get("OPENLITHOHUB_LEADERBOARD_PATH")
            if env_path:
                self._path = Path(env_path)
            else:
                self._path = _DEFAULT_LEADERBOARD_DIR / _LEADERBOARD_FILENAME

    @property
    def path(self) -> Path:
        return self._path

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        text = self._path.read_text(encoding="utf-8")
        data = json.loads(text)
        return data.get("entries", [])  # type: ignore[no-any-return]

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        payload = json.dumps({"entries": entries}, indent=2, default=str)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self._path)

    def submit(self, result: BenchmarkResult) -> str:
        entries = self._read_entries()
        submission_id = _generate_id(result.model_name)
        entry = result.model_dump(mode="json")
        entry["_submission_id"] = submission_id
        entries.append(entry)
        self._write_entries(entries)
        return submission_id

    def query(
        self,
        dataset: str | None = None,
        process_node: str | None = None,
    ) -> list[BenchmarkResult]:
        entries = self._read_entries()
        results: list[BenchmarkResult] = []
        for entry in entries:
            entry_copy = {k: v for k, v in entry.items() if not k.startswith("_")}
            r = BenchmarkResult.model_validate(entry_copy)
            if dataset and r.dataset != dataset:
                continue
            if process_node and r.process_node.value != process_node:
                continue
            results.append(r)
        results.sort(key=lambda r: r.epe_mean_nm)
        return results


def _generate_id(model_name: str) -> str:
    ts_hex = f"{int(time.time() * 1000):x}"[-8:]
    safe_name = model_name.replace(" ", "-").lower()[:20]
    return f"{safe_name}-{ts_hex}"


_default_store: LeaderboardStore | None = None


def _get_store() -> LeaderboardStore:
    global _default_store  # noqa: PLW0603
    if _default_store is None:
        _default_store = LeaderboardStore()
    return _default_store


def submit_result(result: BenchmarkResult, *, store: LeaderboardStore | None = None) -> str:
    """Submit a benchmark result to the leaderboard.

    Args:
        result: Validated BenchmarkResult entry.
        store: Optional explicit store (for testing). Uses default if None.

    Returns:
        Submission ID for tracking.
    """
    s = store or _get_store()
    return s.submit(result)


def get_leaderboard(
    dataset: str | None = None,
    process_node: str | None = None,
    *,
    store: LeaderboardStore | None = None,
) -> list[BenchmarkResult]:
    """Retrieve current leaderboard entries with optional filtering.

    Args:
        dataset: Filter by dataset name.
        process_node: Filter by process node.
        store: Optional explicit store (for testing). Uses default if None.

    Returns:
        Sorted list of BenchmarkResult entries (by EPE ascending).
    """
    s = store or _get_store()
    return s.query(dataset=dataset, process_node=process_node)
