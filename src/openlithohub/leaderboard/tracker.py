"""SOTA tracking and leaderboard management."""

from __future__ import annotations

from openlithohub.leaderboard.schema import BenchmarkResult


def submit_result(result: BenchmarkResult) -> str:
    """Submit a benchmark result to the leaderboard.

    Args:
        result: Validated BenchmarkResult entry.

    Returns:
        Submission ID for tracking.
    """
    raise NotImplementedError(
        "Leaderboard submission not yet implemented. "
        "Planned: validate result, serialize to JSON, "
        "append to leaderboard data store (initially a JSON file in repo, "
        "later a hosted API)."
    )


def get_leaderboard(
    dataset: str | None = None,
    process_node: str | None = None,
) -> list[BenchmarkResult]:
    """Retrieve current leaderboard entries with optional filtering.

    Args:
        dataset: Filter by dataset name.
        process_node: Filter by process node.

    Returns:
        Sorted list of BenchmarkResult entries.
    """
    raise NotImplementedError(
        "Leaderboard retrieval not yet implemented. "
        "Planned: read from leaderboard data store, "
        "filter and sort by EPE (ascending)."
    )
