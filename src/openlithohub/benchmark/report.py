"""Evaluation report generation."""

from __future__ import annotations

from typing import Any


def generate_report(
    metrics: dict[str, Any],
    output_format: str = "table",
) -> str:
    """Generate a formatted evaluation report from computed metrics.

    Args:
        metrics: Dictionary of metric names to values.
        output_format: 'table' (rich terminal), 'json', or 'markdown'.

    Returns:
        Formatted report string.
    """
    raise NotImplementedError(
        "Report generation not yet implemented. "
        "Planned: use rich.table for terminal output, "
        "json.dumps for JSON, and markdown table formatting."
    )
