"""Evaluation report generation."""

from __future__ import annotations

import json
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
    if output_format == "json":
        return json.dumps(metrics, indent=2)

    if output_format == "markdown":
        return _format_markdown(metrics)

    return _format_table(metrics)


def _format_table(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "(no metrics)"

    key_width = max(len(k) for k in metrics)
    lines = ["┌" + "─" * (key_width + 2) + "┬" + "─" * 16 + "┐"]
    lines.append("│ " + "Metric".ljust(key_width) + " │ " + "Value".ljust(14) + " │")
    lines.append("├" + "─" * (key_width + 2) + "┼" + "─" * 16 + "┤")

    for key, val in metrics.items():
        formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append("│ " + key.ljust(key_width) + " │ " + formatted.ljust(14) + " │")

    lines.append("└" + "─" * (key_width + 2) + "┴" + "─" * 16 + "┘")
    return "\n".join(lines)


def _format_markdown(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "(no metrics)"

    lines = ["| Metric | Value |", "|--------|-------|"]
    for key, val in metrics.items():
        formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(f"| {key} | {formatted} |")
    return "\n".join(lines)
