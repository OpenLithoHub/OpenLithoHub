"""Evaluation report generation."""

from __future__ import annotations

import json
from typing import Any

# Metric keys ending in this suffix are produced by ``_aggregate_metrics``
# when one or more samples returned a non-finite value (``nan`` / ``inf``)
# for that metric. Surfacing them as a banner keeps a quietly-noisy run
# from looking like a clean one — e.g. wafer-EPE goes ``inf`` when an
# edge polarity is empty, and dropping that sample silently could hide a
# broken dataset partition.
_DROPPED_SUFFIX = "_dropped_nonfinite"


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


def _dropped_banner(metrics: dict[str, Any], style: str) -> str:
    """Render a one-block warning summarizing any dropped non-finite samples.

    ``style`` selects the surrounding decoration: ``table`` matches the
    box-drawing of ``_format_table``, ``markdown`` renders as a blockquote.
    Returns the empty string when no drops happened.
    """
    drops = {
        k.removesuffix(_DROPPED_SUFFIX): int(v)
        for k, v in metrics.items()
        if k.endswith(_DROPPED_SUFFIX) and isinstance(v, int | float) and v
    }
    if not drops:
        return ""
    items = ", ".join(
        f"{n} samples produced inf/nan {metric}" for metric, n in sorted(drops.items())
    )
    if style == "markdown":
        return f"> **Warning:** dropped non-finite samples — {items}.\n\n"
    return f"⚠ Dropped non-finite samples — {items}.\n\n"


def _format_table(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "(no metrics)"

    visible = {k: v for k, v in metrics.items() if not k.endswith(_DROPPED_SUFFIX)}
    key_width = max(len(k) for k in visible) if visible else 6

    # Pre-format every value so the value column can be sized to fit the
    # longest entry. The previous fixed-width 14 column silently broke box
    # alignment whenever a value (e.g. a long ``l2_error_nm2`` integer or a
    # model-name string) exceeded 14 chars — ``ljust`` does not truncate.
    formatted_values = [
        f"{val:.4f}" if isinstance(val, float) else str(val) for val in visible.values()
    ]
    val_width = max((len(s) for s in formatted_values), default=5)
    val_width = max(val_width, len("Value"))

    lines = ["┌" + "─" * (key_width + 2) + "┬" + "─" * (val_width + 2) + "┐"]
    lines.append("│ " + "Metric".ljust(key_width) + " │ " + "Value".ljust(val_width) + " │")
    lines.append("├" + "─" * (key_width + 2) + "┼" + "─" * (val_width + 2) + "┤")

    for key, formatted in zip(visible.keys(), formatted_values, strict=True):
        lines.append("│ " + key.ljust(key_width) + " │ " + formatted.ljust(val_width) + " │")

    lines.append("└" + "─" * (key_width + 2) + "┴" + "─" * (val_width + 2) + "┘")
    return _dropped_banner(metrics, style="table") + "\n".join(lines)


def _format_markdown(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "(no metrics)"

    visible = {k: v for k, v in metrics.items() if not k.endswith(_DROPPED_SUFFIX)}
    lines = ["| Metric | Value |", "|--------|-------|"]
    for key, val in visible.items():
        formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(f"| {key} | {formatted} |")
    return _dropped_banner(metrics, style="markdown") + "\n".join(lines)
