"""Tests for the eval-report formatter."""

from __future__ import annotations

from openlithohub.benchmark.report import generate_report


def test_dropped_nonfinite_keys_hidden_from_table_rows() -> None:
    """The raw ``*_dropped_nonfinite`` counts should not pollute the metric
    rows — they're surfaced as a banner so they don't get lost in a long
    table of numbers."""
    metrics = {
        "epe_mean_nm": 1.5,
        "epe_wafer_mean_nm": 4.2,
        "epe_wafer_mean_nm_dropped_nonfinite": 3,
    }
    out = generate_report(metrics, output_format="table")
    assert "epe_wafer_mean_nm_dropped_nonfinite" not in out
    assert "Dropped non-finite samples" in out
    assert "3 samples produced inf/nan epe_wafer_mean_nm" in out


def test_dropped_banner_omitted_when_clean() -> None:
    metrics = {"epe_mean_nm": 1.5, "epe_wafer_mean_nm": 4.2}
    out = generate_report(metrics, output_format="table")
    assert "Dropped non-finite" not in out


def test_dropped_banner_in_markdown_form() -> None:
    metrics = {
        "epe_mean_nm": 1.5,
        "epe_mean_nm_dropped_nonfinite": 1,
    }
    out = generate_report(metrics, output_format="markdown")
    assert out.startswith("> **Warning:**")
    assert "1 samples produced inf/nan epe_mean_nm" in out


def test_json_output_keeps_full_metric_dict() -> None:
    """JSON output is machine-consumed — preserve everything, including the
    drop counts, so downstream tools can decide what to do with them."""
    metrics = {
        "epe_mean_nm": 1.5,
        "epe_mean_nm_dropped_nonfinite": 2,
    }
    out = generate_report(metrics, output_format="json")
    assert "epe_mean_nm_dropped_nonfinite" in out
