#!/usr/bin/env python3
"""Generate benchmark comparison charts for the README.

Reads baselines/results.json (from generate_baselines.py) or timing
results (from benchmark_performance.py) and produces transparent-background
SVG charts readable in both light and dark GitHub themes.

Usage:
    python scripts/plot_benchmarks.py --input baselines/results.json --output docs/images/
    python scripts/plot_benchmarks.py --input timing.json --type timing --output docs/images/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    )

NEUTRAL_GRAY = "#888888"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]


def _style_ax(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, color=NEUTRAL_GRAY, fontsize=12, pad=10)
    ax.set_ylabel(ylabel, color=NEUTRAL_GRAY, fontsize=10)
    ax.tick_params(colors=NEUTRAL_GRAY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(NEUTRAL_GRAY)
        spine.set_linewidth(0.5)
    ax.grid(axis="y", color=NEUTRAL_GRAY, alpha=0.2, linewidth=0.5)


def plot_model_quality(records: list[dict], output_dir: Path) -> list[Path]:
    metrics = [
        ("epe_wafer_mean_nm", "Wafer EPE (nm)", "model_quality_epe.svg"),
        ("pvband_mean_nm", "PV Band mean (nm)", "model_quality_pvb.svg"),
        ("l2_error_pixels", "L2 wafer error (px)", "model_quality_l2.svg"),
    ]

    out: list[Path] = []
    for key, ylabel, filename in metrics:
        models: list[str] = []
        values: list[float] = []
        for rec in records:
            v = rec.get("metrics", {}).get(key)
            if v is not None:
                models.append(rec["model"])
                values.append(float(v))

        if not values:
            continue

        fig, ax = plt.subplots(figsize=(max(4, len(models) * 1.2), 4), layout="constrained")
        fig.patch.set_alpha(0)
        bars = ax.bar(models, values, color=PALETTE[: len(models)], edgecolor="none", width=0.6)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=NEUTRAL_GRAY,
            )

        _style_ax(ax, f"{ylabel} — synthetic-8", ylabel)
        ax.set_xlabel("")
        plt.xticks(rotation=25, ha="right")

        for label in ax.get_xticklabels():
            label.set_color(NEUTRAL_GRAY)

        footnote = "Data: baselines/results.json — synthetic-8, 8 nm/px, HopkinsSimulator"
        fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, color=NEUTRAL_GRAY, alpha=0.6)

        path = output_dir / filename
        fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
        plt.close(fig)
        out.append(path)
        print(f"  wrote {path}")

    return out


def plot_model_quality_grouped(records: list[dict], output_dir: Path) -> Path:
    """Grouped bar chart with EPE, PVB, L2 side by side per model."""
    metric_keys = [
        ("epe_wafer_mean_nm", "Wafer EPE (nm)"),
        ("pvband_mean_nm", "PV Band (nm)"),
        ("l2_error_pixels", "L2 (px)"),
    ]

    models = [r["model"] for r in records]
    data: dict[str, list[float]] = {}
    for key, _ in metric_keys:
        data[key] = []
        for rec in records:
            v = rec.get("metrics", {}).get(key)
            data[key].append(float(v) if v is not None else float("nan"))

    n_models = len(models)
    n_metrics = len(metric_keys)
    x = list(range(n_models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.5), 4.5), layout="constrained")
    fig.patch.set_alpha(0)

    for i, (key, label) in enumerate(metric_keys):
        offset = (i - 1) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            data[key],
            width=width,
            label=label,
            color=PALETTE[i],
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    for label in ax.get_xticklabels():
        label.set_color(NEUTRAL_GRAY)

    _style_ax(ax, "Model quality comparison — synthetic-8", "Value")
    ax.legend(
        fontsize=8,
        loc="upper right",
        framealpha=0.7,
        edgecolor=NEUTRAL_GRAY,
        labelcolor=NEUTRAL_GRAY,
    )

    footnote = "Data: baselines/results.json — synthetic-8, 8 nm/px, shared HopkinsSimulator"
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, color=NEUTRAL_GRAY, alpha=0.6)

    path = output_dir / "benchmark_models.svg"
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def plot_timing(payload: dict, output_dir: Path) -> list[Path]:
    results = payload.get("results", [])
    if not results:
        print("  no timing results to plot")
        return []

    benchmarks: dict[str, list[dict]] = {}
    for r in results:
        benchmarks.setdefault(r["benchmark"], []).append(r)

    out: list[Path] = []
    for bench_name, entries in benchmarks.items():
        labels = [e["grid_size"] for e in entries]
        medians = [e["median_ms"] for e in entries]
        p99s = [e["p99_ms"] for e in entries]

        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4), layout="constrained")
        fig.patch.set_alpha(0)

        bar_w = 0.35
        ax.bar([xi - bar_w / 2 for xi in x], medians, bar_w, label="Median", color=PALETTE[0])
        ax.bar([xi + bar_w / 2 for xi in x], p99s, bar_w, label="P99", color=PALETTE[3])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for label in ax.get_xticklabels():
            label.set_color(NEUTRAL_GRAY)

        _style_ax(ax, f"{bench_name} — wall-clock time", "Time (ms)")
        ax.legend(
            fontsize=8,
            framealpha=0.7,
            edgecolor=NEUTRAL_GRAY,
            labelcolor=NEUTRAL_GRAY,
        )

        device = payload.get("environment", {}).get("device_name", "CPU")
        fig.text(
            0.5, -0.02,
            f"Device: {device}",
            ha="center", fontsize=7, color=NEUTRAL_GRAY, alpha=0.6,
        )

        safe_name = bench_name.replace("/", "_")
        path = output_dir / f"timing_{safe_name}.svg"
        fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
        plt.close(fig)
        out.append(path)
        print(f"  wrote {path}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate benchmark charts")
    ap.add_argument("--input", required=True, help="Path to results JSON file")
    ap.add_argument("--output", required=True, help="Output directory for SVG files")
    ap.add_argument(
        "--type",
        choices=["quality", "timing", "auto"],
        default="auto",
        help="Type of chart to generate (default: auto-detect)",
    )
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_timing = isinstance(data, dict) and "environment" in data
    is_quality = isinstance(data, list) or (isinstance(data, dict) and "results" in data and not is_timing)

    chart_type = args.type
    if chart_type == "auto":
        chart_type = "timing" if is_timing else "quality"

    print(f"Generating {chart_type} charts from {args.input} ...")

    if chart_type == "quality":
        records = data if isinstance(data, list) else data.get("records", data.get("results", []))
        plot_model_quality_grouped(records, output_dir)
        plot_model_quality(records, output_dir)
    elif chart_type == "timing":
        plot_timing(data, output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
