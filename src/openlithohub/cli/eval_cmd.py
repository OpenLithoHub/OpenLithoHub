"""The `openlithohub eval` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

eval_app = typer.Typer(no_args_is_help=True)


@eval_app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path to load."),
    dataset: str = typer.Option(
        "lithobench", "--dataset", "-d", help="Dataset to evaluate on (lithobench/lithosim)."
    ),
    data_root: Path = typer.Option(
        ..., "--data-root", "-r", help="Path to dataset root directory."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Path to save evaluation report."
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, or markdown."
    ),
    node: str = typer.Option("45nm", "--node", "-n", help="Process node for evaluation context."),
    pixel_nm: float = typer.Option(1.0, "--pixel-nm", help="Pixel size in nanometers."),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Max samples to evaluate (default: all)."
    ),
    mrc_check: bool = typer.Option(True, "--mrc/--no-mrc", help="Run MRC compliance check."),
    min_width_nm: float = typer.Option(
        40.0, "--min-width-nm", help="Minimum feature width for MRC (nm)."
    ),
    min_spacing_nm: float = typer.Option(
        40.0, "--min-spacing-nm", help="Minimum spacing for MRC (nm)."
    ),
) -> None:
    """Run evaluation of a lithography model on a benchmark dataset."""
    console = Console()

    import openlithohub.models.examples.dummy_model  # noqa: F401 — register built-in models
    from openlithohub.benchmark.metrics.epe import compute_epe
    from openlithohub.benchmark.report import generate_report
    from openlithohub.models.registry import registry

    console.print(f"[bold]Evaluating[/bold] model={model} dataset={dataset} node={node}")

    try:
        litho_model = registry.get(model)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    litho_model.setup()

    try:
        adapter = _load_dataset(dataset, data_root, pixel_nm)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        litho_model.teardown()
        raise typer.Exit(1) from None

    n_samples = min(len(adapter), limit) if limit else len(adapter)
    console.print(f"Running on {n_samples} samples...")

    if mrc_check:
        from openlithohub.benchmark.compliance.mrc import check_mrc

    all_metrics: list[dict[str, float]] = []
    for i in range(n_samples):
        sample = adapter[i]
        result = litho_model.predict(sample.design)

        sample_metrics: dict[str, float] = {}

        if sample.mask is not None:
            epe = compute_epe(result.mask, sample.mask, pixel_size_nm=pixel_nm)
            sample_metrics.update(epe)

        if mrc_check:
            mrc_result = check_mrc(
                result.mask,
                min_width_nm=min_width_nm,
                min_spacing_nm=min_spacing_nm,
                pixel_size_nm=pixel_nm,
            )
            sample_metrics["mrc_violation_rate"] = mrc_result.violation_rate
            sample_metrics["mrc_passed"] = 1.0 if mrc_result.passed else 0.0

        if sample_metrics:
            all_metrics.append(sample_metrics)

    litho_model.teardown()

    if not all_metrics:
        console.print("[yellow]Warning:[/yellow] No metrics computed.")
        raise typer.Exit(1)

    aggregated = _aggregate_metrics(all_metrics)
    aggregated["model"] = model
    aggregated["dataset"] = dataset
    aggregated["node"] = node
    aggregated["num_samples"] = n_samples

    report = generate_report(aggregated, output_format=format)
    console.print(report, highlight=False)

    if output:
        output.write_text(report)
        console.print(f"Report saved to {output}")


def _load_dataset(
    dataset: str, data_root: Path, pixel_nm: float
) -> Any:
    from openlithohub.data import LithoBenchDataset, LithoSimDataset

    if dataset == "lithobench":
        return LithoBenchDataset(root=data_root, pixel_nm=pixel_nm)
    if dataset == "lithosim":
        return LithoSimDataset(cache_dir=str(data_root), pixel_nm=pixel_nm)
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: lithobench, lithosim")


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, Any]:
    """Average per-sample metrics into a single aggregate dict."""
    import torch

    if not metrics_list:
        return {}

    all_keys: set[str] = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    aggregated: dict[str, Any] = {}
    for key in sorted(all_keys):
        vals = [m[key] for m in metrics_list if key in m]
        if vals:
            aggregated[key] = float(torch.tensor(vals).mean().item())
    return aggregated
