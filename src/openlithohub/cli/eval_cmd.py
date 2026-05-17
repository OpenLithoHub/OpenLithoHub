"""The `openlithohub eval` subcommand."""

from __future__ import annotations

from pathlib import Path

import typer

eval_app = typer.Typer(no_args_is_help=True)


@eval_app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path to load."),
    dataset: str = typer.Option(
        "lithobench", "--dataset", "-d", help="Dataset to evaluate on (lithobench/lithosim)."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Path to save evaluation report."
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, or markdown."
    ),
    node: str = typer.Option("45nm", "--node", "-n", help="Process node for evaluation context."),
) -> None:
    """Run evaluation of a lithography model on a benchmark dataset."""
    typer.echo(f"Evaluating model '{model}' on dataset '{dataset}' ({node})")
    typer.echo("This command is not yet implemented.")
    typer.echo("")
    typer.echo("Planned workflow:")
    typer.echo("  1. Load dataset via DatasetAdapter")
    typer.echo("  2. Load model via ModelRegistry")
    typer.echo("  3. Run predict() on each sample")
    typer.echo("  4. Compute metrics (EPE, PV Band, MRC, stochastic robustness)")
    typer.echo("  5. Generate report")
    raise typer.Exit(1)
