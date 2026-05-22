"""CLI entry point for OpenLithoHub."""

from __future__ import annotations

import typer

from openlithohub.cli.data_cmd import data_app
from openlithohub.cli.eval_cmd import eval_app
from openlithohub.cli.export_cmd import export_app
from openlithohub.cli.hackathon_cmd import hackathon_app
from openlithohub.cli.leaderboard_cmd import leaderboard_app
from openlithohub.cli.optimize_cmd import optimize_app
from openlithohub.cli.serve_cmd import serve_app
from openlithohub.cli.simulate_cmd import simulate_app
from openlithohub.cli.synth_cmd import synth_app

app = typer.Typer(
    name="openlithohub",
    help="Open-source computational lithography benchmarking and workflow tool.",
    no_args_is_help=True,
)

app.add_typer(eval_app, name="eval", help="Evaluate a lithography model on benchmarks.")
app.add_typer(data_app, name="data", help="Inspect bundled cell-library adapters (ASAP7, OpenRAM).")
app.add_typer(optimize_app, name="optimize", help="Run mask optimization on a layout.")
app.add_typer(
    leaderboard_app, name="leaderboard", help="View, submit, and export leaderboard results."
)
app.add_typer(simulate_app, name="simulate", help="Run a forward simulator on a mask.")
app.add_typer(synth_app, name="synth", help="Generate synthetic PDK-aware layouts.")
app.add_typer(hackathon_app, name="hackathon", help="Inspect the current hackathon contract.")
app.add_typer(export_app, name="export", help="Export a model to ONNX / TorchScript.")
app.add_typer(serve_app, name="serve", help="Run the HTTP micro-service engine (FastAPI).")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
) -> None:
    """OpenLithoHub — computational lithography benchmarking and workflow tool."""
    if version:
        from openlithohub import __version__

        typer.echo(f"openlithohub {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
