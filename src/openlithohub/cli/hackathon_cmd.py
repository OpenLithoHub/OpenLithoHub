"""The `openlithohub hackathon` subcommand."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from openlithohub.hackathon import load_manifest

hackathon_app = typer.Typer(no_args_is_help=True)


@hackathon_app.command()
def info(
    manifest: Path | None = typer.Option(None, "--manifest", "-m", help="Path to manifest YAML."),
) -> None:
    """Print the current hackathon contract — tag, sample count, gates, target."""
    console = Console()
    m = load_manifest(manifest)

    status_colour = {"charter": "yellow", "open": "green", "closed": "red"}.get(m.status, "white")
    console.print(
        f"[bold]Hackathon track:[/bold] {m.track}  [{status_colour}]({m.status})[/{status_colour}]"
    )
    console.print(f"[bold]Process node:[/bold]  {m.process_node}")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Test-set tag", m.dataset_tag)
    table.add_row("Commit SHA", m.dataset_commit_sha or "[dim]TBD[/dim]")
    table.add_row(
        "Sample count",
        str(m.dataset_sample_count) if m.dataset_sample_count is not None else "[dim]TBD[/dim]",
    )
    table.add_row("MRC max", f"{m.mrc_violation_rate_max}")
    table.add_row("DRC required", "yes" if m.drc_pass_required else "no")
    table.add_row(
        "Target EPE mean (nm)",
        f"{m.target_epe_mean_nm}" if m.has_calibrated_target else "[dim]TBD[/dim]",
    )
    table.add_row("Ranking", " > ".join((m.ranking_primary, *m.ranking_tiebreakers)))
    console.print(table)

    if not m.is_open:
        console.print(
            "\n[yellow]Status is not 'open' — submissions are not yet being scored "
            "against this manifest. Watch docs/hackathon.md for the launch.[/yellow]"
        )
