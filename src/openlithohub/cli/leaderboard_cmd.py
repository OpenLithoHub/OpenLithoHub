"""The `openlithohub leaderboard` subcommand."""

from __future__ import annotations

import json as json_mod
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from openlithohub.leaderboard.schema import BenchmarkResult
from openlithohub.leaderboard.tracker import LeaderboardStore, get_leaderboard, submit_result

leaderboard_app = typer.Typer(no_args_is_help=True)


@leaderboard_app.command()
def view(
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Filter by dataset."),
    node: str | None = typer.Option(None, "--node", "-n", help="Filter by process node."),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, markdown."
    ),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max entries to display."),
    store_path: Path | None = typer.Option(None, "--store", hidden=True),
) -> None:
    """View the leaderboard rankings."""
    console = Console()

    store = LeaderboardStore(store_path) if store_path else None
    results = get_leaderboard(dataset=dataset, process_node=node, store=store)

    if not results:
        if dataset or node:
            console.print(
                f"[yellow]No leaderboard entries match[/yellow] "
                f"(dataset={dataset!r}, node={node!r}). Drop filters to see all entries."
            )
            raise typer.Exit(1)
        console.print("[yellow]No leaderboard entries found.[/yellow]")
        raise typer.Exit(0)

    if limit:
        results = results[:limit]

    if format == "json":
        output = json_mod.dumps([r.model_dump(mode="json") for r in results], indent=2, default=str)
        console.print(output, highlight=False)
    elif format == "markdown":
        console.print(_format_markdown(results), highlight=False)
    else:
        _print_table(console, results)


@leaderboard_app.command()
def submit(
    file: Path | None = typer.Option(
        None, "--file", "-F", help="JSON file with BenchmarkResult fields."
    ),
    model_name: str | None = typer.Option(None, "--model", "-m", help="Model name."),
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Dataset name."),
    node: str | None = typer.Option(None, "--node", "-n", help="Process node (e.g. 7nm, 3nm-euv)."),
    topology: str | None = typer.Option(None, "--topology", "-t", help="manhattan or curvilinear."),
    epe_mean: float | None = typer.Option(None, "--epe-mean", help="Mean EPE in nm."),
    epe_max: float | None = typer.Option(None, "--epe-max", help="Max EPE in nm."),
    pvband_mean: float | None = typer.Option(
        None, "--pvband-mean", help="Mean PV band width in nm."
    ),
    pvband_max: float | None = typer.Option(None, "--pvband-max", help="Max PV band width in nm."),
    paper_url: str | None = typer.Option(None, "--paper-url", help="Paper URL."),
    code_url: str | None = typer.Option(None, "--code-url", help="Code/repo URL."),
    notes: str | None = typer.Option(None, "--notes", help="Additional notes."),
    store_path: Path | None = typer.Option(None, "--store", hidden=True),
) -> None:
    """Submit a benchmark result to the leaderboard."""
    console = Console()

    if file:
        data = json_mod.loads(file.read_text(encoding="utf-8"))
    elif (
        model_name
        and dataset
        and node
        and topology
        and epe_mean is not None
        and epe_max is not None
    ):
        data = {
            "model_name": model_name,
            "dataset": dataset,
            "process_node": node,
            "mask_topology": topology,
            "epe_mean_nm": epe_mean,
            "epe_max_nm": epe_max,
        }
        if pvband_mean is not None:
            data["pvband_mean_nm"] = pvband_mean
        if pvband_max is not None:
            data["pvband_max_nm"] = pvband_max
        if paper_url:
            data["paper_url"] = paper_url
        if code_url:
            data["code_url"] = code_url
        if notes:
            data["notes"] = notes
    else:
        console.print(
            "[red]Error:[/red] Provide --file or all required fields "
            "(--model, --dataset, --node, --topology, --epe-mean, --epe-max)."
        )
        raise typer.Exit(1)

    try:
        result = BenchmarkResult.model_validate(data)
    except ValidationError as e:
        console.print("[red]Validation error:[/red]")
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            console.print(f"  [red]{loc}[/red]: {err['msg']}")
        raise typer.Exit(1) from None

    store = LeaderboardStore(store_path) if store_path else None
    submission_id = submit_result(result, store=store)
    console.print(f"[green]Submitted![/green] ID: {submission_id}")
    console.print(f"  Model: {result.model_name}")
    console.print(f"  EPE mean: {result.epe_mean_nm:.2f} nm")


@leaderboard_app.command()
def export(
    output: Path = typer.Option(..., "--output", "-o", help="Output file path."),
    format: str = typer.Option("json", "--format", "-f", help="Export format: json or markdown."),
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Filter by dataset."),
    node: str | None = typer.Option(None, "--node", "-n", help="Filter by process node."),
    store_path: Path | None = typer.Option(None, "--store", hidden=True),
) -> None:
    """Export the leaderboard to a file."""
    console = Console()

    store = LeaderboardStore(store_path) if store_path else None
    results = get_leaderboard(dataset=dataset, process_node=node, store=store)

    if format == "markdown":
        content = _format_markdown(results)
    else:
        content = json_mod.dumps(
            [r.model_dump(mode="json") for r in results], indent=2, default=str
        )

    output.write_text(content, encoding="utf-8")
    console.print(f"[green]Exported {len(results)} entries to {output}[/green]")


def _print_table(console: Console, results: list[BenchmarkResult]) -> None:
    table = Table(title="OpenLithoHub Leaderboard", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="bold", no_wrap=True, overflow="fold")
    table.add_column("Dataset", no_wrap=True, overflow="fold")
    table.add_column("Node")
    table.add_column("EPE Mean (nm)", justify="right")
    table.add_column("EPE Max (nm)", justify="right")
    table.add_column("PV Band Mean (nm)", justify="right")
    table.add_column("PV Band Max (nm)", justify="right")
    table.add_column("DRC", justify="center")
    table.add_column("Links")

    for i, r in enumerate(results, 1):
        links = []
        if r.paper_url:
            links.append("paper")
        if r.code_url:
            links.append("code")

        table.add_row(
            str(i),
            r.model_name,
            r.dataset,
            r.process_node.value,
            f"{r.epe_mean_nm:.2f}",
            f"{r.epe_max_nm:.2f}",
            f"{r.pvband_mean_nm:.2f}" if r.pvband_mean_nm is not None else "-",
            f"{r.pvband_max_nm:.2f}" if r.pvband_max_nm is not None else "-",
            "pass" if r.drc_pass else ("fail" if r.drc_pass is False else "-"),
            " | ".join(links) if links else "-",
        )

    console.print(table)


def _format_markdown(results: list[BenchmarkResult]) -> str:
    header = (
        "| # | Model | Dataset | Node | EPE Mean (nm) | EPE Max (nm) "
        "| PV Band Mean (nm) | PV Band Max (nm) | DRC | Paper | Code |"
    )
    lines = [
        header,
        "|---|-------|---------|------|---------------|--------------|"
        "-------------------|------------------|-----|-------|------|",
    ]
    for i, r in enumerate(results, 1):
        pvm = f"{r.pvband_mean_nm:.2f}" if r.pvband_mean_nm is not None else "-"
        pvx = f"{r.pvband_max_nm:.2f}" if r.pvband_max_nm is not None else "-"
        drc = "pass" if r.drc_pass else ("fail" if r.drc_pass is False else "-")
        paper = f"[link]({r.paper_url})" if r.paper_url else "-"
        code = f"[link]({r.code_url})" if r.code_url else "-"
        lines.append(
            f"| {i} | {r.model_name} | {r.dataset} | {r.process_node.value} | "
            f"{r.epe_mean_nm:.2f} | {r.epe_max_nm:.2f} | {pvm} | {pvx} | {drc} | {paper} | {code} |"
        )
    return "\n".join(lines)
