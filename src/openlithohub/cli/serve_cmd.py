"""The `openlithohub serve` subcommand — boots the FastAPI engine."""

from __future__ import annotations

import typer
from rich.console import Console

serve_app = typer.Typer(no_args_is_help=False)


@serve_app.callback(invoke_without_command=True)
def run(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="TCP port."),
    workers: int = typer.Option(1, "--workers", "-w", help="Uvicorn worker count."),
    log_level: str = typer.Option("info", "--log-level", help="uvicorn log level."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)."),
) -> None:
    """Run the OpenLithoHub HTTP engine.

    Example:
        openlithohub serve --port 8000

    Then from any client::

        curl -X POST http://localhost:8000/v1/optimize \\
             -F "layout=@chip.oas" -F "model=heuristic-opc" \\
             -o optimized.oas
    """
    console = Console()
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Error:[/red] FastAPI server extras are not installed. "
            "Install with: [bold]pip install openlithohub[server][/bold]"
        )
        raise typer.Exit(1) from None

    console.print(f"[bold]OpenLithoHub HTTP engine[/bold] starting on http://{host}:{port}")
    uvicorn.run(
        "openlithohub.server.app:create_app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
        factory=True,
    )
