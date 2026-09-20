"""``openlithohub flow`` — end-to-end design→litho→manufacturability pipeline.

Accepts an ORFS product directory or a standalone DEF/GDS file, selects
a layer, tiles it, runs the Hopkins forward model + resist, and produces
an aggregated tile-level manufacturability report (EPE, PV Band, DRC, MRC).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

flow_app = typer.Typer(
    help="End-to-end design-to-litho-to-manufacturability pipeline.",
    no_args_is_help=True,
)


@flow_app.command()
def run(
    input_path: Path = typer.Argument(
        ...,
        help="Path to a GDS/OAS/DEF file or an ORFS results directory.",
    ),
    pdk: str = typer.Option(
        "orfs_asap7",
        "--pdk",
        help=(
            "PDK layer mapping name (asap7, freepdk45, orfs_asap7, sky130) "
            "or a path to a custom JSON layermap."
        ),
    ),
    layer: str = typer.Option(
        "metal1",
        "--layer",
        help="Layer name from the PDK layermap (metal1, metal2, via1, etc.).",
    ),
    pixel_nm: float = typer.Option(1.0, "--pixel-nm", help="Pixel size in nm."),
    tile_nm: float = typer.Option(2000.0, "--tile-nm", help="Tile edge length in nm."),
    node: str = typer.Option("45nm", "--node", "-n", help="Process node for litho params."),
    resist_diffusion_nm: float = typer.Option(
        0.0,
        "--resist-diffusion-nm",
        help="Acid diffusion length in nm (0 = legacy CTR).",
    ),
    quencher: float = typer.Option(
        0.0,
        "--quencher",
        help="Quencher concentration (0 = disabled).",
    ),
    drc_check: bool = typer.Option(True, "--drc/--no-drc", help="Run DRC compliance."),
    mrc_check: bool = typer.Option(True, "--mrc/--no-mrc", help="Run MRC compliance."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save JSON report.",
    ),
    report_json: Path | None = typer.Option(
        None,
        "--report-json",
        help=(
            "Path to save the machine-readable JSON report, including the "
            "performance profile section."
        ),
    ),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Print a per-phase performance profile table (parser, forward, metrics, peak RSS).",
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic/--no-deterministic",
        help="Force bit-reproducible backends.",
    ),
) -> None:
    """Run the design→litho→manufacturability pipeline on a layout."""
    console = Console()
    import resource
    import time as _time

    t_start = _time.perf_counter()

    if deterministic:
        from openlithohub._utils.determinism import set_deterministic

        set_deterministic()

    # Resolve PDK layer mapping
    from openlithohub.data._layers import LAYERS, load_layermap

    if Path(pdk).exists() and Path(pdk).suffix == ".json":
        pdk_layers = load_layermap(pdk)
    elif pdk in LAYERS:
        pdk_layers = LAYERS[pdk]
    else:
        available = ", ".join(sorted(LAYERS.keys()))
        console.print(f"[red]Error:[/red] Unknown PDK '{pdk}'. Available: {available}")
        raise typer.Exit(1)

    # Resolve layer number from layer name
    layer_tuple = getattr(pdk_layers, layer, None)
    if layer_tuple is None:
        available_layers = [
            f.name
            for f in pdk_layers.__dataclass_fields__.values()
            if getattr(pdk_layers, f.name, None) is not None
        ]
        console.print(
            f"[red]Error:[/red] Layer '{layer}' not in PDK. "
            f"Available: {', '.join(available_layers)}"
        )
        raise typer.Exit(1)

    console.print(f"[bold]Flow[/bold] input={input_path} pdk={pdk} layer={layer}={layer_tuple}")

    # Resolve input file (directory → find GDS)
    gds_path = _resolve_input(input_path, console)
    if gds_path is None:
        raise typer.Exit(1)

    # Load and tile
    from openlithohub.data.orfs import OrfsArtifactDataset

    dataset = OrfsArtifactDataset(
        gds_path=gds_path,
        design_layer=layer_tuple,
        pixel_nm=pixel_nm,
        tile_nm=tile_nm,
        drop_empty_tiles=True,
    )
    t_load = _time.perf_counter()
    console.print(f"Loaded {len(dataset)} tiles from {gds_path.name}")

    # Build simulator
    from openlithohub.simulators.base import SimulatorConfig
    from openlithohub.simulators.hopkins_sim import HopkinsSimulator
    from openlithohub.workflow.process_node import PROCESS_NODES, get_node

    if node in PROCESS_NODES:
        nc = get_node(node)
        cfg = SimulatorConfig(
            wavelength_nm=nc.wavelength_nm,
            na=nc.numerical_aperture,
            pixel_size_nm=pixel_nm,
            threshold=nc.resist_threshold,
            resist_diffusion_nm=resist_diffusion_nm,
            quencher=quencher,
        )
    else:
        cfg = SimulatorConfig(
            pixel_size_nm=pixel_nm,
            resist_diffusion_nm=resist_diffusion_nm,
            quencher=quencher,
        )

    simulator = HopkinsSimulator(cfg)

    # Run metrics on each tile
    from openlithohub.benchmark.compliance.drc import check_drc
    from openlithohub.benchmark.compliance.mrc import check_mrc
    from openlithohub.benchmark.metrics.epe import compute_epe
    from openlithohub.benchmark.metrics.pvband import compute_pvband

    all_metrics: list[dict[str, float]] = []
    n_tiles = len(dataset)
    t_forward_s = 0.0
    t_metrics_s = 0.0
    for i in range(n_tiles):
        sample = dataset[i]
        _t0 = _time.perf_counter()
        result = simulator.simulate(sample.design)
        t_forward_s += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        tile_metrics: dict[str, float] = {}

        # L2 wafer error
        if sample.mask is not None:
            wafer = result.resist if result.resist is not None else result.aerial
            l2 = float((wafer - sample.mask).abs().sum().item())
            tile_metrics["l2_error_pixels"] = l2

            epe = compute_epe(wafer, sample.mask, pixel_size_nm=pixel_nm)
            tile_metrics["epe_mean_nm"] = epe["epe_mean_nm"]

        pv = compute_pvband(
            sample.design,
            pixel_size_nm=pixel_nm,
            resist_diffusion_nm=resist_diffusion_nm,
            quencher=quencher,
        )
        tile_metrics.update(pv)

        if mrc_check:
            mrc_result = check_mrc(
                result.aerial if result.resist is None else result.resist, pixel_size_nm=pixel_nm
            )
            tile_metrics["mrc_violation_rate"] = mrc_result.violation_rate

        if drc_check:
            drc_result = check_drc(
                result.resist if result.resist is not None else result.aerial,
                pixel_size_nm=pixel_nm,
            )
            tile_metrics["drc_passed"] = float(drc_result.passed)

        all_metrics.append(tile_metrics)
        t_metrics_s += _time.perf_counter() - _t0

    # Aggregate
    aggregated = _aggregate(all_metrics)
    aggregated["pdk"] = pdk
    aggregated["layer"] = layer
    aggregated["layer_number"] = list(layer_tuple)
    aggregated["num_tiles"] = n_tiles

    # Performance profile — opt-in observability: where did the time and
    # memory go? Explicitly requested via --profile / --report-json.
    perf: dict[str, Any] = {
        "parse_and_tile_wall_s": round(t_load - t_start, 6),
        "tile_count": n_tiles,
        "forward_wall_s_total": round(t_forward_s, 6),
        "metrics_wall_s_total": round(t_metrics_s, 6),
        "end_to_end_wall_s": round(_time.perf_counter() - t_start, 6),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "forward_calls": n_tiles,
    }

    # Print report
    _print_report(console, aggregated)

    if profile:
        _print_profile(console, perf)

    if output:
        output.write_text(json.dumps(aggregated, indent=2, default=str))
        console.print(f"Report saved to {output}")

    if report_json:
        report_json.write_text(
            json.dumps({"report": aggregated, "profile": perf}, indent=2, default=str)
        )
        console.print(f"Machine-readable report saved to {report_json}")


def _resolve_input(path: Path, console: Console) -> Path | None:
    """Resolve input path to a GDS file."""
    if path.is_file():
        return path
    if path.is_dir():
        gds_files = list(path.rglob("*.gds")) + list(path.rglob("*.oas"))
        if not gds_files:
            console.print(f"[red]Error:[/red] No GDS/OAS files found under {path}")
            return None
        if len(gds_files) > 1:
            console.print(
                f"[yellow]Warning:[/yellow] Multiple GDS files found; using {gds_files[0]}"
            )
        return gds_files[0]
    console.print(f"[red]Error:[/red] {path} does not exist")
    return None


def _aggregate(metrics: list[dict[str, float]]) -> dict[str, Any]:
    """Simple mean aggregation over tile metrics."""
    if not metrics:
        return {}
    result: dict[str, Any] = {}
    all_keys: set[str] = set()
    for m in metrics:
        all_keys.update(m.keys())
    for key in sorted(all_keys):
        values = [m[key] for m in metrics if key in m and isinstance(m[key], (int, float))]
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            result[key] = float(np.mean(finite))
    return result


def _print_report(console: Console, data: dict[str, Any]) -> None:
    """Print a summary table to the console."""
    table = Table(title="Flow Report", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for key, value in sorted(data.items()):
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    console.print(table)


def _print_profile(console: Console, profile: dict[str, Any]) -> None:
    """Print the opt-in performance profile table."""
    table = Table(title="Flow Profile", show_header=True)
    table.add_column("Phase", style="bold")
    table.add_column("Value", justify="right")
    for key, value in profile.items():
        if key.endswith("_wall_s"):
            table.add_row(key, f"{float(value) * 1000:.1f} ms")
        elif key == "peak_rss_bytes":
            table.add_row(key, f"{int(value) / (1 << 20):.1f} MiB")
        else:
            table.add_row(key, str(value))
    console.print(table)
