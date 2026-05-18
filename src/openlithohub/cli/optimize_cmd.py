"""The `openlithohub optimize` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

optimize_app = typer.Typer(no_args_is_help=True)


@optimize_app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Input layout file (.oas or .gds)."),
    model: str = typer.Option(..., "--model", "-m", help="Optimization model to use."),
    output: Path = typer.Option(..., "--output", "-o", help="Output optimized layout path."),
    writer: str = typer.Option("mbmw", "--writer", "-w", help="Target writer type: mbmw or vsb."),
    node: str = typer.Option("3nm-euv", "--node", "-n", help="Target process node."),
    drc_check: bool = typer.Option(
        False, "--drc-check", help="Run DRC/MRC checks after optimization."
    ),
    tile_size: int = typer.Option(
        2048, "--tile-size", help="Tile size for distributed processing (pixels)."
    ),
    overlap: int = typer.Option(
        128, "--overlap", help="Tile overlap for seamless stitching (pixels)."
    ),
    pixel_nm: float = typer.Option(1.0, "--pixel-nm", help="Pixel size in nanometers."),
    device: str = typer.Option(
        "cpu", "--device", help="Torch device for the forward model (cpu, cuda, mps)."
    ),
    dtype: str = typer.Option(
        "fp32", "--dtype", help="Compute dtype for the forward model: fp32 or bf16."
    ),
    compile_forward: bool = typer.Option(
        False, "--compile/--no-compile", help="Wrap the Hopkins forward with torch.compile."
    ),
    pretrained: bool = typer.Option(
        False,
        "--pretrained/--no-pretrained",
        help="Load pretrained weights for the selected model (when supported).",
    ),
    sha256: str | None = typer.Option(
        None,
        "--sha256",
        help=(
            "Expected SHA256 hex digest for direct-URL weight downloads. "
            "Required when the model resolves weights via a raw HTTPS URL; "
            "ignored for HuggingFace Hub repos."
        ),
    ),
) -> None:
    """Run end-to-end mask optimization on a layout file.

    Example:
        openlithohub optimize --input chip.oas --model diffusion-ilt
        --writer mbmw --node 3nm-euv --drc-check --output optimized.oas
    """
    console = Console()

    import openlithohub.models.examples.dummy_model  # noqa: F401
    import openlithohub.models.levelset_ilt  # noqa: F401
    import openlithohub.models.neural_ilt  # noqa: F401
    from openlithohub.models.registry import registry
    from openlithohub.workflow.process_node import PROCESS_NODES

    # Auto-configure from process node
    if node in PROCESS_NODES:
        from openlithohub.workflow.process_node import get_node

        node_config = get_node(node)
        if pixel_nm == 1.0:
            pixel_nm = node_config.pixel_size_nm

    console.print("[bold]OpenLithoHub Mask Optimization[/bold]")
    console.print(f"  Input:  {input}")
    console.print(f"  Model:  {model}")
    console.print(f"  Writer: {writer}")
    console.print(f"  Node:   {node}")
    console.print()

    try:
        litho_model = registry.get(model, **_build_model_kwargs(pretrained, sha256))
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None
    except TypeError:
        # Selected model does not accept pretrained / sha256 kwargs.
        if pretrained or sha256 is not None:
            console.print(
                f"[yellow]Warning:[/yellow] Model {model!r} does not support "
                "--pretrained / --sha256; ignoring."
            )
        try:
            litho_model = registry.get(model)
        except KeyError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from None

    litho_model.setup()

    console.print("[bold]Step 1:[/bold] Parsing layout...")
    try:
        layout_tensor = _load_layout_as_tensor(input, pixel_nm)
    except (ImportError, FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        litho_model.teardown()
        raise typer.Exit(1) from None

    console.print(f"  Layout size: {layout_tensor.shape[0]}x{layout_tensor.shape[1]} pixels")

    console.print("[bold]Step 2:[/bold] Tiling layout...")
    from openlithohub.workflow.tiling import Tile, stitch_tiles, tile_layout

    tiles = tile_layout(layout_tensor, tile_size=tile_size, overlap=overlap)
    console.print(f"  Generated {len(tiles)} tiles ({tile_size}px, overlap={overlap})")

    console.print("[bold]Step 3:[/bold] Running optimization...")
    tile_results: list[tuple[Tile, torch.Tensor]] = []
    perf_kwargs = _build_perf_kwargs(device, dtype, compile_forward)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing tiles", total=len(tiles))
        for tile in tiles:
            result = litho_model.predict(tile.tensor, **perf_kwargs)
            tile_results.append((tile, result.mask))
            progress.advance(task)

    litho_model.teardown()

    console.print("[bold]Step 4:[/bold] Stitching tiles...")
    h, w = layout_tensor.shape
    optimized = stitch_tiles(tile_results, (h, w))
    optimized = (optimized > 0.5).float()
    console.print(f"  Stitched output: {optimized.shape[0]}x{optimized.shape[1]}")

    if drc_check:
        console.print("[bold]Step 5:[/bold] Running compliance checks...")
        from openlithohub.benchmark.compliance.drc import check_drc
        from openlithohub.benchmark.compliance.mrc import check_mrc

        mrc_result = check_mrc(optimized, pixel_size_nm=pixel_nm)
        drc_result = check_drc(optimized, pixel_size_nm=pixel_nm)

        if mrc_result.passed and drc_result.passed:
            console.print("  [green]All checks passed[/green]")
        else:
            if not mrc_result.passed:
                console.print(
                    f"  [yellow]MRC:[/yellow] {mrc_result.violation_count} violations "
                    f"(rate={mrc_result.violation_rate:.4f})"
                )
            if not drc_result.passed:
                console.print(f"  [yellow]DRC:[/yellow] {drc_result.violation_count} violations")

    console.print(f"[bold]Step {'6' if drc_check else '5'}:[/bold] Exporting...")
    export_mode = "curvilinear" if writer == "mbmw" else "manhattan"

    try:
        from openlithohub.workflow.export import export_oasis

        export_oasis(optimized, output, mode=export_mode, pixel_size_nm=pixel_nm)
        console.print(f"  [green]Output written to {output}[/green]")
    except ImportError as e:
        console.print(f"  [yellow]Warning:[/yellow] {e}")
        console.print("  Falling back to raw tensor export...")
        torch.save(optimized, str(output).replace(".oas", ".pt"))
        console.print(f"  Saved tensor to {str(output).replace('.oas', '.pt')}")

    console.print()
    console.print("[bold green]Optimization complete.[/bold green]")


def _build_perf_kwargs(device: str, dtype: str, compile_forward: bool) -> dict[str, Any]:
    """Translate CLI perf flags into predict() kwargs."""
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype not in dtype_map:
        raise typer.BadParameter(f"--dtype must be 'fp32' or 'bf16'; got {dtype!r}")
    return {
        "device": device,
        "dtype": dtype_map[dtype],
        "compile_forward": compile_forward,
    }


def _build_model_kwargs(pretrained: bool, sha256: str | None) -> dict[str, Any]:
    """Construct registry.get() kwargs for opt-in remote weight loading."""
    kwargs: dict[str, Any] = {}
    if pretrained:
        kwargs["pretrained"] = True
    if sha256 is not None:
        kwargs["url_sha256"] = sha256
    return kwargs


def _load_layout_as_tensor(path: Path, pixel_nm: float) -> torch.Tensor:
    """Load a layout file and rasterize to a binary tensor."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pt":
        return torch.load(str(path), weights_only=True)  # type: ignore[no-any-return]

    if suffix == ".npy":
        import numpy as np

        arr = np.load(str(path))
        return torch.from_numpy(arr).float()

    try:
        import klayout.db as db
    except ImportError:
        raise ImportError(
            "klayout is required for OASIS/GDSII parsing. "
            "Install with: pip install openlithohub[workflow]"
        ) from None

    layout = db.Layout()
    layout.read(str(path))

    top_cell = layout.top_cell()
    bbox = top_cell.bbox()

    width_dbu = bbox.width()
    height_dbu = bbox.height()
    dbu_nm = layout.dbu * 1000.0
    pixels_per_dbu = dbu_nm / pixel_nm

    w_px = max(1, int(width_dbu * pixels_per_dbu))
    h_px = max(1, int(height_dbu * pixels_per_dbu))

    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (w_px, h_px), 0)
    drawer = ImageDraw.Draw(canvas)

    for layer_idx in layout.layer_indices():
        shapes = top_cell.shapes(layer_idx)
        for shape in shapes.each():
            if shape.is_polygon() or shape.is_box():
                poly = shape.polygon if shape.is_polygon() else shape.box.to_poly()
                points = []
                for point in poly.each_point():
                    px = int((point.x - bbox.left) * pixels_per_dbu)
                    py = int((point.y - bbox.bottom) * pixels_per_dbu)
                    px = max(0, min(px, w_px - 1))
                    py = max(0, min(py, h_px - 1))
                    points.append((px, py))
                if len(points) >= 3:
                    drawer.polygon(points, fill=255)

    raster = np.array(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(raster)
