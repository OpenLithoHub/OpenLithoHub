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
        True, "--compile/--no-compile", help="Wrap the Hopkins forward with torch.compile."
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
    layer: str | None = typer.Option(
        None,
        "--layer",
        help=(
            "OASIS/GDSII layer to rasterize, as 'LAYER:DTYPE' (e.g. '1:0'). "
            "Required for multi-layer files; ignored for raw .pt/.npy inputs."
        ),
    ),
    num_gpus: int = typer.Option(
        1,
        "--num-gpus",
        help=(
            "Number of worker processes for tile inference. 1 (default) keeps "
            "the sequential single-device path. >1 spawns one worker per GPU "
            "and shards tiles round-robin; falls back to CPU dispatch when "
            "fewer GPUs are visible than requested."
        ),
    ),
) -> None:
    """Run end-to-end mask optimization on a layout file.

    Example:
        openlithohub optimize --input chip.oas --model diffusion-ilt
        --writer mbmw --node 3nm-euv --drc-check --output optimized.oas
    """
    console = Console()

    if num_gpus < 1:
        raise typer.BadParameter("--num-gpus must be >= 1")

    from openlithohub.models.registry import register_builtin_models, registry

    register_builtin_models()
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

    requested_kwargs = _build_model_kwargs(pretrained, sha256)
    try:
        support = registry.supports_kwargs(model, requested_kwargs)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    if (pretrained or sha256 is not None) and not all(support.values()):
        console.print(
            f"[yellow]Warning:[/yellow] Model {model!r} does not support "
            "--pretrained / --sha256; ignoring."
        )

    litho_model = registry.get(model, **requested_kwargs)

    step = _StepCounter()

    console.print(f"[bold]Step {step.next()}:[/bold] Parsing layout...")
    try:
        layout_tensor = _load_layout_as_tensor(input, pixel_nm, layer=layer)
    except (ImportError, FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    console.print(f"  Layout size: {layout_tensor.shape[0]}x{layout_tensor.shape[1]} pixels")

    console.print(f"[bold]Step {step.next()}:[/bold] Tiling layout...")
    from openlithohub.workflow.tiling import Tile, stitch_tiles, tile_layout

    tiles = tile_layout(layout_tensor, tile_size=tile_size, overlap=overlap)
    console.print(f"  Generated {len(tiles)} tiles ({tile_size}px, overlap={overlap})")

    console.print(f"[bold]Step {step.next()}:[/bold] Running optimization...")
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
        if num_gpus == 1:
            litho_model.setup()
            try:
                for tile in tiles:
                    result = litho_model.predict(tile.tensor, **perf_kwargs)
                    tile_results.append((tile, result.mask))
                    progress.advance(task)
            finally:
                litho_model.teardown()
        else:
            from openlithohub.workflow.parallel import parallel_tile_inference

            tile_results = parallel_tile_inference(
                model_name=model,
                model_kwargs=requested_kwargs,
                tiles=tiles,
                num_gpus=num_gpus,
                base_perf_kwargs=perf_kwargs,
                progress_cb=lambda: progress.advance(task),
            )

    console.print(f"[bold]Step {step.next()}:[/bold] Stitching tiles...")
    h, w = layout_tensor.shape
    optimized = stitch_tiles(tile_results, (h, w))
    optimized = (optimized > 0.5).float()
    console.print(f"  Stitched output: {optimized.shape[0]}x{optimized.shape[1]}")

    if drc_check:
        console.print(f"[bold]Step {step.next()}:[/bold] Running compliance checks...")
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

    console.print(f"[bold]Step {step.next()}:[/bold] Exporting...")
    export_mode = "curvilinear" if writer == "mbmw" else "manhattan"

    try:
        from openlithohub.workflow.export import export_oasis

        export_oasis(optimized, output, mode=export_mode, pixel_size_nm=pixel_nm)
        console.print(f"  [green]Output written to {output}[/green]")
    except ImportError as e:
        console.print(f"  [yellow]Warning:[/yellow] {e}")
        console.print("  Falling back to raw tensor export...")
        fallback_path = output.with_suffix(".pt")
        torch.save(optimized, str(fallback_path))
        console.print(f"  Saved tensor to {fallback_path}")

    console.print()
    console.print("[bold green]Optimization complete.[/bold green]")


class _StepCounter:
    """Auto-increment counter for the human-readable Step N: log lines."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


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


def _load_layout_as_tensor(
    path: Path,
    pixel_nm: float,
    layer: str | None = None,
) -> torch.Tensor:
    """Load a layout file and rasterize to a binary tensor.

    For OASIS/GDSII inputs with more than one layer, ``layer`` must be set
    to a ``"LAYER:DTYPE"`` string (e.g. ``"1:0"``); otherwise the loader
    refuses rather than collapsing every layer onto the same mask.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pt":
        loaded = torch.load(str(path), weights_only=True)
        if not isinstance(loaded, torch.Tensor) or loaded.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2-D torch.Tensor for layout input, "
                f"got {type(loaded).__name__}"
                + (f" ndim={loaded.ndim}" if isinstance(loaded, torch.Tensor) else "")
            )
        return loaded.float()

    if suffix == ".npy":
        import numpy as np

        arr = np.load(str(path), allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2-D ndarray for layout input, got ndim={arr.ndim}"
            )
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

    top_cells = list(layout.top_cells())
    if not top_cells:
        raise ValueError(f"{path}: layout has no top cells.")
    top_cell = top_cells[0]
    bbox = top_cell.bbox()

    width_dbu = bbox.width()
    height_dbu = bbox.height()
    dbu_nm = layout.dbu * 1000.0
    pixels_per_dbu = dbu_nm / pixel_nm

    w_px = max(1, int(width_dbu * pixels_per_dbu))
    h_px = max(1, int(height_dbu * pixels_per_dbu))

    selected_layer_idx = _select_layer(layout, layer)

    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (w_px, h_px), 0)
    drawer = ImageDraw.Draw(canvas)

    shapes = top_cell.shapes(selected_layer_idx)

    def _project(point: Any) -> tuple[int, int]:
        px = int((point.x - bbox.left) * pixels_per_dbu)
        py = int((point.y - bbox.bottom) * pixels_per_dbu)
        return (max(0, min(px, w_px - 1)), max(0, min(py, h_px - 1)))

    for shape in shapes.each():
        if shape.is_polygon() or shape.is_box():
            poly = shape.polygon if shape.is_polygon() else shape.box.to_poly()

            hull = [_project(p) for p in poly.each_point_hull()]
            if len(hull) >= 3:
                drawer.polygon(hull, fill=255)
            for hole_idx in range(poly.holes()):
                hole = [_project(p) for p in poly.each_point_hole(hole_idx)]
                if len(hole) >= 3:
                    drawer.polygon(hole, fill=0)

    raster = np.array(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(raster)


def _select_layer(layout: Any, layer: str | None) -> int:
    """Resolve a CLI --layer 'NUM:DTYPE' to a klayout layer index.

    Refuses multi-layer files when the user did not specify a layer — the
    historical behavior of OR-ing every layer into one mask collapses
    multi-layer designs into nonsense input.
    """
    layer_indices = list(layout.layer_indices())
    if not layer_indices:
        raise ValueError("Layout contains no layers.")

    if layer is None:
        if len(layer_indices) > 1:
            available = ", ".join(
                f"{layout.get_info(idx).layer}:{layout.get_info(idx).datatype}"
                for idx in layer_indices
            )
            raise ValueError(
                f"Layout has {len(layer_indices)} layers; pass --layer LAYER:DTYPE "
                f"(available: [{available}])."
            )
        return int(layer_indices[0])

    if ":" not in layer:
        raise ValueError(f"--layer must be 'LAYER:DTYPE' (e.g. '1:0'); got {layer!r}")
    try:
        layer_num_s, dtype_s = layer.split(":", 1)
        layer_num = int(layer_num_s)
        dtype = int(dtype_s)
    except ValueError:
        raise ValueError(
            f"--layer must be 'LAYER:DTYPE' with integer components; got {layer!r}"
        ) from None

    for idx in layer_indices:
        info = layout.get_info(idx)
        if info.layer == layer_num and info.datatype == dtype:
            return int(idx)
    raise ValueError(f"Layer {layer!r} not found in layout.")
