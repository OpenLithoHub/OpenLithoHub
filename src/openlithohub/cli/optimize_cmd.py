"""The `openlithohub optimize` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
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
    overlap: int | None = typer.Option(
        None,
        "--overlap",
        help=(
            "Tile overlap (halo) in pixels. Mutually exclusive with --halo. "
            "Prefer --halo unless you need a fixed pixel count for back-compat "
            "with pre-RFC-0005 scripts."
        ),
    ),
    halo: str = typer.Option(
        "auto",
        "--halo",
        help=(
            "Tile halo size: 'auto' (default) computes max(optical_radius, "
            "model_receptive_field) from the process node and model, or pass "
            "an integer pixel count. Mutually exclusive with --overlap."
        ),
    ),
    pixel_nm: float | None = typer.Option(
        None,
        "--pixel-nm",
        help=(
            "Pixel size in nanometers. When omitted, the value is taken from "
            "the selected --node (e.g. 0.5 nm for 3nm-euv). Pass an explicit "
            "value to override the node default; '1.0' is treated as a real "
            "value, not a sentinel — same contract as POST /v1/optimize."
        ),
    ),
    threshold: float = typer.Option(
        0.225,
        "--threshold",
        help=(
            "Final mask binarisation threshold. Defaults to 0.225 to match the "
            "LithoBench/Yang2023 evaluation calibration so the optimised "
            "output is scored on the same threshold the benchmark assumes. "
            "Pass --threshold 0.5 for the legacy mid-grey cut."
        ),
    ),
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
            "fewer GPUs are visible than requested. Multi-GPU runs are dense "
            "only (legacy parallel path)."
        ),
    ),
    export_min_area: float = typer.Option(
        0.0,
        "--export-min-area",
        help=(
            "Drop curvilinear shapes below this polygon area (nm^2) at export. "
            "Default 0.0 keeps every shape so academic / Hackathon scoring "
            "stays bit-exact. Set >0 for fab-oriented exports where MRC would "
            "reject the smallest SRAFs an ILT can produce."
        ),
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic/--no-deterministic",
        help=(
            "Force bit-reproducible torch backends (cudnn.deterministic=True, "
            "cudnn.benchmark=False, allow_tf32=False). Slower but required "
            "when two identical optimize runs must produce identical masks."
        ),
    ),
    execution_mode: str = typer.Option(
        "auto",
        "--execution-mode",
        help=(
            "Execution topology. 'auto' (default): dense under the dense "
            "memory policy (OPENLITHOHUB_MAX_DENSE_BYTES bytes per fp32 "
            "raster; 0 = unlimited), streaming via run_streaming for "
            "supported large jobs, and FAIL CLOSED for unsupported large "
            "jobs instead of silently materializing a full-chip raster. "
            "'dense' forces the legacy blend path. 'streaming' forces the "
            "exact-core streaming path (input .npy/.gds/.oas; output "
            "streaming-capable — see --output)."
        ),
    ),
) -> None:
    """Run end-to-end mask optimization on a layout file.

    Example:
        openlithohub optimize --input chip.oas --model diffusion-ilt
        --writer mbmw --node 3nm-euv --drc-check --output optimized.oas

    Output artifact: an .oas/.gds path writes a mask-writer layout;
    an .npy path writes the binarised raster artifact (memmap-backed,
    never fully materialised in RAM, when the planner picks streaming).
    """
    console = Console()

    if deterministic:
        from openlithohub._utils.determinism import set_deterministic

        set_deterministic()

    if num_gpus < 1:
        raise typer.BadParameter("--num-gpus must be >= 1")
    if execution_mode not in ("auto", "dense", "streaming"):
        raise typer.BadParameter(
            f"--execution-mode must be 'auto', 'dense' or 'streaming'; got {execution_mode!r}"
        )

    from openlithohub.models.registry import register_builtin_models, registry

    register_builtin_models()
    from openlithohub.workflow.process_node import PROCESS_NODES

    # Auto-configure from process node
    node_config = None
    if node in PROCESS_NODES:
        from openlithohub.workflow.process_node import get_node

        node_config = get_node(node)
        if pixel_nm is None:
            pixel_nm = node_config.pixel_size_nm
    if pixel_nm is None:
        pixel_nm = 1.0
    if writer not in ("mbmw", "vsb"):
        console.print(f"[red]Error:[/red] --writer must be 'mbmw' or 'vsb', got {writer!r}")
        raise typer.Exit(1)

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

    try:
        effective_overlap = _resolve_halo(
            halo=halo,
            overlap=overlap,
            node_config=node_config,
            litho_model=litho_model,
            pixel_nm=pixel_nm,
            tile_size=tile_size,
            console=console,
        )
    except (ValueError, typer.BadParameter) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2) from None

    perf_kwargs = _build_perf_kwargs(device, dtype, compile_forward)

    if num_gpus > 1:
        if execution_mode == "streaming":
            console.print(
                "[red]Error:[/red] --num-gpus > 1 uses the legacy dense parallel "
                "path; streaming execution is single-device. Use --num-gpus 1."
            )
            raise typer.Exit(2) from None
        _run_dense_parallel_legacy(
            input=input,
            output=output,
            model=model,
            model_kwargs=requested_kwargs,
            num_gpus=num_gpus,
            pixel_nm=pixel_nm,
            layer=layer,
            tile_size=tile_size,
            overlap=effective_overlap,
            threshold=threshold,
            export_min_area=export_min_area,
            writer=writer,
            drc_check=drc_check,
            perf_kwargs=perf_kwargs,
            console=console,
        )
        return

    # ---- single-device: the shared product execution spine ---------------
    from openlithohub.workflow.execution import (
        OptimizeRequest,
        coerce_execution_mode,
        optimize_layout,
        plan_request,
    )

    # An .npy output path selects the raster artifact (memmap-backed on the
    # streaming branch); anything else is a mask-writer layout file.
    output_kind: Literal["oasis", "raster-npy"] = (
        "raster-npy" if output.suffix.lower() == ".npy" else "oasis"
    )
    request = OptimizeRequest(
        model=litho_model,
        pixel_size_nm=pixel_nm,
        input_path=input,
        output_path=output,
        output_kind=output_kind,
        writer=writer,
        layer=layer,
        node=node_config,
        tile_size=tile_size,
        halo_px=effective_overlap,
        threshold=threshold,
        min_area_nm2=export_min_area,
        execution_mode=coerce_execution_mode(execution_mode),
        forward_kwargs=perf_kwargs,
    )

    step = _StepCounter()
    console.print(f"[bold]Step {step.next()}:[/bold] Planning execution...")
    try:
        probe, plan = plan_request(request)
    except (ImportError, FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    console.print(f"  Layout size: {probe.shape[0]}x{probe.shape[1]} pixels ({probe.kind})")
    console.print(
        f"  Plan: {plan.mode.upper()} [{plan.reason}]"
        + (f" — {plan.detail}" if plan.detail else "")
    )
    console.print(
        f"  Memory policy: estimated dense raster {plan.estimated_dense_bytes} B / "
        f"budget {plan.max_dense_bytes} B" + (" (OVER)" if plan.over_memory_policy else "")
    )
    console.print(f"  Backends: input={plan.input_backend} output={plan.output_backend}")

    console.print(f"[bold]Step {step.next()}:[/bold] Running optimization...")
    litho_model.setup()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing tiles", total=None)

            def on_progress(done: int, total: int) -> None:
                progress.update(task, total=total, completed=done)

            result = optimize_layout(request, progress=on_progress, prepared=(probe, plan))
    except (ImportError, FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None
    finally:
        litho_model.teardown()

    if result.plan.mode == "streaming":
        accounting = result.work_accounting or {}
        console.print(
            "  Streaming: "
            f"{result.n_tiles} trusted cores, "
            f"{int(accounting.get('forward_simulator_input_pixels', 0))} forwarded px, "
            f"{int(accounting.get('screened_out_pixels', 0))} screened px"
        )

    raster: torch.Tensor | None = result.mask
    if drc_check:
        console.print(f"[bold]Step {step.next()}:[/bold] Running compliance checks...")
        if raster is None and result.output_format == "npy" and result.output_path:
            console.print("  Loading raster artifact into RAM for compliance checks...")
            raster = torch.from_numpy(np.load(str(result.output_path), mmap_mode="r")[:]).float()
        if raster is None:
            console.print(
                "  [yellow]Skipping:[/yellow] streaming Manhattan export retains no "
                "raster; rerun with an .npy output (or --execution-mode dense) to "
                "run DRC/MRC."
            )
        else:
            from openlithohub.benchmark.compliance.drc import check_drc
            from openlithohub.benchmark.compliance.mrc import check_mrc

            mrc_result = check_mrc(raster, pixel_size_nm=pixel_nm)
            drc_result = check_drc(raster, pixel_size_nm=pixel_nm)

            if mrc_result.passed and drc_result.passed:
                console.print("  [green]All checks passed[/green]")
            else:
                if not mrc_result.passed:
                    console.print(
                        f"  [yellow]MRC:[/yellow] {mrc_result.violation_count} violations "
                        f"(rate={mrc_result.violation_rate:.4f})"
                    )
                if not drc_result.passed:
                    console.print(
                        f"  [yellow]DRC:[/yellow] {drc_result.violation_count} violations"
                    )

    console.print(f"[bold]Step {step.next()}:[/bold] Exporting...")
    if result.output_path is not None:
        console.print(
            f"  [{result.output_format}] Output written to {result.output_path}"
            + (
                f"  ({result.streaming_report.n_rectangles} Manhattan rectangles)"
                if result.output_format == "oasis"
                and result.plan.mode == "streaming"
                and result.streaming_report is not None
                else ""
            )
        )
    else:  # pragma: no cover — every CLI output kind writes a file
        console.print("  No output artifact produced")

    console.print()
    console.print("[bold green]Optimization complete.[/bold green]")


def _run_dense_parallel_legacy(
    *,
    input: Path,
    output: Path,
    model: str,
    model_kwargs: dict[str, Any],
    num_gpus: int,
    pixel_nm: float,
    layer: str | None,
    tile_size: int,
    overlap: int,
    threshold: float,
    export_min_area: float,
    writer: str,
    drc_check: bool,
    perf_kwargs: dict[str, Any],
    console: Console,
) -> None:
    """Legacy multi-GPU dense path (parallel tile inference across GPUs).

    Kept out of the shared spine deliberately: it shards tiles across
    worker processes, which the sequential planners/executors do not own.
    Dense-only by contract (enforced by the caller).
    """
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing tiles", total=len(tiles))
        from openlithohub.workflow.parallel import parallel_tile_inference

        tile_results = parallel_tile_inference(
            model_name=model,
            model_kwargs=model_kwargs,
            tiles=tiles,
            num_gpus=num_gpus,
            base_perf_kwargs=perf_kwargs,
            progress_cb=lambda: progress.advance(task),
        )

    console.print(f"[bold]Step {step.next()}:[/bold] Stitching tiles...")
    h, w = layout_tensor.shape
    optimized = stitch_tiles(tile_results, (h, w))
    optimized = (optimized > threshold).float()
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

        export_oasis(
            optimized,
            output,
            mode=export_mode,
            pixel_size_nm=pixel_nm,
            min_area_nm2=export_min_area,
        )
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


def _resolve_halo(
    *,
    halo: str,
    overlap: int | None,
    node_config: Any,
    litho_model: Any,
    pixel_nm: float,
    tile_size: int,
    console: Console,
) -> int:
    """Resolve --halo / --overlap into a single tile-overlap pixel count.

    Rules:
      - --halo and --overlap are mutually exclusive when both are
        explicit. Detect explicitness via "halo != 'auto'" and
        "overlap is not None".
      - --halo "auto": compute from process node OIR + model RF.
      - --halo N: integer pixel count.
      - --overlap N: legacy fixed pixel count, kept for back-compat.
      - Otherwise: auto.
    """
    from openlithohub.workflow.halo import compute_halo_px, describe_halo

    halo_explicit = halo != "auto"
    overlap_explicit = overlap is not None

    if halo_explicit and overlap_explicit:
        raise typer.BadParameter(
            "--halo and --overlap are mutually exclusive. "
            "Prefer --halo (auto or integer); --overlap is kept only for "
            "scripts that pre-date RFC 0005."
        )

    if overlap_explicit:
        assert overlap is not None  # narrowed by overlap_explicit
        if overlap < 0:
            raise typer.BadParameter(f"--overlap must be >= 0; got {overlap}")
        if overlap >= tile_size:
            raise typer.BadParameter(
                f"--overlap {overlap} >= --tile-size {tile_size}; overlap must "
                "be smaller than the tile."
            )
        console.print(f"  Halo: {overlap} px (from --overlap)")
        return int(overlap)

    if halo_explicit:
        try:
            halo_px = int(halo)
        except ValueError as exc:
            raise typer.BadParameter(
                f"--halo must be 'auto' or a non-negative integer; got {halo!r}"
            ) from exc
        if halo_px < 0:
            raise typer.BadParameter(f"--halo must be >= 0; got {halo_px}")
        if halo_px >= tile_size:
            raise typer.BadParameter(
                f"--halo {halo_px} >= --tile-size {tile_size}; halo must be smaller than the tile."
            )
        console.print(f"  Halo: {halo_px} px (≈{halo_px * pixel_nm:.0f} nm) — explicit --halo")
        return halo_px

    halo_px = compute_halo_px(
        node=node_config,
        model=litho_model,
        pixel_nm=pixel_nm,
        tile_size=tile_size,
    )
    console.print(f"  Halo: {describe_halo(halo_px, node_config, litho_model, pixel_nm)}")
    return halo_px


def _load_layout_as_tensor(
    path: Path,
    pixel_nm: float,
    layer: str | None = None,
) -> torch.Tensor:
    """Backwards-compatible alias for :func:`openlithohub.data.io.load_layout`.

    The canonical loader now lives at ``openlithohub.data.io.load_layout``.
    This shim is kept for callers that still import the underscored name
    (the HTTP server and the API façade have been updated to use the
    public name; remove this shim once the v0.1 import surface is dropped).
    """
    from openlithohub.data.io import load_layout

    return load_layout(path, pixel_nm, layer=layer)


def _select_layer(layout: Any, layer: str | None) -> int:
    """Backwards-compatible alias for :func:`openlithohub.data.io._select_layer`."""
    from openlithohub.data.io import _select_layer as _impl

    return _impl(layout, layer)
