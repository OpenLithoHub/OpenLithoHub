"""The `openlithohub export` subcommand — emit production-ready model artifacts.

The plan calls for `openlithohub export --model X --format onnx --output X.onnx`
specifically because Fab MDP (mask data prep) clusters are C++, not Python.
Exporting to ONNX / TorchScript is the single strongest signal that the
project is industrially deployable — not just a research toy.

Supported formats:

* ``onnx``        — ONNX graph, opset configurable.
* ``torchscript`` — saved ``ScriptModule`` (``.pt``).
* ``tensorrt``    — emits an ONNX file plus a ``trtexec`` command suggestion.
                    We intentionally do NOT call out to TensorRT here (it has
                    huge install footprint and requires a target GPU); shipping
                    the ONNX is sufficient for the Fab side to do the actual
                    TRT compile in their own pipeline.

Usage::

    openlithohub export --model neural-ilt \\
        --format onnx --output neural-ilt.onnx \\
        --shape 256x256 --pretrained
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
import typer
from rich.console import Console

export_app = typer.Typer(no_args_is_help=True)


@export_app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model name from the registry."),
    format: str = typer.Option(
        "onnx",
        "--format",
        "-f",
        help="Export format: onnx | torchscript | tensorrt.",
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path."),
    shape: str = typer.Option(
        "256x256",
        "--shape",
        help="Input HxW shape used to trace the model (e.g. '256x256').",
    ),
    opset: int = typer.Option(
        17,
        "--opset",
        help="ONNX opset version (ignored for non-ONNX formats).",
    ),
    pretrained: bool = typer.Option(
        False,
        "--pretrained/--no-pretrained",
        help="Load pretrained weights for the selected model (when supported).",
    ),
    device: str = typer.Option(
        "cpu", "--device", help="Torch device to trace on (cpu, cuda, mps)."
    ),
    dynamic_batch: bool = typer.Option(
        True,
        "--dynamic-batch/--static-batch",
        help="Mark the batch dimension as dynamic in the exported graph.",
    ),
) -> None:
    """Export a model to a production-friendly artifact (ONNX / TorchScript)."""
    console = Console()

    fmt = format.lower()
    if fmt not in ("onnx", "torchscript", "tensorrt"):
        raise typer.BadParameter(
            f"--format must be one of onnx | torchscript | tensorrt; got {format!r}"
        )

    h_w = _parse_shape(shape)

    # Importing each model module registers it; mirror what eval/optimize do.
    for mod in (
        "openlithohub.models.examples.dummy_model",
        "openlithohub.models.levelset_ilt",
        "openlithohub.models.neural_ilt",
        "openlithohub.models.rule_based_opc",
    ):
        importlib.import_module(mod)
    from openlithohub.models.registry import registry

    kwargs: dict[str, object] = {}
    if pretrained:
        kwargs["pretrained"] = True

    try:
        litho_model = registry.get(model, **kwargs)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    litho_model.setup()
    try:
        try:
            module = litho_model.to_torch_module()
        except NotImplementedError as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print(
                "  Try a model with a static forward graph (e.g. neural-ilt, layout-mae)."
            )
            raise typer.Exit(2) from None

        module = module.to(device).eval()
        dummy = torch.zeros(1, 1, h_w[0], h_w[1], device=device)

        output.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "torchscript":
            scripted = torch.jit.trace(module, dummy)  # type: ignore[no-untyped-call]
            scripted.save(str(output))
            console.print(f"[green]Saved TorchScript module to {output}[/green]")
            return

        if fmt in ("onnx", "tensorrt"):
            _export_onnx(module, dummy, output, opset, dynamic_batch, console)

        if fmt == "tensorrt":
            console.print()
            console.print("[bold]Next step (run on the target GPU host):[/bold]")
            console.print(
                f"  trtexec --onnx={output} --saveEngine={output.with_suffix('.trt')} "
                "--fp16  --minShapes=input:1x1x"
                f"{h_w[0]}x{h_w[1]}"
                " --optShapes=input:1x1x"
                f"{h_w[0]}x{h_w[1]}"
                " --maxShapes=input:8x1x"
                f"{h_w[0]}x{h_w[1]}"
            )
    finally:
        litho_model.teardown()


def _parse_shape(shape: str) -> tuple[int, int]:
    parts = shape.lower().split("x")
    if len(parts) != 2:
        raise typer.BadParameter(f"--shape must be 'HxW' (e.g. '256x256'); got {shape!r}")
    try:
        h, w = int(parts[0]), int(parts[1])
    except ValueError:
        raise typer.BadParameter(
            f"--shape must be 'HxW' with integer components; got {shape!r}"
        ) from None
    if h <= 0 or w <= 0:
        raise typer.BadParameter(f"--shape must be positive; got {shape!r}")
    return h, w


def _export_onnx(
    module: torch.nn.Module,
    dummy: torch.Tensor,
    output: Path,
    opset: int,
    dynamic_batch: bool,
    console: Console,
) -> None:
    # PyTorch 2.9 deprecated the TorchScript-based exporter; the
    # dynamo-based exporter (torch.export.export → ONNX) is now the
    # default. Try dynamo first when onnxscript is available, then fall
    # back to the legacy path — many existing models (any module that
    # can't be torch.export-ed) only work via the legacy path today.
    try:
        import onnxscript  # noqa: F401

        have_dynamo = True
    except ImportError:
        have_dynamo = False

    if have_dynamo:
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None
        if dynamic_batch:
            batch = torch.export.Dim("batch")
            dynamic_shapes = {"input": {0: batch}}
        try:
            torch.onnx.export(
                module,
                (dummy,),
                str(output),
                input_names=["input"],
                output_names=["output"],
                opset_version=opset,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
            )
            console.print(
                f"[green]Saved ONNX graph to {output}[/green] "
                f"(opset={opset}, dynamic_batch={dynamic_batch}, exporter=dynamo)"
            )
            return
        except Exception as e:
            msg = str(e)
            if "Module onnx is not installed" in msg or "No module named 'onnx" in msg:
                console.print(
                    "[red]Error:[/red] ONNX export needs the optional 'onnx' package. "
                    "Install with:\n  pip install 'openlithohub[export]'\n  # or: pip install onnx"
                )
                raise typer.Exit(3) from None
            console.print(
                f"[yellow]dynamo-based ONNX export failed ({type(e).__name__}); "
                "falling back to the deprecated TorchScript exporter.[/yellow]"
            )

    _export_onnx_legacy(module, dummy, output, opset, dynamic_batch, console)


def _export_onnx_legacy(
    module: torch.nn.Module,
    dummy: torch.Tensor,
    output: Path,
    opset: int,
    dynamic_batch: bool,
    console: Console,
) -> None:
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    try:
        torch.onnx.export(
            module,
            (dummy,),
            str(output),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as e:
        msg = str(e)
        if "Module onnx is not installed" in msg or "No module named 'onnx" in msg:
            console.print(
                "[red]Error:[/red] ONNX export needs the optional 'onnx' package. "
                "Install with:\n  pip install 'openlithohub[export]'\n  # or: pip install onnx"
            )
            raise typer.Exit(3) from None
        raise
    console.print(
        f"[green]Saved ONNX graph to {output}[/green] "
        f"(opset={opset}, dynamic_batch={dynamic_batch}, exporter=legacy)"
    )
