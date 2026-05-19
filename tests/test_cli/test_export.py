"""Tests for `openlithohub export` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from openlithohub.cli.app import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_export_torchscript_neural_ilt(runner: CliRunner, tmp_path: Path) -> None:
    """TorchScript export of neural-ilt should work without optional ONNX deps."""
    out = tmp_path / "neural-ilt.pt"
    result = runner.invoke(
        app,
        [
            "export",
            "run",
            "--model",
            "neural-ilt",
            "--format",
            "torchscript",
            "--output",
            str(out),
            "--shape",
            "32x32",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    # The saved artifact is a real ScriptModule we can load and call.
    loaded = torch.jit.load(str(out))
    y = loaded(torch.zeros(1, 1, 32, 32))
    assert y.shape == (1, 1, 32, 32)


def test_export_rejects_unexportable_model(runner: CliRunner, tmp_path: Path) -> None:
    """Models without a static forward graph (e.g. levelset-ilt) refuse export."""
    out = tmp_path / "levelset.pt"
    result = runner.invoke(
        app,
        [
            "export",
            "run",
            "--model",
            "levelset-ilt",
            "--format",
            "torchscript",
            "--output",
            str(out),
            "--shape",
            "32x32",
        ],
    )
    assert result.exit_code == 2
    assert "does not support export" in result.output


def test_export_rejects_bad_shape(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "x.pt"
    result = runner.invoke(
        app,
        [
            "export",
            "run",
            "--model",
            "neural-ilt",
            "--format",
            "torchscript",
            "--output",
            str(out),
            "--shape",
            "garbage",
        ],
    )
    assert result.exit_code != 0
    assert "shape" in result.output.lower()


def test_export_onnx_neural_ilt(runner: CliRunner, tmp_path: Path) -> None:
    """ONNX export of neural-ilt produces a valid graph with a dynamic batch axis."""
    onnx = pytest.importorskip("onnx")
    out = tmp_path / "neural-ilt.onnx"
    result = runner.invoke(
        app,
        [
            "export",
            "run",
            "--model",
            "neural-ilt",
            "--format",
            "onnx",
            "--output",
            str(out),
            "--shape",
            "32x32",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    # Batch dim should be symbolic when --dynamic-batch is on (the default).
    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    assert in_dims[0].dim_param == "batch"


def test_export_rejects_bad_format(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "x.bin"
    result = runner.invoke(
        app,
        [
            "export",
            "run",
            "--model",
            "neural-ilt",
            "--format",
            "coreml",  # not supported
            "--output",
            str(out),
            "--shape",
            "32x32",
        ],
    )
    assert result.exit_code != 0
    assert "format" in result.output.lower()


def test_export_onnx_dynamo_path_used_when_module_supports_it(tmp_path: Path) -> None:
    """The dynamo-based exporter (PyTorch 2.9+ default) is used when onnxscript
    is available and the module is torch.export-compatible. A trivial Linear
    module covers both."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from rich.console import Console

    from openlithohub.cli.export_cmd import _export_onnx

    out = tmp_path / "linear.onnx"
    module = torch.nn.Linear(8, 8).eval()
    dummy = torch.zeros(1, 8)
    console = Console(record=True)
    _export_onnx(module, dummy, out, opset=18, dynamic_batch=True, console=console)

    text = console.export_text()
    assert "exporter=dynamo" in text, text
    assert out.exists()
