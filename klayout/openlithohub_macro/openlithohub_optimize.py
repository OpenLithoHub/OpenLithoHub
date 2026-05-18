"""KLayout reference macro: run `openlithohub optimize` on the active selection.

This macro deliberately shells out to the user's regular Python interpreter
(`openlithohub` CLI on PATH) instead of importing the model in-process.
KLayout ships its own bundled Python without `pip`, so doing it that way
avoids the dep-packaging nightmare of injecting torch + openlithohub into
that interpreter. The cost is per-invocation startup latency (~1-2 s).
"""
from __future__ import annotations

import datetime
import shutil
import subprocess
import tempfile
from pathlib import Path

# `pya` is only available inside KLayout. Importing this module from CI
# (e.g. for a syntax check) must not fail, so the import is deferred to the
# functions that actually touch the GUI.


def _require_pya():  # type: ignore[no-untyped-def]
    try:
        import pya  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "This macro must run inside KLayout (pya unavailable)."
        ) from exc
    return pya


def _find_cli() -> str:
    cli = shutil.which("openlithohub")
    if cli is None:
        raise FileNotFoundError(
            "`openlithohub` CLI not found on PATH. Install with "
            "`pip install openlithohub` in the Python environment whose "
            "`python` KLayout can reach."
        )
    return cli


def _selection_bbox(cell_view):  # type: ignore[no-untyped-def]
    """Return the bounding box of the user's selection, or None if empty."""
    pya = _require_pya()
    sel = list(cell_view.view().each_object_selected())
    if not sel:
        return None
    bbox = pya.Box()
    for obj in sel:
        bbox += obj.shape.bbox().transformed(obj.trans)
    return bbox


def _write_tile_to_oas(cell_view, bbox, output: Path) -> None:  # type: ignore[no-untyped-def]
    """Dump the shapes inside `bbox` to an OASIS file at `output`."""
    pya = _require_pya()
    layout = cell_view.cellview(0).layout()
    options = pya.SaveLayoutOptions()
    options.format = "OASIS"
    options.clip_box = bbox
    layout.write(str(output), options)


def _read_result_into_view(cell_view, oas_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Read the optimized OASIS back and insert as a new top cell."""
    pya = _require_pya()
    layout = cell_view.cellview(0).layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_cell_name = f"OPC_{timestamp}"

    aux = pya.Layout()
    aux.read(str(oas_path))
    if aux.top_cell() is None:
        raise RuntimeError(f"Optimized OASIS at {oas_path} has no top cell.")

    new_cell = layout.create_cell(new_cell_name)
    new_cell.copy_tree(aux.top_cell())
    cell_view.view().select_cell(new_cell.cell_index(), 0)
    cell_view.view().zoom_fit()


def run_from_menu() -> None:  # type: ignore[no-untyped-def]
    """Menu entrypoint: select → write tile → run CLI → read back."""
    pya = _require_pya()
    cv = pya.Application.instance().main_window().current_view()
    if cv is None:
        pya.MessageBox.critical(
            "OpenLithoHub", "No layout open in the active view.", pya.MessageBox.Ok
        )
        return

    bbox = _selection_bbox(cv)
    if bbox is None:
        bbox = cv.box()  # fall back to the current viewport
        if bbox is None or bbox.empty():
            pya.MessageBox.critical(
                "OpenLithoHub", "Nothing selected and no viewport bbox.", pya.MessageBox.Ok
            )
            return

    try:
        cli = _find_cli()
    except FileNotFoundError as exc:
        pya.MessageBox.critical("OpenLithoHub", str(exc), pya.MessageBox.Ok)
        return

    with tempfile.TemporaryDirectory(prefix="openlithohub_") as tmpdir:
        tile_path = Path(tmpdir) / "tile.oas"
        result_path = Path(tmpdir) / "result.oas"
        _write_tile_to_oas(cv, bbox, tile_path)

        cmd = [
            cli,
            "optimize",
            "run",
            "--input", str(tile_path),
            "--output", str(result_path),
            "--model", "rule-based-opc",
            "--tile-size", "1024",
            "--pixel-nm", "1.0",
        ]
        pya.Logger.info(f"OpenLithoHub: running {' '.join(cmd)}")

        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if proc.stdout:
            pya.Logger.info(proc.stdout)
        if proc.returncode != 0:
            pya.MessageBox.critical(
                "OpenLithoHub",
                f"openlithohub optimize failed (exit {proc.returncode}):\n\n{proc.stderr}",
                pya.MessageBox.Ok,
            )
            return

        if not result_path.exists():
            pya.MessageBox.critical(
                "OpenLithoHub",
                f"CLI succeeded but no output file at {result_path}",
                pya.MessageBox.Ok,
            )
            return

        _read_result_into_view(cv, result_path)
        pya.Logger.info("OpenLithoHub: optimization complete")
