"""``openlithohub data`` — quick-look browser for bundled cell adapters.

Surfaces ``Asap7Dataset`` and ``FreePdk45SramDataset`` through the CLI so a
user can inspect available cells and rasterize a single cell to PNG without
writing Python. Intentionally narrow: this is a debugging / research-flow
helper, not the eval path. For benchmark runs use ``openlithohub eval run``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import typer

from openlithohub.data.base import DatasetAdapter

data_app = typer.Typer(
    help="Inspect and render samples from bundled cell-library adapters.",
    no_args_is_help=True,
)

_KNOWN_DATASETS = ("asap7", "freepdk45-sram")


def _validate_dataset(dataset: str) -> str:
    if dataset not in _KNOWN_DATASETS:
        raise typer.BadParameter(
            f"Unknown dataset {dataset!r}. Choose from: {', '.join(_KNOWN_DATASETS)}"
        )
    return dataset


def _parse_design_layer(spec: str) -> tuple[int, int]:
    try:
        layer, datatype = spec.split("/", 1)
        return int(layer), int(datatype)
    except (ValueError, AttributeError) as exc:
        raise typer.BadParameter(
            f"--design-layer must be 'LAYER/DATATYPE' (e.g. '10/0'), got {spec!r}"
        ) from exc


def _save_png(arr: np.ndarray[Any, Any], out_path: Path) -> None:
    """Write a 2D float array in [0, 1] to a grayscale PNG at ``out_path``."""
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Flip Y so the GDS origin (bottom-left) is at the bottom of the PNG —
    # matches what reviewers expect when they cross-reference klayout.
    img = np.flipud(arr)
    img8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(out_path)


def _build_adapter(
    dataset: str,
    *,
    cells: tuple[str, ...],
    layer_spec: tuple[int, int] | None,
    pixel_nm: float,
    data_root: Path | None,
    accept_license: bool,
) -> DatasetAdapter:
    """Construct an ``Asap7Dataset`` or ``FreePdk45SramDataset`` for ``cells``.

    Validation of the license gate / data-root requirements lives here so
    ``show`` and ``show --all`` share the same error messages.
    """
    if dataset == "asap7":
        from openlithohub.data.asap7 import (
            ASAP7_LICENSE,
            ASAP7_LICENSE_URL,
            DEFAULT_DESIGN_LAYER,
            Asap7Dataset,
        )

        if data_root is None:
            raise typer.BadParameter("--data-root is required for --dataset asap7")
        if not accept_license:
            raise typer.BadParameter(
                f"--dataset asap7 requires --accept-license: ASAP7 ships under "
                f"{ASAP7_LICENSE}; see {ASAP7_LICENSE_URL}."
            )
        return Asap7Dataset(
            root=data_root,
            cells=cells,
            design_layer=layer_spec or DEFAULT_DESIGN_LAYER,
            pixel_nm=pixel_nm,
        )

    # freepdk45-sram
    from openlithohub.data.freepdk45_sram import (
        DEFAULT_DESIGN_LAYER as SRAM_DEFAULT_LAYER,
    )
    from openlithohub.data.freepdk45_sram import (
        FreePdk45SramDataset,
    )

    return FreePdk45SramDataset(
        cells=cells,
        design_layer=layer_spec or SRAM_DEFAULT_LAYER,
        pixel_nm=pixel_nm,
    )


def _canonical_cells(dataset: str) -> tuple[str, ...]:
    if dataset == "asap7":
        from openlithohub.data.asap7 import CANONICAL_CELLS

        return CANONICAL_CELLS
    from openlithohub.data.freepdk45_sram import CANONICAL_CELLS as SRAM_CELLS

    return SRAM_CELLS


def _format_sample_line(
    cell_label: str, arr: np.ndarray[Any, Any], md: dict[str, Any], out_path: Path
) -> str:
    h, w = arr.shape
    return (
        f"cell={md.get('cell_name', cell_label)} shape={(h, w)} "
        f"layer={md.get('design_layer')} pixel_nm={md.get('pixel_nm')} "
        f"license={md.get('license')} -> {out_path}"
    )


@data_app.command("list")
def list_cmd(
    dataset: str = typer.Argument(
        ..., help="Dataset id: 'asap7' or 'freepdk45-sram'.", callback=_validate_dataset
    ),
) -> None:
    """List the canonical cells the adapter exposes by default.

    Prints one cell name per line — script-friendly. License and upstream
    source URL go to stderr so piping ``| sort`` / ``| wc -l`` still works.
    """
    if dataset == "asap7":
        from openlithohub.data.asap7 import (
            ASAP7_LICENSE,
            ASAP7_LICENSE_URL,
            ASAP7_UPSTREAM_URL,
            CANONICAL_CELLS,
        )

        typer.echo(
            f"# asap7  license={ASAP7_LICENSE}  upstream={ASAP7_UPSTREAM_URL}  "
            f"license_url={ASAP7_LICENSE_URL}",
            err=True,
        )
        for name in CANONICAL_CELLS:
            typer.echo(name)
        return

    # freepdk45-sram
    from openlithohub.data.freepdk45 import FREEPDK45_LICENSE, FREEPDK45_LICENSE_URL
    from openlithohub.data.freepdk45_sram import (
        CANONICAL_CELLS as SRAM_CELLS,
    )
    from openlithohub.data.freepdk45_sram import (
        OPENRAM_LICENSE,
    )

    typer.echo(
        f"# freepdk45-sram  pdk_license={FREEPDK45_LICENSE}  "
        f"tooling_license={OPENRAM_LICENSE}  pdk_license_url={FREEPDK45_LICENSE_URL}",
        err=True,
    )
    for name in SRAM_CELLS:
        typer.echo(name)


@data_app.command("show")
def show_cmd(
    dataset: str = typer.Argument(
        ..., help="Dataset id: 'asap7' or 'freepdk45-sram'.", callback=_validate_dataset
    ),
    cell: str = typer.Option(
        None,
        "--cell",
        "-c",
        help=(
            "Cell name. For asap7 you can pass shorthand ('INV', 'NAND2', "
            "'DFFHQN') and the resolver expands to the canonical "
            "'<FUNC>x1_ASAP7_75t_R' string; override with --drive/--flavor/--track. "
            "For freepdk45-sram pass the OpenRAM bundle stem (e.g. 'cell_1rw'). "
            "Required unless --all is set."
        ),
    ),
    all_cells: bool = typer.Option(
        False,
        "--all",
        help=(
            "Render every cell in the adapter's CANONICAL_CELLS list. "
            "Mutually exclusive with --cell. With --all, --out is treated as "
            "an output directory and defaults to './<dataset>-cells/'."
        ),
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help=(
            "Single cell: PNG path (default '<cell>.png' in cwd). "
            "With --all: output directory (default '<dataset>-cells/' in cwd); "
            "one '<cell>.png' is written per cell."
        ),
    ),
    data_root: Path = typer.Option(
        None,
        "--data-root",
        "-r",
        help=(
            "Path to a local ASAP7 clone (required for --dataset asap7). "
            "Use `Asap7Dataset.fetch(root, accept_license=True)` to create one."
        ),
    ),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help="Required for --dataset asap7. Acknowledges BSD-3-Clause attribution.",
    ),
    design_layer: str = typer.Option(
        None,
        "--design-layer",
        help=(
            "GDS layer to rasterize, formatted as 'LAYER/DATATYPE' (e.g. '10/0'). "
            "Defaults: asap7 → 10/0 (M1), freepdk45-sram → 11/0 (metal1)."
        ),
    ),
    pixel_nm: float = typer.Option(
        1.0, "--pixel-nm", help="Raster pixel size in nm. Defaults to 1.0."
    ),
    drive: str = typer.Option(
        "x1", "--drive", help="ASAP7 drive-strength suffix for shorthand resolution (default x1)."
    ),
    flavor: str = typer.Option(
        "R", "--flavor", help="ASAP7 flavor for shorthand resolution: R / L / SL / SRAM."
    ),
    track: str = typer.Option(
        "75", "--track", help="ASAP7 track count for shorthand resolution: 75 or 6."
    ),
) -> None:
    """Render a cell's design layer to a PNG for visual inspection.

    Default mode renders one cell. ``--all`` renders every entry in the
    adapter's CANONICAL_CELLS into an output directory — useful for
    populating documentation galleries or batch-inspecting an adapter
    before a benchmark run.
    """
    if all_cells and cell is not None:
        raise typer.BadParameter("--cell and --all are mutually exclusive")
    if not all_cells and cell is None:
        raise typer.BadParameter(
            "--cell is required (or pass --all to render every canonical cell)"
        )

    layer_spec = _parse_design_layer(design_layer) if design_layer else None

    if all_cells:
        out_dir = out if out is not None else Path(f"{dataset}-cells")
        cells = _canonical_cells(dataset)
        adapter = _build_adapter(
            dataset,
            cells=cells,
            layer_spec=layer_spec,
            pixel_nm=pixel_nm,
            data_root=data_root,
            accept_license=accept_license,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, requested in enumerate(cells):
            sample = adapter[i]
            arr = sample.design.detach().cpu().numpy()
            md = sample.metadata
            # Use the resolved canonical name for the filename when available
            # (ASAP7 stores it; FreePDK45-SRAM cell_name == requested name).
            stem = str(md.get("cell_name", requested))
            png_path = out_dir / f"{stem}.png"
            _save_png(arr, png_path)
            typer.echo(_format_sample_line(requested, arr, md, png_path))
        typer.echo(f"# {len(cells)} cells -> {out_dir}", err=True)
        return

    # Single-cell mode. ``cell`` is non-None: validated above.
    out_path = out if out is not None else Path(f"{cell}.png")

    if dataset == "asap7":
        from openlithohub.data.asap7 import resolve_cell_name

        canonical = resolve_cell_name(cell, drive=drive, flavor=flavor, track=track)
        adapter = _build_adapter(
            dataset,
            cells=(canonical,),
            layer_spec=layer_spec,
            pixel_nm=pixel_nm,
            data_root=data_root,
            accept_license=accept_license,
        )
    else:
        adapter = _build_adapter(
            dataset,
            cells=(cell,),
            layer_spec=layer_spec,
            pixel_nm=pixel_nm,
            data_root=data_root,
            accept_license=accept_license,
        )

    sample = adapter[0]
    arr = sample.design.detach().cpu().numpy()
    _save_png(arr, out_path)
    typer.echo(_format_sample_line(cell, arr, sample.metadata, out_path))


@data_app.command("export")
def export_cmd(
    dataset: str = typer.Argument(
        ..., help="Dataset id: 'asap7' or 'freepdk45-sram'.", callback=_validate_dataset
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Directory to write shards into. Created if missing.",
    ),
    fmt: str = typer.Option(
        "webdataset",
        "--format",
        "-f",
        help="Shard format: 'webdataset' (.tar) or 'parquet'.",
    ),
    shards: int = typer.Option(
        None,
        "--shards",
        help="Number of shards to write. Mutually exclusive with --shard-size.",
    ),
    shard_size: str = typer.Option(
        None,
        "--shard-size",
        help=(
            "Target size per shard, e.g. '1GB', '500MB'. Derives shard count "
            "from a probe of the first sample. Mutually exclusive with --shards."
        ),
    ),
    dataset_tag: str = typer.Option(
        None,
        "--tag",
        help="Per-record key prefix. Defaults to the adapter's croissant_name lowercased.",
    ),
    compression: str = typer.Option(
        "snappy",
        "--compression",
        help="Parquet compression codec (parquet only): snappy, gzip, zstd, none.",
    ),
    data_root: Path = typer.Option(
        None,
        "--data-root",
        "-r",
        help="Path to a local ASAP7 clone (required for --dataset asap7).",
    ),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help="Required for --dataset asap7. Acknowledges BSD-3-Clause attribution.",
    ),
    design_layer: str = typer.Option(
        None,
        "--design-layer",
        help="GDS layer to rasterize, formatted as 'LAYER/DATATYPE' (e.g. '10/0').",
    ),
    pixel_nm: float = typer.Option(
        1.0, "--pixel-nm", help="Raster pixel size in nm. Defaults to 1.0."
    ),
) -> None:
    """Export an adapter's canonical cells as WebDataset (.tar) or Parquet shards.

    Produces ``shard-NNNNN.{tar,parquet}`` plus a sibling ``croissant.json``
    with dataset-level metadata. Shard layout is deterministic: sample ``i``
    lands in shard ``i % N``, so re-runs reproduce the same output.

    For foundation-model pretraining: prefer ``webdataset`` for streaming
    pipelines (PyTorch ``IterableDataset``), ``parquet`` for table-shaped
    workflows (HF ``datasets.load_dataset``, polars, duckdb).
    """
    from openlithohub.data.exporters import (
        ParquetExporter,
        WebdatasetExporter,
        parse_size,
    )

    if fmt not in ("webdataset", "parquet"):
        raise typer.BadParameter(f"--format must be 'webdataset' or 'parquet', got {fmt!r}")
    if shards is not None and shard_size is not None:
        raise typer.BadParameter("--shards and --shard-size are mutually exclusive")

    shard_size_bytes = parse_size(shard_size) if shard_size else None

    layer_spec = _parse_design_layer(design_layer) if design_layer else None
    cells = _canonical_cells(dataset)
    adapter = _build_adapter(
        dataset,
        cells=cells,
        layer_spec=layer_spec,
        pixel_nm=pixel_nm,
        data_root=data_root,
        accept_license=accept_license,
    )

    if fmt == "webdataset":
        exporter: WebdatasetExporter | ParquetExporter = WebdatasetExporter(
            adapter,
            output_dir,
            dataset_tag=dataset_tag,
            shards=shards,
            shard_size_bytes=shard_size_bytes,
        )
    else:
        exporter = ParquetExporter(
            adapter,
            output_dir,
            dataset_tag=dataset_tag,
            shards=shards,
            shard_size_bytes=shard_size_bytes,
            compression=compression,
        )

    paths = exporter.export()
    for p in paths:
        typer.echo(str(p))
    typer.echo(
        f"# wrote {len(paths)} shard(s) of {len(adapter)} sample(s) to {output_dir}",
        err=True,
    )
