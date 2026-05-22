"""``openlithohub data`` — quick-look browser for bundled cell adapters.

Surfaces ``Asap7Dataset`` and ``FreePdk45SramDataset`` through the CLI so a
user can inspect available cells and rasterize a single cell to PNG without
writing Python. Intentionally narrow: this is a debugging / research-flow
helper, not the eval path. For benchmark runs use ``openlithohub eval run``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

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


def _save_png(arr: np.ndarray, out_path: Path) -> None:
    """Write a 2D float array in [0, 1] to a grayscale PNG at ``out_path``."""
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Flip Y so the GDS origin (bottom-left) is at the bottom of the PNG —
    # matches what reviewers expect when they cross-reference klayout.
    img = np.flipud(arr)
    img8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(out_path)


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
        ...,
        "--cell",
        "-c",
        help=(
            "Cell name. For asap7 you can pass shorthand ('INV', 'NAND2', "
            "'DFFHQN') and the resolver expands to the canonical "
            "'<FUNC>x1_ASAP7_75t_R' string; override with --drive/--flavor/--track. "
            "For freepdk45-sram pass the OpenRAM bundle stem (e.g. 'cell_1rw')."
        ),
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help=(
            "Where to write the rasterized design layer as a grayscale PNG. "
            "Defaults to '<cell>.png' in the current directory."
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
    """Render a single cell's design layer to a PNG for visual inspection."""
    from openlithohub.data.base import DatasetAdapter

    out_path = out if out is not None else Path(f"{cell}.png")
    layer_spec = _parse_design_layer(design_layer) if design_layer else None
    adapter: DatasetAdapter

    if dataset == "asap7":
        from openlithohub.data.asap7 import (
            ASAP7_LICENSE,
            ASAP7_LICENSE_URL,
            DEFAULT_DESIGN_LAYER,
            Asap7Dataset,
            resolve_cell_name,
        )

        if data_root is None:
            raise typer.BadParameter("--data-root is required for --dataset asap7")
        if not accept_license:
            raise typer.BadParameter(
                f"--dataset asap7 requires --accept-license: ASAP7 ships under "
                f"{ASAP7_LICENSE}; see {ASAP7_LICENSE_URL}."
            )
        canonical = resolve_cell_name(cell, drive=drive, flavor=flavor, track=track)
        adapter = Asap7Dataset(
            root=data_root,
            cells=(canonical,),
            design_layer=layer_spec or DEFAULT_DESIGN_LAYER,
            pixel_nm=pixel_nm,
        )
    else:  # freepdk45-sram
        from openlithohub.data.freepdk45_sram import (
            DEFAULT_DESIGN_LAYER as SRAM_DEFAULT_LAYER,
        )
        from openlithohub.data.freepdk45_sram import (
            FreePdk45SramDataset,
        )

        adapter = FreePdk45SramDataset(
            cells=(cell,),
            design_layer=layer_spec or SRAM_DEFAULT_LAYER,
            pixel_nm=pixel_nm,
        )

    sample = adapter[0]
    arr = sample.design.detach().cpu().numpy()
    _save_png(arr, out_path)

    md = sample.metadata
    h, w = arr.shape
    typer.echo(
        f"cell={md.get('cell_name', cell)} shape={(h, w)} "
        f"layer={md.get('design_layer')} pixel_nm={md.get('pixel_nm')} "
        f"license={md.get('license')} -> {out_path}"
    )
