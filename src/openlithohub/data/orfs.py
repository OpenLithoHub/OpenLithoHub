"""ORFS artifact adapter — load ASAP7-routed RISC-V layouts as tile samples.

OpenROAD-flow-scripts (ORFS) is the open-source RTL→GDSII flow. Its
``flow/designs/asap7/<name>/`` configurations produce real ASAP7-routed
layouts (mock-alu, riscv32i, ibex, …) under
``flow/results/asap7/<name>/base/<name>.gds``.

Phase 3 of issue #4 wires those artifacts into OpenLithoHub. The
adapter rasterizes one design layer of the top cell, then cuts the
result into fixed-size tiles (2 µm or 5 µm by default — the windows
AI-OPC inference is benchmarked on). One ``LithoSample`` per tile.

Why tiling instead of one sample per block: a routed RISC-V ALU block
is hundreds of microns on a side, far too large for the Hopkins
forward model to evaluate as a single tensor. The ICCAD/AI-OPC
literature evaluates on ~2 µm and ~5 µm windows, and that's what the
issue spec (Phase 3) calls for.

License
-------
ORFS itself is BSD-3-Clause; the ``asap7`` platform underneath is also
BSD-3-Clause (same upstream as ``openlithohub.data.asap7``). The
adapter re-uses the ASAP7 license constants — there is no separate
ORFS data-license gate beyond the ASAP7 acknowledgement already
required when fetching the PDK.

This module never redistributes ORFS or ASAP7 bytes. The ``fetch()``
classmethod points at the ``build-asap7-mock-alu`` GitHub Actions
workflow that produces the GDS as a release-style artifact.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub.data._layers import LAYERS
from openlithohub.data.asap7 import (
    ASAP7_LICENSE,
    ASAP7_LICENSE_URL,
    rasterize_cell_layer,
)
from openlithohub.data.base import DatasetAdapter, LithoSample

# ORFS-routed ASAP7 layouts use the platform's stream-out map (defined
# under flow/platforms/asap7/). The post-route GDS numbers metal1 as
# layer 20/0 — not 10/0 like the cell-library source. Sourced from the
# central PDK layer registry (key ``orfs_asap7``) so the value lives in
# exactly one place. Verified against a fresh `make` of asap7/mock-alu
# (issue #4 Phase 3): the top cell has shapes on layers 19, 20, 30, 40,
# 50, 60, 70, with 20/0 being the densest (~45k shapes) — that's M1.
DEFAULT_DESIGN_LAYER: tuple[int, int] = LAYERS["orfs_asap7"].metal1

# AI-OPC inference windows. ICCAD16 hotspots are 1.2 µm; AI-OPC papers
# evaluate on 2 µm and 5 µm tiles. We expose both as canonical sizes.
DEFAULT_TILE_NM: float = 2000.0
SUPPORTED_TILE_NM: tuple[float, ...] = (2000.0, 5000.0)


def tile_design_tensor(
    design: np.ndarray,
    tile_nm: float,
    pixel_nm: float,
    stride_nm: float | None = None,
    drop_empty: bool = True,
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Cut a rasterized design into fixed-size tiles.

    Returns ``[(tile_array, (x_pixels, y_pixels)), ...]`` where the
    second element is the tile's lower-left corner in pixel
    coordinates of the parent design. Tiles smaller than the requested
    size at the right/top edges are dropped — keeping ragged tiles
    would force the eval harness to handle variable-size inputs.

    ``stride_nm`` defaults to ``tile_nm`` (non-overlapping grid).

    ``drop_empty=True`` skips all-zero tiles. Routed layouts have huge
    empty regions outside the core; emitting thousands of zero tiles
    would dominate runtime without producing useful metrics.
    """
    tile_px = max(1, int(round(tile_nm / pixel_nm)))
    stride_px = max(1, int(round((stride_nm if stride_nm is not None else tile_nm) / pixel_nm)))
    h, w = design.shape
    tiles: list[tuple[np.ndarray, tuple[int, int]]] = []
    for y in range(0, h - tile_px + 1, stride_px):
        for x in range(0, w - tile_px + 1, stride_px):
            t = design[y : y + tile_px, x : x + tile_px]
            if drop_empty and not t.any():
                continue
            tiles.append((t.copy(), (x, y)))
    return tiles


class OrfsArtifactDataset(DatasetAdapter):
    """Load an ORFS-produced ASAP7 layout, expose it as N tile samples.

    Args:
        gds_path: Path to a GDS file produced by ``ORFS make`` against
            an ``asap7/<design>`` config (e.g. ``mock-alu.gds``).
        cell_name: Optional explicit top-cell name. Defaults to the
            GDS file's basename (matches ORFS naming convention).
        design_layer: ``(layer, datatype)`` to rasterize. Defaults to
            metal1 (20, 0) — post-route ORFS-ASAP7 GDS numbers M1 as
            20/0, *not* 10/0 like the cell-library source. See the
            module docstring for the full layer-numbering caveat.
        pixel_nm: Raster pixel size in nm. Default 1.0; ASAP7 dbu is
            0.25 nm so the rasterizer downsamples 4×.
        tile_nm: Tile edge length in nm. Default 2000 (2 µm); also
            commonly 5000 (5 µm). Pass ``None`` to disable tiling and
            expose the whole block as a single sample (only feasible
            for very small designs).
        stride_nm: Tile stride. Defaults to ``tile_nm``
            (non-overlapping). Pass a smaller value for overlapping
            inference windows.
        drop_empty_tiles: Skip all-zero tiles. Default True.
        design_name: Optional human-readable design name for metadata
            (e.g. "mock-alu", "riscv32i"). Defaults to ``gds_path.stem``.
        orfs_revision: Optional ORFS git SHA recorded in metadata for
            reproducibility. Set this to the ``orfs_ref`` input of the
            ``build-asap7-mock-alu`` workflow that produced the GDS.
    """

    def __init__(
        self,
        gds_path: str | Path,
        cell_name: str | None = None,
        design_layer: tuple[int, int] = DEFAULT_DESIGN_LAYER,
        pixel_nm: float = 1.0,
        tile_nm: float | None = DEFAULT_TILE_NM,
        stride_nm: float | None = None,
        drop_empty_tiles: bool = True,
        design_name: str | None = None,
        orfs_revision: str | None = None,
    ) -> None:
        self.gds_path = Path(gds_path)
        if not self.gds_path.exists():
            raise FileNotFoundError(f"ORFS GDS not found: {self.gds_path}")
        if tile_nm is not None and tile_nm <= 0:
            raise ValueError(f"tile_nm must be positive or None, got {tile_nm!r}")
        self.cell_name = cell_name
        self.design_layer = design_layer
        self.pixel_nm = float(pixel_nm)
        self.tile_nm = tile_nm
        self.stride_nm = stride_nm
        self.drop_empty_tiles = drop_empty_tiles
        self.design_name = design_name or self.gds_path.stem
        self.orfs_revision = orfs_revision
        # Lazy: rasterize on first __getitem__ so constructor is cheap.
        self._design_arr: np.ndarray | None = None
        self._origin_nm: tuple[float, float] | None = None
        self._dbu_nm: float | None = None
        self._tiles: list[tuple[np.ndarray, tuple[int, int]]] | None = None

    def _ensure_loaded(self) -> None:
        if self._design_arr is not None:
            return
        import klayout.db as kdb

        layout = kdb.Layout()
        layout.read(str(self.gds_path))
        if self.cell_name is not None:
            cell = layout.cell(self.cell_name)
            if cell is None:
                available = sorted(c.name for c in layout.each_cell())[:10]
                raise KeyError(
                    f"Cell {self.cell_name!r} not found in {self.gds_path.name}. "
                    f"First 10 available: {available}"
                )
        else:
            top_cells = list(layout.top_cells())
            if not top_cells:
                raise ValueError(f"GDS {self.gds_path.name} has no top cells.")
            if len(top_cells) > 1:
                names = [c.name for c in top_cells]
                warnings.warn(
                    f"GDS {self.gds_path.name} has {len(top_cells)} top cells "
                    f"({names!r}); picking {names[0]!r}. Pass cell_name=... to "
                    "select explicitly.",
                    stacklevel=2,
                )
            cell = top_cells[0]
        design_arr, origin = rasterize_cell_layer(layout, cell, self.design_layer, self.pixel_nm)
        self._design_arr = design_arr
        self._origin_nm = origin
        self._dbu_nm = layout.dbu * 1000.0
        self._cell_name_resolved = cell.name
        if self.tile_nm is None:
            # Treat the whole block as a single "tile" at offset (0, 0).
            self._tiles = [(design_arr, (0, 0))]
        else:
            self._tiles = tile_design_tensor(
                design_arr,
                tile_nm=self.tile_nm,
                pixel_nm=self.pixel_nm,
                stride_nm=self.stride_nm,
                drop_empty=self.drop_empty_tiles,
            )

    def __len__(self) -> int:
        self._ensure_loaded()
        assert self._tiles is not None
        return len(self._tiles)

    def __getitem__(self, index: int) -> LithoSample:
        self._ensure_loaded()
        assert self._tiles is not None
        if index < 0 or index >= len(self._tiles):
            raise IndexError(f"Index {index} out of range [0, {len(self._tiles)})")
        tile_arr, (tx_px, ty_px) = self._tiles[index]
        ox_nm, oy_nm = self._origin_nm  # type: ignore[misc]
        tile_origin_nm = (
            ox_nm + tx_px * self.pixel_nm,
            oy_nm + ty_px * self.pixel_nm,
        )
        metadata: dict[str, Any] = {
            "dataset": "orfs",
            "pdk": "asap7",
            "design_name": self.design_name,
            "cell_name": self._cell_name_resolved,
            "source_gds": str(self.gds_path),
            "dbu_nm": self._dbu_nm,
            "pixel_nm": self.pixel_nm,
            "design_layer": list(self.design_layer),
            "tile_index": index,
            "tile_nm": self.tile_nm,
            "tile_origin_nm": [tile_origin_nm[0], tile_origin_nm[1]],
            "tile_pixels": list(tile_arr.shape[::-1]),  # (w, h)
            "license": ASAP7_LICENSE,
            "license_url": ASAP7_LICENSE_URL,
        }
        if self.orfs_revision is not None:
            metadata["orfs_revision"] = self.orfs_revision
        return LithoSample(
            design=torch.from_numpy(tile_arr).float(),
            mask=None,
            resist=None,
            metadata=metadata,
        )

    def download(self, root: str) -> None:
        """ORFS artifacts are produced by a CI workflow, not downloaded.

        See ``.github/workflows/build-asap7-mock-alu.yml`` — trigger it
        via ``gh workflow run build-asap7-mock-alu.yml`` and download
        the resulting GDS artifact. There is no remote URL to fetch.
        """
        raise RuntimeError(
            "OrfsArtifactDataset has no download() — the GDS comes from "
            "the build-asap7-mock-alu GitHub Actions workflow. Trigger "
            "it via `gh workflow run build-asap7-mock-alu.yml`, download "
            "the produced artifact, and pass its path to "
            "OrfsArtifactDataset(gds_path=...)."
        )
