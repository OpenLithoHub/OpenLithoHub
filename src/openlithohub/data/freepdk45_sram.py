"""FreePDK45 SRAM bitcell adapter — load OpenRAM's bundled GDS as samples.

OpenRAM (BSD-3-Clause, ``pip install openram``) ships a small set of
hand-crafted FreePDK45 standard cells under ``technology/freepdk45/gds_lib/``:

- ``cell_1rw.gds``        — 6T 1-port SRAM bitcell
- ``cell_2rw.gds``        — 8T dual-port SRAM bitcell
- ``dff.gds``             — D-flip-flop
- ``sense_amp.gds``       — sense amplifier
- ``write_driver.gds``    — write driver
- ``tri_gate.gds``        — tri-state gate
- ``replica_cell_{1,2}rw.gds`` — timing-replica columns
- ``dummy_cell_{1,2}rw.gds``   — row/column edge dummies

These are the exact cells OpenRAM compiles together to build a full SRAM
macro on FreePDK45. Each GDS contains a single top cell whose name matches
the file stem.

This adapter rasterizes one design layer per cell (default: metal1
(11, 0)) and emits one ``LithoSample`` per cell — directly addressing
issue #4 Phase 3's SRAM-bitcell-tile data goal without running OpenRAM's
compiler. The compile path (``sram_compiler.py``) currently has a
numpy-2 scalar-conversion regression in upstream OpenRAM 1.2.48; the
pre-shipped GDS files are the canonical, citation-worthy artifact and
sidestep that bug entirely.

Layer numbering matches the FreePDK45 stream-out map (``layers.map`` in
the openram package): metal1 = 11/0, identical to the mflowgen NanGate
mirror, so the central registry's ``LAYERS["freepdk45"].metal1`` covers
both.

License
-------
- OpenRAM:     BSD-3-Clause (https://github.com/VLSIDA/OpenRAM)
- FreePDK45:   academic / non-commercial (NCSU EDA Wiki).

The adapter does not redistribute either set of bytes — it locates the
bundled GDS via ``importlib.resources.files("openram")`` at runtime, so
the user's pip-installed ``openram`` wheel is the source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from openlithohub.data._layers import LAYERS
from openlithohub.data.asap7 import rasterize_cell_layer
from openlithohub.data.base import DatasetAdapter, LithoSample
from openlithohub.data.freepdk45 import (
    FREEPDK45_LICENSE,
    FREEPDK45_LICENSE_URL,
)

OPENRAM_LICENSE = "BSD-3-Clause (OpenRAM, VLSIDA group at UCSC)"
OPENRAM_LICENSE_URL = "https://github.com/VLSIDA/OpenRAM/blob/master/LICENSE"

# Default design layer: metal1 (11, 0) per FreePDK45 layers.map.
DEFAULT_DESIGN_LAYER: tuple[int, int] = LAYERS["freepdk45"].metal1

# Canonical cells in OpenRAM's bundled gds_lib. The order matches what a
# user typically wants for an SRAM-tile benchmark: 1-port bitcell first
# (the central feature), then peripheral support cells.
CANONICAL_CELLS: tuple[str, ...] = (
    "cell_1rw",
    "cell_2rw",
    "dff",
    "sense_amp",
    "write_driver",
    "tri_gate",
    "replica_cell_1rw",
    "replica_cell_2rw",
    "dummy_cell_1rw",
    "dummy_cell_2rw",
)


def _locate_openram_gds_lib() -> Path:
    """Return the absolute path to OpenRAM's bundled ``freepdk45/gds_lib`` dir.

    Raises a clear ``ImportError`` with installation instructions when
    ``openram`` is not installed — it is an *optional* dependency of
    OpenLithoHub (declared under ``project.optional-dependencies.freepdk45-sram``),
    not a core one.
    """
    try:
        import importlib.resources as ir

        # ``files()`` returns a Traversable; for installed wheels this is
        # a real Path. We need a Path because klayout's reader takes a
        # filesystem path string.
        root = Path(str(ir.files("openram")))
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "FreePdk45SramDataset requires the optional 'openram' package. "
            "Install it with `pip install 'openlithohub[freepdk45-sram]'` "
            "or `pip install openram`."
        ) from exc
    gds_lib = root / "technology" / "freepdk45" / "gds_lib"
    if not gds_lib.is_dir():
        raise FileNotFoundError(
            f"Expected OpenRAM FreePDK45 GDS bundle at {gds_lib}, but it "
            "does not exist. Reinstall openram or check the package layout."
        )
    return gds_lib


class FreePdk45SramDataset(DatasetAdapter):
    """Adapter for OpenRAM's bundled FreePDK45 SRAM-cell GDS files.

    Args:
        cells: Cell names to expose, in order. Defaults to
            ``CANONICAL_CELLS`` (all 10 cells in the bundle).
        design_layer: ``(layer, datatype)`` to rasterize as the design
            tensor. Defaults to metal1 (11, 0) per FreePDK45's
            ``layers.map``.
        pixel_nm: Raster pixel size in nm. Defaults to 1.0; FreePDK45's
            dbu is 0.5 nm so this is a 2× downsample.
        gds_lib_path: Optional explicit path to OpenRAM's ``gds_lib``
            directory. If unset, auto-located via ``importlib.resources``.

    Each ``LithoSample`` has ``mask=None`` and ``resist=None`` — these
    are unmasked design-layer rasterizations, suitable as inputs to OPC
    / mask-optimization research, not paired training data.
    """

    def __init__(
        self,
        cells: Sequence[str] | None = None,
        design_layer: tuple[int, int] = DEFAULT_DESIGN_LAYER,
        pixel_nm: float = 1.0,
        gds_lib_path: str | Path | None = None,
    ) -> None:
        if pixel_nm <= 0:
            raise ValueError(f"pixel_nm must be positive, got {pixel_nm!r}")
        self.design_layer = design_layer
        self.pixel_nm = float(pixel_nm)
        self.cells: tuple[str, ...] = tuple(cells) if cells is not None else CANONICAL_CELLS
        self._gds_lib = (
            Path(gds_lib_path) if gds_lib_path is not None else _locate_openram_gds_lib()
        )
        if not self._gds_lib.is_dir():
            raise FileNotFoundError(f"FreePDK45 SRAM gds_lib not found: {self._gds_lib}")
        self._cache: dict[str, LithoSample] = {}

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, index: int) -> LithoSample:
        if index < 0 or index >= len(self.cells):
            raise IndexError(f"Index {index} out of range [0, {len(self.cells)})")
        name = self.cells[index]
        if name in self._cache:
            return self._cache[name]
        sample = self._load_cell(name)
        self._cache[name] = sample
        return sample

    def _load_cell(self, name: str) -> LithoSample:
        import klayout.db as kdb

        gds_path = self._gds_lib / f"{name}.gds"
        if not gds_path.exists():
            available = sorted(p.stem for p in self._gds_lib.glob("*.gds"))
            raise KeyError(f"Cell {name!r} not found in {self._gds_lib}. Available: {available}")
        layout = kdb.Layout()
        layout.read(str(gds_path))
        cell = layout.cell(name)
        if cell is None:
            # OpenRAM convention: each GDS top cell name = file stem.
            # If that ever drifts, fall back to the unique top cell.
            tops = list(layout.top_cells())
            if len(tops) != 1:
                names = [c.name for c in tops]
                raise KeyError(
                    f"Cell {name!r} not present in {gds_path.name}; expected "
                    f"exactly one top cell matching the file stem, found {names!r}."
                )
            cell = tops[0]

        design_arr, origin = rasterize_cell_layer(layout, cell, self.design_layer, self.pixel_nm)

        metadata: dict[str, Any] = {
            "dataset": "freepdk45-sram",
            "pdk": "freepdk45",
            "pdk_variant": "openram-bundled",
            "cell_name": cell.name,
            "source_gds": str(gds_path),
            "dbu_nm": layout.dbu * 1000.0,
            "pixel_nm": self.pixel_nm,
            "design_layer": list(self.design_layer),
            "origin_nm": [origin[0], origin[1]],
            "license": FREEPDK45_LICENSE,
            "license_url": FREEPDK45_LICENSE_URL,
            "tooling_license": OPENRAM_LICENSE,
            "tooling_license_url": OPENRAM_LICENSE_URL,
        }

        return LithoSample(
            design=torch.from_numpy(design_arr).float(),
            mask=None,
            resist=None,
            metadata=metadata,
        )

    def download(self, root: str) -> None:
        """No-op — the GDS bundle ships in the ``openram`` pip wheel.

        Install via ``pip install 'openlithohub[freepdk45-sram]'`` or
        ``pip install openram``; the adapter then locates the bundle
        automatically via ``importlib.resources``.
        """
        raise RuntimeError(
            "FreePdk45SramDataset has no download() — the GDS bundle ships "
            "inside the openram pip wheel. Install via `pip install openram` "
            "(or `pip install 'openlithohub[freepdk45-sram]'`) and the "
            "adapter will locate it automatically."
        )

    # ---- Croissant metadata ----

    def croissant_name(self) -> str:
        return "FreePDK45-SRAM-OpenRAM"

    def croissant_description(self) -> str:
        return (
            "FreePDK45 SRAM-cell GDS files bundled with OpenRAM (1RW / 2RW "
            "bitcells, sense amp, write driver, DFF, replica and dummy "
            "cells). Each cell is rasterised on one design layer for OPC / "
            "mask-optimisation research."
        )

    def croissant_license_url(self) -> str | None:
        return FREEPDK45_LICENSE_URL

    def croissant_url(self) -> str | None:
        return "https://github.com/VLSIDA/OpenRAM"

    def croissant_citation(self) -> str | None:
        return (
            "Guthaus, M. R., Stine, J. E., Ataei, S., et al. "
            "OpenRAM: An Open-Source Memory Compiler. ICCAD 2016."
        )
