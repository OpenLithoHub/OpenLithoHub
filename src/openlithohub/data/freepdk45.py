"""FreePDK45 + NanGate Open Cell Library adapter (single-GDS standard cells).

FreePDK45 is NCSU's 45nm open-source predictive PDK; NanGate's Open Cell
Library provides the standard cells designed against it. The mflowgen
ASIC design kit at https://github.com/mflowgen/freepdk-45nm bundles the
two together as a convenience drop, including a single ``stdcells.gds``
file with all 135 NanGate cells.

This adapter loads a small canonical list of cells (INV_X1, NAND2_X1,
NOR2_X1, DFF_X1) by name and rasterizes one design layer per cell. The
default is metal1 = (11, 0) per the kit's ``rtk-stream-out.map`` (note:
this is *not* the same numbering as ASAP7, where metal1 = (10, 0)).

License caveat
--------------
Unlike ASAP7's clean BSD-3-Clause, the FreePDK45 distribution is two
licenses stacked:

- **FreePDK45** (NCSU): see https://eda.ncsu.edu/freepdk/freepdk45/.
- **NanGate Open Cell Library** (Si2): see
  https://si2.org/open-cell-library/.

The mflowgen mirror at ``github.com/mflowgen/freepdk-45nm`` does not
ship a top-level LICENSE file, so callers MUST verify the upstream
terms themselves before redistributing any derivative work. As with
ASAP7, ``fetch()`` requires explicit ``accept_license=True`` to
acknowledge this responsibility, and the adapter never bundles PDK
bytes into the OpenLithoHub repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from openlithohub.data._layers import LAYERS
from openlithohub.data.asap7 import rasterize_cell_layer
from openlithohub.data.base import DatasetAdapter, LithoSample

FREEPDK45_UPSTREAM_URL = "https://github.com/mflowgen/freepdk-45nm"
FREEPDK45_LICENSE = "FreePDK45 (see NCSU EDA Wiki) + NanGate Open Cell Library (see Si2)"
FREEPDK45_LICENSE_URL = "https://eda.ncsu.edu/freepdk/freepdk45/"
NANGATE_LICENSE_URL = "https://si2.org/open-cell-library/"

# Path inside the upstream tree to the bundled NanGate stdcells GDS.
_GDS_RELATIVE = "stdcells.gds"

# Canonical small standard cells for the smoke-test benchmark. NanGate's
# `*_X1` drives are the lowest-strength version of each function.
CANONICAL_CELLS: tuple[str, ...] = (
    "INV_X1",
    "NAND2_X1",
    "NOR2_X1",
    "DFF_X1",
)

# Default design layer = metal1 (11, 0) per FreePDK45's
# rtk-stream-out.map. Sourced from the central PDK layer registry so
# the value is asserted in exactly one place. Override via
# ``design_layer`` for metal2 (13, 0) or higher routing layers.
DEFAULT_DESIGN_LAYER: tuple[int, int] = LAYERS["freepdk45"].metal1


class FreePdk45Dataset(DatasetAdapter):
    """Adapter for FreePDK45 + NanGate standard cells via mflowgen mirror.

    Args:
        root: Path to a local clone of ``mflowgen/freepdk-45nm``. Use
            ``FreePdk45Dataset.fetch(root, accept_license=True)`` to
            create one.
        cells: Cell names to expose, in order. Defaults to
            ``CANONICAL_CELLS``.
        design_layer: ``(layer, datatype)`` to rasterize as the design
            tensor. Defaults to metal1 (11, 0) per
            ``rtk-stream-out.map``.
        pixel_nm: Raster pixel size in nm. Defaults to 1.0; the
            FreePDK45 dbu is 0.1 nm so this is a 10× downsample.
        gds_path: Optional explicit override for the GDS file path. If
            unset, the adapter looks for ``stdcells.gds`` directly under
            ``root``.

    The adapter requires ``klayout`` (already pinned in pyproject.toml).
    """

    def __init__(
        self,
        root: str | Path,
        cells: tuple[str, ...] | list[str] | None = None,
        design_layer: tuple[int, int] = DEFAULT_DESIGN_LAYER,
        pixel_nm: float = 1.0,
        gds_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"FreePDK45 root not found: {self.root}")
        from openlithohub._utils.integrity import warn_unverified_data_root

        warn_unverified_data_root(self.root, "freepdk45")
        self.design_layer = design_layer
        self.pixel_nm = float(pixel_nm)
        self.cells: tuple[str, ...] = tuple(cells) if cells is not None else CANONICAL_CELLS
        self._gds_path = Path(gds_path) if gds_path is not None else self.root / _GDS_RELATIVE
        if not self._gds_path.exists():
            raise FileNotFoundError(
                f"FreePDK45 GDS not found at {self._gds_path}. "
                f"Did you clone {FREEPDK45_UPSTREAM_URL} into {self.root}?"
            )
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

        layout = kdb.Layout()
        layout.read(str(self._gds_path))
        cell = layout.cell(name)
        if cell is None:
            available = sorted(c.name for c in layout.each_cell())[:10]
            raise KeyError(
                f"Cell {name!r} not found in {self._gds_path.name}. First 10 available: {available}"
            )

        design_arr, origin = rasterize_cell_layer(layout, cell, self.design_layer, self.pixel_nm)

        metadata: dict[str, Any] = {
            "dataset": "freepdk45",
            "pdk": "freepdk45",
            "pdk_variant": "nangate-openlib",
            "cell_name": name,
            "source_gds": str(self._gds_path),
            "dbu_nm": layout.dbu * 1000.0,
            "pixel_nm": self.pixel_nm,
            "design_layer": list(self.design_layer),
            "origin_nm": [origin[0], origin[1]],
            "license": FREEPDK45_LICENSE,
            "license_url": FREEPDK45_LICENSE_URL,
            "secondary_license_url": NANGATE_LICENSE_URL,
        }

        return LithoSample(
            design=torch.from_numpy(design_arr).float(),
            mask=None,
            resist=None,
            metadata=metadata,
        )

    def download(self, root: str) -> None:
        """Always rejected — use ``fetch()`` instead.

        The base ``DatasetAdapter.download`` signature has no place for
        the license-acknowledgement flag this PDK requires.
        """
        raise RuntimeError(
            "FreePdk45Dataset.download() is intentionally unimplemented "
            "because FreePDK45 + NanGate require explicit license "
            "acknowledgement. Use "
            "`FreePdk45Dataset.fetch(root, accept_license=True)` instead."
        )

    # ---- Croissant metadata ----

    def croissant_name(self) -> str:
        return "FreePDK45"

    def croissant_description(self) -> str:
        return (
            "FreePDK45 is the NCSU 45nm predictive academic PDK paired with NanGate "
            "Open Cell Library. Cell layouts are rasterised on-the-fly for OPC / mask "
            "optimisation research."
        )

    def croissant_license_url(self) -> str | None:
        return FREEPDK45_LICENSE_URL

    def croissant_url(self) -> str | None:
        return "https://eda.ncsu.edu/freepdk/freepdk45/"

    def croissant_citation(self) -> str | None:
        return (
            "Stine, J. E., et al. FreePDK: An Open-Source Variation-Aware Design Kit. "
            "IEEE MSE 2007."
        )

    @classmethod
    def fetch(
        cls,
        root: str | Path,
        accept_license: bool = False,
    ) -> None:
        """Clone the mflowgen FreePDK45 mirror to ``root``.

        FreePDK45 + NanGate ships under a stacked license that the
        mflowgen mirror does *not* declare in a LICENSE file. Callers
        must independently verify both upstream terms before
        redistributing any derivative work, and the adapter requires
        ``accept_license=True`` to acknowledge that responsibility.

        Per ``DATA-LICENSES.md``, OpenLithoHub does not redistribute PDK
        bytes — this method only clones from the mflowgen mirror on the
        user's own machine.
        """
        if not accept_license:
            raise RuntimeError(
                f"FreePDK45 ships under a stacked license: {FREEPDK45_LICENSE}. "
                f"Read the terms at {FREEPDK45_LICENSE_URL} (FreePDK45) and "
                f"{NANGATE_LICENSE_URL} (NanGate OCL) and call "
                f"fetch(..., accept_license=True) to confirm you will comply "
                f"with both."
            )
        target = Path(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(
            f"Cloning FreePDK45 / NanGate OCL into {target} from "
            f"{FREEPDK45_UPSTREAM_URL}\n"
            f"  Verify upstream terms: {FREEPDK45_LICENSE_URL}\n"
            f"                          {NANGATE_LICENSE_URL}\n"
        )
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                FREEPDK45_UPSTREAM_URL,
                str(target),
            ],
            check=True,
        )
