"""ASAP7 predictive PDK adapter (standard cells from a single GDS).

ASAP7 (Clark et al., ASU) ships its standard-cell library as a single GDSII
file containing every cell as a top-level cell in the layout — there is no
per-cell file to read. The canonical release lives at
``https://github.com/The-OpenROAD-Project/asap7`` under BSD-3-Clause; the
7.5-track regular-Vt cells are in submodule ``asap7sc7p5t_27`` at
``GDS/asap7sc7p5t_27_R_*.gds``.

This adapter loads a small canonical list of cells (INVx1, NAND2x1,
NOR2x1, DFFHQNx1) by name and rasterizes one design layer per cell into a
``LithoSample.design`` tensor. The layer choice is configurable; the
default is M1 (10/0), which is the densest mask layer foundry reviewers
ask about first. The cell selection is intentionally narrow — Phase 1 of
issue #4 is a smoke-test, not a full library benchmark.

Per ``DATA-LICENSES.md``, this adapter does **not** redistribute any PDK
bytes. Users must clone the upstream repository themselves and pass the
local path. ``download()`` is a guarded helper that ``git clone``s the
upstream repo only after the caller passes ``accept_license=True``,
acknowledging the BSD-3-Clause attribution requirement.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample

ASAP7_UPSTREAM_URL = "https://github.com/The-OpenROAD-Project/asap7"
ASAP7_LICENSE = "BSD-3-Clause"
ASAP7_LICENSE_URL = "https://github.com/The-OpenROAD-Project/asap7/blob/master/LICENSE"

# Path inside the upstream tree to the regular-Vt 7.5-track GDS file.
# Glob pattern accommodates the date suffix (`_R_201211.gds` today, may
# tick forward in future releases).
_GDS_RELATIVE_GLOB = "asap7sc7p5t_27/GDS/asap7sc7p5t_27_R_*.gds"

# Canonical small standard cells for the smoke-test benchmark. Names are
# top-level cell names in the GDS, verified against the May-2026 head of
# the asap7sc7p5t_27 submodule.
CANONICAL_CELLS: tuple[str, ...] = (
    "INVx1_ASAP7_75t_R",
    "NAND2x1_ASAP7_75t_R",
    "NOR2x1_ASAP7_75t_R",
    "DFFHQNx1_ASAP7_75t_R",
)

# Default design layer to rasterize. ASAP7 GDS uses unnamed layer numbers;
# 10/0 is M1 in the asap7sc7p5t_27 layer map. Override via `design_layer`
# if you want M2 (12/0), M3 (14/0), etc.
DEFAULT_DESIGN_LAYER: tuple[int, int] = (10, 0)


def rasterize_cell_layer(
    layout: Any,
    cell: Any,
    layer_spec: tuple[int, int],
    pixel_nm: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterize one (layer, datatype) of a klayout cell into a {0,1} array.

    Decomposes each polygon into trapezoids via klayout's
    ``Polygon.decompose_trapezoids`` and fills each trapezoid's pixel
    footprint. For Manhattan polygons every trapezoid is an axis-aligned
    rectangle, so the fill is exact even for L-shapes — a plain bbox fill
    would over-fill the concave corner.

    Returns ``(array, origin_nm)`` where ``origin_nm`` is the cell bbox
    lower-left corner in nm, suitable for storing in
    ``LithoSample.metadata['origin_nm']``.

    This helper is shared with future PDK adapters (FreePDK45 in Phase 2);
    keep it generic — no ASAP7-specific assumptions.
    """
    import klayout.db as kdb

    layer_index = layout.find_layer(*layer_spec)
    bbox = cell.bbox()
    dbu_nm = layout.dbu * 1000.0
    origin = (bbox.left * dbu_nm, bbox.bottom * dbu_nm)
    w = max(1, int(np.ceil(bbox.width() * dbu_nm / pixel_nm)))
    h = max(1, int(np.ceil(bbox.height() * dbu_nm / pixel_nm)))
    if layer_index is None:
        return np.zeros((h, w), dtype=np.float32), origin

    arr = np.zeros((h, w), dtype=np.float32)
    ox_nm, oy_nm = origin

    def _fill_box(b: Any) -> None:
        x0 = b.left * dbu_nm - ox_nm
        y0 = b.bottom * dbu_nm - oy_nm
        x1 = b.right * dbu_nm - ox_nm
        y1 = b.top * dbu_nm - oy_nm
        i0 = max(0, int(np.floor(x0 / pixel_nm)))
        j0 = max(0, int(np.floor(y0 / pixel_nm)))
        i1 = min(w, int(np.ceil(x1 / pixel_nm)))
        j1 = min(h, int(np.ceil(y1 / pixel_nm)))
        if i1 > i0 and j1 > j0:
            arr[j0:j1, i0:i1] = 1.0

    for shape_obj in cell.shapes(layer_index).each():
        if shape_obj.is_box():
            _fill_box(shape_obj.box)
            continue
        if shape_obj.is_path():
            poly = shape_obj.path.polygon()
        elif shape_obj.is_polygon():
            poly = shape_obj.polygon
        else:
            continue
        try:
            trapezoids = list(poly.decompose_trapezoids(kdb.Polygon.TD_simple))
        except AttributeError:
            _fill_box(poly.bbox())
            continue
        for tz in trapezoids:
            _fill_box(tz.bbox())

    return arr, origin


class Asap7Dataset(DatasetAdapter):
    """Adapter for the ASAP7 predictive PDK standard cells.

    Args:
        root: Path to a local clone of ``The-OpenROAD-Project/asap7``
            with the ``asap7sc7p5t_27`` submodule initialised. Use
            ``Asap7Dataset.download(root, accept_license=True)`` to
            create one.
        cells: Cell names to expose, in order. Defaults to
            ``CANONICAL_CELLS``.
        design_layer: ``(layer, datatype)`` to rasterize as the design
            tensor. Defaults to M1 (10, 0).
        pixel_nm: Raster pixel size in nm. Defaults to 1.0 to match the
            existing OpenLithoHub grid; ASAP7's manufacturing dbu is
            0.25 nm so this is a 4× downsample.
        gds_path: Optional explicit override for the GDS file path. If
            unset, the adapter globs ``asap7sc7p5t_27/GDS/...`` under
            ``root`` and picks the lexicographically last match.

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
            raise FileNotFoundError(f"ASAP7 root not found: {self.root}")
        self.design_layer = design_layer
        self.pixel_nm = float(pixel_nm)
        self.cells: tuple[str, ...] = tuple(cells) if cells is not None else CANONICAL_CELLS
        self._gds_path = Path(gds_path) if gds_path is not None else self._resolve_gds_path()
        if not self._gds_path.exists():
            raise FileNotFoundError(
                f"ASAP7 GDS not found at {self._gds_path}. "
                f"Did you run `git submodule update --init asap7sc7p5t_27` "
                f"under {self.root}?"
            )
        self._cache: dict[str, LithoSample] = {}

    def _resolve_gds_path(self) -> Path:
        matches = sorted(self.root.glob(_GDS_RELATIVE_GLOB))
        if not matches:
            raise FileNotFoundError(
                f"No GDS matching {_GDS_RELATIVE_GLOB!r} under {self.root}. "
                f"Initialise the asap7sc7p5t_27 submodule."
            )
        return matches[-1]

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
            "dataset": "asap7",
            "pdk": "asap7",
            "pdk_variant": "asap7sc7p5t_27_R",
            "cell_name": name,
            "source_gds": str(self._gds_path),
            "dbu_nm": layout.dbu * 1000.0,
            "pixel_nm": self.pixel_nm,
            "design_layer": list(self.design_layer),
            "origin_nm": [origin[0], origin[1]],
            "license": ASAP7_LICENSE,
            "license_url": ASAP7_LICENSE_URL,
        }

        return LithoSample(
            design=torch.from_numpy(design_arr).float(),
            mask=None,
            resist=None,
            metadata=metadata,
        )

    def download(self, root: str) -> None:
        """Clone ASAP7 to ``root``. Always rejected — use ``fetch()`` instead.

        The base ``DatasetAdapter.download`` signature has no place for the
        license-acknowledgement flag this PDK requires, so this method is a
        guard that points the caller at ``Asap7Dataset.fetch()``.
        """
        raise RuntimeError(
            "Asap7Dataset.download() is intentionally unimplemented because "
            "ASAP7 (BSD-3-Clause) requires explicit license acknowledgement. "
            "Use `Asap7Dataset.fetch(root, accept_license=True)` instead."
        )

    @classmethod
    def fetch(
        cls,
        root: str | Path,
        accept_license: bool = False,
    ) -> None:
        """Clone the ASAP7 repo with submodules to ``root``.

        ASAP7 ships under BSD-3-Clause. The license requires attribution
        in any redistribution; ``accept_license=True`` is the caller's
        explicit acknowledgement that they have read the license at
        ``ASAP7_LICENSE_URL`` and will comply with the attribution
        requirement when sharing derived layouts.

        Per ``DATA-LICENSES.md``, OpenLithoHub does not redistribute PDK
        bytes — this method only clones from the official upstream
        source on the user's own machine.
        """
        if not accept_license:
            raise RuntimeError(
                f"ASAP7 is licensed under {ASAP7_LICENSE}. Read the terms at "
                f"{ASAP7_LICENSE_URL} and call fetch(..., accept_license=True) "
                f"to confirm you will comply with the attribution requirement."
            )
        target = Path(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(
            f"Cloning ASAP7 ({ASAP7_LICENSE}) into {target} from {ASAP7_UPSTREAM_URL}\n"
        )
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--recurse-submodules",
                "--shallow-submodules",
                ASAP7_UPSTREAM_URL,
                str(target),
            ],
            check=True,
        )
