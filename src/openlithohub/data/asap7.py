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

from openlithohub.data._layers import LAYERS
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

# Default design layer to rasterize. Sourced from the central PDK
# layer registry (see openlithohub.data._layers) so the (10, 0) value
# is asserted in exactly one place across the codebase.
DEFAULT_DESIGN_LAYER: tuple[int, int] = LAYERS["asap7"].metal1


# Recognised flavors and tracks for ``resolve_cell_name``. Verified
# against the asap7sc7p5t_28 LEF on 2026-05-22:
#   <FUNC><DRIVE>_ASAP7_<TRACK>t_<FLAVOR>
# where TRACK ∈ {6, 75} (6T or 7.5T library) and
#       FLAVOR ∈ {R (regular-Vt), L (low-Vt), SL (super-low-Vt), SRAM}.
_ASAP7_FLAVORS: frozenset[str] = frozenset({"R", "L", "SL", "SRAM"})
_ASAP7_TRACKS: frozenset[str] = frozenset({"6", "75"})


def resolve_cell_name(
    shorthand: str,
    *,
    drive: str = "x1",
    flavor: str = "R",
    track: str = "75",
) -> str:
    """Expand a function-name shorthand into the ASAP7 canonical cell name.

    The ASAP7 stdcell library names every cell as
    ``<FUNC><DRIVE>_ASAP7_<TRACK>t_<FLAVOR>``. Issue spec language often
    uses the bare function name (``"INV"``, ``"NAND2"``, ``"DFFHQN"``);
    this helper composes the canonical string a downstream reader
    actually expects, with sensible defaults for drive / flavor / track.

    Args:
        shorthand: Function name, with or without trailing ``x...`` drive
            spec. ``"INV"``, ``"INVx1"``, and ``"INVx1_ASAP7_75t_R"`` all
            resolve identically. Already-canonical names pass through
            unchanged.
        drive: Drive-strength suffix (``"x1"``, ``"x2"``, ``"xp33"``,
            ``"x1p5"``, ...). Used only when ``shorthand`` does not
            already include one. ``p`` is the decimal separator
            (e.g. ``"xp5"`` is ½×).
        flavor: ``"R"`` (regular-Vt, default), ``"L"`` (low-Vt),
            ``"SL"`` (super-low-Vt), or ``"SRAM"``.
        track: ``"75"`` for the 7.5-track library (default) or ``"6"``
            for the 6-track sibling library.

    Returns:
        The canonical cell-name string, e.g. ``"INVx1_ASAP7_75t_R"``.

    Raises:
        ValueError: For unknown flavor/track values.

    Examples:
        >>> resolve_cell_name("INV")
        'INVx1_ASAP7_75t_R'
        >>> resolve_cell_name("NAND2", drive="x2", flavor="L")
        'NAND2x2_ASAP7_75t_L'
        >>> resolve_cell_name("INVx1_ASAP7_75t_R")  # passthrough
        'INVx1_ASAP7_75t_R'
    """
    if flavor not in _ASAP7_FLAVORS:
        raise ValueError(f"flavor must be one of {sorted(_ASAP7_FLAVORS)}, got {flavor!r}")
    if track not in _ASAP7_TRACKS:
        raise ValueError(f"track must be one of {sorted(_ASAP7_TRACKS)}, got {track!r}")
    # Already canonical? Pass through unchanged.
    if "_ASAP7_" in shorthand:
        return shorthand
    # Drive baked into the shorthand (e.g. "INVx1", "NAND2xp5")?
    # The "x" must be lowercase and precede a digit or "p".
    func = shorthand
    drive_suffix = drive
    for i, ch in enumerate(shorthand):
        if ch == "x" and i > 0 and i + 1 < len(shorthand):
            tail = shorthand[i + 1 :]
            if tail and (tail[0].isdigit() or tail[0] == "p"):
                func = shorthand[:i]
                drive_suffix = shorthand[i:]
                break
    return f"{func}{drive_suffix}_ASAP7_{track}t_{flavor}"


def rasterize_cell_layer(
    layout: Any,
    cell: Any,
    layer_spec: tuple[int, int],
    pixel_nm: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterize one (layer, datatype) of a klayout cell into a {0,1} array.

    Polygons are rasterized through ``PIL.ImageDraw.polygon`` after
    transforming their hull/holes to pixel coordinates. This is faithful
    for arbitrary (Manhattan and non-Manhattan) shapes — earlier code
    decomposed into trapezoids and filled their bboxes, which over-filled
    angled trapezoids and only happened to be exact because ASAP7 is
    Manhattan-only. Sibling PDK adapters reuse this helper, so the fix
    has to handle non-axis-aligned geometry too.

    Iterates the layer recursively (``begin_shapes_rec``) so geometry
    referenced via cell instances (a stdcell that ``INSTANCE``s a shared
    via array, for example) is included; the previous flat
    ``cell.shapes(...).each()`` silently dropped instanced shapes.

    Returns ``(array, origin_nm)`` where ``origin_nm`` is the cell bbox
    lower-left corner in nm. The returned array follows the same
    orientation convention as :func:`openlithohub.data.io.load_layout`:
    image (y-down) coordinates with ``arr[0]`` at the top of the layout
    viewer (largest y_nm). Earlier asap7 code stored y-up and then
    ``flipud``-d, contradicting the canonical convention and producing
    vertically mirrored masks vs. ``load_layout``-loaded layouts.
    """
    import klayout.db as kdb
    from PIL import Image, ImageDraw

    layer_index = layout.find_layer(*layer_spec)
    bbox = cell.bbox()
    dbu_nm = layout.dbu * 1000.0
    origin = (bbox.left * dbu_nm, bbox.bottom * dbu_nm)
    w = max(1, int(np.ceil(bbox.width() * dbu_nm / pixel_nm)))
    h = max(1, int(np.ceil(bbox.height() * dbu_nm / pixel_nm)))
    if layer_index is None:
        return np.zeros((h, w), dtype=np.float32), origin

    canvas = Image.new("L", (w, h), 0)
    drawer = ImageDraw.Draw(canvas)

    def _to_px(point: Any) -> tuple[int, int]:
        # GDSII uses mathematical y-up; PIL's image surface uses y-down.
        # Convert at projection time and store the array in PIL's native
        # orientation (no later flipud). ``arr[0]`` corresponds to the top
        # of the layout viewer (largest y_nm) — same convention as
        # ``data.io.load_layout``.
        x_dbu = point.x - bbox.left
        y_dbu = point.y - bbox.bottom
        x_nm = x_dbu * dbu_nm
        y_nm_math = y_dbu * dbu_nm
        px = int(round(x_nm / pixel_nm))
        py_math = int(round(y_nm_math / pixel_nm))
        py = (h - 1) - py_math
        return (max(0, min(px, w - 1)), max(0, min(py, h - 1)))

    # Build a Region from the recursive shape iterator so per-polygon hole
    # semantics survive a multi-shape cell with overlapping geometry, and
    # instanced sub-cell geometry (which the previous flat iterator
    # silently dropped) is included. Same contract as data.io.load_layout.
    region = kdb.Region()
    shapes_iter = cell.begin_shapes_rec(layer_index)
    while not shapes_iter.at_end():
        shape = shapes_iter.shape()
        trans = shapes_iter.trans()
        if shape.is_box():
            region.insert(kdb.Polygon(shape.box).transformed(trans))
        elif shape.is_path():
            region.insert(shape.path.polygon().transformed(trans))
        elif shape.is_polygon():
            region.insert(shape.polygon.transformed(trans))
        shapes_iter.next()
    region.merge()

    for poly in region.each():
        # Decompose into convex (hole-free) pieces so a polygon-with-hole
        # cannot erase a separate polygon nested inside its hole — the
        # global-canvas hazard fixed in data.io.load_layout.
        try:
            convex_pieces = list(poly.decompose_convex(kdb.Polygon.PO_any))
        except (AttributeError, TypeError):
            convex_pieces = [poly]
        for piece in convex_pieces:
            iter_points = (
                piece.each_point if hasattr(piece, "each_point") else (piece.each_point_hull)
            )
            hull = [_to_px(p) for p in iter_points()]
            if len(hull) >= 3:
                drawer.polygon(hull, fill=255)

    arr = np.array(canvas, dtype=np.float32) / 255.0
    return arr, origin


class Asap7Dataset(DatasetAdapter):
    """Adapter for the ASAP7 predictive PDK standard cells.

    Args:
        root: Path to a local clone of ``The-OpenROAD-Project/asap7``
            with the ``asap7sc7p5t_27`` submodule initialised. Use
            ``Asap7Dataset.download(root, accept_license=True)`` to
            create one.
        cells: Cell names to expose, in order. Defaults to
            ``CANONICAL_CELLS``. Function-name shorthand (``"INV"``,
            ``"NAND2"``, ``"DFFHQN"``) is accepted alongside canonical
            ASAP7 strings (``"INVx1_ASAP7_75t_R"``) — see
            ``resolve_shorthand`` and ``resolve_cell_name``.
        design_layer: ``(layer, datatype)`` to rasterize as the design
            tensor. Defaults to M1 (10, 0).
        pixel_nm: Raster pixel size in nm. Defaults to 1.0 to match the
            existing OpenLithoHub grid; ASAP7's manufacturing dbu is
            0.25 nm so this is a 4× downsample.
        gds_path: Optional explicit override for the GDS file path. If
            unset, the adapter globs ``asap7sc7p5t_27/GDS/...`` under
            ``root`` and picks the lexicographically last match.
        resolve_shorthand: When True (default), attempt to expand
            function-name shorthand into the canonical ASAP7 cell-name
            (drive=x1, flavor=R, track=75) before raising KeyError.
            ``LithoSample.metadata['cell_name']`` reflects the resolved
            string; ``metadata['requested_cell_name']`` records the
            original input. Set False to require exact-match names.

    The adapter requires ``klayout`` (already pinned in pyproject.toml).
    """

    def __init__(
        self,
        root: str | Path,
        cells: tuple[str, ...] | list[str] | None = None,
        design_layer: tuple[int, int] = DEFAULT_DESIGN_LAYER,
        pixel_nm: float = 1.0,
        gds_path: str | Path | None = None,
        resolve_shorthand: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"ASAP7 root not found: {self.root}")
        # Surface a warning when no MANIFEST.SHA256 sits next to the data
        # — this adapter's `download()` helper clones the upstream repo
        # but does not pin per-file hashes; bytes loaded later are
        # trusted blindly without this signal.
        from openlithohub._utils.integrity import warn_unverified_data_root

        warn_unverified_data_root(self.root, "asap7")
        self.design_layer = design_layer
        self.pixel_nm = float(pixel_nm)
        self.cells: tuple[str, ...] = tuple(cells) if cells is not None else CANONICAL_CELLS
        self.resolve_shorthand = resolve_shorthand
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
        resolved_name = name
        if cell is None and self.resolve_shorthand and "_ASAP7_" not in name:
            # Caller passed a function-name shorthand ("INV", "NAND2"); try the
            # canonical default flavour/drive before giving up. Records the
            # resolved string in metadata so the caller can see what was picked.
            try:
                candidate = resolve_cell_name(name)
            except ValueError:
                candidate = None
            if candidate is not None:
                cell = layout.cell(candidate)
                if cell is not None:
                    resolved_name = candidate
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
            "cell_name": resolved_name,
            "requested_cell_name": name,
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

    # ---- Croissant metadata ----

    def croissant_name(self) -> str:
        return "ASAP7"

    def croissant_description(self) -> str:
        return (
            "ASAP7 is a 7nm predictive academic PDK released by ASU + ARM (BSD-3-Clause). "
            "Cell layouts are rasterised on-the-fly into design-tensor samples for OPC research."
        )

    def croissant_license_url(self) -> str | None:
        return ASAP7_LICENSE_URL

    def croissant_url(self) -> str | None:
        return "https://github.com/The-OpenROAD-Project/asap7"

    def croissant_citation(self) -> str | None:
        return (
            "Clark, L. T., et al. ASAP7: A 7nm finFET predictive process design kit. "
            "Microelectronics Journal 53 (2016): 105-115."
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
