"""ICCAD 2016 Problem C — EUV hotspot detection benchmark adapter.

The benchmark is from the ICCAD 2016 CAD Contest (Problem C, EUV
Simulation). The publicly mirrored copy lives at
https://github.com/phdyang007/ICCAD16-N7M2EUV — four EUV designs at
N7 / 16 nm CD plus simulated hotspot locations recorded under a
process-window sweep.

The dataset is a **hotspot detection benchmark**, not a mask
optimization benchmark — there is no OPC reference mask to compare
against. Two pieces of evidence:

1. The repo's references are both hotspot-detection papers
   (Chen et al., DAC'19; Yang et al., TCAD'20 — "Bridging the Gap
   Between Layout Pattern Sampling and Hotspot Detection via Batch
   Active Learning").
2. The auxiliary layer ``(10000, 0)`` ships 120 small 16×16 nm boxes
   distributed on a regular grid, covering only ~1% of design pixels
   and located 70+ nm away from any CSV hotspot — consistent with
   detection clip / inspection-grid sites, not with an OPC mask.

Files per test case:

- ``testcaseN.oas`` — OASIS layout. The N7M2EUV stack is documented in
  [Yang2020_BatchAL, §III-A, p.4]; the per-layer mapping below applies
  to every test case in this distribution:

  | GDS layer | Datatype | Meaning                                                     |
  |-----------|----------|-------------------------------------------------------------|
  | 1000      | 0        | Design polygons (drawn metal-2 features at N7, 16 nm CD).   |
  | 10000     | 0        | Auxiliary clip-site grid (16×16 nm boxes, hotspot inspection sites). |

  ``(layer=1000, datatype=0)`` is exposed as the loaded ``design``
  tensor. ``(layer=10000, datatype=0)`` is exposed under
  ``metadata['clip_sites']``.
- ``testN.csv`` — hotspot annotations with columns ``def, id,
  category, x, y``. Coordinates are in OASIS database units (1 dbu =
  1 nm for these files); ``category`` is the contest's defect type
  code (raw integers, per-testcase). The README promises three
  semantic kinds (EPE / Bridging / Necking) but does not publish the
  integer mapping, so the code preserves the raw id. The same physical
  site can appear multiple times under different dose/focus
  conditions.

The adapter returns ``LithoSample(design, mask=None, resist=None,
metadata)``. ``LithoSample.mask`` is intentionally left ``None`` —
this dataset does not provide a reference mask. Hotspot annotations
and clip-site centers live in ``metadata``.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub.data.base import DatasetAdapter, LithoSample


@dataclass(frozen=True)
class HotspotAnnotation:
    """One row from the testN.csv hotspot table.

    ``x_nm`` / ``y_nm`` are the contest dbu coordinates converted to nm
    using the OASIS layout's dbu (1 dbu = 1 nm for the published files,
    but the conversion still goes through ``layout.dbu * 1000``).
    ``category_id`` preserves the raw contest code; the README only
    promises three semantic kinds (EPE / Bridging / Necking) but does
    not publish the integer mapping, so callers should treat the id as
    an opaque label until they have the contest's category dictionary.
    """

    hotspot_id: int
    category_id: int
    x_nm: float
    y_nm: float


class Iccad16Dataset(DatasetAdapter):
    """Adapter for the ICCAD 2016 Problem C — EUV hotspot benchmark.

    Args:
        root: Directory containing ``testcase{N}.oas`` and
            ``test{N}.csv`` files. The four published cases use ``N``
            in 1..4.
        cases: Optional explicit list of case indices to expose, e.g.
            ``[1, 3]``. Defaults to whichever cases are present on disk.
        design_layer: ``(layer, datatype)`` tuple selecting the design
            polygons. Defaults to ``(1000, 0)`` per the published files.
        clip_layer: ``(layer, datatype)`` tuple selecting the auxiliary
            clip-site layer. Defaults to ``(10000, 0)``. Exposed via
            ``metadata['clip_sites']``; the layer is empirically used
            for hotspot-detection clip locations, not as an OPC mask.
        pixel_nm: Raster pixel size in nm. The published layouts are
            ~1.9 µm × 1.5 µm so 1 nm/px stays well under 2k×2k.

    The adapter reads each OASIS file lazily on first access and caches
    the rasterized design tensor in memory. ``klayout`` is required and
    is already pinned in ``pyproject.toml``.
    """

    def __init__(
        self,
        root: str | Path,
        cases: list[int] | None = None,
        design_layer: tuple[int, int] = (1000, 0),
        clip_layer: tuple[int, int] = (10000, 0),
        pixel_nm: float = 1.0,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"ICCAD16 root not found: {self.root}")
        from openlithohub._utils.integrity import warn_unverified_data_root

        warn_unverified_data_root(self.root, "iccad16")
        self.design_layer = design_layer
        self.clip_layer = clip_layer
        self.pixel_nm = float(pixel_nm)

        if cases is None:
            cases = sorted(
                int(p.stem.removeprefix("testcase"))
                for p in self.root.glob("testcase*.oas")
                if p.stem.removeprefix("testcase").isdigit()
            )
        if not cases:
            raise FileNotFoundError(f"No testcase*.oas files under {self.root}")
        self._cases = cases
        self._cache: dict[int, LithoSample] = {}

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, index: int) -> LithoSample:
        if index < 0 or index >= len(self._cases):
            raise IndexError(f"Index {index} out of range [0, {len(self._cases)})")
        case_id = self._cases[index]
        if case_id in self._cache:
            return self._cache[case_id]
        sample = self._load_case(case_id)
        self._cache[case_id] = sample
        return sample

    def _load_case(self, case_id: int) -> LithoSample:
        oas_path = self.root / f"testcase{case_id}.oas"
        csv_path = self.root / f"test{case_id}.csv"
        if not oas_path.exists():
            raise FileNotFoundError(f"Missing OASIS file: {oas_path}")

        # OASIS rasterization via klayout. Imported lazily so that
        # importing the package does not require klayout for users who
        # only touch other datasets.
        import klayout.db as kdb

        layout = kdb.Layout()
        layout.read(str(oas_path))
        dbu_nm = layout.dbu * 1000.0  # klayout dbu is in µm

        top = layout.top_cell()
        if top is None:
            raise RuntimeError(f"OASIS file has no top cell: {oas_path}")

        design_arr, origin = self._rasterize_layer(layout, top, self.design_layer)
        clip_sites = self._collect_clip_sites(layout, top, self.clip_layer)

        hotspots = self._load_hotspots(csv_path) if csv_path.exists() else []

        metadata: dict[str, Any] = {
            "dataset": "iccad16",
            "case_id": case_id,
            "source_oas": str(oas_path),
            "source_csv": str(csv_path) if csv_path.exists() else None,
            "dbu_nm": dbu_nm,
            "pixel_nm": self.pixel_nm,
            "design_layer": list(self.design_layer),
            "clip_layer": list(self.clip_layer),
            "origin_nm": [origin[0], origin[1]],  # bbox lower-left in nm
            "hotspots": [h.__dict__ for h in hotspots],
            "num_hotspots": len(hotspots),
            "clip_sites": clip_sites,
            "num_clip_sites": len(clip_sites),
        }

        return LithoSample(
            design=torch.from_numpy(design_arr).float(),
            mask=None,
            resist=None,
            metadata=metadata,
        )

    def _rasterize_layer(
        self,
        layout: Any,
        top: Any,
        layer_spec: tuple[int, int],
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Rasterize a single OASIS layer into a {0,1} numpy array.

        Decomposes each polygon into trapezoids via klayout's
        ``Polygon.decompose_trapezoids`` and fills each trapezoid's pixel
        footprint. For Manhattan polygons every trapezoid is an
        axis-aligned rectangle, so the fill is exact even for L-shapes
        and other concave Manhattan geometry — a plain bbox fill would
        over-fill the concave corner.
        """
        import klayout.db as kdb

        layer_index = layout.find_layer(*layer_spec)
        bbox = top.bbox()
        origin = (
            bbox.left * layout.dbu * 1000.0,
            bbox.bottom * layout.dbu * 1000.0,
        )
        w = max(1, int(np.ceil(bbox.width() * layout.dbu * 1000.0 / self.pixel_nm)))
        h = max(1, int(np.ceil(bbox.height() * layout.dbu * 1000.0 / self.pixel_nm)))
        if layer_index is None:
            return np.zeros((h, w), dtype=np.float32), origin

        arr = np.zeros((h, w), dtype=np.float32)
        ox_nm, oy_nm = origin
        dbu_um = layout.dbu

        def _fill_box(b: Any) -> None:
            x0_nm = b.left * dbu_um * 1000.0 - ox_nm
            y0_nm = b.bottom * dbu_um * 1000.0 - oy_nm
            x1_nm = b.right * dbu_um * 1000.0 - ox_nm
            y1_nm = b.top * dbu_um * 1000.0 - oy_nm
            i0 = max(0, int(np.floor(x0_nm / self.pixel_nm)))
            j0 = max(0, int(np.floor(y0_nm / self.pixel_nm)))
            i1 = min(w, int(np.ceil(x1_nm / self.pixel_nm)))
            j1 = min(h, int(np.ceil(y1_nm / self.pixel_nm)))
            if i1 > i0 and j1 > j0:
                arr[j0:j1, i0:i1] = 1.0

        # Recursive iteration so hierarchical layouts (geometry referenced
        # through cell instances) are not silently dropped. Today's published
        # ICCAD16 files are flat — single TOPCELL — but matching the canonical
        # pattern from data/io.py:128 keeps this robust against future
        # contributions or upstream regenerations that introduce hierarchy.
        shapes_iter = top.begin_shapes_rec(layer_index)
        while not shapes_iter.at_end():
            shape_obj = shapes_iter.shape()
            trans = shapes_iter.trans()
            if shape_obj.is_box():
                _fill_box(shape_obj.box.transformed(trans))
                shapes_iter.next()
                continue
            if shape_obj.is_path():
                poly = shape_obj.path.polygon()
            elif shape_obj.is_polygon():
                poly = shape_obj.polygon
            else:
                shapes_iter.next()
                continue
            poly = poly.transformed(trans)
            try:
                trapezoids = list(poly.decompose_trapezoids(kdb.Polygon.TD_simple))
            except AttributeError:
                # Older klayout: fall back to whole-polygon bbox (still over-fills
                # concavities, but no worse than the historical behavior).
                _fill_box(poly.bbox())
                shapes_iter.next()
                continue
            for tz in trapezoids:
                _fill_box(tz.bbox())
            shapes_iter.next()

        return arr, origin

    def _collect_clip_sites(
        self,
        layout: Any,
        top: Any,
        layer_spec: tuple[int, int],
    ) -> list[dict[str, float]]:
        """Return clip-site bboxes in nm (no rasterization)."""
        layer_index = layout.find_layer(*layer_spec)
        if layer_index is None:
            return []
        dbu_nm = layout.dbu * 1000.0
        out: list[dict[str, float]] = []
        # Recursive iteration matches _rasterize_layer above.
        shapes_iter = top.begin_shapes_rec(layer_index)
        while not shapes_iter.at_end():
            s = shapes_iter.shape()
            trans = shapes_iter.trans()
            if s.is_box():
                b = s.box.transformed(trans)
            elif s.is_polygon():
                b = s.polygon.transformed(trans).bbox()
            else:
                shapes_iter.next()
                continue
            out.append(
                {
                    "x0_nm": b.left * dbu_nm,
                    "y0_nm": b.bottom * dbu_nm,
                    "x1_nm": b.right * dbu_nm,
                    "y1_nm": b.top * dbu_nm,
                }
            )
            shapes_iter.next()
        return out

    def _load_hotspots(self, csv_path: Path) -> list[HotspotAnnotation]:
        out: list[HotspotAnnotation] = []
        n_rows = 0
        n_skipped = 0
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_rows += 1
                try:
                    out.append(
                        HotspotAnnotation(
                            hotspot_id=int(row["id"]),
                            category_id=int(row["category"]),
                            x_nm=float(row["x"]),
                            y_nm=float(row["y"]),
                        )
                    )
                except (KeyError, ValueError) as exc:
                    n_skipped += 1
                    warnings.warn(
                        f"Skipped malformed row in {csv_path.name}: {exc!r}",
                        stacklevel=2,
                    )
        if n_rows > 0 and not out:
            raise ValueError(
                f"All {n_rows} rows in {csv_path} were malformed — the CSV "
                f"header may have changed (expected columns: id, category, x, y)."
            )
        if n_skipped:
            warnings.warn(
                f"{n_skipped}/{n_rows} rows in {csv_path.name} were skipped as malformed.",
                stacklevel=2,
            )
        return out

    def download(self, root: str) -> None:
        raise NotImplementedError(
            "ICCAD16 auto-download is not implemented. Clone manually from "
            "https://github.com/phdyang007/ICCAD16-N7M2EUV and place the "
            "testcase*.oas + test*.csv files under the dataset root."
        )

    @property
    def case_ids(self) -> list[int]:
        return list(self._cases)
