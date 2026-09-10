"""Exact parser-independent vector geometry -> owned horizontal runs.

The theorem-facing geometry semantics are deliberately independent of
KLayout/PIL rasterization:

* vertices live on integer pixel *edges*;
* pixel occupancy is sampled on the pixel-center scanline y+1/2;
* polygon holes subtract only from their parent polygon;
* layout objects are unioned as a binary mask before spectral accumulation;
* contributor identities are preserved until that union is formed;
* tile windows clip a canonical global parent run, so overlapping reads do
  not manufacture new physical owners.

A KLayout adapter is provided at the end of the file.  It is a parser layer:
all proof-sensitive union/ownership logic remains in this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .geometry import BoundingBox
from .physical_identity import InstancePathElement, PhysicalInstanceKey

Point = tuple[int, int]


@dataclass(frozen=True)
class PolygonWithHoles:
    object_id: str
    outer: tuple[Point, ...]
    holes: tuple[tuple[Point, ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.outer) < 3:
            raise ValueError("outer polygon needs at least 3 vertices")
        if any(len(hole) < 3 for hole in self.holes):
            raise ValueError("polygon hole needs at least 3 vertices")

    def translated(self, dx: int, dy: int, *, object_id: str) -> PolygonWithHoles:
        def tr(loop: tuple[Point, ...]) -> tuple[Point, ...]:
            return tuple((x + dx, y + dy) for x, y in loop)

        return PolygonWithHoles(
            object_id=object_id,
            outer=tr(self.outer),
            holes=tuple(tr(h) for h in self.holes),
        )

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.outer]
        ys = [p[1] for p in self.outer]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class CellInstance:
    instance_id: str
    cell_name: str
    dx: int
    dy: int


@dataclass(frozen=True)
class VectorCell:
    name: str
    polygons: tuple[PolygonWithHoles, ...] = ()
    instances: tuple[CellInstance, ...] = ()


@dataclass(frozen=True)
class OwnedRun:
    """Canonical global occupancy run before tile clipping."""

    run_id: str
    y: int
    x0: int
    x1: int
    contributor_object_ids: tuple[str, ...]

    @property
    def length(self) -> int:
        return self.x1 - self.x0


@dataclass(frozen=True)
class OwnedRunSlice:
    """A tile view of a canonical physical occupancy run."""

    parent: OwnedRun
    x0: int
    x1: int
    tile_id: str
    role: str = "INTERFACE"
    analytic_period_shift: tuple[int, int] = (0, 0)
    error_budget_charge: float = 0.0

    def __post_init__(self) -> None:
        if not (self.parent.x0 <= self.x0 < self.x1 <= self.parent.x1):
            raise ValueError("slice escapes canonical parent run")
        if self.error_budget_charge < 0:
            raise ValueError("negative error budget charge")

    @property
    def run_id(self) -> str:
        return self.parent.run_id

    @property
    def y(self) -> int:
        return self.parent.y

    @property
    def analytic_duplicate(self) -> bool:
        return self.analytic_period_shift != (0, 0)


def _ceil_fraction(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _loop_scanline_intersections(loop: tuple[Point, ...], y: int) -> list[Fraction]:
    """Exact x-intersections with the pixel-center scanline y+1/2.

    The half-open vertical edge rule prevents double counting polygon
    vertices.  Vertices are integral, so y+1/2 never hits a vertex.
    """
    sy = Fraction(2 * y + 1, 2)
    xs: list[Fraction] = []
    for (x1, y1), (x2, y2) in zip(loop, loop[1:] + loop[:1], strict=True):
        if y1 == y2:
            continue
        low, high = sorted((y1, y2))
        if not (Fraction(low, 1) <= sy < Fraction(high, 1)):
            continue
        x = Fraction(x1, 1) + (sy - y1) * Fraction(x2 - x1, y2 - y1)
        xs.append(x)
    xs.sort()
    if len(xs) % 2:
        raise ValueError("non-simple polygon or inconsistent scanline intersections")
    return xs


def _loop_row_spans(loop: tuple[Point, ...], y: int) -> list[tuple[int, int]]:
    xs = _loop_scanline_intersections(loop, y)
    spans: list[tuple[int, int]] = []
    half = Fraction(1, 2)
    for left, right in zip(xs[0::2], xs[1::2], strict=True):
        if right <= left:
            continue
        # Pixel center x+1/2 lies in [left,right).
        x0 = _ceil_fraction(left - half)
        x1 = _ceil_fraction(right - half)
        if x1 > x0:
            spans.append((x0, x1))
    return spans


def _subtract_spans(
    base: list[tuple[int, int]],
    cuts: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = base[:]
    for ca, cb in cuts:
        nxt: list[tuple[int, int]] = []
        for a, b in out:
            if cb <= a or ca >= b:
                nxt.append((a, b))
                continue
            if a < ca:
                nxt.append((a, min(b, ca)))
            if cb < b:
                nxt.append((max(a, cb), b))
        out = [(a, b) for a, b in nxt if b > a]
    return out


def polygon_row_spans(poly: PolygonWithHoles, y: int) -> list[tuple[int, int]]:
    spans = _loop_row_spans(poly.outer, y)
    for hole in poly.holes:
        spans = _subtract_spans(spans, _loop_row_spans(hole, y))
    return spans


def _union_with_contributors(
    spans: Iterable[tuple[int, int, str]],
) -> list[tuple[int, int, tuple[str, ...]]]:
    """Binary union while preserving the exact active contributor set."""
    events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for a, b, owner in spans:
        if b <= a:
            continue
        events[a].append((owner, +1))
        events[b].append((owner, -1))
    if not events:
        return []

    counts: dict[str, int] = {}
    positions = sorted(events)
    out: list[tuple[int, int, tuple[str, ...]]] = []
    for i, pos in enumerate(positions):
        # Events at pos affect [pos,next).
        for owner, delta in events[pos]:
            counts[owner] = counts.get(owner, 0) + delta
            if counts[owner] == 0:
                del counts[owner]
        if i + 1 == len(positions):
            break
        nxt = positions[i + 1]
        if nxt <= pos or not counts:
            continue
        owners = tuple(sorted(counts))
        if out and out[-1][1] == pos and out[-1][2] == owners:
            out[-1] = (out[-1][0], nxt, owners)
        else:
            out.append((pos, nxt, owners))
    return out


def _hash_layout(shape: tuple[int, int], cells: dict[str, VectorCell], top: str) -> str:
    def poly_obj(p: PolygonWithHoles) -> dict[str, Any]:
        return {"id": p.object_id, "outer": p.outer, "holes": p.holes}

    payload = {
        "shape": shape,
        "top": top,
        "cells": {
            name: {
                "polygons": [poly_obj(p) for p in cell.polygons],
                "instances": [
                    {"id": i.instance_id, "cell": i.cell_name, "dx": i.dx, "dy": i.dy}
                    for i in cell.instances
                ],
            }
            for name, cell in sorted(cells.items())
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class ExactVectorRunSource:
    """Reference vector TileSource + verification LayoutRunSource.

    This source stores vector geometry, never a full-chip raster.  It may
    generate a *requested-window* tensor for compatibility with RFC 0008,
    while theorem-facing verification consumes the canonical owned runs.
    """

    backend_kind = "exact-vector-scanline"

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        cells: dict[str, VectorCell],
        top: str,
        pixel_size_nm: float = 1.0,
    ) -> None:
        if top not in cells:
            raise ValueError(f"unknown top cell {top!r}")
        if min(shape) <= 0:
            raise ValueError("invalid layout shape")
        self._shape = shape
        self._cells = dict(cells)
        self._top = top
        self._pixel_nm = float(pixel_size_nm)
        self._layout_hash = _hash_layout(shape, cells, top)
        self._flat = tuple(self._flatten())
        self._row_index: dict[int, tuple[PolygonWithHoles, ...]] = {}
        self._last_bbox: BoundingBox | None = None
        self._last_owned: tuple[OwnedRunSlice, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def pixel_size_nm(self) -> float:
        return self._pixel_nm

    @property
    def layout_hash(self) -> str:
        return self._layout_hash

    def _flatten(self) -> Iterator[PolygonWithHoles]:
        stack: list[tuple[str, int, int, tuple[str, ...], tuple[str, ...]]] = [
            (self._top, 0, 0, (self._top,), ())
        ]
        while stack:
            cell_name, dx, dy, path, ancestors = stack.pop()
            if cell_name in ancestors:
                raise ValueError("cyclic cell hierarchy")
            cell = self._cells[cell_name]
            for poly in cell.polygons:
                oid = "/".join(path + (poly.object_id,))
                yield poly.translated(dx, dy, object_id=oid)
            for inst in reversed(cell.instances):
                if inst.cell_name not in self._cells:
                    raise ValueError(f"unknown child cell {inst.cell_name!r}")
                stack.append(
                    (
                        inst.cell_name,
                        dx + inst.dx,
                        dy + inst.dy,
                        path + (inst.instance_id,),
                        ancestors + (cell_name,),
                    )
                )

    def _polygons_for_row(self, y: int) -> tuple[PolygonWithHoles, ...]:
        cached = self._row_index.get(y)
        if cached is not None:
            return cached
        selected = []
        for poly in self._flat:
            _x0, y0, _x1, y1 = poly.bbox
            if y0 <= y < y1:
                selected.append(poly)
        out = tuple(selected)
        self._row_index[y] = out
        return out

    def _global_owned_row(self, y: int) -> tuple[OwnedRun, ...]:
        h, w = self._shape
        if not 0 <= y < h:
            return ()
        spans: list[tuple[int, int, str]] = []
        for poly in self._polygons_for_row(y):
            for a, b in polygon_row_spans(poly, y):
                a, b = max(0, a), min(w, b)
                if b > a:
                    spans.append((a, b, poly.object_id))
        merged = _union_with_contributors(spans)
        runs = []
        for x0, x1, owners in merged:
            raw = (f"{self._layout_hash}|{y}|{x0}|{x1}|{'|'.join(owners)}").encode()
            rid = hashlib.sha256(raw).hexdigest()[:24]
            runs.append(
                OwnedRun(
                    run_id=rid,
                    y=y,
                    x0=x0,
                    x1=x1,
                    contributor_object_ids=owners,
                )
            )
        return tuple(runs)

    def iter_owned_runs_for_bbox(
        self,
        bbox: BoundingBox,
        *,
        tile_id: str = "tile",
        role: str = "INTERFACE",
    ) -> Iterator[OwnedRunSlice]:
        h, w = self._shape
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > w or bbox.y1 > h:
            raise ValueError("bbox escapes finite layout")
        for y in range(bbox.y0, bbox.y1):
            for parent in self._global_owned_row(y):
                x0 = max(parent.x0, bbox.x0)
                x1 = min(parent.x1, bbox.x1)
                if x1 > x0:
                    yield OwnedRunSlice(
                        parent=parent,
                        x0=x0,
                        x1=x1,
                        tile_id=tile_id,
                        role=role,
                    )

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ) -> Iterable[Any]:
        """Satisfy verify.streaming.LayoutRunSource structurally."""
        from openlithohub.verify.interface_runs import HorizontalRun

        x0, y0, x1, y1 = bbox_px
        bbox = BoundingBox(x0, y0, x1, y1)
        return tuple(HorizontalRun(r.y, r.x0, r.x1) for r in self.iter_owned_runs_for_bbox(bbox))

    def read_window(self, bbox: BoundingBox) -> torch.Tensor:
        """RFC-0008 TileSource compatibility using only a window tensor."""
        arr = np.zeros((bbox.height, bbox.width), dtype=np.float32)
        owned = tuple(self.iter_owned_runs_for_bbox(bbox))
        for run in owned:
            arr[
                run.y - bbox.y0,
                run.x0 - bbox.x0 : run.x1 - bbox.x0,
            ] = 1.0
        self._last_bbox = bbox
        self._last_owned = owned
        return torch.from_numpy(arr)

    def verification_metadata(
        self,
        bbox: BoundingBox,
        *,
        tile_id: str,
        core_bbox: BoundingBox,
    ) -> dict[str, Any]:
        if self._last_bbox == bbox:
            owned = self._last_owned
        else:
            owned = tuple(self.iter_owned_runs_for_bbox(bbox, tile_id=tile_id))
        # Rebind the tile id for cached views; parent ids remain unchanged.
        rebound = tuple(
            OwnedRunSlice(
                parent=r.parent,
                x0=r.x0,
                x1=r.x1,
                tile_id=tile_id,
                role="CORE"
                if core_bbox.y0 <= r.y < core_bbox.y1
                and r.x0 >= core_bbox.x0
                and r.x1 <= core_bbox.x1
                else "INTERFACE",
                analytic_period_shift=r.analytic_period_shift,
                error_budget_charge=r.error_budget_charge,
            )
            for r in owned
        )
        return {
            "b04_layout_hash": self._layout_hash,
            "b04_owned_runs": rebound,
            "b04_backend_kind": self.backend_kind,
        }


def rectangle(
    object_id: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    holes: tuple[tuple[int, int, int, int], ...] = (),
) -> PolygonWithHoles:
    outer = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    hole_loops = tuple(((a, b), (c, b), (c, d), (a, d)) for a, b, c, d in holes)
    return PolygonWithHoles(object_id=object_id, outer=outer, holes=hole_loops)


class KLayoutAlignedRunSource(ExactVectorRunSource):
    """GDS/OASIS parser adapter for exactly pixel-aligned vector geometry.

    KLayout remains a parser only.  All run construction, binary union,
    ownership and tile clipping use :class:`ExactVectorRunSource`.

    The adapter intentionally refuses non-integral pixel alignment rather
    than silently rounding theorem-facing geometry.
    """

    backend_kind = "klayout-aligned-vector-scanline"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float,
        layer: str | None = None,
        top_cell: str | None = None,
    ) -> KLayoutAlignedRunSource:
        try:
            import klayout.db as db
        except ImportError:
            raise ImportError("KLayout Python API is required for GDS/OASIS parsing") from None

        from openlithohub.data.io import _select_layer

        path = Path(path)
        layout = db.Layout()
        layout.read(str(path))
        tops = list(layout.top_cells())
        if top_cell is None:
            if len(tops) != 1:
                raise ValueError(
                    "theorem-facing adapter requires exactly one top cell "
                    "or an explicit top_cell name"
                )
            top = tops[0]
        else:
            top = layout.cell(top_cell)
            if top is None:
                raise ValueError(f"unknown top cell {top_cell!r}")
        bbox = top.bbox()
        layer_index = _select_layer(layout, layer)

        dbu_nm = Fraction(str(layout.dbu)) * 1000
        pixel_nm = Fraction(str(pixel_size_nm))
        dbu_per_pixel = pixel_nm / dbu_nm
        if dbu_per_pixel.denominator != 1:
            raise ValueError(
                "theorem-facing KLayout adapter requires an integer DBU-per-pixel ratio"
            )
        step = int(dbu_per_pixel)
        if step <= 0:
            raise ValueError("invalid DBU-per-pixel scale")
        if bbox.width() % step or bbox.height() % step:
            raise ValueError("top-cell bbox is not exactly pixel aligned")

        width = bbox.width() // step
        height = bbox.height() // step

        def _call_or_attr(obj: Any, name: str, default: Any = None) -> Any:
            value = getattr(obj, name, default)
            return value() if callable(value) else value

        def _transform_token(trans: Any) -> str:
            if trans is None:
                return "none"
            to_s = getattr(trans, "to_s", None)
            if callable(to_s):
                return str(to_s())
            return str(trans)

        def _instance_path(it_obj: Any) -> tuple[InstancePathElement, ...]:
            path_value = _call_or_attr(it_obj, "path", ())
            out = []
            for elem in tuple(path_value):
                cell_inst = _call_or_attr(elem, "cell_inst", None)
                cell_index = int(_call_or_attr(cell_inst, "cell_index", -1))
                ia = int(_call_or_attr(elem, "ia", 0))
                ib = int(_call_or_attr(elem, "ib", 0))
                specific = _call_or_attr(elem, "specific_cplx_trans", None)
                out.append(
                    InstancePathElement(
                        source_cell_index=cell_index,
                        array_i=ia,
                        array_j=ib,
                        specific_transform=_transform_token(specific),
                    )
                )
            return tuple(out)

        def _shape_fingerprint(shape_obj: Any) -> str:
            prop_id = int(_call_or_attr(shape_obj, "prop_id", 0) or 0)
            payload: tuple[Any, ...]
            if shape_obj.is_path():
                path_obj = _call_or_attr(shape_obj, "path")
                pts = tuple((int(p.x), int(p.y)) for p in _call_or_attr(path_obj, "each_point", ()))
                payload = (
                    "PATH",
                    pts,
                    int(_call_or_attr(path_obj, "width", 0)),
                    int(_call_or_attr(path_obj, "bgn_ext", 0)),
                    int(_call_or_attr(path_obj, "end_ext", 0)),
                    bool(_call_or_attr(path_obj, "is_round", False)),
                    prop_id,
                )
            elif shape_obj.is_box():
                b = _call_or_attr(shape_obj, "box")
                payload = (
                    "BOX",
                    int(b.left),
                    int(b.bottom),
                    int(b.right),
                    int(b.top),
                    prop_id,
                )
            else:
                p = _call_or_attr(shape_obj, "polygon")
                outer0 = tuple(
                    (int(q.x), int(q.y)) for q in _call_or_attr(p, "each_point_hull", ())
                )
                holes0 = []
                holes_n = int(_call_or_attr(p, "holes", 0) or 0)
                for hi0 in range(holes_n):
                    holes0.append(tuple((int(q.x), int(q.y)) for q in p.each_point_hole(hi0)))
                payload = ("POLYGON", outer0, tuple(holes0), prop_id)
            return hashlib.sha256(repr(payload).encode()).hexdigest()[:24]

        source_format = path.suffix.lower().lstrip(".") or "unknown"
        quant_policy = f"exact-integer-dbu-per-pixel:{step}"
        polys: list[PolygonWithHoles] = []
        occurrence: dict[tuple[int, tuple[InstancePathElement, ...], str], int] = {}
        it = top.begin_shapes_rec(layer_index)
        while not it.at_end():
            shape = it.shape()
            trans_attr = getattr(it, "itrans", None)
            if trans_attr is None:
                trans_attr = it.trans
            trans = trans_attr() if callable(trans_attr) else trans_attr

            poly = None
            if shape.is_polygon() or shape.is_box():
                poly = shape.polygon if shape.is_polygon() else db.Polygon(shape.box)
            elif shape.is_path():
                # KLayout's own path->polygon semantics carry width,
                # begin/end extensions and round-end state.  The resulting
                # polygon must still pass exact pixel-edge quantization below.
                path_obj = shape.path
                poly = path_obj.polygon()

            if poly is not None:
                source_cell_index = int(_call_or_attr(it, "cell_index", -1))
                ipath = _instance_path(it)
                fingerprint = _shape_fingerprint(shape)
                base = (source_cell_index, ipath, fingerprint)
                ordinal = occurrence.get(base, 0)
                occurrence[base] = ordinal + 1

                physical_key = PhysicalInstanceKey(
                    source_format=source_format,
                    source_cell_index=source_cell_index,
                    source_shape_fingerprint=f"{fingerprint}:{ordinal}",
                    instance_path=ipath,
                    quantization_policy=quant_policy,
                )

                poly = poly.transformed(trans)

                def project(pt: Any) -> Point:
                    dx = int(pt.x) - int(bbox.left)
                    dy = int(pt.y) - int(bbox.bottom)
                    if dx % step or dy % step:
                        raise ValueError(
                            "GDS/OASIS transformed vertex is not exactly aligned "
                            "to verifier pixel edges"
                        )
                    x = dx // step
                    y = height - dy // step
                    return x, y

                outer = tuple(project(p) for p in poly.each_point_hull())
                holes_out = []
                holes_fn = getattr(poly, "holes", None)
                n_holes = int(holes_fn()) if callable(holes_fn) else 0
                for hi in range(n_holes):
                    holes_out.append(tuple(project(p) for p in poly.each_point_hole(hi)))

                polys.append(
                    PolygonWithHoles(
                        object_id=f"physical:{physical_key.owner_id}",
                        outer=outer,
                        holes=tuple(holes_out),
                    )
                )
            it.next()

        cells = {"TOP": VectorCell(name="TOP", polygons=tuple(polys), instances=())}
        return cls(
            shape=(height, width),
            cells=cells,
            top="TOP",
            pixel_size_nm=float(pixel_size_nm),
        )
