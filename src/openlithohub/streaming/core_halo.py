"""Tile requests, planning, and the streaming scheduler (RFC 0008).

``TileRequest`` pairs a trusted core with its context halo.  Planning a
request list from a layout shape guarantees an exact core cover; the
scheduler consumes requests (possibly re-queuing refinements) one tile at
a time so full-chip memory stays O(tile area + active batch).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from .geometry import (
    BoundingBox,
    HaloSpec,
    core_grid_boxes,
    halo_actual,
    halo_overhead_stats,
    read_bbox_for,
)


@dataclass(frozen=True)
class TileRequest:
    """Read ``read_bbox`` (core + halo) and commit only ``core_bbox``."""

    core_bbox: BoundingBox
    read_bbox: BoundingBox
    halo: HaloSpec
    tile_id: str

    @property
    def core_size(self) -> tuple[int, int]:
        return (self.core_bbox.height, self.core_bbox.width)

    @property
    def read_size(self) -> tuple[int, int]:
        return (self.read_bbox.height, self.read_bbox.width)


@dataclass
class RefinementRequest:
    """Verifier-agnostic advice to re-run a tile with more context.

    Produced by verification plugins (prompt §12); the scheduler never
    needs to know *why* a tile was inconclusive — only which knob to turn.
    """

    tile_id: str
    action: str = "increase_halo"  # increase_halo | subdivide
    extra_halo_px: int = 0
    subdivision: int = 2

    def __post_init__(self) -> None:
        if self.action not in ("increase_halo", "subdivide"):
            raise ValueError(f"unknown refinement action: {self.action!r}")
        if self.action == "increase_halo" and self.extra_halo_px <= 0:
            raise ValueError("increase_halo refinement needs extra_halo_px > 0")
        if self.action == "subdivide" and self.subdivision < 2:
            raise ValueError("subdivide refinement needs subdivision >= 2")


def plan_tile_requests(
    shape: tuple[int, int],
    core_size: int,
    halo_px: int | HaloSpec,
    *,
    core_stride: int | None = None,
    prefix: str = "tile",
) -> list[TileRequest]:
    """Plan core-exact-cover tile requests for a layout shape."""
    halo = HaloSpec.uniform(halo_px) if isinstance(halo_px, int) else halo_px
    h, w = shape
    requests: list[TileRequest] = []
    for idx, core in enumerate(core_grid_boxes(shape, core_size, core_stride=core_stride)):
        read = read_bbox_for(core, halo, w, h)
        requests.append(
            TileRequest(
                core_bbox=core,
                read_bbox=read,
                halo=halo_actual(core, read),
                tile_id=f"{prefix}_{idx}",
            )
        )
    return requests


def subdivide_request(request: TileRequest, subdivision: int) -> list[TileRequest]:
    """Split a tile's core into ``subdivision^2`` smaller cores."""
    ch, cw = request.core_size
    sub_h = max(1, ch // subdivision)
    sub_w = max(1, cw // subdivision)
    out: list[TileRequest] = []
    core = request.core_bbox
    idx = 0
    for y0 in range(core.y0, core.y1, sub_h):
        for x0 in range(core.x0, core.x1, sub_w):
            sub_core = BoundingBox(
                x0=x0,
                y0=y0,
                x1=min(x0 + sub_w, core.x1),
                y1=min(y0 + sub_h, core.y1),
            )
            if sub_core.width <= 0 or sub_core.height <= 0:
                continue
            extra = HaloSpec.uniform(
                max(request.halo.left, request.halo.top, request.halo.right, request.halo.bottom)
            )
            w = request.read_bbox.x1
            h = request.read_bbox.y1
            read = read_bbox_for(
                sub_core,
                extra,
                width=max(w, sub_core.x1),
                height=max(h, sub_core.y1),
            )
            out.append(
                TileRequest(
                    core_bbox=sub_core,
                    read_bbox=read,
                    halo=halo_actual(sub_core, read),
                    tile_id=f"{request.tile_id}_s{idx}",
                )
            )
            idx += 1
    return out


@dataclass
class TileScheduler:
    """Sequential streaming scheduler with verifier-driven refinement.

    The consumer pulls one request at a time; refinement requests re-enqueue
    work without the scheduler knowing anything about the verifier's maths.
    ``domain`` (global ``(H, W)``) lets grown halos clip to the real layout
    instead of the tile's previous read region; when omitted, the previous
    read extent is used as the clip bound.
    """

    requests: list[TileRequest]
    domain: tuple[int, int] | None = None
    _pending: list[TileRequest] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._pending = list(self.requests)

    def __iter__(self) -> Iterator[TileRequest]:
        return self

    def __next__(self) -> TileRequest:
        if not self._pending:
            raise StopIteration
        return self._pending.pop(0)

    def enqueue(self, requests: list[TileRequest]) -> None:
        self._pending.extend(requests)

    def enqueue_refinement(
        self, request: TileRequest, refinement: RefinementRequest
    ) -> list[TileRequest]:
        """Re-queue refined work for an inconclusive tile."""
        if refinement.action == "subdivide":
            refined = subdivide_request(request, refinement.subdivision)
        else:
            grown = HaloSpec.uniform(
                max(
                    request.halo.left,
                    request.halo.top,
                    request.halo.right,
                    request.halo.bottom,
                )
                + refinement.extra_halo_px
            )
            if self.domain is not None:
                clip_w, clip_h = self.domain
            else:
                clip_w = max(request.read_bbox.x1, request.core_bbox.x1)
                clip_h = max(request.read_bbox.y1, request.core_bbox.y1)
            read = read_bbox_for(request.core_bbox, grown, clip_w, clip_h)
            refined = [
                TileRequest(
                    core_bbox=request.core_bbox,
                    read_bbox=read,
                    halo=halo_actual(request.core_bbox, read),
                    tile_id=request.tile_id,
                )
            ]
        self.enqueue(refined)
        return refined


def tiling_overhead(requests: list[TileRequest]) -> dict[str, float]:
    """Aggregate core/halo duplicate-compute statistics for a plan."""
    cores = [r.core_bbox for r in requests]
    reads = [r.read_bbox for r in requests]
    stats = halo_overhead_stats(cores, reads)
    stats["estimated_duplicate_compute_pct"] = stats["halo_overhead_pct"]
    return stats


def plan_tiling(
    layout_shape: tuple[int, int],
    *,
    memory_budget_bytes: int,
    bytes_per_pixel: float = 4.0,
    halo_px: int = 0,
    min_core_size: int = 32,
    max_core_size: int = 4096,
    batch: int = 1,
) -> list[TileRequest]:
    """Pick the largest core size that fits the per-tile memory budget.

    Approximates peak per-tile memory as ``bytes_per_pixel * (C + 2h)^2 *
    batch`` and maximises core efficiency ``C^2 / (C + 2h)^2`` subject to
    the budget (prompt §16).  Callers may pass an explicit
    ``HaloPolicy``-derived halo; the planner itself stays policy-agnostic.
    """
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    if min_core_size <= 0 or max_core_size < min_core_size:
        raise ValueError(f"invalid core size bounds: {min_core_size}..{max_core_size}")

    best_core = min_core_size
    for core in range(min_core_size, max_core_size + 1, 8):
        read = core + 2 * halo_px
        if bytes_per_pixel * read * read * batch <= memory_budget_bytes:
            best_core = core
    h, w = layout_shape
    if best_core >= min(h, w):
        # Small layout: a single core covering the whole domain needs no halo.
        best_core = max(min(h, w), min_core_size)
        return plan_tile_requests(layout_shape, best_core, HaloSpec.uniform(0))
    return plan_tile_requests(layout_shape, best_core, HaloSpec.uniform(halo_px))
