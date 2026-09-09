"""Streaming full-chip geometry primitives (RFC 0008).

Core/halo ownership model:

- The **core** is the only region whose result may be committed to the
  final full-chip output.
- The **halo** is read-region context that gives the forward model real
  neighbourhood content instead of zero-padded artefacts.
- Adjacent cores must exactly cover the global domain (no gaps, no
  duplicate ownership); adjacent read regions may overlap freely.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Half-open pixel rectangle ``[x0, x1) x [y0, y1)`` in global coords."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"degenerate bbox: x=[{self.x0},{self.x1}) y=[{self.y0},{self.y1})")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped_to(self, width: int, height: int) -> BoundingBox:
        """Clip to the global domain ``[0,width) x [0,height)``.

        Returns a bbox that may shrink; callers must treat an empty
        intersection (degenerate result) as "no overlap".
        """
        return BoundingBox(
            x0=max(0, self.x0),
            y0=max(0, self.y0),
            x1=min(width, self.x1),
            y1=min(height, self.y1),
        )

    def contains(self, other: BoundingBox) -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )


@dataclass(frozen=True)
class HaloSpec:
    """Per-side halo widths requested for a tile read region."""

    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def uniform(cls, px: int) -> HaloSpec:
        if px < 0:
            raise ValueError(f"halo must be nonnegative, got {px}")
        return cls(left=px, top=px, right=px, bottom=px)

    @property
    def total(self) -> int:
        return self.left + self.top + self.right + self.bottom


def core_grid_boxes(
    shape: tuple[int, int],
    core_size: int,
    *,
    core_stride: int | None = None,
) -> list[BoundingBox]:
    """Partition ``shape`` into an exact cover of axis-aligned core boxes.

    The last column/row of cores is pulled back to the domain edge so
    cores cover ``[0,W) x [0,H)`` with no gap and no overlap.  Returns
    boxes in row-major order.
    """
    h, w = shape
    if core_size <= 0:
        raise ValueError(f"core_size must be positive, got {core_size}")
    stride = core_stride if core_stride is not None else core_size
    if stride <= 0:
        raise ValueError(f"core_stride must be positive, got {stride}")

    xs = list(range(0, w, stride))
    ys = list(range(0, h, stride))
    boxes: list[BoundingBox] = []
    for y in ys:
        for x in xs:
            boxes.append(
                BoundingBox(
                    x0=x,
                    y0=y,
                    x1=min(x + core_size, w) if x + core_size > w else x + core_size,
                    y1=min(y + core_size, h) if y + core_size > h else y + core_size,
                )
            )
    return boxes


def iter_core_grid(shape: tuple[int, int], core_size: int) -> Iterator[BoundingBox]:
    yield from core_grid_boxes(shape, core_size)


def read_bbox_for(core: BoundingBox, halo: HaloSpec, width: int, height: int) -> BoundingBox:
    """Expand a core by its halo, clipped to the global domain."""
    return BoundingBox(
        x0=max(0, core.x0 - halo.left),
        y0=max(0, core.y0 - halo.top),
        x1=min(width, core.x1 + halo.right),
        y1=min(height, core.y1 + halo.bottom),
    )


def halo_actual(core: BoundingBox, read: BoundingBox) -> HaloSpec:
    """Recover the per-side halo actually realised after domain clipping."""
    return HaloSpec(
        left=core.x0 - read.x0,
        top=core.y0 - read.y0,
        right=read.x1 - core.x1,
        bottom=read.y1 - core.y1,
    )


def halo_overhead_stats(
    core_boxes: list[BoundingBox],
    read_boxes: list[BoundingBox],
) -> dict[str, float]:
    """Report halo duplicate-compute overhead (prompt §15).

    ``eta_halo = ((C+2h)^2 - C^2) / C^2`` generalised per tile with the
    realised (possibly clipped) halos.
    """
    if len(core_boxes) != len(read_boxes):
        raise ValueError("core/read box lists must have the same length")
    core_px = sum(b.area for b in core_boxes)
    read_px = sum(b.area for b in read_boxes)
    if core_px == 0:
        return {
            "core_pixels": 0.0,
            "read_pixels": 0.0,
            "halo_pixels": 0.0,
            "halo_overhead_pct": 0.0,
            "n_tiles": float(len(core_boxes)),
        }
    return {
        "core_pixels": float(core_px),
        "read_pixels": float(read_px),
        "halo_pixels": float(read_px - core_px),
        "halo_overhead_pct": 100.0 * (read_px - core_px) / core_px,
        "n_tiles": float(len(core_boxes)),
    }
