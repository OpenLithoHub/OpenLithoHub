"""Streaming tile sources (RFC 0008, prompt §2).

A ``TileSource`` produces the *read region* tensor for a tile request on
demand.  Implementations must never require the whole layout as a dense
host tensor:

- :class:`TensorTileSource` wraps an already-materialised tensor
  (compatibility, tests, small benchmarks).
- :class:`MemmapTensorTileSource` reads windows out of an on-disk
  ``np.memmap`` raster (out-of-core big raster).
- :class:`VectorLayoutTileSource` rasterizes only the objects that
  intersect the requested window of a vector GDS/OASIS layout through a
  :class:`SpatialLayoutIndex`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .geometry import BoundingBox


@runtime_checkable
class TileSource(Protocol):
    """Read-only windowed view over a full-chip layout."""

    @property
    def shape(self) -> tuple[int, int]:
        """Global raster shape ``(H, W)`` in pixels."""
        ...

    @property
    def pixel_size_nm(self) -> float: ...

    def read_window(self, bbox: BoundingBox) -> torch.Tensor:
        """Return a ``(H, W)`` float tensor for the half-open bbox."""
        ...


class TensorTileSource:
    """Wrap an existing dense tensor. Zero-copy via narrow/as_strided."""

    def __init__(self, tensor: torch.Tensor, pixel_size_nm: float = 1.0) -> None:
        t = tensor.detach()
        if t.ndim != 2:
            raise ValueError(f"expected a 2-D layout tensor, got shape {tuple(t.shape)}")
        self._t = t
        self._pixel_nm = float(pixel_size_nm)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._t.shape[0], self._t.shape[1])

    @property
    def pixel_size_nm(self) -> float:
        return self._pixel_nm

    def read_window(self, bbox: BoundingBox) -> torch.Tensor:
        return self._t[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1].float()


class MemmapTensorTileSource:
    """Out-of-core windowed reads over an on-disk raster via ``np.memmap``.

    The backing file may be far larger than RAM; only requested windows are
    paged in.  Accepts a ``.npy``/raw path plus dtype/shape, or an existing
    ``np.memmap``.
    """

    def __init__(
        self,
        source: str | Path | np.memmap,
        *,
        dtype: np.dtype[Any] | None = None,
        shape: tuple[int, int] | None = None,
        pixel_size_nm: float = 1.0,
    ) -> None:
        if isinstance(source, (str, Path)):
            if dtype is not None and shape is not None:
                # Raw raster: caller must supply the on-disk layout.
                self._map: np.memmap = np.memmap(source, dtype=dtype, mode="r", shape=shape)
            else:
                # .npy path: np.load with mmap_mode pages windows from disk.
                self._map = np.load(source, mmap_mode="r")
        else:
            self._map = source
        if self._map.ndim != 2:
            raise ValueError(f"expected a 2-D raster, got shape {self._map.shape}")
        self._pixel_nm = float(pixel_size_nm)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._map.shape[0]), int(self._map.shape[1]))

    @property
    def pixel_size_nm(self) -> float:
        return self._pixel_nm

    def read_window(self, bbox: BoundingBox) -> torch.Tensor:
        window = np.asarray(self._map[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1])
        return torch.from_numpy(np.ascontiguousarray(window)).float()


class SpatialLayoutIndex:
    """Minimal grid-bucket spatial index over vector layout boxes.

    Each layout object is an axis-aligned bounding box plus optional fill
    (1.0 foreground).  Objects are bucketed on a coarse uniform grid so a
    window query only visits intersecting buckets.  This is the smallest
    useful realisation of the ``SpatialLayoutIndex`` concept from RFC 0008;
    a klayout-Region-backed implementation can drop in later behind the
    same interface.
    """

    def __init__(
        self,
        boxes: list[tuple[int, int, int, int]],
        raster_shape: tuple[int, int],
        bucket: int = 256,
    ) -> None:
        if not boxes:
            raise ValueError("SpatialLayoutIndex needs at least one layout box")
        h, w = raster_shape
        self._shape = raster_shape
        self._bucket = max(1, int(bucket))
        nbx = max(1, (w + self._bucket - 1) // self._bucket)
        nby = max(1, (h + self._bucket - 1) // self._bucket)
        self._buckets: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for x0, y0, x1, y1 in boxes:
            if x1 <= x0 or y1 <= y0:
                continue
            for by in range(y0 // self._bucket, min((y1 - 1) // self._bucket, nby - 1) + 1):
                for bx in range(x0 // self._bucket, min((x1 - 1) // self._bucket, nbx - 1) + 1):
                    self._buckets.setdefault((bx, by), []).append((x0, y0, x1, y1))

    def query(self, bbox: BoundingBox) -> list[tuple[int, int, int, int]]:
        found: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        nbx = max(1, (self._shape[1] + self._bucket - 1) // self._bucket)
        nby = max(1, (self._shape[0] + self._bucket - 1) // self._bucket)
        for by in range(bbox.y0 // self._bucket, min((bbox.y1 - 1) // self._bucket, nby - 1) + 1):
            for bx in range(
                bbox.x0 // self._bucket, min((bbox.x1 - 1) // self._bucket, nbx - 1) + 1
            ):
                for box in self._buckets.get((bx, by), ()):  # may repeat across buckets
                    if box in seen:
                        continue
                    seen.add(box)
                    ix0, iy0, ix1, iy1 = box
                    ox0, oy0 = max(ix0, bbox.x0), max(iy0, bbox.y0)
                    ox1, oy1 = min(ix1, bbox.x1), min(iy1, bbox.y1)
                    if ox1 > ox0 and oy1 > oy0:
                        found.append((ox0, oy0, ox1, oy1))
        return found


class VectorLayoutTileSource:
    """Rasterize only the objects intersecting each requested window."""

    def __init__(
        self,
        index: SpatialLayoutIndex,
        shape: tuple[int, int],
        pixel_size_nm: float = 1.0,
        background: float = 0.0,
        fill: float = 1.0,
    ) -> None:
        self._index = index
        self._shape = shape
        self._pixel_nm = float(pixel_size_nm)
        self._background = float(background)
        self._fill = float(fill)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def pixel_size_nm(self) -> float:
        return self._pixel_nm

    def read_window(self, bbox: BoundingBox) -> torch.Tensor:
        window = np.full((bbox.height, bbox.width), self._background, dtype=np.float32)
        for x0, y0, x1, y1 in self._index.query(bbox):
            window[y0 - bbox.y0 : y1 - bbox.y0, x0 - bbox.x0 : x1 - bbox.x0] = self._fill
        return torch.from_numpy(window)
