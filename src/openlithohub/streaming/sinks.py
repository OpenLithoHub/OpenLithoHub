"""Streaming tile sinks (RFC 0008, prompt §3).

A ``TileSink`` receives each tile's trusted-core result.  Implementations
decide the output materialisation:

- :class:`TensorTileSink` — assemble a full dense tensor (small layouts,
  backward-compatible behaviour).
- :class:`MemmapTileSink` — page the output to an on-disk memmap
  (out-of-core big rasters).
- :class:`MetricOnlyTileSink` — keep only per-tile metrics/aggregates and
  never materialise a raster at all (the QDM-critical mode).

Because only trusted cores are ever written, no weight map / blend pass
is required: cores are an exact partition of the output domain.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .geometry import BoundingBox


@runtime_checkable
class TileSink(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def finalize(self) -> Any: ...


class TensorTileSink:
    """Assemble the full dense output tensor in host memory."""

    def __init__(self, shape: tuple[int, int], dtype: torch.dtype = torch.float32) -> None:
        self._shape = shape
        self.output = torch.zeros(shape, dtype=dtype)
        self._written: set[str] = set()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tile_id in self._written:
            raise ValueError(f"duplicate core write for {tile_id!r}")
        self._written.add(tile_id)
        expected = (bbox.height, bbox.width)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"core tensor shape {tuple(tensor.shape)} != bbox {expected}")
        self.output[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1] = tensor

    def finalize(self) -> torch.Tensor:
        return self.output


class MemmapTileSink:
    """Page trusted cores straight into an on-disk memmap raster."""

    def __init__(
        self,
        shape: tuple[int, int],
        path: str | Path,
        dtype: np.dtype[Any] | None = None,
    ) -> None:
        self._shape = shape
        self.path = Path(path)
        if dtype is None:
            dtype = np.dtype(np.float32)
        self._map: np.memmap = np.memmap(self.path, dtype=dtype, mode="w+", shape=shape)
        self._written: set[str] = set()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tile_id in self._written:
            raise ValueError(f"duplicate core write for {tile_id!r}")
        self._written.add(tile_id)
        window = tensor.detach().cpu().numpy().astype(self._map.dtype, copy=False)
        self._map[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1] = window

    def finalize(self) -> np.memmap:
        self._map.flush()
        return self._map


class MetricOnlyTileSink:
    """Aggregate per-tile results without ever materialising a raster.

    Reducers receive one tile at a time, so peak memory is O(1) beyond the
    running aggregates.  This is the sink mode QDM-style verification needs
    (certificates, worst bounds, violation lists, provenance).
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self._shape = shape
        self.tiles: list[str] = []
        self.covered_pixels = 0
        self.metrics: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def write_core(
        self,
        tile_id: str,
        bbox: BoundingBox,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tiles.append(tile_id)
        self.covered_pixels += bbox.area
        self.metadata[tile_id] = metadata or {}
        for key, value in (metadata or {}).items():
            if isinstance(value, (int, float)):
                prev = self.metrics.get(f"max_{key}", float("-inf"))
                self.metrics[f"max_{key}"] = max(prev, float(value))

    def finalize(self) -> dict[str, Any]:
        return {
            "n_tiles": len(self.tiles),
            "covered_pixels": self.covered_pixels,
            "metrics": dict(self.metrics),
        }


def covered_pixels(sink: TileSink) -> int:
    """Total trusted-core pixels written (for gap/overlap audits)."""
    written = getattr(sink, "covered_pixels", None)
    if written is not None:
        return int(written)
    written = getattr(sink, "_written", None)
    if isinstance(written, Iterable):
        return len(written)  # type: ignore[arg-type]
    return -1
