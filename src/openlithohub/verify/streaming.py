"""Minimal streaming contracts for future full-chip proof verification."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .interface_runs import HorizontalRun
from .types import CertificateStatus


@dataclass(frozen=True)
class CoreWindow:
    tile_id: str
    core_bbox_px: tuple[int, int, int, int]
    loaded_bbox_px: tuple[int, int, int, int]


class TileSource(Protocol):
    def iter_cores(self) -> Iterable[CoreWindow]: ...


class LayoutRunSource(Protocol):
    def iter_runs_for_bbox(self, bbox_px: tuple[int, int, int, int]) -> Iterable[HorizontalRun]: ...


class CertificateSink(Protocol):
    def emit(self, tile_id: str, status: CertificateStatus, artifact_sha256: str) -> None: ...
