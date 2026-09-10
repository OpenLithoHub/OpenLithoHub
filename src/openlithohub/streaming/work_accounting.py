"""Dual work ledgers for the streaming pipeline (R17 C4a/C4b).

Two ledgers answer two different questions and must never be merged:

- **Terminal coverage ledger** (owned by :class:`~.ownership.OwnershipTree`):
  *how was every physical pixel finally discharged?*
  ``terminal_active + terminal_exact_skip + terminal_verification_skip
  == full_chip_pixels``.

- **Historical work ledger** (this module): *which physical area's
  ancestor chain ever executed the expensive forward model?*
  ``unique_forward_pixels + pre_forward_avoided_pixels == full_chip_pixels``.

The historical ledger is region-keyed: forwarded core rectangles are
tracked as a nested-or-disjoint union, so a parent forward is never
double-counted when its subdivided children re-run, and a screened child
of a forwarded parent correctly remains part of the unique-forward area.

Compatibility aliases (Inc28 surface): ``active_pixels`` projects
``unique_forward_pixels`` and ``screened_out_pixels`` projects
``pre_forward_avoided_pixels``.  Repeated-work counters (forward calls,
read windows, refinement re-runs) are historical *volume*, not coverage,
and stay separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import BoundingBox


@dataclass
class WorkAccounting:
    """Per-run counters for work done vs work avoided (dual ledger)."""

    full_chip_pixels: int = 0
    reused_work_units: int = 0
    rigorous_work_units: int = 0
    forward_simulator_calls: int = 0
    forward_simulator_input_pixels: int = 0
    read_window_calls: int = 0
    read_window_pixels: int = 0
    screen_queries: int = 0
    process_points_skipped: int = 0
    tiles_skipped: int = 0
    tiles_refined: int = 0
    tiles_screened_out: int = 0
    dense_allocation_events: int = 0
    n_subdivisions: int = 0
    _forward_union: list[BoundingBox] = field(default_factory=list, repr=False)
    _unique_forward_pixels: int = 0
    _tile_areas: dict[str, int] = field(default_factory=dict, repr=False)
    _screened_tiles: set[str] = field(default_factory=set, repr=False)

    # ------------------------------------------------------------------
    # Historical work ledger (canonical, region-keyed)
    # ------------------------------------------------------------------

    def record_forward_core(self, core_bbox: BoundingBox) -> None:
        """Register a forwarded core area in the unique-forward union.

        Rectangles arrive nested-or-disjoint by construction (subdivision
        partitions parents exactly; halo refinement keeps the same core),
        which keeps the incremental union exact.
        """
        contained = any(_contains(cover, core_bbox) for cover in self._forward_union)
        if contained:
            return
        absorbed = [cover for cover in self._forward_union if _contains(core_bbox, cover)]
        for cover in absorbed:
            self._unique_forward_pixels -= cover.area
            self._forward_union.remove(cover)
        self._forward_union.append(core_bbox)
        self._unique_forward_pixels += core_bbox.area

    @property
    def unique_forward_pixels(self) -> int:
        """Area whose ancestor chain executed at least one full forward."""
        return self._unique_forward_pixels

    @property
    def pre_forward_avoided_pixels(self) -> int:
        """Area certified irrelevant before any forward in its ancestry."""
        if self.full_chip_pixels <= 0:
            return 0
        return max(0, self.full_chip_pixels - self._unique_forward_pixels)

    # ------------------------------------------------------------------
    # Compatibility aliases (Inc28 surface)
    # ------------------------------------------------------------------

    @property
    def active_pixels(self) -> int:
        """Alias: unique forward-work area (historical ledger)."""
        return self.unique_forward_pixels

    @property
    def screened_out_pixels(self) -> int:
        """Alias: pre-forward avoided area (historical ledger)."""
        return self.pre_forward_avoided_pixels

    # ------------------------------------------------------------------
    # Event bookkeeping
    # ------------------------------------------------------------------

    def record_active_core(self, tile_id: str, core_bbox: BoundingBox) -> None:
        """Record that a tile entered the active (expensive) path."""
        if tile_id not in self._tile_areas and tile_id not in self._screened_tiles:
            self._tile_areas[tile_id] = core_bbox.area

    def record_read_window(self, pixels: int) -> None:
        self.read_window_calls += 1
        self.read_window_pixels += int(pixels)

    def record_forward(self, input_pixels: int) -> None:
        self.forward_simulator_calls += 1
        self.forward_simulator_input_pixels += int(input_pixels)

    def record_screen_query(self) -> None:
        self.screen_queries += 1

    def record_tile(
        self,
        tile_id: str,
        core_bbox: BoundingBox,
        *,
        simulator_called: bool = True,
    ) -> None:
        """Backward-compatible active-tile recording."""
        self.record_active_core(tile_id, core_bbox)
        if simulator_called:
            self.record_forward(core_bbox.area)
            self.record_forward_core(core_bbox)

    def record_screened_out(self, tile_id: str, area: int) -> None:
        """Record a uniquely owned core discharged before any forward."""
        del area
        if tile_id in self._tile_areas or tile_id in self._screened_tiles:
            return
        self._screened_tiles.add(tile_id)
        self.tiles_skipped += 1
        self.tiles_screened_out += 1

    def record_subdivision(self, parent_core: BoundingBox) -> None:
        """Record one subdivision event (historical work is area-unioned)."""
        del parent_core
        self.n_subdivisions += 1

    def record_refinement(self, old_tile_id: str, new_tile_id: str, area: int) -> None:
        """Record repeated expensive work without changing unique area."""
        del old_tile_id, new_tile_id
        self.tiles_refined += 1
        self.reused_work_units += int(area)

    def record_dense_allocation(self, pixels: int) -> None:
        del pixels
        self.dense_allocation_events += 1

    @property
    def n_tiles(self) -> int:
        return len(self._tile_areas) + len(self._screened_tiles)

    @property
    def active_fraction(self) -> float:
        if self.full_chip_pixels <= 0:
            return 0.0
        return self.unique_forward_pixels / self.full_chip_pixels

    @property
    def screened_fraction(self) -> float:
        if self.full_chip_pixels <= 0:
            return 0.0
        return self.pre_forward_avoided_pixels / self.full_chip_pixels

    @property
    def avoided_pct(self) -> float:
        return 100.0 * self.screened_fraction

    @property
    def read_amplification(self) -> float:
        if self.unique_forward_pixels <= 0:
            return 0.0
        return self.read_window_pixels / self.unique_forward_pixels

    def summary(self) -> dict[str, float | int]:
        accounted = self.unique_forward_pixels + self.pre_forward_avoided_pixels
        accounted_pct = (
            100.0 * accounted / self.full_chip_pixels if self.full_chip_pixels > 0 else 0.0
        )
        return {
            "full_chip_pixels": self.full_chip_pixels,
            # canonical historical-ledger keys (R17 C4b)
            "unique_forward_pixels": self.unique_forward_pixels,
            "pre_forward_avoided_pixels": self.pre_forward_avoided_pixels,
            # Inc28 compatibility aliases
            "active_pixels": self.active_pixels,
            "active_fraction": self.active_fraction,
            "screened_out_pixels": self.screened_out_pixels,
            "screened_fraction": self.screened_fraction,
            "avoided_pct": round(self.avoided_pct, 4),
            "accounted_pct": round(accounted_pct, 4),
            "reused_work_units": self.reused_work_units,
            "rigorous_work_units": self.rigorous_work_units,
            "forward_simulator_calls": self.forward_simulator_calls,
            "forward_simulator_input_pixels": self.forward_simulator_input_pixels,
            "read_window_calls": self.read_window_calls,
            "read_window_pixels": self.read_window_pixels,
            "read_amplification": self.read_amplification,
            "screen_queries": self.screen_queries,
            "process_points_skipped": self.process_points_skipped,
            "tiles_skipped": self.tiles_skipped,
            "tiles_screened_out": self.tiles_screened_out,
            "tiles_refined": self.tiles_refined,
            "n_subdivisions": self.n_subdivisions,
            "dense_allocation_events": self.dense_allocation_events,
        }


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.x0 <= inner.x0
        and outer.y0 <= inner.y0
        and outer.x1 >= inner.x1
        and outer.y1 >= inner.y1
    )
