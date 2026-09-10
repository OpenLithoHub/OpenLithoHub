"""Work-avoidance accounting for the streaming pipeline.

The counters distinguish *unique expensive core area* from repeated reads and
refinement attempts.  This makes the accounting suitable for scale-first
benchmarks: a refinement may increase simulator calls/read work without
silently inflating the fraction of the chip classified as active.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import BoundingBox


@dataclass
class WorkAccounting:
    """Per-run counters for work done vs work avoided."""

    full_chip_pixels: int = 0
    active_pixels: int = 0
    screened_out_pixels: int = 0
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
    dense_allocation_events: int = 0
    _tile_areas: dict[str, int] = field(default_factory=dict, repr=False)
    _screened_tiles: set[str] = field(default_factory=set, repr=False)

    def record_active_core(self, tile_id: str, core_bbox: BoundingBox) -> None:
        """Record unique core area that requires expensive processing."""
        if tile_id not in self._tile_areas and tile_id not in self._screened_tiles:
            area = core_bbox.area
            self._tile_areas[tile_id] = area
            self.active_pixels += area

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

    def record_screened_out(self, tile_id: str, area: int) -> None:
        """Record a uniquely owned core that is certified irrelevant."""
        if tile_id in self._tile_areas or tile_id in self._screened_tiles:
            return
        self._screened_tiles.add(tile_id)
        self.screened_out_pixels += int(area)
        self.tiles_skipped += 1

    def record_refinement(self, old_tile_id: str, new_tile_id: str, area: int) -> None:
        """Record repeated expensive work without changing unique active area."""
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
        return self.active_pixels / self.full_chip_pixels

    @property
    def screened_fraction(self) -> float:
        if self.full_chip_pixels <= 0:
            return 0.0
        return self.screened_out_pixels / self.full_chip_pixels

    @property
    def avoided_pct(self) -> float:
        return 100.0 * self.screened_fraction

    @property
    def read_amplification(self) -> float:
        if self.active_pixels <= 0:
            return 0.0
        return self.read_window_pixels / self.active_pixels

    def summary(self) -> dict[str, float | int]:
        accounted = self.active_pixels + self.screened_out_pixels
        accounted_pct = (
            100.0 * accounted / self.full_chip_pixels if self.full_chip_pixels > 0 else 0.0
        )
        return {
            "full_chip_pixels": self.full_chip_pixels,
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
            "tiles_refined": self.tiles_refined,
            "dense_allocation_events": self.dense_allocation_events,
        }
