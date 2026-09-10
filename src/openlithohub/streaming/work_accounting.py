"""Work-avoidance accounting for the streaming pipeline (philosophy §6).

Tracks what expensive work B04 avoided relative to a dense whole-chip
baseline, so the code can answer "what did B04 avoid?" rather than merely
"how long did it run?".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import BoundingBox


@dataclass
class WorkAccounting:
    """Per-run counters for work done vs work avoided.

    Units are pixel-areas unless stated otherwise.  ``full_chip_pixels``
    is the denominator that makes all other counters meaningful as
    percentages of a dense whole-chip baseline.
    """

    full_chip_pixels: int = 0
    active_pixels: int = 0
    screened_out_pixels: int = 0
    reused_work_units: int = 0
    rigorous_work_units: int = 0
    forward_simulator_calls: int = 0
    process_points_skipped: int = 0
    tiles_skipped: int = 0
    tiles_refined: int = 0
    dense_allocation_events: int = 0
    _tile_areas: dict[str, int] = field(default_factory=dict, repr=False)

    def record_tile(
        self,
        tile_id: str,
        core_bbox: BoundingBox,
        *,
        simulator_called: bool = True,
    ) -> None:
        """Record one tile's contribution to the active work set."""
        area = core_bbox.area
        self._tile_areas[tile_id] = area
        self.active_pixels += area
        if simulator_called:
            self.forward_simulator_calls += 1

    def record_screened_out(self, tile_id: str, area: int) -> None:
        """Record a tile that was cheaply proven irrelevant."""
        self.screened_out_pixels += area
        self.tiles_skipped += 1

    def record_refinement(self, old_tile_id: str, new_tile_id: str, area: int) -> None:
        self.tiles_refined += 1
        self.active_pixels += area

    def record_dense_allocation(self, pixels: int) -> None:
        """Flag a full-chip dense allocation (scaling-contract violation)."""
        self.dense_allocation_events += 1

    @property
    def n_tiles(self) -> int:
        return len(self._tile_areas)

    @property
    def avoided_pct(self) -> float:
        """Percentage of full-chip area that did NOT need expensive work."""
        if self.full_chip_pixels == 0:
            return 0.0
        return 100.0 * (1.0 - self.active_pixels / self.full_chip_pixels)

    def summary(self) -> dict[str, float | int]:
        return {
            "full_chip_pixels": self.full_chip_pixels,
            "active_pixels": self.active_pixels,
            "screened_out_pixels": self.screened_out_pixels,
            "avoided_pct": round(self.avoided_pct, 2),
            "reused_work_units": self.reused_work_units,
            "rigorous_work_units": self.rigorous_work_units,
            "forward_simulator_calls": self.forward_simulator_calls,
            "process_points_skipped": self.process_points_skipped,
            "tiles_skipped": self.tiles_skipped,
            "tiles_refined": self.tiles_refined,
            "dense_allocation_events": self.dense_allocation_events,
        }
