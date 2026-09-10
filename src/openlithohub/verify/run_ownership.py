"""B04 ownership guard as an RFC-0008 VerificationPlugin dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openlithohub.streaming.vector_runs import OwnedRunSlice
from openlithohub.streaming.verification import (
    TileContext,
    TileVerificationResult,
    VerificationContext,
)


@dataclass
class OwnershipState:
    seen_run_ids: set[str] = field(default_factory=set)
    seen_view_keys: set[tuple[str, int, int, str]] = field(default_factory=set)
    duplicate_views: int = 0
    unique_charge_by_run: dict[str, float] = field(default_factory=dict)


class RunOwnershipVerifier:
    """Semantic guard for physical ownership and analytic-copy firewalls.

    PASS here means only that the streamed geometry/ownership contract is
    internally valid.  It is a dependency of, not a replacement for, the
    process-window verifier.
    """

    name = "b04_run_ownership"
    version = "1.0"

    def __init__(self) -> None:
        self.state = OwnershipState()

    def required_halo(self, context: VerificationContext) -> None:
        return None

    def prepare(self, context: VerificationContext) -> None:
        self.state = OwnershipState()

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        raw = tile.metadata.get("b04_owned_runs")
        if raw is None:
            return TileVerificationResult(
                tile_id=tile.tile_id,
                core_bbox=tile.core_bbox,
                status="INCONCLUSIVE",
                halo_provenance="b04_owned_runs",
                metrics={"halo_error_bound": 0.0},
            )

        for run in raw:
            if not isinstance(run, OwnedRunSlice):
                return TileVerificationResult(
                    tile_id=tile.tile_id,
                    core_bbox=tile.core_bbox,
                    status="INCONCLUSIVE",
                    halo_provenance="b04_owned_runs",
                    metrics={"halo_error_bound": 0.0},
                )
            if run.analytic_duplicate:
                return TileVerificationResult(
                    tile_id=tile.tile_id,
                    core_bbox=tile.core_bbox,
                    status="INCONCLUSIVE",
                    halo_provenance="b04_owned_runs",
                    metrics={"halo_error_bound": 0.0},
                )

            self.state.seen_run_ids.add(run.run_id)
            view_key = (run.run_id, run.x0, run.x1, tile.tile_id)
            if view_key in self.state.seen_view_keys:
                self.state.duplicate_views += 1
            self.state.seen_view_keys.add(view_key)
            self.state.unique_charge_by_run.setdefault(run.run_id, float(run.error_budget_charge))

        total_charge = float(sum(self.state.unique_charge_by_run.values()))
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status="PASS",
            halo_provenance="b04_owned_runs",
            metrics={
                "halo_error_bound": total_charge,
                "owned_unique_runs": float(len(self.state.seen_run_ids)),
                "owned_duplicate_views": float(self.state.duplicate_views),
            },
        )

    def reduce(self, results: Any) -> Any:
        # The repository's existing StreamingVerificationReducer remains
        # authoritative; this plugin does not create a parallel reducer.
        return results

    def finalize(self) -> None:
        return None
