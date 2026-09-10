"""R17 C0 — freeze the Inc28 no-subdivision accounting surface.

These are the compatibility values the R17 correctness migration must
preserve bit-for-bit on the canonical exact-empty screening scenario
(2x1 tiles of 64 px each: one screened empty, one active).

The whole ``work_accounting`` summary dict is pinned: if R17 changes any
of these numbers on a no-subdivision run, the migration — not the
benchmark — is wrong.  Canonical ledger keys added by R17 (C4a/C4b) are
additive and checked separately; they must not disturb this projection.
"""

import sys
from pathlib import Path

from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.work_accounting import WorkAccounting

sys.path.insert(0, str(Path(__file__).parent))
from test_screening_work_accounting_inc28 import (  # noqa: E402
    CounterForward,
    ExactEmptyContextScreeningPolicy,
    SparseExactSource,
)

FROZEN_SUMMARY: dict[str, float | int] = {
    "accounted_pct": 100.0,
    "active_fraction": 0.5,
    "active_pixels": 64,
    "avoided_pct": 50.0,
    "dense_allocation_events": 0,
    "forward_simulator_calls": 1,
    "forward_simulator_input_pixels": 64,
    "full_chip_pixels": 128,
    "process_points_skipped": 0,
    "read_amplification": 1.0,
    "read_window_calls": 1,
    "read_window_pixels": 64,
    "reused_work_units": 0,
    "rigorous_work_units": 0,
    "screen_queries": 2,
    "screened_fraction": 0.5,
    "screened_out_pixels": 64,
    "tiles_refined": 0,
    "tiles_skipped": 1,
}


def _canonical_run() -> tuple[
    WorkAccounting, SparseExactSource, CounterForward, MetricOnlyTileSink
]:
    source = SparseExactSource()
    forward = CounterForward()
    sink = MetricOnlyTileSink(source.shape)
    accounting = WorkAccounting()
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=ExactEmptyContextScreeningPolicy(
            context_certified=True,
            zero_response_certified=True,
            fill_value=0.0,
            model_provenance="r17-freeze",
        ),
        work_accounting=accounting,
    )
    assert report.n_tiles == 2
    return accounting, source, forward, sink


def test_inc28_summary_projection_is_frozen():
    accounting, source, forward, _sink = _canonical_run()
    summary = accounting.summary()
    # R17 C4b adds canonical ledger keys additively; every Inc28 key must
    # keep its exact frozen value.
    for key, expected in FROZEN_SUMMARY.items():
        assert summary[key] == expected, f"{key}: {summary[key]!r} != {expected!r}"
    assert source.read_calls == 1
    assert forward.calls == 1


def test_behavioral_witnesses_unchanged():
    _accounting, source, forward, sink = _canonical_run()
    # tile_0 is active (read + forward once, no verifier metadata);
    # tile_1 is certified-screened before any read/forward.
    assert sink.covered_pixels == 128
    assert sink.tiles == ["tile_0", "tile_1"]
    assert sink.metadata["tile_0"] == {}
    assert source.read_calls == 1
    assert forward.calls == 1
    screened_meta = sink.metadata["tile_1"]
    assert screened_meta["screening_status"] == "SCREENED_OUT"
    assert screened_meta["screening_certified"] is True
    assert isinstance(screened_meta["screening_certificate_ref"], str)
    assert screened_meta["screened_core_pixels"] == 64.0


def test_accounting_invariants_hold_without_subdivision():
    accounting, _source, _forward, _sink = _canonical_run()
    s = accounting.summary()
    assert s["accounted_pct"] == 100.0
    assert s["active_pixels"] + s["screened_out_pixels"] == s["full_chip_pixels"]
    assert s["reused_work_units"] == 0
    assert s["dense_allocation_events"] == 0
