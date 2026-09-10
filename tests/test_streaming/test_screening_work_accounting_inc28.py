from dataclasses import dataclass

import torch

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.screening import (
    ExactEmptyContextScreeningPolicy,
)
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.verify.interface_runs import HorizontalRun


@dataclass
class SparseExactSource:
    shape: tuple[int, int] = (8, 16)
    pixel_size_nm: float = 1.0
    read_calls: int = 0

    def iter_runs_for_bbox(self, bbox_px):
        x0, y0, x1, y1 = bbox_px
        run = HorizontalRun(2, 1, 3)
        if y0 <= run.y < y1 and min(run.x1, x1) > max(run.x0, x0):
            return (
                HorizontalRun(
                    run.y,
                    max(run.x0, x0),
                    min(run.x1, x1),
                ),
            )
        return ()

    def read_window(self, bbox: BoundingBox) -> torch.Tensor:
        self.read_calls += 1
        out = torch.zeros((bbox.height, bbox.width), dtype=torch.float32)
        for run in self.iter_runs_for_bbox((bbox.x0, bbox.y0, bbox.x1, bbox.y1)):
            out[
                run.y - bbox.y0,
                run.x0 - bbox.x0 : run.x1 - bbox.x0,
            ] = 1.0
        return out


class CounterForward:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return tile


def policy(*, context_certified: bool = True):
    return ExactEmptyContextScreeningPolicy(
        context_certified=context_certified,
        zero_response_certified=True,
        model_provenance="identity-zero-preserving",
    )


def test_certified_empty_screen_skips_read_and_forward():
    source = SparseExactSource()
    forward = CounterForward()
    sink = MetricOnlyTileSink(source.shape)
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=policy(),
    )
    work = report.work_accounting
    assert report.n_tiles == 2
    assert source.read_calls == 1
    assert forward.calls == 1
    assert work["active_pixels"] == 64
    assert work["screened_out_pixels"] == 64
    assert work["forward_simulator_calls"] == 1
    assert work["read_window_calls"] == 1
    assert work["screen_queries"] == 2
    assert work["avoided_pct"] == 50.0
    assert work["accounted_pct"] == 100.0
    assert sink.covered_pixels == 128


def test_uncertified_context_fails_open_to_active_processing():
    source = SparseExactSource()
    forward = CounterForward()
    sink = MetricOnlyTileSink(source.shape)
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=policy(context_certified=False),
    )
    work = report.work_accounting
    assert source.read_calls == 2
    assert forward.calls == 2
    assert work["screened_out_pixels"] == 0
    assert work["active_pixels"] == 128
    assert work["accounted_pct"] == 100.0


def test_screen_does_not_bypass_uncertified_verifier_scope():
    from openlithohub.streaming.verification import DummyVerifier

    source = SparseExactSource()
    forward = CounterForward()
    sink = MetricOnlyTileSink(source.shape)
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=(DummyVerifier(halo_px=0),),
        screening_policy=policy(),
        max_requeues=0,
    )
    # The screen does not claim to certify arbitrary verification plugins.
    assert forward.calls == 2
    assert report.verification is not None
    assert report.verification.n_tiles == 2


def test_public_verify_layout_exposes_integrated_work_accounting():
    from openlithohub.streaming.verify_layout import verify_layout

    source = SparseExactSource()
    forward = CounterForward()
    result = verify_layout(
        source,
        simulator=forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=policy(),
    )
    assert result.total_tiles == 2
    assert result.work_accounting["active_pixels"] == 64
    assert result.work_accounting["screened_out_pixels"] == 64
    assert result.work_accounting["forward_simulator_calls"] == 1
    assert "screen=exact-empty-context" in result.model_provenance
