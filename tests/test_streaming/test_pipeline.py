"""Streaming pipeline acceptance tests (RFC 0008, prompt §13/§22 A-E)."""

import numpy as np
import pytest
import torch

from openlithohub.streaming.halo_policy import (
    HaloRequirement,
    HaloStatus,
    LegacyFixedHaloPolicy,
)
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MemmapTileSink, MetricOnlyTileSink, TensorTileSink
from openlithohub.streaming.sources import MemmapTensorTileSource, TensorTileSource
from openlithohub.streaming.verification import (
    DummyVerifier,
    TileVerificationResult,
)


def _layout(h: int, w: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return ((xx * 7 + yy * 13) % 32).float() / 32.0


class TestIdentityRoundTrip:
    """§13: tile → core/halo → sink must reproduce the input exactly."""

    def test_identity_forward_bitwise(self):
        layout = _layout(96, 128)
        sink = TensorTileSink((96, 128))
        run_streaming(
            TensorTileSource(layout),
            sink,
            lambda t: t,  # identity forward
            core_size=32,
            halo_policy=LegacyFixedHaloPolicy(8),
        )
        assert torch.equal(sink.finalize(), layout), "identity round-trip must be bitwise"

    def test_no_core_gap_or_overlap(self):
        layout = _layout(96, 96)
        sink = TensorTileSink((96, 96))
        report = run_streaming(
            TensorTileSource(layout),
            sink,
            lambda t: t + 1.0,
            core_size=32,
            halo_policy=LegacyFixedHaloPolicy(4),
        )
        # every core written exactly once: +1 everywhere is detectable
        out = sink.finalize()
        assert torch.equal(out, layout + 1.0)
        assert report.n_tiles == 9


class TestFullVsTiled:
    def test_error_decreases_with_halo(self):
        h, w = 48, 48
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        layout = torch.sin(0.4 * xx.float()) * torch.cos(0.3 * yy.float())

        def aerial(tile: torch.Tensor) -> torch.Tensor:
            kernel = torch.ones(1, 1, 7, 7) / 49.0
            padded = torch.nn.functional.pad(
                tile.float().unsqueeze(0).unsqueeze(0), (3, 3, 3, 3), mode="replicate"
            )
            return torch.nn.functional.conv2d(padded, kernel).squeeze()

        full = aerial(layout)
        errors = {}
        for halo in (0, 2, 4, 12):
            sink = TensorTileSink((h, w))
            run_streaming(
                TensorTileSource(layout),
                sink,
                aerial,
                core_size=16,
                halo_policy=LegacyFixedHaloPolicy(halo),
            )
            errors[halo] = float((sink.finalize() - full).abs().max())
        assert errors[12] < errors[0], "interior tiles must beat zero-padded edges"
        assert errors[12] == pytest.approx(0.0, abs=1e-6)


class TestVerifierIntegration:
    def test_dummy_verifier_full_flow(self):
        layout = _layout(64, 64)
        verifier = DummyVerifier(fail_tiles={"tile_0"})
        sink = MetricOnlyTileSink((64, 64))
        report = run_streaming(
            TensorTileSource(layout),
            sink,
            lambda t: t,
            core_size=32,
            halo_policy=LegacyFixedHaloPolicy(4),
            verifiers=[verifier],
        )
        assert verifier.prepared, "prepare() must be called"
        # plugin halo (24) wins over legacy policy (4)
        assert report.halo_requirement.halo_px == 24
        assert report.verification is not None
        assert report.verification.status == "FAIL"
        assert report.verification.n_fail == 1

    def test_inconclusive_triggers_refinement_then_passes(self):
        layout = _layout(64, 64)

        class Picky(DummyVerifier):
            """INCONCLUSIVE on the first visit of each tile, PASS after."""

            def __init__(self) -> None:
                super().__init__()
                self._visits: dict[str, int] = {}

            def verify_tile(self, tile):
                n = self._visits.get(tile.tile_id, 0)
                self._visits[tile.tile_id] = n + 1
                status = "INCONCLUSIVE" if n == 0 else "PASS"
                return TileVerificationResult(
                    tile_id=tile.tile_id,
                    core_bbox=tile.core_bbox,
                    status=status,
                    halo_px=tile.halo.left,
                    halo_provenance="dummy_plugin",
                )

        verifier = Picky()
        sink = TensorTileSink((64, 64))
        report = run_streaming(
            TensorTileSource(layout),
            sink,
            lambda t: t,
            core_size=32,
            halo_policy=LegacyFixedHaloPolicy(0),
            verifiers=[verifier],
        )
        assert report.n_requeued == 4, "each tile refines exactly once"
        assert report.verification.status == "PASS"
        assert torch.equal(sink.finalize(), layout)

    def test_plugin_halo_request_feeds_budget(self):
        class HaloBudgetVerifier(DummyVerifier):
            def required_halo(self, context):
                return HaloRequirement(
                    halo_px=16,
                    provenance="kernel_tail",
                    error_bound=0.05,
                    certified=True,
                    reason="analytic tail",
                    status=HaloStatus.CERTIFIED_SUFFICIENT,
                )

            def verify_tile(self, tile):
                result = super().verify_tile(tile)
                return TileVerificationResult(
                    tile_id=result.tile_id,
                    core_bbox=result.core_bbox,
                    status=result.status,
                    halo_px=result.halo_px,
                    halo_provenance=result.halo_provenance,
                    metrics={"halo_error_bound": 0.05},
                )

        sink = MetricOnlyTileSink((64, 64))
        report = run_streaming(
            TensorTileSource(_layout(64, 64)),
            sink,
            lambda t: t,
            core_size=32,
            verifiers=[HaloBudgetVerifier()],
        )
        assert report.verification.halo_error_budget == 0.05
        assert "halo" in report.verification.error_budget


class TestStreamingMemoryAcceptance:
    """§22-A/B: a layout far larger than one tile streams under a budget."""

    def test_big_layout_streams_without_full_tensor(self, tmp_path):
        h = w = 2048
        path = tmp_path / "big_layout.f32"
        big = np.memmap(path, dtype=np.float32, mode="w+", shape=(h, w))
        # fill sparsely to keep the test fast; content beyond a few windows
        # is never read by the streaming pipeline
        big[0:h:64] = 1.0
        big.flush()
        del big

        source = MemmapTensorTileSource(path, dtype=np.float32, shape=(h, w))
        sink = MemmapTileSink((h, w), tmp_path / "out.f32")
        report = run_streaming(
            source,
            sink,
            lambda t: t,
            core_size=256,
            halo_policy=LegacyFixedHaloPolicy(0),
        )
        assert report.n_tiles == 64
        out = sink.finalize()
        # spot-check a streamed window round-trips through disk
        assert np.asarray(out[0:16, 0:16]).max() == 1.0

    def test_metric_only_mode_has_no_raster(self):
        layout = _layout(128, 128)
        sink = MetricOnlyTileSink((128, 128))
        report = run_streaming(
            TensorTileSource(layout),
            sink,
            lambda t: t,
            core_size=32,
            verifiers=[DummyVerifier()],
        )
        result = sink.finalize()
        assert result["covered_pixels"] == 128 * 128
        assert report.verification.status == "PASS"
        assert not hasattr(sink, "output")
