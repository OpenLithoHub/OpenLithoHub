"""R17 C5 — metric and tolerance ownership.

Covers the required matrix items:
26. EPE projection comes only from the unique epe/nm verifier;
27. Hausdorff projection comes only from the unique hausdorff/nm verifier;
28. duplicate convenience-metric ownership is rejected (fail closed);
29. per-verifier tolerances are independent (no scalar shared tolerance).
"""

import pytest
import torch

from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.verification import (
    MetricDescriptor,
    TileContext,
    TileVerificationResult,
    VerificationContext,
    VerifierIdentity,
    project_metric_bound,
)
from openlithohub.streaming.verify_layout import verify_layout


class MetricVerifier:
    """PASS everywhere, owns a typed metric with an upper bound."""

    def __init__(
        self,
        name: str,
        descriptor: MetricDescriptor | None,
        upper: float | None = 1.5,
    ) -> None:
        self.name = name
        self.version = "1.0"
        self.metric_descriptor = descriptor
        self.upper = upper

    def required_halo(self, context: VerificationContext):
        return None

    def prepare(self, context: VerificationContext) -> None:
        return None

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status="PASS",
            upper_bound=self.upper,
        )

    def reduce(self, results):
        return None

    def finalize(self):
        return None


def _source() -> TensorTileSource:
    return TensorTileSource(torch.zeros((16, 16), dtype=torch.float32))


def test_epe_projection_only_from_epe_verifier():
    epe = MetricVerifier(
        "epe-v", MetricDescriptor(quantity="epe", unit="nm", tolerance=2.0), upper=1.5
    )
    other = MetricVerifier(
        "own-v", MetricDescriptor(quantity="contour", unit="nm", tolerance=1.0), upper=9.9
    )
    result = verify_layout(
        _source(),
        simulator=lambda tile: tile,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=[epe, other],
    )
    assert result.continuous_epe_upper_nm == 1.5
    assert result.hausdorff_upper_nm is None  # no hausdorff owner


def test_hausdorff_projection_only_from_hausdorff_verifier():
    hd = MetricVerifier(
        "hd-v", MetricDescriptor(quantity="hausdorff", unit="nm", tolerance=4.0), upper=3.25
    )
    result = verify_layout(
        _source(),
        simulator=lambda tile: tile,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=[hd],
    )
    assert result.hausdorff_upper_nm == 3.25
    assert result.continuous_epe_upper_nm is None


def test_duplicate_metric_ownership_is_ambiguous():
    v1 = MetricVerifier("a", MetricDescriptor(quantity="epe", unit="nm"), upper=1.0)
    v2 = MetricVerifier("b", MetricDescriptor(quantity="epe", unit="nm"), upper=2.0)
    with pytest.raises(ValueError, match="2 verifiers own the epe/nm metric"):
        verify_layout(
            _source(),
            simulator=lambda tile: tile,
            core_size=8,
            halo_policy=LegacyFixedHaloPolicy(0),
            verifiers=[v1, v2],
        )


def test_multi_verifier_scalar_tolerance_is_rejected():
    v1 = MetricVerifier("a", MetricDescriptor(quantity="epe", unit="nm"))
    v2 = MetricVerifier("b", MetricDescriptor(quantity="hausdorff", unit="nm"))
    with pytest.raises(ValueError, match="single-verifier legacy convenience"):
        run_streaming(
            _source(),
            MetricOnlyTileSink((16, 16)),
            lambda tile: tile,
            core_size=8,
            halo_policy=LegacyFixedHaloPolicy(0),
            verifiers=[v1, v2],
            tolerance_nm=3.0,
        )


def test_per_verifier_tolerance_is_independent():
    class HaloBoundVerifier(MetricVerifier):
        """PASS with a halo error bound; the budget gate uses own tolerance."""

        def verify_tile(self, tile: TileContext) -> TileVerificationResult:
            return TileVerificationResult(
                tile_id=tile.tile_id,
                core_bbox=tile.core_bbox,
                status="PASS",
                upper_bound=0.0,
                halo_provenance="test_kernel",
                metrics={"halo_error_bound": 1.0},
            )

    tight = HaloBoundVerifier("tight", MetricDescriptor(quantity="epe", unit="nm", tolerance=0.5))
    loose = HaloBoundVerifier(
        "loose", MetricDescriptor(quantity="hausdorff", unit="nm", tolerance=50.0)
    )
    report = run_streaming(
        _source(),
        MetricOnlyTileSink((16, 16)),
        lambda tile: tile,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=[tight, loose],
    )
    res = report.verification_results
    # identical halo error (1.0): the tight verifier's budget gate demotes it
    # to INCONCLUSIVE while the loose verifier stays PASS — independently.
    assert res[VerifierIdentity("tight", "1.0")].status == "INCONCLUSIVE"
    assert res[VerifierIdentity("loose", "1.0")].status == "PASS"


def test_project_metric_bound_unit_directly():
    identity = VerifierIdentity("v", "1")
    from openlithohub.streaming.verification import GlobalVerificationResult

    def result(upper):
        return GlobalVerificationResult(
            status="PASS",
            worst_upper_bound=upper,
            worst_lower_bound=None,
            n_tiles=1,
            n_pass=1,
            n_fail=0,
            n_inconclusive=0,
            coverage="NOT_APPLICABLE",
            halo_error_budget=0.0,
            error_budget={},
        )

    descriptors = {identity: MetricDescriptor(quantity="epe", unit="nm")}
    assert (
        project_metric_bound({identity: result(2.5)}, descriptors, quantity="epe", unit="nm") == 2.5
    )
    assert (
        project_metric_bound({identity: result(None)}, descriptors, quantity="epe", unit="nm")
        is None
    )
    assert (
        project_metric_bound({identity: result(2.5)}, descriptors, quantity="hausdorff", unit="nm")
        is None
    )
