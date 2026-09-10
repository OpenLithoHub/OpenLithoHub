"""R17 C2 — sink capabilities and tensor-free certified commits.

Covers the required matrix items:
24. metric-only tensor-free certified commit (no raster allocation);
25. tensor/memmap sinks reject a verification-only skip;
plus the exact-fill synthesis path for tensor-materializing sinks and the
legacy-sink fallback in the pipeline.
"""

import sys
from pathlib import Path

import pytest
import torch

from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import (
    CertifiedCommitNotRepresentableError,
    MemmapTileSink,
    MetricOnlyTileSink,
    TensorTileSink,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_screening_work_accounting_inc28 import (  # noqa: E402  # noqa: E402
    CounterForward,
    ExactEmptyContextScreeningPolicy,
    SparseExactSource,
)


def _screening() -> ExactEmptyContextScreeningPolicy:
    return ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c2",
    )


def test_metric_only_certified_commit_is_tensor_free():
    source = SparseExactSource()
    sink = MetricOnlyTileSink(source.shape)
    forward = CounterForward()
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=_screening(),
    )
    assert report.n_tiles == 2
    assert sink.certified_records == {"tile_1": "EXACT_OUTPUT_SKIP"}
    assert sink.certified_pixels == 64
    assert sink.covered_pixels == 128
    meta = sink.metadata["tile_1"]
    assert meta["certified_exact_fill"] == 0.0
    assert meta["screen_disposition"] == "EXACT_OUTPUT_SKIP"


def test_tensor_sink_synthesizes_exact_fill():
    source = SparseExactSource()
    sink = TensorTileSink(source.shape, dtype=torch.float32)
    forward = CounterForward()
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=_screening(),
    )
    assert report.n_tiles == 2
    output = sink.finalize()
    # the screened core is exactly filled; the active core comes from the
    # identity forward of the mask tensor
    assert float(output[0:8, 8:16].min()) == 0.0
    assert float(output[0:8, 8:16].max()) == 0.0
    assert float(output[2, 1]) == 1.0


def test_tensor_sink_rejects_verification_only_skip():
    sink = TensorTileSink((8, 8))
    from openlithohub.streaming.geometry import BoundingBox

    with pytest.raises(CertifiedCommitNotRepresentableError, match="verification-only"):
        sink.record_certified_core(
            "t",
            BoundingBox(0, 0, 4, 4),
            exact_fill=None,
            metadata={},
        )


def test_memmap_sink_rejects_verification_only_skip(tmp_path):
    sink = MemmapTileSink((8, 8), tmp_path / "out.f32")
    from openlithohub.streaming.geometry import BoundingBox

    with pytest.raises(CertifiedCommitNotRepresentableError):
        sink.record_certified_core("t", BoundingBox(0, 0, 4, 4), exact_fill=None, metadata={})
    assert float(sink.finalize()[0:4, 0:4].max()) == 0.0


def test_legacy_sink_without_capability_falls_back_to_write_core():
    class LegacySink:
        def __init__(self) -> None:
            self.shape = (16, 16)
            self.written: list[str] = []

        def write_core(self, tile_id, bbox, tensor, metadata=None):
            self.written.append(tile_id)

        def finalize(self):
            return None

    source = SparseExactSource()
    sink = LegacySink()
    forward = CounterForward()
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=_screening(),
    )
    assert report.n_tiles == 2
    assert sink.written == ["tile_0", "tile_1"]
