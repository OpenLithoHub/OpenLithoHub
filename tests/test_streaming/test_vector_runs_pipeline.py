from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.vector_runs import ExactVectorRunSource, VectorCell, rectangle
from openlithohub.verify.run_ownership import RunOwnershipVerifier


def test_run_streaming_passes_source_ownership_metadata_to_verifier():
    src = ExactVectorRunSource(
        shape=(16, 24),
        cells={
            "TOP": VectorCell(
                "TOP",
                polygons=(
                    rectangle("long", 2, 4, 22, 8),
                    rectangle("edge", 0, 10, 3, 14),
                ),
            )
        },
        top="TOP",
        pixel_size_nm=8.0,
    )
    sink = MetricOnlyTileSink(src.shape)
    verifier = RunOwnershipVerifier()
    report = run_streaming(
        src,
        sink,
        lambda x: x,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=(verifier,),
        pixel_nm=8.0,
        max_requeues=0,
    )
    assert report.n_tiles == 6
    assert sink.covered_pixels == 16 * 24
    assert report.verification.status == "PASS"
    assert report.verification.metrics["max_owned_unique_runs"] >= 1.0
