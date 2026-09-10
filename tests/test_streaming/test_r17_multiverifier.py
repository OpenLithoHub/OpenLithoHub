"""R17 C1a — per-verifier reducer sessions and the report facade.

Covers the required matrix items:
1. single-verifier backward ``report.verification`` facade;
2. multi-verifier canonical ``verification_results``;
3. multi-verifier summary carries no generic numeric upper bound;
4. cross-verifier same tile-id does not collide;
5. the legacy batch ``reduce(results)`` is not invoked by the pipeline;
6. optional ``make_reducer`` capability is honored per session.
"""

from dataclasses import dataclass, field

import pytest
import torch

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.verification import (
    MultiVerifierSummary,
    StreamingVerificationReducer,
    TileContext,
    TileVerificationResult,
    VerificationContext,
    VerifierIdentity,
    fold_global,
)


@dataclass
class ScriptedVerifier:
    """Minimal verifier returning scripted per-tile-id statuses."""

    name: str
    statuses: dict[str, str] = field(default_factory=dict)
    default_status: str = "PASS"
    reduce_calls: int = 0
    make_reducer_calls: int = 0

    @property
    def version(self) -> str:
        return "1.0"

    def required_halo(self, context: VerificationContext):
        return None

    def prepare(self, context: VerificationContext) -> None:
        return None

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        status = self.statuses.get(tile.tile_id, self.default_status)
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status=status,
            upper_bound={"PASS": 0.5, "FAIL": 9.9, "INCONCLUSIVE": None}[status],
        )

    def reduce(self, results):
        self.reduce_calls += 1
        return None

    def finalize(self):
        return None


def _source() -> TensorTileSource:
    return TensorTileSource(torch.zeros((16, 16), dtype=torch.float32))


def _run(verifiers):
    sink = MetricOnlyTileSink((16, 16))
    return run_streaming(
        _source(),
        sink,
        lambda tile: tile,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=verifiers,
    )


def test_single_verifier_facade_preserved():
    report = _run([ScriptedVerifier(name="solo")])
    from openlithohub.streaming.verification import GlobalVerificationResult

    assert isinstance(report.verification, GlobalVerificationResult)
    assert report.verification.status == "PASS"
    assert report.verification.n_tiles == 4
    assert len(report.verification_results) == 1
    identity = next(iter(report.verification_results))
    assert identity == VerifierIdentity(name="solo", version="1.0")


def test_multi_verifier_canonical_results_and_facade():
    v1 = ScriptedVerifier(name="alpha")
    v2 = ScriptedVerifier(name="beta", statuses={"tile_0": "FAIL"})
    report = _run([v1, v2])
    assert set(report.verification_results) == {
        VerifierIdentity(name="alpha", version="1.0"),
        VerifierIdentity(name="beta", version="1.0"),
    }
    assert report.verification_results[VerifierIdentity("beta", "1.0")].status == "FAIL"
    summary = report.verification
    assert isinstance(summary, MultiVerifierSummary)
    assert summary.status == "FAIL"
    assert summary.n_verifiers == 2
    assert summary.n_fail == 1
    assert summary.n_pass == 1
    assert not hasattr(summary, "worst_upper_bound")


def test_inconclusive_summary_without_fail_is_inconclusive():
    v1 = ScriptedVerifier(name="alpha")
    v2 = ScriptedVerifier(name="beta", statuses={"tile_2": "INCONCLUSIVE"})
    report = _run([v1, v2])
    summary = report.verification
    assert isinstance(summary, MultiVerifierSummary)
    assert summary.status == "INCONCLUSIVE"
    assert summary.n_inconclusive == 1


def test_cross_verifier_same_tile_id_does_not_collide():
    # FAIL vs PASS on the SAME tile ids: a shared reducer would let the
    # PASS clobber the FAIL (latest-wins).  Per-verifier sessions must not.
    v1 = ScriptedVerifier(name="strict", default_status="FAIL")
    v2 = ScriptedVerifier(name="lenient", default_status="PASS")
    report = _run([v1, v2])
    strict = report.verification_results[VerifierIdentity("strict", "1.0")]
    lenient = report.verification_results[VerifierIdentity("lenient", "1.0")]
    assert strict.status == "FAIL"
    assert strict.n_fail == 4
    assert lenient.status == "PASS"
    assert lenient.n_pass == 4
    assert report.verification.status == "FAIL"


def test_streaming_pipeline_never_calls_legacy_batch_reduce():
    v1 = ScriptedVerifier(name="alpha")
    _run([v1])
    assert v1.reduce_calls == 0


def test_optional_make_reducer_is_honored():
    class CustomReducerVerifier(ScriptedVerifier):
        instance_key = "custom"

        def make_reducer(self, context, identity):
            self.make_reducer_calls += 1
            return StreamingVerificationReducer()

    v = CustomReducerVerifier(name="custom")
    report = _run([v])
    assert v.make_reducer_calls == 1
    identity = next(iter(report.verification_results))
    assert identity.instance_key == "custom"


def test_make_reducer_must_return_stock_reducer_type():
    class BadReducerVerifier(ScriptedVerifier):
        def make_reducer(self, context, identity):
            return object()

    with pytest.raises(TypeError, match="StreamingVerificationReducer"):
        _run([BadReducerVerifier(name="bad")])


def test_reducer_retire_removes_tile_from_finalization():
    reducer = StreamingVerificationReducer()
    result = TileVerificationResult(
        tile_id="tile_0",
        core_bbox=BoundingBox(0, 0, 1, 1),
        status="INCONCLUSIVE",
    )
    reducer.add(result)
    assert reducer.n_tiles == 1
    reducer.retire("tile_0")
    assert reducer.n_tiles == 0
    final = reducer.finalize()
    assert final.n_tiles == 0
    assert final.status == "INCONCLUSIVE"


def test_fold_global_firewall():
    identity = VerifierIdentity("v", "1")
    from openlithohub.streaming.verification import GlobalVerificationResult

    passed = GlobalVerificationResult(
        status="PASS",
        worst_upper_bound=0.0,
        worst_lower_bound=None,
        n_tiles=1,
        n_pass=1,
        n_fail=0,
        n_inconclusive=0,
        coverage="NOT_APPLICABLE",
        halo_error_budget=0.0,
        error_budget={},
    )
    failed = GlobalVerificationResult(
        status="FAIL",
        worst_upper_bound=None,
        worst_lower_bound=None,
        n_tiles=1,
        n_pass=0,
        n_fail=1,
        n_inconclusive=0,
        coverage="NOT_APPLICABLE",
        halo_error_budget=0.0,
        error_budget={},
    )
    inconclusive = GlobalVerificationResult(
        status="INCONCLUSIVE",
        worst_upper_bound=None,
        worst_lower_bound=None,
        n_tiles=1,
        n_pass=0,
        n_fail=0,
        n_inconclusive=1,
        coverage="NOT_APPLICABLE",
        halo_error_budget=0.0,
        error_budget={},
    )
    assert fold_global({identity: passed}).status == "PASS"
    assert fold_global({identity: inconclusive}).status == "INCONCLUSIVE"
    assert (
        fold_global({identity: passed, VerifierIdentity("w", "1"): inconclusive}).status
        == "INCONCLUSIVE"
    )
    assert fold_global({identity: passed, VerifierIdentity("w", "1"): failed}).status == "FAIL"
