"""R17 C1b — typed screen proof facts replace blanket verifier authority.

Covers the required matrix items:
19. legacy exact-empty adapter (SCREENED_OUT + certified fill adapts to
    EXACT_OUTPUT_SKIP facts);
20. deprecated blanket ``certifies_verifiers`` authority is recorded but
    never honored;
21. invalid legacy screens fail closed;
23. all attached verifiers must accept the facts for a verification skip;
    a single refusal sends the tile down the ordinary active path.
"""

import sys
from pathlib import Path

import torch

from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.screening import (
    ExactEmptyContextScreeningPolicy,
    ExactOutputFact,
    ScreenProofFact,
    screen_decision_to_facts,
)
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.verification import (
    TileContext,
    TileVerificationResult,
    VerificationContext,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_screening_work_accounting_inc28 import (  # noqa: E402
    CounterForward,
    SparseExactSource,
)


class FactAcceptingVerifier:
    """Accepts EXACT_OUTPUT_SKIP facts only when told to."""

    name = "fact-acceptor"
    version = "1.0"

    def __init__(self, *, accept: bool = True, refuse_after: int | None = None):
        self.accept = accept
        self.refuse_after = refuse_after
        self.accepted: list[str] = []

    def required_halo(self, context: VerificationContext):
        return None

    def prepare(self, context: VerificationContext) -> None:
        return None

    def accept_screen_facts(
        self, tile: TileContext, facts: list[ScreenProofFact]
    ) -> TileVerificationResult | None:
        kinds = {fact.kind for fact in facts}
        if not self.accept or "EXACT_OUTPUT_SKIP" not in kinds:
            return None
        if self.refuse_after is not None and len(self.accepted) >= self.refuse_after:
            return None
        self.accepted.append(tile.tile_id)
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status="PASS",
            certificate_ref=next(f.certificate_ref for f in facts if f.kind == "EXACT_OUTPUT_SKIP"),
        )

    def verify_tile(self, tile: TileContext) -> TileVerificationResult:
        return TileVerificationResult(
            tile_id=tile.tile_id,
            core_bbox=tile.core_bbox,
            status="PASS",
            upper_bound=0.0,
        )

    def reduce(self, results):
        return None

    def finalize(self):
        return None


def _run(source, verifiers=(), sink=None, screening=None):
    sink = sink or MetricOnlyTileSink(source.shape)
    forward = CounterForward()
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=verifiers,
        screening_policy=screening,
    )
    return report, sink, forward


def test_legacy_exact_empty_adapter_produces_exact_output_fact():
    source = SparseExactSource()
    screening = ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c1b",
    )
    from openlithohub.streaming.core_halo import TileRequest
    from openlithohub.streaming.geometry import BoundingBox, HaloSpec

    request = TileRequest(
        tile_id="tile_probe",
        core_bbox=BoundingBox(8, 0, 16, 8),
        read_bbox=BoundingBox(8, 0, 16, 8),
        halo=HaloSpec.uniform(0),
    )
    decision = screening.screen(source, request)
    assert decision.status == "SCREENED_OUT"
    exact, facts, deprecated = screen_decision_to_facts(decision)
    assert isinstance(exact, ExactOutputFact)
    assert exact.fill_value == 0.0
    assert facts[0].kind == "EXACT_OUTPUT_SKIP"
    assert deprecated is False


def test_zero_verifier_exact_skip_unchanged():
    source = SparseExactSource()
    screening = ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c1b",
    )
    report, sink, forward = _run(source, screening=screening)
    assert report.n_tiles == 2
    assert source.read_calls == 1
    assert forward.calls == 1
    assert sink.metadata["tile_1"]["screen_disposition"] == "EXACT_OUTPUT_SKIP"


def test_all_verifiers_accepting_facts_skip_forward():
    source = SparseExactSource()
    screening = ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c1b",
    )
    v = FactAcceptingVerifier(accept=True)
    report, sink, forward = _run(source, verifiers=[v], screening=screening)
    assert forward.calls == 1  # only the active tile forwards
    assert v.accepted == ["tile_1"]
    assert sink.metadata["tile_1"]["screen_disposition"] == "EXACT_OUTPUT_SKIP"
    # the accepted verdict (not a fabricated one) is in the verifier session
    assert report.verification.n_tiles == 2
    assert report.verification.n_pass == 2


def test_single_verifier_refusal_forces_active_path():
    source = SparseExactSource()
    screening = ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c1b",
    )
    v = FactAcceptingVerifier(accept=False)
    report, sink, forward = _run(source, verifiers=[v], screening=screening)
    # the geometrically-empty tile is NOT skipped: the verifier did not
    # discharge the facts, so the ordinary active path runs for both tiles.
    assert forward.calls == 2
    assert source.read_calls == 2
    assert v.accepted == []
    rejected = sink.metadata["tile_1"]
    assert rejected["screening_status"] == "SCREENED_OUT_NOT_DISCHARGED"


def test_deprecated_blanket_authority_recorded_but_not_honored():
    source = SparseExactSource()
    screening = ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance="c1b",
        certifies_verifiers=True,
        verification_upper_bound=0.25,
    )
    # legacy verifier WITHOUT accept_screen_facts: the blanket flag must not
    # authorize the skip
    from test_r17_multiverifier import ScriptedVerifier

    v = ScriptedVerifier(name="legacy")
    _report, sink, forward = _run(source, verifiers=[v], screening=screening)
    assert forward.calls == 2
    assert sink.metadata["tile_1"]["screen_authority"] == (
        "deprecated-blanket-input-recorded-not-honored"
    )

    # zero verifiers: the skip happens, and the deprecated input is still
    # only recorded as provenance
    source2 = SparseExactSource()
    report2, sink2, forward2 = _run(source2, screening=screening)
    assert forward2.calls == 1
    assert sink2.metadata["tile_1"]["screen_authority"] == (
        "deprecated-blanket-input-recorded-not-honored"
    )
    assert report2.n_tiles == 2


def test_invalid_legacy_screen_fails_closed():
    source = TensorTileSource(torch.zeros((8, 8), dtype=torch.float32))

    class UncertifiedPolicy:
        name = "broken"

        def screen(self, src, request):
            from openlithohub.streaming.screening import TileScreenDecision

            return TileScreenDecision(
                status="SCREENED_OUT",
                reason="claims a skip without certification",
                certified=False,
                fill_value=None,
            )

    import pytest

    with pytest.raises(ValueError, match="SCREENED_OUT requires a certified decision"):
        _run(source, screening=UncertifiedPolicy())
