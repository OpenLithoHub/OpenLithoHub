from openlithohub.verify.replay_contract import (
    HaloDecisionProvenance,
    Interval1D,
    one_way_reference_enclosed,
)


def test_one_way_enclosure_accepts_reference_plus_certified_bridge():
    assert one_way_reference_enclosed(
        certified=Interval1D(0.9, 1.2),
        reference=Interval1D(1.0, 1.1),
        certified_bridge_error=0.05,
    )


def test_one_way_enclosure_rejects_bridge_escape():
    assert not one_way_reference_enclosed(
        certified=Interval1D(0.99, 1.11),
        reference=Interval1D(1.0, 1.1),
        certified_bridge_error=0.02,
    )


def test_halo_requires_explicit_certified_tail_to_be_theorem_ready():
    open_halo = HaloDecisionProvenance(
        core_size_px=64,
        halo_px=32,
        selection_method="heuristic-effective-radius",
        optical_tail_upper=None,
        required_error_budget=1e-3,
        kernel_snapshot_hash="k",
        source_snapshot_hash="s",
        boundary_model="FINITE_PHYSICAL_CONTEXT",
    )
    assert not open_halo.theorem_ready

    closed = HaloDecisionProvenance(
        core_size_px=64,
        halo_px=36,
        selection_method="certified-SOCS-tail",
        optical_tail_upper=8e-4,
        required_error_budget=1e-3,
        kernel_snapshot_hash="k",
        source_snapshot_hash="s",
        boundary_model="FINITE_PHYSICAL_CONTEXT",
    )
    assert closed.theorem_ready
