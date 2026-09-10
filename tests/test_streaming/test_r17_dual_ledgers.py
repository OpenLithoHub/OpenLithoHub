"""R17 C4 — dual ledgers: terminal coverage vs historical work.

The two ledgers answer different questions and may disagree (matrix item
18); each must partition the chip exactly (items 16, 17).

Guide §21 minimal witnesses:

- Case A — forward then subdivide: parent area 100 forwarded, child1 (40)
  verification-only, child2 (60) active  =>
      terminal_verify=40, terminal_active=60, unique_forward=100, avoided=0
- Case B — subdivide before forward: child1 (40) verification-only never
  forwarded, child2 (60) forwarded  =>
      terminal_verify=40, terminal_active=60, unique_forward=60, avoided=40
"""

import pytest

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.ownership import (
    TERMINAL_ACTIVE,
    TERMINAL_EXACT_SKIP,
    TERMINAL_VERIFY_SKIP,
    OwnershipTree,
)
from openlithohub.streaming.work_accounting import WorkAccounting

_CHILD_40 = BoundingBox(0, 0, 10, 4)
_CHILD_60 = BoundingBox(0, 4, 10, 10)
_PARENT = BoundingBox(0, 0, 10, 10)


def test_case_a_forward_then_subdivide_ledgers_disagree():
    tree = OwnershipTree()
    tree.add_root("p", _PARENT)
    tree.mark_forward("p")
    children = [
        ("p#c1", _CHILD_40),
        ("p#c2", _CHILD_60),
    ]
    tree.subdivide("p", children)
    # child1 discharged by verification only (no exact tensor known)
    tree.set_terminal("p#c1", TERMINAL_VERIFY_SKIP)
    # child2 stays active
    tree.set_terminal("p#c2", TERMINAL_ACTIVE)
    terminal = tree.verify_total_coverage(100)
    assert terminal[TERMINAL_VERIFY_SKIP] == 40
    assert terminal[TERMINAL_ACTIVE] == 60
    assert terminal[TERMINAL_EXACT_SKIP] == 0
    assert sum(terminal.values()) == 100

    w = WorkAccounting(full_chip_pixels=100)
    w.record_forward_core(_PARENT)  # the parent's forward is history
    w.record_forward_core(_CHILD_60)  # child2 re-forward (same area union)
    assert w.unique_forward_pixels == 100
    assert w.pre_forward_avoided_pixels == 0
    assert w.active_pixels == 100 and w.screened_out_pixels == 0


def test_case_b_subdivide_before_forward():
    tree = OwnershipTree()
    tree.add_root("p", _PARENT)  # never forwarded
    tree.subdivide("p", [("p#c1", _CHILD_40), ("p#c2", _CHILD_60)])
    tree.set_terminal("p#c1", TERMINAL_VERIFY_SKIP)
    tree.set_terminal("p#c2", TERMINAL_ACTIVE)
    terminal = tree.verify_total_coverage(100)
    assert terminal[TERMINAL_VERIFY_SKIP] == 40
    assert terminal[TERMINAL_ACTIVE] == 60

    w = WorkAccounting(full_chip_pixels=100)
    w.record_forward_core(_CHILD_60)  # only child2 forwarded
    assert w.unique_forward_pixels == 60
    assert w.pre_forward_avoided_pixels == 40


def test_screened_child_of_forwarded_parent_is_unique_forward():
    # a screened child inherits its ancestor's forward history: it is NOT
    # pre-forward avoided work
    w = WorkAccounting(full_chip_pixels=100)
    w.record_forward_core(_PARENT)
    assert w.unique_forward_pixels == 100
    assert w.pre_forward_avoided_pixels == 0


def test_pipeline_terminal_ledger_partition_exact_empty_case():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_screening_work_accounting_inc28 import (
        CounterForward,
        ExactEmptyContextScreeningPolicy,
        SparseExactSource,
    )

    from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
    from openlithohub.streaming.pipeline import run_streaming
    from openlithohub.streaming.sinks import MetricOnlyTileSink
    from openlithohub.streaming.work_accounting import WorkAccounting

    source = SparseExactSource()
    report = run_streaming(
        source,
        MetricOnlyTileSink(source.shape),
        CounterForward(),
        core_size=8,
        halo_policy=LegacyFixedHaloPolicy(0),
        screening_policy=ExactEmptyContextScreeningPolicy(
            context_certified=True,
            zero_response_certified=True,
            fill_value=0.0,
            model_provenance="c4",
        ),
        work_accounting=WorkAccounting(),
    )
    work = report.work_accounting
    assert work["terminal_active_pixels"] == 64
    assert work["terminal_exact_skip_pixels"] == 64
    assert work["terminal_verification_skip_pixels"] == 0
    assert (
        work["terminal_active_pixels"]
        + work["terminal_exact_skip_pixels"]
        + work["terminal_verification_skip_pixels"]
        == work["full_chip_pixels"]
    )
    # both ledgers partition the chip, and here they agree
    assert work["unique_forward_pixels"] + work["pre_forward_avoided_pixels"] == 128


def test_unterminated_final_leaf_fails_closed():
    tree = OwnershipTree()
    tree.add_root("t", _PARENT)
    with pytest.raises(ValueError, match="no terminal disposition"):
        tree.verify_total_coverage(100)


def test_double_terminal_disposition_fails_closed():
    tree = OwnershipTree()
    tree.add_root("t", _PARENT)
    tree.set_terminal("t", TERMINAL_ACTIVE)
    with pytest.raises(ValueError, match="already has terminal disposition"):
        tree.set_terminal("t", TERMINAL_EXACT_SKIP)


def test_unknown_disposition_fails_closed():
    tree = OwnershipTree()
    tree.add_root("t", _PARENT)
    with pytest.raises(ValueError, match="unknown terminal disposition"):
        tree.set_terminal("t", "MAGIC")
