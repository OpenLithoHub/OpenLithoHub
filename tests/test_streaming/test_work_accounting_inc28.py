"""Inc28 WorkAccounting semantics under the R17 C4b dual ledger.

The canonical unique-forward ledger is region-keyed
(``record_forward_core``); ``active_pixels``/``screened_out_pixels`` are
compatibility aliases projecting ``unique_forward_pixels`` /
``pre_forward_avoided_pixels``.  The Inc28 values are preserved.
"""

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.work_accounting import WorkAccounting


def test_active_area_is_unique_across_refinement_attempts():
    w = WorkAccounting(full_chip_pixels=100)
    box = BoundingBox(0, 0, 5, 10)
    w.record_active_core("tile", box)
    w.record_active_core("tile", box)
    w.record_read_window(60)
    w.record_forward(60)
    w.record_forward_core(box)
    w.record_refinement("tile", "tile", 50)
    w.record_read_window(80)
    w.record_forward(80)
    w.record_forward_core(box)  # same physical area re-forwarded
    assert w.active_pixels == 50
    assert w.unique_forward_pixels == 50
    assert w.forward_simulator_calls == 2
    assert w.forward_simulator_input_pixels == 140
    assert w.tiles_refined == 1
    assert w.reused_work_units == 50


def test_screened_and_active_partition_reports_avoidance():
    w = WorkAccounting(full_chip_pixels=128)
    w.record_active_core("a", BoundingBox(0, 0, 8, 8))
    w.record_forward_core(BoundingBox(0, 0, 8, 8))
    w.record_screened_out("b", 64)
    s = w.summary()
    assert s["active_pixels"] == 64
    assert s["unique_forward_pixels"] == 64
    assert s["screened_out_pixels"] == 64
    assert s["pre_forward_avoided_pixels"] == 64
    assert s["avoided_pct"] == 50.0
    assert s["accounted_pct"] == 100.0


def test_forwarded_parent_area_is_not_double_counted_by_children():
    w = WorkAccounting(full_chip_pixels=100)
    parent = BoundingBox(0, 0, 10, 10)
    w.record_forward_core(parent)
    # parent subdivides into four 5x5 children; one child re-forwards
    w.record_forward_core(BoundingBox(0, 0, 5, 5))
    assert w.unique_forward_pixels == 100
    assert w.pre_forward_avoided_pixels == 0
