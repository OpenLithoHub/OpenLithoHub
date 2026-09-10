from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.work_accounting import WorkAccounting


def test_active_area_is_unique_across_refinement_attempts():
    w = WorkAccounting(full_chip_pixels=100)
    box = BoundingBox(0, 0, 5, 10)
    w.record_active_core("tile", box)
    w.record_active_core("tile", box)
    w.record_read_window(60)
    w.record_forward(60)
    w.record_refinement("tile", "tile", 50)
    w.record_read_window(80)
    w.record_forward(80)
    assert w.active_pixels == 50
    assert w.forward_simulator_calls == 2
    assert w.forward_simulator_input_pixels == 140
    assert w.tiles_refined == 1
    assert w.reused_work_units == 50


def test_screened_and_active_partition_reports_avoidance():
    w = WorkAccounting(full_chip_pixels=128)
    w.record_active_core("a", BoundingBox(0, 0, 8, 8))
    w.record_screened_out("b", 64)
    s = w.summary()
    assert s["active_pixels"] == 64
    assert s["screened_out_pixels"] == 64
    assert s["avoided_pct"] == 50.0
    assert s["accounted_pct"] == 100.0
