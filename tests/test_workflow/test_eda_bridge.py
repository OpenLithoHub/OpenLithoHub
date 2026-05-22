"""Tests for openlithohub.workflow.eda_bridge template emitters."""

from __future__ import annotations

from openlithohub.workflow import (
    BridgeRules,
    emit_bridge_bundle,
    emit_calibre_svrf,
    emit_icv_runset,
)


def test_emit_calibre_svrf_writes_file(tmp_path):
    oasis = tmp_path / "mask.oas"
    oasis.write_bytes(b"")  # placeholder, content irrelevant

    rules = BridgeRules(min_width_nm=40.0, min_spacing_nm=40.0)
    out = emit_calibre_svrf(oasis, rules, cell_name="TOP")

    assert out.exists()
    text = out.read_text()
    assert "LAYOUT PATH" in text
    # Comment carries the human-readable nm value; the rule literal is in
    # microns (issue #50: SVRF interprets bare numerics as user units, so
    # `< 40` for a 40-nm rule would mean 40 µm).
    assert "is 40.0 nm" in text
    assert "< 0.04 ABUT" in text
    assert "TOP" in text


def test_emit_icv_runset_writes_file(tmp_path):
    oasis = tmp_path / "mask.oas"
    oasis.write_bytes(b"")

    rules = BridgeRules(min_width_nm=32.0, min_spacing_nm=36.0, layer=2, datatype=0)
    out = emit_icv_runset(oasis, rules, cell_name="DESIGN")

    assert out.exists()
    text = out.read_text()
    assert 'layout("' in text
    # Issue #50: literals must be in microns to match ICV's user-unit
    # convention. 32 nm -> 0.032 µm, 36 nm -> 0.036 µm.
    assert "< 0.032)" in text
    assert "< 0.036)" in text
    assert "{ 2, 0 }" in text


def test_emit_calibre_svrf_emits_microns_not_nm_for_threshold(tmp_path):
    """Issue #50: a 40-nm rule must be emitted as `< 0.04` (microns), not
    `< 40` (which SVRF would interpret as 40 µm = 1000× too lax)."""
    oasis = tmp_path / "mask.oas"
    oasis.write_bytes(b"")
    rules = BridgeRules(min_width_nm=40.0, min_spacing_nm=40.0)
    out = emit_calibre_svrf(oasis, rules, cell_name="TOP")
    text = out.read_text()
    # Bare `< 40` (without decimal point) would be the bug.
    assert "< 40 ABUT" not in text
    assert "< 40 SINGULAR" not in text
    # Microns form is present and correct.
    assert "< 0.04 ABUT" in text


def test_emit_bridge_bundle_creates_three_files(tmp_path):
    oasis = tmp_path / "mask.oas"
    oasis.write_bytes(b"")

    rules = BridgeRules(min_width_nm=40.0, min_spacing_nm=40.0)
    bundle = emit_bridge_bundle(oasis, rules)

    assert set(bundle.keys()) == {"svrf", "icv", "readme"}
    for path in bundle.values():
        assert path.exists()
        assert path.read_text().strip()

    readme = bundle["readme"].read_text()
    assert "Calibre" in readme
    assert "icv" in readme
