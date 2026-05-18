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
    assert "40.0" in text
    assert "TOP" in text


def test_emit_icv_runset_writes_file(tmp_path):
    oasis = tmp_path / "mask.oas"
    oasis.write_bytes(b"")

    rules = BridgeRules(min_width_nm=32.0, min_spacing_nm=36.0, layer=2, datatype=0)
    out = emit_icv_runset(oasis, rules, cell_name="DESIGN")

    assert out.exists()
    text = out.read_text()
    assert 'layout("' in text
    assert "32.0" in text
    assert "36.0" in text
    assert "{ 2, 0 }" in text


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
