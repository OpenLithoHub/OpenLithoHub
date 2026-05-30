"""Tests for examples/eda_bridge_demo.py — validates the full end-to-end chain."""

from __future__ import annotations

from pathlib import Path

import torch

from examples.eda_bridge_demo import (
    cross_check_tachyon,
    export_to_eda,
    main,
    make_test_mask,
    run_co_design,
)


class TestMakeTestMask:
    def test_shape(self) -> None:
        mask = make_test_mask()
        assert mask.shape == (64, 64)

    def test_has_centered_rectangle(self) -> None:
        mask = make_test_mask()
        center = mask.shape[0] // 2
        assert mask[center, center] == 1.0
        assert mask[0, 0] == 0.0


class TestRunCoDesign:
    def test_returns_clamped_mask(self) -> None:
        mask = make_test_mask()
        target = make_test_mask()
        result = run_co_design(mask, target, steps=3)
        assert result.shape == mask.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_produces_nontrivial_output(self) -> None:
        mask = make_test_mask()
        target = make_test_mask()
        result = run_co_design(mask, target, steps=2)
        assert result.sum().item() > 0


class TestCrossCheckTachyon:
    def test_returns_expected_keys(self) -> None:
        mask = make_test_mask()
        target = make_test_mask()
        info = cross_check_tachyon(mask, target)
        assert "preflight_ok" in info
        assert "aerial_shape" in info
        assert "mse_vs_target" in info
        assert "mock" in info

    def test_preflight_passes(self) -> None:
        info = cross_check_tachyon(make_test_mask(), make_test_mask())
        assert info["preflight_ok"] is True

    def test_mock_mode_active(self) -> None:
        info = cross_check_tachyon(make_test_mask(), make_test_mask())
        assert info["mock"] is True

    def test_aerial_shape_matches_mask(self) -> None:
        mask = make_test_mask()
        info = cross_check_tachyon(mask, make_test_mask())
        assert info["aerial_shape"] == (64, 64)

    def test_aerial_in_valid_range(self) -> None:
        info = cross_check_tachyon(make_test_mask(), make_test_mask())
        lo, hi = info["aerial_range"]
        assert lo >= 0.0
        assert hi <= 1.0


class TestExportToEda:
    def test_creates_all_files(self, tmp_path: Path) -> None:
        mask = make_test_mask()
        files = export_to_eda(mask, tmp_path)
        assert set(files.keys()) == {"mask", "svrf", "icv", "readme"}
        for path in files.values():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_mask_file_contains_data(self, tmp_path: Path) -> None:
        mask = make_test_mask()
        files = export_to_eda(mask, tmp_path)
        content = Path(files["mask"]).read_text()
        assert "OpenLithoHub mask export" in content

    def test_svrf_has_correct_rules(self, tmp_path: Path) -> None:
        mask = make_test_mask()
        files = export_to_eda(mask, tmp_path, min_width_nm=32.0, min_spacing_nm=48.0)
        svrf = Path(files["svrf"]).read_text()
        assert "LAYOUT PATH" in svrf
        assert "is 32.0 nm" in svrf
        assert "< 0.032 ABUT" in svrf
        assert "< 0.048 ABUT" in svrf

    def test_custom_cell_name(self, tmp_path: Path) -> None:
        mask = make_test_mask()
        files = export_to_eda(mask, tmp_path, cell_name="MY_CELL")
        svrf = Path(files["svrf"]).read_text()
        assert "MY_CELL" in svrf


class TestMain:
    def test_runs_end_to_end(self, tmp_path: Path) -> None:
        result = main(output_dir=str(tmp_path))
        assert "tachyon_info" in result
        assert "files" in result
        assert "output_dir" in result

    def test_all_outputs_valid(self, tmp_path: Path) -> None:
        result = main(output_dir=str(tmp_path))
        tachyon = result["tachyon_info"]
        assert tachyon["preflight_ok"] is True
        assert tachyon["aerial_shape"] == (64, 64)
        for path in result["files"].values():
            assert Path(path).exists()

    def test_non_empty_aerial(self, tmp_path: Path) -> None:
        result = main(output_dir=str(tmp_path))
        lo, hi = result["tachyon_info"]["aerial_range"]
        assert hi > 0.0, "Aerial image should not be all zeros"
