"""Tests for openlithohub.models.rule_based_opc."""

import pytest
import torch

from openlithohub.models.base import PredictionResult
from openlithohub.models.registry import registry
from openlithohub.models.rule_based_opc import RuleBasedOPCModel


class TestRuleBasedOPCModel:
    def test_registered_in_registry(self) -> None:
        model = registry.get("rule-based-opc")
        assert isinstance(model, RuleBasedOPCModel)

    def test_properties(self) -> None:
        model = RuleBasedOPCModel()
        assert model.name == "rule-based-opc"
        assert model.supports_curvilinear is False

    def test_predict_returns_prediction_result(self) -> None:
        model = RuleBasedOPCModel()
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape

    def test_mask_is_binary(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=2)
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0
        result = model.predict(design)
        unique_vals = result.mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_dilation_grows_foreground(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=2, line_end_extra_px=0)
        design = torch.zeros(32, 32)
        design[14:18, 14:18] = 1.0
        result = model.predict(design)
        assert result.mask.sum() > design.sum()

    def test_zero_radius_preserves_design(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=0, line_end_extra_px=0)
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0
        result = model.predict(design)
        assert torch.equal(result.mask, design)

    def test_kwargs_override_constructor(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=5)
        design = torch.zeros(32, 32)
        design[14:18, 14:18] = 1.0
        result = model.predict(design, bias_radius_px=0, line_end_extra_px=0)
        assert torch.equal(result.mask, design)
        assert result.metadata["bias_radius_px"] == 0

    def test_line_end_bias_extends_tip(self) -> None:
        """A short horizontal line gets its right tip extended."""
        model = RuleBasedOPCModel(bias_radius_px=0, line_end_extra_px=1)
        design = torch.zeros(16, 16)
        design[8, 4:9] = 1.0
        result = model.predict(design)
        assert result.mask.sum() > design.sum()

    def test_empty_design_stays_empty(self) -> None:
        model = RuleBasedOPCModel()
        design = torch.zeros(16, 16)
        result = model.predict(design)
        assert result.mask.sum().item() == 0.0

    def test_full_design_stays_full(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=3)
        design = torch.ones(16, 16)
        result = model.predict(design)
        assert torch.equal(result.mask, design)

    def test_metadata_keys(self) -> None:
        model = RuleBasedOPCModel()
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        assert "bias_radius_px" in result.metadata
        assert "line_end_extra_px" in result.metadata

    def test_handles_extra_dims(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=1)
        design = torch.zeros(1, 1, 16, 16)
        design[..., 4:12, 4:12] = 1.0
        result = model.predict(design)
        assert result.mask.ndim == 2

    def test_directional_line_end_extends_only_along_axis(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=0, line_end_extra_px=2, directional_line_end=True)
        design = torch.zeros(16, 16)
        design[8, 4:9] = 1.0
        result = model.predict(design)
        # Right tip at (8, 8) should extend rightward to (8, 10) but NOT
        # bleed onto rows 7 or 9.
        assert result.mask[8, 10].item() == 1.0
        assert result.mask[7, 10].item() == 0.0
        assert result.mask[9, 10].item() == 0.0
        assert result.mask[7, 8].item() == 0.0
        assert result.mask[9, 8].item() == 0.0

    def test_directional_line_end_disabled_uses_isotropic(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=0, line_end_extra_px=1, directional_line_end=False)
        design = torch.zeros(16, 16)
        design[8, 4:9] = 1.0
        result = model.predict(design)
        # Isotropic dilation puts pixels above/below the right tip too.
        assert result.mask[7, 8].item() == 1.0
        assert result.mask[9, 8].item() == 1.0

    def test_inner_corner_serif_fills_concave_notch(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=0, line_end_extra_px=0, inner_corner_extra_px=1)
        design = torch.zeros(16, 16)
        # L-shape: vertical bar + horizontal bar meeting at (8,8)
        design[4:9, 8] = 1.0
        design[8, 8:13] = 1.0
        result = model.predict(design)
        # Concave notch sits at (7, 9) — bg pixel adjacent to fg below (the
        # horizontal bar) and fg to its left (the vertical bar).
        assert result.mask[7, 9].item() == 1.0
        assert result.metadata["n_inner_corners"] >= 1

    def test_inner_corner_serif_inactive_when_zero(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=0, line_end_extra_px=0, inner_corner_extra_px=0)
        design = torch.zeros(16, 16)
        design[4:9, 8] = 1.0
        design[8, 8:13] = 1.0
        result = model.predict(design)
        assert torch.equal(result.mask, design)

    def test_iso_dense_split_uses_iso_radius_in_sparse_region(self) -> None:
        model = RuleBasedOPCModel(
            bias_radius_px=0,
            line_end_extra_px=0,
            iso_radius_px=3,
            dense_radius_px=1,
            density_window_px=7,
            density_threshold=0.3,
        )
        sparse = torch.zeros(48, 48)
        sparse[22:26, 22:26] = 1.0
        sparse_result = model.predict(sparse)
        sparse_growth = sparse_result.metadata["mask_area_growth"]

        dense = torch.zeros(48, 48)
        for col in range(8, 40, 2):
            dense[8:40, col] = 1.0
        dense_result = model.predict(dense)
        dense_growth = dense_result.metadata["mask_area_growth"]

        assert sparse_growth > dense_growth

    def test_iso_dense_inactive_when_unset(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=2, line_end_extra_px=0)
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0
        baseline = model.predict(design)
        # iso/dense both None → identical to single-radius dilation.
        explicit = model.predict(design, iso_radius_px=None, dense_radius_px=None)
        assert torch.equal(baseline.mask, explicit.mask)

    def test_metadata_reports_min_space_and_growth(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=1, line_end_extra_px=0)
        design = torch.zeros(32, 32)
        design[8:24, 10] = 1.0
        design[8:24, 18] = 1.0  # two vertical lines, gap = 7px
        result = model.predict(design)
        assert result.metadata["min_space_px"] > 0
        assert result.metadata["min_space_px"] < 8
        assert result.metadata["mask_area_growth"] > 1.0
        assert result.metadata["min_width_px"] >= 1

    def test_mrc_retreat_resolves_violation(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=2, line_end_extra_px=0)
        design = torch.zeros(32, 32)
        design[8:24, 12] = 1.0
        design[8:24, 16] = 1.0  # gap = 3px, dilate by 2 → spaces collapse
        violated = model.predict(design, mrc_min_space_px=0)
        # baseline with violation reporting off
        assert violated.metadata["mrc_violated"] is False
        # now demand min_space_px >= 3 — retreat should clean it
        cleaned = model.predict(design, mrc_min_space_px=3)
        assert cleaned.metadata["mrc_violated"] is False
        assert cleaned.metadata["min_space_px"] == 0 or cleaned.metadata["min_space_px"] >= 3

    def test_bias_radius_nm_converts_with_pixel_size(self) -> None:
        model = RuleBasedOPCModel(bias_radius_px=0, line_end_extra_px=0)
        design = torch.zeros(32, 32)
        design[14:18, 14:18] = 1.0
        result_nm = model.predict(design, bias_radius_nm=4.0, pixel_size_nm=2.0)
        result_px = model.predict(design, bias_radius_px=2)
        assert torch.equal(result_nm.mask, result_px.mask)
        assert result_nm.metadata["bias_radius_nm"] == 4.0
        assert result_nm.metadata["bias_radius_px"] == 2

    def test_bias_radius_nm_without_pixel_size_raises(self) -> None:
        model = RuleBasedOPCModel()
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        with pytest.raises(ValueError):
            model.predict(design, bias_radius_nm=4.0)
        with pytest.raises(ValueError):
            model.predict(design, pixel_size_nm=2.0)
