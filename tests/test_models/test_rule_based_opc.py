"""Tests for openlithohub.models.rule_based_opc."""

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
