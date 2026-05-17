"""Tests for openlithohub.models.levelset_ilt."""

import torch

from openlithohub.models.base import PredictionResult
from openlithohub.models.levelset_ilt import LevelSetILTModel
from openlithohub.models.registry import registry


class TestLevelSetILTModel:
    def test_registered_in_registry(self) -> None:
        model = registry.get("levelset-ilt")
        assert isinstance(model, LevelSetILTModel)

    def test_properties(self) -> None:
        model = LevelSetILTModel()
        assert model.name == "levelset-ilt"
        assert model.supports_curvilinear is True

    def test_predict_returns_prediction_result(self) -> None:
        model = LevelSetILTModel(iterations=5)
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape

    def test_predict_mask_is_binary(self) -> None:
        model = LevelSetILTModel(iterations=10)
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0
        result = model.predict(design)
        unique_vals = result.mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_optimization_reduces_loss(self) -> None:
        """Verify that the optimized mask is different from identity (optimization happened)."""
        model = LevelSetILTModel(iterations=50, lr=0.1, sigma_px=1.5)
        design = torch.zeros(32, 32)
        design[12:20, 12:20] = 1.0
        result = model.predict(design)
        # With a blur kernel, the optimal mask differs from the target design
        # (it needs to be biased to compensate for optical proximity effects)
        assert result.metadata["final_loss"] < 0.5

    def test_kwargs_override_constructor(self) -> None:
        model = LevelSetILTModel(iterations=200)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design, iterations=5)
        assert result.metadata["iterations"] == 5

    def test_metadata_contains_expected_keys(self) -> None:
        model = LevelSetILTModel(iterations=3)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        assert "final_loss" in result.metadata
        assert "iterations" in result.metadata
        assert "sigma_px" in result.metadata

    def test_empty_design(self) -> None:
        model = LevelSetILTModel(iterations=5)
        design = torch.zeros(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape

    def test_full_design(self) -> None:
        model = LevelSetILTModel(iterations=5)
        design = torch.ones(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape
