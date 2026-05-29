"""Tests for openlithohub.models.surrogate_ilt."""

import torch

from openlithohub._utils.hopkins import HopkinsParams, clear_kernel_cache
from openlithohub.models.base import PredictionResult
from openlithohub.models.registry import registry
from openlithohub.models.surrogate_ilt import SurrogateILTModel


class TestSurrogateILTModel:
    def test_registered_in_registry(self) -> None:
        model = registry.get("surrogate-ilt")
        assert isinstance(model, SurrogateILTModel)

    def test_properties(self) -> None:
        model = SurrogateILTModel()
        assert model.name == "surrogate-ilt"
        assert model.supports_curvilinear is True

    def test_predict_returns_prediction_result(self) -> None:
        model = SurrogateILTModel(
            iterations=10,
            surrogate_train_samples=64,
            surrogate_epochs=3,
        )
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape

    def test_predict_mask_is_binary(self) -> None:
        model = SurrogateILTModel(
            iterations=10,
            surrogate_train_samples=64,
            surrogate_epochs=3,
        )
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0
        result = model.predict(design)
        unique_vals = result.mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_optimization_reduces_loss(self) -> None:
        model = SurrogateILTModel(
            iterations=30,
            surrogate_train_samples=128,
            surrogate_epochs=5,
        )
        design = torch.zeros(32, 32)
        design[12:20, 12:20] = 1.0
        result = model.predict(design)
        assert result.metadata["final_loss"] < 0.5

    def test_metadata_contains_expected_keys(self) -> None:
        model = SurrogateILTModel(
            iterations=5,
            surrogate_train_samples=64,
            surrogate_epochs=2,
        )
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        assert "final_loss" in result.metadata
        assert "iterations" in result.metadata
        assert "sigma_px" in result.metadata
        assert "surrogate_correction_interval" in result.metadata
        assert "true_forward_calls" in result.metadata

    def test_correction_reduces_forward_calls(self) -> None:
        model = SurrogateILTModel(
            iterations=20,
            correction_interval=5,
            surrogate_train_samples=64,
            surrogate_epochs=2,
        )
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        assert result.metadata["true_forward_calls"] < 20

    def test_empty_design(self) -> None:
        model = SurrogateILTModel(
            iterations=3,
            surrogate_train_samples=32,
            surrogate_epochs=1,
        )
        design = torch.zeros(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape

    def test_full_design(self) -> None:
        model = SurrogateILTModel(
            iterations=3,
            surrogate_train_samples=32,
            surrogate_epochs=1,
        )
        design = torch.ones(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape


class TestSurrogateILTHopkinsForwardModel:
    def test_hopkins_mode_runs_end_to_end(self) -> None:
        clear_kernel_cache()
        model = SurrogateILTModel(
            iterations=10,
            forward_model="hopkins",
            hopkins_params=HopkinsParams(num_kernels=4, pixel_size_nm=2.0, sigma=0.7),
            surrogate_train_samples=64,
            surrogate_epochs=2,
        )
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape
        assert result.metadata["forward_model"] == "hopkins"
