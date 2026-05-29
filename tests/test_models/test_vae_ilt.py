"""Tests for openlithohub.models.vae_ilt."""

import torch

from openlithohub._utils.hopkins import HopkinsParams, clear_kernel_cache
from openlithohub.models.base import PredictionResult
from openlithohub.models.registry import registry
from openlithohub.models.vae_ilt import VAEILTModel


class TestVAEILTModel:
    def test_registered_in_registry(self) -> None:
        model = registry.get("vae-ilt")
        assert isinstance(model, VAEILTModel)

    def test_properties(self) -> None:
        model = VAEILTModel()
        assert model.name == "vae-ilt"
        assert model.supports_curvilinear is True

    def test_predict_returns_prediction_result(self) -> None:
        model = VAEILTModel(
            iterations=5,
            vae_train_masks=64,
            vae_epochs=2,
        )
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape

    def test_predict_mask_is_binary(self) -> None:
        model = VAEILTModel(
            iterations=10,
            vae_train_masks=64,
            vae_epochs=2,
        )
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0
        result = model.predict(design)
        unique_vals = result.mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_optimization_reduces_loss(self) -> None:
        model = VAEILTModel(
            iterations=30,
            lr=0.05,
            vae_train_masks=128,
            vae_epochs=5,
        )
        design = torch.zeros(32, 32)
        design[12:20, 12:20] = 1.0
        result = model.predict(design)
        assert result.metadata["final_loss"] < 0.5

    def test_metadata_contains_expected_keys(self) -> None:
        model = VAEILTModel(
            iterations=3,
            vae_train_masks=64,
            vae_epochs=2,
        )
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        assert "final_loss" in result.metadata
        assert "iterations" in result.metadata
        assert "latent_dim" in result.metadata
        assert "forward_model" in result.metadata

    def test_empty_design(self) -> None:
        model = VAEILTModel(
            iterations=3,
            vae_train_masks=32,
            vae_epochs=1,
        )
        design = torch.zeros(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape

    def test_full_design(self) -> None:
        model = VAEILTModel(
            iterations=3,
            vae_train_masks=32,
            vae_epochs=1,
        )
        design = torch.ones(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape


class TestVAEILTHopkinsForwardModel:
    def test_hopkins_mode_runs_end_to_end(self) -> None:
        clear_kernel_cache()
        model = VAEILTModel(
            iterations=5,
            forward_model="hopkins",
            hopkins_params=HopkinsParams(num_kernels=4, pixel_size_nm=2.0, sigma=0.7),
            vae_train_masks=64,
            vae_epochs=2,
        )
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape
        assert result.metadata["forward_model"] == "hopkins"
