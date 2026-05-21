"""Tests for openlithohub.models.openilt."""

import torch

from openlithohub._utils.hopkins import HopkinsParams, clear_kernel_cache
from openlithohub.models.base import PredictionResult
from openlithohub.models.openilt import OpenILTModel, PVBandCorners
from openlithohub.models.registry import registry


class TestOpenILTModel:
    def test_registered_in_registry(self) -> None:
        model = registry.get("openilt")
        assert isinstance(model, OpenILTModel)

    def test_properties(self) -> None:
        model = OpenILTModel()
        assert model.name == "openilt"
        assert model.supports_curvilinear is True

    def test_predict_returns_prediction_result(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=5)
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape

    def test_predict_mask_is_binary(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=10)
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0
        result = model.predict(design)
        unique_vals = result.mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_optimization_reduces_loss(self) -> None:
        """Verify the optimizer drives the loss below a sane threshold.

        Mirrors the LevelSet-ILT smoke threshold; at 50 iterations on a
        small block under the Gaussian forward, both losses should sit
        well below 0.5.
        """
        torch.manual_seed(0)
        model = OpenILTModel(iterations=50, lr=2.0)
        design = torch.zeros(32, 32)
        design[12:20, 12:20] = 1.0
        result = model.predict(design)
        assert result.metadata["final_loss"] < 0.5

    def test_metadata_keys(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=3)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        for key in (
            "final_loss",
            "l2_nom",
            "pvb_loss",
            "iterations",
            "forward_model",
            "pvb_weight",
        ):
            assert key in result.metadata

    def test_kwargs_override_constructor(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=200)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design, iterations=5)
        assert result.metadata["iterations"] == 5

    def test_deterministic_with_seed(self) -> None:
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        torch.manual_seed(42)
        m1 = OpenILTModel(iterations=10).predict(design).mask
        torch.manual_seed(42)
        m2 = OpenILTModel(iterations=10).predict(design).mask
        assert torch.equal(m1, m2)

    def test_empty_design(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=5)
        design = torch.zeros(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape

    def test_full_design(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=5)
        design = torch.ones(16, 16)
        result = model.predict(design)
        assert result.mask.shape == design.shape

    def test_pvb_weight_zero_isolates_l2(self) -> None:
        """With pvb_weight=0 the loss equals the nominal L2 term exactly."""
        torch.manual_seed(0)
        model = OpenILTModel(iterations=3, pvb_weight=0.0)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        assert abs(result.metadata["final_loss"] - result.metadata["l2_nom"]) < 1e-6

    def test_custom_corners(self) -> None:
        torch.manual_seed(0)
        corners = PVBandCorners(nom_dose=1.0, max_dose=1.10, min_dose=0.90)
        model = OpenILTModel(iterations=3, corners=corners)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design)
        assert result.mask.shape == design.shape

    def test_kwarg_lr_and_momentum_overrides(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=3)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        # Just exercise the override paths — assert no crash and shape preserved.
        result = model.predict(
            design, lr=0.5, momentum=0.5, pvb_weight=0.25, forward_model="gaussian"
        )
        assert result.mask.shape == design.shape
        assert result.metadata["pvb_weight"] == 0.25

    def test_higher_dim_design_squeezed(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=2)
        design = torch.zeros(1, 1, 16, 16)
        design[0, 0, 4:12, 4:12] = 1.0
        result = model.predict(design)
        assert result.mask.shape == (16, 16)

    def test_device_kwarg_cpu_passthrough(self) -> None:
        torch.manual_seed(0)
        model = OpenILTModel(iterations=2)
        design = torch.zeros(16, 16)
        design[4:12, 4:12] = 1.0
        result = model.predict(design, device="cpu")
        assert result.mask.device.type == "cpu"


class TestOpenILTHopkinsForwardModel:
    def test_hopkins_mode_runs_end_to_end(self) -> None:
        clear_kernel_cache()
        torch.manual_seed(0)
        model = OpenILTModel(
            iterations=3,
            forward_model="hopkins",
            hopkins_params=HopkinsParams(num_kernels=4, pixel_size_nm=2.0, sigma=0.7),
        )
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape
        assert result.metadata["forward_model"] == "hopkins"

    def test_hopkins_kernel_cache_reused(self) -> None:
        """Two predict() calls on the same shape reuse the SOCS kernels."""
        clear_kernel_cache()
        torch.manual_seed(0)
        model = OpenILTModel(
            iterations=2,
            forward_model="hopkins",
            hopkins_params=HopkinsParams(num_kernels=4, pixel_size_nm=2.0),
        )
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        model.predict(design)
        cached_id = id(model._cached_kernels_nom)
        model.predict(design)
        assert id(model._cached_kernels_nom) == cached_id

    def test_hopkins_param_change_invalidates_cache(self) -> None:
        """Changing HopkinsParams via kwargs forces fresh kernel computation."""
        clear_kernel_cache()
        torch.manual_seed(0)
        model = OpenILTModel(
            iterations=2,
            forward_model="hopkins",
            hopkins_params=HopkinsParams(num_kernels=4, pixel_size_nm=2.0, sigma=0.7),
        )
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        model.predict(design)
        old_kernels = model._cached_kernels_nom
        model.predict(
            design,
            hopkins_params=HopkinsParams(num_kernels=4, pixel_size_nm=2.0, sigma=0.5),
        )
        # Different params → re-cache happened.
        assert model._cached_kernels_nom is not old_kernels
