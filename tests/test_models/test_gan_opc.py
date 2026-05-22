"""Tests for openlithohub.models.gan_opc."""

import warnings

import pytest
import torch

from openlithohub.models.base import PredictionResult
from openlithohub.models.gan_opc import GanOpcModel
from openlithohub.models.registry import register_builtin_models, registry


@pytest.fixture(autouse=True)
def _silence_no_weights_warning():
    # Building GanOpcModel without trained weights is intentional here —
    # these tests exercise the registry / IO plumbing, not the
    # mask-quality which would require a published checkpoint we don't
    # ship yet.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="GanOpcModel running without trained weights",
            category=UserWarning,
        )
        yield


class TestGanOpcModel:
    def test_registered_in_registry(self) -> None:
        register_builtin_models()
        model = registry.get("gan-opc")
        assert isinstance(model, GanOpcModel)

    def test_properties(self) -> None:
        model = GanOpcModel()
        assert model.name == "gan-opc"
        assert model.supports_curvilinear is True

    def test_predict_returns_prediction_result(self) -> None:
        model = GanOpcModel()
        model.setup()
        design = torch.zeros(64, 64)
        design[16:48, 16:48] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape

    def test_predict_preserves_3d_batch(self) -> None:
        model = GanOpcModel()
        design = torch.zeros(1, 64, 64)
        design[..., 16:48, 16:48] = 1.0
        result = model.predict(design)
        assert result.mask.shape == design.shape

    def test_auto_setup_on_predict(self) -> None:
        model = GanOpcModel()
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)

    def test_to_torch_module_returns_eval_module(self) -> None:
        model = GanOpcModel()
        module = model.to_torch_module()
        assert isinstance(module, torch.nn.Module)
        assert not module.training

    def test_warns_when_no_weights_loaded(self) -> None:
        model = GanOpcModel()
        with pytest.warns(UserWarning, match="without trained weights"):
            model.setup()
