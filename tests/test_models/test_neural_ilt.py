"""Tests for openlithohub.models.neural_ilt."""

import warnings

import pytest
import torch

from openlithohub.models.base import PredictionResult
from openlithohub.models.neural_ilt import NeuralILTModel
from openlithohub.models.registry import registry


@pytest.fixture(autouse=True)
def _silence_no_weights_warning():
    # Constructing NeuralILTModel without weights is intentional in these
    # tests — they exercise plumbing, not prediction quality. The
    # "no weights loaded" UserWarning is asserted explicitly below in
    # test_warns_when_no_weights_loaded.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="NeuralILTModel is running with random-initialized weights",
            category=UserWarning,
        )
        yield


class TestNeuralILTModel:
    def test_registered_in_registry(self) -> None:
        model = registry.get("neural-ilt")
        assert isinstance(model, NeuralILTModel)

    def test_properties(self) -> None:
        model = NeuralILTModel()
        assert model.name == "neural-ilt"
        assert model.supports_curvilinear is True

    def test_predict_returns_prediction_result(self) -> None:
        model = NeuralILTModel()
        model.setup()
        design = torch.zeros(64, 64)
        design[16:48, 16:48] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)
        assert result.mask.shape == design.shape

    def test_predict_mask_is_binary(self) -> None:
        model = NeuralILTModel()
        model.setup()
        design = torch.rand(64, 64)
        result = model.predict(design)
        unique_vals = result.mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_auto_setup_on_predict(self) -> None:
        model = NeuralILTModel()
        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)
        assert isinstance(result, PredictionResult)

    def test_teardown_clears_network(self) -> None:
        model = NeuralILTModel()
        model.setup()
        assert model._net is not None
        model.teardown()
        assert model._net is None

    def test_metadata_contains_logits_range(self) -> None:
        model = NeuralILTModel()
        design = torch.rand(32, 32)
        result = model.predict(design)
        assert "logits_range" in result.metadata
        lo, hi = result.metadata["logits_range"]
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_different_input_sizes(self) -> None:
        model = NeuralILTModel()
        model.setup()
        for size in [16, 32, 64]:
            design = torch.rand(size, size)
            result = model.predict(design)
            assert result.mask.shape == (size, size)

    def test_warns_when_no_weights_loaded(self) -> None:
        # BatchNorm in eval() with default running stats produces meaningless
        # output. Surface this as a UserWarning so users running the model
        # without --pretrained / --weights are not silently misled.
        model = NeuralILTModel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.setup()
        assert any(
            issubclass(w.category, UserWarning) and "random-initialized weights" in str(w.message)
            for w in caught
        )
