"""Tests for model layer interfaces."""

import pytest
import torch

from openlithohub.models import PredictionResult
from openlithohub.models.examples.dummy_model import DummyModel


def test_dummy_model_predict(sample_design):
    model = DummyModel()
    result = model.predict(sample_design)
    assert isinstance(result, PredictionResult)
    assert torch.equal(result.mask, sample_design)


def test_dummy_model_properties():
    model = DummyModel()
    assert model.name == "dummy-identity"
    assert model.supports_curvilinear is False


def test_registry_get():
    from openlithohub.models.registry import registry

    model = registry.get("dummy-identity")
    assert isinstance(model, DummyModel)


def test_registry_missing():
    from openlithohub.models.registry import registry

    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent-model")


def test_registry_list():
    from openlithohub.models.registry import registry

    models = registry.list_models()
    assert "dummy-identity" in models
