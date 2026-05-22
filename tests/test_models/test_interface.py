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


def test_registry_warns_on_name_collision():
    """Re-registering a different class with an existing NAME must warn, not silently overwrite."""
    import warnings

    from openlithohub.models.registry import ModelRegistry

    class A:
        NAME = "collide-me"

        def predict(self, *_, **__):  # pragma: no cover - shape only
            ...

    class B:
        NAME = "collide-me"

        def predict(self, *_, **__):  # pragma: no cover - shape only
            ...

    reg = ModelRegistry()
    reg.register(A)  # type: ignore[arg-type]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.register(B)  # type: ignore[arg-type]
    assert any(
        issubclass(w.category, UserWarning) and "re-registered" in str(w.message) for w in caught
    )


def test_registry_idempotent_on_same_class():
    """Re-registering the exact same class must NOT warn.

    ``register_builtin_models`` is called from both the parent and worker
    processes, so the same class can pass through ``register`` more than once.
    """
    import warnings

    from openlithohub.models.registry import ModelRegistry

    class C:
        NAME = "idempotent"

        def predict(self, *_, **__):  # pragma: no cover - shape only
            ...

    reg = ModelRegistry()
    reg.register(C)  # type: ignore[arg-type]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.register(C)  # type: ignore[arg-type]
    assert not any(issubclass(w.category, UserWarning) for w in caught)
