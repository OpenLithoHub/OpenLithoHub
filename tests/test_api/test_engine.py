"""Tests for the `LitheEngine` driver."""

from __future__ import annotations

import pytest
import torch

from openlithohub import LitheEngine, Mask
from openlithohub.models.base import LithographyModel, PredictionResult


def test_list_models_includes_builtins() -> None:
    names = LitheEngine.list_models()
    for expected in ("dummy-identity", "neural-ilt", "rule-based-opc", "levelset-ilt"):
        assert expected in names


def test_optimize_with_mask_preserves_shape(sample_design: torch.Tensor) -> None:
    mask = Mask.from_tensor(sample_design, pixel_size_nm=0.5, layer="1:0")
    engine = LitheEngine(model="dummy-identity")
    out = engine.optimize(mask)
    assert isinstance(out, Mask)
    assert out.shape == mask.shape
    assert out.pixel_size_nm == 0.5
    assert out.layer == "1:0"


def test_optimize_accepts_raw_tensor(sample_design: torch.Tensor) -> None:
    """Backward-compat: optimize() takes either Mask or torch.Tensor."""
    engine = LitheEngine(model="dummy-identity")
    out = engine.optimize(sample_design)
    assert isinstance(out, Mask)
    assert out.shape == tuple(sample_design.shape)


def test_optimize_rejects_other_types() -> None:
    engine = LitheEngine(model="dummy-identity")
    with pytest.raises(TypeError):
        engine.optimize("not-a-mask")  # type: ignore[arg-type]


def test_dummy_identity_is_identity_after_binarise(sample_design: torch.Tensor) -> None:
    engine = LitheEngine(model="dummy-identity")
    out = engine.optimize(Mask.from_tensor(sample_design))
    # `dummy-identity` returns input unchanged; the engine binarises at 0.5,
    # so the output equals (input > 0.5).
    expected = (sample_design > 0.5).float()
    assert torch.equal(out.tensor, expected)


def test_setup_called_once_on_construction() -> None:
    """Engine should call setup() exactly once at construction, not per optimize()."""

    class CountingModel(LithographyModel):
        NAME = "counting-model"
        SUPPORTS_CURVILINEAR = False
        RECEPTIVE_FIELD_PX = 0

        def __init__(self) -> None:
            self.setup_calls = 0

        def setup(self) -> None:
            self.setup_calls += 1

        def predict(self, design: torch.Tensor, **_: object) -> PredictionResult:
            return PredictionResult(mask=design.clone())

    model = CountingModel()
    engine = LitheEngine(model=model)
    t = torch.zeros(64, 64)
    t[16:48, 16:48] = 1.0
    engine.optimize(t)
    engine.optimize(t)
    assert model.setup_calls == 1


def test_instance_model_rejects_kwargs() -> None:
    """Passing model_kwargs alongside an already-built model is a user error."""

    class IdentityModel(LithographyModel):
        NAME = "instance-identity"
        SUPPORTS_CURVILINEAR = False
        RECEPTIVE_FIELD_PX = 0

        def predict(self, design: torch.Tensor, **_: object) -> PredictionResult:
            return PredictionResult(mask=design.clone())

    with pytest.raises(ValueError, match="model_kwargs"):
        LitheEngine(model=IdentityModel(), pretrained=True)


def test_unknown_model_name_raises() -> None:
    with pytest.raises(KeyError):
        LitheEngine(model="not-a-real-model")


def test_node_overrides_default_pixel_pitch(sample_design: torch.Tensor) -> None:
    """When the input mask uses the default 1.0 nm/px, the node's pitch wins."""
    engine = LitheEngine(model="dummy-identity", node="3nm-euv")
    mask = Mask.from_tensor(sample_design)  # default pixel_size_nm = 1.0
    out = engine.optimize(mask)
    # 3nm-euv has pixel_size_nm = 0.5 in PROCESS_NODES
    assert out.pixel_size_nm == 0.5


def test_node_does_not_override_explicit_pitch(sample_design: torch.Tensor) -> None:
    engine = LitheEngine(model="dummy-identity", node="3nm-euv")
    mask = Mask.from_tensor(sample_design, pixel_size_nm=0.25)
    out = engine.optimize(mask)
    assert out.pixel_size_nm == 0.25
