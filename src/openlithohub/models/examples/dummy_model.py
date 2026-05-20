"""Dummy identity model for testing the evaluation pipeline."""

from __future__ import annotations

from typing import Any

import torch

from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


@registry.register
class DummyModel(LithographyModel):
    """Trivial model that returns the input design as the mask (identity)."""

    NAME = "dummy-identity"
    SUPPORTS_CURVILINEAR = False
    RECEPTIVE_FIELD_PX = 0

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        return PredictionResult(mask=design.clone())


@registry.register
class FailingDummyModel(LithographyModel):
    """Dummy model whose ``predict`` always raises.

    Exists so the multi-GPU tile pipeline can exercise worker error
    propagation across processes — pickling a closure that raises does
    not survive the spawn boundary, but a registered model name does.
    """

    NAME = "dummy-failing"
    SUPPORTS_CURVILINEAR = False
    RECEPTIVE_FIELD_PX = 0

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        raise RuntimeError("dummy-failing: deliberate predict failure")
