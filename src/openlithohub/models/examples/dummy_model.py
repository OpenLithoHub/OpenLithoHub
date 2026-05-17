"""Dummy identity model for testing the evaluation pipeline."""

from __future__ import annotations

from typing import Any

import torch

from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


@registry.register
class DummyModel(LithographyModel):
    """Trivial model that returns the input design as the mask (identity)."""

    @property
    def name(self) -> str:
        return "dummy-identity"

    @property
    def supports_curvilinear(self) -> bool:
        return False

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        return PredictionResult(mask=design.clone())
