"""Layer 3: Model Integration — abstract interface and registry for lithography models."""

from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import ModelRegistry

__all__ = ["LithographyModel", "PredictionResult", "ModelRegistry"]
