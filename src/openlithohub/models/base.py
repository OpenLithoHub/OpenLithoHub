"""Abstract base class for lithography optimization models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch


@dataclass
class PredictionResult:
    """Result from a model prediction."""

    mask: torch.Tensor
    contour: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LithographyModel(ABC):
    """Abstract interface for lithography optimization models.

    Any model (heuristic OPC, U-Net, diffusion-based ILT, curvyILT)
    can join the evaluation pipeline by implementing predict().

    Subclasses MUST set the class-level ``NAME`` attribute. The registry
    reads it without instantiating the class, so it cannot be set in
    ``__init__``.
    """

    NAME: ClassVar[str]
    SUPPORTS_CURVILINEAR: ClassVar[bool] = False

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        return type(self).NAME

    @property
    def supports_curvilinear(self) -> bool:
        """Whether this model produces curvilinear (non-Manhattan) output."""
        return type(self).SUPPORTS_CURVILINEAR

    @abstractmethod
    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Run model inference on a design layout tensor.

        Args:
            design: Input design tensor of shape (H, W) or (B, C, H, W).
            **kwargs: Model-specific parameters (process node, dose, etc.)

        Returns:
            PredictionResult with the optimized mask and optional contour.
        """
        ...

    def setup(self) -> None:
        """Optional setup hook (load weights, initialize GPU, etc.)."""

    def teardown(self) -> None:
        """Optional cleanup hook."""

    def to_torch_module(self) -> torch.nn.Module:
        """Return a single ``nn.Module`` that maps an input mask/design tensor
        to an output mask tensor, suitable for ONNX / TorchScript export.

        Default raises ``NotImplementedError``. Override this on models that
        wrap a static ``nn.Module`` (e.g. Neural-ILT). Iterative optimizers
        like Level-Set ILT do not have a single forward graph and cannot be
        exported — those should keep the default.

        The returned module must be in ``eval()`` mode and accept tensors
        of shape ``(B, 1, H, W)`` in ``[0, 1]``, returning the same shape.
        """
        raise NotImplementedError(
            f"Model {type(self).NAME!r} does not support export to a single "
            "nn.Module — override LithographyModel.to_torch_module() if it "
            "wraps a static forward graph."
        )
