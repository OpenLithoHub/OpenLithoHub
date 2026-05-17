"""Neural-ILT: U-Net based mask prediction model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


@registry.register
class NeuralILTModel(LithographyModel):
    """Neural ILT model using a U-Net architecture for mask prediction.

    Predicts an optimized mask directly from the design layout in a single
    forward pass. Much faster than iterative methods at inference time,
    but requires pretrained weights for good results.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        pretrained: bool = False,
        device: str = "cpu",
    ) -> None:
        self._weights_path = weights
        self._pretrained = pretrained
        self._device = device
        self._net: torch.nn.Module | None = None

    @property
    def name(self) -> str:
        return "neural-ilt"

    @property
    def supports_curvilinear(self) -> bool:
        return True

    def setup(self) -> None:
        from openlithohub.models._unet import UNet

        self._net = UNet(in_channels=1, out_channels=1).to(self._device)

        if self._weights_path is not None:
            weights_path = Path(self._weights_path)
            if weights_path.exists():
                state_dict = torch.load(
                    str(weights_path), map_location=self._device, weights_only=True
                )
                self._net.load_state_dict(state_dict)
        elif self._pretrained:
            from openlithohub.models.hub import ModelHub

            hub = ModelHub()
            path = hub.download_weights("openlithohub/neural-ilt-v1")
            state_dict = torch.load(str(path), map_location=self._device, weights_only=True)
            self._net.load_state_dict(state_dict)

        self._net.eval()

    def teardown(self) -> None:
        self._net = None

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        if self._net is None:
            self.setup()
        assert self._net is not None

        inp = design.detach().float()
        if inp.ndim == 2:
            inp = inp.unsqueeze(0).unsqueeze(0)
        elif inp.ndim == 3:
            inp = inp.unsqueeze(0)

        inp = inp.to(self._device)

        with torch.no_grad():
            logits = self._net(inp)
            mask = torch.sigmoid(logits).squeeze()

        mask_binary = (mask > 0.5).float()
        return PredictionResult(
            mask=mask_binary,
            contour=None,
            metadata={"logits_range": (float(logits.min()), float(logits.max()))},
        )
