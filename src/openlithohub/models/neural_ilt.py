"""Neural-ILT: U-Net based mask prediction model."""

from __future__ import annotations

import warnings
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

    Architecture lineage: ``Jiang2020_NeuralILT`` (ICCAD'20, paywalled).
    Open-access substitute is [Yang2023_LithoBench, §4.3] (NeurIPS 2023),
    which describes the same architecture and training
    schedule. OpenLithoHub's v0.1 baseline diverges from the paper in two
    ways: encoder/decoder channel widths are halved (32→64→128→256 vs
    paper's 64→128→256→512), and the differentiable ILT correction layer
    is not packaged inside the adapter — eval-time forward simulation
    lives in the leaderboard scoring pipeline. See
    ``docs/audits/neural-ilt-architecture.md`` for the full audit and
    re-audit triggers.
    """

    NAME = "neural-ilt"
    SUPPORTS_CURVILINEAR = True
    RECEPTIVE_FIELD_PX = 64

    def __init__(
        self,
        weights: str | Path | None = None,
        pretrained: bool = False,
        device: str = "cpu",
        repo_id: str = "openlithohub/neural-ilt-v0.1",
        repo_filename: str = "model.pt",
        url_sha256: str | None = None,
    ) -> None:
        self._weights_path = weights
        self._pretrained = pretrained
        self._device = device
        self._repo_id = repo_id
        self._repo_filename = repo_filename
        self._url_sha256 = url_sha256
        self._net: torch.nn.Module | None = None

    def setup(self) -> None:
        from openlithohub.models._unet import UNet

        self._net = UNet(in_channels=1, out_channels=1).to(self._device)

        weights_loaded = False
        if self._weights_path is not None:
            weights_path = Path(self._weights_path)
            if weights_path.exists():
                state_dict = torch.load(
                    str(weights_path), map_location=self._device, weights_only=True
                )
                self._net.load_state_dict(state_dict)
                weights_loaded = True
        elif self._pretrained:
            from openlithohub.models.hub import ModelHub

            hub = ModelHub()
            path = hub.download_weights(
                self._repo_id, filename=self._repo_filename, sha256=self._url_sha256
            )
            state_dict = torch.load(str(path), map_location=self._device, weights_only=True)
            self._net.load_state_dict(state_dict)
            weights_loaded = True

        if not weights_loaded:
            # BatchNorm in eval() mode with default running stats produces
            # near-arbitrary outputs that are then thresholded to a binary
            # mask. Surface this so users know predictions are meaningless.
            warnings.warn(
                "NeuralILTModel is running with random-initialized weights. "
                "Predictions are not meaningful. Pass --pretrained or --weights "
                "to load trained parameters.",
                UserWarning,
                stacklevel=2,
            )

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
            metadata={"logits_range": (logits.min().item(), logits.max().item())},
        )

    def to_torch_module(self) -> torch.nn.Module:
        """Return the U-Net wrapped so the export forward emits a sigmoid mask
        in ``[0, 1]`` directly — what a downstream production pipeline expects.
        """
        if self._net is None:
            self.setup()
        assert self._net is not None
        unet = self._net

        class _NeuralILTExportWrapper(torch.nn.Module):
            def __init__(self, net: torch.nn.Module) -> None:
                super().__init__()
                self.net = net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self.net(x))

        wrapper = _NeuralILTExportWrapper(unet).eval()
        return wrapper
