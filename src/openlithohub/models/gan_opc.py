"""GAN-OPC baseline — generator-only inference of the Yang2018_GANOPC model.

PLACEHOLDER / UNIMPLEMENTED — Without trained weights this model emits
near-random masks. Even with weights, the GAN discriminator and
lithography-guided training loop are NOT implemented (generator-only
inference). Do not use for evaluation without pretrained weights.

Architecture follows Yang et al., *GAN-OPC: Mask Optimization with
Lithography-guided Generative Adversarial Nets*, DAC 2018
([doi:10.1145/3195970.3196056](https://doi.org/10.1145/3195970.3196056)) §IV.A — the
generator is an encoder-decoder with skip connections trained against a
discriminator on the GAN-OPC paired dataset (target layout → OPC mask).
This adapter only ships the generator side; the discriminator is a
training-time concern.

For inference we reuse the in-tree :class:`UNet` (encoder/decoder
channels 32→64→128→256, bilinear upsampling) as a faithful enough
generator silhouette — the paper's published architecture diagram (§IV
Fig. 4, confidence **C** since read from the figure) is a 4-level
encoder-decoder of the same shape. Channel widths in the v0 baseline
are halved relative to the paper for consistency with our Neural-ILT
v0.1 baseline; bumping them is a hyperparameter knob, not a rewrite.

Without trained weights this baseline emits an untrained-mask
prediction — useful as a registry-level smoke test, not a competitive
leaderboard entry. Pin a checkpoint via the ``weights`` arg or
``pretrained=True`` (HF Hub) to score real numbers.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry


@registry.register
class GanOpcModel(LithographyModel):
    """GAN-OPC generator baseline (Yang2018_GANOPC, DAC 2018).

    PLACEHOLDER / UNIMPLEMENTED — Generator-only inference without the
    GAN discriminator or lithography-guided training loop. Without
    pretrained weights, predictions are near-random.

    Args:
        weights: Optional path to a state-dict ``.pt`` file. Takes
            precedence over ``pretrained``.
        pretrained: If True, fetch weights from HuggingFace Hub at
            ``repo_id``/``repo_filename``. Mutually exclusive with
            ``weights``.
        device: Torch device (``"cpu"``, ``"cuda"``, …).
        repo_id: HF Hub repo for pretrained weights. Default points at
            the placeholder slot reserved for a future v0.1 release.
        repo_filename: Weights filename inside ``repo_id``.
        url_sha256: Optional SHA-256 to verify against the downloaded
            weights file.
    """

    # NOTE: Placeholder implementation — not functional, do not use for evaluation

    NAME = "gan-opc"
    SUPPORTS_CURVILINEAR = True
    RECEPTIVE_FIELD_PX = 64

    def __init__(
        self,
        weights: str | Path | None = None,
        pretrained: bool = False,
        device: str = "cpu",
        repo_id: str = "openlithohub/gan-opc-v0.1",
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
        self._weights_loaded: bool = False

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
            # Untrained generators emit near-random masks; warn so a
            # leaderboard run isn't accidentally scored on noise.
            warnings.warn(
                "GanOpcModel running without trained weights — predictions "
                "will be near-random. Pass `weights=...` or `pretrained=True` "
                "for a meaningful baseline.",
                stacklevel=2,
            )

        # Always switch to eval mode after setup. Otherwise BatchNorm uses
        # per-batch statistics at inference, producing different outputs for
        # the same mask depending on what else is in the batch.
        self._net.eval()
        self._weights_loaded = weights_loaded

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        if self._net is None:
            self.setup()
        assert self._net is not None  # for type checker

        x = design.to(self._device)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            logits = self._net(x)
            mask = torch.sigmoid(logits)

        # Binarize to match the LithographyModel.predict() contract: the
        # downstream metric stack (DRC/MRC, PV-band, shot-count) operates
        # on {0, 1} masks, and the Neural-ILT baseline binarizes too.
        # Emitting a soft mask here would silently bias the leaderboard.
        mask = (mask > 0.5).float()

        if design.ndim == 2:
            mask = mask.squeeze(0).squeeze(0)
        elif design.ndim == 3:
            mask = mask.squeeze(0)

        return PredictionResult(
            mask=mask.cpu(),
            metadata={"model": self.NAME, "weights_loaded": self._weights_loaded},
        )

    def to_torch_module(self) -> torch.nn.Module:
        if self._net is None:
            self.setup()
        assert self._net is not None

        class _GanOpcExportWrapper(torch.nn.Module):
            def __init__(self, net: torch.nn.Module) -> None:
                super().__init__()
                self.net = net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self.net(x))

        return _GanOpcExportWrapper(self._net).eval()
