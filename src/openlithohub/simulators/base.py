"""Simulator base class and result/config dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class SimulatorConfig:
    """Vendor-neutral simulator configuration.

    Backends are free to ignore fields they don't model and to read
    extra options from :attr:`extra`. We deliberately keep the surface
    small — fields here must mean the same thing across every backend.

    Attributes:
        wavelength_nm: Exposure wavelength. 193 = ArF, 13.5 = EUV.
        na: Numerical aperture (image-side).
        sigma: Outer partial-coherence factor.
        sigma_inner: Inner sigma for annular/dipole/quasar (0 = circular).
        pixel_size_nm: Physical size of one mask pixel.
        defocus_nm: Defocus offset.
        dose: Linear dose multiplier.
        threshold: Resist intensity threshold for binarization (0–1
            relative to ``dose``). Backends that do not expose a
            threshold knob round at the model's nominal value.
        extra: Backend-specific options. Treated as opaque by the ABC.
    """

    wavelength_nm: float = 193.0
    na: float = 1.35
    sigma: float = 0.7
    sigma_inner: float = 0.0
    pixel_size_nm: float = 1.0
    defocus_nm: float = 0.0
    dose: float = 1.0
    # Per Yang2023_LithoBench §3.2 (NeurIPS 2023),
    # threshold = 0.225 is the canonical resist cutoff used to score
    # ICCAD16 mask layouts on the simulated wafer image. Confidence A.
    threshold: float = 0.225
    # Acid diffusion length in nm. 0.0 (default) produces bit-identical
    # results to the legacy hard-threshold CTR model; positive values
    # enable gaussian acid diffusion before binarization.
    resist_diffusion_nm: float = 0.0
    # Base quencher concentration subtracted from acid field after
    # diffusion. 0.0 disables quencher neutralization.
    quencher: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorResult:
    """Output of a simulator forward pass.

    Attributes:
        aerial: Aerial intensity image, same spatial shape as mask.
        resist: Optional binarized resist contour (0/1) at
            :attr:`SimulatorConfig.threshold`. ``None`` if the backend
            does not produce one (callers should threshold ``aerial``).
        backend: Name of the simulator that produced the result.
        metadata: Free-form per-backend metadata (e.g. license info,
            vendor version, kernel count). Treated as opaque.
    """

    aerial: torch.Tensor
    resist: torch.Tensor | None = None
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        from openlithohub.jupyter._html import (
            kv_table,
            mask_thumbnail_png_b64,
            panel,
            png_b64_to_img_tag,
        )

        shape = "x".join(str(d) for d in tuple(self.aerial.shape))
        rows = [
            ("backend", self.backend or "—"),
            ("aerial shape", shape),
            ("aerial dtype", str(self.aerial.dtype)),
            ("resist", "yes" if self.resist is not None else "no"),
        ]
        for k, v in list(self.metadata.items())[:6]:
            rows.append((f"meta:{k}", str(v)))

        preview_tensor = self.resist if self.resist is not None else self.aerial
        preview = preview_tensor
        if preview_tensor is not None:
            preview = preview_tensor.detach().cpu()
            if preview.numel() > 0:
                lo, hi = float(preview.min()), float(preview.max())
                if hi > lo:
                    preview = (preview - lo) / (hi - lo)
        img_html = png_b64_to_img_tag(
            mask_thumbnail_png_b64(preview),
            alt=self.backend or "simulation",
        )
        body = (
            f'<div style="display:flex;gap:10px;align-items:flex-start;">'
            f'<div style="flex:0 0 auto;">{img_html}</div>'
            f'<div style="flex:1 1 auto;">{kv_table(rows)}</div>'
            f"</div>"
        )
        return panel(title="SimulatorResult", header_html="", body_html=body)


class BaseSimulator(ABC):
    """Abstract simulator backend.

    A simulator maps ``mask -> SimulatorResult``. The ABC makes no
    assumption about differentiability; callers that need gradients
    should pick a backend that documents support (e.g.
    :class:`HopkinsSimulator`).

    Implementations should be cheap to construct — heavy state (kernel
    caches, tool licenses) belongs in :meth:`prepare` so that callers
    can decide when to pay the cost.
    """

    name: str = "base"
    differentiable: bool = False

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()

    def prepare(self) -> None:
        """Eagerly initialise backend state (kernels, tool sessions).

        Default no-op. Override when there is meaningful setup cost so
        that callers can amortise it across many simulate() calls.
        """

    def with_config(self, config: SimulatorConfig) -> BaseSimulator:
        """Return a sibling simulator using ``config``, sharing cached state where possible.

        Default builds a fresh instance via ``type(self)(config)``. Subclasses
        with expensive per-config setup (SOCS kernels, vendor sessions) should
        override to clone cheaply when the new config only differs in fields
        the cached state does not depend on (typically ``dose`` / ``threshold``).
        """
        return type(self)(config)

    @abstractmethod
    def simulate(self, mask: torch.Tensor) -> SimulatorResult:
        """Simulate the aerial image (and resist contour, if available).

        Args:
            mask: Real-valued mask. Shape ``(H, W)`` or ``(B, 1, H, W)``,
                values in ``[0, 1]``.

        Returns:
            A :class:`SimulatorResult` with the same spatial shape as
            ``mask``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, differentiable={self.differentiable})"
