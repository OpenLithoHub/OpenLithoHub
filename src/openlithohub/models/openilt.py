"""OpenILT-style ILT baseline (SimpleILT / MOSAIC formulation).

Clean-room PyTorch reimplementation of the *SimpleILT* formulation described
in the upstream OpenILT README — no source code is copied or vendored. The
formulation itself is the standard MOSAIC L2 + PVBand objective from the
academic literature; this module reuses OpenLithoHub's existing forward
optical model (``_utils.forward_model.simulate_aerial_image`` for the fast
Gaussian path, ``_utils.hopkins.simulate_aerial_image_hopkins`` for the
SOCS-truncated Hopkins path) rather than introducing a parallel optical core.

What distinguishes this baseline from ``LevelSetILTModel`` is the *loss
decomposition* and *initialization*:

- **PVBand-as-loss**: total loss is L2 at nominal + (∥Z_max − Z_T∥² +
  ∥Z_min − Z_T∥²) at the max/min process corners, where max/min are dose+sigma
  perturbations of the nominal corner. ``LevelSetILTModel`` uses a single
  fidelity term (or a 5-corner weighted-mean MSE under ``process_window=True``).
- **PixelInit**: mask logit is initialized to ``2 * target - 1`` (per the
  PixelInit scheme described in the OpenILT README), giving the optimizer
  a non-saturated starting point.
- **SGD**: gradient descent rather than Adam, matching the SimpleILT default.

These choices are themselves common ILT-textbook material; cross-referencing
them keeps the comparator honest as a *different* baseline, not a near-clone
of LevelSet-ILT.

References (no source code copied):

- OpenILT — OpenOPC/OpenILT, MIT license, pinned commit
  ``dabb97c6ca3dfd159362e48273c436444c77353b`` (2024-08-26).
  Algorithmic description sourced from the upstream README only.
- Gao et al., "MOSAIC: Mask Optimizing Solution With Process Window Aware
  Inverse Correction", DAC 2014 — original L2 + PVBand formulation.
- Banerjee et al., "ICCAD-2013 CAD Contest in Mask Optimization", ICCAD 2013 —
  benchmark suite the SimpleILT loss is calibrated against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as functional

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.hopkins import (
    HopkinsParams,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry

ForwardModelKind = Literal["gaussian", "hopkins"]


@dataclass(frozen=True)
class PVBandCorners:
    """Three-corner sweep for SimpleILT's PVBand loss.

    Nominal is the design exposure point. ``max`` and ``min`` are
    multiplicative dose offsets paired with a defocus broadening of the
    Gaussian-PSF path (or a defocus_nm offset on the Hopkins path). The
    asymmetric default (1.05 / 0.95) matches conventional ±5% dose latitude
    used in academic ILT studies.
    """

    nom_dose: float = 1.0
    max_dose: float = 1.05
    min_dose: float = 0.95
    nom_sigma_px: float = 2.0
    defocus_sigma_px: float = 2.5
    nom_defocus_nm: float = 0.0
    edge_defocus_nm: float = 50.0


@registry.register
class OpenILTModel(LithographyModel):
    """OpenILT-style ILT baseline using L2 + PVBand on a 3-corner sweep.

    Optimizes a continuous mask via SGD against:

        loss = ‖Z_nom − Z_T‖² + α · (‖Z_max − Z_T‖² + ‖Z_min − Z_T‖²)

    where Z_{nom,max,min} are simulated resist images at the nominal,
    max-dose+defocus, and min-dose+defocus corners. ``α`` is ``pvb_weight``.

    Two forward models are supported, both reused from the existing
    OpenLithoHub stack:

    - ``gaussian`` (default): Gaussian-PSF aerial image — fast, used in
      tests.
    - ``hopkins``: SOCS-truncated partial-coherent imaging via
      ``_utils/hopkins.py``.

    Hopkins-mode defocus is applied via ``HopkinsParams.defocus_nm``; the
    Gaussian path emulates defocus by widening the PSF sigma at the
    PVBand corners.
    """

    NAME = "openilt"
    SUPPORTS_CURVILINEAR = True
    RECEPTIVE_FIELD_PX = 0

    def __init__(
        self,
        iterations: int = 200,
        lr: float = 1.0,
        momentum: float = 0.9,
        pvb_weight: float = 0.5,
        resist_steepness: float = 50.0,
        forward_model: ForwardModelKind = "gaussian",
        corners: PVBandCorners | None = None,
        hopkins_params: HopkinsParams | None = None,
    ) -> None:
        self._iterations = iterations
        self._lr = lr
        self._momentum = momentum
        self._pvb_weight = pvb_weight
        self._resist_steepness = resist_steepness
        self._forward_model = forward_model
        self._corners = corners or PVBandCorners()
        self._hopkins_params = hopkins_params or HopkinsParams()
        self._cached_kernels_nom: torch.Tensor | None = None
        self._cached_weights_nom: torch.Tensor | None = None
        self._cached_kernels_def: torch.Tensor | None = None
        self._cached_weights_def: torch.Tensor | None = None
        self._cached_grid: int | None = None
        self._cached_defocus_nm: float | None = None
        self._cached_hopkins_params: HopkinsParams | None = None

    def _ensure_hopkins_kernels(
        self,
        grid_size: int,
        device: torch.device,
        defocus_nm: float,
        hopkins_params: HopkinsParams,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cache validity is keyed on the full (grid, device, defocus,
        # params) tuple — caller-supplied ``hopkins_params`` overrides
        # never mutate ``self._hopkins_params`` and a one-off override
        # cannot poison subsequent calls that use the default.
        if (
            self._cached_kernels_nom is None
            or self._cached_weights_nom is None
            or self._cached_kernels_def is None
            or self._cached_weights_def is None
            or self._cached_grid != grid_size
            or self._cached_defocus_nm != defocus_nm
            or self._cached_kernels_nom.device != device
            or self._cached_hopkins_params != hopkins_params
        ):
            kernels_nom, weights_nom = compute_socs_kernels(hopkins_params, grid_size, device)
            from dataclasses import replace

            def_params = replace(hopkins_params, defocus_nm=defocus_nm)
            kernels_def, weights_def = compute_socs_kernels(def_params, grid_size, device)
            self._cached_kernels_nom = kernels_nom
            self._cached_weights_nom = weights_nom
            self._cached_kernels_def = kernels_def
            self._cached_weights_def = weights_def
            self._cached_grid = grid_size
            self._cached_defocus_nm = defocus_nm
            self._cached_hopkins_params = hopkins_params
        assert self._cached_kernels_nom is not None
        assert self._cached_weights_nom is not None
        assert self._cached_kernels_def is not None
        assert self._cached_weights_def is not None
        return (
            self._cached_kernels_nom,
            self._cached_weights_nom,
            self._cached_kernels_def,
            self._cached_weights_def,
        )

    def _aerial(
        self,
        mask: torch.Tensor,
        *,
        forward_model: ForwardModelKind,
        dose: float,
        sigma_px: float,
        kernels: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if forward_model == "hopkins":
            assert kernels is not None and weights is not None
            return simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights, dose=dose)
        return simulate_aerial_image(mask, sigma_px=sigma_px, dose=dose)

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Optimize a mask via the SimpleILT L2 + PVBand objective.

        Args:
            design: Target design pattern, shape (H, W), binary.
            **kwargs: Optional overrides — ``iterations``, ``lr``,
                ``momentum``, ``pvb_weight``, ``forward_model``,
                ``corners``, ``hopkins_params``, ``device``.
        """
        target = design.detach().float()
        if target.ndim > 2:
            target = target.squeeze()

        iterations = int(kwargs.get("iterations", self._iterations))
        lr = float(kwargs.get("lr", self._lr))
        momentum = float(kwargs.get("momentum", self._momentum))
        pvb_weight = float(kwargs.get("pvb_weight", self._pvb_weight))
        forward_model: ForwardModelKind = kwargs.get("forward_model", self._forward_model)
        corners: PVBandCorners = kwargs.get("corners", self._corners)
        hopkins_params: HopkinsParams = kwargs.get("hopkins_params", self._hopkins_params)
        device = kwargs.get("device")

        if device is not None:
            target = target.to(device)

        kernels_nom: torch.Tensor | None = None
        weights_nom: torch.Tensor | None = None
        kernels_def: torch.Tensor | None = None
        weights_def: torch.Tensor | None = None
        if forward_model == "hopkins":
            (
                kernels_nom,
                weights_nom,
                kernels_def,
                weights_def,
            ) = self._ensure_hopkins_kernels(
                target.shape[0], target.device, corners.edge_defocus_nm, hopkins_params
            )

        # PixelInit: mask_logit starts at 2*target - 1 (per OpenILT README).
        # After sigmoid, mask ≈ 0 where target=0 and mask ≈ 0.73 where target=1,
        # which is the non-saturated starting point SimpleILT uses.
        mask_logit = (target * 2.0 - 1.0).clone().detach().requires_grad_(True)

        optimizer = torch.optim.SGD([mask_logit], lr=lr, momentum=momentum)

        best_loss = float("inf")
        best_mask: torch.Tensor = target.clone()
        last_l2_nom = float("inf")
        last_pvb = float("inf")

        for _ in range(iterations):
            optimizer.zero_grad()

            mask_continuous = torch.sigmoid(mask_logit)

            aerial_nom = self._aerial(
                mask_continuous,
                forward_model=forward_model,
                dose=corners.nom_dose,
                sigma_px=corners.nom_sigma_px,
                kernels=kernels_nom,
                weights=weights_nom,
            )
            aerial_max = self._aerial(
                mask_continuous,
                forward_model=forward_model,
                dose=corners.max_dose,
                sigma_px=corners.defocus_sigma_px,
                kernels=kernels_def,
                weights=weights_def,
            )
            aerial_min = self._aerial(
                mask_continuous,
                forward_model=forward_model,
                dose=corners.min_dose,
                sigma_px=corners.defocus_sigma_px,
                kernels=kernels_def,
                weights=weights_def,
            )

            resist_nom = differentiable_threshold(
                aerial_nom.float(), threshold=0.5, steepness=self._resist_steepness
            )
            resist_max = differentiable_threshold(
                aerial_max.float(), threshold=0.5, steepness=self._resist_steepness
            )
            resist_min = differentiable_threshold(
                aerial_min.float(), threshold=0.5, steepness=self._resist_steepness
            )

            l2_nom = functional.mse_loss(resist_nom, target)
            pvb = functional.mse_loss(resist_max, target) + functional.mse_loss(resist_min, target)
            loss = l2_nom + pvb_weight * pvb

            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_mask = (mask_continuous > 0.5).float().detach()
                last_l2_nom = l2_nom.item()
                last_pvb = pvb.item()

        return PredictionResult(
            mask=best_mask,
            metadata={
                "final_loss": best_loss,
                "l2_nom": last_l2_nom,
                "pvb_loss": last_pvb,
                "iterations": iterations,
                "forward_model": forward_model,
                "pvb_weight": pvb_weight,
            },
        )
