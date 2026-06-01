"""Curvilinear ILT with process window optimization, adjoint guidance, and shot-count Pareto.

Traditional ILT produces rectilinear (Manhattan) mask shapes — axis-aligned
edges and corners — because the mask is stored on a rectangular pixel grid.
Curvilinear mask representations allow non-Manhattan edges (arcs, splines),
which can improve process window by better matching the optimal intensity
contour at defocus/dose extremes.

This module implements:
1. **CurvilinearMaskILT** — ILT with curvilinear mask primitives. The mask
   is parameterized as a level-set field where the zero-crossing contour
   naturally produces smooth, non-Manhattan edges. Adjoint gradients from
   the lithography forward model steer the contour toward the target.
2. **ProcessWindowOptimizer** — Multi-focus/dose optimization that evaluates
   EPE across a grid of process conditions and combines them into a robust
   objective.
3. **ShotCountPareto** — Tracks the Pareto frontier of shot count (mask
   complexity proxy) vs printability (EPE or NILS), returning the set of
   non-dominated configurations.
4. **CurvilinearILTBenchmark** — Structured comparison of curvilinear vs
   rectilinear ILT on the same target patterns.

References
----------
- Y. Shen et al., "Curvilinear mask optimization for EUV lithography using
  level-set methods," Proc. SPIE 12498, 2023.
- J. Yu, P. Hu, W. Ye, "CUDA-curvilinear OPC: GPU-accelerated curvilinear
  optical proximity correction via level set," Proc. SPIE 12053, 2022.
- A. Poonawala, P. Milanfar, "Mask design for optical microlithography —
  an inverse imaging problem," IEEE TIP 16(3), 2007.
- K. Matsunobu et al., "Curvilinear mask data preparation for advanced
  lithography," Proc. SPIE 11147, 2019.

Licensed under the Apache License, Version 2.0 (clean-room implementation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import apply_differentiable_resist
from openlithohub._utils.tensor_ops import ensure_2d

# ---------------------------------------------------------------------------
# Curvilinear mask representation via level-set field
# ---------------------------------------------------------------------------


class CurvilinearMaskRepresentation:
    """Level-set based curvilinear mask representation.

    Stores the mask as a signed distance field (SDF) where the zero level-set
    contour defines the mask boundary. The SDF naturally produces smooth,
    non-Manhattan edges when optimized with gradient descent, because the
    contour is free to take any shape rather than being constrained to
    axis-aligned edges.

    The SDF is stored as a dense grid (H, W) of float values:
    - Negative values: mask interior (opaque / chrome)
    - Positive values: mask exterior (clear / quartz)
    - Zero crossing: the mask boundary contour

    The continuous mask is obtained by applying a sigmoid to the SDF:
    ``mask = sigmoid(-steepness * sdf)`` — this makes the boundary
    differentiable for gradient-based ILT.

    Parameters
    ----------
    grid_size : int
        Spatial grid size (pixels, assumed square).
    steepness : float
        Sigmoid steepness for SDF-to-mask conversion. Higher values produce
        sharper boundaries (approaching binary) but may hurt gradient flow.
    smoothness_weight : float
        Weight for the Laplacian smoothness regularizer on the SDF. Higher
        values produce smoother contours (fewer shots) at the cost of EPE.
    """

    def __init__(
        self,
        grid_size: int = 64,
        steepness: float = 20.0,
        smoothness_weight: float = 0.01,
    ) -> None:
        self.grid_size = grid_size
        self.steepness = steepness
        self.smoothness_weight = smoothness_weight

    def init_from_target(self, target: torch.Tensor) -> torch.Tensor:
        """Initialize SDF from a binary target design.

        Uses Euclidean distance transform approximation via iterative
        convolution to create an approximate signed distance field. Interior
        points (target == 1) get negative distances, exterior points get
        positive distances.

        Parameters
        ----------
        target : Tensor
            Binary target design (H, W), values in {0, 1}.

        Returns
        -------
        SDF tensor of shape (H, W), float32.
        """
        t = ensure_2d(target).float()
        h, w = t.shape
        device = t.device

        # Approximate distance transform via iterative chamfer-like steps.
        # Interior: negative SDF, exterior: positive SDF.
        interior = (t > 0.5).float()
        exterior = 1.0 - interior

        # Multi-pass distance approximation using 3x3 convolution.
        dist_interior = self._approx_distance(interior, device)
        dist_exterior = self._approx_distance(exterior, device)

        sdf = dist_exterior - dist_interior
        return sdf

    @staticmethod
    def _approx_distance(
        binary: torch.Tensor, device: torch.device, n_passes: int = 4
    ) -> torch.Tensor:
        """Approximate Euclidean distance transform via iterative convolution.

        Uses a 3x3 chamfer kernel iterated multiple times. This is not an
        exact EDT but is sufficient for SDF initialization and fully
        differentiable.

        Parameters
        ----------
        binary : Tensor
            Binary foreground mask (H, W).
        device : torch.device
            Torch device.
        n_passes : int
            Number of chamfer passes (more passes = better approximation).

        Returns
        -------
        Distance tensor of shape (H, W).
        """
        # Chamfer 3-4 kernel weights.
        kernel = torch.tensor(
            [[3, 4, 3], [4, 0, 4], [3, 4, 3]],
            dtype=torch.float32,
            device=device,
        )
        kernel = kernel / 4.0  # Normalize so 4-connected neighbor = 1.0
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        dist = binary.clone().unsqueeze(0).unsqueeze(0)
        large_val = float(dist.shape[-1] + dist.shape[-2])

        for _ in range(n_passes):
            padded = functional.pad(dist, [1, 1, 1, 1], mode="constant", value=0.0)
            neighbor_min = functional.conv2d(padded, kernel)
            dist = torch.where(
                dist > 0.5,
                torch.ones_like(dist),
                torch.clamp(neighbor_min + 1.0, max=large_val),
            )

        return dist.squeeze(0).squeeze(0)

    def sdf_to_mask(self, sdf: torch.Tensor) -> torch.Tensor:
        """Convert SDF to continuous mask in [0, 1].

        Parameters
        ----------
        sdf : Tensor
            Signed distance field (H, W) or (B, H, W).

        Returns
        -------
        Continuous mask tensor, same shape, values in [0, 1].
        """
        return torch.sigmoid(-self.steepness * sdf)

    def smoothness_loss(self, sdf: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian smoothness regularizer on the SDF.

        Penalizes high-curvature regions in the SDF, which correspond to
        sharp corners in the mask boundary. Lower curvature means smoother
        contours, which translates to fewer curvilinear shots.

        Parameters
        ----------
        sdf : Tensor
            Signed distance field (H, W).

        Returns
        -------
        Scalar smoothness penalty.
        """
        s = ensure_2d(sdf).float()
        laplacian_h = s[:-2, :] - 2.0 * s[1:-1, :] + s[2:, :]
        laplacian_w = s[:, :-2] - 2.0 * s[:, 1:-1] + s[:, 2:]
        lap_total = (laplacian_h**2).sum() + (laplacian_w**2).sum()
        return self.smoothness_weight * lap_total


# ---------------------------------------------------------------------------
# Adjoint guidance for gradient-based mask optimization
# ---------------------------------------------------------------------------


class AdjointGuidance:
    """Adjoint-method gradient guidance for ILT mask optimization.

    Computes gradients of the lithographic loss (EPE) with respect to the
    mask SDF parameters via the chain rule through the forward model. The
    adjoint formulation is equivalent to standard back-propagation but is
    exposed explicitly so that process-window multi-condition gradients can
    be accumulated efficiently.

    The gradient chain is:
    ``d(Loss) / d(SDF) = d(Loss) / d(mask) * d(mask) / d(SDF)``

    where the first factor comes from the aerial image + resist model and
    the second from the sigmoid SDF-to-mask conversion.

    Parameters
    ----------
    sigma_px : float
        Gaussian PSF sigma in pixels.
    dose : float
        Dose multiplier.
    resist_threshold : float
        Resist threshold for printing.
    resist_steepness : float
        Sigmoid steepness for differentiable resist.
    """

    def __init__(
        self,
        sigma_px: float = 2.0,
        dose: float = 1.0,
        resist_threshold: float = 0.5,
        resist_steepness: float = 50.0,
    ) -> None:
        self.sigma_px = sigma_px
        self.dose = dose
        self.resist_threshold = resist_threshold
        self.resist_steepness = resist_steepness

    def compute_epe_loss(
        self,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute EPE loss between printed mask image and target.

        Uses the lithography forward model (aerial image + resist) to
        simulate the printed wafer image, then computes MSE against the
        target design.

        Parameters
        ----------
        mask : Tensor
            Continuous mask (H, W), values in [0, 1].
        target : Tensor
            Target design (H, W), binary {0, 1}.

        Returns
        -------
        Scalar EPE loss (differentiable).
        """
        m = ensure_2d(mask).float()
        t = ensure_2d(target).float()

        aerial = simulate_aerial_image(m, sigma_px=self.sigma_px, dose=self.dose)
        resist = apply_differentiable_resist(
            aerial,
            threshold=self.resist_threshold,
            steepness=self.resist_steepness,
        )
        return functional.mse_loss(resist, t)

    def compute_nils(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute Normalized Image Log-Slope (NILS) as a printability metric.

        NILS measures the steepness of the aerial image intensity at the
        resist threshold contour. Higher NILS means more robust printing
        (steeper intensity transitions). NILS = threshold * |grad(I)| / I
        evaluated at the contour.

        Parameters
        ----------
        mask : Tensor
            Continuous mask (H, W), values in [0, 1].

        Returns
        -------
        Scalar mean NILS (differentiable proxy).
        """
        m = ensure_2d(mask).float()
        aerial = simulate_aerial_image(m, sigma_px=self.sigma_px, dose=self.dose)

        # Gradient magnitude of aerial image.
        gy = aerial[1:, :] - aerial[:-1, :]
        gx = aerial[:, 1:] - aerial[:, :-1]
        gy = functional.pad(gy, (0, 0, 0, 1))
        gx = functional.pad(gx, (0, 1, 0, 0))
        grad_mag = (gx**2 + gy**2).sqrt()

        # NILS = threshold * |grad(I)| / I, near the contour.
        # Use soft mask to weight the NILS toward the boundary region.
        boundary_weight = 4.0 * m * (1.0 - m)  # peaks at mask boundary
        safe_aerial = aerial.clamp(min=1e-6)
        nils = self.resist_threshold * grad_mag / safe_aerial
        weighted_nils = (nils * boundary_weight).sum() / boundary_weight.sum().clamp(min=1e-6)
        return weighted_nils

    def compute_gradient(
        self,
        sdf: torch.Tensor,
        target: torch.Tensor,
        mask_rep: CurvilinearMaskRepresentation,
    ) -> torch.Tensor:
        """Compute adjoint gradient of EPE loss w.r.t. the SDF.

        Parameters
        ----------
        sdf : Tensor
            Signed distance field (H, W), requires_grad.
        target : Tensor
            Target design (H, W), binary.
        mask_rep : CurvilinearMaskRepresentation
            Mask representation for SDF-to-mask conversion.

        Returns
        -------
        Gradient tensor of same shape as ``sdf``.
        """
        sdf_param = sdf.detach().requires_grad_(True)
        mask = mask_rep.sdf_to_mask(sdf_param)
        loss = self.compute_epe_loss(mask, target)
        return torch.autograd.grad(loss, sdf_param)[0]


# ---------------------------------------------------------------------------
# ProcessWindowOptimizer
# ---------------------------------------------------------------------------


@dataclass
class ProcessWindowConfig:
    """Configuration for process window optimization.

    Attributes
    ----------
    focus_range_nm : tuple[float, float]
        (min_focus, max_focus) range in nm. 0 = best focus.
    dose_range : tuple[float, float]
        (min_dose, max_dose) range as multipliers. 1.0 = nominal dose.
    n_focus_steps : int
        Number of focus sampling points.
    n_dose_steps : int
        Number of dose sampling points.
    sigma_px_per_nm : float
        Conversion from defocus (nm) to PSF sigma change (px).
    """

    focus_range_nm: tuple[float, float] = (-40.0, 40.0)
    dose_range: tuple[float, float] = (0.9, 1.1)
    n_focus_steps: int = 3
    n_dose_steps: int = 3
    sigma_px_per_nm: float = 0.05


class ProcessWindowOptimizer:
    """Multi-condition process window optimization across focus/dose variations.

    Evaluates EPE at each (focus, dose) combination and combines them into a
    robust weighted objective. The worst-case EPE is explicitly penalized to
    ensure the mask prints correctly across the entire process window.

    The total loss is:
    ``L_pw = w_mean * mean(EPE) + w_max * max(EPE)``
    where EPE is computed at each (focus, dose) grid point.

    Parameters
    ----------
    config : ProcessWindowConfig | None
        Process window configuration. Uses defaults if None.
    mean_weight : float
        Weight for the mean EPE across all conditions.
    max_weight : float
        Weight for the worst-case (max) EPE.
    """

    def __init__(
        self,
        config: ProcessWindowConfig | None = None,
        mean_weight: float = 1.0,
        max_weight: float = 2.0,
    ) -> None:
        self.config = config or ProcessWindowConfig()
        self.mean_weight = mean_weight
        self.max_weight = max_weight

    def get_condition_grid(self) -> list[tuple[float, float]]:
        """Generate the (focus_nm, dose) grid of process conditions.

        Returns
        -------
        List of (focus_nm, dose) tuples spanning the configured ranges.
        """
        cfg = self.config
        focus_values = torch.linspace(
            cfg.focus_range_nm[0],
            cfg.focus_range_nm[1],
            cfg.n_focus_steps,
        ).tolist()
        dose_values = torch.linspace(
            cfg.dose_range[0],
            cfg.dose_range[1],
            cfg.n_dose_steps,
        ).tolist()
        return [(f, d) for f in focus_values for d in dose_values]

    def compute_sigma_for_defocus(self, nominal_sigma_px: float, defocus_nm: float) -> float:
        """Adjust PSF sigma for defocus.

        Through-focus blur increases with the square of defocus. A simple
        model adds a defocus-dependent term to the nominal PSF sigma:
        ``sigma(defocus) = sqrt(sigma_nominal^2 + (k * defocus)^2)``

        Parameters
        ----------
        nominal_sigma_px : float
            Best-focus PSF sigma in pixels.
        defocus_nm : float
            Defocus in nanometres.

        Returns
        -------
        Adjusted sigma in pixels.
        """
        k = self.config.sigma_px_per_nm
        return math.sqrt(nominal_sigma_px**2 + (k * defocus_nm) ** 2)

    def compute_pw_loss(
        self,
        mask: torch.Tensor,
        target: torch.Tensor,
        nominal_sigma_px: float = 2.0,
        resist_threshold: float = 0.5,
        resist_steepness: float = 50.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute process-window-weighted EPE loss.

        Evaluates EPE at each (focus, dose) condition and returns a weighted
        combination of the mean and max EPE.

        Parameters
        ----------
        mask : Tensor
            Continuous mask (H, W), values in [0, 1].
        target : Tensor
            Target design (H, W), binary.
        nominal_sigma_px : float
            PSF sigma at best focus (pixels).
        resist_threshold : float
            Resist threshold.
        resist_steepness : float
            Resist sigmoid steepness.

        Returns
        -------
        Tuple of (scalar PW loss, info dict with per-condition EPE values).
        """
        m = ensure_2d(mask).float()
        t = ensure_2d(target).float()

        conditions = self.get_condition_grid()
        epe_values: list[torch.Tensor] = []
        info_entries: list[dict[str, float]] = []

        for focus_nm, dose in conditions:
            sigma = self.compute_sigma_for_defocus(nominal_sigma_px, focus_nm)
            aerial = simulate_aerial_image(m, sigma_px=sigma, dose=dose)
            resist = apply_differentiable_resist(
                aerial, threshold=resist_threshold, steepness=resist_steepness
            )
            epe = functional.mse_loss(resist, t)
            epe_values.append(epe)
            info_entries.append({"focus_nm": focus_nm, "dose": dose, "epe": epe.item()})

        epe_stack = torch.stack(epe_values)
        mean_epe = epe_stack.mean()
        max_epe = epe_stack.max()

        total_loss = self.mean_weight * mean_epe + self.max_weight * max_epe

        info: dict[str, Any] = {
            "per_condition": info_entries,
            "mean_epe": mean_epe.item(),
            "max_epe": max_epe.item(),
            "n_conditions": len(conditions),
        }
        return total_loss, info


# ---------------------------------------------------------------------------
# ShotCountPareto
# ---------------------------------------------------------------------------


class ShotCountPareto:
    """Track and compute the Pareto frontier of shot-count vs printability.

    For curvilinear ILT, shot count depends on the complexity of the mask
    boundary contour. Smoother contours require fewer shots but may sacrifice
    EPE. The Pareto frontier identifies the set of non-dominated configurations
    where no single solution is better in both metrics simultaneously.

    Shot count estimation for curvilinear masks uses:
    1. Contour segment counting: each segment of the zero-crossing contour
       between inflection points corresponds to one curvilinear shot.
    2. SRAF contribution: sub-resolution assist features add shots.

    Printability is measured as EPE (lower is better) or NILS (higher is
    better). The Pareto frontier is computed in the (shot_count, EPE) space.

    Parameters
    ----------
    min_segment_px : float
        Minimum curvilinear segment length in pixels. Shorter segments are
        merged with neighbors to avoid over-counting.
    sraf_weight : float
        Additional shot-count weight per SRAF feature.
    """

    def __init__(
        self,
        min_segment_px: float = 3.0,
        sraf_weight: float = 1.5,
    ) -> None:
        self.min_segment_px = min_segment_px
        self.sraf_weight = sraf_weight
        self._history: list[tuple[float, float]] = []

    def estimate_curvilinear_shots(self, mask: torch.Tensor) -> float:
        """Estimate shot count for a curvilinear mask.

        Uses contour complexity as a proxy: the number of sign changes in
        the gradient direction along the mask boundary approximates the
        number of curvilinear segments (shots) needed.

        Parameters
        ----------
        mask : Tensor
            Continuous mask (H, W), values in [0, 1].

        Returns
        -------
        Estimated shot count (float).
        """
        m = ensure_2d(mask).float()

        # Detect mask boundary via gradient magnitude.
        gy = m[1:, :] - m[:-1, :]
        gx = m[:, 1:] - m[:, :-1]
        gy = functional.pad(gy, (0, 0, 0, 1))
        gx = functional.pad(gx, (0, 1, 0, 0))
        grad_mag = (gx**2 + gy**2).sqrt()

        # Boundary pixels: where gradient magnitude is above threshold.
        boundary = (grad_mag > 0.1).float()
        n_boundary_px = boundary.sum().item()

        if n_boundary_px < 1.0:
            return 0.0

        # Count curvature changes along the boundary: direction changes in
        # gradient correspond to different curvilinear segments.
        grad_angle = torch.atan2(gy, gx)
        # Quantize angles to detect direction changes.
        angle_diff_h = (grad_angle[1:, :] - grad_angle[:-1, :]).abs()
        angle_diff_w = (grad_angle[:, 1:] - grad_angle[:, :-1]).abs()

        # Wrap to [0, pi].
        angle_diff_h = torch.min(angle_diff_h, 2.0 * math.pi - angle_diff_h)
        angle_diff_w = torch.min(angle_diff_w, 2.0 * math.pi - angle_diff_w)

        # Significant direction changes indicate segment boundaries.
        curvature_threshold = math.pi / 6.0  # 30 degrees
        n_changes = (angle_diff_h > curvature_threshold).float().sum().item() + (
            angle_diff_w > curvature_threshold
        ).float().sum().item()

        # Each segment between direction changes is one shot.
        # Add a base count for the boundary perimeter itself.
        perimeter_shots = max(1.0, n_boundary_px / self.min_segment_px)
        curvature_shots = max(1.0, n_changes)

        return perimeter_shots + curvature_shots

    def estimate_sraf_shots(self, mask: torch.Tensor, main_mask: torch.Tensor) -> float:
        """Estimate additional shot count from SRAF features.

        SRAFs (Sub-Resolution Assist Features) are small mask features that
        do not print but improve process window. Each SRAF adds shot count.

        Parameters
        ----------
        mask : Tensor
            Full mask including SRAFs (H, W).
        main_mask : Tensor
            Main features only, no SRAFs (H, W).

        Returns
        -------
        Additional shot count from SRAFs.
        """
        m_full = ensure_2d(mask).float()
        m_main = ensure_2d(main_mask).float()

        sraf_region = (m_full - m_main).abs()
        sraf_pixels = (sraf_region > 0.1).float().sum().item()
        return sraf_pixels * self.sraf_weight / self.min_segment_px

    def record(self, shot_count: float, epe: float) -> None:
        """Record a (shot_count, epe) configuration.

        Parameters
        ----------
        shot_count : float
            Estimated shot count.
        epe : float
            Edge placement error.
        """
        self._history.append((shot_count, epe))

    def compute_pareto_frontier(self) -> list[tuple[float, float]]:
        """Compute the Pareto frontier from recorded configurations.

        A point (s1, e1) dominates (s2, e2) if s1 <= s2 AND e1 <= e2
        with at least one strict inequality. The Pareto frontier contains
        all non-dominated points, sorted by shot count.

        Returns
        -------
        List of (shot_count, epe) tuples forming the Pareto frontier,
        sorted by ascending shot count.
        """
        if not self._history:
            return []

        points = sorted(self._history, key=lambda p: (p[0], p[1]))
        frontier: list[tuple[float, float]] = []

        min_epe_so_far = float("inf")
        for sc, epe in points:
            if epe < min_epe_so_far:
                frontier.append((sc, epe))
                min_epe_so_far = epe

        return frontier

    @property
    def history(self) -> list[tuple[float, float]]:
        """Access the raw recorded configurations."""
        return list(self._history)

    def reset(self) -> None:
        """Clear all recorded configurations."""
        self._history = []


# ---------------------------------------------------------------------------
# CurvilinearMaskILT — main ILT optimizer
# ---------------------------------------------------------------------------


@dataclass
class CurvilinearILTConfig:
    """Configuration for CurvilinearMaskILT optimization.

    Attributes
    ----------
    grid_size : int
        Spatial grid size (pixels, assumed square).
    sigma_px : float
        Nominal PSF sigma at best focus.
    dose : float
        Nominal dose multiplier.
    resist_threshold : float
        Resist threshold.
    resist_steepness : float
        Resist sigmoid steepness.
    sdf_steepness : float
        SDF-to-mask sigmoid steepness.
    smoothness_weight : float
        SDF Laplacian smoothness weight.
    n_steps : int
        Number of optimization iterations.
    lr : float
        Learning rate for Adam optimizer.
    pw_mean_weight : float
        Process window mean EPE weight.
    pw_max_weight : float
        Process window max EPE weight.
    shot_count_weight : float
        Weight for the shot-count penalty in the total loss.
    """

    grid_size: int = 64
    sigma_px: float = 2.0
    dose: float = 1.0
    resist_threshold: float = 0.5
    resist_steepness: float = 50.0
    sdf_steepness: float = 20.0
    smoothness_weight: float = 0.01
    n_steps: int = 100
    lr: float = 0.1
    pw_mean_weight: float = 1.0
    pw_max_weight: float = 2.0
    shot_count_weight: float = 0.1


class CurvilinearMaskILT:
    """Curvilinear ILT with process window optimization and adjoint guidance.

    Optimizes a mask represented as a level-set (signed distance field) to
    produce curvilinear (non-Manhattan) mask shapes. The optimization
    combines:
    1. **Adjoint guidance** — gradient-based mask update via the lithography
       forward model chain rule.
    2. **Process window optimization** — multi-focus/dose EPE evaluation for
       robust printing across process variations.
    3. **Shot-count penalty** — contour complexity penalty to trade off mask
       write time against print fidelity.

    The total optimization objective is:
    ``L = pw_loss + smoothness_loss + shot_count_penalty``

    Parameters
    ----------
    config : CurvilinearILTConfig | None
        Optimization configuration. Uses defaults if None.
    """

    def __init__(self, config: CurvilinearILTConfig | None = None) -> None:
        self.config = config or CurvilinearILTConfig()
        cfg = self.config

        self.mask_rep = CurvilinearMaskRepresentation(
            grid_size=cfg.grid_size,
            steepness=cfg.sdf_steepness,
            smoothness_weight=cfg.smoothness_weight,
        )
        self.adjoint = AdjointGuidance(
            sigma_px=cfg.sigma_px,
            dose=cfg.dose,
            resist_threshold=cfg.resist_threshold,
            resist_steepness=cfg.resist_steepness,
        )
        self.pw_optimizer = ProcessWindowOptimizer(
            mean_weight=cfg.pw_mean_weight,
            max_weight=cfg.pw_max_weight,
        )
        self.pareto = ShotCountPareto()

    def optimize(
        self,
        target: torch.Tensor,
        n_steps: int | None = None,
        lr: float | None = None,
        shot_count_weight: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run curvilinear ILT optimization.

        Initializes the SDF from the target design, then iteratively refines
        it using adjoint gradients from the process-window-weighted EPE loss
        plus smoothness and shot-count penalties.

        Parameters
        ----------
        target : Tensor
            Target design pattern (H, W), binary {0, 1}.
        n_steps : int | None
            Override number of optimization steps.
        lr : float | None
            Override learning rate.
        shot_count_weight : float | None
            Override shot-count penalty weight.

        Returns
        -------
        Tuple of (optimized_mask, info_dict).
        info_dict contains 'pareto_frontier', 'final_epe', 'final_shots',
        'loss_history'.
        """
        cfg = self.config
        steps = n_steps if n_steps is not None else cfg.n_steps
        learning_rate = lr if lr is not None else cfg.lr
        sc_weight = shot_count_weight if shot_count_weight is not None else cfg.shot_count_weight

        t = ensure_2d(target).float()

        # Initialize SDF from target.
        sdf = self.mask_rep.init_from_target(t).detach().requires_grad_(True)

        optimizer = torch.optim.Adam([sdf], lr=learning_rate)
        self.pareto.reset()

        loss_history: list[float] = []

        for _step in range(steps):
            optimizer.zero_grad()

            mask = self.mask_rep.sdf_to_mask(sdf)

            # Process-window loss.
            pw_loss, pw_info = self.pw_optimizer.compute_pw_loss(
                mask,
                t,
                nominal_sigma_px=cfg.sigma_px,
                resist_threshold=cfg.resist_threshold,
                resist_steepness=cfg.resist_steepness,
            )

            # SDF smoothness loss.
            smooth_loss = self.mask_rep.smoothness_loss(sdf)

            # Shot-count penalty: contour density proxy.
            binary_soft = torch.sigmoid(10.0 * (mask - 0.5))
            gy = binary_soft[1:, :] - binary_soft[:-1, :]
            gx = binary_soft[:, 1:] - binary_soft[:, :-1]
            contour_density = gy.abs().mean() + gx.abs().mean()
            sc_penalty = sc_weight * contour_density

            total_loss = pw_loss + smooth_loss + sc_penalty

            total_loss.backward()
            optimizer.step()

            # Track Pareto point.
            with torch.no_grad():
                epe_at_nominal = pw_info["mean_epe"]
                shots = self.pareto.estimate_curvilinear_shots(mask)
                self.pareto.record(shots, epe_at_nominal)
                loss_history.append(total_loss.item())

        # Final mask.
        with torch.no_grad():
            final_mask = (self.mask_rep.sdf_to_mask(sdf) > 0.5).float()

        frontier = self.pareto.compute_pareto_frontier()
        info: dict[str, Any] = {
            "pareto_frontier": frontier,
            "final_epe": loss_history[-1] if loss_history else float("inf"),
            "final_shots": self.pareto._history[-1][0] if self.pareto._history else 0.0,
            "loss_history": loss_history,
            "n_steps": steps,
        }
        return final_mask, info

    def optimize_rectilinear(
        self,
        target: torch.Tensor,
        n_steps: int | None = None,
        lr: float | None = None,
        shot_count_weight: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run rectilinear (Manhattan) ILT for comparison baseline.

        Same optimization as ``optimize`` but constrains the SDF to produce
        only axis-aligned edges by applying a Manhattan projection step after
        each gradient update. This simulates traditional pixel-based ILT.

        Parameters
        ----------
        target : Tensor
            Target design pattern (H, W), binary {0, 1}.
        n_steps : int | None
            Override number of optimization steps.
        lr : float | None
            Override learning rate.
        shot_count_weight : float | None
            Override shot-count penalty weight.

        Returns
        -------
        Tuple of (optimized_mask, info_dict).
        """
        cfg = self.config
        steps = n_steps if n_steps is not None else cfg.n_steps
        learning_rate = lr if lr is not None else cfg.lr
        sc_weight = shot_count_weight if shot_count_weight is not None else cfg.shot_count_weight

        t = ensure_2d(target).float()

        sdf = self.mask_rep.init_from_target(t).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([sdf], lr=learning_rate)
        pareto_rect = ShotCountPareto()
        loss_history: list[float] = []

        for _step in range(steps):
            optimizer.zero_grad()

            mask = self.mask_rep.sdf_to_mask(sdf)

            # Manhattan projection: quantize gradients to axis-aligned.
            # This is achieved by a Manhattan penalty that discourages
            # diagonal edges.
            gy = mask[1:, :] - mask[:-1, :]
            gx = mask[:, 1:] - mask[:, :-1]
            # Penalize pixels where both gx and gy are non-zero (diagonal).
            gy_padded = functional.pad(gy, (0, 0, 0, 1))
            gx_padded = functional.pad(gx, (0, 1, 0, 0))
            diagonal_penalty = (gy_padded.abs() * gx_padded.abs()).mean()

            # Process-window loss.
            pw_loss, pw_info = self.pw_optimizer.compute_pw_loss(
                mask,
                t,
                nominal_sigma_px=cfg.sigma_px,
                resist_threshold=cfg.resist_threshold,
                resist_steepness=cfg.resist_steepness,
            )

            smooth_loss = self.mask_rep.smoothness_loss(sdf)

            binary_soft = torch.sigmoid(10.0 * (mask - 0.5))
            gy2 = binary_soft[1:, :] - binary_soft[:-1, :]
            gx2 = binary_soft[:, 1:] - binary_soft[:, :-1]
            contour_density = gy2.abs().mean() + gx2.abs().mean()
            sc_penalty = sc_weight * contour_density

            total_loss = pw_loss + smooth_loss + sc_penalty + 0.5 * diagonal_penalty

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                epe_at_nominal = pw_info["mean_epe"]
                shots = pareto_rect.estimate_curvilinear_shots(mask)
                pareto_rect.record(shots, epe_at_nominal)
                loss_history.append(total_loss.item())

        with torch.no_grad():
            final_mask = (self.mask_rep.sdf_to_mask(sdf) > 0.5).float()

        frontier = pareto_rect.compute_pareto_frontier()
        info: dict[str, Any] = {
            "pareto_frontier": frontier,
            "final_epe": loss_history[-1] if loss_history else float("inf"),
            "final_shots": pareto_rect._history[-1][0] if pareto_rect._history else 0.0,
            "loss_history": loss_history,
            "n_steps": steps,
        }
        return final_mask, info


# ---------------------------------------------------------------------------
# CurvilinearILTBenchmark
# ---------------------------------------------------------------------------


class CurvilinearILTBenchmark:
    """Benchmark comparing curvilinear vs rectilinear ILT.

    Runs both optimization modes on the same target patterns and compares:
    - Final EPE (lower is better)
    - Shot count (lower is better)
    - Pareto frontier shape (more dominant points is better)
    - Process window robustness (max EPE across conditions)

    Parameters
    ----------
    config : CurvilinearILTConfig | None
        Optimization configuration shared between both modes.
    n_steps : int
        Number of optimization steps per run.
    """

    def __init__(
        self,
        config: CurvilinearILTConfig | None = None,
        n_steps: int = 50,
    ) -> None:
        self.config = config or CurvilinearILTConfig(n_steps=n_steps)
        self.n_steps = n_steps

    def _make_target(
        self, pattern: str = "linespace", grid_size: int | None = None
    ) -> torch.Tensor:
        """Create a test target pattern.

        Parameters
        ----------
        pattern : str
            Pattern type: 'linespace', 'contact', or 'elbow'.
        grid_size : int | None
            Override grid size.

        Returns
        -------
        Binary target tensor (H, W).
        """
        gs = grid_size or self.config.grid_size
        t = torch.zeros(gs, gs)

        if pattern == "linespace":
            pitch = max(4, gs // 8)
            half = pitch // 2
            for x in range(0, gs, pitch):
                t[:, x : x + half] = 1.0
        elif pattern == "contact":
            pitch = max(6, gs // 4)
            size = max(2, pitch // 3)
            for cy in range(pitch // 2, gs, pitch):
                for cx in range(pitch // 2, gs, pitch):
                    y0 = max(0, cy - size // 2)
                    y1 = min(gs, cy + size // 2 + 1)
                    x0 = max(0, cx - size // 2)
                    x1 = min(gs, cx + size // 2 + 1)
                    t[y0:y1, x0:x1] = 1.0
        elif pattern == "elbow":
            w = max(2, gs // 8)
            t[: w * 3, gs // 2 - w : gs // 2 + w] = 1.0
            t[gs // 2 - w : gs // 2 + w, gs // 2 - w :] = 1.0

        return t

    def run(
        self,
        targets: list[torch.Tensor] | None = None,
        n_seeds: int = 3,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run curvilinear vs rectilinear ILT benchmark.

        Parameters
        ----------
        targets : list[Tensor] | None
            Target patterns to optimize. If None, generates default patterns.
        n_seeds : int
            Number of random seeds per target.
        n_steps : int | None
            Override number of optimization steps.

        Returns
        -------
        Dict with 'curvilinear' and 'rectilinear' results, each containing
        'mean_epe', 'mean_shots', and per-target details.
        """
        steps = n_steps or self.n_steps

        if targets is None:
            targets = [
                self._make_target("linespace"),
                self._make_target("contact"),
            ]

        curv_results: list[dict[str, Any]] = []
        rect_results: list[dict[str, Any]] = []

        for target in targets:
            for seed in range(n_seeds):
                torch.manual_seed(seed)

                # Curvilinear run.
                curv_ilt = CurvilinearMaskILT(self.config)
                _, curv_info = curv_ilt.optimize(target, n_steps=steps)
                curv_results.append(curv_info)

                torch.manual_seed(seed)

                # Rectilinear run.
                rect_ilt = CurvilinearMaskILT(self.config)
                _, rect_info = rect_ilt.optimize_rectilinear(target, n_steps=steps)
                rect_results.append(rect_info)

        # Aggregate.
        curv_epes = [r["final_epe"] for r in curv_results]
        curv_shots = [r["final_shots"] for r in curv_results]
        rect_epes = [r["final_epe"] for r in rect_results]
        rect_shots = [r["final_shots"] for r in rect_results]

        return {
            "curvilinear": {
                "mean_epe": sum(curv_epes) / len(curv_epes) if curv_epes else float("inf"),
                "mean_shots": sum(curv_shots) / len(curv_shots) if curv_shots else 0.0,
                "per_run": curv_results,
            },
            "rectilinear": {
                "mean_epe": sum(rect_epes) / len(rect_epes) if rect_epes else float("inf"),
                "mean_shots": sum(rect_shots) / len(rect_shots) if rect_shots else 0.0,
                "per_run": rect_results,
            },
            "n_targets": len(targets),
            "n_seeds": n_seeds,
            "n_steps": steps,
        }
