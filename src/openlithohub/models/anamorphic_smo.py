"""High-NA EUV anamorphic Source-Mask Optimization with shot-count cost.

High-NA EUV scanners (ASML EXE:5000 class) use an anamorphic projection
optics with different magnifications in x (scanning) and y (cross-scan)
to maintain acceptable chief-ray angles while pushing NA from 0.33 to
0.55. The asymmetric magnification (typically 4x/8x) creates
direction-dependent imaging that must be modelled in SMO.

Central obscuration from the off-axis reflective projection optics
produces a pupil-plane hole that further modulates the point spread
function, degrading contrast for low-frequency content.

The mask-3D shadow effect arises because EUV mask absorber features have
finite thickness (~50-70 nm) and the chief ray strikes the mask at
oblique incidence (~6 deg for standard NA, up to ~9 deg for high-NA).
This causes asymmetric CD bias and pattern shift that depend on feature
orientation relative to the incidence plane.

Shot-count optimization penalises curvilinear mask complexity to keep
mask write time within production budgets. The gradient proxy uses
contour density (gradient magnitude of the soft-binarised mask) which
is differentiable even though the actual shot-count estimator is not.

References
----------
- M. van de Kerkhof et al., "High-NA EUV lithography: realizing
  Moore's law in the next decade," Proc. SPIE 12498, 2023.
- Synopsys / imec High-NA EUV joint reviews, SPIE 2024-2025.
- Science Tokyo High-NA EUV STCC, 2026-05.
- ASML High-NA anamorphic imaging fundamentals, white paper 2024.

Licensed under the Apache License, Version 2.0 (clean-room implementation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from openlithohub._constants import WAVELENGTH_EUV_NM as _WAVELENGTH_EUV_NM
from openlithohub._utils.tensor_ops import ensure_2d
from openlithohub.benchmark.metrics.shot_count import estimate_shot_count

# ---------------------------------------------------------------------------
# AnamorphicParams
# ---------------------------------------------------------------------------


@dataclass
class AnamorphicParams:
    """High-NA EUV anamorphic imaging parameters.

    Attributes:
        na_x: Numerical aperture in x (scanning direction).
        na_y: Numerical aperture in y (cross-scan direction).
        mag_x: Magnification in x. Standard high-NA: 4x.
        mag_y: Magnification in y. Standard high-NA: 8x.
        central_obscuration_ratio: Ratio of obscuration radius to pupil
            radius. Off-axis reflective optics block the central pupil.
        wavelength_nm: Exposure wavelength in nanometres (13.5 for EUV).
        pixel_size_nm: Physical pixel pitch in nanometres.
    """

    na_x: float = 0.55
    na_y: float = 0.55
    mag_x: float = 4.0
    mag_y: float = 8.0
    central_obscuration_ratio: float = 0.2
    wavelength_nm: float = _WAVELENGTH_EUV_NM
    pixel_size_nm: float = 1.0


# ---------------------------------------------------------------------------
# AnamorphicImaging
# ---------------------------------------------------------------------------


class AnamorphicImaging:
    """High-NA EUV imaging model with anamorphic magnification and central
    obscuration.

    The PSF is constructed from a pupil function that accounts for:
    1. Anamorphic NA (different cutoffs in x and y).
    2. Central obscuration (annular pupil).
    3. Mask-to-wafer anamorphic magnification scaling.

    The aerial image is computed via FFT convolution with the anamorphic
    PSF, preserving differentiability for gradient-based SMO.

    Optionally accepts polarization and triple-beam illumination to apply
    first-order polarization corrections to the aerial image.
    """

    def __init__(
        self,
        params: AnamorphicParams | None = None,
        polarization: PolarizationState | None = None,
        triple_beam: TripleBeamIllumination | None = None,
    ) -> None:
        self.params = params or AnamorphicParams()
        self.polarization = polarization
        self.triple_beam = triple_beam

    def compute_psf(self, grid_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Compute the anamorphic point spread function.

        Builds a pupil function with elliptical NA and central obscuration,
        then inverse-FFTs to get the spatial PSF. The PSF is normalised
        so that an open-frame mask produces unit aerial intensity.

        Args:
            grid_size: Spatial grid size (pixels).
            device: Torch device.

        Returns:
            PSF tensor of shape (grid_size, grid_size), float32, normalised.
        """
        if device is None:
            device = torch.device("cpu")

        p = self.params
        freq = torch.fft.fftfreq(grid_size, d=p.pixel_size_nm, device=device).float()
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")

        f_cutoff_x = p.na_x / p.wavelength_nm
        f_cutoff_y = p.na_y / p.wavelength_nm

        # Elliptical pupil normalised to [0, 1] within the NA ellipse.
        r_norm = torch.sqrt((fx / f_cutoff_x) ** 2 + (fy / f_cutoff_y) ** 2)

        # Annular pupil: pass between obscuration ratio and 1.0.
        outer_mask = r_norm <= 1.0
        inner_mask = r_norm >= p.central_obscuration_ratio
        pupil = (outer_mask & inner_mask).float()

        # PSF = |IFFT(pupil)|^2 (incoherent imaging).
        field = torch.fft.ifft2(pupil.to(torch.complex64))
        psf = field.real**2 + field.imag**2
        psf_sum = psf.sum()
        if psf_sum > 0:
            psf = psf / psf_sum

        # Apply anamorphic magnification scaling: the mask-to-wafer
        # demagnification differs in x and y. The PSF on the mask grid
        # must be stretched by the magnification ratio.
        if p.mag_x != p.mag_y:
            psf = self._anamorphic_scale(psf, p.mag_x / p.mag_y)

        return psf

    @staticmethod
    def _anamorphic_scale(psf: torch.Tensor, scale_y: float) -> torch.Tensor:
        """Scale PSF along one axis to account for anamorphic magnification.

        Uses bilinear interpolation to stretch/compress the PSF.

        Args:
            psf: Square PSF tensor (N, N).
            scale_y: Scaling factor in y relative to x.

        Returns:
            Scaled PSF tensor of the same spatial size.
        """
        if abs(scale_y - 1.0) < 1e-6:
            return psf

        n = psf.shape[0]
        # Build sampling grid centered at origin.
        coords = torch.linspace(-1.0, 1.0, n, device=psf.device)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")

        # Scale y coordinates.
        gy_scaled = gy / scale_y
        gy_scaled = gy_scaled.clamp(-1.0, 1.0)

        grid = torch.stack([gx, gy_scaled], dim=-1).unsqueeze(0)  # (1, N, N, 2)

        inp = psf.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        scaled = functional.grid_sample(
            inp,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        result = scaled.squeeze(0).squeeze(0)

        # Re-normalise.
        s = result.sum()
        if s > 0:
            result = result / s
        return result

    def simulate_aerial(self, mask: torch.Tensor) -> torch.Tensor:
        """Simulate anamorphic aerial image.

        Args:
            mask: Mask tensor (H, W), values in [0, 1].

        Returns:
            Aerial image tensor of shape (H, W).
        """
        m = ensure_2d(mask).float()
        h, w = m.shape
        if h != w:
            raise ValueError(f"Expected square mask; got {h}x{w}")

        psf = self.compute_psf(h, m.device)

        # Circular convolution via FFT.
        mask_c = m.to(torch.complex64)
        psf_c = psf.to(torch.complex64)
        aerial_f = torch.fft.fft2(mask_c) * torch.fft.fft2(torch.fft.ifftshift(psf_c))
        aerial = torch.fft.ifft2(aerial_f).real

        aerial = aerial.clamp(min=0.0)

        # Apply polarization correction if provided.
        if self.polarization is not None:
            na = max(self.params.na_x, self.params.na_y)
            factor = self.polarization.contrast_factor(na, self.params.wavelength_nm)
            aerial = aerial * factor

        # Apply triple-beam modulation if provided.
        if self.triple_beam is not None:
            aerial = self.triple_beam.modify_aerial_image(aerial)

        return aerial.clamp(min=0.0)

    def mask_3d_shadow_correction(
        self, mask: torch.Tensor, incident_angle_deg: float = 6.0
    ) -> torch.Tensor:
        """First-order mask-3D shadow correction.

        The oblique illumination at EUV wavelengths causes a lateral shift
        and CD bias proportional to absorber thickness and the tangent of
        the chief-ray angle. This correction applies a direction-dependent
        shift to the mask pattern.

        Args:
            mask: Mask tensor (H, W).
            incident_angle_deg: Chief-ray angle of incidence at mask (degrees).

        Returns:
            Corrected mask tensor of shape (H, W).
        """
        m = ensure_2d(mask).float()
        theta = math.radians(incident_angle_deg)

        # Shadow shift in pixels: proportional to absorber thickness / pixel size
        # times tan(theta). Using typical absorber thickness ~50 nm.
        absorber_nm = 50.0
        shift_px = absorber_nm * math.tan(theta) / self.params.pixel_size_nm

        if abs(shift_px) < 1e-4:
            return m

        # Apply directional shift via phase ramp in Fourier domain.
        h, w = m.shape
        freq_x = torch.fft.fftfreq(w, device=m.device)
        freq_y = torch.fft.fftfreq(h, device=m.device)

        # Anamorphic magnification makes the shadow effect asymmetric.
        shift_x = shift_px / self.params.mag_x
        shift_y = shift_px / self.params.mag_y

        m_f = torch.fft.fft2(m.to(torch.complex64))
        fx, fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        phase = torch.exp(-2j * math.pi * (fx * shift_x + fy * shift_y)).to(torch.complex64)
        shifted = torch.fft.ifft2(m_f * phase).real

        # Blend with original to model partial shadow transmission.
        shadow_attenuation = math.exp(-0.5 * absorber_nm / self.params.wavelength_nm)
        corrected = shadow_attenuation * shifted + (1.0 - shadow_attenuation) * m
        return corrected.clamp(0.0, 1.0)

    def compute_epe(self, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge placement error between mask and target.

        EPE measures the distance from each contour point of the printed
        image to the nearest target contour point. This implementation
        uses the L2 difference of the aerial images as a differentiable
        proxy.

        Args:
            mask: Optimised mask tensor (H, W).
            target: Target design tensor (H, W).

        Returns:
            Scalar EPE loss (mean squared difference of aerial images).
        """
        aerial_mask = self.simulate_aerial(mask)
        aerial_target = self.simulate_aerial(target)
        return ((aerial_mask - aerial_target) ** 2).mean()


# ---------------------------------------------------------------------------
# ShotCountCost
# ---------------------------------------------------------------------------


class ShotCountCost:
    """Differentiable shot-count penalty for curvilinear masks.

    Provides a gradient proxy via contour density (gradient magnitude of
    the soft-binarised mask) so that gradient-based SMO can trade off
    mask complexity against lithographic fidelity.

    The actual shot count is estimated via
    :func:`estimate_shot_count`, but the gradient uses the differentiable
    contour-density proxy from the morphology module.
    """

    def __init__(self, weight: float = 0.01, writer_type: str = "mbmw") -> None:
        self.weight = weight
        self.writer_type = writer_type

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute weighted shot-count cost.

        Args:
            mask: Continuous mask tensor (H, W), values in [0, 1].

        Returns:
            Scalar cost tensor (differentiable).
        """
        m = ensure_2d(mask).float()

        # Differentiable contour density proxy.
        binary_soft = torch.sigmoid(20.0 * (m - 0.5))
        gy = binary_soft[1:, :] - binary_soft[:-1, :]
        gx = binary_soft[:, 1:] - binary_soft[:, :-1]
        gy = functional.pad(gy, (0, 0, 0, 1))
        gx = functional.pad(gx, (0, 1, 0, 0))
        contour_density = (gy.abs() + gx.abs()).mean()

        return self.weight * contour_density

    def evaluate(self, mask: torch.Tensor) -> dict[str, Any]:
        """Evaluate actual shot count (non-differentiable).

        Args:
            mask: Binary or continuous mask tensor (H, W).

        Returns:
            Dict with 'shot_count' and 'estimated_write_time_s'.
        """
        m = ensure_2d(mask)
        return estimate_shot_count(
            m,
            writer_type=self.writer_type,
            pixel_size_nm=1.0,
        )


# ---------------------------------------------------------------------------
# AnamorphicSMO
# ---------------------------------------------------------------------------


class AnamorphicSMO:
    """Joint Source-Mask Optimization for high-NA EUV with anamorphic imaging.

    Optimises both the mask and a parametric source representation to
    minimise a multi-objective cost:

        total_cost = EPE_loss + PVB_loss + shot_count_cost

    The Pareto frontier between fidelity and shot count is tracked at each
    optimisation step.
    """

    def __init__(
        self,
        params: AnamorphicParams | None = None,
        shot_count_weight: float = 0.01,
    ) -> None:
        self.params = params or AnamorphicParams()
        self.imaging = AnamorphicImaging(self.params)
        self.shot_count = ShotCountCost(weight=shot_count_weight)
        self.pareto_history: list[tuple[float, float]] = []

    def optimize_source_mask(
        self,
        target: torch.Tensor,
        n_steps: int = 200,
        lr: float = 0.05,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run joint source-mask optimisation.

        The mask is initialised as the target design and iteratively
        refined. The source is represented as a small set of parameters
        that control the anamorphic sigma and central obscuration ratio.

        Args:
            target: Target design pattern (H, W), binary {0, 1}.
            n_steps: Number of optimisation steps.
            lr: Learning rate for Adam.

        Returns:
            Tuple of (optimised_mask, info_dict).
            info_dict contains 'pareto_history', 'final_epe', 'final_shot_cost'.
        """
        t = ensure_2d(target).float()
        device = t.device

        # Initialise mask as target with requires_grad.
        mask_param = t.clone().detach().requires_grad_(True)

        # Source parameters: anamorphic sigma_x, sigma_y.
        source_param = torch.tensor([0.7, 0.7], device=device, requires_grad=True)

        optimizer = torch.optim.Adam([mask_param, source_param], lr=lr)

        self.pareto_history = []

        for _step in range(n_steps):
            optimizer.zero_grad()

            # Clamp mask to [0, 1] via sigmoid proxy.
            mask_opt = torch.sigmoid(mask_param)

            # Source-controlled anamorphic PSF.
            self.params.na_x = 0.55 * source_param[0].sigmoid()
            self.params.na_y = 0.55 * source_param[1].sigmoid()
            self.imaging = AnamorphicImaging(self.params)

            # Forward model.
            aerial = self.imaging.simulate_aerial(mask_opt)

            # EPE loss: MSE between aerial image of mask and target.
            aerial_target = self.imaging.simulate_aerial(t)
            epe_loss = ((aerial - aerial_target) ** 2).mean()

            # PVB proxy loss: penalise gradient magnitude ( encourages
            # steep edges = narrow process window band).
            gy = aerial[1:, :] - aerial[:-1, :]
            gx = aerial[:, 1:] - aerial[:, :-1]
            pvb_loss = gy.abs().mean() + gx.abs().mean()

            # Shot-count cost.
            sc_cost = self.shot_count.forward(mask_opt)

            total_loss = epe_loss + 0.1 * pvb_loss + sc_cost

            total_loss.backward()
            optimizer.step()

            # Track Pareto point.
            with torch.no_grad():
                fidelity = epe_loss.item()
                sc_val = sc_cost.item()
                self.pareto_history.append((fidelity, sc_val))

        # Binarise final mask.
        with torch.no_grad():
            final_mask = (torch.sigmoid(mask_param) > 0.5).float()

        info: dict[str, Any] = {
            "pareto_history": list(self.pareto_history),
            "final_epe": self.pareto_history[-1][0] if self.pareto_history else float("inf"),
            "final_shot_cost": self.pareto_history[-1][1] if self.pareto_history else 0.0,
            "source_params": source_param.detach().sigmoid().tolist(),
        }
        return final_mask, info


# ---------------------------------------------------------------------------
# AnamorphicSMOBenchmark
# ---------------------------------------------------------------------------


class AnamorphicSMOBenchmark:
    """Compare anamorphic-aware vs isotropic SMO.

    Runs the same target pattern through both an anamorphic SMO (different
    NA/mag in x and y) and an isotropic SMO (same NA/mag in both axes)
    and reports the comparison.
    """

    def __init__(self, grid_size: int = 64, n_steps: int = 50, lr: float = 0.05) -> None:
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.lr = lr

    def _make_target(self) -> torch.Tensor:
        """Create a simple target pattern (dense line-space)."""
        t = torch.zeros(self.grid_size, self.grid_size)
        pitch = max(4, self.grid_size // 8)
        half = pitch // 2
        for x in range(0, self.grid_size, pitch):
            t[:, x : x + half] = 1.0
        return t

    def run(self, n_seeds: int = 3) -> dict[str, Any]:
        """Run comparison benchmark.

        Args:
            n_seeds: Number of random seeds to average over.

        Returns:
            Dict with 'anamorphic' and 'isotropic' results, each containing
            'mean_epe', 'mean_shot_cost', and per-seed details.
        """
        target = self._make_target()

        aniso_params = AnamorphicParams(na_x=0.55, na_y=0.55, mag_x=4.0, mag_y=8.0)
        iso_params = AnamorphicParams(na_x=0.55, na_y=0.55, mag_x=4.0, mag_y=4.0)

        results: dict[str, Any] = {"anamorphic": [], "isotropic": []}

        for seed in range(n_seeds):
            torch.manual_seed(seed)

            # Anamorphic run.
            smo_aniso = AnamorphicSMO(aniso_params, shot_count_weight=0.01)
            _, info_aniso = smo_aniso.optimize_source_mask(
                target,
                n_steps=self.n_steps,
                lr=self.lr,
            )
            results["anamorphic"].append(info_aniso)

            torch.manual_seed(seed)

            # Isotropic run.
            smo_iso = AnamorphicSMO(iso_params, shot_count_weight=0.01)
            _, info_iso = smo_iso.optimize_source_mask(
                target,
                n_steps=self.n_steps,
                lr=self.lr,
            )
            results["isotropic"].append(info_iso)

        # Aggregate.
        for key in ("anamorphic", "isotropic"):
            epes = [r["final_epe"] for r in results[key]]
            scs = [r["final_shot_cost"] for r in results[key]]
            results[f"mean_{key}_epe"] = sum(epes) / len(epes)
            results[f"mean_{key}_shot_cost"] = sum(scs) / len(scs)

        return results


# ---------------------------------------------------------------------------
# Half-field stitching (High-NA EUV)
# ---------------------------------------------------------------------------


class HalfFieldStitching:
    """Half-field stitching boundary model for High-NA EUV lithography.

    High-NA EUV scanners (ASML EXE:5000 class) have a reduced field size
    (26 mm x 16.5 mm vs 26 mm x 33 mm for standard NA) due to the
    anamorphic optics. To print a full-field die, two half-field exposures
    are stitched together at the wafer, creating a stitching boundary that
    can introduce dose/intensity discontinuities and edge placement errors.

    This class models the stitching boundary effects including:
    1. Dose discontinuity from misalignment between the two half-fields.
    2. Differentiable EPE penalty at the stitching boundary for ILT/SMO.

    Parameters
    ----------
    overlap_px : int
        Width of the overlap region between the two half-fields (pixels).
    max_misalignment_px : float
        Maximum allowed misalignment in pixels.

    References
    ----------
    - imec / Vandenberghe, "The case for High-NA EUV", Semicon Korea 2026.
    - M. van de Kerkhof et al., "High-NA EUV lithography: realizing
      Moore's law in the next decade," Proc. SPIE 12498, 2023.
    """

    def __init__(self, overlap_px: int = 20, max_misalignment_px: float = 2.0) -> None:
        if overlap_px < 1:
            raise ValueError(f"overlap_px must be >= 1, got {overlap_px}")
        if max_misalignment_px < 0.0:
            raise ValueError(f"max_misalignment_px must be >= 0, got {max_misalignment_px}")
        self.overlap_px = overlap_px
        self.max_misalignment_px = max_misalignment_px

    def compute_stitching_boundary(
        self, mask: torch.Tensor, misalignment_px: float
    ) -> torch.Tensor:
        """Simulate the stitching boundary with dose/intensity discontinuity.

        Creates a blending weight map that models the overlap region between
        two half-fields. When perfectly aligned, the blend is smooth across
        the boundary. Misalignment introduces a dose jump proportional to
        the lateral shift.

        The boundary is placed at the horizontal centre of the mask (the
        cross-scan stitching direction for High-NA EUV half-fields).

        Parameters
        ----------
        mask : Tensor
            Mask tensor (H, W), values in [0, 1].
        misalignment_px : float
            Lateral misalignment at the stitching boundary in pixels.

        Returns
        -------
        Tensor of shape (H, W) representing the stitching-modified mask.
        """
        m = ensure_2d(mask).float()
        h, w = m.shape
        device = m.device

        # Stitching boundary at the vertical centre (cross-scan direction).
        boundary_y = h // 2

        # Build a blending weight map: 1.0 in the upper half-field,
        # 0.0 in the lower half-field, with a smooth transition across
        # the overlap region.
        y_coords = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)

        # Smooth sigmoid blend across overlap region.
        steepness = 10.0 / max(self.overlap_px, 1)
        blend = torch.sigmoid(steepness * (y_coords.expand(h, w) - boundary_y - misalignment_px))

        # Upper field gets weight (1 - blend), lower field gets blend.
        upper = m * (1.0 - blend)
        lower = m * blend

        # Dose jump from misalignment: proportional to overlap confusion.
        dose_error = abs(misalignment_px) * blend * (1.0 - blend)

        result = upper + lower + dose_error * 0.5

        return result.clamp(0.0, 1.0)

    def stitching_epe_penalty(
        self, mask: torch.Tensor, target: torch.Tensor, misalignment_px: float
    ) -> torch.Tensor:
        """Compute differentiable EPE penalty at the stitching boundary.

        Compares the stitched mask output to the target, weighted by
        proximity to the stitching boundary. Pixels near the boundary
        receive higher weight because stitching defects are concentrated
        there.

        Parameters
        ----------
        mask : Tensor
            Optimised mask tensor (H, W).
        target : Tensor
            Target design tensor (H, W).
        misalignment_px : float
            Misalignment at the stitching boundary in pixels.

        Returns
        -------
        Scalar penalty tensor (differentiable).
        """
        m = ensure_2d(mask).float()
        t = ensure_2d(target).float()
        h, w = m.shape
        device = m.device

        # Stitched output with misalignment.
        stitched = self.compute_stitching_boundary(m, misalignment_px)

        # Boundary proximity weight: peaks at the stitching seam.
        boundary_y = h // 2
        y_coords = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)
        sigma = max(self.overlap_px, 1.0)
        proximity = torch.exp(-0.5 * ((y_coords.expand(h, w) - boundary_y) / sigma) ** 2)

        # Weighted MSE at the stitching boundary.
        diff = (stitched - t) ** 2
        penalty = (diff * proximity).sum() / proximity.sum().clamp(min=1e-8)

        return penalty


# ---------------------------------------------------------------------------
# Stitching-aware SMO
# ---------------------------------------------------------------------------


class StitchingAwareSMO(AnamorphicSMO):
    """Source-Mask Optimization with stitching-defect-aware ILT for High-NA EUV.

    Extends :class:`AnamorphicSMO` to include stitching boundary penalties
    in the optimisation objective. The additional penalty encourages the
    optimizer to find mask layouts that are robust to half-field stitching
    misalignment, reducing EPE at the stitching seam.

    The total cost becomes::

        total_cost = EPE + PVB + shot_count + stitching_penalty

    Pareto history tracks both stitching-boundary EPE and non-stitching EPE.

    Parameters
    ----------
    params : AnamorphicParams | None
        Anamorphic imaging parameters.
    shot_count_weight : float
        Weight for the shot-count cost term.
    stitching_weight : float
        Weight for the stitching boundary penalty.
    stitching : HalfFieldStitching | None
        Stitching model; defaults to standard overlap/misalignment.
    """

    def __init__(
        self,
        params: AnamorphicParams | None = None,
        shot_count_weight: float = 0.01,
        stitching_weight: float = 0.5,
        stitching: HalfFieldStitching | None = None,
    ) -> None:
        super().__init__(params, shot_count_weight)
        self.stitching_weight = stitching_weight
        self.stitching = stitching or HalfFieldStitching()
        self.stitching_history: list[tuple[float, float]] = []

    def optimize_source_mask(
        self,
        target: torch.Tensor,
        n_steps: int = 200,
        lr: float = 0.05,
        misalignment_px: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run stitching-aware source-mask optimisation.

        Extends the base SMO loop with a stitching boundary EPE penalty.
        At each step, the stitching penalty is computed for the current
        mask at the specified misalignment and added to the total cost.

        Parameters
        ----------
        target : Tensor
            Target design pattern (H, W), binary {0, 1}.
        n_steps : int
            Number of optimisation steps.
        lr : float
            Learning rate for Adam.
        misalignment_px : float
            Assumed misalignment at the stitching boundary.

        Returns
        -------
        Tuple of (optimised_mask, info_dict).
        info_dict additionally contains 'stitching_history',
        'final_stitching_epe', and 'final_non_stitching_epe'.
        """
        t = ensure_2d(target).float()
        device = t.device

        mask_param = t.clone().detach().requires_grad_(True)
        source_param = torch.tensor([0.7, 0.7], device=device, requires_grad=True)

        optimizer = torch.optim.Adam([mask_param, source_param], lr=lr)

        self.pareto_history = []
        self.stitching_history = []

        h, w = t.shape
        boundary_y = h // 2
        y_coords = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)
        sigma = max(self.stitching.overlap_px, 1.0)
        proximity = torch.exp(-0.5 * ((y_coords.expand(h, w) - boundary_y) / sigma) ** 2)
        inv_proximity = 1.0 - proximity

        for _step in range(n_steps):
            optimizer.zero_grad()

            mask_opt = torch.sigmoid(mask_param)

            self.params.na_x = 0.55 * source_param[0].sigmoid()
            self.params.na_y = 0.55 * source_param[1].sigmoid()
            self.imaging = AnamorphicImaging(self.params)

            aerial = self.imaging.simulate_aerial(mask_opt)
            aerial_target = self.imaging.simulate_aerial(t)
            aerial_diff2 = (aerial - aerial_target) ** 2

            epe_loss = aerial_diff2.mean()

            # PVB proxy loss.
            gy = aerial[1:, :] - aerial[:-1, :]
            gx = aerial[:, 1:] - aerial[:, :-1]
            pvb_loss = gy.abs().mean() + gx.abs().mean()

            # Shot-count cost.
            sc_cost = self.shot_count.forward(mask_opt)

            # Stitching boundary penalty.
            stitch_penalty = self.stitching.stitching_epe_penalty(mask_opt, t, misalignment_px)

            total_loss = (
                epe_loss + 0.1 * pvb_loss + sc_cost + self.stitching_weight * stitch_penalty
            )

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                fidelity = epe_loss.item()
                sc_val = sc_cost.item()

                # Stitching-boundary EPE vs non-stitching EPE.
                stitch_epe = (aerial_diff2 * proximity).sum() / proximity.sum().clamp(min=1e-8)
                non_stitch_epe = (aerial_diff2 * inv_proximity).sum() / inv_proximity.sum().clamp(
                    min=1e-8
                )

                self.pareto_history.append((fidelity, sc_val))
                self.stitching_history.append((stitch_epe.item(), non_stitch_epe.item()))

        with torch.no_grad():
            final_mask = (torch.sigmoid(mask_param) > 0.5).float()

        info: dict[str, Any] = {
            "pareto_history": list(self.pareto_history),
            "stitching_history": list(self.stitching_history),
            "final_epe": (self.pareto_history[-1][0] if self.pareto_history else float("inf")),
            "final_shot_cost": (self.pareto_history[-1][1] if self.pareto_history else 0.0),
            "source_params": source_param.detach().sigmoid().tolist(),
            "final_stitching_epe": (
                self.stitching_history[-1][0] if self.stitching_history else float("inf")
            ),
            "final_non_stitching_epe": (
                self.stitching_history[-1][1] if self.stitching_history else float("inf")
            ),
        }
        return final_mask, info


# ---------------------------------------------------------------------------
# Polarization state
# ---------------------------------------------------------------------------


@dataclass
class PolarizationState:
    """Polarization state of the illumination source.

    Models the polarization of EUV illumination, which affects image
    contrast at high NA where the vector nature of light matters.
    At NA > 0.45, unpolarized illumination loses contrast compared to
    TM (transverse magnetic) polarized light for dense lines.

    Parameters
    ----------
    angle_deg : float
        Polarization angle in degrees.  0 = TM (parallel to the
        scanning direction), 90 = TE (perpendicular to scanning).
    ellipticity : float
        Ellipticity parameter in [0, 1].  0 = fully linear, 1 =
        fully circular.  Intermediate values give elliptical polarization.
    tm_weight : float
        Relative weight of the TM component.  For partially polarized
        light, ``tm_weight=1`` is fully TM, ``0.5`` is unpolarized.
    """

    angle_deg: float = 0.0
    ellipticity: float = 0.0
    tm_weight: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.ellipticity <= 1.0):
            raise ValueError(f"ellipticity must be in [0, 1], got {self.ellipticity}")
        if not (0.0 <= self.tm_weight <= 1.0):
            raise ValueError(f"tm_weight must be in [0, 1], got {self.tm_weight}")

    def contrast_factor(self, na: float, wavelength_nm: float = 13.5) -> float:
        """Compute the polarisation-dependent contrast factor.

        At high NA the vector-image contrast depends on polarization.
        TM polarization preserves contrast better than TE for line/space
        patterns.  This factor multiplies the scalar aerial image
        intensity.

        Parameters
        ----------
        na : float
            Numerical aperture.
        wavelength_nm : float
            Exposure wavelength.

        Returns
        -------
        Contrast factor in [0, 1].  Unpolarized (tm_weight=0.5) returns
        a lower value than full TM.
        """
        sin_theta = min(na, 0.99)
        cos_theta = math.sqrt(1.0 - sin_theta**2)

        # TM contrast is cos(theta), TE is 1/cos(theta) (obliquity)
        tm_factor = cos_theta
        te_factor = 1.0 / max(cos_theta, 1e-6)

        angle_rad = math.radians(self.angle_deg)
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        # Blend TM and TE based on angle and weight
        linear_factor = c * tm_factor + s * te_factor

        # Ellipticity reduces effective polarization
        effective = (1.0 - self.ellipticity) * linear_factor + self.ellipticity * (
            tm_factor + te_factor
        ) / 2.0

        # tm_weight mixes with unpolarized
        unpolarized = (tm_factor + te_factor) / 2.0
        result = self.tm_weight * effective + (1.0 - self.tm_weight) * unpolarized

        return max(0.0, min(1.0, result))


# ---------------------------------------------------------------------------
# Triple-beam illumination
# ---------------------------------------------------------------------------


class TripleBeamIllumination:
    """Three-beam illumination model with polarization control.

    Models a conventional three-beam (0th and +/- 1st diffraction orders)
    illumination scheme with per-beam polarization.  The interference of
    the three orders produces the aerial image, and polarization
    controls the contrast of the interference fringes.

    The first-order polarization terms modify the aerial image by:
      1. Weighting each diffraction order by its polarization contrast
         factor.
      2. Applying a cos(pi*pitch/pitch_cutoff) modulation that captures
         the polarization-dependent contrast roll-off for sub-resolution
         features.

    Parameters
    ----------
    polarization : PolarizationState
        Polarization state for all three beams.
    pitch_nm : float
        Nominal pattern pitch in nm.
    pixel_size_nm : float
        Pixel pitch in nm.
    wavelength_nm : float
        Exposure wavelength in nm.
    na : float
        Numerical aperture.

    References
    ----------
    - IBM, "demonstrates High-NA EUV below 2 nm nodes at SPIE 2026",
      research.ibm.com, 2026-02.
    """

    def __init__(
        self,
        polarization: PolarizationState | None = None,
        pitch_nm: float = 32.0,
        pixel_size_nm: float = 1.0,
        wavelength_nm: float = 13.5,
        na: float = 0.55,
    ) -> None:
        self.polarization = polarization or PolarizationState()
        self.pitch_nm = pitch_nm
        self.pixel_size_nm = pixel_size_nm
        self.wavelength_nm = wavelength_nm
        self.na = na

    def _diffraction_efficiency(self) -> float:
        """Compute the first-order diffraction efficiency.

        Simplified model: efficiency scales with the ratio of pitch to
        the resolution limit.  Features at the resolution limit have
        zero first-order efficiency.
        """
        pitch_cutoff = self.wavelength_nm / (2.0 * self.na)
        if self.pitch_nm <= pitch_cutoff:
            return 0.0
        ratio = pitch_cutoff / self.pitch_nm
        return max(0.0, 1.0 - ratio**2)

    def modify_aerial_image(self, aerial: torch.Tensor) -> torch.Tensor:
        """Apply polarization-weighted three-beam modulation to an aerial image.

        The three-beam interference produces a sinusoidal intensity
        modulation.  Polarization scales the modulation depth
        (contrast) of this interference pattern.

        Parameters
        ----------
        aerial : Tensor
            ``(H, W)`` scalar aerial image from the base imaging model.

        Returns
        -------
        Modified aerial image tensor of shape ``(H, W)``.
        """
        a = ensure_2d(aerial).float()
        h, w = a.shape

        # Polarization contrast factor
        pc = self.polarization.contrast_factor(self.na, self.wavelength_nm)

        # Diffraction efficiency of +/- 1st orders
        eta = self._diffraction_efficiency()

        # Three-beam modulation: the interference of 0th and +/- 1st
        # orders creates a sinusoidal pattern along x with period = pitch.
        # Modulation depth depends on diffraction efficiency and
        # polarization contrast.
        x = torch.arange(w, dtype=torch.float32, device=a.device)
        pitch_px = self.pitch_nm / self.pixel_size_nm
        spatial_freq = 2.0 * math.pi / max(pitch_px, 1.0)

        # Modulation envelope: polarisation-weighted
        modulation = 1.0 + 2.0 * eta * pc * torch.cos(spatial_freq * x)
        modulation = modulation / modulation.max().clamp(min=1e-6)

        # Apply row-wise (broadcast along y)
        result = a * modulation.unsqueeze(0)

        return result.clamp(min=0.0)

    def compute_effective_contrast(self) -> float:
        """Compute the effective image contrast with polarization.

        Returns
        -------
        Effective contrast in [0, 1], combining diffraction efficiency
        and polarization weighting.
        """
        eta = self._diffraction_efficiency()
        pc = self.polarization.contrast_factor(self.na, self.wavelength_nm)
        return (2.0 * eta * pc) / (1.0 + 2.0 * eta * pc)
