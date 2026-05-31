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

from openlithohub.benchmark.metrics.shot_count import estimate_shot_count
from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.tensor_ops import ensure_2d


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
    wavelength_nm: float = 13.5
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
    """

    def __init__(self, params: AnamorphicParams | None = None) -> None:
        self.params = params or AnamorphicParams()

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
        psf = (field.real ** 2 + field.imag ** 2)
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
            inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
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
        phase = torch.exp(
            -2j * math.pi * (fx * shift_x + fy * shift_y)
        ).to(torch.complex64)
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
        source_param = torch.tensor(
            [0.7, 0.7], device=device, requires_grad=True
        )

        optimizer = torch.optim.Adam([mask_param, source_param], lr=lr)

        self.pareto_history = []

        for step in range(n_steps):
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
            pvb_loss = (gy.abs().mean() + gx.abs().mean())

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
                target, n_steps=self.n_steps, lr=self.lr,
            )
            results["anamorphic"].append(info_aniso)

            torch.manual_seed(seed)

            # Isotropic run.
            smo_iso = AnamorphicSMO(iso_params, shot_count_weight=0.01)
            _, info_iso = smo_iso.optimize_source_mask(
                target, n_steps=self.n_steps, lr=self.lr,
            )
            results["isotropic"].append(info_iso)

        # Aggregate.
        for key in ("anamorphic", "isotropic"):
            epes = [r["final_epe"] for r in results[key]]
            scs = [r["final_shot_cost"] for r in results[key]]
            results[f"mean_{key}_epe"] = sum(epes) / len(epes)
            results[f"mean_{key}_shot_cost"] = sum(scs) / len(scs)

        return results
