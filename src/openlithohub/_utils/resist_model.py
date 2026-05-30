"""Chemically-amplified resist simulation with acid diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from openlithohub._utils.forward_model import _build_gaussian_kernel, _circular_pad_clamped


def differentiable_threshold(
    aerial_image: torch.Tensor,
    threshold: float = 0.5,
    steepness: float = 50.0,
) -> torch.Tensor:
    """Smooth, differentiable substitute for a hard resist threshold.

    Returns ``sigmoid(steepness * (aerial - threshold))``. As ``steepness``
    increases the output approaches a step function while remaining
    differentiable everywhere — required for gradient-based ILT.
    """
    return torch.sigmoid(steepness * (aerial_image - threshold))


def _diffuse_acid(acid: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Apply Gaussian acid-diffusion blur with circular padding.

    Resist diffusion shares the periodic-boundary contract documented in
    ``forward_model``; zero-padding here would cause near-edge resist to
    underexpose relative to the Hopkins forward and silently bias EPE on
    layouts with features close to the frame.
    """
    kernel = _build_gaussian_kernel(sigma_px, acid.device)
    padding = kernel.shape[-1] // 2
    inp = acid.unsqueeze(0).unsqueeze(0)
    inp_padded = _circular_pad_clamped(inp, padding)
    return functional.conv2d(inp_padded, kernel).squeeze(0).squeeze(0)


def simulate_resist(
    aerial_image: torch.Tensor,
    acid_diffusion_length_nm: float = 5.0,
    pixel_size_nm: float = 1.0,
    threshold: float = 0.5,
    quencher_concentration: float = 0.1,
) -> torch.Tensor:
    """Simulate chemically-amplified resist response with acid diffusion.

    Models a physically-motivated resist development process:
    1. Photoacid generation proportional to aerial image intensity
    2. Acid diffusion via Gaussian blur (diffusion length determines spread)
    3. Quencher neutralization (constant subtraction)
    4. Threshold to binary resist pattern

    Args:
        aerial_image: Aerial image intensity (H, W), values in [0, 1].
        acid_diffusion_length_nm: Acid diffusion length in nanometers.
        pixel_size_nm: Physical pixel size for unit conversion.
        threshold: Development threshold *applied to the post-quencher acid
            field*. The quencher is subtracted before thresholding, so an
            aerial intensity of ``threshold + quencher_concentration`` is
            the dose where development just kicks in. This is the standard
            CAR convention (more quencher → more acid needed) — not a
            normalized intensity threshold. To keep a fixed effective dose
            cutoff regardless of quencher, set
            ``threshold = nominal_threshold - quencher_concentration``.
        quencher_concentration: Base quencher level subtracted from acid.

    Returns:
        Binary resist pattern (H, W), 1.0 where resist remains.
    """
    acid = aerial_image.clone()

    sigma_diffusion_px = acid_diffusion_length_nm / max(pixel_size_nm, 1e-6)
    if sigma_diffusion_px > 0.1:
        acid = _diffuse_acid(acid, sigma_diffusion_px)

    acid = (acid - quencher_concentration).clamp(min=0.0)
    return (acid >= threshold).float()


def simulate_resist_soft(
    aerial_image: torch.Tensor,
    acid_diffusion_length_nm: float = 5.0,
    pixel_size_nm: float = 1.0,
    threshold: float = 0.5,
    quencher_concentration: float = 0.1,
    steepness: float = 50.0,
) -> torch.Tensor:
    """Differentiable resist simulation using sigmoid instead of hard threshold.

    Same physics as `simulate_resist` but uses a smooth sigmoid for the
    development step, making it suitable for gradient-based optimization.
    """
    acid = aerial_image.clone()

    sigma_diffusion_px = acid_diffusion_length_nm / max(pixel_size_nm, 1e-6)
    if sigma_diffusion_px > 0.1:
        acid = _diffuse_acid(acid, sigma_diffusion_px)

    acid = (acid - quencher_concentration).clamp(min=0.0)
    return differentiable_threshold(acid, threshold=threshold, steepness=steepness)


def apply_differentiable_resist(
    aerial_image: torch.Tensor,
    threshold: float = 0.5,
    steepness: float = 50.0,
    resist_diffusion_nm: float = 0.0,
    pixel_size_nm: float = 1.0,
    quencher: float = 0.0,
) -> torch.Tensor:
    """Apply differentiable resist with optional acid diffusion.

    When ``resist_diffusion_nm`` and ``quencher`` are both zero (default),
    falls through to :func:`differentiable_threshold` for bit-identical
    behavior with the legacy path. When diffusion is enabled, delegates
    to :func:`simulate_resist_soft`.

    This is the single dispatch point ILT optimizers should use.
    """
    if resist_diffusion_nm <= 0.0 and quencher <= 0.0:
        return differentiable_threshold(aerial_image, threshold, steepness)
    return simulate_resist_soft(
        aerial_image,
        acid_diffusion_length_nm=resist_diffusion_nm,
        pixel_size_nm=pixel_size_nm,
        threshold=threshold,
        quencher_concentration=quencher,
        steepness=steepness,
    )


class ResistCalibration:
    """Least-squares resist parameter calibration from SEM CD anchors.

    .. deprecated::
        The scalar CD model used by :meth:`fit` is a placeholder — it reduces
        the full 2D resist simulation to a single binary check per anchor,
        making the calibration trivially satisfiable (any threshold below the
        minimum aerial intensity scores zero error regardless of real resist
        behaviour). Do not rely on the returned parameters for production
        resist modelling. A full image-level calibration (simulate resist on
        the 2D aerial image, measure printed CD from the binary contour, compare
        against ``target_cd``) will replace this stub in a future release.

    Given one or more ``(aerial_intensity, measured_cd_nm)`` pairs, find
    ``(threshold, resist_diffusion_nm, quencher)`` that minimises the
    squared CD error. This is a brute-force grid search — fast enough for
    the typical handful of anchor points.

    Usage::

        cal = ResistCalibration(pixel_size_nm=1.0)
        params = cal.fit([
            (0.45, 40.0),   # (nominal aerial intensity, measured CD)
            (0.55, 45.0),
        ])
        # params.threshold, params.resist_diffusion_nm, params.quencher
    """

    @staticmethod
    def fit(
        anchors: list[tuple[float, float]],
        pixel_size_nm: float = 1.0,
        threshold_range: tuple[float, float, float] = (0.1, 0.5, 0.05),
        diffusion_range: tuple[float, float, float] = (0.0, 10.0, 2.0),
        quencher_range: tuple[float, float, float] = (0.0, 0.2, 0.05),
    ) -> tuple[float, float, float]:
        """Find (threshold, resist_diffusion_nm, quencher) minimizing CD error.

        .. warning::
            The underlying CD model is a scalar placeholder, not a physically
            meaningful simulation. See class docstring for details.

        Args:
            anchors: ``[(aerial_intensity, measured_cd_nm), ...]``.
            pixel_size_nm: Physical pixel size.
            threshold_range: ``(start, stop, step)`` grid for threshold.
            diffusion_range: ``(start, stop, step)`` grid for diffusion length.
            quencher_range: ``(start, stop, step)`` grid for quencher.

        Returns:
            ``(threshold, resist_diffusion_nm, quencher)`` with lowest error.
        """
        import warnings

        import numpy as np

        warnings.warn(
            "ResistCalibration.fit uses a scalar CD placeholder model. "
            "Returned parameters are not physically meaningful. "
            "See ResistCalibration class docstring for details.",
            UserWarning,
            stacklevel=2,
        )

        best_params = (0.225, 0.0, 0.0)
        best_error = float("inf")

        thresholds = np.arange(*threshold_range)
        diffusions = np.arange(*diffusion_range)
        quenchers = np.arange(*quencher_range)

        for th in thresholds:
            for df in diffusions:
                for qu in quenchers:
                    error = 0.0
                    for aerial_val, target_cd in anchors:
                        acid = aerial_val
                        if df > 0.0:
                            sigma_px = df / max(pixel_size_nm, 1e-6)
                            acid = float(acid * min(1.0, 1.0 / (1.0 + sigma_px * 0.1)))
                        acid = float(max(0.0, acid - qu))
                        resist = 1.0 if acid >= th else 0.0
                        cd_nm = resist * target_cd
                        error += (cd_nm - target_cd) ** 2
                    if error < best_error:
                        best_error = error
                        best_params = (float(th), float(df), float(qu))

        return best_params
