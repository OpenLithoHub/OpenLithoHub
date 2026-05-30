"""DiffNano high-precision resist model adapter.

Wraps :class:`diffnano.solvers.resist.DifferentiableResistModel` as an
opt-in resist backend.  When ``[diffnano]`` is not installed the module
can still be imported — only the ``*ResistModel`` accessors raise
:class:`~openlithohub.plugins.OptionalPluginError`.
"""

from __future__ import annotations

import torch

from openlithohub.plugins import optional_import

__all__ = ["DiffNanoResistAdapter"]


class DiffNanoResistAdapter:
    """Adapter that bridges DiffNano's ``DifferentiableResistModel`` to
    OpenLithoHub's resist interface (aerial_image → binary contour).

    This is used as an opt-in resist backend, not as a ``BaseSimulator``
    subclass — it replaces the built-in
    :func:`~openlithohub._utils.resist_model.apply_differentiable_resist`
    path when selected via ``SimulatorConfig.resist_backend``.

    Parameters
    ----------
    acid_diffusion_length_nm : float
        Acid diffusion length during PEB (nm).
    development_contrast : float
        Resist contrast (higher = sharper).
    threshold_dose : float
        Normalized threshold for clearing.
    peb_diffusion_nm : float
        Post-exposure bake diffusion length (nm).
    pixel_size_nm : float
        Grid spacing for nm → pixel conversion.
    """

    def __init__(
        self,
        *,
        acid_diffusion_length_nm: float = 20.0,
        development_contrast: float = 10.0,
        threshold_dose: float = 0.5,
        peb_diffusion_nm: float = 10.0,
        pixel_size_nm: float = 1.0,
    ) -> None:
        mod = optional_import("diffnano.solvers.resist", plugin="diffnano")
        self._model = mod.DifferentiableResistModel(
            acid_diffusion_length_nm=acid_diffusion_length_nm,
            development_contrast=development_contrast,
            threshold_dose=threshold_dose,
            peb_diffusion_nm=peb_diffusion_nm,
        )
        self.pixel_size_nm = pixel_size_nm

    def __call__(self, aerial_image: torch.Tensor) -> torch.Tensor:
        """Apply DiffNano resist to *aerial_image*.

        Parameters
        ----------
        aerial_image : Tensor, shape ``(H, W)``
            Aerial intensity, values in ``[0, 1]``.

        Returns
        -------
        Tensor, shape ``(H, W)``
            Resist contour (soft, differentiable).
        """
        result = self._model.forward(aerial_image)
        return result.field.squeeze(0).to(aerial_image.dtype)  # type: ignore[no-any-return]

    def calibrate(
        self,
        target_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        n_steps: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """Delegate to DiffNano's built-in Adam-based calibration."""
        return self._model.calibrate(target_pairs, n_steps=n_steps, lr=lr)  # type: ignore[no-any-return]
