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

# ---------------------------------------------------------------------------
# Single source of truth for plugin parameter defaults (WS-B centralization)
# ---------------------------------------------------------------------------
# These values define the canonical defaults for the DiffNano resist plugin.
# When users do not explicitly provide parameters, the adapter falls back to
# these values.  If you need to change a default, change it HERE.
# ---------------------------------------------------------------------------

RESIST_DEFAULTS: dict[str, float] = {
    "acid_diffusion_length_nm": 20.0,   # nm — acid diffusion length during PEB
    "development_contrast": 10.0,       # dimensionless — higher = sharper development
    "threshold_dose": 0.5,              # normalized — clearing threshold
    "peb_diffusion_nm": 10.0,           # nm — post-exposure bake diffusion length
    "pixel_size_nm": 1.0,               # nm — grid spacing for nm-to-pixel conversion
}


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
        acid_diffusion_length_nm: float | None = None,
        development_contrast: float | None = None,
        threshold_dose: float | None = None,
        peb_diffusion_nm: float | None = None,
        pixel_size_nm: float | None = None,
    ) -> None:
        mod = optional_import("diffnano.solvers.resist", plugin="diffnano")
        _acid = acid_diffusion_length_nm if acid_diffusion_length_nm is not None else RESIST_DEFAULTS["acid_diffusion_length_nm"]
        _contrast = development_contrast if development_contrast is not None else RESIST_DEFAULTS["development_contrast"]
        _threshold = threshold_dose if threshold_dose is not None else RESIST_DEFAULTS["threshold_dose"]
        _peb = peb_diffusion_nm if peb_diffusion_nm is not None else RESIST_DEFAULTS["peb_diffusion_nm"]
        _pixel = pixel_size_nm if pixel_size_nm is not None else RESIST_DEFAULTS["pixel_size_nm"]
        self._model = mod.DifferentiableResistModel(
            acid_diffusion_length_nm=_acid,
            development_contrast=_contrast,
            threshold_dose=_threshold,
            peb_diffusion_nm=_peb,
        )
        self.pixel_size_nm = _pixel

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
