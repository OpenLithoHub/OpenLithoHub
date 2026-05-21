"""L2 wafer error — Neural-ILT canonical mask-printability metric.

The standard academic OPC scoring contract, as established by
[Neural-ILT (ICCAD'20)](https://github.com/cuhk-eda/neural-ilt) and used
by GAN-OPC / MOSAIC, is:

    wafer = lithosim(mask, dose=1.0, threshold=0.225)
    L2    = (wafer - target).abs().sum()

i.e. forward-simulate the predicted mask through SOCS optics and the
resist threshold, then sum the absolute pixel-wise error against the
target *layout* (not against the input mask). The result is in pixel
units; multiply by ``pixel_size_nm**2`` for an area in nm² if needed.

Naming note: the published paper calls this "L2 error", but the closed-form
above is the L1 norm of ``(wafer - target)`` (sum of absolute differences).
The ``l2_error_pixels`` field name is preserved for cross-paper comparability;
do not "fix" it to L1 without coordinating against the upstream tables.
For a binary wafer/target the two norms are equal anyway — ``|x|`` and
``x²`` agree on ``{0,1}`` — so the underlying scalar is the same number
either way.

Like :func:`openlithohub.benchmark.metrics.epe.compute_wafer_epe`, this
metric requires the forward simulator in the loop. The
:func:`compute_epe` mask-level metric scores 0 for an Identity model;
``compute_l2_error`` does not, because diffraction reshapes the printed
contour even when the mask is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from openlithohub.simulators.base import BaseSimulator


class L2ErrorResult(TypedDict):
    """Per-sample L2 wafer-error summary.

    Attributes:
        l2_error_pixels: ``(wafer - target).abs().sum()`` in pixel units —
            the literal Neural-ILT contract value.
        l2_error_nm2: Same quantity expressed as a physical area in nm²
            (``l2_error_pixels * pixel_size_nm**2``). Useful when comparing
            results computed at different pitches.
        wafer_pixels: Number of foreground pixels in the simulated wafer
            image. Reported alongside the error so a normalised ratio can
            be derived downstream without re-running the simulator.
        target_pixels: Foreground pixel count of the target layout.
    """

    l2_error_pixels: float
    l2_error_nm2: float
    wafer_pixels: int
    target_pixels: int


def compute_l2_error(
    predicted_mask: torch.Tensor,
    target: torch.Tensor,
    pixel_size_nm: float = 1.0,
    simulator: BaseSimulator | None = None,
) -> L2ErrorResult:
    """Compute L2 wafer error per the Neural-ILT eval contract.

    Args:
        predicted_mask: Predicted mask (H, W), values in [0, 1].
        target: Target layout (H, W), values in {0, 1}. Compared against
            the *simulated wafer*, not against the predicted mask.
        pixel_size_nm: Physical pixel size, used only to convert the
            pixel-unit error to an nm² area. Does not affect simulator
            sampling — pass a configured ``simulator`` for that.
        simulator: Forward simulator. Defaults to a fresh
            :class:`HopkinsSimulator`. Pass an explicit instance to keep
            dose / threshold / illumination consistent across a run.

    Returns:
        :class:`L2ErrorResult` with the raw Neural-ILT scalar plus its
        nm² conversion and the supporting pixel counts.
    """
    if predicted_mask.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_mask.shape} vs target {target.shape}"
        )

    if simulator is None:
        # Local import: simulators package builds SOCS kernels on init,
        # which we don't want to pay at metric module import time.
        from openlithohub.simulators.hopkins_sim import HopkinsSimulator

        simulator = HopkinsSimulator()

    sim_result = simulator.simulate(predicted_mask)
    if sim_result.resist is not None:
        wafer = sim_result.resist
    else:
        threshold = simulator.config.threshold * simulator.config.dose
        wafer = (sim_result.aerial >= threshold).to(sim_result.aerial.dtype)

    target_f = target.to(wafer.dtype)
    l2_pixels = float((wafer - target_f).abs().sum().item())

    return {
        "l2_error_pixels": l2_pixels,
        "l2_error_nm2": l2_pixels * pixel_size_nm * pixel_size_nm,
        "wafer_pixels": int(wafer.sum().item()),
        "target_pixels": int(target_f.sum().item()),
    }
