"""Edge Placement Error (EPE) computation.

Two flavors live here:

* :func:`compute_epe` — mask-level. Compares predicted mask edges directly
  to target edges. An Identity model (mask passed straight through) scores
  0 by construction, which is useful as a sanity baseline but does NOT
  reflect what would actually print on the wafer.
* :func:`compute_wafer_epe` — wafer-level. Pushes the predicted mask
  through a forward optical/resist simulator and compares the *resist*
  contour to the target. This is the physically meaningful quantity for
  OPC quality: a square mask will round at the corners after diffraction,
  so an Identity model lands at a nonzero EPE.

Both report the same ``EPEResult`` schema; the leaderboard surfaces them
under separate keys (``epe_*`` vs ``epe_wafer_*``) so existing dashboards
that compare against historical mask-level numbers stay valid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch
import torch.nn.functional as functional

if TYPE_CHECKING:
    from openlithohub.simulators.base import BaseSimulator


class EPEResult(TypedDict):
    """Per-sample EPE summary. Numeric fields are always ``float`` so callers
    can do arithmetic on them without first narrowing away ``bool``."""

    epe_mean_nm: float
    epe_max_nm: float
    epe_std_nm: float
    valid: bool


def _extract_edges(binary: torch.Tensor) -> torch.Tensor:
    """Extract edge pixels from a binary mask using Sobel filtering.

    The input is thresholded at 0.5 first so a soft mask passed in by
    mistake (e.g. a raw resist field that hasn't been binarized) still
    yields meaningful edges instead of the noisy gradient of a continuous
    field.

    Returns a boolean tensor marking edge pixel locations.
    """
    inp = (binary > 0.5).float().unsqueeze(0).unsqueeze(0)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=binary.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=binary.device,
    ).reshape(1, 1, 3, 3)

    gx = functional.conv2d(inp, sobel_x, padding=1)
    gy = functional.conv2d(inp, sobel_y, padding=1)
    magnitude = (gx.square() + gy.square()).sqrt().squeeze()

    # Zero-padded Sobel produces phantom 1-pixel edges along every image border
    # whenever foreground touches the frame. Strip the border so a fully-clear
    # edge of features (typical in tile-based optimization) does not bias EPE.
    magnitude[0, :] = 0.0
    magnitude[-1, :] = 0.0
    magnitude[:, 0] = 0.0
    magnitude[:, -1] = 0.0

    return magnitude > 0.0


def compute_epe(
    predicted: torch.Tensor,
    target: torch.Tensor,
    pixel_size_nm: float = 1.0,
) -> EPEResult:
    """Compute Edge Placement Error between predicted and target contours.

    Extracts edges from both binary masks via Sobel operators, then computes
    the minimum Euclidean distance from each predicted edge pixel to the
    nearest target edge pixel.

    Args:
        predicted: Binary mask of predicted pattern (H, W), values in {0, 1}.
        target: Binary mask of target/reference pattern (H, W), values in {0, 1}.
        pixel_size_nm: Physical size of each pixel in nanometers.

    Returns:
        Dictionary with keys ``epe_mean_nm``, ``epe_max_nm``, ``epe_std_nm``,
        and ``valid``. Empty-edge cases are reported explicitly:

        - both edge sets empty → all zeros, ``valid=True`` (degenerate match).
        - exactly one edge set empty → all values ``inf`` and ``valid=False``;
            callers must not treat the result as a "perfect" score.
        - exactly one matched edge pixel → ``epe_std_nm`` is ``nan`` (std over
            a single sample is undefined); ``valid=True``.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}")

    pred_edges = _extract_edges(predicted)
    tgt_edges = _extract_edges(target)

    pred_pts = pred_edges.nonzero(as_tuple=False).float()
    tgt_pts = tgt_edges.nonzero(as_tuple=False).float()

    pred_empty = pred_pts.numel() == 0
    tgt_empty = tgt_pts.numel() == 0
    if pred_empty and tgt_empty:
        return {"epe_mean_nm": 0.0, "epe_max_nm": 0.0, "epe_std_nm": 0.0, "valid": True}
    if pred_empty or tgt_empty:
        inf = float("inf")
        # Std is undefined when one edge set is empty — return nan rather
        # than 0.0 so callers can distinguish "no data" from a real zero
        # spread, matching the single-edge-pixel convention below.
        return {"epe_mean_nm": inf, "epe_max_nm": inf, "epe_std_nm": float("nan"), "valid": False}

    # Compute pairwise distances in chunks along BOTH axes to keep peak
    # memory at chunk_size^2 floats regardless of edge count. With a single
    # axis chunked, large target patterns still blow the memory budget.
    chunk_size = 4096
    min_dists = []
    for i in range(0, pred_pts.shape[0], chunk_size):
        pred_chunk = pred_pts[i : i + chunk_size]
        running = torch.full(
            (pred_chunk.shape[0],),
            float("inf"),
            device=pred_chunk.device,
            dtype=pred_chunk.dtype,
        )
        for j in range(0, tgt_pts.shape[0], chunk_size):
            tgt_chunk = tgt_pts[j : j + chunk_size]
            dists = torch.cdist(pred_chunk, tgt_chunk)
            running = torch.minimum(running, dists.min(dim=1).values)
        min_dists.append(running)

    min_distances = torch.cat(min_dists) * pixel_size_nm

    return {
        "epe_mean_nm": float(min_distances.mean().item()),
        "epe_max_nm": float(min_distances.max().item()),
        # std over a single edge pixel is undefined, not zero — return nan so
        # downstream filters can distinguish a degenerate single-edge result
        # from a genuine zero-spread multi-edge match.
        "epe_std_nm": (
            float(min_distances.std().item()) if min_distances.numel() > 1 else float("nan")
        ),
        "valid": True,
    }


def compute_wafer_epe(
    predicted_mask: torch.Tensor,
    target: torch.Tensor,
    pixel_size_nm: float = 1.0,
    simulator: BaseSimulator | None = None,
) -> EPEResult:
    """Compute EPE between the *printed wafer contour* and the target.

    Pushes ``predicted_mask`` through a forward optical/resist simulator
    and compares the resulting binarised resist image to ``target`` using
    the same edge-distance routine as :func:`compute_epe`. This is the
    physically meaningful EPE for OPC quality — an Identity model (mask
    returned unchanged) lands at a nonzero value here because diffraction
    rounds corners that the original mask had as right angles.

    Args:
        predicted_mask: Predicted mask (H, W), values in [0, 1]. The
            simulator will be applied to this tensor.
        target: Target wafer/contour pattern (H, W), values in {0, 1}.
        pixel_size_nm: Physical pixel size in nanometers.
        simulator: Forward simulator. Defaults to a fresh
            :class:`HopkinsSimulator` with default config — callers that
            need specific dose / threshold / illumination should pass an
            explicit instance to keep results comparable across runs.

    Returns:
        Same ``EPEResult`` schema as :func:`compute_epe`.
    """
    if simulator is None:
        # Local import: simulators package pulls in heavy SOCS kernel state,
        # so we don't want benchmark.metrics.epe to drag it in at import time.
        from openlithohub.simulators.hopkins_sim import HopkinsSimulator

        simulator = HopkinsSimulator()

    sim_result = simulator.simulate(predicted_mask)
    # Prefer the binarised resist contour the simulator already produced.
    # Fall back to thresholding the aerial image at the configured threshold
    # for backends that only return aerial intensity.
    if sim_result.resist is not None:
        wafer = sim_result.resist
    else:
        threshold = simulator.config.threshold * simulator.config.dose
        wafer = (sim_result.aerial >= threshold).to(sim_result.aerial.dtype)

    return compute_epe(wafer, target, pixel_size_nm=pixel_size_nm)
