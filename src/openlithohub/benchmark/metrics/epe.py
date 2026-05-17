"""Edge Placement Error (EPE) computation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _extract_edges(binary: torch.Tensor) -> torch.Tensor:
    """Extract edge pixels from a binary mask using Sobel filtering.

    Returns a boolean tensor marking edge pixel locations.
    """
    inp = binary.float().unsqueeze(0).unsqueeze(0)

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

    return magnitude > 0.0


def compute_epe(
    predicted: torch.Tensor,
    target: torch.Tensor,
    pixel_size_nm: float = 1.0,
) -> dict[str, float]:
    """Compute Edge Placement Error between predicted and target contours.

    Extracts edges from both binary masks via Sobel operators, then computes
    the minimum Euclidean distance from each predicted edge pixel to the
    nearest target edge pixel.

    Args:
        predicted: Binary mask of predicted pattern (H, W), values in {0, 1}.
        target: Binary mask of target/reference pattern (H, W), values in {0, 1}.
        pixel_size_nm: Physical size of each pixel in nanometers.

    Returns:
        Dictionary with 'epe_mean_nm', 'epe_max_nm', 'epe_std_nm'.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}")

    pred_edges = _extract_edges(predicted)
    tgt_edges = _extract_edges(target)

    pred_pts = pred_edges.nonzero(as_tuple=False).float()
    tgt_pts = tgt_edges.nonzero(as_tuple=False).float()

    if pred_pts.numel() == 0 or tgt_pts.numel() == 0:
        return {"epe_mean_nm": 0.0, "epe_max_nm": 0.0, "epe_std_nm": 0.0}

    # Compute pairwise distances in chunks to limit memory usage.
    chunk_size = 4096
    min_dists = []
    for i in range(0, pred_pts.shape[0], chunk_size):
        chunk = pred_pts[i : i + chunk_size]
        dists = torch.cdist(chunk, tgt_pts)
        min_dists.append(dists.min(dim=1).values)

    min_distances = torch.cat(min_dists) * pixel_size_nm

    return {
        "epe_mean_nm": float(min_distances.mean().item()),
        "epe_max_nm": float(min_distances.max().item()),
        "epe_std_nm": float(min_distances.std().item()) if min_distances.numel() > 1 else 0.0,
    }
