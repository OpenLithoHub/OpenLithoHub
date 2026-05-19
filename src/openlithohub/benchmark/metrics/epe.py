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
) -> dict[str, float | bool]:
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
        return {"epe_mean_nm": inf, "epe_max_nm": inf, "epe_std_nm": 0.0, "valid": False}

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
