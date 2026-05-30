"""Cross-tile consistency metrics for tiled ILT workflows.

Full-chip ILT partitions the layout into overlapping tiles and optimises each
independently. Tile-boundary SRAF/curve inconsistency is identified in the
Light:Sci.Appl. 2025 survey as a major artifact source. This module quantifies
that inconsistency so it can be minimised during stitching or used as a
diagnostic in benchmarks.

Two metrics are provided:

* :func:`tile_boundary_consistency` — compare mask values in the overlap region
  between adjacent tile pairs. Ideal tiled ILT produces identical masks in the
  overlapping area; any discrepancy is a boundary artifact.
* :func:`cross_tile_sraf_consistency` — check that SRAF assist features
  (sub-resolution bars) are continuous across tile edges. Discontinuous SRAFs
  degrade process window locally at the boundary.
"""

from __future__ import annotations

import torch

from openlithohub.workflow.tiling import Tile


def tile_boundary_consistency(
    tiles: list[Tile],
    tile_results: list[torch.Tensor],
    overlap: int = 0,
) -> dict[str, float]:
    """Measure consistency of mask features across tile boundaries.

    For each pair of adjacent tiles, compare the mask values in the overlap
    region. Inconsistent tiles produce different mask patterns in the
    overlapping area.

    Args:
        tiles: Original tiles (from :func:`tile_layout`).
        tile_results: Optimised mask tensors, one per tile. Must be the same
            length as ``tiles`` and have shape compatible with each tile's
            spatial extent.
        overlap: Override for the overlap width to compare. When ``0`` (the
            default), each tile's own ``Tile.overlap`` field is used.

    Returns:
        Dictionary with:

        - ``'boundary_mse'``: MSE of overlapping regions between adjacent tiles.
        - ``'boundary_max_diff'``: maximum absolute difference at boundaries.
        - ``'sraf_consistency'``: fraction of SRAF features (pixels in
          ``[0.1, 0.5]``) that agree across the boundary (both tiles classify
          the pixel the same way). ``1.0`` = perfect consistency.
    """
    if len(tiles) != len(tile_results):
        raise ValueError(
            f"tiles and tile_results must have same length; "
            f"got {len(tiles)} vs {len(tile_results)}"
        )
    if not tiles:
        return {"boundary_mse": 0.0, "boundary_max_diff": 0.0, "sraf_consistency": 1.0}

    mse_accum: list[float] = []
    max_diff_accum: list[float] = []
    sraf_match_accum: list[float] = []
    sraf_total_accum: list[float] = []

    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            ti, ri = tiles[i], _squeeze(tile_results[i])
            tj, rj = tiles[j], _squeeze(tile_results[j])

            ol_width = overlap if overlap > 0 else min(ti.overlap, tj.overlap)
            if ol_width <= 0:
                continue

            # Check horizontal adjacency (same row, columns adjacent)
            regions = _overlap_regions(ti, tj, ri, rj, ol_width)
            for ra, rb in regions:
                if ra is None:
                    continue
                diff = ra - rb
                mse_accum.append(float(diff.pow(2).mean().item()))
                max_diff_accum.append(float(diff.abs().max().item()))

                # SRAF consistency: both classify as SRAF or both don't
                sraf_a = (ra > 0.1) & (ra < 0.5)
                sraf_b = (rb > 0.1) & (rb < 0.5)
                n_sraf = float((sraf_a | sraf_b).sum().item())
                if n_sraf > 0:
                    matches = float((sraf_a == sraf_b).sum().item())
                    sraf_match_accum.append(matches)
                    sraf_total_accum.append(float(sraf_a.numel()))

    if not mse_accum:
        return {"boundary_mse": 0.0, "boundary_max_diff": 0.0, "sraf_consistency": 1.0}

    avg_mse = sum(mse_accum) / len(mse_accum)
    max_diff = max(max_diff_accum)
    sraf_consistency = (
        sum(sraf_match_accum) / sum(sraf_total_accum) if sraf_total_accum else 1.0
    )

    return {
        "boundary_mse": avg_mse,
        "boundary_max_diff": max_diff,
        "sraf_consistency": sraf_consistency,
    }


def cross_tile_sraf_consistency(
    layout_mask: torch.Tensor,
    tile_size: int,
    sraf_threshold: float = 0.3,
) -> dict[str, float]:
    """Check SRAF pattern continuity across tile boundaries.

    Splits mask into tiles, checks that SRAF features (small assist features
    below ``sraf_threshold``) are continuous across tile edges.

    Args:
        layout_mask: Full layout mask tensor ``(H, W)``.
        tile_size: Tile size used for the tiling partition.
        sraf_threshold: Pixels with mask value in ``(0.05, sraf_threshold]``
            are classified as SRAF. Pixels above are main features; pixels
            below are background.

    Returns:
        Dictionary with:

        - ``'sraf_discontinuity_rate'``: fraction of tile-edge pixels where an
          SRAF feature is present on one side but absent on the other.
          ``0.0`` = perfectly continuous.
        - ``'n_boundary_pixels'``: total number of boundary pixels examined.
    """
    mask = _ensure_2d(layout_mask)
    h, w = mask.shape

    if tile_size >= max(h, w):
        return {"sraf_discontinuity_rate": 0.0, "n_boundary_pixels": 0}

    is_sraf = (mask > 0.05) & (mask <= sraf_threshold)

    total_boundary = 0
    discontinuities = 0

    # Horizontal tile boundaries (rows that fall on tile edges)
    y = tile_size
    while y < h:
        above = is_sraf[y - 1, :]  # last row of tile above
        below = is_sraf[y, :]  # first row of tile below
        mismatch = above != below
        discontinuities += int(mismatch.sum().item())
        total_boundary += w
        y += tile_size

    # Vertical tile boundaries
    x = tile_size
    while x < w:
        left = is_sraf[:, x - 1]  # last col of tile left
        right = is_sraf[:, x]  # first col of tile right
        mismatch = left != right
        discontinuities += int(mismatch.sum().item())
        total_boundary += h
        x += tile_size

    if total_boundary == 0:
        return {"sraf_discontinuity_rate": 0.0, "n_boundary_pixels": 0}

    return {
        "sraf_discontinuity_rate": discontinuities / total_boundary,
        "n_boundary_pixels": total_boundary,
    }


def _squeeze(t: torch.Tensor) -> torch.Tensor:
    """Remove leading singleton dims to get a 2D (H, W) tensor."""
    while t.ndim > 2:
        t = t.squeeze(0)
    return t


def _overlap_regions(
    ti: Tile,
    tj: Tile,
    ri: torch.Tensor,
    rj: torch.Tensor,
    ol_width: int,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Extract matching overlap patches from two overlapping tile results.

    Handles the real tiling scheme where tiles have overlapping extents in
    global coordinates. For a pair of horizontally overlapping tiles, the
    overlap region in global coords is ``[max(left_i, left_j), min(right_i, right_j))``.
    We extract that same region from each tile's local result tensor.

    Returns list of (region_i, region_j) pairs for each overlap found.
    """
    results: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    # Horizontal overlap detection
    x_overlap_start = max(ti.origin_x, tj.origin_x)
    x_overlap_end = min(ti.origin_x + ti.width, tj.origin_x + tj.width)
    y_overlap_start = max(ti.origin_y, tj.origin_y)
    y_overlap_end = min(ti.origin_y + ti.height, tj.origin_y + tj.height)

    # Check if tiles actually overlap in both dimensions
    if x_overlap_start >= x_overlap_end or y_overlap_start >= y_overlap_end:
        results.append((None, None))
        return results

    # Determine adjacency: tiles should be neighbours (same row or same column)
    # in at least one axis for the overlap to be a boundary region.
    same_row = ti.origin_y == tj.origin_y and ti.height == tj.height
    same_col = ti.origin_x == tj.origin_x and ti.width == tj.width
    if not same_row and not same_col:
        results.append((None, None))
        return results

    # Clip the overlap region to at most ol_width pixels in the overlap axis
    if same_row:
        ol = min(x_overlap_end - x_overlap_start, ol_width)
        x_start = x_overlap_start
        x_end = x_start + ol
        y_start = y_overlap_start
        y_end = y_overlap_end
    else:
        ol = min(y_overlap_end - y_overlap_start, ol_width)
        x_start = x_overlap_start
        x_end = x_overlap_end
        y_start = y_overlap_start
        y_end = y_start + ol

    # Map global coordinates to local tile coordinates
    # Tile i local: (y - ti.origin_y, x - ti.origin_x)
    ri_y0 = y_start - ti.origin_y
    ri_x0 = x_start - ti.origin_x
    ri_y1 = ri_y0 + (y_end - y_start)
    ri_x1 = ri_x0 + (x_end - x_start)

    rj_y0 = y_start - tj.origin_y
    rj_x0 = x_start - tj.origin_x
    rj_y1 = rj_y0 + (y_end - y_start)
    rj_x1 = rj_x0 + (x_end - x_start)

    # Bounds check
    h_i, w_i = ri.shape[-2], ri.shape[-1]
    h_j, w_j = rj.shape[-2], rj.shape[-1]
    if (
        ri_y0 < 0 or ri_x0 < 0 or ri_y1 > h_i or ri_x1 > w_i
        or rj_y0 < 0 or rj_x0 < 0 or rj_y1 > h_j or rj_x1 > w_j
    ):
        results.append((None, None))
        return results

    patch_i = ri[..., ri_y0:ri_y1, ri_x0:ri_x1]
    patch_j = rj[..., rj_y0:rj_y1, rj_x0:rj_x1]

    if patch_i.shape != patch_j.shape:
        results.append((None, None))
        return results

    results.append((patch_i, patch_j))
    return results


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Coerce to 2D (H, W) by squeezing leading singletons."""
    while t.ndim > 2:
        if t.shape[0] == 1:
            t = t.squeeze(0)
        else:
            break
    return t


def schwarz_convergence_metrics(
    history: list[float],
) -> dict[str, float | bool]:
    """Summarize Schwarz iteration convergence from boundary MSE history.

    Args:
        history: Per-iteration boundary MSE values returned by
            :func:`openlithohub.workflow.tiling.schwarz_tiled_ilt`.

    Returns:
        Dict with:

        - ``'initial_mse'``: boundary MSE at Schwarz iteration 0.
        - ``'final_mse'``: boundary MSE at the last iteration.
        - ``'reduction_ratio'``: ``final_mse / initial_mse`` (``1.0`` if
          history is empty or ``initial_mse`` is zero).
        - ``'converged_monotone'``: ``True`` if the MSE decreased at every
          step (no oscillation).
    """
    if not history:
        return {
            "initial_mse": 0.0,
            "final_mse": 0.0,
            "reduction_ratio": 1.0,
            "converged_monotone": True,
        }

    initial = history[0]
    final = history[-1]
    ratio = final / initial if initial > 0 else 1.0

    monotone = all(history[i] <= history[i - 1] + 1e-12 for i in range(1, len(history)))

    return {
        "initial_mse": initial,
        "final_mse": final,
        "reduction_ratio": ratio,
        "converged_monotone": monotone,
    }
