"""Cross-tile consistency metrics for tiled ILT workflows.

Full-chip ILT partitions the layout into overlapping tiles and optimises each
independently. Tile-boundary SRAF/curve inconsistency is identified in the
Light:Sci.Appl. 2025 survey as a major artifact source. This module quantifies
that inconsistency so it can be minimised during stitching or used as a
diagnostic in benchmarks.

Metrics provided:

* :func:`tile_boundary_consistency` — compare mask values in the overlap region
  between adjacent tile pairs. Ideal tiled ILT produces identical masks in the
  overlapping area; any discrepancy is a boundary artifact.
* :func:`cross_tile_sraf_consistency` — check that SRAF assist features
  (sub-resolution bars) are continuous across tile edges. Discontinuous SRAFs
  degrade process window locally at the boundary.
* :func:`cross_tile_epe_residual` — edge placement error at tile boundaries.
* :func:`cross_tile_contour_residual` — contour position offset at tile
  boundaries.
* :func:`sweep_overlap_convergence` — sweep overlap width and Schwarz iteration
  count to produce convergence surfaces.
* :func:`schwarz_vs_naive_comparison` — compare Schwarz stitching against naive
  zero-fill stitching.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn.functional as functional

from openlithohub.workflow.tiling import Tile, tile_layout


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


# ---------------------------------------------------------------------------
# Cross-tile boundary EPE and contour residuals
# ---------------------------------------------------------------------------


def cross_tile_epe_residual(
    tiles: list[torch.Tensor],
    tile_results: list[torch.Tensor],
    overlap: int,
    pixel_size_nm: float = 1.0,
) -> dict[str, float]:
    """Compute edge placement error at tile boundaries.

    For each pair of adjacent tiles, extract the overlapping region from each
    tile's result, compute edge maps via Sobel filtering, and measure the
    minimum symmetric distance between edge pixels in the two overlap strips.

    Args:
        tiles: Original tile tensors ``(H_tile, W_tile)``, one per tile.
            Only the spatial shape is used — the mask values are not inspected.
        tile_results: Optimised / simulated tensors, one per tile.
        overlap: Overlap width in pixels used when tiling.
        pixel_size_nm: Physical pixel pitch for converting distances to nm.

    Returns:
        Dictionary with:

        - ``'mean_epe'``: mean symmetric EPE across all boundaries (nm).
        - ``'max_epe'``: maximum EPE observed at any boundary (nm).
        - ``'epe_per_boundary'``: list of mean EPE values, one per adjacent
          tile pair that has a nonzero overlap.
    """
    if len(tiles) != len(tile_results):
        raise ValueError(
            f"tiles and tile_results must have same length; "
            f"got {len(tiles)} vs {len(tile_results)}"
        )
    if not tiles or overlap <= 0:
        return {"mean_epe": 0.0, "max_epe": 0.0, "epe_per_boundary": []}

    epe_per_boundary: list[float] = []

    for i in range(len(tile_results)):
        for j in range(i + 1, len(tile_results)):
            ri = _squeeze(tile_results[i])
            rj = _squeeze(tile_results[j])
            ti_2d = _squeeze(tiles[i])
            tj_2d = _squeeze(tiles[j])

            # Extract overlap strips from each tile's result
            strips = _boundary_strips(ti_2d, tj_2d, ri, rj, overlap)
            for sa, sb in strips:
                if sa is None:
                    continue
                epe = _symmetric_edge_distance(sa, sb, pixel_size_nm)
                if epe is not None:
                    epe_per_boundary.append(epe)

    if not epe_per_boundary:
        return {"mean_epe": 0.0, "max_epe": 0.0, "epe_per_boundary": []}

    return {
        "mean_epe": sum(epe_per_boundary) / len(epe_per_boundary),
        "max_epe": max(epe_per_boundary),
        "epe_per_boundary": epe_per_boundary,
    }


def cross_tile_contour_residual(
    tiles: list[torch.Tensor],
    tile_results: list[torch.Tensor],
    overlap: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute contour difference at tile boundaries.

    Binarise each tile's result at *threshold*, extract contours via gradient
    magnitude, then measure the mean and maximum offset between contour
    positions across the boundary.

    Args:
        tiles: Original tile tensors ``(H_tile, W_tile)``.
        tile_results: Optimised / simulated tensors, one per tile.
        overlap: Overlap width in pixels.
        threshold: Binarisation threshold for contour extraction.

    Returns:
        Dictionary with:

        - ``'mean_contour_offset'``: mean contour position offset (pixels).
        - ``'max_contour_offset'``: maximum offset (pixels).
        - ``'n_boundary_pixels'``: total number of boundary pixels examined.
    """
    if len(tiles) != len(tile_results):
        raise ValueError(
            f"tiles and tile_results must have same length; "
            f"got {len(tiles)} vs {len(tile_results)}"
        )
    if not tiles or overlap <= 0:
        return {"mean_contour_offset": 0.0, "max_contour_offset": 0.0, "n_boundary_pixels": 0}

    offsets: list[float] = []
    total_pixels = 0

    for i in range(len(tile_results)):
        for j in range(i + 1, len(tile_results)):
            ri = _squeeze(tile_results[i])
            rj = _squeeze(tile_results[j])
            ti_2d = _squeeze(tiles[i])
            tj_2d = _squeeze(tiles[j])

            strips = _boundary_strips(ti_2d, tj_2d, ri, rj, overlap)
            for sa, sb in strips:
                if sa is None:
                    continue
                bin_a = (sa >= threshold).float()
                bin_b = (sb >= threshold).float()
                grad_a = _gradient_magnitude(bin_a)
                grad_b = _gradient_magnitude(bin_b)
                # Contour pixels: where gradient is nonzero
                contour_mask = (grad_a > 0) | (grad_b > 0)
                n_px = int(contour_mask.sum().item())
                if n_px == 0:
                    continue
                diff = (bin_a - bin_b).abs()
                pixel_offsets = diff[contour_mask]
                offsets.append(float(pixel_offsets.mean().item()))
                total_pixels += n_px

    if not offsets:
        return {"mean_contour_offset": 0.0, "max_contour_offset": 0.0, "n_boundary_pixels": 0}

    return {
        "mean_contour_offset": sum(offsets) / len(offsets),
        "max_contour_offset": max(offsets),
        "n_boundary_pixels": float(total_pixels),
    }


def sweep_overlap_convergence(
    mask: torch.Tensor,
    tile_size: int,
    forward_fn: Callable,
    overlap_range: list[int] | None = None,
    schwarz_iter_range: list[int] | None = None,
) -> dict[str, dict]:
    """Sweep overlap x Schwarz iterations and collect convergence metrics.

    For each (overlap, n_iterations) combination:

    1. Tile the mask with the given overlap.
    2. Run Schwarz-style iterations (apply *forward_fn* per tile, stitch).
    3. Measure seam residual, EPE, and contour offset.

    Args:
        mask: Full-chip mask tensor ``(H, W)``.
        tile_size: Tile size in pixels.
        forward_fn: Callable ``(H_tile, W_tile) -> (H_tile, W_tile)`` applied
            per tile per iteration.
        overlap_range: List of overlap widths to sweep.
        schwarz_iter_range: List of Schwarz iteration counts to sweep.

    Returns:
        Nested dict ``results[overlap_str][n_iter_str]`` mapping to a dict with
        keys ``seam_residual``, ``epe``, ``contour_offset``, ``time``.
    """
    if overlap_range is None:
        overlap_range = [4, 8, 16, 32]
    if schwarz_iter_range is None:
        schwarz_iter_range = [1, 3, 5, 10]

    if mask.ndim > 2:
        mask = mask.squeeze()
    h, w = mask.shape

    results: dict[str, dict] = {}

    for overlap in overlap_range:
        if overlap >= tile_size:
            continue
        ol_key = str(overlap)
        results[ol_key] = {}

        tiles = tile_layout(mask, tile_size=tile_size, overlap=overlap)
        tile_data = [t.tensor.clone() for t in tiles]

        cumulative_results = list(tile_data)

        for n_iter in schwarz_iter_range:
            t0 = time.perf_counter()

            # Run additional iterations (cumulative)
            cumulative_results = _run_schwarz_iteration(
                tiles, cumulative_results, forward_fn, overlap
            )

            elapsed = time.perf_counter() - t0

            # Measure seam residual (boundary MSE)
            consistency = tile_boundary_consistency(tiles, cumulative_results, overlap=overlap)

            # Measure EPE residual
            epe_result = cross_tile_epe_residual(
                [t.tensor for t in tiles], cumulative_results, overlap=overlap
            )

            # Measure contour residual
            contour_result = cross_tile_contour_residual(
                [t.tensor for t in tiles], cumulative_results, overlap=overlap
            )

            results[ol_key][str(n_iter)] = {
                "seam_residual": consistency["boundary_mse"],
                "epe": epe_result["mean_epe"],
                "contour_offset": contour_result["mean_contour_offset"],
                "time": elapsed,
            }

    return results


def schwarz_vs_naive_comparison(
    mask: torch.Tensor,
    tile_size: int,
    overlap: int,
    n_schwarz_iter: int = 5,
    forward_fn: Callable | None = None,
) -> dict[str, float]:
    """Compare Schwarz stitching vs naive zero-fill stitching.

    The naive approach simply runs *forward_fn* on each tile independently and
    stitches with linear blending (no boundary exchange). The Schwarz approach
    exchanges overlap data between adjacent tiles between iterations.

    Args:
        mask: Full-chip mask tensor ``(H, W)``.
        tile_size: Tile size in pixels.
        overlap: Overlap width in pixels.
        n_schwarz_iter: Number of Schwarz iterations.
        forward_fn: Per-tile forward callable. Defaults to Gaussian aerial
            image simulation with ``sigma_px=2.0``.

    Returns:
        Dictionary with:

        - ``'schwarz_seam_error'``: boundary MSE after Schwarz iterations.
        - ``'naive_seam_error'``: boundary MSE with independent tiles.
        - ``'improvement_ratio'``: ``naive / schwarz`` (> 1 means Schwarz is
          better).
    """
    if forward_fn is None:
        from openlithohub._utils.forward_model import simulate_aerial_image

        def forward_fn(tile: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
            return simulate_aerial_image(tile, sigma_px=2.0)

    if mask.ndim > 2:
        mask = mask.squeeze()
    h, w = mask.shape

    tiles = tile_layout(mask, tile_size=tile_size, overlap=overlap)

    # --- Naive: independent per-tile forward, no boundary exchange ---
    naive_results: list[torch.Tensor] = []
    for tile in tiles:
        result = tile.tensor.clone()
        for _ in range(n_schwarz_iter):
            result = forward_fn(result)
        naive_results.append(result)

    naive_consistency = tile_boundary_consistency(tiles, naive_results, overlap=overlap)
    naive_mse = naive_consistency["boundary_mse"]

    # --- Schwarz: iterative boundary exchange ---
    schwarz_results = [t.tensor.clone() for t in tiles]
    for _ in range(n_schwarz_iter):
        schwarz_results = _run_schwarz_iteration(tiles, schwarz_results, forward_fn, overlap)

    schwarz_consistency = tile_boundary_consistency(tiles, schwarz_results, overlap=overlap)
    schwarz_mse = schwarz_consistency["boundary_mse"]

    improvement = naive_mse / schwarz_mse if schwarz_mse > 0 else 1.0

    return {
        "schwarz_seam_error": schwarz_mse,
        "naive_seam_error": naive_mse,
        "improvement_ratio": improvement,
    }


# ---------------------------------------------------------------------------
# Private helpers for the new metrics
# ---------------------------------------------------------------------------


def _run_schwarz_iteration(
    tiles: list[Tile],
    current_results: list[torch.Tensor],
    forward_fn: Callable,
    overlap: int,
) -> list[torch.Tensor]:
    """Run one Schwarz iteration: exchange boundaries, then apply forward_fn."""
    from openlithohub.workflow.tiling import _inject_boundary_data

    new_results: list[torch.Tensor] = []
    for idx, tile in enumerate(tiles):
        updated = _inject_boundary_data(tile, current_results, idx, tiles, overlap)
        updated = forward_fn(updated)
        new_results.append(updated)
    return new_results


def _boundary_strips(
    ti: torch.Tensor,
    tj: torch.Tensor,
    ri: torch.Tensor,
    rj: torch.Tensor,
    overlap: int,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Extract boundary-adjacent overlap strips from two tile results.

    Returns pairs of (strip_i, strip_j) where strips are *overlap* pixels wide
    along the shared edge. This differs from ``_overlap_regions`` which
    extracts the exact overlap patch — here we just want the boundary-adjacent
    strip for EPE/contour comparison.
    """
    hi, wi = ri.shape[-2], ri.shape[-1]
    hj, wj = rj.shape[-2], rj.shape[-1]
    ol = min(overlap, wi, wj, hi, hj)
    if ol <= 0:
        return [(None, None)]

    strips: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    # Right edge of ri vs left edge of rj (horizontal adjacency)
    strip_r = ri[..., :, -ol:]
    strip_l = rj[..., :, :ol]
    if strip_r.shape == strip_l.shape:
        strips.append((strip_r, strip_l))

    # Bottom edge of ri vs top edge of rj (vertical adjacency)
    strip_b = ri[..., -ol:, :]
    strip_t = rj[..., :ol, :]
    if strip_b.shape == strip_t.shape:
        strips.append((strip_b, strip_t))

    return strips if strips else [(None, None)]


def _symmetric_edge_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    pixel_size_nm: float,
) -> float | None:
    """Compute mean symmetric edge distance between two same-shape tensors."""
    edges_a = _extract_edge_pixels(a)
    edges_b = _extract_edge_pixels(b)

    pts_a = edges_a.nonzero(as_tuple=False).float()
    pts_b = edges_b.nonzero(as_tuple=False).float()

    if pts_a.numel() == 0 and pts_b.numel() == 0:
        return 0.0
    if pts_a.numel() == 0 or pts_b.numel() == 0:
        return None

    dists_a = _min_pairwise_row_distances(pts_a, pts_b)
    dists_b = _min_pairwise_row_distances(pts_b, pts_a)
    combined = torch.cat([dists_a, dists_b]) * pixel_size_nm
    return float(combined.mean().item())


def _extract_edge_pixels(tensor: torch.Tensor) -> torch.Tensor:
    """Extract a boolean edge map via Sobel filtering."""
    inp = (tensor > 0.5).float().unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=tensor.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=tensor.device,
    ).reshape(1, 1, 3, 3)
    gx = functional.conv2d(inp, sobel_x, padding=1)
    gy = functional.conv2d(inp, sobel_y, padding=1)
    magnitude = (gx.square() + gy.square()).sqrt().squeeze()
    return magnitude > 0.0


def _min_pairwise_row_distances(
    source: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """For each row in source, distance to nearest row in reference."""
    chunk = 4096
    parts: list[torch.Tensor] = []
    for i in range(0, source.shape[0], chunk):
        s = source[i : i + chunk]
        running = torch.full((s.shape[0],), float("inf"), device=s.device)
        for j in range(0, reference.shape[0], chunk):
            r = reference[j : j + chunk]
            d = torch.cdist(s, r)
            running = torch.minimum(running, d.min(dim=1).values)
        parts.append(running)
    return torch.cat(parts)


def _gradient_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude of a 2D tensor via finite differences."""
    dy = torch.zeros_like(tensor)
    dx = torch.zeros_like(tensor)
    dy[:-1, :] = tensor[1:, :] - tensor[:-1, :]
    dx[:, :-1] = tensor[:, 1:] - tensor[:, :-1]
    return (dy.square() + dx.square()).sqrt()
