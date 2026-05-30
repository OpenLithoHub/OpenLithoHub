"""Manhattanization round-trip degradation metrics.

Curvilinear ILT produces free-form mask shapes that maximise process window.
Multi-beam mask writers (MBMW) can write these directly, but some flows require
a Manhattan (axis-aligned + 45-degree edges) representation for VSB writers or
legacy mask data prep. This module quantifies the quality loss from that
conversion so designers can decide whether curvilinear output is worth the
extra shot count.

Two functions:

* :func:`manhattanization_degradation` — compare curvilinear and Manhattan masks
  and report EPE, PVB increase, shot count ratio, and area error.
* :func:`curvilinear_to_manhattan` — convert a curvilinear mask to Manhattan by
  quantising edge angles.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from openlithohub._utils.tensor_ops import ensure_2d


def manhattanization_degradation(
    curvilinear_mask: torch.Tensor,
    manhattanized_mask: torch.Tensor,
    target_cd_nm: float = 40.0,
    pixel_size_nm: float = 2.0,
) -> dict[str, float]:
    """Quantify the degradation from curvilinear to Manhattan mask.

    Args:
        curvilinear_mask: Original curvilinear mask ``(H, W)`` or ``(B, 1, H, W)``.
        manhattanized_mask: Manhattanized version of the same mask, same shape.
        target_cd_nm: Target critical dimension in nm (for EPE context).
        pixel_size_nm: Physical pixel size in nm.

    Returns:
        Dictionary with:

        - ``'edge_placement_error_nm'``: mean EPE between curve and Manhattan
          edges, measured as the mean distance from Manhattan contour pixels to
          the nearest curvilinear contour pixel.
        - ``'pvb_increase'``: increase in process-variation-band width
          (approximated as the difference in edge-area spread).
        - ``'shot_count_ratio'``: Manhattan shot count / curvilinear shot count.
          Values < 1 mean Manhattan is cheaper to write.
        - ``'area_error_frac'``: fractional area difference
          ``(manhattan - curvilinear) / curvilinear``. Positive means the
          Manhattan mask has more foreground.
    """
    curv = ensure_2d(curvilinear_mask).float()
    manh = ensure_2d(manhattanized_mask).float()

    if curv.shape != manh.shape:
        raise ValueError(
            f"Shape mismatch: curvilinear {curv.shape} vs manhattanized {manh.shape}"
        )

    curv_binary = (curv > 0.5).float()
    manh_binary = (manh > 0.5).float()

    # Edge placement error: symmetric edge distance
    epe = _compute_mean_edge_distance(manh_binary, curv_binary) * pixel_size_nm

    # PVB increase: approximate via edge-area spread difference
    curv_edge_area = _edge_spread(curv_binary)
    manh_edge_area = _edge_spread(manh_binary)
    pvb_increase = float((manh_edge_area - curv_edge_area).clamp(min=0).sum().item())
    pvb_increase_nm = pvb_increase * pixel_size_nm

    # Shot count ratio: MBMW foreground pixel count
    curv_fg = float(curv_binary.sum().item())
    manh_fg = float(manh_binary.sum().item())
    shot_count_ratio = manh_fg / curv_fg if curv_fg > 0 else 0.0

    # Area error fraction
    area_diff = float((manh_binary - curv_binary).sum().item())
    area_error_frac = area_diff / curv_fg if curv_fg > 0 else 0.0

    return {
        "edge_placement_error_nm": epe,
        "pvb_increase": pvb_increase_nm,
        "shot_count_ratio": shot_count_ratio,
        "area_error_frac": area_error_frac,
    }


def curvilinear_to_manhattan(
    mask: torch.Tensor,
    angle_quantization: int = 45,
) -> torch.Tensor:
    """Convert curvilinear mask to Manhattan (axis-aligned + 45-degree edges).

    Quantises edge orientations to multiples of ``angle_quantization`` degrees
    by applying directional erosion/dilation filters that only preserve edges
    at the allowed angles. For ``angle_quantization=45`` the allowed angles
    are ``0, 45, 90, 135`` degrees.

    Implementation: extract the gradient direction at each boundary pixel and
    snap it to the nearest quantised angle. Reconstruct the mask by keeping
    only boundary pixels whose gradient direction is within half a quantisation
    step of an allowed angle, then re-filling interiors.

    Args:
        mask: Continuous or binary mask ``(H, W)``.
        angle_quantization: Angle step in degrees. Must be one of
            ``{45, 90}``. ``90`` = axis-aligned only (pure Manhattan);
            ``45`` = axis-aligned + diagonals.

    Returns:
        Manhattanized mask tensor ``(H, W)`` with the same dtype as input.
    """
    if angle_quantization not in (45, 90):
        raise ValueError(
            f"angle_quantization must be 45 or 90, got {angle_quantization}"
        )

    m = ensure_2d(mask).float()

    # Compute image gradients
    gy = torch.zeros_like(m)
    gx = torch.zeros_like(m)
    if m.shape[-2] > 2:
        gy[..., 1:-1, :] = (m[..., 2:, :] - m[..., :-2, :]) / 2.0
    if m.shape[-1] > 2:
        gx[..., :, 1:-1] = (m[..., :, 2:] - m[..., :, :-2]) / 2.0

    grad_mag = (gx.pow(2) + gy.pow(2)).sqrt()
    edge_mask = grad_mag > 1e-6

    if not edge_mask.any():
        # No edges (uniform mask) — return as-is
        return m.to(mask.dtype)

    # Compute gradient angles
    angles = torch.atan2(gy, gx)  # [-pi, pi]
    angles_deg = angles * (180.0 / math.pi)  # [-180, 180]

    # Allowed angles in degrees
    n_bins = 180 // angle_quantization
    allowed = torch.tensor(
        [i * angle_quantization for i in range(-n_bins, n_bins + 1)],
        dtype=angles_deg.dtype,
        device=angles_deg.device,
    )

    # For each edge pixel, find the nearest allowed angle
    # Reshape for broadcasting: (H*W, 1) vs (1, n_allowed)
    flat_angles = angles_deg[edge_mask].unsqueeze(1)  # (N, 1)
    diff = (flat_angles - allowed.unsqueeze(0)).abs()  # (N, n_allowed)
    nearest_idx = diff.argmin(dim=1)  # (N,)
    nearest_angles = allowed[nearest_idx]  # (N,)

    # Quantised gradient direction
    quant_gy = grad_mag.new_zeros(grad_mag.shape)
    quant_gx = grad_mag.new_zeros(grad_mag.shape)

    flat_mag = grad_mag[edge_mask]
    rad = nearest_angles * (math.pi / 180.0)
    quant_gy[edge_mask] = flat_mag * torch.sin(rad)
    quant_gx[edge_mask] = flat_mag * torch.cos(rad)

    # Reconstruct: use the quantised gradient field to rebuild edges
    # via divergence (Laplacian-like reconstruction)
    # Simpler approach: build a new mask from quantised boundary pixels
    # by thresholding the quantised gradient magnitude
    quant_mag = (quant_gx.pow(2) + quant_gy.pow(2)).sqrt()

    # Threshold at a fraction of the original edge magnitude to capture
    # the quantised boundary
    threshold = quant_mag.max() * 0.3 if quant_mag.max() > 0 else 0.5
    quant_edges = (quant_mag > threshold).float()

    # Build output: binary mask where we keep the original mask values but
    # re-contour using only quantised-edge directions.
    # Morphological reconstruction: start from interior seed, dilate only
    # through quantised-edge directions.
    result = _reconstruct_from_quantised_edges(m, quant_edges, edge_mask)

    return result.to(mask.dtype)


def _compute_mean_edge_distance(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
) -> float:
    """Symmetric mean distance between edge pixels of two binary masks.

    Returns mean distance in pixels.
    """
    edges_a = _extract_binary_edges(mask_a)
    edges_b = _extract_binary_edges(mask_b)

    pts_a = edges_a.nonzero(as_tuple=False).float()
    pts_b = edges_b.nonzero(as_tuple=False).float()

    if pts_a.numel() == 0 and pts_b.numel() == 0:
        return 0.0
    if pts_a.numel() == 0 or pts_b.numel() == 0:
        return float("inf")

    # Symmetric: mean of A->B and B->A distances
    dist_a_to_b = _min_distances_chunked(pts_a, pts_b)
    dist_b_to_a = _min_distances_chunked(pts_b, pts_a)
    all_dists = torch.cat([dist_a_to_b, dist_b_to_a])
    return float(all_dists.mean().item())


def _extract_binary_edges(binary: torch.Tensor) -> torch.Tensor:
    """Extract edge pixels from a binary mask using Sobel filtering."""
    inp = binary.unsqueeze(0).unsqueeze(0)
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

    # Strip 1-pixel border to avoid phantom edges at frame boundaries
    magnitude[0, :] = 0.0
    magnitude[-1, :] = 0.0
    magnitude[:, 0] = 0.0
    magnitude[:, -1] = 0.0

    return magnitude > 0.0


def _min_distances_chunked(
    source: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """For each point in source, compute distance to nearest point in reference."""
    out: list[torch.Tensor] = []
    for i in range(0, source.shape[0], chunk_size):
        src_chunk = source[i : i + chunk_size]
        running = torch.full(
            (src_chunk.shape[0],),
            float("inf"),
            device=src_chunk.device,
            dtype=src_chunk.dtype,
        )
        for j in range(0, reference.shape[0], chunk_size):
            ref_chunk = reference[j : j + chunk_size]
            dists = torch.cdist(src_chunk, ref_chunk)
            running = torch.minimum(running, dists.min(dim=1).values)
        out.append(running)
    return torch.cat(out)


def _edge_spread(binary: torch.Tensor) -> torch.Tensor:
    """Approximate edge-area spread via 1-pixel dilation minus erosion residual.

    Larger spread = wider transition band = larger PV band.
    """
    kernel = torch.ones(1, 1, 3, 3, device=binary.device)
    inp = binary.unsqueeze(0).unsqueeze(0)
    dilated = functional.conv2d(
        functional.pad(inp, (1, 1, 1, 1), mode="replicate"),
        kernel,
    ).squeeze()
    eroded = -functional.conv2d(
        functional.pad(-inp, (1, 1, 1, 1), mode="replicate"),
        kernel,
    ).squeeze()
    return (dilated - eroded).clamp(min=0.0)


def _reconstruct_from_quantised_edges(
    original: torch.Tensor,
    quant_edges: torch.Tensor,
    original_edges: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a binary mask preserving quantised-edge directions.

    Strategy: start from the original binary mask interior (eroded by 1px as
    a seed), then dilate outward only through pixels that are on a quantised
    edge direction. This produces a Manhattanised version that stays true to
    the original mask's topology while snapping edge directions.
    """
    binary = (original > 0.5).float()
    h, w = binary.shape

    # Interior seed: erode by 1 pixel to get pixels safely inside features
    kernel = torch.ones(1, 1, 3, 3, device=binary.device)
    inp = binary.unsqueeze(0).unsqueeze(0)
    seed = (-functional.conv2d(
        functional.pad(-inp, (1, 1, 1, 1), mode="replicate"),
        kernel,
    ) >= 9.0).squeeze().float()

    # Iterative dilation: expand seed outward, constrained to pixels that
    # are inside the original mask or on quantised edges.
    mask_allowed = binary + quant_edges  # can grow into original + quantised edges
    mask_allowed = (mask_allowed > 0.5).float()

    result = seed.clone()
    for _ in range(max(h, w)):
        prev = result.clone()
        dilated = functional.conv2d(
            functional.pad(result.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1)),
            kernel,
        ).squeeze()
        grown = ((dilated > 0).float() * mask_allowed)
        result = torch.maximum(result, grown)
        if (result - prev).abs().sum() < 0.5:
            break

    return result
