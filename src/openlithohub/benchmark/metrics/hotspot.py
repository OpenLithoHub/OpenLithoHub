"""Hotspot detection metric — recall / precision / F1 with distance-tolerant
matching against a ground-truth point list.

This is the canonical evaluation used by ICCAD'16 Problem C and the
hotspot-detection literature (e.g. Yang et al., TCAD 2020): a predicted
point counts as a true positive if any ground-truth point lies within a
configurable radius (``match_radius_nm``). Each GT point may be matched
at most once — duplicate predictions inside the same tolerance disk
become false positives. GT points with no predictor inside the disk are
false negatives.

The matching is point-based, not pixel-based. If your predictor outputs
a binary heatmap, run connected-components and feed the centroids (in
nm) as ``predicted_points``. ``openlithohub._utils.morphology`` has the
primitives — there is no need to reinvent them.

Coordinates are in nanometers throughout to match the rest of the
benchmark stack (LithoSample.metadata exposes nm units consistently).
"""

from __future__ import annotations

import numpy as np
import torch


def compute_hotspot_detection(
    predicted_points: torch.Tensor,
    ground_truth_points: torch.Tensor,
    match_radius_nm: float = 1.0,
) -> dict[str, float]:
    """Score a hotspot predictor against a ground-truth point list.

    A predicted point is a true positive iff it can be paired with a GT
    point within ``match_radius_nm``, under a *maximum-cardinality
    minimum-cost* assignment (Hungarian algorithm) — independent of the
    order ``predicted_points`` arrives in.

    Args:
        predicted_points: ``(N, 2)`` tensor of predicted hotspot
            centers in nm. Pass an empty ``(0, 2)`` tensor for an empty
            prediction.
        ground_truth_points: ``(M, 2)`` tensor of ground-truth hotspot
            centers in nm. Pass an empty ``(0, 2)`` tensor when no
            hotspots exist for the case.
        match_radius_nm: Maximum nm distance at which a predicted point
            is considered to have located a GT hotspot. ICCAD'16
            literature commonly uses 1 nm (exact-pixel match) or a few
            nm to allow for centroid jitter.

    Returns:
        Dict with ``num_tp``, ``num_fp``, ``num_fn``, ``recall``,
        ``precision``, ``f1``. Counts are returned as floats so the
        result merges cleanly with other ``dict[str, float]`` metrics.

        Edge cases:

        - No GT and no predictions → recall/precision/F1 = 1.0 (vacuous
            perfect score). This convention matches sklearn's behavior
            when ``y_true`` and ``y_pred`` are both empty.
        - GT present but no predictions → recall=0, precision=1.0
            (vacuously: nothing predicted, so nothing is wrong), F1=0.
        - Predictions present but no GT → recall=1.0, precision=0, F1=0.
    """
    if predicted_points.ndim != 2 or predicted_points.shape[-1] != 2:
        raise ValueError(
            f"predicted_points must have shape (N, 2), got {tuple(predicted_points.shape)}"
        )
    if ground_truth_points.ndim != 2 or ground_truth_points.shape[-1] != 2:
        raise ValueError(
            f"ground_truth_points must have shape (M, 2), got {tuple(ground_truth_points.shape)}"
        )
    if match_radius_nm < 0:
        raise ValueError(f"match_radius_nm must be >= 0, got {match_radius_nm}")

    n_pred = predicted_points.shape[0]
    n_gt = ground_truth_points.shape[0]

    if n_pred == 0 and n_gt == 0:
        return {
            "num_tp": 0.0,
            "num_fp": 0.0,
            "num_fn": 0.0,
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
        }
    if n_pred == 0:
        return {
            "num_tp": 0.0,
            "num_fp": 0.0,
            "num_fn": float(n_gt),
            "recall": 0.0,
            "precision": 1.0,
            "f1": 0.0,
        }
    if n_gt == 0:
        return {
            "num_tp": 0.0,
            "num_fp": float(n_pred),
            "num_fn": 0.0,
            "recall": 1.0,
            "precision": 0.0,
            "f1": 0.0,
        }

    pred = predicted_points.float()
    gt = ground_truth_points.float()
    dists = torch.cdist(pred, gt)  # (N, M)

    radius = float(match_radius_nm)
    # Mark out-of-disk pairs with a large finite cost so the Hungarian
    # solver still runs on rectangular matrices (it requires a square cost
    # internally, but linear_sum_assignment handles non-square fine; only
    # finite costs matter). We post-filter those infeasible pairings.
    # Lazy scipy import: scipy lives in the [workflow] extra, but
    # importing this module unconditionally pulls scipy into every test
    # shard via benchmark/metrics/__init__.py. Defer to call time so
    # only callers that actually score hotspots pay the dependency.
    from scipy.optimize import linear_sum_assignment

    cost = dists.detach().cpu().numpy().astype(np.float64)
    big = radius + 1.0  # any value strictly greater than radius
    cost_for_solver = np.where(cost <= radius, cost, big)
    pred_idx, gt_idx = linear_sum_assignment(cost_for_solver)
    # A pair counts only if the *original* distance was inside the disk.
    valid = cost[pred_idx, gt_idx] <= radius
    num_tp = int(valid.sum())

    num_fp = n_pred - num_tp
    num_fn = n_gt - num_tp
    recall = num_tp / n_gt
    precision = num_tp / n_pred
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return {
        "num_tp": float(num_tp),
        "num_fp": float(num_fp),
        "num_fn": float(num_fn),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
    }
