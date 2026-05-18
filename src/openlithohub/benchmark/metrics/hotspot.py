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

import torch


def compute_hotspot_detection(
    predicted_points: torch.Tensor,
    ground_truth_points: torch.Tensor,
    match_radius_nm: float = 1.0,
) -> dict[str, float]:
    """Score a hotspot predictor against a ground-truth point list.

    A predicted point is a true positive iff at least one *unmatched* GT
    point lies within ``match_radius_nm`` of it. Matching is greedy in
    predicted-point order; this is standard for hotspot detection
    benchmarks where the predictor's confidence ranking is not part of
    the contest scoring.

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
            f"ground_truth_points must have shape (M, 2), got "
            f"{tuple(ground_truth_points.shape)}"
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

    matched_gt = torch.zeros(n_gt, dtype=torch.bool, device=gt.device)
    num_tp = 0
    radius = float(match_radius_nm)
    for i in range(n_pred):
        candidate_mask = (dists[i] <= radius) & (~matched_gt)
        if not candidate_mask.any():
            continue
        # Closest unmatched GT inside the disk wins this prediction.
        candidate_dists = torch.where(
            candidate_mask, dists[i], torch.full_like(dists[i], float("inf"))
        )
        j = int(candidate_dists.argmin().item())
        matched_gt[j] = True
        num_tp += 1

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
