"""Batch active sampling for hotspot pattern selection (Yang2020 §III).

Implements the layout-pattern sampling acquisition function from
[Yang2020_BatchAL] §III (arXiv:1807.06446). The paper proposes an
active-learning loop that iteratively (a) trains a hotspot detector,
(b) scores an unlabelled clip pool with the detector, (c) selects a
batch of `k` clips that are simultaneously high-uncertainty and
mutually diverse, (d) labels them via lithography simulation and adds
them to the training set.

This module ships only the **acquisition function** — i.e. the (b)+(c)
selection step given pre-computed clip features and per-clip predictive
probabilities. The detector training loop (a) and the labelling oracle
(d) are out of scope: OpenLithoHub does not currently ship a hotspot
detector, and the publicly mirrored ICCAD-2016 corpus on disk has only
one testcase, which is too small to demonstrate the full AL loop
honestly. Wiring a real loop is tracked in
``out/plans/external-resource-utilization.md`` Task #1 v0.2.

Algorithm faithfulness vs. paper:

- **Uncertainty score** uses Eq. (8) ``c(i) = p(y=1 | x_i; w_t)`` —
  the predicted hotspot probability — exactly as the paper recommends
  for hotspot tasks (preferred over the Eq. (7) entropy because
  problematic / hotspot instances are of more interest than uncertain
  non-hotspots).
- **Diversity matrix** uses Eq. (9) ``D[i,j] = x_i^T x_j`` on
  L2-normalised feature vectors. ``D`` is a positive-semidefinite Gram
  matrix; its diagonal equals 1 by construction.
- **Batch selection** diverges from the paper's QP relaxation
  (Eq. (10)/(11)). The paper relaxes the binary 0/1 selection to
  ``[0,1]^n``, solves a quadratic program ``min m^T D m s.t. sum=k``,
  and recovers the integer solution by picking the `k` largest entries
  of `m`. Theorem 1 bounds the rounding gap by
  ``2 λ_n (k − k²/n)`` — small at the paper's recommended ``k=60,
  n=90``. Here we use a simpler **greedy max-min selection**: from the
  top-``n`` highest-uncertainty candidates, repeatedly add the
  candidate that maximises its minimum cosine distance to the already-
  selected set. This avoids a QP solver dependency, runs in
  ``O(n·k)``, is deterministic, and empirically tracks the QP solution
  on Gram-matrix problems of this size.
"""

from __future__ import annotations

import torch


def _l2_normalise(features: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalise rows of ``features`` for cosine-similarity diversity."""
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True).clamp_min(eps)
    out: torch.Tensor = features / norms
    return out


def batch_active_select(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    k: int,
    n: int | None = None,
) -> torch.Tensor:
    """Pick ``k`` pool indices with high uncertainty and mutual diversity.

    Implements the [Yang2020_BatchAL] §III acquisition function with the
    greedy-rounding variant documented at module level.

    Args:
        features: ``(P, F)`` feature tensor for a pool of ``P`` clips.
            Rows are L2-normalised internally; callers do not need to
            pre-normalise.
        probabilities: ``(P,)`` predicted hotspot probability for each
            pool entry, i.e. ``p(y=1 | x_i; w_t)`` (Eq. 8). Must be in
            ``[0, 1]``.
        k: Number of clips to select into the batch. Must satisfy
            ``1 <= k <= P``.
        n: Size of the candidate prefilter — pre-select the top-``n``
            most-uncertain clips, then run greedy diversity selection on
            that subset (Algorithm 1, line 5: "sample n instances with
            highest probability"). Defaults to ``min(P, max(k, 3*k//2))``
            to give the diversity step room to work without scanning the
            entire pool. Must satisfy ``k <= n <= P``.

    Returns:
        ``(k,)`` ``torch.long`` tensor of selected indices into
        ``features``. Ordering is the order in which the greedy step
        added each clip — first index is the highest-uncertainty seed,
        subsequent indices are the diversity-driven additions.

    Raises:
        ValueError: if shapes/ranges are inconsistent.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (P, F); got shape {tuple(features.shape)}")
    pool_size, _ = features.shape
    if probabilities.shape != (pool_size,):
        raise ValueError(
            f"probabilities must have shape ({pool_size},); got {tuple(probabilities.shape)}"
        )
    if float(probabilities.min()) < 0 or float(probabilities.max()) > 1:
        raise ValueError("probabilities must lie in [0, 1]")
    if not 1 <= k <= pool_size:
        raise ValueError(f"k={k} must satisfy 1 <= k <= P={pool_size}")
    if n is None:
        n = min(pool_size, max(k, (3 * k) // 2))
    if not k <= n <= pool_size:
        raise ValueError(f"n={n} must satisfy k={k} <= n <= P={pool_size}")

    candidate_idx = torch.topk(probabilities, k=n, largest=True, sorted=True).indices
    cand_features = _l2_normalise(features[candidate_idx])

    selected_local: list[int] = [0]
    similarity_to_selected = cand_features @ cand_features[0]

    for _ in range(k - 1):
        masked = similarity_to_selected.clone()
        masked[selected_local] = float("inf")
        next_local = int(torch.argmin(masked).item())
        selected_local.append(next_local)
        new_sims = cand_features @ cand_features[next_local]
        similarity_to_selected = torch.maximum(similarity_to_selected, new_sims)

    return candidate_idx[torch.tensor(selected_local, dtype=torch.long)]


def extract_clip_features(
    design: torch.Tensor,
    clip_sites: list[dict[str, float]],
    *,
    pixel_nm: float,
    origin_nm: tuple[float, float] = (0.0, 0.0),
    feature_dim: int = 16,
) -> torch.Tensor:
    """Build a per-clip feature pool from the design tensor.

    For each entry in ``clip_sites`` (the auxiliary inspection-grid
    boxes from the ICCAD16 ``(layer=10000, datatype=0)`` layer), crop
    the design tensor at the clip footprint and downsample to a
    ``feature_dim × feature_dim`` patch. The flattened patch is the
    feature vector. This is a deliberately simple density-style
    featuriser (paper §3.3 says they use the "second channel of the
    feature tensor" as the diversity feature, i.e. a frequency-domain
    summary, but that channel is not available here without the
    [Yang2014_FeatureTensor] extractor — which is upstream of OpenLitho-
    Hub's scope). Callers swapping in a learned encoder should call
    :func:`batch_active_select` directly with their own features.

    Args:
        design: ``(H, W)`` binary design tensor, in pixel coordinates.
        clip_sites: list of dicts with ``x0_nm, y0_nm, x1_nm, y1_nm``
            keys (the format produced by
            :class:`openlithohub.data.iccad16.Iccad16Dataset`).
        pixel_nm: nm-per-pixel of the design tensor.
        origin_nm: ``(ox_nm, oy_nm)`` design origin in nm. Clip-site
            coordinates are translated by this offset before pixel
            conversion.
        feature_dim: target side length in pixels for each pooled patch.

    Returns:
        ``(P, feature_dim*feature_dim)`` float tensor.
    """
    if design.ndim != 2:
        raise ValueError(f"design must be 2-D (H, W); got shape {tuple(design.shape)}")
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be >= 1; got {feature_dim}")
    height_px, width_px = int(design.shape[0]), int(design.shape[1])
    ox_nm, oy_nm = origin_nm
    patches: list[torch.Tensor] = []
    design_f = design.to(torch.float32)
    for site in clip_sites:
        x0_px = int(round((site["x0_nm"] - ox_nm) / pixel_nm))
        y0_px = int(round((site["y0_nm"] - oy_nm) / pixel_nm))
        x1_px = int(round((site["x1_nm"] - ox_nm) / pixel_nm))
        y1_px = int(round((site["y1_nm"] - oy_nm) / pixel_nm))
        x0_px, x1_px = sorted((max(0, x0_px), min(width_px, x1_px)))
        y0_px, y1_px = sorted((max(0, y0_px), min(height_px, y1_px)))
        if x1_px - x0_px < 1 or y1_px - y0_px < 1:
            patches.append(torch.zeros(feature_dim * feature_dim, dtype=torch.float32))
            continue
        crop = design_f[y0_px:y1_px, x0_px:x1_px]
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            crop.unsqueeze(0).unsqueeze(0), output_size=(feature_dim, feature_dim)
        )
        patches.append(pooled.flatten())
    if not patches:
        return torch.zeros(0, feature_dim * feature_dim, dtype=torch.float32)
    return torch.stack(patches, dim=0)
