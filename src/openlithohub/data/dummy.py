"""Procedural dummy layout generator for CI, debugging, and onboarding.

These layouts are *not* representative of real cell libraries — they exist so
that you can exercise the OpenLithoHub pipeline end-to-end without downloading
LithoBench or LithoSim, and so CI can run hermetically without network or large
data fixtures.

The generator only uses ``numpy`` and ``torch``. It does not depend on KLayout
or any of the heavy ``[workflow]`` extras, which keeps it usable in Colab and
on minimal CI images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from openlithohub._utils.morphology import binary_dilation, binary_erosion


@dataclass(frozen=True)
class DummyLayoutSpec:
    """Parameters controlling the generated layout."""

    size: int = 256
    pixel_size_nm: float = 1.0
    min_width_nm: float = 40.0
    min_spacing_nm: float = 40.0
    fill_ratio: float = 0.25
    seed: int | None = 0


def _enforce_min_rules(mask: torch.Tensor, min_width_px: int, min_spacing_px: int) -> torch.Tensor:
    """Apply opening (width) + closing-of-background (spacing) to satisfy DRC."""
    width_radius = max(1, min_width_px // 2)
    eroded = binary_erosion(mask, radius=width_radius)
    opened = binary_dilation(eroded, radius=width_radius)

    spacing_radius = max(1, min_spacing_px // 2)
    inverted = 1.0 - opened
    inv_eroded = binary_erosion(inverted, radius=spacing_radius)
    inv_opened = binary_dilation(inv_eroded, radius=spacing_radius)
    return 1.0 - inv_opened


def generate_dummy_layout(
    spec: DummyLayoutSpec | None = None,
    *,
    size: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a deterministic dummy binary layout that satisfies basic DRC.

    The result is a 2D ``torch.Tensor`` of shape (size, size) with values in
    {0.0, 1.0}. Polygons are placed by random rectangle splatting and then
    cleaned with morphological opening/closing so that minimum width and
    spacing rules are met for the configured pixel pitch.

    Args:
        spec: Full configuration; if omitted, a default 256 px / 40 nm spec is
            used and overridden by the keyword arguments.
        size: Convenience override for ``spec.size``.
        seed: Convenience override for ``spec.seed``.

    Returns:
        Binary mask tensor of shape (size, size).

    Examples:
        >>> mask = generate_dummy_layout(size=128, seed=0)
        >>> mask.shape
        torch.Size([128, 128])
    """
    if spec is None:
        spec = DummyLayoutSpec()
    if size is not None or seed is not None:
        spec = DummyLayoutSpec(
            size=size if size is not None else spec.size,
            pixel_size_nm=spec.pixel_size_nm,
            min_width_nm=spec.min_width_nm,
            min_spacing_nm=spec.min_spacing_nm,
            fill_ratio=spec.fill_ratio,
            seed=seed if seed is not None else spec.seed,
        )

    if spec.size < 32:
        raise ValueError(f"size must be >= 32, got {spec.size}")
    if not 0.0 < spec.fill_ratio < 1.0:
        raise ValueError(f"fill_ratio must be in (0, 1), got {spec.fill_ratio}")

    rng = np.random.default_rng(spec.seed)
    canvas = np.zeros((spec.size, spec.size), dtype=np.float32)

    min_width_px = max(2, int(round(spec.min_width_nm / spec.pixel_size_nm)))
    min_spacing_px = max(2, int(round(spec.min_spacing_nm / spec.pixel_size_nm)))

    target_pixels = spec.fill_ratio * spec.size * spec.size
    placed = 0
    attempts = 0
    max_attempts = 4000
    rect_min = min(min_width_px * 2, max(2, spec.size // 8))
    rect_max = max(rect_min + 1, min(spec.size // 4, spec.size - 1))

    while placed < target_pixels and attempts < max_attempts:
        attempts += 1
        w = int(rng.integers(rect_min, rect_max))
        h = int(rng.integers(rect_min, rect_max))
        if w >= spec.size or h >= spec.size:
            continue
        x = int(rng.integers(0, spec.size - w))
        y = int(rng.integers(0, spec.size - h))
        canvas[y : y + h, x : x + w] = 1.0
        placed = int(canvas.sum())

    mask = torch.from_numpy(canvas)
    mask = _enforce_min_rules(mask, min_width_px, min_spacing_px)
    return (mask > 0.5).float()


def generate_dummy_pair(
    spec: DummyLayoutSpec | None = None, **kwargs: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a ``(design, mask)`` pair where the mask is a dilated design.

    Useful for sanity-checking OPC pipelines without real ground truth.
    """
    design = generate_dummy_layout(spec, **kwargs)
    mask = binary_dilation(design, radius=2)
    return design, mask
