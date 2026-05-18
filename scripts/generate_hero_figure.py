"""Generate the OPC before/after hero figure used by docs and the marketing site.

Renders three panels for one synthetic layout:

1. **Design** — the target wafer pattern.
2. **No OPC** — aerial image when the design is printed verbatim
   (shows diffraction-induced rounding / pull-back at line ends and corners).
3. **With ILT** — aerial image after running the registered ``levelset-ilt``
   baseline, overlaid with the resist contour and the target outline so the
   optimisation gain is visible at a glance.

The forward model used here is the Gaussian-PSF surrogate from
``openlithohub._utils.forward_model``. It is a simplified stand-in for full
Hopkins partial coherence — chosen because its threshold/dose convention is
self-consistent at any grid size, which keeps this hero script deterministic
and easy to read. The point of the figure is the qualitative gap between
"verbatim print" and "ILT-optimized print"; both panels use the same physics.

Outputs:

- ``<output>/hero.png`` — the figure itself.
- ``<output>/hero.json`` — sidecar with the EPE numbers shown in the figure,
  so downstream consumers (mkdocs, marketing site) can render the numbers
  without hard-coding them.

Deterministic: same inputs produce the same PNG bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import openlithohub.models.levelset_ilt  # noqa: F401  (register)
from openlithohub._utils.forward_model import apply_resist_threshold, simulate_aerial_image
from openlithohub.benchmark.metrics.epe import compute_epe
from openlithohub.models.registry import registry

GRID = 128
SIGMA_PX = 4.5  # Gaussian-PSF radius — wide enough that small features pull back visibly.
RESIST_THRESHOLD = 0.5
PIXEL_SIZE_NM = 2.0


def build_layout(grid: int = GRID) -> torch.Tensor:
    """Synthetic via + line layout that makes diffraction loss visually obvious."""
    layout = torch.zeros(grid, grid)
    for cy in (grid // 4, 3 * grid // 4):
        for cx in (grid // 4, 3 * grid // 4):
            layout[cy - 6 : cy + 6, cx - 6 : cx + 6] = 1.0
    layout[grid // 2 - 3 : grid // 2 + 3, grid // 4 + 6 : 3 * grid // 4 - 6] = 1.0
    return layout


def render(
    design: torch.Tensor,
    aerial_no_opc: torch.Tensor,
    aerial_opc: torch.Tensor,
    resist_opc: torch.Tensor,
    epe_no_opc: float,
    epe_opc: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), dpi=150)

    titles = [
        "Design",
        f"Aerial — no OPC\nEPE = {epe_no_opc:.1f} nm",
        f"Aerial — with ILT\nEPE = {epe_opc:.1f} nm",
    ]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(design.numpy(), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")

    aerial_vmax = 1.0
    axes[1].imshow(aerial_no_opc.numpy(), cmap="inferno", vmin=0, vmax=aerial_vmax)
    axes[1].contour(design.numpy(), levels=[0.5], colors="white", linewidths=0.7, linestyles="--")

    axes[2].imshow(aerial_opc.numpy(), cmap="inferno", vmin=0, vmax=aerial_vmax)
    axes[2].contour(design.numpy(), levels=[0.5], colors="white", linewidths=0.7, linestyles="--")
    axes[2].contour(resist_opc.numpy(), levels=[0.5], colors="#22d3ee", linewidths=1.0)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets"),
        help="Directory to write hero.png and hero.json into.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    design = build_layout()

    aerial_no_opc = simulate_aerial_image(design, sigma_px=SIGMA_PX)
    resist_no_opc = apply_resist_threshold(aerial_no_opc, threshold=RESIST_THRESHOLD)
    epe_no_opc = compute_epe(resist_no_opc, design, pixel_size_nm=PIXEL_SIZE_NM)["epe_mean_nm"]

    model = registry.get(
        "levelset-ilt",
        iterations=600,
        lr=0.2,
        sigma_px=SIGMA_PX,
        tv_weight=0.0005,
        forward_model="gaussian",
    )
    model.setup()
    try:
        result = model.predict(design)
    finally:
        model.teardown()
    mask_opc = result.mask.detach().to(torch.float32)

    aerial_opc = simulate_aerial_image(mask_opc, sigma_px=SIGMA_PX)
    resist_opc = apply_resist_threshold(aerial_opc, threshold=RESIST_THRESHOLD)
    epe_opc = compute_epe(resist_opc, design, pixel_size_nm=PIXEL_SIZE_NM)["epe_mean_nm"]

    out_dir = args.output
    render(design, aerial_no_opc, aerial_opc, resist_opc, epe_no_opc, epe_opc, out_dir / "hero.png")

    sidecar = {
        "grid": GRID,
        "pixel_size_nm": PIXEL_SIZE_NM,
        "sigma_px": SIGMA_PX,
        "forward_model": "gaussian",
        "model": model.name,
        "epe_nm_no_opc": round(epe_no_opc, 3),
        "epe_nm_with_opc": round(epe_opc, 3),
        "resist_threshold": RESIST_THRESHOLD,
    }
    (out_dir / "hero.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Wrote {out_dir / 'hero.png'} and {out_dir / 'hero.json'}")
    print(f"  EPE  no-OPC = {epe_no_opc:.3f} nm")
    print(f"  EPE with-OPC = {epe_opc:.3f} nm")


if __name__ == "__main__":
    main()
