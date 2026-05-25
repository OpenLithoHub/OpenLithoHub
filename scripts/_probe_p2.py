"""P2 probe — 4-corner PVB regulariser sanity (R2 construction, R3 stop-the-world).

Per gan-opc-v0.3-improvements.md §2.5 P2:

  (a) L_pvb is non-zero on a frozen v0.1 mask.
  (b) Gradient flows through L_pvb to the input mask (autograd survives all
      four forward_gaussian calls and the mean across corners).
  (c) **Stop-the-world physics check.** On a perfectly-printing mask
      (predicted == target) L_pvb is small but non-zero; on a deliberately
      blurred mask (Gaussian sigma=2 px applied to v0.1 output) L_pvb is
      LARGER. Run this check separately under:
        - dose-only path (sigma fixed at sigma_nom, dose ± delta)
        - focus-only path (dose fixed at 1.0, sigma in {sigma_hi, sigma_lo})
      Both paths must show L_pvb(sharp) < L_pvb(blurred). If either path
      shows the opposite, training MUST NOT proceed; written justification
      required (R3 Q2).

Construction parameters (§2.3 Change 5, training px=4.0):
    delta      = 0.05
    sigma_nom  = 2.0
    sigma_def  = 20.0 / (2.0 * 4.0) = 2.5
    sigma_hi   = sigma_nom + sigma_def       = 4.5
    sigma_lo   = max(0.5, 2.0 - 0.5*2.5)     = 0.75
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openlithohub._utils.forward_model import simulate_aerial_image  # noqa: E402
from openlithohub.data.iccad16 import Iccad16Dataset  # noqa: E402
from openlithohub.models.gan_opc import GanOpcModel  # noqa: E402

PIXEL_NM = 4.0
DELTA = 0.05
SIGMA_NOM = 2.0
SIGMA_DEF = 20.0 / (2.0 * PIXEL_NM)  # 2.5
SIGMA_HI = SIGMA_NOM + SIGMA_DEF  # 4.5
SIGMA_LO = max(0.5, SIGMA_NOM - SIGMA_DEF * 0.5)  # 0.75

CORNERS_FULL = [
    (1.0 + DELTA, SIGMA_HI),
    (1.0 + DELTA, SIGMA_LO),
    (1.0 - DELTA, SIGMA_HI),
    (1.0 - DELTA, SIGMA_LO),
]
# Dose-only: vary dose, hold sigma at nominal.
CORNERS_DOSE_ONLY = [
    (1.0 + DELTA, SIGMA_NOM),
    (1.0 - DELTA, SIGMA_NOM),
]
# Focus-only: hold dose, vary sigma.
CORNERS_FOCUS_ONLY = [
    (1.0, SIGMA_HI),
    (1.0, SIGMA_LO),
]


def _l_pvb(
    mask: torch.Tensor,
    target: torch.Tensor,
    corners: list[tuple[float, float]],
) -> torch.Tensor:
    target_aerial = target  # treat target mask itself as the aerial reference
    total = mask.new_zeros(())
    for dose, sigma in corners:
        aerial = simulate_aerial_image(mask, sigma_px=sigma, dose=dose)
        total = total + functional.mse_loss(aerial, target_aerial)
    return total / len(corners)


def _gaussian_blur(mask: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Apply a Gaussian blur to a (H,W) mask."""
    kernel_size = max(3, int(2 * round(3 * sigma_px) + 1))
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - half
    g = torch.exp(-0.5 * (coords / sigma_px) ** 2)
    g = (g / g.sum()).to(mask.device)
    k1 = g.view(1, 1, 1, -1)
    k2 = g.view(1, 1, -1, 1)
    x = mask.float().unsqueeze(0).unsqueeze(0)
    x = functional.pad(x, (half, half, 0, 0), mode="replicate")
    x = functional.conv2d(x, k1)
    x = functional.pad(x, (0, 0, half, half), mode="replicate")
    x = functional.conv2d(x, k2)
    return x.squeeze(0).squeeze(0)


def main() -> int:
    print(
        f"P2 probe: pixel_nm={PIXEL_NM} delta={DELTA} sigma_nom={SIGMA_NOM} "
        f"sigma_hi={SIGMA_HI:.3f} sigma_lo={SIGMA_LO:.3f}"
    )

    ckpt = Path("checkpoints/gan_opc_v0_1.pt")
    if not ckpt.exists():
        print(f"FAIL: missing v0.1 checkpoint at {ckpt}", file=sys.stderr)
        return 2

    model = GanOpcModel(weights=ckpt, device="cpu")
    model.setup()

    ds = Iccad16Dataset(root=Path("data/iccad16"), pixel_nm=PIXEL_NM)
    sample = ds[0]
    target = sample.mask.float() if sample.mask is not None else sample.design.float()

    pred = model.predict(sample.design)
    sharp_mask = pred.mask.float()
    blurred_mask = _gaussian_blur(sharp_mask, sigma_px=2.0)

    # Use target as aerial reference. Resize target to sharp_mask if shapes differ.
    if target.shape != sharp_mask.shape:
        target_4 = target.unsqueeze(0).unsqueeze(0)
        target = functional.interpolate(
            target_4, size=sharp_mask.shape, mode="bilinear", align_corners=False
        )
        target = target.squeeze(0).squeeze(0)

    # (a) non-zero
    sharp_full = _l_pvb(sharp_mask, target, CORNERS_FULL)
    nonzero_a = float(sharp_full.item()) > 0.0

    # (b) gradient flows
    test_mask = sharp_mask.clone().requires_grad_(True)
    grad_loss = _l_pvb(test_mask, target, CORNERS_FULL)
    grad_loss.backward()
    grad_present = test_mask.grad is not None and float(test_mask.grad.abs().sum().item()) > 0.0

    # (c) sharp vs blurred — full / dose-only / focus-only
    blurred_full = _l_pvb(blurred_mask, target, CORNERS_FULL)
    sharp_dose = _l_pvb(sharp_mask, target, CORNERS_DOSE_ONLY)
    blurred_dose = _l_pvb(blurred_mask, target, CORNERS_DOSE_ONLY)
    sharp_focus = _l_pvb(sharp_mask, target, CORNERS_FOCUS_ONLY)
    blurred_focus = _l_pvb(blurred_mask, target, CORNERS_FOCUS_ONLY)

    full_mono = float(sharp_full.item()) < float(blurred_full.item())
    dose_mono = float(sharp_dose.item()) < float(blurred_dose.item())
    focus_mono = float(sharp_focus.item()) < float(blurred_focus.item())

    out = {
        "probe": "P2",
        "pixel_nm": PIXEL_NM,
        "delta": DELTA,
        "sigma_nom": SIGMA_NOM,
        "sigma_def": SIGMA_DEF,
        "sigma_hi": SIGMA_HI,
        "sigma_lo": SIGMA_LO,
        "checks": {
            "a_nonzero": nonzero_a,
            "b_gradient_flows": grad_present,
            "c_full_sharp_lt_blurred": full_mono,
            "c_dose_only_sharp_lt_blurred": dose_mono,
            "c_focus_only_sharp_lt_blurred": focus_mono,
        },
        "values": {
            "sharp_full": float(sharp_full.item()),
            "blurred_full": float(blurred_full.item()),
            "sharp_dose": float(sharp_dose.item()),
            "blurred_dose": float(blurred_dose.item()),
            "sharp_focus": float(sharp_focus.item()),
            "blurred_focus": float(blurred_focus.item()),
        },
    }
    print(json.dumps(out, indent=2))

    out_path = Path("out/probes/v0_3_p2_pvb_sanity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Stop-the-world if (c) fails on any path.
    all_ok = nonzero_a and grad_present and full_mono and dose_mono and focus_mono
    if not all_ok:
        print("\nP2 FAILED — block training. R3 Q2 requires written justification before retry.")
        if not full_mono or not dose_mono or not focus_mono:
            print("  Sharp-vs-blurred monotonicity FAILED — loss-surface physics wrong.")
        return 1
    print("\nP2 PASSED — sharp < blurred under all three paths. PVB term physics OK.")
    model.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
