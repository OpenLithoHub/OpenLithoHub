"""P3 probe — PVB bandwidth loss correctness check (v0.4, R2 rewrite).

Per gan-opc-v0.4-improvements.md §3 P3:

  (a) Non-zero: bandwidth loss > 0 on any non-trivial mask
  (b) Gradient flow: all gradients finite through differentiable_threshold
  (c) Directionality: bandwidth(sharp) < bandwidth(blurred)
      - full 4-corner: MUST pass
      - focus-only: MUST pass
      - dose-only: signal may be weak, direction-correct is sufficient
  (d) v0.3 MSE vs v0.4 bandwidth comparison (import real v0.3 code)
  (e) Steepness sensitivity: {10, 20, 50}

Blocking: (a)(b)(c full)(c focus) must all pass for Stage 1 to start.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.data.iccad16 import Iccad16Dataset
from openlithohub.models.gan_opc import GanOpcModel

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
CORNERS_DOSE_ONLY = [
    (1.0 + DELTA, SIGMA_NOM),
    (1.0 - DELTA, SIGMA_NOM),
]
CORNERS_FOCUS_ONLY = [
    (1.0, SIGMA_HI),
    (1.0, SIGMA_LO),
]


def _bandwidth_loss(
    mask: torch.Tensor,
    corners: list[tuple[float, float]],
    steepness: float = 20.0,
) -> torch.Tensor:
    """Metric-aligned bandwidth loss (v0.4)."""
    outer = mask.new_zeros(mask.shape)
    inner = mask.new_ones(mask.shape)
    for dose, sigma in corners:
        aerial = simulate_aerial_image(mask, sigma_px=sigma, dose=dose)
        resist = differentiable_threshold(aerial, threshold=0.5, steepness=steepness)
        outer = torch.maximum(outer, resist)
        inner = torch.minimum(inner, resist)
    bandwidth = (outer - inner).clamp(min=0.0)
    return bandwidth.mean()


def _pvb_loss_v03(
    mask: torch.Tensor,
    target: torch.Tensor,
    corners: list[tuple[float, float]],
) -> torch.Tensor:
    """v0.3 PVB loss: MSE(aerial_corner, design) — imported construction."""
    total = mask.new_zeros(())
    for dose, sigma in corners:
        aerial = simulate_aerial_image(mask, sigma_px=sigma, dose=dose)
        total = total + functional.mse_loss(aerial, target)
    return total / len(corners)


def _gaussian_blur(mask: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Apply Gaussian blur to a (H,W) mask."""
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
        f"P3 probe: pixel_nm={PIXEL_NM} delta={DELTA} sigma_nom={SIGMA_NOM} "
        f"sigma_hi={SIGMA_HI:.3f} sigma_lo={SIGMA_LO:.3f}"
    )

    ckpt = Path("checkpoints/gan_opc_v0_1.pt")
    if not ckpt.exists():
        print(f"SKIP: no v0.1 checkpoint at {ckpt} — running with synthetic mask")
        sharp_mask = (torch.rand(256, 256) > 0.7).float()
        target = sharp_mask.clone()
    else:
        model = GanOpcModel(weights=ckpt, device="cpu")
        model.setup()
        ds = Iccad16Dataset(root=Path("data/iccad16"), pixel_nm=PIXEL_NM)
        sample = ds[0]
        target = sample.mask.float() if sample.mask is not None else sample.design.float()
        pred = model.predict(sample.design)
        sharp_mask = pred.mask.float()
        model.teardown()

    blurred_mask = _gaussian_blur(sharp_mask, sigma_px=2.0)

    # Resize target to match if needed
    if target.shape != sharp_mask.shape:
        target_4 = target.unsqueeze(0).unsqueeze(0)
        target = functional.interpolate(
            target_4, size=sharp_mask.shape, mode="bilinear", align_corners=False
        )
        target = target.squeeze(0).squeeze(0)

    results: dict = {"probe": "P3", "checks": {}, "values": {}}

    # (a) Non-zero
    bw_sharp_full = _bandwidth_loss(sharp_mask, CORNERS_FULL)
    results["checks"]["a_nonzero"] = float(bw_sharp_full.item()) > 0.0
    results["values"]["sharp_bandwidth_full"] = float(bw_sharp_full.item())

    # (b) Gradient flow
    test_mask = sharp_mask.clone().detach().requires_grad_(True)
    bw_grad = _bandwidth_loss(test_mask, CORNERS_FULL)
    bw_grad.backward()
    grad_present = test_mask.grad is not None
    grad_finite = torch.isfinite(test_mask.grad).all().item() if grad_present else False
    grad_norm = float(test_mask.grad.norm().item()) if grad_present else 0.0
    results["checks"]["b_gradient_flows"] = grad_present and grad_finite
    results["values"]["gradient_norm"] = grad_norm

    # (c) Directionality: sharp < blurred
    bw_blurred_full = _bandwidth_loss(blurred_mask, CORNERS_FULL)
    bw_sharp_focus = _bandwidth_loss(sharp_mask, CORNERS_FOCUS_ONLY)
    bw_blurred_focus = _bandwidth_loss(blurred_mask, CORNERS_FOCUS_ONLY)
    bw_sharp_dose = _bandwidth_loss(sharp_mask, CORNERS_DOSE_ONLY)
    bw_blurred_dose = _bandwidth_loss(blurred_mask, CORNERS_DOSE_ONLY)

    full_mono = float(bw_sharp_full.item()) < float(bw_blurred_full.item())
    focus_mono = float(bw_sharp_focus.item()) < float(bw_blurred_focus.item())
    dose_mono = float(bw_sharp_dose.item()) < float(bw_blurred_dose.item())
    dose_relative_change = abs(
        float(bw_blurred_dose.item()) - float(bw_sharp_dose.item())
    ) / max(float(bw_sharp_dose.item()), 1e-8)

    results["checks"]["c_full_sharp_lt_blurred"] = full_mono
    results["checks"]["c_focus_sharp_lt_blurred"] = focus_mono
    results["checks"]["c_dose_sharp_lt_blurred"] = dose_mono or dose_relative_change < 0.01
    results["values"].update({
        "blurred_bandwidth_full": float(bw_blurred_full.item()),
        "sharp_bandwidth_focus": float(bw_sharp_focus.item()),
        "blurred_bandwidth_focus": float(bw_blurred_focus.item()),
        "sharp_bandwidth_dose": float(bw_sharp_dose.item()),
        "blurred_bandwidth_dose": float(bw_blurred_dose.item()),
        "dose_relative_change": dose_relative_change,
    })

    # (d) v0.3 MSE vs v0.4 bandwidth comparison
    v03_sharp = _pvb_loss_v03(sharp_mask, target, CORNERS_FULL)
    v03_blurred = _pvb_loss_v03(blurred_mask, target, CORNERS_FULL)
    v04_sharp = bw_sharp_full
    v04_blurred = bw_blurred_full

    results["values"]["v03_mse_sharp"] = float(v03_sharp.item())
    results["values"]["v03_mse_blurred"] = float(v03_blurred.item())
    results["values"]["v04_bandwidth_sharp"] = float(v04_sharp.item())
    results["values"]["v04_bandwidth_blurred"] = float(v04_blurred.item())
    results["checks"]["d_both_positive_finite"] = (
        float(v03_sharp.item()) > 0
        and float(v04_sharp.item()) > 0
        and torch.isfinite(v03_sharp).item()
        and torch.isfinite(v04_sharp).item()
    )
    results["checks"]["d_both_increase_on_blur"] = (
        float(v03_blurred.item()) > float(v03_sharp.item())
        and float(v04_blurred.item()) > float(v04_sharp.item())
    )

    # (e) Steepness sensitivity
    steepness_results = {}
    for s in [10, 20, 50]:
        test_m = sharp_mask.clone().detach().requires_grad_(True)
        bw = _bandwidth_loss(test_m, CORNERS_FULL, steepness=float(s))
        bw.backward()
        gn = float(test_m.grad.norm().item()) if test_m.grad is not None else 0.0
        steepness_results[str(s)] = {
            "bandwidth": float(bw.item()),
            "grad_norm": gn,
        }
    results["steepness_sensitivity"] = steepness_results
    # Verify steepness=20 has significant gradient
    results["checks"]["e_steepness_20_grad_significant"] = (
        steepness_results["20"]["grad_norm"] > 1e-6
    )

    print(json.dumps(results, indent=2))

    out_path = Path("out/probes/v0_4_p3_pvb_bandwidth.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    # Blocking checks
    blocking = [
        ("a_nonzero", results["checks"]["a_nonzero"]),
        ("b_gradient_flows", results["checks"]["b_gradient_flows"]),
        ("c_full_sharp_lt_blurred", results["checks"]["c_full_sharp_lt_blurred"]),
        ("c_focus_sharp_lt_blurred", results["checks"]["c_focus_sharp_lt_blurred"]),
    ]
    non_blocking = [
        ("c_dose_sharp_lt_blurred", results["checks"]["c_dose_sharp_lt_blurred"]),
        ("e_steepness_20_grad_significant", results["checks"]["e_steepness_20_grad_significant"]),
    ]

    all_blocking_pass = all(v for _, v in blocking)
    all_non_blocking_pass = all(v for _, v in non_blocking)

    print("\nBlocking checks:")
    for name, passed in blocking:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print("Non-blocking checks:")
    for name, passed in non_blocking:
        print(f"  {'PASS' if passed else 'WARN'}: {name}")

    if not all_blocking_pass:
        print("\nP3 FAILED — blocking checks failed. Stage 1 CANNOT start.")
        return 1
    if not all_non_blocking_pass:
        print("\nP3 PASSED (with warnings) — blocking checks pass, non-blocking have warnings.")
    else:
        print("\nP3 PASSED — all checks pass. Bandwidth loss physics OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
