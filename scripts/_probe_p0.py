"""P0 probe — radius-2 parity tripwire at v0.3 settings.

Per gan-opc-v0.3-improvements.md §2.5 P0:
At ``min_width_nm=20, pixel_size_nm=4.0``:
  - Loss radius   = floor(20 / (2*4))           = 2
  - Checker radius= max(0, (floor(20/4) - 1)//2) = (5-1)//2 = 2
The plan claims structural parity at radius=2. This probe is a *tripwire*
that fires if the formulas in ``mrc_loss.py`` / ``mrc.py`` have drifted.

Pass conditions:

* Strips of width 12 nm and 16 nm flag as MRC violations on **both** the
  loss-side opening and the checker-side opening.
* Strips of width 20 nm and 24 nm flag on **neither**.

Constructs a synthetic 256x256 mask containing parallel horizontal strips
of fixed width separated by wide gaps, evaluates both sides, and checks the
loss-side residual is non-zero iff width < 20 nm and the checker-side
violation count is non-zero iff width < 20 nm. Block training until pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openlithohub.benchmark.compliance.mrc import check_mrc  # noqa: E402
from openlithohub.benchmark.metrics.mrc_loss import curvilinear_mrc_loss  # noqa: E402

PIXEL_NM = 4.0
MIN_WIDTH_NM = 20.0
MIN_SPACING_NM = 20.0
HEIGHT = 256
WIDTH = 256


def _strip_mask(width_nm: float) -> torch.Tensor:
    """Build a (H, W) mask with horizontal strips of ``width_nm``, gap=64 nm."""
    width_px = max(1, int(round(width_nm / PIXEL_NM)))
    gap_px = max(1, int(round(64.0 / PIXEL_NM)))
    period = width_px + gap_px
    mask = torch.zeros(HEIGHT, WIDTH)
    y = 4  # leave a margin so the boundary band doesn't dominate
    while y + width_px < HEIGHT - 4:
        mask[y : y + width_px, 8 : WIDTH - 8] = 1.0
        y += period
    return mask


def _loss_flag(mask: torch.Tensor) -> float:
    """Return the cd term of curvilinear_mrc_loss with spacing/curvature off."""
    val = curvilinear_mrc_loss(
        mask,
        min_width_nm=MIN_WIDTH_NM,
        min_spacing_nm=MIN_SPACING_NM,
        pixel_size_nm=PIXEL_NM,
        weight_min_cd=1.0,
        weight_min_spacing=0.0,
        weight_min_curvature=0.0,
    )
    return float(val.item())


def _checker_flag(mask: torch.Tensor) -> int:
    res = check_mrc(
        mask,
        min_width_nm=MIN_WIDTH_NM,
        min_spacing_nm=MIN_SPACING_NM,
        pixel_size_nm=PIXEL_NM,
    )
    return int(res.width_violation_count)


def main() -> int:
    print(f"P0 probe: pixel_nm={PIXEL_NM} min_width_nm={MIN_WIDTH_NM}")
    print("Expected loss radius = floor(20/(2*4)) = 2")
    print("Expected checker radius = (floor(20/4) - 1)//2 = (5-1)//2 = 2")
    print()

    rows = []
    for w_nm in (12.0, 16.0, 20.0, 24.0):
        mask = _strip_mask(w_nm)
        loss_v = _loss_flag(mask)
        chk_v = _checker_flag(mask)
        should_violate = w_nm < MIN_WIDTH_NM
        loss_violates = loss_v > 1e-9
        chk_violates = chk_v > 0
        ok = (loss_violates == should_violate) and (chk_violates == should_violate)
        rows.append(
            {
                "width_nm": w_nm,
                "loss_value": loss_v,
                "loss_violates": loss_violates,
                "checker_count": chk_v,
                "checker_violates": chk_violates,
                "expected_violation": should_violate,
                "ok": ok,
            }
        )
        flag = "OK " if ok else "FAIL"
        print(
            f"  [{flag}] w={w_nm:5.1f} nm | loss={loss_v:.4e} ({loss_violates}) | "
            f"checker={chk_v:5d} ({chk_violates}) | expect_violate={should_violate}"
        )

    all_ok = all(r["ok"] for r in rows)

    out = Path("out/probes/v0_3_p0_parity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "probe": "P0",
                "pixel_nm": PIXEL_NM,
                "min_width_nm": MIN_WIDTH_NM,
                "expected_loss_radius": 2,
                "expected_checker_radius": 2,
                "rows": rows,
                "passed": all_ok,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    if not all_ok:
        print("\nP0 FAILED — radius parity drift; STOP and re-derive.")
        return 1
    print("\nP0 PASSED — radius-2 parity holds at (px=4, w=20).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
