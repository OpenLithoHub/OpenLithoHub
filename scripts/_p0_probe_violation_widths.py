"""P0 probe: histogram v0.1 GAN-OPC mask MRC width violations on ICCAD16 testcase1.

Decision tree (per gan-opc-v0.2-improvements.md "v8 finding"):
- Mass at width_nm <= 16: proceed with v0.2 as planned (radius-1 MRC loss).
- Mass at 16-24 nm AND significant spacing violations: enable
  weight_min_spacing > 0 in Change 1.
- Mass at width_nm >= 24 nm: abort v0.2 plan as designed.

Runs at eval-time settings (node=7nm, pixel_nm=1.0, min_width=28, min_spacing=28)
because that's what the leaderboard checker fires.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openlithohub.benchmark.compliance.mrc import check_mrc  # noqa: E402
from openlithohub.data.iccad16 import Iccad16Dataset  # noqa: E402
from openlithohub.models.gan_opc import GanOpcModel  # noqa: E402
from openlithohub.workflow.process_node import get_node  # noqa: E402


def main() -> int:
    node = get_node("7nm")
    pixel_nm = node.pixel_size_nm
    min_width_nm = node.min_feature_nm
    min_spacing_nm = node.min_spacing_nm
    print(
        f"node=7nm pixel_nm={pixel_nm} min_width={min_width_nm} "
        f"min_spacing={min_spacing_nm}"
    )

    ckpt = Path("checkpoints/gan_opc_v0_1.pt")
    if not ckpt.exists():
        print(f"missing v0.1 checkpoint at {ckpt}", file=sys.stderr)
        return 2

    model = GanOpcModel(weights=ckpt, device="cpu")
    model.setup()

    adapter = Iccad16Dataset(root=Path("data/iccad16"), pixel_nm=pixel_nm)
    sample = adapter[0]
    print(f"design shape: {tuple(sample.design.shape)}")

    result_pred = model.predict(sample.design)
    mask = result_pred.mask
    print(f"mask shape: {tuple(mask.shape)} fg_pixels={int((mask > 0.5).sum())}")

    # Run MRC checker at eval-time settings.
    mrc = check_mrc(
        mask,
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
        pixel_size_nm=pixel_nm,
    )
    print(
        f"violation_count={mrc.violation_count} rate={mrc.violation_rate:.4%} "
        f"width_count={mrc.width_violation_count} "
        f"spacing_count={mrc.spacing_violation_count}"
    )
    print(f"violations sampled: {len(mrc.violations)}")

    # Histogram per-violation widths. The MRCResult.violations entries
    # use type_code (0=width, 1=spacing) and actual_nm (distance to
    # nearest non-feature for width, to nearest feature for spacing).
    widths_by_type: dict[str, list[float]] = {"width": [], "spacing": []}
    for v in mrc.violations:
        tc = int(v.get("type_code", -1))
        vt = "width" if tc == 0 else ("spacing" if tc == 1 else "unknown")
        w = v.get("actual_nm")
        if w is None:
            continue
        widths_by_type.setdefault(vt, []).append(float(w))

    for vt, vals in widths_by_type.items():
        if not vals:
            continue
        bins = [0, 8, 16, 24, 32, 40, 1000]
        labels = ["≤8", "8-16", "16-24", "24-32", "32-40", ">40"]
        counts = Counter()
        for v in vals:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    counts[labels[i]] += 1
                    break
        print(f"\nViolation type={vt} (n={len(vals)})")
        for lbl in labels:
            n = counts.get(lbl, 0)
            print(f"  {lbl:>6} nm: {n:5d}")
        if vals:
            print(
                f"  min={min(vals):.2f} max={max(vals):.2f} "
                f"mean={sum(vals) / len(vals):.2f}"
            )

    # Decision summary.
    width_vals = widths_by_type.get("width", [])
    spacing_vals = widths_by_type.get("spacing", [])
    decision = {
        "mrc_violation_count": mrc.violation_count,
        "mrc_violation_rate": mrc.violation_rate,
        "width_violation_count": mrc.width_violation_count,
        "spacing_violation_count": mrc.spacing_violation_count,
        "n_width_samples": len(width_vals),
        "n_spacing_samples": len(spacing_vals),
        "width_p50": (
            sorted(width_vals)[len(width_vals) // 2] if width_vals else None
        ),
        "width_max": max(width_vals) if width_vals else None,
    }
    print("\n--- decision summary ---")
    print(json.dumps(decision, indent=2))

    out = Path("out/p0_probe_v01_violation_widths.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "probe": "v0.1 ICCAD16 testcase1 MRC width-violation histogram",
                "node_settings": {
                    "pixel_nm": pixel_nm,
                    "min_width_nm": min_width_nm,
                    "min_spacing_nm": min_spacing_nm,
                },
                "summary": decision,
                "violations_sample": mrc.violations[:50],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")

    if width_vals:
        below_16 = sum(1 for v in width_vals if v <= 16)
        between_16_24 = sum(1 for v in width_vals if 16 < v <= 24)
        above_24 = sum(1 for v in width_vals if v > 24)
        n = len(width_vals)
        print(
            f"\nDistribution: <=16: {below_16}/{n} "
            f"16-24: {between_16_24}/{n} >24: {above_24}/{n}"
        )
        if above_24 > 0.5 * n:
            print("DECISION: ABORT — mass at >=24 nm; v0.2 as designed cannot fix.")
            return 1
        if below_16 < 0.3 * n and spacing_vals:
            print(
                "DECISION: enable weight_min_spacing > 0 in Change 1; "
                "spacing dominates over width."
            )
        else:
            print("DECISION: PROCEED — radius-1 MRC loss targets the right failure mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
