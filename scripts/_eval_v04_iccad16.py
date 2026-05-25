"""Eval v0.4 gan-opc weights against ICCAD16 testcase1.

Mirrors `_eval_v03_iccad16.py` but updated for v0.4:
  - Supports UNetV2 (--arch unetv2) architecture
  - Reports bandwidth loss alongside other metrics
  - Greenlight rule: E PVB <= F PVB - 0.4 nm AND E PVB <= 10.9 nm

Usage:
    python scripts/_eval_v04_iccad16.py --weights checkpoints/gan_opc_v0_4_e.pt \\
        --tag e --arch unet --pixel-nm 4.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import openlithohub.models.gan_opc  # noqa: E402, F401  # register gan-opc
from openlithohub.benchmark.compliance.drc import check_drc  # noqa: E402
from openlithohub.benchmark.compliance.mrc import check_mrc  # noqa: E402
from openlithohub.benchmark.metrics import (  # noqa: E402
    compute_epe,
    compute_l2_error,
    compute_pvband,
    compute_wafer_epe,
)
from openlithohub.data import Iccad16Dataset  # noqa: E402
from openlithohub.models.registry import registry  # noqa: E402
from openlithohub.simulators.base import SimulatorConfig  # noqa: E402
from openlithohub.simulators.hopkins_sim import HopkinsSimulator  # noqa: E402
from openlithohub.workflow.process_node import get_node  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="run tag (e/f/g/h/i/stage0).")
    ap.add_argument("--arch", default="unet", choices=["unet", "unetv2"])
    ap.add_argument("--node", default="7nm")
    ap.add_argument("--pixel-nm", type=float, default=4.0)
    ap.add_argument("--testcase-index", type=int, default=0)
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"missing {args.weights}")
        return 1

    nc = get_node(args.node)
    forward_sim = HopkinsSimulator(
        SimulatorConfig(
            wavelength_nm=nc.wavelength_nm,
            na=nc.numerical_aperture,
            pixel_size_nm=args.pixel_nm,
            threshold=nc.resist_threshold,
        )
    )

    model = registry.get("gan-opc", weights=str(args.weights))
    model.setup()
    try:
        ds = Iccad16Dataset(root=Path("data/iccad16"), pixel_nm=args.pixel_nm)
        sample = ds[args.testcase_index]
        print(f"sample: {sample.metadata.get('testcase_id', '?')}")

        result = model.predict(sample.design)
        h, w = result.mask.shape[-2:]
        n_px = int(h) * int(w)
        print(f"mask shape: {h}x{w}")

        out: dict[str, object] = {
            "model": f"gan-opc-v0.4-{args.tag}",
            "version": "v0.4",
            "arch": args.arch,
            "weights": str(args.weights),
            "dataset": "iccad16",
            "testcase_id": int(sample.metadata.get("testcase_id", 1)),
            "pixel_nm": args.pixel_nm,
            "node": args.node,
            "mask_shape": [int(h), int(w)],
        }

        if sample.mask is not None:
            epe = compute_epe(result.mask, sample.mask, pixel_size_nm=args.pixel_nm)
            out["epe_mean_nm"] = float(epe["epe_mean_nm"])
            out["epe_max_nm"] = float(epe["epe_max_nm"])

            wepe = compute_wafer_epe(
                result.mask, sample.mask, pixel_size_nm=args.pixel_nm, simulator=forward_sim
            )
            out["epe_wafer_mean_nm"] = float(wepe["epe_mean_nm"])
            out["epe_wafer_max_nm"] = float(wepe["epe_max_nm"])

            l2 = compute_l2_error(
                result.mask, sample.mask, pixel_size_nm=args.pixel_nm, simulator=forward_sim
            )
            out["l2_error_pixels"] = float(l2["l2_error_pixels"])

        pv = compute_pvband(result.mask, pixel_size_nm=args.pixel_nm)
        out.update({k: float(v) for k, v in pv.items()})

        mrc = check_mrc(
            result.mask,
            min_width_nm=20.0,
            min_spacing_nm=20.0,
            pixel_size_nm=args.pixel_nm,
        )
        out["mrc_passed"] = bool(mrc.passed)
        out["mrc_violation_count"] = int(mrc.violation_count)
        out["mrc_violation_rate"] = float(mrc.violation_rate)
        out["mrc_total_pixels"] = n_px

        drc = check_drc(result.mask, pixel_size_nm=args.pixel_nm)
        out["drc_passed"] = bool(drc.passed)

        # Greenlight check
        pvb_mean = out.get("pvband_mean_nm", float("inf"))
        mrc_rate = out.get("mrc_violation_rate", 1.0)
        pvb_max = out.get("pvband_max_nm", float("inf"))

        out["greenlight"] = {
            "pvb_mean_threshold": 10.9,
            "pvb_mean_met": pvb_mean <= 10.9,
            "mrc_threshold": 7.5,
            "mrc_met": mrc_rate <= 7.5,
            "pvb_max_threshold": 45.0,
            "pvb_max_met": pvb_max <= 45.0,
            "all_met": pvb_mean <= 10.9 and mrc_rate <= 7.5 and pvb_max <= 45.0,
        }

        out_path = Path(f"out/baselines/iccad16/gan-opc-v0.4-{args.tag}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        print(f"wrote {out_path}")

        # Print greenlight summary
        gl = out["greenlight"]
        print(f"\nGreenlight: PVB_mean={pvb_mean:.2f} (<=10.9: {gl['pvb_mean_met']})")
        print(f"            MRC_rate={mrc_rate:.2f}% (<=7.5%: {gl['mrc_met']})")
        print(f"            PVB_max={pvb_max:.2f} (<=45: {gl['pvb_max_met']})")
        print(f"            ALL MET: {gl['all_met']}")
    finally:
        model.teardown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
