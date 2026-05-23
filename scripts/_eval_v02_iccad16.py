"""Eval v0.2 gan-opc weights against ICCAD16 testcase1 (private scaffolding).

Mirrors `openlithohub eval run --model gan-opc --dataset iccad16 ...` but
points at a local checkpoint via the GanOpc ``weights=`` kwarg, which the
CLI does not expose. Only the resulting numbers are gating-publishable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openlithohub.data import Iccad16Dataset  # noqa: E402
from openlithohub.benchmark.compliance.drc import check_drc  # noqa: E402
from openlithohub.benchmark.compliance.mrc import check_mrc  # noqa: E402
from openlithohub.benchmark.metrics import (  # noqa: E402
    compute_epe,
    compute_l2_error,
    compute_pvband,
    compute_wafer_epe,
)
import openlithohub.models.gan_opc  # noqa: E402, F401  # register gan-opc
from openlithohub.models.registry import registry  # noqa: E402
from openlithohub.simulators.base import SimulatorConfig  # noqa: E402
from openlithohub.simulators.hopkins_sim import HopkinsSimulator  # noqa: E402
from openlithohub.workflow.process_node import get_node  # noqa: E402


def main() -> int:
    weights = Path("checkpoints/gan_opc_v0_2.pt")
    if not weights.exists():
        print(f"missing {weights}")
        return 1

    pixel_nm = 4.0
    node = "7nm"
    nc = get_node(node)
    forward_sim = HopkinsSimulator(
        SimulatorConfig(
            wavelength_nm=nc.wavelength_nm,
            na=nc.numerical_aperture,
            pixel_size_nm=pixel_nm,
            threshold=nc.resist_threshold,
        )
    )

    model = registry.get("gan-opc", weights=str(weights))
    model.setup()
    try:
        ds = Iccad16Dataset(root=Path("data/iccad16"), pixel_nm=pixel_nm)
        sample = ds[0]
        print(f"sample: {sample.metadata.get('testcase_id', '?')}")

        result = model.predict(sample.design)
        h, w = result.mask.shape[-2:]
        n_px = int(h) * int(w)
        print(f"mask shape: {h}x{w}")

        out: dict[str, object] = {
            "model": "gan-opc-v0.2",
            "weights": str(weights),
            "dataset": "iccad16",
            "testcase_id": int(sample.metadata.get("testcase_id", 1)),
            "pixel_nm": pixel_nm,
            "node": node,
            "mask_shape": [int(h), int(w)],
        }

        if sample.mask is not None:
            epe = compute_epe(result.mask, sample.mask, pixel_size_nm=pixel_nm)
            out["epe_mean_nm"] = float(epe["epe_mean_nm"])
            out["epe_max_nm"] = float(epe["epe_max_nm"])

            wepe = compute_wafer_epe(
                result.mask, sample.mask, pixel_size_nm=pixel_nm, simulator=forward_sim
            )
            out["epe_wafer_mean_nm"] = float(wepe["epe_mean_nm"])
            out["epe_wafer_max_nm"] = float(wepe["epe_max_nm"])

            l2 = compute_l2_error(
                result.mask, sample.mask, pixel_size_nm=pixel_nm, simulator=forward_sim
            )
            out["l2_error_pixels"] = float(l2["l2_error_pixels"])

        pv = compute_pvband(result.mask, pixel_size_nm=pixel_nm)
        out.update({k: float(v) for k, v in pv.items()})

        # 7nm node defaults: min_width 20, min_spacing 20.
        mrc = check_mrc(
            result.mask, min_width_nm=20.0, min_spacing_nm=20.0, pixel_size_nm=pixel_nm
        )
        out["mrc_passed"] = bool(mrc.passed)
        out["mrc_violation_count"] = int(mrc.violation_count)
        out["mrc_violation_rate"] = float(mrc.violation_rate)
        out["mrc_total_pixels"] = n_px

        drc = check_drc(result.mask, pixel_size_nm=pixel_nm)
        out["drc_passed"] = bool(drc.passed)

        out_path = Path("out/baselines/iccad16/gan-opc-v0.2.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        print(f"wrote {out_path}")
    finally:
        model.teardown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
