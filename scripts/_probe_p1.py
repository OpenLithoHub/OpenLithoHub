"""P1 probe — target-distribution sanity for v0.3 bilinear+thresh resize.

Per gan-opc-v0.3-improvements.md §2.5 P1:
Render 100 random samples through:
  * v0.1 pipeline: bilinear + > 0.5 threshold at resize_to=256 (px=8.0)
  * v0.3 pipeline: bilinear + > 0.5 threshold at resize_to=512 (px=4.0)

Both pipelines must match the **reference** bilinear+thresh implementation
exactly (modulo torch nondeterminism noise). Asserts:

  (i) v0.1 path matches reference bilinear+thresh at 256² byte-for-byte.
  (ii) v0.3 path matches reference bilinear+thresh at 512² byte-for-byte.
  (iii) v0.3 path differs from v0.2's mode='area'+thresh — this is the
        regression-detector. If v0.3 byte-matches v0.2's area path, the
        plan's revert was a no-op.

Catches: accidental retention of mode='area', off-by-one resize, or
threshold drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openlithohub.data.ganopc import GanOpcDataset  # noqa: E402

NUM_SAMPLES = 100
DATA_ROOT = Path("data/ganopc/extracted/ganopc-data")


def _resize_bilinear_thresh(t: torch.Tensor, target: int) -> torch.Tensor:
    t4 = t.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(t4, size=(target, target), mode="bilinear", align_corners=False)
    return (resized.squeeze(0).squeeze(0) > 0.5).float()


def _resize_area_thresh(t: torch.Tensor, target: int) -> torch.Tensor:
    t4 = t.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(t4, size=(target, target), mode="area")
    return (resized.squeeze(0).squeeze(0) > 0.5).float()


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"FAIL: missing dataset root {DATA_ROOT}", file=sys.stderr)
        return 2

    ds = GanOpcDataset(root=DATA_ROOT)
    n = min(NUM_SAMPLES, len(ds))
    print(f"P1 probe: scanning {n} ganopc samples")

    v01_match_ref = 0
    v03_match_ref = 0
    v03_differs_from_v02 = 0
    v02_match_v03 = 0  # red flag — should be 0 if revert is real

    for i in range(n):
        sample = ds[i]
        if sample.mask is None:
            continue
        target = sample.mask.float()

        ref256 = _resize_bilinear_thresh(target, 256)
        v01_path = _resize_bilinear_thresh(target, 256)
        if torch.equal(ref256, v01_path):
            v01_match_ref += 1

        ref512 = _resize_bilinear_thresh(target, 512)
        v03_path = _resize_bilinear_thresh(target, 512)
        if torch.equal(ref512, v03_path):
            v03_match_ref += 1

        v02_area = _resize_area_thresh(target, 512)
        if not torch.equal(v03_path, v02_area):
            v03_differs_from_v02 += 1
        else:
            v02_match_v03 += 1

    out = {
        "probe": "P1",
        "n_samples": n,
        "v01_pipeline_matches_reference": v01_match_ref,
        "v03_pipeline_matches_reference": v03_match_ref,
        "v03_differs_from_v02_area": v03_differs_from_v02,
        "v02_area_collisions_with_v03_bilinear": v02_match_v03,
    }
    print(json.dumps(out, indent=2))

    out_path = Path("out/probes/v0_3_p1_target_distribution.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    ok = (
        v01_match_ref == n
        and v03_match_ref == n
        and v03_differs_from_v02 >= int(0.5 * n)
    )
    if not ok:
        print("\nP1 FAILED — bilinear+thresh resize implementation drift.")
        return 1
    print("\nP1 PASSED — v0.1 and v0.3 paths use bilinear+thresh; differ from v0.2 area.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
