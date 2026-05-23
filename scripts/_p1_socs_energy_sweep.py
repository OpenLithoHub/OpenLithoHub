"""P1: SOCS cumulative-energy spectrum sweep at v0.2 settings.

Per gan-opc-v0.2-improvements.md Change 2 prep step P1 — print the
energy spectrum at 8 nm/px, 256×256, num_kernels=24 and pick the
smallest N in {2, 4, 8, 16, 24} where cumulative energy >= 0.99.

Runs in a SEPARATE Python process from training (to avoid evicting the
training-time entry from _KERNEL_CACHE during sweep).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openlithohub._utils.hopkins import (  # noqa: E402
    HopkinsParams,
    clear_kernel_cache,
    compute_socs_kernels,
)


def main() -> int:
    pixel_nm = 8.0
    grid = 256
    base_n = 24
    threshold = 0.99

    params = HopkinsParams(num_kernels=base_n, pixel_size_nm=pixel_nm)
    kernels, weights = compute_socs_kernels(
        params, grid_size=grid, device=torch.device("cpu")
    )
    weights_np = weights.cpu().tolist()
    print(f"K={len(weights_np)} pixel_nm={pixel_nm} grid={grid}")
    total = sum(weights_np)
    print(f"sum_weights={total:.6f}")

    # Cumulative-energy table (already sorted descending in compute_socs_kernels).
    rows = []
    cum = 0.0
    for i, w in enumerate(weights_np, start=1):
        cum += w
        frac = cum / total if total > 0 else 0.0
        rows.append({"k": i, "weight": w, "cum": cum, "frac": frac})
    print(f"\n{'k':>3} {'weight':>10} {'cum':>10} {'frac':>8}")
    for r in rows:
        print(f"{r['k']:>3d} {r['weight']:>10.6f} {r['cum']:>10.6f} {r['frac']:>8.4f}")

    # Pick smallest N from candidate set.
    candidates = [2, 4, 8, 16, 24]
    chosen = base_n
    for n in candidates:
        if n - 1 < len(rows) and rows[n - 1]["frac"] >= threshold:
            chosen = n
            break
    print(f"\nselected N={chosen} (smallest with cumulative >= {threshold})")

    out = Path("out/p1_socs_kernel_selection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "probe": "P1 SOCS cumulative energy at v0.2 settings",
                "pixel_size_nm": pixel_nm,
                "grid_size": grid,
                "base_num_kernels": base_n,
                "threshold": threshold,
                "selected_num_kernels": chosen,
                "energy_table": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    clear_kernel_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
