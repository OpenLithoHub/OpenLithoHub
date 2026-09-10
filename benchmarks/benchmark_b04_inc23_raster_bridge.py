#!/usr/bin/env python3
"""B04 Increment 23 raster-bridge replay on the Increment-21 fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openlithohub.data.io import load_layout
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource
from openlithohub.verify.raster_bridge import compare_binary_rasters


def generate_fixture(path: Path, *, h: int = 512, w: int = 512) -> None:
    import klayout.db as db

    step = 8
    layout = db.Layout()
    layout.dbu = 0.001
    geom = layout.layer(1, 0)
    anchor = layout.layer(99, 0)
    child = layout.create_cell("CHILD")
    top = layout.create_cell("TOP")
    child.shapes(geom).insert(db.Box(0, 0, 24 * step, 16 * step))

    for y in range(32, h - 32, 48):
        top.shapes(geom).insert(
            db.Box(
                4 * step,
                (h - (y + 12)) * step,
                (w - 4) * step,
                (h - y) * step,
            )
        )

    for y in range(24, h - 48, 96):
        for x in range(20, w - 48, 96):
            top.shapes(geom).insert(
                db.Box(
                    x * step,
                    (h - (y + 20)) * step,
                    (x + 36) * step,
                    (h - y) * step,
                )
            )
            top.insert(
                db.CellInstArray(
                    child.cell_index(),
                    db.Trans((x + 8) * step, (h - (y + 48)) * step),
                )
            )

    outer = db.Box(120 * step, (h - 220) * step, 280 * step, (h - 80) * step)
    hole = db.Box(160 * step, (h - 180) * step, 240 * step, (h - 120) * step)
    donut = db.Polygon(outer)
    donut.insert_hole(hole)
    top.shapes(geom).insert(donut)

    top.shapes(geom).insert(db.Box(0, (h - 64) * step, 8 * step, (h - 16) * step))
    top.shapes(geom).insert(db.Box((w - 8) * step, (h - 64) * step, w * step, (h - 16) * step))

    top.shapes(anchor).insert(db.Box(0, 0, 1, 1))
    top.shapes(anchor).insert(db.Box(w * step - 1, h * step - 1, w * step, h * step))
    layout.write(str(path))


def exact_fixture_mask(*, h: int = 512, w: int = 512) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for y in range(32, h - 32, 48):
        mask[y : y + 12, 4 : w - 4] = True
    for y in range(24, h - 48, 96):
        for x in range(20, w - 48, 96):
            mask[y : y + 20, x : x + 36] = True
            mask[y + 32 : y + 48, x + 8 : x + 32] = True
    donut = np.zeros_like(mask)
    donut[80:220, 120:280] = True
    donut[120:180, 160:240] = False
    mask |= donut
    mask[16:64, 0:8] = True
    mask[16:64, w - 8 : w] = True
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gds = args.out / "b04_inc23_raster_bridge.gds"
    generate_fixture(gds)

    exact = exact_fixture_mask()
    if int(exact.sum()) != 90032:
        raise RuntimeError(
            f"analytic fixture area drifted: {int(exact.sum())} != 90032"
        )

    source = KLayoutAlignedRunSource.from_file(gds, pixel_size_nm=8.0, layer="1:0", top_cell="TOP")
    streamed = (
        source.read_window(BoundingBox(0, 0, source.shape[1], source.shape[0])).numpy().astype(bool)
    )
    if not np.array_equal(exact, streamed):
        raise RuntimeError("exact-vector source no longer matches analytic fixture")

    dense = load_layout(gds, pixel_nm=8.0, layer="1:0").numpy().astype(bool)
    bridge = compare_binary_rasters(exact, dense, max_radius=4)

    result = {
        "status": "PASS_DIAGNOSTIC_BRIDGE_CHARACTERIZED",
        "exact_reference_ones": bridge.reference_ones,
        "dense_loader_ones": bridge.candidate_ones,
        "false_positive_pixels": bridge.false_positive_pixels,
        "false_negative_pixels": bridge.false_negative_pixels,
        "symmetric_difference_pixels": bridge.symmetric_difference_pixels,
        "candidate_in_reference_dilation_px": bridge.candidate_in_reference_dilation_px,
        "reference_in_candidate_dilation_px": bridge.reference_in_candidate_dilation_px,
        "mutual_chebyshev_radius_px": bridge.mutual_chebyshev_radius_px,
        "euclidean_pixel_center_hausdorff_upper_nm": bridge.euclidean_center_hausdorff_upper_nm(
            pixel_nm=8.0
        ),
        "relative_area_bias_vs_exact": (bridge.candidate_ones - bridge.reference_ones)
        / bridge.reference_ones,
        "semantic_firewall": {
            "discrete_mask_bridge": "DIAGNOSTIC/FINITE-GRID ENCLOSURE",
            "continuous_contour_bridge": "NOT_YET_PROVED",
            "hopkins_socs_bridge": "NOT_YET_PROVED",
            "epe_cd_process_window": "NOT_YET_PROVED",
        },
    }
    out = args.out / "B04_INC23_RASTER_BRIDGE.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
