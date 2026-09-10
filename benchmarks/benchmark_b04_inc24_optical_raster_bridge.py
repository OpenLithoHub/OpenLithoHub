#!/usr/bin/env python3
"""K24 Hopkins diagnostic for the dense-vs-exact raster bridge.

This benchmark intentionally uses the same *periodic* 128x128 Hopkins tile for
both mask representations.  It isolates the optical sensitivity to the mask
rasterization discrepancy; it does not certify a finite-chip boundary model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openlithohub._utils.hopkins import (
    HopkinsParams,
    clear_kernel_cache,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub.benchmark.metrics.epe import compute_epe
from openlithohub.data.io import load_layout
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

PIXEL_NM = 8.0
THRESHOLD = 0.225
CROP = 128


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
            db.Box(4 * step, (h - (y + 12)) * step, (w - 4) * step, (h - y) * step)
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


def choose_crop(exact: np.ndarray, dense: np.ndarray) -> tuple[int, int, int]:
    diff = exact != dense
    best = (-1, 0, 0)
    # Step 32 so the diagnostic crop sees the most raster disagreement while
    # remaining deterministic.
    for y0 in range(0, exact.shape[0] - CROP + 1, 32):
        for x0 in range(0, exact.shape[1] - CROP + 1, 32):
            n = int(diff[y0 : y0 + CROP, x0 : x0 + CROP].sum())
            if n > best[0]:
                best = (n, y0, x0)
    return best


def gradient_band_diagnostic(
    image: np.ndarray,
    *,
    threshold: float,
    band_width: float,
) -> dict[str, float | int | None]:
    gy, gx = np.gradient(image, PIXEL_NM, PIXEL_NM)
    grad = np.hypot(gx, gy)
    width = max(float(band_width), 1e-8)
    band = np.abs(image - threshold) <= width
    # Strip the one-pixel finite-difference boundary.
    band[[0, -1], :] = False
    band[:, [0, -1]] = False
    count = int(band.sum())
    if count == 0:
        return {
            "band_pixels": 0,
            "kappa_grid_per_nm": None,
            "delta_over_kappa_grid_nm": None,
        }
    kappa = float(grad[band].min())
    return {
        "band_pixels": count,
        "kappa_grid_per_nm": kappa,
        "delta_over_kappa_grid_nm": (None if kappa <= 0 else float(band_width) / kappa),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gds = args.out / "b04_inc24_optical_bridge.gds"
    generate_fixture(gds)

    exact = exact_fixture_mask()
    if int(exact.sum()) != 90032:
        raise RuntimeError("exact fixture area drifted")

    source = KLayoutAlignedRunSource.from_file(
        gds, pixel_size_nm=PIXEL_NM, layer="1:0", top_cell="TOP"
    )
    streamed = (
        source.read_window(BoundingBox(0, 0, source.shape[1], source.shape[0])).numpy().astype(bool)
    )
    if not np.array_equal(exact, streamed):
        raise RuntimeError("exact-vector source escaped analytic oracle")

    dense = load_layout(gds, pixel_nm=PIXEL_NM, layer="1:0").numpy().astype(bool)
    n_diff, y0, x0 = choose_crop(exact, dense)
    ex = exact[y0 : y0 + CROP, x0 : x0 + CROP].astype(np.float32)
    de = dense[y0 : y0 + CROP, x0 : x0 + CROP].astype(np.float32)

    if int((ex != de).sum()) != n_diff or n_diff <= 0:
        raise RuntimeError("crop selection failed")

    exact_t = torch.from_numpy(ex)
    dense_t = torch.from_numpy(de)

    focus_nodes = (-10.0, 0.0, 10.0)
    dose_nodes = (0.95, 1.0, 1.05)
    rows = []
    worst_delta = (-1.0, None)
    worst_epe = (-1.0, None)

    clear_kernel_cache()
    for focus in focus_nodes:
        params = HopkinsParams(
            wavelength_nm=193.0,
            na=1.35,
            sigma=0.7,
            pixel_size_nm=PIXEL_NM,
            num_kernels=24,
            defocus_nm=focus,
        )
        kernels, weights = compute_socs_kernels(params, CROP, device="cpu")
        kernels_f = torch.fft.fft2(torch.fft.ifftshift(kernels.to(torch.complex64), dim=(-2, -1)))
        for dose in dose_nodes:
            ie = simulate_aerial_image_hopkins(
                exact_t,
                params=params,
                kernels=kernels,
                weights=weights,
                dose=dose,
                precomputed_kernels_f=kernels_f,
            ).detach()
            idense = simulate_aerial_image_hopkins(
                dense_t,
                params=params,
                kernels=kernels,
                weights=weights,
                dose=dose,
                precomputed_kernels_f=kernels_f,
            ).detach()

            delta = (idense - ie).abs()
            delta_max = float(delta.max().item())
            delta_mean = float(delta.mean().item())
            resist_e = (ie >= THRESHOLD).to(torch.float32)
            resist_d = (idense >= THRESHOLD).to(torch.float32)
            epe = compute_epe(resist_d, resist_e, pixel_size_nm=PIXEL_NM)
            gd = gradient_band_diagnostic(
                ie.numpy(),
                threshold=THRESHOLD,
                band_width=delta_max,
            )
            row = {
                "focus_nm": focus,
                "dose": dose,
                "max_abs_intensity_delta": delta_max,
                "mean_abs_intensity_delta": delta_mean,
                "printed_raster_epe_dense_vs_exact_nm": epe,
                **gd,
            }
            rows.append(row)
            if delta_max > worst_delta[0]:
                worst_delta = (delta_max, row)
            epe_max = float(epe["epe_max_nm"])
            if epe["valid"] and epe_max > worst_epe[0]:
                worst_epe = (epe_max, row)

    result = {
        "status": "PASS_NUMERICAL_DIAGNOSTIC",
        "boundary_semantics": "PERIODIC_128PX_HOPKINS_TILE_FOR_BOTH_MASKS",
        "claim_firewall": {
            "intensity_deltas": "NUMERICAL_DIAGNOSTIC",
            "grid_gradient": "NUMERICAL_DIAGNOSTIC_NOT_INTERVAL_CERTIFIED",
            "delta_over_kappa_grid": "NUMERICAL_DIAGNOSTIC_NOT_A_CONTOUR_CERTIFICATE",
            "printed_raster_epe": "SOBEL_PIXEL_METRIC_DIAGNOSTIC",
            "finite_chip_halo": "NOT_PROVED_BY_THIS_BENCHMARK",
            "continuous_epe_cd_process_window": "NOT_YET_CERTIFIED",
        },
        "fixture": {
            "full_shape": [512, 512],
            "exact_ones": int(exact.sum()),
            "dense_ones": int(dense.sum()),
            "full_disagreement_pixels": int((exact != dense).sum()),
            "selected_crop_xyxy": [x0, y0, x0 + CROP, y0 + CROP],
            "crop_disagreement_pixels": n_diff,
            "crop_exact_ones": int(ex.sum()),
            "crop_dense_ones": int(de.sum()),
        },
        "optics": {
            "wavelength_nm": 193.0,
            "na": 1.35,
            "sigma": 0.7,
            "num_kernels": 24,
            "pixel_nm": PIXEL_NM,
            "threshold": THRESHOLD,
            "focus_nodes_nm": list(focus_nodes),
            "dose_nodes": list(dose_nodes),
        },
        "nodes": rows,
        "worst_intensity_node": worst_delta[1],
        "worst_printed_raster_epe_node": worst_epe[1],
    }
    out = args.out / "B04_INC24_OPTICAL_RASTER_BRIDGE_DIAGNOSTIC.json"
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
