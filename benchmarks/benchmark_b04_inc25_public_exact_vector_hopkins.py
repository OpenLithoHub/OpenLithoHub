#!/usr/bin/env python3
"""Public-GDS exact-vector -> run-spectrum -> Hopkins replay for B04 Increment 25.

This benchmark intentionally bypasses ``load_layout``.  KLayout is used only
as a parser into ``KLayoutAlignedRunSource``; the optical mask spectrum is
constructed from canonical exact-vector runs.

The public Sky130 geometry is real, while the optical parameters are a stated
synthetic 193 nm / NA 1.35 / sigma 0.7 model.  This is therefore an algorithmic
lithography benchmark, not foundry calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from openlithohub._utils.hopkins import (
    HopkinsParams,
    clear_kernel_cache,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource
from openlithohub.verify.exact_vector_optics import (
    exact_vector_normalized_spectrum,
    materialized_reference_spectrum,
)
from openlithohub.verify.run_spectrum import coherent_spectrum_from_layout

GRID_N = 576


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_public_cell_layer(path: Path) -> tuple[str, str, float]:
    import klayout.db as db

    layout = db.Layout()
    layout.read(str(path))
    cell = layout.cell("sky130_fd_sc_hd__inv_1")
    if cell is None:
        tops = list(layout.top_cells())
        if not tops:
            raise RuntimeError("public layout has no top cells")
        cell = tops[0]

    # Prefer the licon1/contact layer if present; otherwise use the most
    # populated geometric layer.  66/44 is a real physical mask layer in
    # Sky130, but this benchmark does not claim PDK-calibrated optics.
    wanted = layout.layer(66, 44)
    it = cell.begin_shapes_rec(wanted)
    count = 0
    while not it.at_end():
        count += 1
        it.next()
    if count > 0:
        layer = "66:44"
    else:
        best_count = -1
        layer = None
        for li in layout.layer_indexes():
            it = cell.begin_shapes_rec(li)
            c = 0
            while not it.at_end():
                c += 1
                if c >= 20000:
                    break
                it.next()
            if c > best_count:
                info = layout.get_info(li)
                layer = f"{int(info.layer)}:{int(info.datatype)}"
                best_count = c
        if layer is None:
            raise RuntimeError("no geometric layer found")

    dbu_nm = float(layout.dbu) * 1000.0
    return cell.name, layer, dbu_nm


def choose_densest_square(
    source: KLayoutAlignedRunSource,
    *,
    grid_n: int,
) -> tuple[tuple[int, int, int, int], int]:
    h, w = source.shape
    if h < grid_n or w < grid_n:
        raise RuntimeError(f"public cell shape {source.shape} is smaller than {grid_n} square")
    stride = max(1, grid_n // 2)
    best = (-1, (0, 0, grid_n, grid_n))
    y_starts = list(range(0, h - grid_n + 1, stride))
    x_starts = list(range(0, w - grid_n + 1, stride))
    if y_starts[-1] != h - grid_n:
        y_starts.append(h - grid_n)
    if x_starts[-1] != w - grid_n:
        x_starts.append(w - grid_n)

    for y0 in y_starts:
        for x0 in x_starts:
            bbox = (x0, y0, x0 + grid_n, y0 + grid_n)
            occupied = sum(int(run.x1 - run.x0) for run in source.iter_runs_for_bbox(bbox))
            if occupied > best[0]:
                best = (occupied, bbox)
    return best[1], best[0]


def direct_aerial_from_run_spectrum(
    *,
    layout_coeff: np.ndarray,
    kernels: torch.Tensor,
    weights: torch.Tensor,
    dose: float,
) -> np.ndarray:
    n = layout_coeff.shape[0]
    kernels_c64 = kernels.to(torch.complex64)
    kernel_f = (
        torch.fft.fft2(torch.fft.ifftshift(kernels_c64, dim=(-2, -1)))
        .cpu()
        .numpy()
        .astype(np.complex128)
    )
    kernel_coeff = kernel_f / (n * n)

    coherent_coeff = coherent_spectrum_from_layout(
        normalized_kernel_spectrum=kernel_coeff,
        normalized_layout_spectrum=layout_coeff,
    )
    coherent = np.fft.ifft2((n * n) * coherent_coeff, axes=(-2, -1))
    w = weights.detach().cpu().numpy().astype(np.float64)
    intensity = np.sum(
        w[:, None, None] * (coherent.real**2 + coherent.imag**2),
        axis=0,
    )
    return intensity * dose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cell, layer, dbu_nm = choose_public_cell_layer(args.gds)
    source = KLayoutAlignedRunSource.from_file(
        args.gds,
        pixel_size_nm=dbu_nm,
        layer=layer,
        top_cell=cell,
    )

    bbox, occupied = choose_densest_square(source, grid_n=GRID_N)

    t0 = time.perf_counter()
    spectrum = exact_vector_normalized_spectrum(
        source,
        bbox_xyxy=bbox,
    )
    run_spectrum_wall = time.perf_counter() - t0

    x0, y0, x1, y1 = bbox
    exact_window = source.read_window(BoundingBox(x0, y0, x1, y1)).numpy().astype(np.float64)
    fft_reference = materialized_reference_spectrum(exact_window)
    spectrum_error = float(np.max(np.abs(spectrum.normalized_spectrum - fft_reference)))

    params = HopkinsParams(
        wavelength_nm=193.0,
        na=1.35,
        sigma=0.7,
        pixel_size_nm=dbu_nm,
        num_kernels=24,
        defocus_nm=0.0,
    )
    clear_kernel_cache()
    t0 = time.perf_counter()
    kernels, weights = compute_socs_kernels(
        params,
        GRID_N,
        device="cpu",
    )
    kernel_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    direct = direct_aerial_from_run_spectrum(
        layout_coeff=spectrum.normalized_spectrum,
        kernels=kernels,
        weights=weights,
        dose=1.0,
    )
    direct_wall = time.perf_counter() - t0

    ref = (
        simulate_aerial_image_hopkins(
            torch.from_numpy(exact_window.astype(np.float32)),
            params=params,
            kernels=kernels,
            weights=weights,
            dose=1.0,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    aerial_error = float(np.max(np.abs(direct - ref)))
    aerial_mean_error = float(np.mean(np.abs(direct - ref)))

    result = {
        "status": "PASS_EXACT_VECTOR_TO_HOPKINS"
        if spectrum_error < 1e-11 and aerial_error < 5e-5
        else "FAIL",
        "claim_firewall": {
            "public_geometry": "REAL_SKY130_GDS",
            "optics": "SYNTHETIC_DECLARED_HOPKINS_MODEL",
            "foundry_calibration": "NOT_CLAIMED",
            "dense_loader_in_proof_path": False,
            "finite_chip_boundary": "NOT_YET_CERTIFIED",
            "continuous_epe_cd_process_window": "NOT_YET_CERTIFIED",
        },
        "fixture": {
            "sha256": sha256(args.gds),
            "top_cell": cell,
            "layer_datatype": layer,
            "pixel_nm": dbu_nm,
            "source_shape_px": list(source.shape),
            "selected_bbox_xyxy": list(bbox),
            "grid_n": GRID_N,
            "occupied_pixels": occupied,
            "run_count": spectrum.run_count,
            "layout_hash": source.layout_hash,
            "backend_kind": source.backend_kind,
        },
        "spectrum": {
            "representation_bridge_upper": spectrum.contract.representation_bridge_upper,
            "dense_loader_used": spectrum.contract.dense_loader_used,
            "run_vs_same_exact_indicator_fft_max_error": spectrum_error,
            "run_spectrum_wall_s": run_spectrum_wall,
        },
        "optics": {
            "wavelength_nm": 193.0,
            "na": 1.35,
            "sigma": 0.7,
            "num_kernels": 24,
            "focus_nm": 0.0,
            "dose": 1.0,
            "tile_extent_nm": GRID_N * dbu_nm,
            "four_lambda_over_na_nm": 4.0 * 193.0 / 1.35,
            "kernel_build_wall_s": kernel_wall,
            "direct_spectral_aerial_wall_s": direct_wall,
            "direct_vs_materialized_exact_source_aerial_max_error": aerial_error,
            "direct_vs_materialized_exact_source_aerial_mean_error": aerial_mean_error,
        },
    }
    out = args.out / "B04_INC25_PUBLIC_EXACT_VECTOR_HOPKINS.json"
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_EXACT_VECTOR_TO_HOPKINS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
