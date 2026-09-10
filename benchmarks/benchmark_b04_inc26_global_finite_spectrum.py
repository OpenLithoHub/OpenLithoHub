#!/usr/bin/env python3
"""Public Sky130 global finite-spectral compression benchmark.

This benchmark validates a non-periodic, arbitrary-physical-frequency mask
summary on the complete finite exact-vector layout.

Proof path:
    GDS -> KLayoutAlignedRunSource -> canonical runs -> global spectral moments.

Diagnostic oracle:
    materialize a window from the SAME exact-vector source and sum the same
    physical phases over occupied pixel centers.

The oracle is not the legacy dense/PIL loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource
from openlithohub.verify.global_finite_spectrum import (
    PlaneWaveMode,
    SpectralNode,
    evaluate_intensity_jet,
    global_finite_spectral_summary,
)

WAVELENGTH_NM = 193.0
NA = 1.35
N_NODES = 48


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


def pupil_like_nodes() -> tuple[SpectralNode, ...]:
    """Deterministic non-lattice disk nodes; diagnostic, not quadrature-certified."""

    fmax = NA / WAVELENGTH_NM
    golden = math.pi * (3.0 - math.sqrt(5.0))
    nodes = [SpectralNode(0.0, 0.0)]
    for j in range(1, N_NODES):
        radius = fmax * math.sqrt((j - 0.5) / (N_NODES - 0.5))
        theta = golden * j
        nodes.append(
            SpectralNode(
                radius * math.cos(theta),
                radius * math.sin(theta),
            )
        )
    return tuple(nodes)


def explicit_reference_moments(
    mask: np.ndarray,
    nodes: tuple[SpectralNode, ...],
    *,
    pixel_nm: float,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    xn = (xs.astype(np.float64) + 0.5) * pixel_nm
    yn = (ys.astype(np.float64) + 0.5) * pixel_nm
    out = np.empty(len(nodes), dtype=np.complex128)
    for q, node in enumerate(nodes):
        phase = -2j * np.pi * (node.fx_per_nm * xn + node.fy_per_nm * yn)
        out[q] = np.exp(phase).sum(dtype=np.complex128)
    return out


def synthetic_modes(nodes: tuple[SpectralNode, ...]) -> tuple[PlaneWaveMode, ...]:
    """Three deterministic coherent modes for continuous-jet smoke testing."""

    q = len(nodes)
    idx = np.arange(q, dtype=np.float64)
    radial = np.array(
        [math.hypot(n.fx_per_nm, n.fy_per_nm) for n in nodes],
        dtype=np.float64,
    )
    fmax = max(float(radial.max()), 1e-15)
    envelope = np.exp(-0.75 * (radial / fmax) ** 2) / q

    modes = []
    for s, weight in enumerate((0.5, 0.3, 0.2)):
        phase = np.exp(1j * (0.37 + 0.41 * s) * idx)
        amplitudes = envelope * phase
        modes.append(
            PlaneWaveMode(
                weight=weight,
                amplitudes=amplitudes.astype(np.complex128),
            )
        )
    return tuple(modes)


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
    nodes = pupil_like_nodes()

    t0 = time.perf_counter()
    summary = global_finite_spectral_summary(source, nodes=nodes)
    summary_wall = time.perf_counter() - t0

    # Diagnostic algebraic oracle from the same exact-vector source.
    h, w = source.shape
    t0 = time.perf_counter()
    exact_mask = source.read_window(BoundingBox(0, 0, w, h)).numpy().astype(bool)
    materialize_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    reference = explicit_reference_moments(
        exact_mask,
        nodes,
        pixel_nm=dbu_nm,
    )
    reference_wall = time.perf_counter() - t0

    abs_error = np.abs(summary.moments - reference)
    scaled_error = abs_error / np.maximum(1.0, np.abs(reference))
    max_abs_error = float(abs_error.max())
    max_scaled_error = float(scaled_error.max())

    if summary.occupied_pixels != int(exact_mask.sum()):
        raise RuntimeError("run occupancy disagrees with exact-vector oracle")

    # Continuous finite-plane-wave value/gradient/Hessian smoke test.
    modes = synthetic_modes(nodes)
    xs = np.linspace(0.2 * w * dbu_nm, 0.8 * w * dbu_nm, 5)
    ys = np.linspace(0.2 * h * dbu_nm, 0.8 * h * dbu_nm, 5)
    points = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    jet = evaluate_intensity_jet(
        summary,
        modes,
        points_nm=points,
    )
    jet_finite = all(
        np.all(np.isfinite(arr))
        for arr in (
            jet.value,
            jet.dx,
            jet.dy,
            jet.dxx,
            jet.dxy,
            jet.dyy,
        )
    )

    spectral_state_bytes = summary.moments.nbytes + 16 * len(nodes)
    raster_bytes = exact_mask.nbytes

    result = {
        "status": (
            "PASS_GLOBAL_FINITE_SPECTRAL_SUMMARY"
            if max_scaled_error < 1e-8 and jet_finite
            else "FAIL"
        ),
        "claim_firewall": {
            "public_geometry": "REAL_SKY130_GDS",
            "mask_semantics": "FINITE_EXACT_VECTOR_PIXEL_CENTER_MEASURE",
            "periodic_mask_copies": False,
            "dense_pil_loader_used": False,
            "same_exact_source_materialization": "DIAGNOSTIC_ORACLE_ONLY",
            "spatial_halo_error_for_declared_finite_quadrature_model": 0.0,
            "pupil_source_quadrature": "NOT_YET_CERTIFIED",
            "continuous_hopkins_equivalence": "NOT_YET_CERTIFIED",
            "foundry_calibration": "NOT_CLAIMED",
        },
        "fixture": {
            "sha256": sha256(args.gds),
            "top_cell": cell,
            "layer_datatype": layer,
            "shape_px": list(source.shape),
            "pixel_nm": dbu_nm,
            "layout_hash": source.layout_hash,
            "backend_kind": source.backend_kind,
            "run_count": summary.run_count,
            "occupied_pixels": summary.occupied_pixels,
            "full_layout_pixels": h * w,
        },
        "spectral_summary": {
            "node_count": len(nodes),
            "fmax_per_nm": NA / WAVELENGTH_NM,
            "summary_wall_s": summary_wall,
            "max_abs_moment_error_vs_same_exact_source": max_abs_error,
            "max_scaled_moment_error_vs_same_exact_source": max_scaled_error,
            "spatial_halo_error_upper": (summary.contract.spatial_halo_error_upper),
            "persistent_spectral_state_bytes_estimate": spectral_state_bytes,
        },
        "diagnostic_reference": {
            "materialized_exact_vector_mask_bytes": raster_bytes,
            "materialize_wall_s": materialize_wall,
            "moment_reference_wall_s": reference_wall,
            "spectral_state_vs_mask_byte_ratio": (spectral_state_bytes / max(1, raster_bytes)),
        },
        "continuous_jet_smoke_test": {
            "points": len(points),
            "all_finite": bool(jet_finite),
            "intensity_min": float(jet.value.min()),
            "intensity_max": float(jet.value.max()),
            "gradient_norm_max_per_nm": float(np.hypot(jet.dx, jet.dy).max()),
            "hessian_frobenius_max_per_nm2": float(
                np.sqrt(jet.dxx**2 + 2.0 * jet.dxy**2 + jet.dyy**2).max()
            ),
            "status": "NUMERICAL_DIAGNOSTIC",
        },
    }

    out = args.out / "B04_INC26_GLOBAL_FINITE_SPECTRUM.json"
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_GLOBAL_FINITE_SPECTRAL_SUMMARY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
