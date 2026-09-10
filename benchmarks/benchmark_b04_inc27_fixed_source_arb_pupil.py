#!/usr/bin/env python3
"""Fixed-source certified hard-pupil benchmark for Increment 27."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource
from openlithohub.verify.arb_fixed_source_pupil import (
    FixedSourcePupilParams,
    certified_fixed_source_coherent_field,
)
from openlithohub.verify.continuous_square_mask import (
    continuous_square_mask_contract,
    continuous_square_mask_transform,
)
from openlithohub.verify.global_finite_spectrum import (
    SpectralNode,
    global_finite_spectral_summary,
)


def generate_fixture(path: Path) -> None:
    import klayout.db as db

    layout = db.Layout()
    layout.dbu = 0.001
    geom = layout.layer(1, 0)
    anchor = layout.layer(99, 0)
    top = layout.create_cell("TOP")
    for box in (
        db.Box(16, 16, 48, 32),
        db.Box(64, 24, 104, 40),
        db.Box(32, 56, 80, 72),
        db.Box(96, 64, 120, 96),
    ):
        top.shapes(geom).insert(box)
    top.shapes(anchor).insert(db.Box(0, 0, 1, 1))
    top.shapes(anchor).insert(db.Box(127, 127, 128, 128))
    layout.write(str(path))


def normalized_sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def transform_lift_replay(source) -> dict[str, float | int]:
    nodes = (
        SpectralNode(0.0, 0.0),
        SpectralNode(0.0021, -0.0017),
        SpectralNode(0.0043, 0.0032),
        SpectralNode(-0.0051, 0.0011),
    )
    summary = global_finite_spectral_summary(source, nodes=nodes)
    p = source.pixel_size_nm
    errors = []
    for q, node in enumerate(nodes):
        cont = continuous_square_mask_transform(
            source,
            fx_per_nm=node.fx_per_nm,
            fy_per_nm=node.fy_per_nm,
        )
        lifted = (
            p
            * p
            * normalized_sinc(p * node.fx_per_nm)
            * normalized_sinc(p * node.fy_per_nm)
            * summary.moments[q]
        )
        errors.append(abs(cont - lifted))
    return {
        "max_continuous_vs_point_moment_sinc_error": float(max(errors)),
        "run_count": summary.run_count,
        "occupied_pixels": summary.occupied_pixels,
    }


def choose_sparse_public_layer(path: Path) -> tuple[str, str, float, int]:
    import klayout.db as db

    layout = db.Layout()
    layout.read(str(path))
    cell = layout.cell("sky130_fd_sc_hd__inv_1")
    if cell is None:
        tops = list(layout.top_cells())
        if not tops:
            raise RuntimeError("public layout has no top cells")
        cell = tops[0]
    candidates = []
    for layer_index in layout.layer_indexes():
        info = layout.get_info(layer_index)
        it = cell.begin_shapes_rec(layer_index)
        count = 0
        while not it.at_end():
            shape = it.shape()
            if shape.is_box() or shape.is_polygon() or shape.is_path():
                count += 1
                if count > 128:
                    break
            it.next()
        if 1 <= count <= 128:
            candidates.append((count, int(info.layer), int(info.datatype)))
    if not candidates:
        raise RuntimeError("no sparse nonempty public geometric layer found")
    count, layer, datatype = min(candidates)
    return cell.name, f"{layer}:{datatype}", float(layout.dbu) * 1000.0, count


def certify_source(source, *, sx_frac: float, sy_frac: float) -> dict[str, object]:
    fp = 1.35 / 193.0
    h, w = source.shape
    params = FixedSourcePupilParams(
        wavelength_nm=193.0,
        na=1.35,
        source_fx_per_nm=sx_frac * fp,
        source_fy_per_nm=sy_frac * fp,
        defocus_nm=5.0,
    )
    t0 = time.perf_counter()
    cert = certified_fixed_source_coherent_field(
        source,
        params=params,
        observation_x_nm=0.5 * w * source.pixel_size_nm,
        observation_y_nm=0.5 * h * source.pixel_size_nm,
        precision_bits=96,
        abs_tol_decimal="1e-8",
        rel_tol_decimal="1e-8",
        eval_limit=60000,
        depth_limit=24,
    )
    return {
        **asdict(cert),
        "max_component_radius": cert.max_component_radius,
        "wall_s": time.perf_counter() - t0,
        "source_fraction_of_pupil": [sx_frac, sy_frac],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--public-gds", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fixture = args.out / "b04_inc27_continuous_fixture.gds"
    generate_fixture(fixture)
    source = KLayoutAlignedRunSource.from_file(
        fixture,
        pixel_size_nm=8.0,
        layer="1:0",
        top_cell="TOP",
    )
    contract = continuous_square_mask_contract(source)
    replay = transform_lift_replay(source)
    toy_cert = certify_source(source, sx_frac=0.15, sy_frac=-0.10)
    status = (
        "PASS_FIXED_SOURCE_ARB_PUPIL"
        if contract.representation_bridge_upper == 0.0
        and replay["max_continuous_vs_point_moment_sinc_error"] < 1e-10
        and toy_cert["finite"]
        and toy_cert["max_component_radius"] < 1e-5
        else "FAIL"
    )
    result = {
        "status": status,
        "claim_firewall": {
            "continuous_mask_semantics": "FINITE_EXACT_VECTOR_SQUARE_APERTURE_MASK",
            "continuous_mask_representation_bridge_upper": 0.0,
            "fixed_source_hard_pupil_integral": "ARB_CERTIFIED",
            "source_plane_quadrature": "OPEN",
            "partial_coherence_intensity": "NOT_YET_CERTIFIED",
            "openlithohub_socs_equivalence": "NOT_YET_CERTIFIED",
            "foundry_calibration": "NOT_CLAIMED",
        },
        "generated_fixture": {
            "shape_px": list(source.shape),
            "pixel_nm": source.pixel_size_nm,
            "contract": asdict(contract),
            "lift_replay": replay,
            "fixed_source_certificate": toy_cert,
        },
    }

    if args.public_gds is not None and args.public_gds.exists():
        cell, layer, pixel_nm, shape_count = choose_sparse_public_layer(args.public_gds)
        public = KLayoutAlignedRunSource.from_file(
            args.public_gds,
            pixel_size_nm=pixel_nm,
            layer=layer,
            top_cell=cell,
        )
        public_replay = transform_lift_replay(public)
        public_cert = None
        if public_replay["run_count"] <= 24:
            public_cert = certify_source(public, sx_frac=0.0, sy_frac=0.0)
            public_status = (
                "PASS_PUBLIC_FIXED_SOURCE_ARB"
                if public_replay["max_continuous_vs_point_moment_sinc_error"] < 1e-9
                and public_cert["finite"]
                and public_cert["max_component_radius"] < 1e-5
                else "FAIL"
            )
        else:
            public_status = (
                "PASS_PUBLIC_CONTINUOUS_LIFT_ARB_SKIPPED_COMPLEXITY"
                if public_replay["max_continuous_vs_point_moment_sinc_error"] < 1e-9
                else "FAIL"
            )
        result["public_gds"] = {
            "top_cell": cell,
            "layer_datatype": layer,
            "recursive_shape_count": shape_count,
            "shape_px": list(public.shape),
            "pixel_nm": pixel_nm,
            "lift_replay": public_replay,
            "fixed_source_certificate": public_cert,
            "arb_run_complexity_guard": 24,
            "status": public_status,
        }
        if public_status == "FAIL":
            result["status"] = "FAIL"

    out = args.out / "B04_INC27_FIXED_SOURCE_ARB_PUPIL.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_FIXED_SOURCE_ARB_PUPIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
