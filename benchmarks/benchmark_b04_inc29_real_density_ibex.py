#!/usr/bin/env python3
"""B04 Increment 29 realistic-density routed-block benchmark.

Fixture
-------
Pinned public PDB Physical Design Database, Sky130HD ``ibex``:
- top cell: ibex_core
- real routed GDS
- design metadata reports ~57.9% standard-cell utilization.

This benchmark takes deterministic nested center crops from that *real routed
block* and replays the Increment-28 work-avoidance pipeline.

The purpose is hostile external-validity testing:
does exact-empty-context screening still avoid meaningful work when the layout
is no longer an artificially sparse single-cell placement?

The forward model remains the same deterministic finite-support 9x9,
zero-preserving architecture benchmark used in Increment 28.  This isolates
the effect of layout density.  It is not foundry-calibrated lithography.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from openlithohub.streaming.crop_source import ExactVectorCropSource
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.screening import ExactEmptyContextScreeningPolicy
from openlithohub.streaming.sinks import MetricOnlyTileSink, TensorTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

FORWARD_RADIUS = 4
FORWARD_KERNEL = 2 * FORWARD_RADIUS + 1
PIXEL_NM = 1.0
TOP_CELL = "ibex_core"
LAYER = "66:44"


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def make_forward():
    ax = torch.arange(FORWARD_KERNEL, dtype=torch.float32) - FORWARD_RADIUS
    g = torch.exp(-(ax**2) / (2.0 * 1.6**2))
    g = g / g.sum()
    kx = g.reshape(1, 1, 1, -1)
    ky = g.reshape(1, 1, -1, 1)

    def forward(tile: torch.Tensor) -> torch.Tensor:
        x = tile.float().unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.conv2d(x, kx, padding=(0, FORWARD_RADIUS))
        x = torch.nn.functional.conv2d(x, ky, padding=(FORWARD_RADIUS, 0))
        return x.squeeze(0).squeeze(0)

    return forward


def selective_policy() -> ExactEmptyContextScreeningPolicy:
    return ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance=("inc29-separable-9x9-finite-support-radius4-zero-preserving"),
    )


def centered_crop(parent, size: int) -> ExactVectorCropSource:
    h, w = parent.shape
    if size > h or size > w:
        raise ValueError(f"requested {size} crop exceeds parent shape {parent.shape}")
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return ExactVectorCropSource(
        parent,
        BoundingBox(x0, y0, x0 + size, y0 + size),
    )


def load_parent(gds: Path):
    return KLayoutAlignedRunSource.from_file(
        gds,
        pixel_size_nm=PIXEL_NM,
        layer=LAYER,
        top_cell=TOP_CELL,
    )


def crop_run_stats(crop: ExactVectorCropSource) -> dict[str, Any]:
    h, w = crop.shape
    t0 = time.perf_counter()
    runs = tuple(crop.iter_runs_for_bbox((0, 0, w, h)))
    scan_wall = time.perf_counter() - t0
    occupied = sum(r.x1 - r.x0 for r in runs)
    return {
        "run_count": len(runs),
        "occupied_pixels": occupied,
        "mask_occupancy_fraction": occupied / (h * w),
        "run_scan_wall_s": scan_wall,
    }


def run_mode(
    *,
    mode: str,
    gds: Path,
    size: int,
    core: int,
) -> dict[str, Any]:
    all_t0 = time.perf_counter()
    t0 = time.perf_counter()
    parent = load_parent(gds)
    parent_load_wall = time.perf_counter() - t0
    crop = centered_crop(parent, size)
    stats = crop_run_stats(crop)
    forward = make_forward()
    h, w = crop.shape
    full_pixels = h * w

    result = {
        "mode": mode,
        "size_px": [h, w],
        "physical_extent_um": [w * PIXEL_NM / 1000.0, h * PIXEL_NM / 1000.0],
        "parent_shape_px": list(parent.shape),
        "crop_bbox_parent_px": [
            crop.crop_bbox_parent.x0,
            crop.crop_bbox_parent.y0,
            crop.crop_bbox_parent.x1,
            crop.crop_bbox_parent.y1,
        ],
        "parent_load_wall_s": parent_load_wall,
        "crop_stats": stats,
        "dense_input_bytes_structural": full_pixels * 4,
        "structural_max_window_bytes": (
            min(h, core + 2 * FORWARD_RADIUS) * min(w, core + 2 * FORWARD_RADIUS) * 4
        ),
    }

    exec_t0 = time.perf_counter()
    if mode == "dense_full":
        dense = crop.read_window(BoundingBox(0, 0, w, h))
        out = forward(dense)
        checksum = float(out.sum().item())
        work = {
            "full_chip_pixels": full_pixels,
            "active_pixels": full_pixels,
            "active_fraction": 1.0,
            "screened_out_pixels": 0,
            "screened_fraction": 0.0,
            "avoided_pct": 0.0,
            "forward_simulator_calls": 1,
            "forward_simulator_input_pixels": full_pixels,
            "read_window_calls": 1,
            "read_window_pixels": full_pixels,
            "dense_allocation_events": 1,
        }
        n_tiles = 1
        del dense, out
    elif mode == "tiled_raster":
        dense = crop.read_window(BoundingBox(0, 0, w, h))
        source = TensorTileSource(dense, pixel_size_nm=PIXEL_NM)
        sink = MetricOnlyTileSink((h, w))
        report = run_streaming(
            source,
            sink,
            forward,
            core_size=core,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
        )
        checksum = float("nan")
        work = dict(report.work_accounting)
        work["dense_allocation_events"] = 1
        n_tiles = report.n_tiles
        del dense
    elif mode == "b04_vector":
        sink = MetricOnlyTileSink((h, w))
        report = run_streaming(
            crop,
            sink,
            forward,
            core_size=core,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
        )
        checksum = float("nan")
        work = dict(report.work_accounting)
        n_tiles = report.n_tiles
    elif mode == "b04_selective":
        sink = MetricOnlyTileSink((h, w))
        report = run_streaming(
            crop,
            sink,
            forward,
            core_size=core,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
            screening_policy=selective_policy(),
        )
        checksum = float("nan")
        work = dict(report.work_accounting)
        n_tiles = report.n_tiles
    elif mode == "screen_only":
        # Measures the theorem-safe active-set discovery cost without running
        # the expensive forward model.  This is allowed only as a diagnostic
        # at larger sizes and never contributes to a speedup crossover claim.
        from openlithohub.streaming.core_halo import plan_tile_requests

        requests = plan_tile_requests(crop.shape, core, FORWARD_RADIUS)
        policy = selective_policy()
        active = 0
        screened = 0
        q0 = time.perf_counter()
        for req in requests:
            decision = policy.screen(crop, req)
            if decision.status == "SCREENED_OUT":
                screened += req.core_bbox.area
            else:
                active += req.core_bbox.area
        qwall = time.perf_counter() - q0
        checksum = float("nan")
        work = {
            "full_chip_pixels": full_pixels,
            "active_pixels": active,
            "active_fraction": active / full_pixels,
            "screened_out_pixels": screened,
            "screened_fraction": screened / full_pixels,
            "avoided_pct": 100.0 * screened / full_pixels,
            "forward_simulator_calls": 0,
            "forward_simulator_input_pixels": 0,
            "read_window_calls": 0,
            "read_window_pixels": 0,
            "dense_allocation_events": 0,
            "screen_queries": len(requests),
            "screen_query_wall_s": qwall,
            "accounted_pct": 100.0 * (active + screened) / full_pixels,
        }
        n_tiles = len(requests)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    result.update(
        {
            "execution_wall_s": time.perf_counter() - exec_t0,
            "end_to_end_wall_s": time.perf_counter() - all_t0,
            "peak_rss_bytes": peak_rss_bytes(),
            "n_tiles": n_tiles,
            "checksum": checksum,
            "work": work,
        }
    )
    return result


def worker(script: Path, *, mode: str, gds: Path, size: int, core: int) -> dict[str, Any]:
    # Fixed argv re-invoking this script's own worker mode; no shell, no untrusted input.
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script),
            "--worker",
            "--mode",
            mode,
            "--gds",
            str(gds),
            "--size",
            str(size),
            "--core",
            str(core),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed mode={mode} size={size}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout)


def median_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(records[0])
    for key in (
        "parent_load_wall_s",
        "execution_wall_s",
        "end_to_end_wall_s",
        "peak_rss_bytes",
    ):
        vals = [float(r[key]) for r in records]
        value = statistics.median(vals)
        out[key] = int(value) if key == "peak_rss_bytes" else value
    out["repeats"] = len(records)
    return out


def first_crossover(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    contender: str,
    time_key: str,
) -> int | None:
    by_size: dict[int, dict[str, float]] = {}
    for row in rows:
        size = int(row["size_px"][0])
        by_size.setdefault(size, {})[row["mode"]] = float(row[time_key])
    for size in sorted(by_size):
        pair = by_size[size]
        if baseline in pair and contender in pair and pair[contender] < pair[baseline]:
            return size
    return None


def correctness_witness(gds: Path, *, size: int = 2048, core: int = 512) -> dict[str, Any]:
    parent = load_parent(gds)
    crop = centered_crop(parent, size)
    forward = make_forward()
    dense = crop.read_window(BoundingBox(0, 0, size, size))
    ref = forward(dense)
    sink = TensorTileSink((size, size))
    report = run_streaming(
        crop,
        sink,
        forward,
        core_size=core,
        halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
        screening_policy=selective_policy(),
    )
    got = sink.finalize()
    return {
        "size_px": size,
        "max_abs_error": float((got - ref).abs().max().item()),
        "reference_checksum": float(ref.sum().item()),
        "selective_checksum": float(got.sum().item()),
        "work": dict(report.work_accounting),
    }


def parse_metadata(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    overview = raw["design_metrics"]["overview"]
    design = raw["design_info"]
    return {
        "name": design["name"],
        "topcell": design["topcell"],
        "description": design["description"],
        "pdk": design["design_flow"]["technology"]["pdk"],
        "cell_count": int(overview["instance"]["cell"]),
        "core_area_um2": float(overview["area"]["core_area"]),
        "die_area_um2": float(overview["area"]["die_area"]),
        "stdcell_area_um2": float(overview["area"]["stdcell_area"]),
        "utilization": float(overview["area"]["utilization"]),
        "stdcell_utilization": float(overview["area"]["stdcell_utilization"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gds", type=Path, required=True)
    ap.add_argument("--metadata", type=Path)
    ap.add_argument("--sizes", default="4096,8192,16384")
    ap.add_argument("--screen-only-sizes", default="32768")
    ap.add_argument("--core", type=int, default=512)
    ap.add_argument("--dense-max-size", type=int, default=4096)
    ap.add_argument("--vector-max-size", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode")
    ap.add_argument("--size", type=int)
    args = ap.parse_args()

    if args.worker:
        print(
            json.dumps(
                run_mode(
                    mode=args.mode,
                    gds=args.gds,
                    size=args.size,
                    core=args.core,
                ),
                sort_keys=True,
            )
        )
        return 0

    if args.out is None or args.metadata is None:
        raise SystemExit("--out and --metadata are required")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = parse_metadata(args.metadata)
    if metadata["topcell"] != TOP_CELL or metadata["pdk"] != "sky130hd":
        raise RuntimeError(f"unexpected PDB metadata: {metadata}")
    if not (0.4 <= metadata["stdcell_utilization"] <= 0.8):
        raise RuntimeError("fixture no longer lies in the intended realistic-density band")

    witness = correctness_witness(args.gds)
    if witness["max_abs_error"] > 2e-6:
        raise RuntimeError(f"correctness witness failed: {witness}")

    script = Path(__file__).resolve()
    rows = []
    for size in [int(x) for x in args.sizes.split(",") if x]:
        modes = ["b04_selective"]
        if size <= args.vector_max_size:
            modes.insert(0, "b04_vector")
        if size <= args.dense_max_size:
            modes = ["dense_full", "tiled_raster"] + modes
        for mode in modes:
            reps = [
                worker(
                    script,
                    mode=mode,
                    gds=args.gds,
                    size=size,
                    core=args.core,
                )
                for _ in range(args.repeats)
            ]
            rows.append(median_record(reps))

    screen_rows = []
    for size in [int(x) for x in args.screen_only_sizes.split(",") if x]:
        screen_rows.append(
            worker(
                script,
                mode="screen_only",
                gds=args.gds,
                size=size,
                core=args.core,
            )
        )

    selective = [r for r in rows if r["mode"] == "b04_selective"]
    no_dense = all(int(r["work"]["dense_allocation_events"]) == 0 for r in selective)
    accounting_closed = all(
        abs(float(r["work"]["accounted_pct"]) - 100.0) < 1e-8 for r in selective
    )

    max_size_row = max(selective, key=lambda r: int(r["size_px"][0]))
    active_at_max = float(max_size_row["work"]["active_fraction"])
    if active_at_max <= 0.5:
        empty_screen_verdict = "STRONG_ON_REALISTIC_DENSITY"
    elif active_at_max <= 0.8:
        empty_screen_verdict = "MODERATE_ON_REALISTIC_DENSITY"
    else:
        empty_screen_verdict = "WEAK_ON_REALISTIC_DENSITY"

    result = {
        "status": "PASS_REAL_DENSITY_EXTERNAL_VALIDITY"
        if no_dense and accounting_closed
        else "FAIL",
        "fixture": {
            "dataset": "PDB Physical Design Database",
            "design_metadata": metadata,
            "gds_path": args.gds.name,
            "pixel_nm": PIXEL_NM,
            "layer_datatype": LAYER,
            "crop_rule": "DETERMINISTIC_NESTED_CENTER_CROPS",
        },
        "correctness_witness": witness,
        "rows": rows,
        "large_screen_only_rows": screen_rows,
        "crossover": {
            "post_parse_selective_vs_dense_full_px": first_crossover(
                rows, baseline="dense_full", contender="b04_selective", time_key="execution_wall_s"
            ),
            "end_to_end_selective_vs_dense_full_px": first_crossover(
                rows, baseline="dense_full", contender="b04_selective", time_key="end_to_end_wall_s"
            ),
            "post_parse_selective_vs_tiled_raster_px": first_crossover(
                rows,
                baseline="tiled_raster",
                contender="b04_selective",
                time_key="execution_wall_s",
            ),
            "end_to_end_selective_vs_vector_no_prune_px": first_crossover(
                rows, baseline="b04_vector", contender="b04_selective", time_key="end_to_end_wall_s"
            ),
        },
        "empty_context_screening_verdict": empty_screen_verdict,
        "decision_rule": {
            "STRONG_ON_REALISTIC_DENSITY": (
                "continue integrating survivor-only continuous verification"
            ),
            "MODERATE_ON_REALISTIC_DENSITY": (
                "retain empty screening but add stronger nonempty safe screening"
            ),
            "WEAK_ON_REALISTIC_DENSITY": (
                "redirect next work-avoidance mainline to nonempty tile margin/edge screening"
            ),
        },
        "claim_firewall": {
            "real_layout": "PDB_SKY130HD_IBEX_ROUTED_GDS",
            "real_design_density": "FROM_PINNED_PUBLIC_METADATA",
            "forward_model": "SYNTHETIC_FINITE_SUPPORT_9X9",
            "foundry_calibration": "NOT_CLAIMED",
            "full_chip_scope": "REAL_ROUTED_BLOCK_CROPS_NOT_ENTIRE_IBEX_CORE",
            "continuous_certification": "NOT_EXERCISED_BY_THIS_PERFORMANCE_GATE",
            "speedup_claim": "ONLY_FOR_RETURNED_COMMON_SIZE_COMPARISONS",
        },
    }
    out = args.out / "B04_INC29_REAL_DENSITY_IBEX.json"
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_REAL_DENSITY_EXTERNAL_VALIDITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
