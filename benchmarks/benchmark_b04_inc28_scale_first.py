#!/usr/bin/env python3
"""B04 Increment 28 scale-first work-avoidance benchmark.

Four execution modes use the same finite-support zero-preserving forward model:

1. ``dense_full``: full exact-vector mask materialized, one whole-raster forward.
2. ``tiled_raster``: full mask materialized, ordinary tiled forward, no pruning.
3. ``b04_vector``: exact-vector window streaming, no pruning.
4. ``b04_selective``: exact-vector streaming + certified empty-context screen.

The benchmark contains:
- a synthetic sparse size ladder: 512², 2048², 4096², 8192² by default;
- an optional public Sky130 hierarchical sparse-placement ladder:
  4096², 8192², 16384² by default.

The forward model is a deterministic 9x9 separable finite-support blur.  This
benchmark measures architecture and work avoidance, not calibrated lithography
physics.  Empty-context screening is theorem-safe for this benchmark because
the halo equals the exact four-pixel support radius and the model maps an
all-zero read context to an all-zero trusted core.

Dense modes are guarded by ``--dense-max-size`` to prevent a benchmark from
creating an unsafe full-chip allocation merely to prove that B04 avoids one.
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

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.screening import ExactEmptyContextScreeningPolicy
from openlithohub.streaming.sinks import MetricOnlyTileSink, TensorTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.vector_runs import (
    ExactVectorRunSource,
    KLayoutAlignedRunSource,
    PolygonWithHoles,
    VectorCell,
)

FORWARD_RADIUS = 4
FORWARD_KERNEL = 2 * FORWARD_RADIUS + 1


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def make_forward():
    ax = torch.arange(FORWARD_KERNEL, dtype=torch.float32) - FORWARD_RADIUS
    g = torch.exp(-(ax**2) / (2.0 * 1.6**2))
    g = g / g.sum()
    kx = g.reshape(1, 1, 1, -1)
    ky = g.reshape(1, 1, -1, 1)

    def forward(tile: torch.Tensor) -> torch.Tensor:
        x = tile.float().unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.conv2d(
            x,
            kx,
            padding=(0, FORWARD_RADIUS),
        )
        x = torch.nn.functional.conv2d(
            x,
            ky,
            padding=(FORWARD_RADIUS, 0),
        )
        return x.squeeze(0).squeeze(0)

    return forward


def rect_poly(
    object_id: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> PolygonWithHoles:
    return PolygonWithHoles(
        object_id=object_id,
        outer=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
    )


def make_synthetic_source(size: int) -> ExactVectorRunSource:
    side = max(24, min(96, size // 16))
    anchors = (
        (0.12, 0.14),
        (0.30, 0.72),
        (0.52, 0.24),
        (0.76, 0.62),
        (0.84, 0.16),
        (0.18, 0.48),
    )
    polys = []
    for i, (fx, fy) in enumerate(anchors):
        x0 = min(size - side - 1, max(1, int(fx * size)))
        y0 = min(size - side - 1, max(1, int(fy * size)))
        polys.append(rect_poly(f"island-{i}", x0, y0, x0 + side, y0 + side))
    cell = VectorCell(name="TOP", polygons=tuple(polys), instances=())
    return ExactVectorRunSource(
        shape=(size, size),
        cells={"TOP": cell},
        top="TOP",
        pixel_size_nm=8.0,
    )


def load_public_source(
    wrapper: Path,
    *,
    top_cell: str,
) -> KLayoutAlignedRunSource:
    return KLayoutAlignedRunSource.from_file(
        wrapper,
        pixel_size_nm=1.0,
        layer="66:44",
        top_cell=top_cell,
    )


def make_public_wrapper(
    public_gds: Path,
    out: Path,
    *,
    size_px: int,
) -> str:
    import klayout.db as db

    layout = db.Layout()
    layout.read(str(public_gds))
    child = layout.cell("sky130_fd_sc_hd__inv_1")
    if child is None:
        raise RuntimeError("Sky130 inverter cell not found")

    cb = child.bbox()
    if cb.width() > size_px or cb.height() > size_px:
        raise RuntimeError(f"public motif {cb.width()}x{cb.height()} does not fit {size_px}")

    top_name = f"B04_INC28_SCALE_{size_px}"
    top = layout.create_cell(top_name)
    x0 = (size_px - cb.width()) // 2 - cb.left
    y0 = (size_px - cb.height()) // 2 - cb.bottom
    top.insert(db.CellInstArray(child.cell_index(), db.Trans(x0, y0)))

    anchor = layout.layer(999, 0)
    top.shapes(anchor).insert(db.Box(0, 0, 1, 1))
    top.shapes(anchor).insert(db.Box(size_px - 1, size_px - 1, size_px, size_px))
    layout.write(str(out))
    return top_name


def make_source(
    *,
    source_kind: str,
    size: int,
    wrapper: Path | None,
    top_cell: str | None,
):
    if source_kind == "synthetic":
        return make_synthetic_source(size)
    if source_kind == "public":
        if wrapper is None or top_cell is None:
            raise ValueError("public source needs wrapper and top_cell")
        return load_public_source(wrapper, top_cell=top_cell)
    raise ValueError(f"unknown source_kind {source_kind!r}")


def global_run_count(source) -> int:
    h, w = source.shape
    return sum(1 for _ in source.iter_runs_for_bbox((0, 0, w, h)))


def selective_policy() -> ExactEmptyContextScreeningPolicy:
    return ExactEmptyContextScreeningPolicy(
        context_certified=True,
        zero_response_certified=True,
        fill_value=0.0,
        model_provenance=("benchmark-separable-9x9-finite-support-radius4-zero-preserving"),
    )


def run_mode(
    *,
    mode: str,
    source_kind: str,
    size: int,
    core: int,
    wrapper: Path | None,
    top_cell: str | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    source = make_source(
        source_kind=source_kind,
        size=size,
        wrapper=wrapper,
        top_cell=top_cell,
    )
    source_build_s = time.perf_counter() - t0
    forward = make_forward()
    h, w = source.shape
    full_pixels = h * w
    dense_input_bytes = full_pixels * 4

    result: dict[str, Any] = {
        "mode": mode,
        "source_kind": source_kind,
        "size_px": [h, w],
        "full_chip_pixels": full_pixels,
        "source_build_wall_s": source_build_s,
        "global_run_count": global_run_count(source),
        "dense_input_bytes_structural": dense_input_bytes,
        "process_points_evaluated": 1,
        "rigorous_cells": 0,
    }

    t0 = time.perf_counter()
    if mode == "dense_full":
        dense = source.read_window(BoundingBox(0, 0, w, h))
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
        del out, dense
        n_tiles = 1
    elif mode == "tiled_raster":
        dense = source.read_window(BoundingBox(0, 0, w, h))
        dense_source = TensorTileSource(dense, pixel_size_nm=source.pixel_size_nm)
        sink = MetricOnlyTileSink((h, w))
        report = run_streaming(
            dense_source,
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
            source,
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
            source,
            sink,
            forward,
            core_size=core,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
            screening_policy=selective_policy(),
        )
        checksum = float("nan")
        work = dict(report.work_accounting)
        n_tiles = report.n_tiles
    else:
        raise ValueError(f"unknown mode {mode!r}")

    result.update(
        {
            "execution_wall_s": time.perf_counter() - t0,
            "peak_rss_bytes": peak_rss_bytes(),
            "n_tiles": n_tiles,
            "checksum": checksum,
            "work": work,
            "structural_max_window_bytes": (
                min(h, core + 2 * FORWARD_RADIUS) * min(w, core + 2 * FORWARD_RADIUS) * 4
            ),
        }
    )
    return result


def correctness_witness() -> dict[str, float]:
    size = 512
    source = make_synthetic_source(size)
    forward = make_forward()
    dense = source.read_window(BoundingBox(0, 0, size, size))
    reference = forward(dense)

    sink = TensorTileSink((size, size))
    run_streaming(
        source,
        sink,
        forward,
        core_size=128,
        halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
        screening_policy=selective_policy(),
    )
    selective = sink.finalize()
    return {
        "size": size,
        "max_abs_error": float((selective - reference).abs().max().item()),
        "reference_checksum": float(reference.sum().item()),
        "selective_checksum": float(selective.sum().item()),
    }


def run_worker_subprocess(
    script: Path,
    *,
    mode: str,
    source_kind: str,
    size: int,
    core: int,
    wrapper: Path | None,
    top_cell: str | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--mode",
        mode,
        "--source-kind",
        source_kind,
        "--size",
        str(size),
        "--core",
        str(core),
    ]
    if wrapper is not None:
        cmd += ["--wrapper", str(wrapper)]
    if top_cell is not None:
        cmd += ["--top-cell", top_cell]
    # Fixed argv re-invoking this script's own worker mode; no shell, no untrusted input.
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed mode={mode} size={size}: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def median_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no records")
    out = dict(records[0])
    out["execution_wall_s"] = statistics.median(float(r["execution_wall_s"]) for r in records)
    out["source_build_wall_s"] = statistics.median(float(r["source_build_wall_s"]) for r in records)
    out["peak_rss_bytes"] = int(statistics.median(int(r["peak_rss_bytes"]) for r in records))
    out["repeats"] = len(records)
    return out


def first_crossover(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    contender: str,
) -> int | None:
    by_size: dict[int, dict[str, float]] = {}
    for row in rows:
        by_size.setdefault(int(row["size_px"][0]), {})[row["mode"]] = float(row["execution_wall_s"])
    for size in sorted(by_size):
        pair = by_size[size]
        if pair.get(baseline) is not None and pair[contender] < pair[baseline]:
            return size
    return None


def ladder(
    *,
    script: Path,
    source_kind: str,
    sizes: list[int],
    core: int,
    dense_max_size: int,
    no_prune_max_size: int,
    repeats: int,
    wrappers: dict[int, tuple[Path, str]] | None = None,
) -> dict[str, Any]:
    rows = []
    for size in sizes:
        wrapper = None
        top_cell = None
        if wrappers is not None:
            wrapper, top_cell = wrappers[size]
        for mode in (
            "dense_full",
            "tiled_raster",
            "b04_vector",
            "b04_selective",
        ):
            if mode in ("dense_full", "tiled_raster") and size > dense_max_size:
                continue
            if mode == "b04_vector" and size > no_prune_max_size:
                continue
            reps = []
            for _ in range(repeats):
                reps.append(
                    run_worker_subprocess(
                        script,
                        mode=mode,
                        source_kind=source_kind,
                        size=size,
                        core=core,
                        wrapper=wrapper,
                        top_cell=top_cell,
                    )
                )
            rows.append(median_record(reps))

    return {
        "source_kind": source_kind,
        "rows": rows,
        "crossover": {
            "selective_vs_dense_full_px": first_crossover(
                rows,
                baseline="dense_full",
                contender="b04_selective",
            ),
            "selective_vs_tiled_raster_px": first_crossover(
                rows,
                baseline="tiled_raster",
                contender="b04_selective",
            ),
            "selective_vs_vector_no_prune_px": first_crossover(
                rows,
                baseline="b04_vector",
                contender="b04_selective",
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--public-gds", type=Path, default=None)
    ap.add_argument("--sizes", default="512,2048,4096,8192")
    ap.add_argument("--public-sizes", default="4096,8192,16384")
    ap.add_argument("--core", type=int, default=256)
    ap.add_argument("--public-core", type=int, default=512)
    ap.add_argument("--dense-max-size", type=int, default=4096)
    ap.add_argument("--no-prune-max-size", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=2)

    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode")
    ap.add_argument("--source-kind")
    ap.add_argument("--size", type=int)
    ap.add_argument("--wrapper", type=Path)
    ap.add_argument("--top-cell")
    args = ap.parse_args()

    if args.worker:
        record = run_mode(
            mode=args.mode,
            source_kind=args.source_kind,
            size=args.size,
            core=args.core,
            wrapper=args.wrapper,
            top_cell=args.top_cell,
        )
        print(json.dumps(record, sort_keys=True))
        return 0

    if args.out is None:
        raise SystemExit("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()

    witness = correctness_witness()
    if witness["max_abs_error"] > 2e-6:
        raise RuntimeError(f"selective correctness witness failed: {witness}")

    synthetic = ladder(
        script=script,
        source_kind="synthetic",
        sizes=[int(x) for x in args.sizes.split(",")],
        core=args.core,
        dense_max_size=args.dense_max_size,
        no_prune_max_size=args.no_prune_max_size,
        repeats=args.repeats,
    )

    public = None
    if args.public_gds is not None and args.public_gds.exists():
        wrappers: dict[int, tuple[Path, str]] = {}
        for size in [int(x) for x in args.public_sizes.split(",")]:
            path = args.out / f"sky130_sparse_hierarchy_{size}.gds"
            top = make_public_wrapper(args.public_gds, path, size_px=size)
            wrappers[size] = (path, top)
        public = ladder(
            script=script,
            source_kind="public",
            sizes=sorted(wrappers),
            core=args.public_core,
            dense_max_size=args.dense_max_size,
            no_prune_max_size=args.no_prune_max_size,
            repeats=max(1, args.repeats),
            wrappers=wrappers,
        )

    all_rows = list(synthetic["rows"])
    if public is not None:
        all_rows += list(public["rows"])

    selective_rows = [row for row in all_rows if row["mode"] == "b04_selective"]
    no_dense_alloc = all(
        int(row["work"].get("dense_allocation_events", 0)) == 0 for row in selective_rows
    )
    accounted = all(
        abs(float(row["work"].get("accounted_pct", 0.0)) - 100.0) < 1e-8 for row in selective_rows
    )

    result = {
        "status": ("PASS_SCALE_FIRST_WORK_AVOIDANCE" if no_dense_alloc and accounted else "FAIL"),
        "benchmark_scope": {
            "forward_model": ("SYNTHETIC_FINITE_SUPPORT_9X9_ZERO_PRESERVING"),
            "physics_claim": "NOT_FOUNDRY_CALIBRATED",
            "correctness_witness": witness,
            "dense_guard_max_size_px": args.dense_max_size,
            "vector_no_prune_guard_max_size_px": args.no_prune_max_size,
            "repeats": args.repeats,
        },
        "synthetic_sparse_ladder": synthetic,
        "public_sky130_sparse_hierarchy_ladder": public,
        "claim_firewall": {
            "large_layout_memory_scaling": ("MEASURED_FOR_THIS_ARCHITECTURE_BENCHMARK"),
            "work_avoidance": ("MEASURED_FROM_INTEGRATED_PIPELINE_ACCOUNTING"),
            "speed_crossover": ("CLAIM_ONLY_WHERE_CROSSOVER_FIELD_IS_NON_NULL"),
            "public_layout_realism": (
                "REAL_SKY130_MOTIF_IN_SYNTHETIC_SPARSE_HIERARCHICAL_PLACEMENT"
            ),
            "production_full_chip_speedup": "NOT_YET_CLAIMED",
            "continuous_certification": ("NOT_EXERCISED_BY_THIS_PERFORMANCE_BENCHMARK"),
        },
    }

    out = args.out / "B04_INC28_SCALE_FIRST_WORK_AVOIDANCE.json"
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_SCALE_FIRST_WORK_AVOIDANCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
