#!/usr/bin/env python3
"""B04 Increment 21 real-vector streaming benchmark.

Generates a deterministic aligned GDS fixture through KLayout, then compares:
1. canonical full-raster load_layout;
2. exact vector scanline window reads;
3. RFC-0008 MetricOnlyTileSink + ownership verifier.

No speedup theorem is inferred from wall time.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from openlithohub.data.io import load_layout
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MetricOnlyTileSink
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource
from openlithohub.verify.run_ownership import RunOwnershipVerifier


def generate_gds(path: Path, *, h: int = 512, w: int = 512, pixel_nm: int = 8) -> None:
    import klayout.db as db

    if pixel_nm != 8:
        raise ValueError("benchmark fixture currently fixes pixel_nm=8")
    step = 8
    ly = db.Layout()
    ly.dbu = 0.001
    l1 = ly.layer(1, 0)
    anchor = ly.layer(99, 0)
    child = ly.create_cell("CHILD")
    top = ly.create_cell("TOP")

    # Child hierarchy cell.
    child.shapes(l1).insert(db.Box(0, 0, 24 * step, 16 * step))

    # Long strips crossing many tile boundaries.
    for y in range(32, h - 32, 48):
        py0 = (h - (y + 12)) * step
        py1 = (h - y) * step
        top.shapes(l1).insert(db.Box(4 * step, py0, (w - 4) * step, py1))

    # Sparse blocks and repeated child instances.
    for y in range(24, h - 48, 96):
        for x in range(20, w - 48, 96):
            py0 = (h - (y + 20)) * step
            py1 = (h - y) * step
            top.shapes(l1).insert(db.Box(x * step, py0, (x + 36) * step, py1))
            top.insert(
                db.CellInstArray(
                    child.cell_index(),
                    db.Trans((x + 8) * step, (h - (y + 48)) * step),
                )
            )

    # A donut/hole.
    outer = db.Box(120 * step, (h - 220) * step, 280 * step, (h - 80) * step)
    hole = db.Box(160 * step, (h - 180) * step, 240 * step, (h - 120) * step)
    donut = db.Polygon(outer)
    donut.insert_hole(hole)
    top.shapes(l1).insert(donut)

    # Left/right physical-edge features.
    top.shapes(l1).insert(db.Box(0, (h - 64) * step, 8 * step, (h - 16) * step))
    top.shapes(l1).insert(db.Box((w - 8) * step, (h - 64) * step, w * step, (h - 16) * step))

    # Force exact full physical bbox on a non-selected layer.
    top.shapes(anchor).insert(db.Box(0, 0, 1, 1))
    top.shapes(anchor).insert(db.Box(w * step - 1, h * step - 1, w * step, h * step))
    ly.write(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--core", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gds = args.out / "b04_inc21_benchmark.gds"
    generate_gds(gds)

    tracemalloc.start()
    t0 = time.perf_counter()
    dense = load_layout(gds, pixel_nm=8.0, layer="1:0")
    dense_wall = time.perf_counter() - t0
    _, dense_py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    src = KLayoutAlignedRunSource.from_file(gds, pixel_size_nm=8.0, layer="1:0")
    source_build_wall = time.perf_counter() - t0

    stream_sum = 0.0
    max_window_bytes = 0
    n_windows = 0
    t0 = time.perf_counter()
    for y0 in range(0, src.shape[0], args.core):
        for x0 in range(0, src.shape[1], args.core):
            box = BoundingBox(
                x0, y0, min(src.shape[1], x0 + args.core), min(src.shape[0], y0 + args.core)
            )
            tile = src.read_window(box)
            stream_sum += float(tile.sum().item())
            max_window_bytes = max(max_window_bytes, tile.numel() * tile.element_size())
            n_windows += 1
            del tile
    stream_read_wall = time.perf_counter() - t0
    _, stream_py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sink = MetricOnlyTileSink(src.shape)
    verifier = RunOwnershipVerifier()
    t0 = time.perf_counter()
    report = run_streaming(
        src,
        sink,
        lambda x: x,
        core_size=args.core,
        halo_policy=LegacyFixedHaloPolicy(0),
        verifiers=(verifier,),
        pixel_nm=8.0,
        max_requeues=0,
    )
    pipeline_wall = time.perf_counter() - t0
    sink_summary = sink.finalize()

    dense_bytes = dense.numel() * dense.element_size()
    result = {
        "status": "PASS" if report.verification.status == "PASS" else report.verification.status,
        "fixture": str(gds),
        "shape": list(src.shape),
        "core_size": args.core,
        "n_windows": n_windows,
        "dense": {
            "wall_s": dense_wall,
            "python_tracemalloc_peak_bytes": dense_py_peak,
            "tensor_bytes": dense_bytes,
            "occupied_sum": float(dense.sum().item()),
        },
        "streamed": {
            "source_build_wall_s": source_build_wall,
            "window_read_wall_s": stream_read_wall,
            "python_tracemalloc_peak_bytes": stream_py_peak,
            "max_window_tensor_bytes": max_window_bytes,
            "occupied_sum": stream_sum,
            "dense_to_max_window_tensor_ratio": dense_bytes / max_window_bytes,
        },
        "pipeline": {
            "wall_s": pipeline_wall,
            "n_tiles": report.n_tiles,
            "n_requeued": report.n_requeued,
            "verification_status": report.verification.status,
            "verification_metrics": report.verification.metrics,
            "sink": sink_summary,
        },
        "semantic_note": (
            "dense loader and exact scanline source may expose raster-convention "
            "differences; no equality is assumed for the timing comparison"
        ),
        "timing_status": "NUMERICAL-DIAGNOSTIC",
    }
    (args.out / "B04_INC21_BENCHMARK.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
