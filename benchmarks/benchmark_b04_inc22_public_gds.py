#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import klayout.db as db

from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_cell_and_layer(path: Path):
    ly = db.Layout()
    ly.read(str(path))
    cell = ly.cell("sky130_fd_sc_hd__inv_1")
    if cell is None:
        tops = list(ly.top_cells())
        if not tops:
            raise RuntimeError("no top cells")
        cell = tops[0]
    best = None
    best_count = -1
    for li in ly.layer_indexes():
        it = cell.begin_shapes_rec(li)
        c = 0
        while not it.at_end():
            c += 1
            if c >= 20000:
                break
            it.next()
        if c > best_count:
            info = ly.get_info(li)
            best = f"{int(info.layer)}:{int(info.datatype)}"
            best_count = c
    if best is None or best_count <= 0:
        raise RuntimeError("selected cell has no geometric layer")
    # sky130 stores dbu = 0.001 µm, which in IEEE-754 reads back as
    # 0.0009999999999999998 µm. Return nm-per-dbu as the file itself
    # encodes it so the DBU-per-pixel ratio is exactly 1.0 downstream —
    # a hard-coded 1.0 nm request fails the theorem adapter's exactness
    # guard (1.0 / 0.9999999999999998 = 1.0000000000000002).
    return cell.name, best, best_count, ly.dbu * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=128)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cell, layer, n_shapes, dbu_nm = choose_cell_and_layer(args.gds)
    tracemalloc.start()
    t0 = time.perf_counter()
    src = KLayoutAlignedRunSource.from_file(
        args.gds, pixel_size_nm=dbu_nm, layer=layer, top_cell=cell
    )
    build_s = time.perf_counter() - t0
    _, build_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    h, w = src.shape
    run_count = 0
    run_pixels = 0
    parent_ids = set()
    max_window_bytes = 0
    n_windows = 0
    t0 = time.perf_counter()
    for y0 in range(0, h, args.window):
        for x0 in range(0, w, args.window):
            box = BoundingBox(x0, y0, min(w, x0 + args.window), min(h, y0 + args.window))
            rows = list(src.iter_owned_runs_for_bbox(box, tile_id=f"{x0}:{y0}"))
            run_count += len(rows)
            run_pixels += sum(r.x1 - r.x0 for r in rows)
            parent_ids.update(r.run_id for r in rows)
            tile = src.read_window(box)
            max_window_bytes = max(max_window_bytes, tile.numel() * tile.element_size())
            n_windows += 1
    stream_s = time.perf_counter() - t0

    result = {
        "status": "PASS",
        "fixture_sha256": sha256(args.gds),
        "top_cell": cell,
        "layer_datatype": layer,
        "recursive_shape_count_capped": n_shapes,
        "shape_px": list(src.shape),
        "window_px": args.window,
        "windows": n_windows,
        "run_views": run_count,
        "unique_parent_runs": len(parent_ids),
        "run_pixels_with_read_overlap": run_pixels,
        "source_build_wall_s": build_s,
        "stream_window_wall_s": stream_s,
        "python_build_peak_bytes": build_peak,
        "max_window_tensor_bytes": max_window_bytes,
        "backend_kind": src.backend_kind,
        "physical_metrics": {
            "EPE": "NOT_RUN_THIS_GATE",
            "CD": "NOT_RUN_THIS_GATE",
            "contour_Hausdorff": "NOT_RUN_THIS_GATE",
            "process_window": "NOT_RUN_THIS_GATE",
        },
        "claim_firewall": (
            "Parser/ownership benchmark only; no speedup or certified-process-window claim."
        ),
    }
    out = args.out / "PUBLIC_SKY130_GDS_BENCHMARK.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
