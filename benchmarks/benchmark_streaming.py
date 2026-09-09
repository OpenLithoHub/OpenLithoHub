#!/usr/bin/env python3
"""Streaming core/halo pipeline vs legacy whole-raster tiling (RFC 0008, §22-G).

Compares, on the same synthetic layout:

- legacy path: `tile_layout` + per-tile forward + `stitch_tiles`
  (requires dense input + dense output + dense weight map);
- streaming path: `run_streaming` with `MemmapTileSink` (windowed reads,
  core-only writes, disk-paged output, no weight map).

Reports wall time, tile count, halo overhead, and result discrepancy, plus
the analytically required *resident raster bytes* per mode — the structural
difference the RFC is about. Also sweeps halo candidates to report the
empirically stable minimum halo and the kernel-tail certified halo (§14).

Usage:  python benchmarks/benchmark_streaming.py [--size 2048] [--tile 512]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from openlithohub.streaming.core_halo import plan_tile_requests
from openlithohub.streaming.halo_policy import (
    HaloContext,
    HaloRequirement,
    KernelTailHaloPolicy,
    estimate_minimum_halo,
    kernel_tail_mass,
)
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.sinks import MemmapTileSink
from openlithohub.streaming.sources import MemmapTensorTileSource
from openlithohub.workflow.tiling import stitch_tiles, tile_layout


def make_layout(size: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return (((xx * 7 + yy * 13) % 32) / 32.0).astype(np.float32)


def make_forward(sigma_px: float = 3.0, kernel_size: int = 19):
    """Gaussian-ish separable blur standing in for the forward model."""
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(ax**2) / (2 * sigma_px**2))
    kx = (g / g.sum()).reshape(1, 1, 1, -1)
    ky = (g / g.sum()).reshape(1, 1, -1, 1)

    def forward(tile: torch.Tensor) -> torch.Tensor:
        x = tile.float().unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.conv2d(x, kx, padding=(0, kernel_size // 2))
        x = torch.nn.functional.conv2d(x, ky, padding=(kernel_size // 2, 0))
        return x.squeeze(0).squeeze(0)

    return forward


def bench_legacy(layout: np.ndarray, forward, tile_size: int, overlap: int) -> dict:
    h, w = layout.shape
    t0 = time.perf_counter()
    tiles = tile_layout(torch.from_numpy(layout), tile_size=tile_size, overlap=overlap)
    results = []
    for tile in tiles:
        window = layout[
            tile.origin_y : tile.origin_y + tile.height,
            tile.origin_x : tile.origin_x + tile.width,
        ]
        results.append(forward(torch.from_numpy(window.copy())))
    stitched = stitch_tiles([(t, r) for t, r in zip(tiles, results, strict=False)], (h, w))
    elapsed = time.perf_counter() - t0
    read_px = sum(t.height * t.width for t in tiles)
    # Legacy structurally requires: dense input + dense output + weight map.
    resident_raster_bytes = 3 * h * w * layout.itemsize
    return {
        "wall_time_s": elapsed,
        "n_tiles": len(tiles),
        "resident_raster_mb": resident_raster_bytes / 1e6,
        "halo_overhead_pct": 0.0,  # overlap is not tracked as halo in legacy
        "output": stitched,
        "read_px_total": int(read_px),
    }


def bench_streaming(
    layout: np.ndarray, forward, layout_path, out_path, tile_size: int, halo: int
) -> dict:
    h, w = layout.shape
    t0 = time.perf_counter()
    source = MemmapTensorTileSource(layout_path, dtype=np.float32, shape=(h, w))
    sink = MemmapTileSink((h, w), out_path)
    report = run_streaming(
        source,
        sink,
        forward,
        core_size=tile_size,
        halo_policy=_Fixed(halo),
    )
    elapsed = time.perf_counter() - t0
    requests = plan_tile_requests((h, w), tile_size, halo)
    core_px = sum(r.core_bbox.area for r in requests)
    read_px = sum(r.read_bbox.area for r in requests)
    # Streaming keeps only the current read window resident; the output is
    # paged to disk as trusted cores are produced.
    resident_raster_bytes = max(r.read_bbox.area for r in requests) * layout.itemsize
    return {
        "wall_time_s": elapsed,
        "n_tiles": report.n_tiles,
        "resident_raster_mb": resident_raster_bytes / 1e6,
        "halo_overhead_pct": 100.0 * (read_px - core_px) / core_px,
        "output": sink.finalize(),
        "read_px_total": int(read_px),
    }


class _Fixed:
    """Minimal fixed-halo policy adapter for the benchmark."""

    name = "fixed"

    def __init__(self, halo_px: int) -> None:
        self._halo = halo_px

    def required_halo(self, context):
        return HaloRequirement(
            halo_px=self._halo,
            provenance="benchmark_fixed",
            error_bound=None,
            certified=False,
            reason="benchmark fixed halo",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--halo", type=int, default=16)
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="olh_streaming_bench_"))
    layout = make_layout(args.size)
    layout_path = tmp / "layout.f32"
    raw = np.memmap(layout_path, dtype=np.float32, mode="w+", shape=layout.shape)
    raw[...] = layout
    raw.flush()
    del raw

    forward = make_forward()

    print(f"layout={args.size}x{args.size} tile={args.tile} halo={args.halo}")
    legacy = bench_legacy(layout, forward, args.tile, args.halo)
    streaming = bench_streaming(layout, forward, layout_path, tmp / "out.f32", args.tile, args.halo)

    # result discrepancy vs full-context forward on the whole layout
    full = forward(torch.from_numpy(layout)).numpy()
    out = np.memmap(tmp / "out.f32", dtype=np.float32, mode="r", shape=layout.shape)
    streaming_disc = float(np.abs(np.asarray(out) - full).max())
    legacy_disc = float(np.abs(legacy.pop("output").numpy() - full).max())

    rows = [
        ("wall_time_s", round(legacy["wall_time_s"], 3), round(streaming["wall_time_s"], 3)),
        ("n_tiles", legacy["n_tiles"], streaming["n_tiles"]),
        (
            "resident_raster_mb",
            round(legacy["resident_raster_mb"], 1),
            round(streaming["resident_raster_mb"], 1),
        ),
        (
            "halo_overhead_pct",
            round(legacy["halo_overhead_pct"], 2),
            round(streaming["halo_overhead_pct"], 2),
        ),
        ("max_discrepancy", round(legacy_disc, 6), round(streaming_disc, 6)),
    ]
    print(f"{'metric':<24}{'legacy':>14}{'streaming':>14}")
    for name, left, right in rows:
        print(f"{name:<24}{str(left):>14}{str(right):>14}")

    # §14: halo adequacy — kernel-tail certified halo + empirical sweep
    kernel = torch.ones(19, 19) / (19 * 19)
    tail_policy = KernelTailHaloPolicy(tolerance=1e-3)
    tail_req = tail_policy.required_halo(
        HaloContext(pixel_nm=1.0, tile_core_px=args.tile, kernel=kernel)
    )
    corner = 64 if args.tile > 256 else 8
    sweep = estimate_minimum_halo(
        forward,
        core_bbox_px=(corner, corner, args.tile - corner, args.tile - corner),
        full_context=torch.from_numpy(layout),
        candidate_halos=(0, 4, 8, 12, 16, 24, 32),
        tolerance=1e-4,
    )
    print("\nhalo adequacy (§14):")
    print(
        f"  kernel-tail halo  : {tail_req.halo_px} px "
        f"(status={tail_req.status.value}, bound={tail_req.error_bound})"
    )
    print(f"  empirical halo    : {sweep.halo_px} px (status={sweep.status.value})")
    print(f"  kernel tail mass at empirical halo: {kernel_tail_mass(kernel, sweep.halo_px):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
