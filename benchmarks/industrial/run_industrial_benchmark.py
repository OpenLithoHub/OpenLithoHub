#!/usr/bin/env python3
"""Industrial Benchmark v1 harness — real routed GDS, apples-to-apples.

Measures OpenLithoHub's streaming/exact-vector pipeline against dense
baselines on a real public routed layout (OpenROAD-routed Ibex RISC-V
core, sky130hd, PDB Physical Design Database), with quality comparisons
on the same optical model, on this machine, and writes everything as
``OpenLithoHub.industrial-benchmark.v1`` JSON artifacts.

Stages (each checkpointed; safe to interrupt and resume):

- ``prepare``   correctness witnesses + deterministic GDS fixtures.
- ``runtime``   dense vs tiled vs exact-vector vs selective streaming
                ladder on real center crops (4 sizes), 5 repeats per
                row, median/p10/p90, fresh worker process per timed run.
- ``quality``   same-optical-model (Hopkins SOCS) quality comparison of
                registered models on the highest-occupancy real tiles.
- ``fulldie``   tile-grid streaming coverage of the FULL die — the dense
                raster equivalent is structurally infeasible (1.23 TB).
- ``manifest``  assemble artifacts + SHA-256 index.

Physics claim scope, identical to B04 INC28/29: the runtime/memory
ladder uses a deterministic 9x9 separable finite-support zero-preserving
forward model to measure ARCHITECTURE (work avoidance, memory scaling),
not calibrated lithography physics.  Quality/runtime-at-quality stages
use the built-in Hopkins SOCS forward model with identical parameters
for every compared method; numbers are benchmark-relative and NOT
foundry calibrated.  No commercial tool is compared anywhere.

Claim firewall: this harness produces measurements with provenance; it
never produces marketing numbers.  Public claims are derived only by
``scripts/generate_industrial_claims.py`` from these artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from openlithohub.benchmark.industrial import (
    SCHEMA_NAME,
    STATUS_INFEASIBLE_ON_REFERENCE_MACHINE,
    STATUS_NOT_RUN_MEMORY_POLICY,
    STATUS_SUCCESS,
    _sanitize,
    environment_snapshot,
    measurement_source,
    memory_reduction_pct,
    relative_reduction_pct,
    sha256_file,
    summarize,
    validate_artifact,
)
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.screening import ExactEmptyContextScreeningPolicy
from openlithohub.streaming.sinks import MetricOnlyTileSink, TensorTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

FORWARD_RADIUS = 4
FORWARD_SUPPORT_PX = 2 * FORWARD_RADIUS + 1
LAYER = "66:44"  # sky130hd metal1 (li1 in PDB GDS convention used by B04 INC29)
SEED = 0

# Dataset identity: OpenROAD-routed Ibex core from the PDB Physical
# Design Database (public).  Pinned by content hash at measurement time.
DATASET = {
    "name": "pdb-sky130hd-ibex",
    "design": "ibex_core (32-bit RISC-V CPU core, two-stage pipeline)",
    "pdk": "sky130hd",
    "source": "SJTU-YONGFU-RESEARCH-GRP/PDB-Physical-Design-Database",
    "source_commit": "9e1e3399b1b707f26fee853bce1ff91ab466ce24",
    "layer": LAYER,
    "pixel_size_nm": 1.0,
    "license_note": "public dataset; shipped as adapter-only, not redistributed",
}


def fixture_identity(gds: Path) -> dict[str, Any]:
    """Content identity of the measured fixture (P0.3: provenance closure)."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(gds))
    top = layout.top_cells()[0]
    bb = top.bbox()
    return {
        "filename": gds.name,
        "sha256": sha256_file(gds),
        "bytes": gds.stat().st_size,
        "source_repo": DATASET["source"],
        "source_commit": DATASET["source_commit"],
        "top_cell": top.name,
        "layer": LAYER,
        "pixel_size_nm": DATASET["pixel_size_nm"],
        "die_bbox_dbu": [bb.left, bb.bottom, bb.right, bb.top],
        "dbu_nm": 1.0,
        "cell_count": layout.cells(),
    }


PHYSICS_SCOPE = {
    "ladder_forward_model": (
        "deterministic 9x9 separable finite-support zero-preserving blur (identical to B04 INC28)"
    ),
    "quality_forward_model": (
        "Hopkins SOCS, 193nm, NA 1.35, sigma 0.7, 24 kernels, CTR threshold 0.225"
    ),
    "physics_claim": "NOT_FOUNDRY_CALIBRATED",
    "commercial_tool_comparison": "NONE",
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


_RUN_LOG: Path | None = None


def init_run_log(out: Path) -> None:
    global _RUN_LOG
    out.mkdir(parents=True, exist_ok=True)
    _RUN_LOG = out / "RUN.log"
    with open(_RUN_LOG, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== run start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _RUN_LOG is not None:
        with open(_RUN_LOG, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def make_forward():
    """Deterministic 9x9 separable finite-support blur (B04 INC28-identical)."""
    ax = torch.arange(FORWARD_SUPPORT_PX, dtype=torch.float32) - FORWARD_RADIUS
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
        model_provenance=("benchmark-separable-9x9-finite-support-radius4-zero-preserving"),
    )


def dense_allowed(size: int, *, bytes_per_px: int = 4, tensors: int = 3, budget_bytes: int) -> bool:
    return size * size * bytes_per_px * tensors <= budget_bytes


# ---------------------------------------------------------------------------
# fixture preparation
# ---------------------------------------------------------------------------


def clip_gds(parent_gds: Path, out_gds: Path, *, top_name: str, size_dbu: int) -> dict[str, Any]:
    """Center-crop the parent die to size x size dbu (1 dbu = 1 nm here)."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(parent_gds))
    top = layout.top_cells()[0]
    bb = top.bbox()
    half = size_dbu // 2
    cx, cy = bb.center().x, bb.center().y
    box = kdb.Box(cx - half, cy - half, cx + half, cy + half)
    clipped_idx = layout.clip(top.cell_index(), box)
    layout.rename_cell(clipped_idx, top_name)
    out_ly = kdb.Layout()
    out_ly.dbu = layout.dbu
    new_top = out_ly.create_cell(top_name)
    new_top.copy_tree(layout.cell(clipped_idx))
    out_ly.write(str(out_gds))
    actual = new_top.bbox()
    return {
        "gds": str(out_gds),
        "top_cell": top_name,
        "size_px": [actual.height(), actual.width()],
        "sha256": sha256_file(out_gds),
    }


def make_die_sample_tiles(
    parent_gds: Path, out_dir: Path, *, tile_px: int, tiles_per_side: int
) -> dict[str, Any]:
    """Pre-clip a deterministic sample of die tiles (center + 4 corners).

    The full-die dense raster equivalent is structurally infeasible (see
    stage_fulldie); this fixture set supports a measured *sampled* die
    screening survey, clearly labeled as an estimate when aggregated.
    """
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(parent_gds))
    top = layout.top_cells()[0]
    bb = top.bbox()
    last = tiles_per_side - 1
    sample_positions = [(0, 0), (last, 0), (0, last), (last, last), (last // 2, last // 2)]
    entries = []
    for ix, iy in sample_positions:
        x0 = bb.left + ix * tile_px
        y0 = bb.bottom + iy * tile_px
        box = kdb.Box(x0, y0, min(x0 + tile_px, bb.right), min(y0 + tile_px, bb.top))
        name = f"IBEX_DIE_{ix}_{iy}"
        clipped_idx = layout.clip(top.cell_index(), box)
        layout.rename_cell(clipped_idx, name)
        out_ly = kdb.Layout()
        out_ly.dbu = layout.dbu
        new_top = out_ly.create_cell(name)
        new_top.copy_tree(layout.cell(clipped_idx))
        out_path = out_dir / f"die_tile_{ix:02d}_{iy:02d}.gds"
        out_ly.write(str(out_path))
        entries.append(
            {
                "tile": [ix, iy],
                "gds": str(out_path),
                "top_cell": name,
                "bbox_dbu": [box.left, box.bottom, box.right, box.top],
            }
        )
        layout.delete_cell(clipped_idx)
    return {
        "die_bbox_dbu": [bb.left, bb.bottom, bb.right, bb.top],
        "die_size_px": [bb.height(), bb.width()],
        "tile_px": tile_px,
        "tiles_per_side": tiles_per_side,
        "n_grid_tiles": tiles_per_side * tiles_per_side,
        "n_sample_tiles": len(entries),
        "tiles": entries,
    }


# ---------------------------------------------------------------------------
# worker jobs (each timed run executes in a fresh subprocess)
# ---------------------------------------------------------------------------


def load_source(gds: Path, top_cell: str) -> KLayoutAlignedRunSource:
    return KLayoutAlignedRunSource.from_file(gds, pixel_size_nm=1.0, layer=LAYER, top_cell=top_cell)


def job_runtime_row(args: argparse.Namespace) -> dict[str, Any]:
    """One timed ladder row in a fresh process: parse + execute + peak RSS."""
    torch.manual_seed(SEED)
    t0 = time.perf_counter()
    source = load_source(Path(args.gds), args.top_cell)
    parse_s = time.perf_counter() - t0
    forward = make_forward()
    h, w = source.shape
    full_pixels = h * w
    mode = args.mode

    result: dict[str, Any] = {
        "mode": mode,
        "size_px": [h, w],
        "full_chip_pixels": full_pixels,
        "parse_wall_s": parse_s,
        "core_px": args.core,
        "dense_input_bytes_structural": full_pixels * 4,
        "peak_rss_bytes": peak_rss_bytes(),
    }

    t0 = time.perf_counter()
    screened_fraction = 0.0
    if mode == "dense_full":
        dense = source.read_window(BoundingBox(0, 0, w, h))
        out = forward(dense)
        result["checksum"] = float(out.sum().item())
        n_tiles = 1
        del dense, out
    elif mode == "tiled_raster":
        dense = source.read_window(BoundingBox(0, 0, w, h))
        dense_source = TensorTileSource(dense, pixel_size_nm=source.pixel_size_nm)
        report = run_streaming(
            dense_source,
            MetricOnlyTileSink((h, w)),
            forward,
            core_size=args.core,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
        )
        n_tiles = report.n_tiles
        screened_fraction = float(report.work_accounting.get("screened_fraction", 0.0))
        del dense
    elif mode == "b04_vector":
        report = run_streaming(
            source,
            MetricOnlyTileSink((h, w)),
            forward,
            core_size=args.core,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
        )
        n_tiles = report.n_tiles
        screened_fraction = float(report.work_accounting.get("screened_fraction", 0.0))
    elif mode == "b04_selective":
        report = run_streaming(
            source,
            MetricOnlyTileSink((h, w)),
            forward,
            core_size=args.core,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
            screening_policy=selective_policy(),
        )
        n_tiles = report.n_tiles
        screened_fraction = float(report.work_accounting.get("screened_fraction", 0.0))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    result.update(
        {
            "execution_wall_s": time.perf_counter() - t0,
            "end_to_end_wall_s": parse_s + (time.perf_counter() - t0),
            "n_tiles": n_tiles,
            "screened_fraction": screened_fraction,
            "peak_rss_bytes": peak_rss_bytes(),
        }
    )
    return result


def job_real_witness(args: argparse.Namespace) -> dict[str, Any]:
    """Dense vs selective equivalence on a real occupied crop tile."""
    torch.manual_seed(SEED)
    source = load_source(Path(args.gds), args.top_cell)
    forward = make_forward()
    x0 = args.window
    dense = source.read_window(BoundingBox(x0, x0, x0 + 1024, x0 + 1024))
    reference = forward(dense)
    sink = TensorTileSink((1024, 1024))

    def offset_source_bbox(bbox: BoundingBox) -> BoundingBox:
        return BoundingBox(bbox.x0 + x0, bbox.y0 + x0, bbox.x1 + x0, bbox.y1 + x0)

    # Stream the same 1024x1024 window through the full pipeline by wrapping.
    class _WindowSource(TensorTileSource):
        def read_window(self, bbox: BoundingBox) -> torch.Tensor:  # type: ignore[override]
            return source.read_window(offset_source_bbox(bbox))

    window_src = _WindowSource(dense, pixel_size_nm=source.pixel_size_nm)
    run_streaming(
        window_src,
        sink,
        forward,
        core_size=256,
        halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
        screening_policy=selective_policy(),
    )
    streamed = sink.finalize()
    occupancy = float((dense > 0.5).float().mean())
    return {
        "window_px": 1024,
        "occupancy": occupancy,
        "max_abs_error": float((streamed - reference).abs().max().item()),
        "reference_checksum": float(reference.sum().item()),
        "streamed_checksum": float(streamed.sum().item()),
    }


def _evaluate_mask(
    mask: torch.Tensor,
    design: torch.Tensor,
    *,
    simulator: Any,
    pixel_size_nm: float,
) -> dict[str, float]:
    from openlithohub.benchmark.compliance.mrc import check_mrc
    from openlithohub.benchmark.metrics.epe import compute_epe, compute_wafer_epe
    from openlithohub.benchmark.metrics.l2_error import compute_l2_error
    from openlithohub.benchmark.metrics.pvband import compute_pvband
    from openlithohub.benchmark.metrics.shot_count import estimate_shot_count

    binary = (mask > 0.5).float()
    epe = compute_epe(binary, design, pixel_size_nm=pixel_size_nm)
    wafer = compute_wafer_epe(binary, design, pixel_size_nm=pixel_size_nm, simulator=simulator)
    pvb = compute_pvband(binary, pixel_size_nm=pixel_size_nm, simulator=simulator)
    mrc = check_mrc(binary, min_width_nm=40.0, min_spacing_nm=40.0, pixel_size_nm=pixel_size_nm)
    l2 = compute_l2_error(binary, design)
    shots = estimate_shot_count(binary, writer_type="mbmw", pixel_size_nm=pixel_size_nm)
    total_px = int(binary.numel())
    metrics = {
        "epe_mean_nm": float(epe["epe_mean_nm"]),
        "epe_max_nm": float(epe["epe_max_nm"]),
        "wafer_epe_mean_nm": float(wafer["epe_mean_nm"]),
        "wafer_epe_max_nm": float(wafer["epe_max_nm"]),
        "l2_error_pixels": float(l2["l2_error_pixels"]),
        "pvband_mean_nm": float(pvb["pvband_mean_nm"]),
        "pvband_max_nm": float(pvb["pvband_max_nm"]),
        "mrc_violation_rate": float(mrc.violation_count) / max(total_px, 1),
        "mrc_passed": bool(mrc.passed),
        "shot_count": int(shots["shot_count"]),
        "valid": bool(epe.get("valid", True)) and bool(wafer.get("valid", True)),
    }
    # Strict-JSON contract: a metric that cannot be computed (e.g. wafer EPE
    # of an empty mask) becomes null + an explicit reason, never Infinity.
    import math as _math

    non_finite = [k for k, v in metrics.items() if isinstance(v, float) and not _math.isfinite(v)]
    for k in non_finite:
        metrics[k] = None
    metrics["non_finite_metrics"] = non_finite
    if non_finite:
        metrics["non_finite_reason"] = "degenerate output (empty/degenerate mask): metric undefined"
    return metrics


def job_quality_tile(args: argparse.Namespace) -> dict[str, Any]:
    """All models x metrics on one real tile, identical Hopkins config."""
    # Registration imports: the registry only knows a model after its module
    # is imported (same pattern as scripts/generate_baselines.py).
    import openlithohub.models.examples.dummy_model  # noqa: F401
    import openlithohub.models.levelset_ilt  # noqa: F401
    import openlithohub.models.rule_based_opc  # noqa: F401
    import openlithohub.models.surrogate_ilt  # noqa: F401
    from openlithohub._utils.hopkins import HopkinsParams
    from openlithohub.models.levelset_ilt import LevelSetILTModel
    from openlithohub.models.registry import registry
    from openlithohub.models.rule_based_opc import RuleBasedOPCModel
    from openlithohub.models.surrogate_ilt import SurrogateILTModel
    from openlithohub.simulators import HopkinsSimulator, SimulatorConfig

    torch.manual_seed(SEED)
    source = load_source(Path(args.gds), args.top_cell)
    x0, y0 = args.tile_x, args.tile_y
    side = args.tile_size
    design = (source.read_window(BoundingBox(x0, y0, x0 + side, y0 + side)) > 0.5).float()

    hopkins_params = HopkinsParams(
        wavelength_nm=193.0,
        na=1.35,
        sigma=0.7,
        num_kernels=24,
        pixel_size_nm=1.0,
        defocus_nm=0.0,
    )
    sim_cfg = SimulatorConfig(
        pixel_size_nm=1.0, wavelength_nm=193.0, na=1.35, sigma=0.7, threshold=0.225
    )
    simulator = HopkinsSimulator(sim_cfg)

    def make_model(name: str) -> Any:
        if name == "dummy-identity":
            return registry.get("dummy-identity")
        if name == "rule-based-opc":
            return RuleBasedOPCModel()
        if name == "levelset-ilt":
            return LevelSetILTModel(
                iterations=args.iters,
                forward_model="hopkins",
                hopkins_params=hopkins_params,
                pixel_size_nm=1.0,
            )
        if name == "surrogate-ilt":
            return SurrogateILTModel(
                iterations=args.iters,
                forward_model="hopkins",
                hopkins_params=hopkins_params,
                correction_interval=10,
                pixel_size_nm=1.0,
                # Reduced on-the-fly training budget: the library defaults
                # (256 samples x 20 epochs of TRUE forward passes) are
                # intractable at 1024px on CPU. Recorded in the artifact
                # policy so the runtime comparison is scoped honestly.
                surrogate_train_samples=args.surrogate_train_samples,
                surrogate_epochs=args.surrogate_epochs,
            )
        raise ValueError(f"unknown model {name!r}")

    # forward-only wall for the shared physics model (spec §4 forward_wall_s);
    # one untimed warmup so SOCS kernel compilation does not pollute p90.
    from openlithohub._utils.hopkins import simulate_aerial_image_hopkins

    simulate_aerial_image_hopkins(design, hopkins_params)
    fwd_walls = []
    for _ in range(5):
        t0 = time.perf_counter()
        simulate_aerial_image_hopkins(design, hopkins_params)
        fwd_walls.append(time.perf_counter() - t0)

    rows = []
    for name in args.models.split(","):
        # One instance per method, untimed warmup predict primes the
        # per-instance SOCS kernel cache so compilation stays out of the
        # timed window (same protocol as job_quality_iccad16).
        model = make_model(name)
        # Warmup predict primes the per-instance SOCS kernel cache. No
        # no_grad here: gradient-based ILT models run their optimization
        # loop inside predict() and need autograd.
        _ = model.predict(design)
        reps = []
        quality_by_rep = []
        for rep in range(args.reps):
            torch.manual_seed(SEED + rep)
            t0 = time.perf_counter()
            prediction = model.predict(design)
            wall = time.perf_counter() - t0
            metrics = _evaluate_mask(
                prediction.mask, design, simulator=simulator, pixel_size_nm=1.0
            )
            fg = float((prediction.mask > 0.5).float().mean())
            metrics["mask_foreground_fraction"] = fg
            metrics["degenerate_blank"] = bool(fg < 0.01)
            quality_by_rep.append(metrics)
            reps.append(wall)
        base = quality_by_rep[0]
        deterministic = all(q == base for q in quality_by_rep[1:])
        rows.append(
            {
                "model": name,
                "optimization_wall_s": summarize(reps),
                "metrics": base,
                "deterministic_across_reps": deterministic,
                "iterations": args.iters,
            }
        )

    return {
        "tile_px": [args.tile_x, args.tile_y, side],
        "occupancy": float(design.mean()),
        "forward_wall_s": summarize(fwd_walls),
        "rows": rows,
    }


def job_quality_iccad16(args: argparse.Namespace) -> dict[str, Any]:
    """Quality comparison on ICCAD16 testcase1 (N7 EUV, 4 nm/px).

    Second quality dataset: a print-critical N7 EUV layout where optical
    effects actually discriminate methods (the sky130hd tiles are
    comfortably above the k1 limit, so all models print near-identically
    there — measured and reported as such).  Same models, same metric
    implementations, same optical config for every method.
    """
    import openlithohub.models.examples.dummy_model  # noqa: F401
    import openlithohub.models.levelset_ilt  # noqa: F401
    import openlithohub.models.rule_based_opc  # noqa: F401
    import openlithohub.models.surrogate_ilt  # noqa: F401
    from openlithohub._utils.hopkins import HopkinsParams
    from openlithohub.data.iccad16 import Iccad16Dataset
    from openlithohub.models.levelset_ilt import LevelSetILTModel
    from openlithohub.models.registry import registry
    from openlithohub.models.rule_based_opc import RuleBasedOPCModel
    from openlithohub.models.surrogate_ilt import SurrogateILTModel
    from openlithohub.simulators import HopkinsSimulator, SimulatorConfig

    torch.manual_seed(SEED)
    pixel_nm = 4.0
    ds = Iccad16Dataset(root=Path(args.iccad16_dir), cases=[1], pixel_nm=pixel_nm)
    sample = ds[0]
    design = (sample.design > 0.5).float()
    # Center-crop to a square clip, then zero-pad symmetrically to square.
    # Every model and every metric sees the same input, so comparisons
    # stay apples-to-apples; the crop is recorded in the artifact.
    h, w = design.shape
    side = min(args.iccad16_crop_px, h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    design = design[y0 : y0 + side, x0 : x0 + side]
    h, w = design.shape
    pad_h, pad_w = side - h, side - w
    if pad_h or pad_w:
        design = torch.nn.functional.pad(
            design, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        )
    padding = {
        "original_px": [int(sample.design.shape[0]), int(sample.design.shape[1])],
        "crop_px": [side, side],
        "crop_offset_px": [y0, x0],
        "padded_px": [int(design.shape[0]), int(design.shape[1])],
        "mode": "center crop + symmetric zero pad (same input for all models)",
    }

    # N7 EUV-class optics (canonical repo eval config, scripts/_eval_v04):
    # identical for model forward and evaluation in this benchmark.
    hopkins_params = HopkinsParams(
        wavelength_nm=13.5,
        na=0.33,
        sigma=0.7,
        num_kernels=24,
        pixel_size_nm=pixel_nm,
        defocus_nm=0.0,
    )
    simulator = HopkinsSimulator(
        SimulatorConfig(
            wavelength_nm=13.5,
            na=0.33,
            pixel_size_nm=pixel_nm,
            threshold=0.5,
        )
    )

    def make_model(name: str) -> Any:
        if name == "dummy-identity":
            return registry.get("dummy-identity")
        if name == "rule-based-opc":
            return RuleBasedOPCModel()
        if name == "levelset-ilt":
            return LevelSetILTModel(
                iterations=args.iters,
                forward_model="hopkins",
                hopkins_params=hopkins_params,
                pixel_size_nm=pixel_nm,
            )
        if name == "surrogate-ilt":
            return SurrogateILTModel(
                iterations=args.iters,
                forward_model="hopkins",
                hopkins_params=hopkins_params,
                correction_interval=10,
                pixel_size_nm=pixel_nm,
                surrogate_train_samples=args.surrogate_train_samples,
                surrogate_epochs=args.surrogate_epochs,
            )
        raise ValueError(f"unknown model {name!r}")

    rows = []
    for name in args.models.split(","):
        # One model instance per method, reused across timing reps: the
        # SOCS kernel cache is per-instance, and EUV kernel compilation
        # costs minutes — an untimed warmup predict primes it so kernel
        # compilation never lands in the timed window.
        model = make_model(name)
        # Warmup predict primes the per-instance SOCS kernel cache. No
        # no_grad here: gradient-based ILT models run their optimization
        # loop inside predict() and need autograd.
        _ = model.predict(design)
        reps = []
        quality_by_rep = []
        for rep in range(args.reps):
            torch.manual_seed(SEED + rep)
            t0 = time.perf_counter()
            prediction = model.predict(design)
            wall = time.perf_counter() - t0
            metrics = _evaluate_mask(
                prediction.mask,
                design,
                simulator=simulator,
                pixel_size_nm=pixel_nm,
            )
            # Degenerate-output guard: an (near-)blank mask must be
            # labeled as such in the artifact, never scored as a win.
            fg = float((prediction.mask > 0.5).float().mean())
            metrics["mask_foreground_fraction"] = fg
            metrics["degenerate_blank"] = bool(fg < 0.01)
            quality_by_rep.append(metrics)
            reps.append(wall)
        base = quality_by_rep[0]
        rows.append(
            {
                "model": name,
                "optimization_wall_s": summarize(reps),
                "metrics": base,
                "deterministic_across_reps": all(q == base for q in quality_by_rep[1:]),
                "iterations": args.iters,
            }
        )

    return {
        "tile_px": [int(design.shape[0]), int(design.shape[1])],
        "occupancy": float(design.mean()),
        "padding": padding,
        "rows": rows,
    }


def job_fulldie_tile(args: argparse.Namespace) -> dict[str, Any]:
    """Certified empty-context screening of one die tile — no forward model.

    Measures the theorem-safe active-set discovery cost on real die
    geometry (B04 INC29 ``screen_only`` semantics): forward_simulator_calls
    stays 0 and no window is ever materialized.
    """
    torch.manual_seed(SEED)
    from openlithohub.streaming.core_halo import plan_tile_requests

    t0 = time.perf_counter()
    source = load_source(Path(args.gds), args.top_cell)
    parse_s = time.perf_counter() - t0
    h, w = source.shape
    requests = list(plan_tile_requests(source.shape, args.core, FORWARD_RADIUS))
    policy = selective_policy()
    active = 0
    screened = 0
    t0 = time.perf_counter()
    for req in requests:
        decision = policy.screen(source, req)
        if decision.status == "SCREENED_OUT":
            screened += req.core_bbox.area
        else:
            active += req.core_bbox.area
    total = h * w
    return {
        "tile": [args.tile_ix, args.tile_iy],
        "size_px": [h, w],
        "parse_wall_s": parse_s,
        "screen_query_wall_s": time.perf_counter() - t0,
        "n_tiles": len(requests),
        "screen_queries": len(requests),
        "core_px": args.core,
        "active_pixels": active,
        "screened_pixels": screened,
        "full_chip_pixels": total,
        "screened_fraction": screened / max(total, 1),
        "accounted_pct": 100.0 * (active + screened) / max(total, 1),
        "forward_simulator_calls": 0,
        "peak_rss_bytes": peak_rss_bytes(),
    }


# ---------------------------------------------------------------------------
# worker dispatch
# ---------------------------------------------------------------------------


def run_worker(script: Path, job_args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, str(script), "--worker", "--parent-pid", str(os.getpid()), *job_args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed {job_args}: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


_PROGRESS_INSTANCES: dict[str, Progress] = {}


def get_progress(path: Path) -> Progress:
    """One Progress instance per path, so stages accumulate in one file."""
    key = str(path.resolve())
    if key not in _PROGRESS_INSTANCES:
        _PROGRESS_INSTANCES[key] = Progress(path)
    return _PROGRESS_INSTANCES[key]


class Progress:
    """Durable progress file: stage, done/total, in-flight row, ETA.

    Written atomically after every checkpoint append so "where is the run,
    how much longer" is answerable from a single JSON at any moment —
    including after an interrupt.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stages: dict[str, dict[str, Any]] = {}

    def stage(self, name: str, total: int) -> None:
        self.stages[name] = {
            "total": total,
            "done": 0,
            "skipped_on_resume": 0,
            "current": None,
            "last_row_wall_s": None,
            "eta_s": None,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.flush(name, None)

    def begin_row(self, stage: str, key: str) -> None:
        self.stages[stage]["current"] = key
        self.flush(stage, None)

    def end_row(self, stage: str, key: str, wall_s: float, skipped: bool = False) -> None:
        s = self.stages[stage]
        s["current"] = None
        s["last_row_wall_s"] = round(wall_s, 3)
        if skipped:
            s["skipped_on_resume"] += 1
        else:
            s["done"] += 1
        remaining = s["total"] - s["done"] - s["skipped_on_resume"]
        s["eta_s"] = round(max(0, remaining) * (s["last_row_wall_s"] or 0.0), 1)
        self.flush(stage, wall_s)

    def flush(self, stage: str, wall_s: float | None) -> None:
        s = self.stages[stage]
        remaining = s["total"] - s["done"] - s["skipped_on_resume"]
        if s["eta_s"] is None and s["last_row_wall_s"]:
            s["eta_s"] = round(max(0, remaining) * s["last_row_wall_s"], 1)
        payload = {
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stages": self.stages,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        if wall_s is not None:
            log(
                f"progress [{stage}] done={s['done']}/{s['total']} "
                f"(resumed {s['skipped_on_resume']}) eta~{s['eta_s']}s"
            )


class Checkpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))

    def has(self, key: str) -> bool:
        return any(row.get("_key") == key for row in self.rows)

    def get(self, key: str) -> dict[str, Any] | None:
        for row in self.rows:
            if row.get("_key") == key:
                return row
        return None

    def append(self, key: str, row: dict[str, Any]) -> None:
        entry = {"_key": key, **row}
        self.rows.append(entry)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# stage implementations
# ---------------------------------------------------------------------------


def tracked_row(
    ckpt: Checkpoint,
    progress: Progress,
    stage: str,
    key: str,
    produce: callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Resume-aware checkpointed row with progress bookkeeping."""
    progress.begin_row(stage, key)
    t0 = time.perf_counter()
    rec = ckpt.get(key)
    if rec is not None:
        progress.end_row(stage, key, 0.0, skipped=True)
        return rec
    rec = produce()
    wall = time.perf_counter() - t0
    ckpt.append(key, rec)
    progress.end_row(stage, key, wall, skipped=False)
    return rec


def fingerprint(args: argparse.Namespace) -> str:
    payload = json.dumps(
        {
            "gds_sha256": sha256_file(args.gds) if Path(args.gds).exists() else None,
            "repeats": args.repeats,
            "core": args.core,
            "dense_max_bytes": args.dense_max_bytes,
            "ladder": args.sizes,
            "max_selective": args.max_selective_size,
            "iccad16_dir": str(args.iccad16_dir) if args.iccad16_dir else None,
            "ilt_iterations": args.ilt_iterations,
            "quality_reps": args.quality_reps,
            "max_tiles": args.max_tiles,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def stage_prepare(args: argparse.Namespace, out: Path, script: Path) -> dict[str, Any]:
    ckpt = Checkpoint(out / "checkpoints" / f"prepare_{fingerprint(args)}.jsonl")
    progress = get_progress(out / "progress.json")
    fixture_dir = out / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    crop_sizes = sorted(
        {int(s) for s in args.sizes.split(",")}
        | ({args.max_selective_size} if args.max_selective_size > 0 else set())
    )
    progress.stage("prepare", 2 + len(crop_sizes) + 1)  # witnesses + crops + die grid
    log(f"prepare: {2 + len(crop_sizes) + 1} checkpointed steps")

    def witness_synthetic() -> dict[str, Any]:
        from openlithohub.streaming.vector_runs import (
            ExactVectorRunSource,
            PolygonWithHoles,
            VectorCell,
        )

        size = 512
        side = 48
        polys = []
        for i, (fx, fy) in enumerate(((0.12, 0.14), (0.52, 0.24), (0.76, 0.62))):
            px = int(fx * size)
            py = int(fy * size)
            polys.append(
                {
                    "object_id": f"island-{i}",
                    "outer": (
                        (px, py),
                        (px + side, py),
                        (px + side, py + side),
                        (px, py + side),
                    ),
                }
            )
        cell = VectorCell(
            name="TOP",
            polygons=tuple(
                PolygonWithHoles(object_id=p["object_id"], outer=p["outer"]) for p in polys
            ),
            instances=(),
        )
        source = ExactVectorRunSource(
            shape=(size, size), cells={"TOP": cell}, top="TOP", pixel_size_nm=8.0
        )
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
        return {
            "size": size,
            "max_abs_error": float((sink.finalize() - reference).abs().max().item()),
        }

    key = "witness_synthetic"
    if not ckpt.has(key):
        progress.begin_row("prepare", key)
        t0 = time.perf_counter()
        row = witness_synthetic()
        ckpt.append(key, row)
        progress.end_row("prepare", key, time.perf_counter() - t0)
    witness_synth = ckpt.get(key)
    if witness_synth is None:
        raise RuntimeError("prepare checkpoint lost synthetic witness row")
    if witness_synth["max_abs_error"] > 2e-6:
        raise RuntimeError(f"synthetic witness failed: {witness_synth}")

    # real-layout equivalence witness on an occupied window of the 4096 crop
    key = "witness_real"
    if not ckpt.has(key):
        progress.begin_row("prepare", key)
        t0 = time.perf_counter()
        crop4096 = out / "fixtures" / "ibex_crop_4096.gds"
        if not crop4096.exists():
            info = clip_gds(Path(args.gds), crop4096, top_name="IBEX_CROP_4096", size_dbu=4096)
            log(f"prepared {crop4096.name}: sha256={info['sha256'][:16]}")
        row = run_worker(
            script,
            [
                "--job",
                "real_witness",
                "--gds",
                str(crop4096),
                "--top-cell",
                "IBEX_CROP_4096",
                "--window",
                "2048",
            ],
        )
        ckpt.append(key, row)
        progress.end_row("prepare", key, time.perf_counter() - t0)
    witness_real = ckpt.get(key)
    if witness_real is None:
        raise RuntimeError("prepare checkpoint lost real-layout witness row")
    if witness_real["max_abs_error"] > 2e-6:
        raise RuntimeError(f"real-layout witness failed: {witness_real}")

    # fixtures: center crops (ladder sizes + the selective-only maximum)
    crops: dict[str, Any] = {}
    for size in crop_sizes:
        key = f"crop_{size}"
        path = fixture_dir / f"ibex_crop_{size}.gds"
        if not ckpt.has(key):
            progress.begin_row("prepare", key)
            t0 = time.perf_counter()
            info = clip_gds(Path(args.gds), path, top_name=f"IBEX_CROP_{size}", size_dbu=size)
            ckpt.append(key, info)
            progress.end_row("prepare", key, time.perf_counter() - t0)
        rec = ckpt.get(key)
        if rec is None:
            raise RuntimeError(f"prepare checkpoint lost crop fixture row for {size}")
        crops[str(size)] = rec

    # die sample tiles (center + 4 corners of the grid)
    key = "die_grid"
    if not ckpt.has(key):
        progress.begin_row("prepare", key)
        t0 = time.perf_counter()
        die_dir = fixture_dir / "die_tiles"
        die_dir.mkdir(exist_ok=True)
        info = make_die_sample_tiles(
            Path(args.gds),
            die_dir,
            tile_px=args.die_tile_px,
            tiles_per_side=args.die_tiles_per_side,
        )
        ckpt.append(key, info)
        progress.end_row("prepare", key, time.perf_counter() - t0)
    die_grid_rec = ckpt.get(key)
    if die_grid_rec is None:
        raise RuntimeError("prepare checkpoint lost die-grid row")

    return {
        "witness_synthetic": witness_synth,
        "witness_real": witness_real,
        "crops": crops,
        "die_grid": die_grid_rec,
    }


def stage_runtime(
    args: argparse.Namespace, out: Path, script: Path, prepared: dict[str, Any]
) -> dict[str, Any]:
    ckpt = Checkpoint(out / "checkpoints" / f"runtime_{fingerprint(args)}.jsonl")
    progress = get_progress(out / "progress.json")
    sizes = [int(s) for s in args.sizes.split(",")]
    if args.max_selective_size > 0:
        sizes.append(args.max_selective_size)
    sizes = sorted(set(sizes))

    # planned worker rows (policy-blocked dense rows cost nothing and are
    # not tracked as progress rows)
    total_rows = 0
    for size in sizes:
        if prepared["crops"].get(str(size)) is None:
            continue
        n_modes = 0
        if dense_allowed(size, budget_bytes=args.dense_max_bytes):
            n_modes += 2
        if size <= args.max_vector_size:
            n_modes += 1
        n_modes += 1  # selective
        reps_n = args.repeats if size <= 16384 else min(args.repeats, args.large_repeats)
        total_rows += n_modes * reps_n
    progress.stage("runtime", total_rows)
    log(f"runtime: {total_rows} checkpointed worker rows")

    rows: list[dict[str, Any]] = []
    for size in sizes:
        crop = prepared["crops"].get(str(size))
        if crop is None:
            log(f"SKIP size {size}: no crop fixture")
            continue
        gds, top = crop["gds"], crop["top_cell"]
        modes = []
        if dense_allowed(size, budget_bytes=args.dense_max_bytes):
            modes += ["dense_full", "tiled_raster"]
        else:
            rows.append(
                {
                    "mode": "dense_full",
                    "size_px": [size, size],
                    "status": STATUS_NOT_RUN_MEMORY_POLICY,
                    "dense_input_bytes_structural": size * size * 4,
                    "note": (
                        "dense input exceeds --dense-max-bytes policy budget; a "
                        "policy decision, not a structural impossibility"
                    ),
                }
            )
        modes += ["b04_vector"] if size <= args.max_vector_size else []
        modes += ["b04_selective"]
        # 5 repeats on the claim-bearing sizes; the largest scaling-only
        # rows cost minutes of parse per rep, so they record fewer and
        # carry their own "repeats" field.  Headline claims are only
        # derived from full-repeat rows (see generate_industrial_claims).
        repeats = args.repeats if size <= 16384 else min(args.repeats, args.large_repeats)
        for mode in modes:
            reps = []
            for rep in range(repeats):
                key = f"{mode}_{size}_{rep}"
                rec = ckpt.get(key)
                if rec is not None:
                    progress.end_row("runtime", key, 0.0, skipped=True)
                    reps.append(rec)
                    continue
                progress.begin_row("runtime", key)
                t0 = time.perf_counter()
                log(f"runtime {mode} {size}px rep {rep + 1}/{repeats}")
                rec = run_worker(
                    script,
                    [
                        "--job",
                        "runtime_row",
                        "--mode",
                        mode,
                        "--gds",
                        str(gds),
                        "--top-cell",
                        str(top),
                        "--core",
                        str(args.core),
                    ],
                )
                ckpt.append(key, rec)
                progress.end_row("runtime", key, time.perf_counter() - t0)
                reps.append(rec)
            merged = dict(reps[0])
            for fieldname in ("parse_wall_s", "execution_wall_s", "end_to_end_wall_s"):
                merged[fieldname] = summarize([float(r[fieldname]) for r in reps])
            merged["peak_rss_bytes"] = summarize([float(r["peak_rss_bytes"]) for r in reps])
            merged["repeats"] = repeats
            rows.append(merged)

    comparisons: dict[str, Any] = {}
    by_size: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") not in (None, STATUS_SUCCESS):
            continue
        by_size.setdefault(int(row["size_px"][0]), {})[str(row["mode"])] = row
    for size, per_mode in sorted(by_size.items()):
        entry: dict[str, Any] = {}
        if "dense_full" in per_mode and "b04_selective" in per_mode:
            entry["dense_vs_selective"] = _runtime_compare(
                per_mode["dense_full"], per_mode["b04_selective"]
            )
        if "tiled_raster" in per_mode and "b04_selective" in per_mode:
            entry["tiled_vs_selective"] = _runtime_compare(
                per_mode["tiled_raster"], per_mode["b04_selective"]
            )
        comparisons[str(size)] = entry

    memory: dict[str, Any] = {"per_size": {}}
    for size, per_mode in sorted(by_size.items()):
        size_entry: dict[str, Any] = {}
        dense_rss = per_mode.get("dense_full", {}).get("peak_rss_bytes")
        for mode in ("dense_full", "tiled_raster", "b04_vector", "b04_selective"):
            row = per_mode.get(mode)
            if row is not None:
                size_entry[mode] = {
                    "peak_rss_bytes_median": float(row["peak_rss_bytes"]["median"]),
                    "peak_rss_bytes_p10": float(row["peak_rss_bytes"]["p10"]),
                    "peak_rss_bytes_p90": float(row["peak_rss_bytes"]["p90"]),
                }
        if dense_rss is not None and "b04_selective" in per_mode:
            size_entry["streaming_memory_reduction_pct"] = memory_reduction_pct(
                float(dense_rss["median"]),
                float(per_mode["b04_selective"]["peak_rss_bytes"]["median"]),
            )
        memory["per_size"][str(size)] = size_entry

    streamed_ok = [
        int(row["size_px"][0])
        for row in rows
        if row["mode"] == "b04_selective" and row.get("status") != STATUS_NOT_RUN_MEMORY_POLICY
    ]
    dense_policy_blocked = [
        row for row in rows if row.get("status") == STATUS_NOT_RUN_MEMORY_POLICY
    ]
    memory["max_streamed_size_px"] = max(streamed_ok) if streamed_ok else None
    memory["dense_not_run_under_memory_policy_px"] = [
        row["size_px"][0] for row in dense_policy_blocked
    ]

    return {
        "rows": rows,
        "comparisons": comparisons,
        "memory": memory,
        "policy": {
            "dense_max_bytes": args.dense_max_bytes,
            "max_vector_size_px": args.max_vector_size,
            "max_selective_size_px": args.max_selective_size,
            "core_px": args.core,
            "repeats": args.repeats,
        },
    }


def _runtime_compare(baseline_row: dict[str, Any], candidate_row: dict[str, Any]) -> dict[str, Any]:
    from openlithohub.benchmark.industrial import speedup

    base = float(baseline_row["execution_wall_s"]["median"])
    cand = float(candidate_row["execution_wall_s"]["median"])
    return {
        "baseline_median_s": base,
        "candidate_median_s": cand,
        "speedup": speedup(base, cand),
        "metric": "execution_wall_s",
    }


def _safe_summarize(values: list[float]) -> dict[str, Any]:
    """summarize() that tolerates "no finite data" (strict-JSON contract)."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {"n": 0, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return summarize(finite)


def _aggregate_quality(block_rows: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    """Aggregate per-tile quality rows into per-model summaries + comparisons."""
    metric_keys = [
        "epe_mean_nm",
        "wafer_epe_mean_nm",
        "pvband_mean_nm",
        "pvband_max_nm",
        "mrc_violation_rate",
        "shot_count",
        "mask_foreground_fraction",
    ]
    aggregate: dict[str, Any] = {}
    for model in models:
        per_tile = [
            row["metrics"] for rec in block_rows for row in rec["rows"] if row["model"] == model
        ]
        agg: dict[str, Any] = {
            k: _safe_summarize([m.get(k) for m in per_tile if m.get(k) is not None])
            for k in metric_keys
        }
        # Degenerate-output firewall: a model that produced a (near-)blank
        # mask on any tile must never headline a quality comparison — a
        # blank mask trivially "passes" MRC and prints nothing.
        agg["degenerate_blank_any_rep"] = any(
            bool(row["metrics"].get("degenerate_blank", False))
            for rec in block_rows
            for row in rec["rows"]
            if row["model"] == model
        )
        agg["optimization_wall_s"] = summarize(
            [
                float(row["optimization_wall_s"]["median"])
                for rec in block_rows
                for row in rec["rows"]
                if row["model"] == model
            ]
        )
        aggregate[model] = agg

    def agg_reduction(metric: str, baseline_model: str, candidate_model: str) -> dict[str, Any]:
        base = aggregate[baseline_model][metric]["median"]
        cand = aggregate[candidate_model][metric]["median"]
        entry: dict[str, Any] = {
            "metric": metric,
            "baseline": baseline_model,
            "candidate": candidate_model,
            "baseline_median": base,
            "candidate_median": cand,
            "reduction_pct": None,
        }
        if base is None or cand is None:
            entry["status"] = "NOT_MEASURABLE"
            return entry
        entry["reduction_pct"] = relative_reduction_pct(float(base), float(cand))
        return entry

    surrogate_speed: dict[str, Any] | None = None
    if "levelset-ilt" in aggregate and "surrogate-ilt" in aggregate:
        lv = float(aggregate["levelset-ilt"]["optimization_wall_s"]["median"])
        sg = float(aggregate["surrogate-ilt"]["optimization_wall_s"]["median"])
        surrogate_speed = {
            "baseline": "levelset-ilt",
            "candidate": "surrogate-ilt",
            "baseline_median_s": lv,
            "candidate_median_s": sg,
        }
    comparisons = {
        "identity_vs_levelset": {
            "pvband_mean_nm": agg_reduction("pvband_mean_nm", "dummy-identity", "levelset-ilt"),
            "mrc_violation_rate": agg_reduction(
                "mrc_violation_rate", "dummy-identity", "levelset-ilt"
            ),
            "wafer_epe_mean_nm": agg_reduction(
                "wafer_epe_mean_nm", "dummy-identity", "levelset-ilt"
            ),
        },
        "levelset_vs_surrogate_runtime": surrogate_speed,
    }
    return {"aggregate": aggregate, "comparisons": comparisons}


def stage_quality(
    args: argparse.Namespace, out: Path, script: Path, prepared: dict[str, Any]
) -> dict[str, Any]:
    ckpt = Checkpoint(out / "checkpoints" / f"quality_{fingerprint(args)}.jsonl")
    progress = get_progress(out / "progress.json")
    crop = prepared["crops"]["4096"]
    models = ["dummy-identity", "rule-based-opc", "levelset-ilt", "surrogate-ilt"]

    # deterministic tile selection: rank 1024px tiles of the 4096 crop by
    # occupancy (measured once, in-driver, untimed), keep the top K.
    key = "tile_selection"
    if not ckpt.has(key):
        source = load_source(Path(crop["gds"]), crop["top_cell"])
        occ = []
        side = args.tile_size
        for ty in range(0, 4096, side):
            for tx in range(0, 4096, side):
                w = source.read_window(BoundingBox(tx, ty, tx + side, ty + side))
                o = float((w > 0.5).float().mean())
                if o >= args.min_tile_occupancy:
                    occ.append((o, tx, ty))
        occ.sort(key=lambda t: (-t[0], t[1], t[2]))
        tiles = [{"x": tx, "y": ty, "occupancy": o} for o, tx, ty in occ[: args.max_tiles]]
        ckpt.append(key, {"tiles": tiles})
    sel = ckpt.get(key)
    if sel is None:
        raise RuntimeError("quality checkpoint lost tile-selection row")

    quality_total = 1 + len(sel["tiles"]) + (1 if args.iccad16_dir else 0)
    progress.stage("quality", quality_total)
    log(f"quality: {quality_total} checkpointed rows")
    tile_rows = []
    for tile in sel["tiles"]:
        key = f"tile_{tile['x']}_{tile['y']}"
        rec = ckpt.get(key)
        if rec is not None:
            progress.end_row("quality", key, 0.0, skipped=True)
            tile_rows.append(rec)
            continue
        progress.begin_row("quality", key)
        t0 = time.perf_counter()
        log(f"quality tile ({tile['x']},{tile['y']}) occ={tile['occupancy']:.3f}")
        rec = run_worker(
            script,
            [
                "--job",
                "quality_tile",
                "--gds",
                str(crop["gds"]),
                "--top-cell",
                str(crop["top_cell"]),
                "--tile-x",
                str(tile["x"]),
                "--tile-y",
                str(tile["y"]),
                "--tile-size",
                str(args.tile_size),
                "--iters",
                str(args.ilt_iterations),
                "--reps",
                str(args.quality_reps),
                "--surrogate-train-samples",
                str(args.surrogate_train_samples),
                "--surrogate-epochs",
                str(args.surrogate_epochs),
                "--models",
                ",".join(models),
            ],
        )
        ckpt.append(key, rec)
        progress.end_row("quality", key, time.perf_counter() - t0)
        tile_rows.append(rec)

    datasets: dict[str, Any] = {
        "sky130hd-ibex-tiles": {
            "description": "real routed sky130hd ibex tiles (1024px @ 1nm/px), 193nm NA1.35 optics",
            "policy": {
                "tile_size_px": args.tile_size,
                "min_tile_occupancy": args.min_tile_occupancy,
                "max_tiles": args.max_tiles,
                "n_tiles": len(tile_rows),
                "ilt_iterations": args.ilt_iterations,
                "quality_reps": args.quality_reps,
                "surrogate_train_samples": args.surrogate_train_samples,
                "surrogate_epochs": args.surrogate_epochs,
                "optical_config": PHYSICS_SCOPE["quality_forward_model"],
            },
            **_aggregate_quality(tile_rows, models),
        }
    }
    all_rows = {"sky130hd-ibex-tiles": tile_rows}

    if args.iccad16_dir:
        key = "iccad16_testcase1"
        rec = ckpt.get(key)
        if rec is not None:
            progress.end_row("quality", key, 0.0, skipped=True)
        else:
            progress.begin_row("quality", key)
            t0 = time.perf_counter()
            log("quality iccad16 testcase1 (4nm/px, 13.5nm EUV-class optics)")
            rec = run_worker(
                script,
                [
                    "--job",
                    "quality_iccad16",
                    "--iccad16-dir",
                    str(args.iccad16_dir),
                    "--iccad16-crop-px",
                    str(args.iccad16_crop_px),
                    "--iters",
                    str(args.ilt_iterations_iccad16),
                    "--reps",
                    str(args.quality_reps),
                    "--surrogate-train-samples",
                    str(args.surrogate_train_samples),
                    "--surrogate-epochs",
                    str(args.surrogate_epochs),
                    "--models",
                    ",".join(models),
                ],
            )
            ckpt.append(key, rec)
            progress.end_row("quality", key, time.perf_counter() - t0)
        datasets["iccad16-testcase1"] = {
            "description": (
                "ICCAD16 Problem C testcase1 (N7 EUV, 4nm/px, 13.5nm NA0.33 optics); "
                "design-as-target, no reference OPC mask ships with the dataset"
            ),
            "policy": {
                "tile_size_px": [rec["tile_px"][0], rec["tile_px"][1]],
                "n_tiles": 1,
                "ilt_iterations": args.ilt_iterations_iccad16,
                "quality_reps": args.quality_reps,
                "surrogate_train_samples": args.surrogate_train_samples,
                "surrogate_epochs": args.surrogate_epochs,
                "iccad16_crop_px": args.iccad16_crop_px,
                "ilt_iterations_iccad16": args.ilt_iterations_iccad16,
                "optical_config": (
                    "Hopkins SOCS, 13.5nm, NA 0.33, sigma 0.7, 24 kernels, threshold 0.5"
                ),
            },
            **_aggregate_quality([rec], models),
        }
        all_rows["iccad16-testcase1"] = [rec]

    return {
        "datasets": datasets,
        "tile_rows": all_rows,
        "tile_selection": sel,
    }


def stage_fulldie(
    args: argparse.Namespace, out: Path, script: Path, prepared: dict[str, Any]
) -> dict[str, Any]:
    """Die-scale characteristics: structural infeasibility + sampled survey.

    The full-die dense raster equivalent (die_px^2 x 4 bytes) is compared
    against physical RAM — a structural statement, not a measurement.  A
    measured screen-only survey runs on a deterministic sample of die
    tiles; the full-die survey cost is then an explicit ESTIMATE
    (mean per-tile cost x grid tile count), never presented as measured.
    """
    ckpt = Checkpoint(out / "checkpoints" / f"fulldie_{fingerprint(args)}.jsonl")
    progress = get_progress(out / "progress.json")
    grid = prepared["die_grid"]
    progress.stage("fulldie", len(grid["tiles"]))
    log(f"fulldie: {len(grid['tiles'])} checkpointed screen-only tiles")
    rows = []
    for entry in grid["tiles"]:
        ix, iy = entry["tile"]
        key = f"tile_{ix}_{iy}"
        rec = ckpt.get(key)
        if rec is not None:
            progress.end_row("fulldie", key, 0.0, skipped=True)
            rows.append(rec)
            continue
        progress.begin_row("fulldie", key)
        t0 = time.perf_counter()
        log(f"fulldie screen-only tile ({ix},{iy})")
        rec = run_worker(
            script,
            [
                "--job",
                "fulldie_tile",
                "--gds",
                entry["gds"],
                "--top-cell",
                entry["top_cell"],
                "--core",
                str(args.die_core_px),
                "--tile-ix",
                str(ix),
                "--tile-iy",
                str(iy),
            ],
        )
        ckpt.append(key, rec)
        progress.end_row("fulldie", key, time.perf_counter() - t0)
        rows.append(rec)

    die_px = grid["die_size_px"]
    dense_die_bytes = die_px[0] * die_px[1] * 4
    ram = environment_snapshot()["hardware"]["physical_ram_bytes"] or 0
    mean_screen_s = float(sum(float(r["screen_query_wall_s"]) for r in rows) / len(rows))
    total_grid_tiles = grid["n_grid_tiles"]
    return {
        "die_size_px": die_px,
        "tile_px": grid["tile_px"],
        "grid_tiles_per_side": grid["tiles_per_side"],
        "n_grid_tiles": total_grid_tiles,
        "n_sample_tiles": len(rows),
        "sample_rows": rows,
        "sample_screen_query_wall_s_mean": mean_screen_s,
        "sampled_screened_fraction_range": [
            min(float(r["screened_fraction"]) for r in rows),
            max(float(r["screened_fraction"]) for r in rows),
        ],
        "full_die_survey_wall_s_estimate": {
            "value": mean_screen_s * total_grid_tiles,
            "method": "mean sample per-tile screen cost x n_grid_tiles",
            "status": "ESTIMATE_NOT_MEASUREMENT",
        },
        "dense_die_raster_bytes_structural": dense_die_bytes,
        "dense_die_status": (
            STATUS_INFEASIBLE_ON_REFERENCE_MACHINE
            if dense_die_bytes > ram
            else "FEASIBLE_BUT_NOT_RUN"
        ),
        "physical_ram_bytes": ram,
        "known_scaling_limit": (
            "exact-vector window/screen queries build a per-row polygon index on "
            "first touch; cost grows with tile rows x layout polygons. A spatial "
            "index is future work (see docs/industrial-benchmarks.md)."
        ),
    }


def build_artifact(
    kind: str,
    status: str,
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = environment_snapshot()
    source = measurement_source()
    if source["commit"] is None:
        raise RuntimeError("measurement source commit undeterminable; refusing to measure")
    artifact = {
        "schema": SCHEMA_NAME,
        "kind": kind,
        "status": status,
        "git_commit": source["commit"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": env["hardware"],
        "software": env["software"],
        "dataset": DATASET,
        "fixture": fixture or args.fixture_identity,
        "measurement_source": source,
        "claim_scope": dict(PHYSICS_SCOPE),
        "reproducibility": {
            "command": "python benchmarks/industrial/run_industrial_benchmark.py "
            f"--gds <ibex.gds> --out {args.out} --repeats {args.repeats}",
            "seed": SEED,
        },
    }
    artifact.update(payload)
    return artifact


def write_artifact(out: Path, name: str, artifact: dict[str, Any]) -> Path:
    # Strict-JSON contract (P0.2): non-finite values are sanitized to None
    # before writing, and allow_nan=False makes any leak a hard error.
    artifact = _sanitize(artifact)
    problems = validate_artifact(artifact)
    # Local-iteration escape hatch: a dirty tree can produce *provisional*
    # artifacts, but they still record working_tree_dirty=true and the CI
    # verifier rejects any dirty artifact from the repository.
    if os.environ.get("OPENLITHOHUB_ALLOW_DIRTY_MEASUREMENT") == "1":
        problems = [p for p in problems if "working_tree_dirty" not in p]
        if problems:
            raise RuntimeError(f"artifact {name} failed validation: {problems}")
        log("WARNING: provisional artifact from a DIRTY tree (not CI-admissible)")
    elif problems:
        raise RuntimeError(f"artifact {name} failed validation: {problems}")
    path = out / name
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gds", type=Path, help="parent routed GDS (ibex.gds)")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/industrial"))
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "prepare", "runtime", "quality", "fulldie", "manifest"],
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--core", type=int, default=1024, help="streaming core size (px)")
    ap.add_argument("--sizes", default="4096,8192,16384,32768")
    ap.add_argument("--dense-max-bytes", type=int, default=30 * (1 << 30))
    ap.add_argument("--max-vector-size", type=int, default=32768)
    ap.add_argument("--max-selective-size", type=int, default=65536)
    ap.add_argument("--die-tile-px", type=int, default=32768)
    ap.add_argument("--die-tiles-per-side", type=int, default=17)
    ap.add_argument("--die-core-px", type=int, default=4096)
    ap.add_argument("--tile-size", type=int, default=1024, help="quality tile side (px)")
    ap.add_argument("--min-tile-occupancy", type=float, default=0.02)
    ap.add_argument("--max-tiles", type=int, default=6)
    ap.add_argument("--ilt-iterations", type=int, default=50)
    ap.add_argument("--quality-reps", type=int, default=3)
    ap.add_argument("--surrogate-train-samples", type=int, default=16)
    ap.add_argument("--surrogate-epochs", type=int, default=3)
    ap.add_argument("--iccad16-crop-px", type=int, default=256)
    ap.add_argument(
        "--ilt-iterations-iccad16",
        type=int,
        default=200,
        help="ILT iteration budget for the ICCAD16 dataset (matches the historical table config)",
    )
    ap.add_argument(
        "--iccad16-dir",
        type=Path,
        default=None,
        help="optional: directory with testcase1.oas/test1.csv for the second quality dataset",
    )
    ap.add_argument("--large-repeats", type=int, default=3, help="repeats for sizes > 16384px")

    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--job")
    ap.add_argument("--mode")
    ap.add_argument("--top-cell")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--tile-x", type=int)
    ap.add_argument("--tile-y", type=int)
    ap.add_argument("--tile-ix", type=int)
    ap.add_argument("--tile-iy", type=int)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--models", default="dummy-identity,rule-based-opc,levelset-ilt,surrogate-ilt")
    args = ap.parse_args()

    script = Path(__file__).resolve()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        init_run_log(out)

    if args.worker:
        if args.parent_pid and args.parent_pid > 0:
            import threading

            def _orphan_watchdog() -> None:
                # If the driver dies (Ctrl-C/kill), exit instead of burning
                # CPU as an orphan for the remainder of a long row.
                while True:
                    if os.getppid() != args.parent_pid:
                        os._exit(97)
                    time.sleep(5)

            threading.Thread(target=_orphan_watchdog, daemon=True).start()
    elif args.gds is None or not Path(args.gds).exists():
        raise SystemExit("--gds is required and must exist")
    else:
        # One content-identity record for the parent fixture, shared by all
        # artifacts of this run (P0.3: fixture provenance closure).
        args.fixture_identity = fixture_identity(Path(args.gds))
        log(
            "fixture sha256="
            f"{args.fixture_identity['sha256'][:16]} bytes={args.fixture_identity['bytes']}"
        )

    if args.worker:
        jobs = {
            "runtime_row": job_runtime_row,
            "real_witness": job_real_witness,
            "quality_tile": job_quality_tile,
            "quality_iccad16": job_quality_iccad16,
            "fulldie_tile": job_fulldie_tile,
        }
        if args.job not in jobs:
            raise SystemExit(f"unknown job {args.job!r}")
        print(json.dumps(jobs[args.job](args), sort_keys=True))
        return 0

    stages = (
        ["prepare", "runtime", "quality", "fulldie", "manifest"]
        if args.stage == "all"
        else [args.stage]
    )
    prepared: dict[str, Any] = {}
    if "prepare" in stages:
        prepared = stage_prepare(args, out, script)
    if "runtime" in stages:
        if not prepared:
            prepared = stage_prepare(args, out, script)
        runtime = stage_runtime(args, out, script, prepared)
        write_artifact(
            out,
            "industrial-runtime.json",
            build_artifact("runtime", STATUS_SUCCESS, runtime, args),
        )
    if "quality" in stages:
        if not prepared:
            prepared = stage_prepare(args, out, script)
        quality = stage_quality(args, out, script, prepared)
        write_artifact(
            out,
            "industrial-quality.json",
            build_artifact("quality", STATUS_SUCCESS, quality, args),
        )
    if "fulldie" in stages:
        if not prepared:
            prepared = stage_prepare(args, out, script)
        fulldie = stage_fulldie(args, out, script, prepared)
        write_artifact(
            out,
            "industrial-fulldie.json",
            build_artifact("full_die", STATUS_SUCCESS, fulldie, args),
        )
    if "manifest" in stages:
        files = sorted(out.glob("industrial-*.json"))
        entries = [
            {"file": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files
        ]
        manifest = build_artifact(
            "manifest",
            STATUS_SUCCESS,
            {"artifacts": entries},
            args,
        )
        path = write_artifact(out, "manifest.json", manifest)
        with contextlib.suppress(OSError):
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            (out / "SHA256SUMS.txt").write_text(
                "".join(f"{e['sha256']}  {e['file']}\n" for e in entries)
                + f"{sha}  manifest.json\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
