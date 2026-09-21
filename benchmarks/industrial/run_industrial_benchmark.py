#!/usr/bin/env python3
"""Industrial Benchmark v1 harness — real routed GDS, apples-to-apples.

Measures OpenLithoHub's streaming/exact-vector pipeline against dense
baselines on a real public routed layout (OpenROAD-routed Ibex RISC-V
core, sky130hd, PDB Physical Design Database), with quality comparisons
on the same optical model, on this machine, and writes everything as
``OpenLithoHub.industrial-benchmark.v1`` artifacts.

Run identity (audit G0.1): a single SHA-256 over the measurement source
(commit + harness/core/generator hashes), the environment lock, the
fixture hashes and EVERY semantic CLI argument.  All mutable run state
lives under ``<out>/runs/<run_identity>/`` (fixtures, checkpoints,
progress, RUN.log, run-config.json); a checkpoint row from any other
identity is a hard error.  Final artifacts are published to ``<out>``
with ``manifest.json`` and ``SHA256SUMS.txt``.

Stages (each checkpointed; safe to interrupt and resume within the same
identity):

- ``prepare``   correctness witnesses + deterministic GDS fixtures.
- ``runtime``   dense vs tiled vs exact-vector vs selective streaming
                ladder on real center crops (4 sizes), ≥ 5 repeats per
                row, median/p10/p90, fresh worker process per timed run.
- ``quality``   same-optical-model (Hopkins SOCS) quality comparison of
                registered models on the highest-occupancy real tiles.
- ``fulldie``   sampled die-tile screening survey; the dense raster
                equivalent of the full die exceeds the reference
                machine's physical RAM (recorded as such, not as
                "structurally impossible").
- ``manifest``  publish artifacts + SHA-256 index.

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
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jobs  # noqa: E402 - sibling module of this script
import run_support as rs  # noqa: E402 - sibling module of this script

from openlithohub.benchmark.industrial import (  # noqa: E402
    SCHEMA_NAME,
    STATUS_INFEASIBLE_ON_REFERENCE_MACHINE,
    STATUS_NOT_RUN_MEMORY_POLICY,
    STATUS_SUCCESS,
    _sanitize,
    environment_snapshot,
    memory_reduction_pct,
    relative_reduction_pct,
    summarize,
    validate_artifact,
)
from openlithohub.streaming.geometry import BoundingBox  # noqa: E402
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy  # noqa: E402
from openlithohub.streaming.pipeline import run_streaming  # noqa: E402
from openlithohub.streaming.sinks import TensorTileSink  # noqa: E402

FORWARD_RADIUS = jobs.FORWARD_RADIUS
LAYER = jobs.LAYER
SEED = jobs.SEED

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


def log(msg: str) -> None:
    rs.log(msg)


def dense_allowed(size: int, *, bytes_per_px: int = 4, tensors: int = 3, budget_bytes: int) -> bool:
    return size * size * bytes_per_px * tensors <= budget_bytes


# ---------------------------------------------------------------------------
# fixture preparation (fail-closed identity, audit G0.2/G0.5)
# ---------------------------------------------------------------------------


def clip_gds(
    parent_gds: Path,
    out_gds: Path,
    *,
    parent_sha256: str,
    source: dict[str, Any],
    top_name: str,
    size_dbu: int,
    layer: str,
    pixel_size_nm: float,
) -> dict[str, Any]:
    """Center-crop the parent die to size x size dbu, with a reuse record."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(parent_gds))
    top = layout.cell("ibex_core")  # fail-closed: never top_cells()[0]
    if top is None:
        raise RuntimeError(f"{parent_gds}: expected top cell 'ibex_core' not found")
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
    return rs.fixture_record(
        out_gds,
        parent_sha256=parent_sha256,
        crop_bbox_dbu=[box.left, box.bottom, box.right, box.top],
        top_cell=top_name,
        layer=layer,
        pixel_size_nm=pixel_size_nm,
        generator_source=source,
    ) | {"size_px": [actual.height(), actual.width()]}


def make_die_sample_tiles(
    parent_gds: Path,
    out_dir: Path,
    *,
    parent_sha256: str,
    source: dict[str, Any],
    tile_px: int,
    tiles_per_side_override: int | None,
    layer: str,
    pixel_size_nm: float,
) -> dict[str, Any]:
    """Pre-clip a deterministic die-tile sample (center + 4 corners).

    ``tiles_per_side`` is DERIVED from the die bbox and tile size (G0.5);
    an explicit override is recorded and validated to actually cover the
    die.  Each tile ships with a full fixture identity record.
    """
    import klayout.db as kdb

    facts = rs.validate_parent_layout(
        parent_gds,
        expected_top_cell="ibex_core",
        layer=layer,
        expected_dbu_nm=1.0,
    )
    bb = facts["die_bbox_dbu"]
    die_w = bb[2] - bb[0]
    die_h = bb[3] - bb[1]
    # C4: rectangular dies get independent grid counts per axis.
    derived_x = -(-die_w // tile_px)
    derived_y = -(-die_h // tile_px)
    if tiles_per_side_override is None:
        grid_x, grid_y = derived_x, derived_y
    else:
        # A single override applies to both axes and must cover BOTH.
        grid_x, grid_y = tiles_per_side_override, tiles_per_side_override
        if grid_x < derived_x or grid_y < derived_y:
            raise RuntimeError(
                f"die tiles_per_side override {tiles_per_side_override} does not "
                f"cover the die (needs >= {derived_x}x{derived_y})"
            )
    last_x, last_y = grid_x - 1, grid_y - 1
    sample_positions = [
        (0, 0),
        (last_x, 0),
        (0, last_y),
        (last_x, last_y),
        (last_x // 2, last_y // 2),
    ]
    entries = []
    layout = kdb.Layout()
    layout.read(str(parent_gds))
    top = layout.cell("ibex_core")
    for ix, iy in sorted(set(sample_positions)):
        x0 = bb[0] + ix * tile_px
        y0 = bb[1] + iy * tile_px
        box = kdb.Box(x0, y0, min(x0 + tile_px, bb[2]), min(y0 + tile_px, bb[3]))
        name = f"IBEX_DIE_{ix}_{iy}"
        clipped_idx = layout.clip(top.cell_index(), box)
        layout.rename_cell(clipped_idx, name)
        out_ly = kdb.Layout()
        out_ly.dbu = layout.dbu
        new_top = out_ly.create_cell(name)
        new_top.copy_tree(layout.cell(clipped_idx))
        out_path = out_dir / f"die_tile_{ix:02d}_{iy:02d}.gds"
        out_ly.write(str(out_path))
        record = rs.fixture_record(
            out_path,
            parent_sha256=parent_sha256,
            crop_bbox_dbu=[box.left, box.bottom, box.right, box.top],
            top_cell=name,
            layer=layer,
            pixel_size_nm=pixel_size_nm,
            generator_source=source,
        )
        entries.append(
            {
                "tile": [ix, iy],
                "gds": str(out_path),
                "top_cell": name,
                "record": record,
            }
        )
    return {
        "die_bbox_dbu": bb,
        "die_size_px": facts["die_size_px"],
        "tile_px": tile_px,
        "grid_tiles_x": grid_x,
        "grid_tiles_y": grid_y,
        "tiles_per_side_derived": [derived_x, derived_y],
        "tiles_per_side_override": tiles_per_side_override,
        "n_grid_tiles": grid_x * grid_y,
        "n_sample_tiles": len(entries),
        "tiles": entries,
    }


# ---------------------------------------------------------------------------
# stage implementations
# ---------------------------------------------------------------------------


def stage_prepare(
    args: argparse.Namespace, run_dir: Path, script: Path, identity: str, source: dict[str, Any]
) -> dict[str, Any]:
    ckpt = rs.Checkpoint(run_dir / "checkpoints" / "prepare.jsonl", identity, "prepare")
    progress = rs.get_progress(run_dir / "progress.json")
    fixture_dir = run_dir / "fixtures"
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
        source_src = ExactVectorRunSource(
            shape=(size, size), cells={"TOP": cell}, top="TOP", pixel_size_nm=8.0
        )
        forward = jobs.make_forward()
        dense = source_src.read_window(BoundingBox(0, 0, size, size))
        reference = forward(dense)
        sink = TensorTileSink((size, size))
        run_streaming(
            source_src,
            sink,
            forward,
            core_size=128,
            halo_policy=LegacyFixedHaloPolicy(FORWARD_RADIUS),
            screening_policy=jobs.selective_policy(),
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
    assert witness_synth is not None
    if witness_synth["max_abs_error"] > 2e-6:
        raise RuntimeError(f"synthetic witness failed: {witness_synth}")

    def ensure_crop4096() -> Path:
        crop4096 = fixture_dir / "ibex_crop_4096.gds"
        rec_key = "crop_4096"
        rec = ckpt.get(rec_key)
        if rec is not None and crop4096.exists():
            rs.verify_fixture_reuse(
                crop4096, rec, parent_sha256=args.parent_gds_sha256, generator_source=source
            )
            return crop4096
        progress.begin_row("prepare", rec_key)
        t0 = time.perf_counter()
        info = clip_gds(
            Path(args.gds),
            crop4096,
            parent_sha256=args.parent_gds_sha256,
            source=source,
            top_name="IBEX_CROP_4096",
            size_dbu=4096,
            layer=LAYER,
            pixel_size_nm=1.0,
        )
        ckpt.append(rec_key, info)
        progress.end_row("prepare", rec_key, time.perf_counter() - t0)
        return crop4096

    key = "witness_real"
    if not ckpt.has(key):
        progress.begin_row("prepare", key)
        t0 = time.perf_counter()
        crop4096 = ensure_crop4096()
        row = jobs.run_worker(
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
    assert witness_real is not None
    if witness_real["max_abs_error"] > 2e-6:
        raise RuntimeError(f"real-layout witness failed: {witness_real}")

    # fixtures: center crops (ladder sizes + the selective-only maximum)
    crops: dict[str, Any] = {}
    for size in crop_sizes:
        key = f"crop_{size}"
        path = fixture_dir / f"ibex_crop_{size}.gds"
        rec = ckpt.get(key)
        if rec is not None and path.exists():
            rs.verify_fixture_reuse(
                path, rec, parent_sha256=args.parent_gds_sha256, generator_source=source
            )
        else:
            progress.begin_row("prepare", key)
            t0 = time.perf_counter()
            rec = clip_gds(
                Path(args.gds),
                path,
                parent_sha256=args.parent_gds_sha256,
                source=source,
                top_name=f"IBEX_CROP_{size}",
                size_dbu=size,
                layer=LAYER,
                pixel_size_nm=1.0,
            )
            ckpt.append(key, rec)
            progress.end_row("prepare", key, time.perf_counter() - t0)
        crops[str(size)] = rec | {"gds": str(path)}

    # die sample tiles (center + 4 corners of the derived grid)
    key = "die_grid"
    if not ckpt.has(key):
        progress.begin_row("prepare", key)
        t0 = time.perf_counter()
        die_dir = fixture_dir / "die_tiles"
        die_dir.mkdir(exist_ok=True)
        info = make_die_sample_tiles(
            Path(args.gds),
            die_dir,
            parent_sha256=args.parent_gds_sha256,
            source=source,
            tile_px=args.die_tile_px,
            tiles_per_side_override=args.die_tiles_per_side,
            layer=LAYER,
            pixel_size_nm=1.0,
        )
        ckpt.append(key, info)
        progress.end_row("prepare", key, time.perf_counter() - t0)
    die_grid_rec = ckpt.get(key)
    assert die_grid_rec is not None

    return {
        "witness_synthetic": witness_synth,
        "witness_real": witness_real,
        "crops": crops,
        "die_grid": die_grid_rec,
    }


def _ladder_plan(
    args: argparse.Namespace, prepared: dict[str, Any]
) -> list[tuple[int, list[str], int]]:
    """Deterministic (size, modes, repeats) plan for the runtime ladder."""
    sizes = [int(s) for s in args.sizes.split(",")]
    if args.max_selective_size > 0:
        sizes.append(args.max_selective_size)
    plan = []
    for size in sorted(set(sizes)):
        if prepared["crops"].get(str(size)) is None:
            continue
        modes: list[str] = []
        if dense_allowed(size, budget_bytes=args.dense_max_bytes):
            modes += ["dense_full", "tiled_raster"]
        if size <= args.max_vector_size:
            modes.append("b04_vector")
        modes.append("b04_selective")
        repeats = args.repeats if size <= 16384 else min(args.repeats, args.large_repeats)
        plan.append((size, modes, repeats))
    return plan


def stage_runtime(
    args: argparse.Namespace,
    run_dir: Path,
    script: Path,
    identity: str,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    ckpt = rs.Checkpoint(run_dir / "checkpoints" / "runtime.jsonl", identity, "runtime")
    progress = rs.get_progress(run_dir / "progress.json")
    plan = _ladder_plan(args, prepared)
    total_rows = sum(len(modes) * repeats for _, modes, repeats in plan)
    progress.stage("runtime", total_rows)
    log(f"runtime: {total_rows} checkpointed worker rows")

    rows: list[dict[str, Any]] = []
    for size, modes, repeats in plan:
        crop = prepared["crops"][str(size)]
        gds, top = crop["gds"], crop["top_cell"]
        policy_blocked = not dense_allowed(size, budget_bytes=args.dense_max_bytes)
        for mode in modes:
            if policy_blocked and mode in ("dense_full", "tiled_raster"):
                rows.append(
                    {
                        "mode": mode,
                        "size_px": [size, size],
                        "status": STATUS_NOT_RUN_MEMORY_POLICY,
                        "dense_input_bytes_structural": size * size * 4,
                        "note": (
                            "dense input exceeds --dense-max-bytes policy budget; a "
                            "policy decision, not a structural impossibility"
                        ),
                    }
                )
                continue
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
                rec = jobs.run_worker(
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
            merged["status"] = STATUS_SUCCESS
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
        if row["mode"] == "b04_selective" and row.get("status") == STATUS_SUCCESS
    ]
    policy_blocked_sizes = sorted(
        {
            int(row["size_px"][0])
            for row in rows
            if row.get("status") == STATUS_NOT_RUN_MEMORY_POLICY
        }
    )
    memory["max_streamed_size_px"] = max(streamed_ok) if streamed_ok else None
    memory["dense_not_run_under_memory_policy_px"] = policy_blocked_sizes

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
            "large_repeats": args.large_repeats,
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
        "direction": "baseline_dense_over_candidate_streaming",
    }


def _safe_summarize(values: list[float | None]) -> dict[str, Any]:
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
        "identity_vs_rulebased": {
            "pvband_mean_nm": agg_reduction("pvband_mean_nm", "dummy-identity", "rule-based-opc"),
            "mrc_violation_rate": agg_reduction(
                "mrc_violation_rate", "dummy-identity", "rule-based-opc"
            ),
            "wafer_epe_mean_nm": agg_reduction(
                "wafer_epe_mean_nm", "dummy-identity", "rule-based-opc"
            ),
        },
        "levelset_vs_surrogate_runtime": surrogate_speed,
    }
    return {"aggregate": aggregate, "comparisons": comparisons}


def stage_quality(
    args: argparse.Namespace,
    run_dir: Path,
    script: Path,
    identity: str,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    ckpt = rs.Checkpoint(run_dir / "checkpoints" / "quality.jsonl", identity, "quality")
    progress = rs.get_progress(run_dir / "progress.json")
    crop = prepared["crops"]["4096"]
    models = list(args.parsed_models)

    # deterministic tile selection: rank tiles of the 4096 crop by
    # occupancy (measured once, in-driver, untimed), keep the top K.
    key = "tile_selection"
    if not ckpt.has(key):
        source = jobs.load_source(Path(crop["gds"]), crop["top_cell"])
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
    assert sel is not None

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
        rec = jobs.run_worker(
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
            "description": (
                "real routed sky130hd ibex tiles (1024px @ 1nm/px), 193nm NA1.35 optics"
            ),
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
            rec = jobs.run_worker(
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
                "optical_config": (
                    "Hopkins SOCS, 13.5nm, NA 0.33, sigma 0.7, 24 kernels, threshold 0.5"
                ),
            },
            **_aggregate_quality([rec], models),
        }
        all_rows["iccad16-testcase1"] = [rec]

    return {
        "datasets": datasets,
        "tile_rows": _sanitize(all_rows),
        "tile_selection": sel,
    }


def stage_fulldie(
    args: argparse.Namespace,
    run_dir: Path,
    script: Path,
    identity: str,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    """Die-scale characteristics: RAM feasibility + sampled survey.

    The full-die dense raster equivalent (die_px^2 x 4 bytes) is compared
    against physical RAM — machine-relative arithmetic, recorded with an
    explicit machine-relative status.  A measured screen-only survey runs
    on a deterministic sample of die tiles; the full-die survey cost is an
    explicit ESTIMATE (mean per-tile cost x grid tile count), never
    presented as measured.
    """
    ckpt = rs.Checkpoint(run_dir / "checkpoints" / "fulldie.jsonl", identity, "fulldie")
    progress = rs.get_progress(run_dir / "progress.json")
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
        rec = jobs.run_worker(
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
        "grid_tiles_per_side": [grid["grid_tiles_x"], grid["grid_tiles_y"]],
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
    env_lock: dict[str, Any],
) -> dict[str, Any]:
    env = environment_snapshot()
    source = args.measurement_source
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
        "fixture": args.fixture_block,
        "measurement_source": source,
        "environment_lock": env_lock,
        "run_identity": args.run_identity,
        "claim_scope": dict(PHYSICS_SCOPE),
        "reproducibility": {
            "command": "python benchmarks/industrial/run_industrial_benchmark.py "
            f"--gds <ibex.gds> --out {args.out} --repeats {args.repeats}",
            "seed": SEED,
        },
    }
    artifact.update(payload)
    return artifact


def write_artifact(run_dir: Path, name: str, artifact: dict[str, Any]) -> Path:
    """Write an artifact into the RUN WORKSPACE only (audit B0.1).

    The public artifact root is populated exclusively by
    :func:`publish_family` after the complete family passes closure
    validation — a partial or mixed-run family can never appear there.
    """
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
    path = run_dir / name
    path.write_text(rs.strict_dumps(artifact) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    return path


def _iccad_fixture_hashes(iccad_dir: Path) -> dict[str, str]:
    hashes = {}
    for rel in ("testcase1.oas", "test1.csv"):
        p = iccad_dir / rel
        if p.exists():
            hashes[rel] = rs.sha256_file(p)
    return hashes


def compute_run_identity_from_args(
    args: argparse.Namespace, source: dict[str, Any]
) -> tuple[str, dict[str, Any], str]:
    """Compute the run identity + full run config from parsed harness args.

    This is the SINGLE source of the identity — ``--print-run-identity``
    and the preflight script both use it, so a preflight identity is by
    construction the formal run identity (audit B0.3).
    """
    # P0.6: canonicalize/validate the model list HERE so every identity
    # consumer (formal run, preflight, verifier recompute) shares exactly
    # one model-list contract.
    args.parsed_models = jobs.canonicalize_models(args.models)
    env_lock, freeze_text = rs.environment_lock(script_repo_root())
    iccad_hashes = _iccad_fixture_hashes(args.iccad16_dir) if args.iccad16_dir else None
    args_payload = {
        "repeats": args.repeats,
        "large_repeats": args.large_repeats,
        "core": args.core,
        "sizes": args.sizes,
        "dense_max_bytes": args.dense_max_bytes,
        "max_vector_size": args.max_vector_size,
        "max_selective_size": args.max_selective_size,
        "die_tile_px": args.die_tile_px,
        "die_tiles_per_side": args.die_tiles_per_side,
        "die_core_px": args.die_core_px,
        "tile_size": args.tile_size,
        "min_tile_occupancy": args.min_tile_occupancy,
        "max_tiles": args.max_tiles,
        "ilt_iterations": args.ilt_iterations,
        "ilt_iterations_iccad16": args.ilt_iterations_iccad16,
        "quality_reps": args.quality_reps,
        "surrogate_train_samples": args.surrogate_train_samples,
        "surrogate_epochs": args.surrogate_epochs,
        "iccad16_crop_px": args.iccad16_crop_px,
        "iccad16_dir": str(args.iccad16_dir) if args.iccad16_dir else None,
        "models": ",".join(args.parsed_models),
        "layer": LAYER,
        "pixel_size_nm": 1.0,
        "seed": SEED,
    }
    identity = rs.compute_run_identity(
        source=source,
        environment_lock_sha256=env_lock["lock_sha256"],
        parent_gds_sha256=args.parent_gds_sha256,
        iccad_fixture_hashes=iccad_hashes,
        args_payload=args_payload,
    )
    config = {
        "schema": "OpenLithoHub.industrial-run-config.v1",
        "run_identity": identity,
        "source": source,
        "environment_lock": env_lock,
        "fixtures": {
            "parent_gds_sha256": args.parent_gds_sha256,
            "iccad": iccad_hashes or {},
        },
        "args": args_payload,
    }
    return identity, config, freeze_text


def script_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    """The harness argument parser — shared with the preflight script so
    a preflight identity is computed from exactly the same configuration."""
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
    ap.add_argument(
        "--die-tiles-per-side",
        type=int,
        default=None,
        help="override the derived grid side; recorded and validated",
    )
    ap.add_argument("--die-core-px", type=int, default=4096)
    ap.add_argument("--tile-size", type=int, default=1024, help="quality tile side (px)")
    ap.add_argument("--min-tile-occupancy", type=float, default=0.02)
    ap.add_argument("--max-tiles", type=int, default=6)
    ap.add_argument("--ilt-iterations", type=int, default=50)
    ap.add_argument("--ilt-iterations-iccad16", type=int, default=200)
    ap.add_argument("--quality-reps", type=int, default=3)
    ap.add_argument("--surrogate-train-samples", type=int, default=16)
    ap.add_argument("--surrogate-epochs", type=int, default=3)
    ap.add_argument("--iccad16-crop-px", type=int, default=256)
    ap.add_argument(
        "--iccad16-dir",
        type=Path,
        default=None,
        help="optional: directory with testcase1.oas/test1.csv for the second quality dataset",
    )
    ap.add_argument("--large-repeats", type=int, default=3, help="repeats for sizes > 16384px")
    ap.add_argument("--worker", action="store_true")
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
    ap.add_argument(
        "--models",
        default=",".join(jobs.MODELS),
        help="comma-separated registered model names; validated, canonicalized"
        " and bound into the run identity",
    )
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument(
        "--print-run-identity",
        action="store_true",
        help="compute the run identity/config for the given args and exit "
        "without measuring (used by the preflight gate)",
    )
    return ap


def publish_family(run_dir: Path, out: Path, identity: str) -> None:
    """Family-closure gate + commit-marker publication (audit B0.1/P0.1).

    JSON members are strict-parsed and schema-validated; the freeze is a
    TEXT blob handled only as bytes (json.loads on it is a bug).  The
    manifest is the COMMIT MARKER written to the public root last —
    per-file replacement is not cross-file atomic, so readers must
    validate the manifest/SHA256SUMS family before consuming.
    """
    import openlithohub.benchmark.industrial as industrial_mod

    json_members = [
        "industrial-runtime.json",
        "industrial-quality.json",
        "industrial-fulldie.json",
        "industrial-run-config.json",
    ]
    blob_members = ["industrial-distribution-freeze.txt"]
    expected = json_members + blob_members

    # P0.9: the run workspace must carry EXACTLY the expected family —
    # unexpected industrial-* members refuse promotion.
    actual_candidates = {p.name for p in run_dir.glob("industrial-*")}
    unexpected = actual_candidates - set(expected)
    if unexpected:
        raise RuntimeError(
            f"run workspace carries unexpected industrial-* members {sorted(unexpected)}; "
            "refusing promotion (exact family set enforced)"
        )
    missing = [n for n in expected if not (run_dir / n).exists()]
    if missing:
        raise RuntimeError(
            f"refusing to publish an incomplete family from {run_dir}: "
            f"missing {missing} — run the missing stages under the SAME identity"
        )

    dirty_override = os.environ.get("OPENLITHOHUB_ALLOW_DIRTY_MEASUREMENT") == "1"
    parsed: dict[str, dict[str, Any]] = {}
    problems: list[str] = []

    # --- JSON members: strict parse + role-specific validation (P0.1) ---
    for name in json_members:
        try:
            data = json.loads(
                (run_dir / name).read_text(encoding="utf-8"),
                parse_constant=lambda tok: (_ for _ in ()).throw(
                    ValueError(f"non-finite token {tok}")
                ),
            )
        except ValueError as e:
            problems.append(f"{name}: NOT strict JSON: {e}")
            continue
        role = name.replace("industrial-", "").replace(".json", "")
        if role == "run-config":
            role_problems = industrial_mod.validate_run_config(data)
        else:
            role_problems = validate_artifact(data)
            source_commit = str((data.get("measurement_source") or {}).get("commit") or "")
            git_commit = str(data.get("git_commit") or "")
            if source_commit and git_commit != source_commit:
                role_problems.append(
                    f"git_commit {git_commit} != measurement_source.commit {source_commit}"
                )
        if dirty_override:
            role_problems = [p for p in role_problems if "working_tree_dirty" not in p]
        problems.extend(f"{name}: {p}" for p in role_problems)
        parsed[role] = data

    # --- BLOB member: bytes only, never json.loads (P0.1) ---
    freeze_name = blob_members[0]
    freeze_bytes = (run_dir / freeze_name).read_bytes()
    freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
    config = parsed.get("run-config")
    lock_freeze = (
        str((config.get("environment_lock") or {}).get("distribution_freeze_sha256") or "")
        if config
        else ""
    )
    if not lock_freeze:
        problems.append(f"{freeze_name}: run-config carries no freeze hash to bind")
    elif freeze_sha != lock_freeze:
        problems.append(
            f"{freeze_name}: sha256 {freeze_sha} does not match "
            f"environment_lock.distribution_freeze_sha256 {lock_freeze}"
        )

    # --- cross-member family closure (single validator) ---
    problems.extend(f"family: {p}" for p in industrial_mod.validate_artifact_family(parsed))
    if "run-config" in parsed:
        recomputed = industrial_mod.recompute_run_identity(parsed["run-config"])
        for role, data in sorted(parsed.items()):
            claimed = str(data.get("run_identity"))
            if claimed != recomputed:
                problems.append(
                    f"{role}: run_identity {claimed} != recomputed {recomputed} "
                    "from the published run-config"
                )
    if problems:
        raise RuntimeError(f"family closure failed; nothing published: {problems}")

    # --- manifest (commit marker) over the five data members ---
    entries = [
        {"file": n, "sha256": rs.sha256_file(run_dir / n), "bytes": (run_dir / n).stat().st_size}
        for n in expected
    ]
    manifest = build_artifact(
        "manifest",
        STATUS_SUCCESS,
        {
            "artifacts": entries,
            "run_identity": identity,
            "run_config": "industrial-run-config.json",
        },
        _PUBLISH_ARGS,
        env_lock=parsed["run-config"]["environment_lock"],
    )
    manifest_problems = (
        [p for p in validate_artifact(_sanitize(manifest)) if "working_tree_dirty" not in p]
        if dirty_override
        else validate_artifact(_sanitize(manifest))
    )
    if manifest_problems:
        raise RuntimeError(f"manifest failed validation: {manifest_problems}")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(rs.strict_dumps(_sanitize(manifest)) + "\n", encoding="utf-8")

    # --- publication: manifest LAST (commit marker) ---
    staging = out / ".publish-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in expected:
        shutil.copyfile(run_dir / name, staging / name)
    sums = ""
    for name in expected:
        digest = rs.sha256_file(staging / name)
        sums += f"{digest}  {name}\n"
    (staging / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    for name in expected:
        shutil.copyfile(staging / name, out / name)
    (staging / "SHA256SUMS.txt").replace(out / "SHA256SUMS.txt")
    # Commit marker last.
    shutil.copyfile(manifest_path, out / "manifest.json")
    shutil.rmtree(staging, ignore_errors=True)
    log(f"published complete family for identity {identity[:16]} to {out}")


# Set by main() before stages run; publish_family needs the args context
# (fixture block, measurement source) to build the manifest.
_PUBLISH_ARGS: argparse.Namespace | None = None


# Set by main() before stages run; publish_family needs the args context
# (fixture block, measurement source) to build the manifest.
_PUBLISH_ARGS: argparse.Namespace | None = None


def main() -> int:
    global _PUBLISH_ARGS
    args = build_arg_parser().parse_args()

    script = Path(__file__).resolve()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.worker:
        if args.parent_pid and args.parent_pid > 0:
            import threading

            def _orphan_watchdog() -> None:
                while True:
                    if os.getppid() != args.parent_pid:
                        os._exit(97)
                    time.sleep(5)

            threading.Thread(target=_orphan_watchdog, daemon=True).start()
        job_map = {
            "runtime_row": jobs.job_runtime_row,
            "real_witness": jobs.job_real_witness,
            "quality_tile": jobs.job_quality_tile,
            "quality_iccad16": jobs.job_quality_iccad16,
            "fulldie_tile": jobs.job_fulldie_tile,
        }
        if args.job not in job_map:
            raise SystemExit(f"unknown job {args.job!r}")
        print(json.dumps(job_map[args.job](args), sort_keys=True, allow_nan=False))
        return 0

    if args.gds is None or not Path(args.gds).exists():
        raise SystemExit("--gds is required and must exist")

    repo_root = script_repo_root()
    source = rs.measurement_source(repo_root)
    args.measurement_source = source
    if source["commit"] is None or not source["commit_valid"]:
        raise SystemExit("measurement source commit is not a full 40-hex commit")
    if (
        source["working_tree_dirty"]
        and os.environ.get("OPENLITHOHUB_ALLOW_DIRTY_MEASUREMENT") != "1"
    ):
        raise SystemExit(
            "working tree has uncommitted tracked changes — commit (or set the "
            "explicit OPENLITHOHUB_ALLOW_DIRTY_MEASUREMENT=1 override for "
            "provisional, CI-inadmissible runs)"
        )
    if source["working_tree_dirty"]:
        log("WARNING: dirty-tree run allowed by override; artifacts are provisional")

    # P0.6: --models is a real configuration knob — canonicalize, validate
    # against the registry, and bind the result into the run identity.
    try:
        args.parsed_models = jobs.canonicalize_models(args.models)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    args.parent_gds_sha256 = rs.sha256_file(Path(args.gds))
    identity, run_config, freeze_text = compute_run_identity_from_args(args, source)
    args.run_identity = identity
    run_dir = out / "runs" / identity
    run_dir.mkdir(parents=True, exist_ok=True)
    # Publishable run config (audit B0.9): the public family includes it.
    (run_dir / "industrial-run-config.json").write_text(
        rs.strict_dumps(_sanitize(run_config)) + "\n", encoding="utf-8"
    )
    # Distribution freeze persisted + PUBLISHED next to the artifacts it
    # locks (B0.2 / third-pass P0.6): the bytes, not just the hash, are
    # part of the public family.
    (run_dir / "industrial-distribution-freeze.txt").write_text(freeze_text, encoding="utf-8")
    (run_dir / "run-config.json").write_text(rs.strict_dumps(run_config) + "\n", encoding="utf-8")
    rs.init_run_log(run_dir / "RUN.log")
    args.fixture_block = rs.validate_parent_layout(
        Path(args.gds),
        expected_top_cell="ibex_core",
        layer=LAYER,
        expected_dbu_nm=1.0,
    ) | {
        "sha256": args.parent_gds_sha256,
        "bytes": Path(args.gds).stat().st_size,
        "filename": Path(args.gds).name,
        "pixel_size_nm": 1.0,
    }
    _PUBLISH_ARGS = args
    log(f"run identity {identity[:16]} workspace {run_dir}")

    if args.print_run_identity:
        print(
            json.dumps(
                {"run_identity": identity, "run_config": run_config},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    stages = (
        ["prepare", "runtime", "quality", "fulldie", "manifest"]
        if args.stage == "all"
        else [args.stage]
    )
    prepared: dict[str, Any] = {}
    if "prepare" in stages:
        prepared = stage_prepare(args, run_dir, script, identity, source)
    if "runtime" in stages:
        if not prepared:
            prepared = stage_prepare(args, run_dir, script, identity, source)
        runtime = stage_runtime(args, run_dir, script, identity, prepared)
        write_artifact(
            run_dir,
            "industrial-runtime.json",
            build_artifact(
                "runtime",
                STATUS_SUCCESS,
                runtime,
                args,
                env_lock=run_config["environment_lock"],
            ),
        )
    if "quality" in stages:
        if not prepared:
            prepared = stage_prepare(args, run_dir, script, identity, source)
        quality = stage_quality(args, run_dir, script, identity, prepared)
        write_artifact(
            run_dir,
            "industrial-quality.json",
            build_artifact(
                "quality",
                STATUS_SUCCESS,
                quality,
                args,
                env_lock=run_config["environment_lock"],
            ),
        )
    if "fulldie" in stages:
        if not prepared:
            prepared = stage_prepare(args, run_dir, script, identity, source)
        fulldie = stage_fulldie(args, run_dir, script, identity, prepared)
        write_artifact(
            run_dir,
            "industrial-fulldie.json",
            build_artifact(
                "full_die",
                STATUS_SUCCESS,
                fulldie,
                args,
                env_lock=run_config["environment_lock"],
            ),
        )
    if "manifest" in stages:
        publish_family(run_dir, out, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
