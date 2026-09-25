"""Industrial Scale streaming benchmark harness (scale track, S2).

Executes the frozen charter (docs/industrial-scale-benchmark.md) on top
of the PRODUCTION streaming spine — the same ``run_streaming`` tiler,
ownership ledger and sinks used by the product path.  No second tiler
exists here.

Protocol properties:

* exact-vector source: ``KLayoutAlignedRunSource`` over the PREPARED
  fixture GDS with the manifest's EXPLICIT selected layer; measured
  windows are centered die crops (the crop is the executed layout);
* full-layout firewall: a full-die run REQUIRES an out-of-core sink
  (``memmap_npy`` / ``manhattan``).  A resident full-layout tensor is a
  protocol violation and FAILS the run — it is never silently allowed;
* forward profiles: ``P0_IDENTITY`` (plumbing, headline-ineligible),
  ``P1_FINITE_SUPPORT`` (deterministic blur, headline-eligible),
  ``P2_HOPKINS_BOUNDED`` (lands with the GPU phase — recorded
  UNSUPPORTED, never faked);
* bounded metrics per repeat: wall, host peak RSS, per-device GPU peaks
  (CUDA only), tiles, forward batches, output bytes;
* claim-bearing statistics (median/p10/p90/n) are computed by THIS
  driver over repeats; a SUCCESS row always has n == repeat_count —
  a stale/missing metric fails the row;
* honest statuses: a window the host cannot run is recorded
  NOT_RUN_MEMORY_POLICY / NOT_RUN_ENVIRONMENT / UNSUPPORTED with a
  reason, never skipped;
* run identity + environment lock come from the authority core
  (``openlithohub.benchmark.industrial_scale``); artifacts land under
  ``benchmarks/results/industrial-scale/runs/<identity>/`` and are
  PROVISIONAL unless ``--formal`` closes every formal blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess  # noqa: S404 — fixed-argv worker re-invocation / git only
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
HARNESS_PATH = Path(__file__).resolve()
CORE_PATH = _REPO / "src" / "openlithohub" / "benchmark" / "industrial_scale.py"
VERIFIER_PATH = _REPO / "scripts" / "verify_industrial_scale_artifacts.py"
CLAIM_GENERATOR_PATH = _REPO / "scripts" / "generate_industrial_scale_claims.py"

WITNESS_WINDOW_PX = 512  # bounded correctness witness (dense-checkable)
FLOAT_TOLERANCE = 1e-5

LANE_A = "A_LARGE_LAYOUT_STREAMING"
LANE_B = "B_MULTI_GPU_SCALING"


class FirewallViolationError(Exception):
    """A run attempted something the charter forbids (e.g. a resident
    full-layout tensor).  Always FAILS the run."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_measurement_commit(repo: Path) -> tuple[str, bool]:
    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603,S607 — fixed-argv git query
            ["/usr/bin/git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    return commit, dirty == ""


def build_source_hashes() -> dict[str, str]:
    def sha_or_empty(path: Path) -> str:
        return sha256_file(path) if path.is_file() else ""

    return {
        "harness": sha256_file(HARNESS_PATH),
        "core": sha256_file(CORE_PATH),
        "verifier": sha_or_empty(VERIFIER_PATH),
        "claim_generator": sha_or_empty(CLAIM_GENERATOR_PATH),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def summarize_repeats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median_s": statistics.median(values) if values else 0.0,
        "p10_s": percentile(values, 0.10),
        "p90_s": percentile(values, 0.90),
    }


# ---- forward profiles (charter §S5) ---------------------------------------------


def resolve_forward(profile: str, radius: int, sigma_nm: float, device: str):
    """Return (forward_fn, batched_forward_fn, headline_eligible)."""
    if profile == "P0_IDENTITY":
        return (lambda tile: tile), None, False
    if profile == "P1_FINITE_SUPPORT":
        from openlithohub.benchmark.industrial_v2 import BatchedFiniteSupportBlur

        blur = BatchedFiniteSupportBlur(radius=radius, sigma=sigma_nm)
        if device.startswith("cuda"):
            blur.to(device)
        return (
            lambda tile: blur.window_forward(tile).to(tile.device),
            (lambda batch: blur.batch_forward(batch)) if device.startswith("cuda") else None,
            True,
        )
    if profile == "P2_HOPKINS_BOUNDED":
        raise ProfileUnavailableError(
            "P2_HOPKINS_BOUNDED lands with the GPU phase (G4) on a CUDA host; "
            "it is never emulated on CPU"
        )
    raise ProfileUnavailableError(f"unknown forward profile {profile!r}")


class ProfileUnavailableError(Exception):
    pass


def _multi_worker_module():
    import sys

    if str(HARNESS_PATH.parent) not in sys.path:
        sys.path.insert(0, str(HARNESS_PATH.parent))
    import multi_worker

    return multi_worker


# ---- full-layout firewall (charter §Full-layout firewall) ------------------------


def enforce_full_layout_firewall(
    *, full_die: bool, sink_kind: str, window: int, die_px: tuple[int, int]
) -> None:
    """A resident full-layout tensor is a protocol violation.  Full-die
    runs REQUIRE an out-of-core sink; TensorTileSink is dev-only and can
    never carry a full-die or large-layout authority row."""
    out_of_core = ("memmap_npy", "manhattan")
    if not full_die:
        return
    if window < max(die_px):
        return
    if sink_kind not in out_of_core:
        raise FirewallViolationError(
            f"full-die window {window} with sink {sink_kind!r} would require a "
            f"resident full-layout tensor; use an out-of-core sink {out_of_core}"
        )


# ---- one streamed window execution ------------------------------------------------


def build_window_source(gds: str, layer: str, pixel_nm: float, window: int):
    from openlithohub.streaming.crop_source import ExactVectorCropSource
    from openlithohub.streaming.geometry import BoundingBox
    from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

    parse_start = time.perf_counter()
    parent = KLayoutAlignedRunSource.from_file(gds, pixel_size_nm=pixel_nm, layer=layer)
    parse_wall = time.perf_counter() - parse_start
    side = min(window, parent.shape[0], parent.shape[1])
    y0 = max(0, (parent.shape[0] - side) // 2)
    x0 = max(0, (parent.shape[1] - side) // 2)
    crop = ExactVectorCropSource(parent, BoundingBox(x0, y0, x0 + side, y0 + side))
    return crop, parse_wall, side


def stream_once(
    *,
    source: Any,
    forward_fn: Any,
    batched_forward_fn: Any,
    sink: Any,
    tile: int,
    microbatch: int,
    queue_depth: int,
    device: str,
    pixel_nm: float,
) -> dict[str, Any]:
    """One bounded streaming execution over the production spine."""
    import torch

    from openlithohub.streaming import run_streaming

    del queue_depth  # run_streaming's pending window is bounded by microbatch
    peak: dict[str, int] = {}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    report = run_streaming(
        source,
        sink,
        lambda tile_tensor: forward_fn(tile_tensor).cpu(),
        core_size=tile,
        pixel_nm=pixel_nm,
        batch_size=microbatch,
        batched_forward_fn=(
            (lambda batch: batched_forward_fn(batch).cpu()) if batched_forward_fn else None
        ),
    )
    wall = time.perf_counter() - start
    if device.startswith("cuda"):
        peak = {
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    import resource

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "wall_s": wall,
        "host_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "max_memory_allocated": peak.get("max_memory_allocated", 0),
        "max_memory_reserved": peak.get("max_memory_reserved", 0),
        "n_tiles": report.n_tiles,
        "n_forward_batches": report.forward_batches,
    }


def stream_once_multi(
    *,
    source: Any,
    forward_fn: Any,
    sink: Any,
    tile: int,
    halo: int,
    queue_depth: int,
    worker_count: int,
    device_backend: str,
    gpu_count: int,
) -> dict[str, Any]:
    """One bounded multi-worker streaming execution (lane B).  Each tile
    is owned by exactly one worker; commits are deterministic and
    exactly-once (charter §S6.2).  CPU hosts run cpu-worker-emulation;
    CUDA hosts bind worker i to cuda:(i mod gpu_count)."""
    import resource

    import torch

    multi_worker = _multi_worker_module()

    devices_used: list[str] = []

    def device_for_worker(index: int) -> str:
        if device_backend == "cuda" and gpu_count > 0:
            device = f"cuda:{index % gpu_count}"
        else:
            device = "cpu"
        devices_used.append(device)
        return device

    report = multi_worker.run_multi_worker_stream(
        source=source,
        sink=sink,
        forward_fn=forward_fn,
        core_size=tile,
        halo_px=halo,
        worker_count=worker_count,
        queue_depth=queue_depth,
        device_for_worker=device_for_worker,
    )
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peaks: dict[str, int] = {}
    per_gpu: dict[str, dict[str, int]] = {}
    for device in sorted(set(devices_used)):
        if device.startswith("cuda"):
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
            per_gpu[device] = {
                "max_memory_allocated": allocated,
                "max_memory_reserved": reserved,
            }
            peaks["max_memory_allocated"] = max(peaks.get("max_memory_allocated", 0), allocated)
            peaks["max_memory_reserved"] = max(peaks.get("max_memory_reserved", 0), reserved)
    return {
        "wall_s": report.wall_s,
        "host_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "max_memory_allocated": peaks.get("max_memory_allocated", 0),
        "max_memory_reserved": peaks.get("max_memory_reserved", 0),
        "per_gpu": per_gpu,
        "n_tiles": report.n_tiles,
        "n_forward_batches": report.n_tiles,
        "worker_counts": report.worker_counts,
        "max_resident_results": report.max_resident_results,
    }


def correctness_witness(
    *, gds: str, layer: str, pixel_nm: float, profile: str, radius: int, sigma_nm: float
) -> dict[str, Any]:
    """Bounded (<= 512²) dense-checkable parity witness: the ACTUAL
    streaming pipeline (production tiler + ownership + sink) over a
    small centered crop must equal the directly computed dense reference
    of the same window.  This witness gates the whole run; the window is
    small enough that its dense reference is not a full-layout tensor."""
    import numpy as np
    import torch

    from openlithohub.streaming import run_streaming
    from openlithohub.streaming.crop_source import ExactVectorCropSource
    from openlithohub.streaming.geometry import BoundingBox
    from openlithohub.streaming.sinks import TensorTileSink
    from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

    parent = KLayoutAlignedRunSource.from_file(gds, pixel_size_nm=pixel_nm, layer=layer)
    side = min(WITNESS_WINDOW_PX, parent.shape[0], parent.shape[1])
    y0 = max(0, (parent.shape[0] - side) // 2)
    x0 = max(0, (parent.shape[1] - side) // 2)
    bbox = BoundingBox(x0, y0, x0 + side, y0 + side)
    dense_ref = parent.read_window(bbox)

    forward_fn, _, _ = resolve_forward(profile, radius, sigma_nm, "cpu")
    # the dense reference is the SAME forward applied directly to the window
    dense_forwarded = forward_fn(dense_ref)
    crop = ExactVectorCropSource(parent, bbox)
    sink = TensorTileSink(crop.shape)
    run_streaming(
        crop,
        sink,
        lambda tile: forward_fn(tile).cpu(),
        core_size=min(64, side),
        pixel_nm=pixel_nm,
    )
    streamed = sink.finalize()
    if profile == "P0_IDENTITY":
        parity = bool(torch.equal(streamed, dense_forwarded))
    else:
        parity = bool(
            torch.allclose(streamed, dense_forwarded, atol=FLOAT_TOLERANCE, rtol=0.0)
            and np.isfinite(streamed.numpy()).all()
        )
    return {
        "witness_window": side,
        "correctness_witness_pass": parity,
        "finite_witness": bool(np.isfinite(streamed.numpy()).all()),
    }


# ---- window row (lane A / lane B baseline) ----------------------------------------


def run_window_row(
    *,
    gds: str,
    layer: str,
    pixel_nm: float,
    window: int,
    full_die: bool,
    die_px: tuple[int, int],
    profile: str,
    radius: int,
    sigma_nm: float,
    sink_kind: str,
    tile: int,
    halo: int,
    microbatch: int,
    queue_depth: int,
    device: str,
    worker_count: int,
    gpu_count: int,
    lane: str,
    warmup: int,
    repeats: int,
    workspace: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "lane": lane,
        "window": window,
        "worker_count": worker_count,
        "forward_profile": profile,
        "sink_kind": sink_kind,
        "device": device,
        "device_requires_cuda": device.startswith("cuda"),
        "timing_method": "cuda_synchronized" if device.startswith("cuda") else "host_perf_counter",
        "headline_eligible": profile == "P1_FINITE_SUPPORT",
        "warmup_discarded": warmup,
        "repeat_count": repeats,
    }
    if profile == "P2_HOPKINS_BOUNDED":
        row["status"] = "UNSUPPORTED"
        row["reason"] = "P2 lands with the GPU phase (G4); never faked on CPU"
        return row
    if device.startswith("cuda") and not torch_cuda_available():
        row["status"] = "NOT_RUN_ENVIRONMENT"
        row["reason"] = "CUDA measurement environment unavailable"
        return row

    try:
        enforce_full_layout_firewall(
            full_die=full_die, sink_kind=sink_kind, window=window, die_px=die_px
        )
    except FirewallViolationError as exc:
        row["status"] = "FAILED"
        row["reason"] = f"full-layout firewall: {exc}"
        return row

    witness = correctness_witness(
        gds=gds, layer=layer, pixel_nm=pixel_nm, profile=profile, radius=radius, sigma_nm=sigma_nm
    )
    row.update(witness)

    source, parse_wall, side = build_window_source(gds, layer, pixel_nm, window)
    row["parse_wall_s"] = parse_wall
    row["executed_window"] = side
    row["full_die"] = bool(side >= max(die_px))

    forward_fn, batched_fn, _ = resolve_forward(profile, radius, sigma_nm, device)
    from openlithohub.streaming.sinks import MemmapTileSink

    # Lane B uses the sharded executor at EVERY worker count — the T1
    # baseline must share the exact execution semantics of T2/T3, or the
    # scaling ratios would compare different code paths.
    multi_worker = lane == LANE_B
    walls: list[float] = []
    metrics: list[dict[str, Any]] = []
    output_bytes = 0
    for index in range(max(0, warmup) + max(1, repeats)):
        sink_path = workspace / f"w{window}-r{index}.npy"
        sink = MemmapTileSink(source.shape, sink_path, npy=True)
        if multi_worker:
            one = stream_once_multi(
                source=source,
                forward_fn=lambda tile_tensor: forward_fn(tile_tensor),
                sink=sink,
                tile=tile,
                halo=halo,
                queue_depth=queue_depth,
                worker_count=worker_count,
                device_backend="cuda" if device.startswith("cuda") else "cpu-worker-emulation",
                gpu_count=gpu_count,
            )
        else:
            one = stream_once(
                source=source,
                forward_fn=forward_fn,
                batched_forward_fn=batched_fn,
                sink=sink,
                tile=tile,
                microbatch=microbatch,
                queue_depth=queue_depth,
                device=device,
                pixel_nm=pixel_nm,
            )
        one["output_bytes"] = sink_path.stat().st_size
        if index >= warmup:
            walls.append(one["wall_s"])
            metrics.append(one)
            output_bytes = one["output_bytes"]

    aggregate = summarize_repeats(walls)
    row["aggregate_n"] = aggregate["n"]
    row["aggregate_median_s"] = aggregate["median_s"]
    row["aggregate_p10_s"] = aggregate["p10_s"]
    row["aggregate_p90_s"] = aggregate["p90_s"]
    executed_px = side * side
    row["layout_equivalent_pixels"] = executed_px
    # trusted cores are an exact partition of the executed window, so the
    # forwarded (core) pixel count per repeat equals the window area
    row["forwarded_pixels"] = executed_px * len(metrics)
    row["n_tiles"] = int(statistics.fmean([m["n_tiles"] for m in metrics])) if metrics else 0
    row["n_forward_batches"] = (
        int(statistics.fmean([m["n_forward_batches"] for m in metrics])) if metrics else 0
    )
    row["output_bytes"] = output_bytes
    row["peak_host_rss_bytes"] = max((m["host_peak_rss_bytes"] for m in metrics), default=0)
    row["max_memory_allocated"] = max((m["max_memory_allocated"] for m in metrics), default=0)
    row["max_memory_reserved"] = max((m["max_memory_reserved"] for m in metrics), default=0)
    median_wall = float(aggregate["median_s"])
    row["throughput_gpx_s"] = round(executed_px / median_wall / 1e9, 6) if median_wall else None
    row["per_repeat"] = metrics

    # claim-bearing closure: n must equal the configured repeat count
    if aggregate["n"] != max(1, repeats):
        row["status"] = "FAILED"
        row["reason"] = (
            f"repeat metric produced n={aggregate['n']} for repeat_count={repeats} "
            "(stale/missing metric key)"
        )
    elif not witness["correctness_witness_pass"]:
        row["status"] = "FAILED_CORRECTNESS"
        row["reason"] = "bounded dense parity witness failed"
    else:
        row["status"] = "SUCCESS"
    return row


def torch_cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


# ---- workspace artifact family -----------------------------------------------------


def build_scale_family_in_workspace(
    *,
    workspace_dir: Path,
    family_dir: Path,
    identity: str,
    measurement_commit: str,
    tracked_tree_clean: bool,
    provisional: bool,
    source_hashes: dict[str, str],
    env_lock: dict[str, Any],
    env_lock_sha: str,
    run_config_payload: dict[str, Any],
    fixture_manifest_payload: dict[str, Any],
    lane_rows: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Write the 8-member scale family into the workspace — fail-closed:
    formal runs with blockers write NOTHING (provisional runs always
    keep their workspace rows for development evidence)."""
    from openlithohub.benchmark.industrial_scale import (
        RUN_CONFIG_SCHEMA,
        SCALE_CANONICAL_FAMILY,
        SCHEMA_NAME,
        ScaleRunConfig,
        formal_scale_blockers,
        write_strict_json,
    )

    run_config = ScaleRunConfig(
        **{
            **run_config_payload,
            "lanes": tuple(run_config_payload["lanes"]),
            "window_sizes": tuple(run_config_payload["window_sizes"]),
        }
    )
    blockers: list[str] = []
    if not provisional:
        blockers = formal_scale_blockers(
            env_lock=env_lock,
            run_config=run_config,
            lane_rows={lane: (rows[0] if rows else {}) for lane, rows in lane_rows.items() if rows},
            git_clean=tracked_tree_clean,
            provisional=provisional,
        )
        if not tracked_tree_clean:
            blockers.append("dirty tracked tree cannot build the scale family (B2-A)")
        if blockers:
            return blockers

    write_strict_json(
        family_dir / "industrial-scale-run-config.json",
        {
            "schema": RUN_CONFIG_SCHEMA,
            "measurement_commit": measurement_commit,
            "tracked_tree_clean": tracked_tree_clean,
            "provisional": provisional,
            "source_hashes": source_hashes,
            "environment_lock_sha256": env_lock_sha,
            "run_identity": identity,
            "run_config": run_config_payload,
        },
    )
    write_strict_json(
        family_dir / "industrial-scale-environment.json",
        {"environment_lock": env_lock, "environment_lock_sha256": env_lock_sha},
    )
    write_strict_json(
        family_dir / "industrial-scale-distribution-freeze.txt",
        {"gpu": env_lock, "environment_lock_sha256": env_lock_sha},
    )
    write_strict_json(
        family_dir / "industrial-scale-fixture.json",
        {
            "schema": "OpenLithoHub.industrial-scale-benchmark.v1",
            "run_identity": identity,
            "fixture": fixture_manifest_payload,
        },
    )
    lane_summary: dict[str, dict[str, Any]] = {}
    for lane, rows in lane_rows.items():
        successes = [r for r in rows if r.get("status") == "SUCCESS"]
        lane_summary[lane] = {
            "schema": SCHEMA_NAME,
            "run_identity": identity,
            "lane": lane,
            "rows": rows,
            "repeats_recorded": sum(int(r.get("repeat_count") or 0) for r in rows),
            "correctness_witness_pass": bool(rows)
            and all(bool(r.get("correctness_witness_pass")) for r in successes),
            "headline_eligible": bool(rows)
            and all(bool(r.get("headline_eligible")) for r in successes),
        }
    write_strict_json(family_dir / "industrial-scale-index.json", lane_summary.get(LANE_A, {}))
    write_strict_json(family_dir / "industrial-scale-runtime.json", lane_summary.get(LANE_B, {}))

    members = sorted(SCALE_CANONICAL_FAMILY - {"manifest.json", "SHA256SUMS.txt"})
    write_strict_json(
        family_dir / "manifest.json",
        {
            "run_identity": identity,
            "members": [
                {"name": name, "bytes": (family_dir / name).stat().st_size} for name in members
            ],
        },
    )
    lines = [
        f"{sha256_file(family_dir / name)}  {name}"
        for name in sorted(SCALE_CANONICAL_FAMILY - {"SHA256SUMS.txt"})
    ]
    (family_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    return []


# ---- driver ------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-manifest", required=True)
    parser.add_argument("--gds", required=True, help="prepared GDS matching the manifest")
    parser.add_argument("--lanes", default="a", help="comma list: a,b")
    parser.add_argument("--windows", default="4096,8192")
    parser.add_argument("--full-die", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-backend", default="cpu-worker-emulation")
    parser.add_argument("--gpu-count", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--forward-profile", default="P1_FINITE_SUPPORT")
    parser.add_argument("--sink", default="memmap_npy", choices=("memmap_npy", "manhattan"))
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--halo", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--pixel-nm", type=float, default=None)
    parser.add_argument("--out-root", default="benchmarks/results/industrial-scale")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]

    from openlithohub.benchmark.industrial_scale import (
        LANE_A,
        LANE_B,
        ScaleRunConfig,
        compute_scale_run_identity,
        load_scale_fixture_manifest,
        scale_environment_lock,
        scale_environment_lock_sha256,
        write_strict_json,
    )

    manifest = load_scale_fixture_manifest(args.fixture_manifest)
    gds_sha = sha256_file(args.gds)
    if gds_sha != manifest.gds_sha256:
        raise SystemExit(
            f"--gds sha256 {gds_sha!r} != manifest {manifest.gds_sha256!r} — "
            "the prepared fixture bytes must match the frozen manifest"
        )
    pixel_nm = args.pixel_nm if args.pixel_nm is not None else manifest.pixel_nm

    commit, clean = git_measurement_commit(repo)
    source_hashes = build_source_hashes()
    env_lock = scale_environment_lock(requested_gpu_count=args.gpu_count)
    env_lock_sha = scale_environment_lock_sha256(env_lock)

    lane_names = {
        "a": LANE_A,
        "b": LANE_B,
    }
    lanes = tuple(lane_names[lane.strip()] for lane in args.lanes.split(",") if lane.strip())
    windows = tuple(int(w) for w in args.windows.split(",") if w.strip())
    run_config = ScaleRunConfig(
        lanes=lanes,
        device_backend=args.device_backend,
        gpu_count=args.gpu_count,
        worker_count=args.worker_count,
        tile_size=args.tile,
        halo_px=args.halo,
        microbatch=args.microbatch,
        queue_depth=args.queue_depth,
        forward_profile=args.forward_profile,
        sink_kind=args.sink,
        warmup_count=args.warmup,
        repeat_count=args.repeats,
        window_sizes=windows,
        fixture_manifest_sha256=sha256_file(args.fixture_manifest),
        fixture_gds_sha256=manifest.gds_sha256,
        selected_layer=manifest.selected_layer,
        pixel_nm=pixel_nm,
    )
    identity = compute_scale_run_identity(
        run_config,
        measurement_commit=commit,
        harness_sha256=source_hashes["harness"],
        core_sha256=source_hashes["core"],
        verifier_sha256=source_hashes["verifier"],
        claim_generator_sha256=source_hashes["claim_generator"],
        fixture_manifest_sha256=run_config.fixture_manifest_sha256,
        environment_lock_sha256=env_lock_sha,
    )
    provisional = not args.formal
    workspace = Path(args.out_root) / "runs" / identity
    workspace.mkdir(parents=True, exist_ok=True)

    lane_rows: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lanes}
    for lane in lanes:
        for window in windows:
            row = run_window_row(
                gds=args.gds,
                layer=manifest.selected_layer,
                pixel_nm=pixel_nm,
                window=window,
                full_die=args.full_die,
                die_px=manifest.die_size_px,
                profile=args.forward_profile,
                radius=4,
                sigma_nm=1.6,
                sink_kind=args.sink,
                tile=args.tile,
                halo=args.halo,
                microbatch=args.microbatch,
                queue_depth=args.queue_depth,
                device=args.device,
                worker_count=args.worker_count,
                gpu_count=args.gpu_count,
                lane=lane,
                warmup=args.warmup,
                repeats=args.repeats,
                workspace=workspace,
            )
            lane_rows[lane].append(row)
            write_strict_json(workspace / f"row-{lane}-{window}.json", row)

    run_config_payload = run_config.to_payload()
    blockers = build_scale_family_in_workspace(
        workspace_dir=workspace,
        family_dir=workspace / "family",
        identity=identity,
        measurement_commit=commit,
        tracked_tree_clean=clean,
        provisional=provisional,
        source_hashes=source_hashes,
        env_lock=env_lock,
        env_lock_sha=env_lock_sha,
        run_config_payload=run_config_payload,
        fixture_manifest_payload=manifest.to_payload(),
        lane_rows=lane_rows,
    )
    summary = {
        "run_identity": identity,
        "workspace": str(workspace),
        "measurement_commit": commit,
        "tracked_tree_clean": clean,
        "provisional": provisional,
        "status": "PROVISIONAL CPU DEVELOPMENT DRY RUN / NOT PERFORMANCE AUTHORITY"
        if provisional
        else "FORMAL SCALE RUN (canonical promotion via verifier)",
        "lanes": {
            lane: [{"window": r.get("window"), "status": r.get("status")} for r in rows]
            for lane, rows in lane_rows.items()
        },
        "family_blockers": blockers,
    }
    write_strict_json(workspace / "run-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if blockers:
        print("FORMAL RUN: canonical family NOT built — blockers above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
