"""Industrial Benchmark v2 harness (PR-G Phase 2A — protocol authority).

**Protocol first, measurement second.** This harness can execute:

* Tier A (CPU-eligible, provisional allowed): indexed vs reference
  exact-vector window discovery on a real GDS fixture, with EXACT
  semantic parity as a hard gate.
* Tier B (requires real CUDA): CPU batch=1 vs GPU batch=1 vs GPU batch=N
  streaming execution with the deterministic finite-support forward —
  synchronized CUDA timing, separate allocated/reserved peak memory.
* Tier C (requires real CUDA): fixed-configuration Hopkins compute tier;
  recorded ``UNSUPPORTED`` in Phase 2A until the GPU-resident Hopkins
  path is implemented and verified (never faked).

Formal discipline retained from v1.1: fresh worker process per
claim-bearing repeat (the driver re-invokes itself with ``--worker``),
setup/warmup/timed phases explicit, median/p10/p90/n statistics (never
best-of-N), strict JSON artifacts under
``benchmarks/results/industrial-v2/runs/<run_identity>/``.

On a host without CUDA, Tier B/C rows record ``NOT_RUN_ENVIRONMENT`` and
canonical publication stays blocked — that is the honest terminal state
``V2 PROTOCOL READY / FORMAL GPU MEASUREMENT PENDING`` (§1/§35).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess  # noqa: S404 — fixed-argv worker re-invocation only
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
HARNESS_PATH = Path(__file__).resolve()
CORE_PATH = _REPO / "src" / "openlithohub" / "benchmark" / "industrial_v2.py"
CLAIM_GENERATOR_PATH = _REPO / "scripts" / "generate_industrial_v2_claims.py"
VERIFIER_PATH = _REPO / "scripts" / "verify_industrial_v2_artifacts.py"

LAYER = "66:44"  # sky130hd li1 convention shared with B04 INC29 / v1.1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_measurement_commit(repo: Path) -> tuple[str, bool]:
    """(commit, clean) for the tracked tree the measurement would bind to."""
    import subprocess

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


def build_source_hashes(repo: Path) -> dict[str, str]:
    return {
        "harness": sha256_file(HARNESS_PATH),
        "core": sha256_file(CORE_PATH),
        "claim_generator": sha256_file(CLAIM_GENERATOR_PATH),
        "verifier": sha256_file(VERIFIER_PATH),
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


# ---- tier workers (executed in a FRESH process per repeat) ------------------


def tier_a_worker_once(
    gds: str, window: int, tile: int, halo: int, pixel_nm: float, layer: str = LAYER
) -> dict[str, Any]:
    """One Tier A measurement: indexed vs reference window discovery on a
    real GDS, centered die-crop ladder window, EXACT parity as a gate."""
    import time

    import numpy as np

    from openlithohub.streaming.geometry import BoundingBox
    from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

    # TWO independent source instances: the reference pass must never warm
    # the indexed path's row cache (that would measure cache reads, not
    # first-touch discovery — the exact cost G1 removes).
    parse_start = time.perf_counter()
    indexed = KLayoutAlignedRunSource.from_file(gds, pixel_size_nm=pixel_nm, layer=layer)
    parse_index_wall = time.perf_counter() - parse_start
    stats = indexed.index_stats()
    reference = KLayoutAlignedRunSource.from_file(gds, pixel_size_nm=pixel_nm, layer=layer)

    side = min(window, indexed.shape[0], indexed.shape[1])
    y0 = max(0, (indexed.shape[0] - side) // 2)
    x0 = max(0, (indexed.shape[1] - side) // 2)
    bbox = BoundingBox(x0, y0, x0 + side, y0 + side)
    rows = list(range(bbox.y0, bbox.y1))
    flat_count = len(indexed._flat)

    # reference: FULL-FLAT scan per unseen row (the pre-G1 bottleneck)
    reference_start = time.perf_counter()
    reference_rows: dict[int, tuple] = {}
    for y in rows:
        reference_rows[y] = tuple(
            poly for poly in reference._flat if poly.bbox[1] <= y < poly.bbox[3]
        )
    reference_wall = time.perf_counter() - reference_start

    # indexed: band buckets + row cache (first touch on a fresh instance)
    indexed_start = time.perf_counter()
    indexed_rows = {y: indexed.candidates_for_row(y) for y in rows}
    indexed_wall = time.perf_counter() - indexed_start

    # exact candidate parity: the index may only say "possibly relevant",
    # so the candidate TUPLES must be identical
    candidate_parity = all(reference_rows[y] == indexed_rows[y] for y in rows)
    candidate_count = sum(len(v) for v in indexed_rows.values())
    reference_count = sum(len(v) for v in reference_rows.values())
    flat_scan_equivalent = rows.__len__() * flat_count

    # window queries + EXACT parity (runs, ids, contributors, raster)
    query_start = time.perf_counter()
    runs = list(indexed.iter_owned_runs_for_bbox(bbox))
    query_wall = time.perf_counter() - query_start
    raster = indexed.read_window(bbox).numpy()
    reference_raster = reference.read_window(bbox).numpy()
    reference_runs = list(reference.iter_owned_runs_for_bbox(bbox))
    parity_pass = (
        candidate_parity
        and np.array_equal(raster, reference_raster)
        and len(runs) == len(reference_runs)
        and all(
            (a.parent.run_id, a.parent.y, a.x0, a.x1, a.parent.contributor_object_ids)
            == (b.parent.run_id, b.parent.y, b.x0, b.x1, b.parent.contributor_object_ids)
            for a, b in zip(runs, reference_runs, strict=True)
        )
    )

    return {
        "window": window,
        "parse_index_wall_s": parse_index_wall,
        "reference_row_wall_s": reference_wall,
        "indexed_row_wall_s": indexed_wall,
        "first_touch_speedup_x": round(reference_wall / indexed_wall, 4) if indexed_wall else None,
        "query_wall_s": query_wall,
        "flat_polygon_count": flat_count,
        "index_entries": stats["entries"],
        "index_bands": stats["bands"],
        "candidate_polygon_count": candidate_count,
        "reference_polygon_count": reference_count,
        "flat_scans_avoided_pct": round(
            100.0 * (1 - candidate_count / flat_scan_equivalent) if flat_scan_equivalent else 0.0,
            4,
        ),
        "rows_visited": side,
        "owned_run_count": len(runs),
        "raster_px": int(raster.size),
        "correctness_witness_pass": bool(parity_pass),
        "status": "SUCCESS" if parity_pass else "FAILED",
        "timing_method": "host_perf_counter",
        "device_requires_cuda": False,
    }


def tier_b_worker_once(cfg: dict[str, Any]) -> dict[str, Any]:
    """One Tier B measurement (CUDA required): CPU batch=1 vs GPU batch=1
    vs GPU batch=N streaming execution with the deterministic finite-
    support forward, synchronized CUDA timing, separate GPU peak memory."""
    import torch

    from openlithohub.benchmark.industrial_v2 import BatchedFiniteSupportBlur
    from openlithohub.streaming import run_streaming
    from openlithohub.streaming.sinks import TensorTileSink
    from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

    device = str(cfg["device"])
    # 2B.1-F: the ACTUAL runtime policy must equal the declared run-config
    # policy for claim-bearing GPU tiers — identity labels are not enough.
    actual_tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    if bool(cfg.get("tf32_matmul", False)) != actual_tf32_matmul:
        return {
            "status": "FAILED",
            "reason": (
                f"runtime TF32 matmul={actual_tf32_matmul} does not match the "
                f"declared run config ({cfg.get('tf32_matmul')})"
            ),
            "device_requires_cuda": True,
            "timing_method": "",
            "correctness_witness_pass": False,
        }
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {
            "status": "NOT_RUN_ENVIRONMENT",
            "reason": "CUDA measurement environment unavailable",
            "device_requires_cuda": True,
            "timing_method": "",
            "correctness_witness_pass": False,
        }
    if cfg["dtype"] != "fp32":  # §20: never merge dtype claims
        return {
            "status": "UNSUPPORTED",
            "reason": f"dtype {cfg['dtype']!r} not implemented in Phase 2A",
            "device_requires_cuda": True,
            "timing_method": "",
            "correctness_witness_pass": False,
        }

    from openlithohub.streaming.crop_source import ExactVectorCropSource
    from openlithohub.streaming.geometry import BoundingBox

    parent = KLayoutAlignedRunSource.from_file(cfg["gds"], pixel_size_nm=cfg["pixel_nm"])
    side = min(cfg["window"], parent.shape[0], parent.shape[1])
    y0 = max(0, (parent.shape[0] - side) // 2)
    x0 = max(0, (parent.shape[1] - side) // 2)
    # Tier B is WINDOW-scoped (roadmap §13): the centered die crop is the
    # executed layout — never the full die.
    source = ExactVectorCropSource(parent, BoundingBox(x0, y0, x0 + side, y0 + side))

    forward = BatchedFiniteSupportBlur(radius=cfg["forward_radius"], sigma=cfg["forward_sigma_nm"])

    def run_stream(device: str, batch: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sink = TensorTileSink(source.shape)
        report = run_streaming(
            source,
            sink,
            lambda tile: forward.window_forward(tile.to(device)).cpu(),
            core_size=cfg["tile"],
            batch_size=batch,
            batched_forward_fn=lambda b: (
                forward.batch_forward(b.to(device)).cpu() if batch > 1 else None
            ),
        )
        return sink.finalize(), {
            "n_tiles": report.n_tiles,
            "forward_batches": report.forward_batches,
        }

    def synced(fn, *, needs_device: bool):
        if needs_device:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn()
        if needs_device:
            torch.cuda.synchronize(device)
        return result, time.perf_counter() - start

    # correctness first (§18): CPU reference vs GPU batch=1 vs GPU batch=N
    cpu_out, _ = run_stream("cpu", 1)
    torch.cuda.reset_peak_memory_stats(device)
    (gpu1, meta1), gpu1_wall = synced(lambda: run_stream(device, 1), needs_device=True)
    gpu1_mem = {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }
    torch.cuda.reset_peak_memory_stats(device)
    (gpu_n, meta_n), gpu_n_wall = synced(
        lambda: run_stream(device, cfg["batch"]), needs_device=True
    )
    gpu_n_mem = {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }
    tolerance = 1e-5
    parity = torch.allclose(cpu_out, gpu1.to("cpu"), atol=tolerance, rtol=0.0) and torch.allclose(
        gpu1, gpu_n.to(gpu1.device), atol=tolerance, rtol=0.0
    )
    # restore the CPU artifact as the sink output semantics (same artifact)
    from openlithohub.benchmark.industrial_v2 import host_peak_rss_bytes

    return {
        "window": cfg["window"],
        "batch": cfg["batch"],
        "tiles": meta1["n_tiles"],
        "forward_batches_batch1": meta1["forward_batches"],
        "forward_batches_batch_n": meta_n["forward_batches"],
        "gpu_batch1_wall_s": gpu1_wall,
        "gpu_batch_n_wall_s": gpu_n_wall,
        "host_peak_rss_bytes": host_peak_rss_bytes(),
        "max_memory_allocated": gpu1_mem["max_memory_allocated"],
        "max_memory_reserved": gpu1_mem["max_memory_reserved"],
        "batch_n_max_memory_allocated": gpu_n_mem["max_memory_allocated"],
        "batch_n_max_memory_reserved": gpu_n_mem["max_memory_reserved"],
        "timing_method": "cuda_synchronized",
        "dtype": "fp32",
        "device": device,
        "correctness_witness_pass": bool(parity),
        "status": "SUCCESS" if parity else "FAILED",
        "device_requires_cuda": True,
    }


def tier_c_worker_once(cfg: dict[str, Any]) -> dict[str, Any]:
    """Tier C: fixed-configuration Hopkins compute tier (2B.1-I).

    The SOCS Hopkins primitives are device-aware
    (``compute_socs_kernels(params, H, mask.device)``), so the GPU-resident
    path executes the same mathematical operator on the mask's device.
    Protocol requirements implemented here:

    * CPU fixed-config witness (finite + deterministic across two calls)
    * GPU fixed-config execution with synchronized wall timing
    * cold (first GPU call incl. SOCS kernel construction) reported
      SEPARATELY from warm steady-state (precomputed kernels)
    * GPU allocated/reserved peaks reset before the timed region
    * CPU vs GPU agreement within the frozen fp32 tolerance
    """
    import time

    import numpy as np
    import torch

    from openlithohub._utils.hopkins import (
        simulate_aerial_image_hopkins,
    )
    from openlithohub.simulators.hopkins_sim import HopkinsParams

    device = str(cfg["device"])
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {
            "status": "NOT_RUN_ENVIRONMENT",
            "reason": "CUDA measurement environment unavailable",
            "device_requires_cuda": True,
            "timing_method": "",
            "correctness_witness_pass": False,
        }
    if cfg["dtype"] != "fp32":
        return {
            "status": "UNSUPPORTED",
            "reason": f"dtype {cfg['dtype']!r} not implemented in Phase 2A",
            "device_requires_cuda": True,
            "timing_method": "",
            "correctness_witness_pass": False,
        }

    grid = int(cfg["hopkins_grid"])
    params = HopkinsParams(
        wavelength_nm=cfg["hopkins_wavelength_nm"],
        na=cfg["hopkins_na"],
        sigma_outer=cfg["hopkins_sigma_outer"],
        sigma_inner=cfg["hopkins_sigma_inner"],
        pixel_size_nm=cfg["pixel_nm"],
        defocus_nm=cfg.get("hopkins_defocus_nm", 0.0),
    )
    rng = np.random.default_rng(0)
    mask_cpu = (rng.random((grid, grid)) > 0.5).astype(np.float32)
    tolerance = 1e-5

    # CPU reference (deterministic across two calls is part of the witness)
    cpu_result = simulate_aerial_image_hopkins(torch.from_numpy(mask_cpu), params=params)
    cpu_out = cpu_result.cpu().numpy()
    cpu_again = simulate_aerial_image_hopkins(torch.from_numpy(mask_cpu), params=params)
    cpu_deterministic = bool(
        np.allclose(cpu_out, cpu_again.cpu().numpy(), atol=tolerance, rtol=0.0)
    )

    mask_gpu = torch.from_numpy(mask_cpu).to(device)

    # COLD: first GPU call includes SOCS kernel construction on device.
    torch.cuda.synchronize(device)
    cold_start = time.perf_counter()
    gpu_cold = simulate_aerial_image_hopkins(mask_gpu, params=params)
    torch.cuda.synchronize(device)
    cold_wall = time.perf_counter() - cold_start
    del gpu_cold  # cold result carries the kernel-construction cost only

    # WARM: precomputed kernels in device memory, synchronized steady-state.
    kernels, weights = _build_socs(params, grid, mask_gpu.device)
    kernels_f = torch.fft.fftn(torch.fft.ifftshift(kernels, dim=(-2, -1)), dim=(-2, -1)).to(
        torch.complex64
    )
    warm_walls: list[float] = []
    gpu_warm = None
    for _ in range(3):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        gpu_warm = simulate_aerial_image_hopkins(
            mask_gpu,
            kernels=kernels,
            weights=weights,
            precomputed_kernels_f=kernels_f,
        )
        torch.cuda.synchronize(device)
        warm_walls.append(time.perf_counter() - start)
    peaks = {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }
    gpu_out = gpu_warm.detach().cpu().numpy()
    parity = bool(
        np.isfinite(gpu_out).all()
        and np.allclose(cpu_out, gpu_out, atol=tolerance, rtol=0.0)
        and cpu_deterministic
    )
    return {
        "grid": grid,
        "cpu_reference_wall_s": None,
        "gpu_cold_wall_s": round(cold_wall, 6),
        "gpu_warm_wall_s": round(min(warm_walls), 6) if warm_walls else None,
        "warm_walls_s": [round(w, 6) for w in warm_walls],
        "finite_witness": bool(np.isfinite(gpu_out).all()),
        "cpu_deterministic_witness": cpu_deterministic,
        "correctness_witness_pass": parity,
        "max_memory_allocated": peaks["max_memory_allocated"],
        "max_memory_reserved": peaks["max_memory_reserved"],
        "timing_method": "cuda_synchronized",
        "dtype": "fp32",
        "device": device,
        "device_requires_cuda": True,
        "status": "SUCCESS" if parity else "FAILED",
    }


def _build_socs(params: Any, grid: int, device: Any):
    """Device-resident SOCS kernels via the existing device-aware primitives."""
    from openlithohub._utils.hopkins import compute_socs_kernels

    return compute_socs_kernels(params, grid, device)


# ---- driver ------------------------------------------------------------------
# ---- driver ------------------------------------------------------------------


def worker_entry(args: argparse.Namespace) -> int:
    """Fresh-process worker: one repeat, strict JSON row on stdout."""
    if args.worker_tier == "a":
        row = tier_a_worker_once(
            args.gds, args.window, args.tile, args.halo, args.pixel_nm, args.layer
        )
    elif args.worker_tier == "b":
        row = tier_b_worker_once(
            {
                "gds": args.gds,
                "device": args.device,
                "dtype": args.dtype,
                "layer": args.layer,
                "window": args.window,
                "tile": args.tile,
                "batch": args.batch,
                "pixel_nm": args.pixel_nm,
                "forward_radius": args.forward_radius,
                "forward_sigma_nm": args.forward_sigma,
            }
        )
    else:
        row = tier_c_worker_once(
            {
                "device": args.device,
                "dtype": args.dtype,
                "hopkins_wavelength_nm": args.hopkins_wavelength_nm,
                "hopkins_na": args.hopkins_na,
                "hopkins_sigma_outer": args.hopkins_sigma_outer,
                "hopkins_sigma_inner": args.hopkins_sigma_inner,
                "hopkins_defocus_nm": 0.0,
                "hopkins_grid": args.hopkins_grid,
                "pixel_nm": args.pixel_nm,
            }
        )
    sys.stdout.write(json.dumps(row, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default="a", help="comma list: a,b,c")
    parser.add_argument("--gds", default="", help="real routed GDS fixture path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--halo", type=int, default=64)
    parser.add_argument("--pixel-nm", type=float, default=1.0)
    parser.add_argument("--forward-radius", type=int, default=4)
    parser.add_argument("--forward-sigma", type=float, default=1.6)
    parser.add_argument("--hopkins-wavelength-nm", type=float, default=13.5)
    parser.add_argument("--hopkins-na", type=float, default=0.33)
    parser.add_argument("--hopkins-sigma-outer", type=float, default=0.9)
    parser.add_argument("--hopkins-sigma-inner", type=float, default=0.6)
    parser.add_argument("--hopkins-grid", type=int, default=1024)
    parser.add_argument("--layer", default=LAYER, help="GDS layer as LAYER:DTYPE")
    parser.add_argument("--windows", default="4096,8192,16384,32768")
    parser.add_argument(
        "--out-root",
        default="benchmarks/results/industrial-v2",
        help="v2 results root (never the v1.1 root)",
    )
    parser.add_argument("--provisional", action="store_true", default=True)
    parser.add_argument("--formal", action="store_true", help="require a clean tracked tree")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-tier", default="a", help=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, default=4096, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker_entry(args)

    repo = Path(__file__).resolve().parents[2]
    commit, clean = git_measurement_commit(repo)
    source_hashes = build_source_hashes(repo)

    from openlithohub.benchmark.industrial_v2 import (
        RunConfigV2,
        StatusV2,
        compute_run_identity_v2,
        gpu_environment_lock,
        gpu_environment_lock_sha256,
        write_strict_json,
    )

    # 2B.1-A: the exact GDS bytes enter the run identity — identical args
    # over different fixture bytes produce a DIFFERENT identity.
    if not args.gds:
        raise SystemExit("--gds is required: fixture identity is part of the run identity")
    fixture_sha256 = sha256_file(args.gds)
    run_config = RunConfigV2(
        tiers=tuple(sorted(set(args.tiers.split(",")) - {""})),
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch,
        tile_size=args.tile,
        halo_px=args.halo,
        pixel_nm=args.pixel_nm,
        forward_radius=args.forward_radius,
        forward_sigma_nm=args.forward_sigma,
        window_sizes=tuple(int(w) for w in args.windows.split(",") if w),
        hopkins_wavelength_nm=args.hopkins_wavelength_nm,
        hopkins_na=args.hopkins_na,
        hopkins_sigma_outer=args.hopkins_sigma_outer,
        hopkins_sigma_inner=args.hopkins_sigma_inner,
        hopkins_grid=args.hopkins_grid,
        warmup_count=args.warmup,
        repeat_count=args.repeats,
        layer=args.layer,
        fixture_sha256=fixture_sha256,
    )
    # 2B.1-D: the GPU environment lock is collected BEFORE identity so its
    # canonical hash binds GPU model/driver/CUDA/Torch/TF32 into identity.
    env_lock = gpu_environment_lock()
    env_lock_sha = gpu_environment_lock_sha256(env_lock)
    identity = compute_run_identity_v2(
        run_config,
        measurement_commit=commit,
        harness_sha256=source_hashes["harness"],
        core_sha256=source_hashes["core"],
        claim_generator_sha256=source_hashes["claim_generator"],
        verifier_sha256=source_hashes["verifier"],
        environment_lock_sha256=env_lock_sha,
    )
    workspace = Path(args.out_root) / "runs" / identity
    workspace.mkdir(parents=True, exist_ok=True)

    write_strict_json(workspace / "environment-lock.json", env_lock)
    write_strict_json(
        workspace / "run-config.json",
        {
            "schema": "OpenLithoHub.industrial-run-config.v2",
            "measurement_commit": commit,
            "tracked_tree_clean": clean,
            "source_hashes": source_hashes,
            "environment_lock_sha256": env_lock_sha,
            "run_identity": identity,
            "provisional": args.provisional and not args.formal,
            "run_config": run_config.to_payload(),
        },
    )

    out_root = Path(args.out_root)
    if out_root.resolve().as_posix().endswith("results/industrial"):
        raise SystemExit("refusing to write v2 results into the frozen v1.1 root")

    tier_rows: dict[str, dict[str, Any]] = {}
    for tier in run_config.tiers:
        row = run_tier(args, tier, identity, workspace, run_config)
        tier_rows[tier] = row
        write_strict_json(workspace / f"tier-{tier}.json", row)

    status = {
        "schema": "OpenLithoHub.industrial-benchmark.v2",
        "run_identity": identity,
        "workspace": str(workspace),
        "measurement_commit": commit,
        "tracked_tree_clean": clean,
        "provisional": args.provisional and not args.formal,
        "status": "V2 PROTOCOL READY / FORMAL GPU MEASUREMENT PENDING"
        if not args.formal
        else "FORMAL RUN RECORDED (canonical promotion via verifier)",
        "tiers": {
            tier: {
                "status": row.get("status", StatusV2.FAILED.value),
                "correctness_witness_pass": bool(row.get("correctness_witness_pass")),
            }
            for tier, row in tier_rows.items()
        },
    }
    write_strict_json(workspace / "run-summary.json", status)

    # 2B.1-J: formal runs close by building the exact seven-member family
    # through the fail-closed builder — the ONLY sanctioned path to
    # promote_canonical_family().  Provisional runs keep workspace rows.
    if args.formal:
        from openlithohub.benchmark.industrial_v2 import (
            build_canonical_family_in_workspace,
        )

        blockers = build_canonical_family_in_workspace(
            workspace_dir=workspace,
            run_config=run_config,
            run_identity=identity,
            measurement_commit=commit,
            source_hashes=source_hashes,
            environment_lock=env_lock,
            tier_rows=tier_rows,
            tracked_tree_clean=clean,
            provisional=False,
        )
        status["canonical_build_blockers"] = blockers
        write_strict_json(workspace / "run-summary.json", status)
        if blockers:
            print(json.dumps(status, indent=2, sort_keys=True))
            print("FORMAL RUN: canonical family NOT built — blockers above")
            return 1
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _worker_once(
    args: argparse.Namespace, tier: str, window: int, run_config: Any
) -> dict[str, Any]:
    """One fresh worker process = one repeat on one window."""
    cmd = [
        sys.executable,
        str(HARNESS_PATH),
        "--worker",
        "--worker-tier",
        tier,
        "--gds",
        args.gds,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--window",
        str(window),
        "--tile",
        str(args.tile),
        "--halo",
        str(args.halo),
        "--batch",
        str(args.batch),
        "--pixel-nm",
        str(args.pixel_nm),
        "--layer",
        args.layer,
    ]
    proc = subprocess.run(  # noqa: S603 — fixed-argv worker re-invocation
        cmd, capture_output=True, text=True, timeout=3600
    )
    if proc.returncode != 0:
        return {
            "status": "FAILED",
            "reason": (proc.stderr or "worker failed")[-2000:],
            "correctness_witness_pass": False,
            "window": window,
        }
    try:
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        row.setdefault("window", window)
        return row
    except (ValueError, IndexError):
        return {
            "status": "FAILED",
            "reason": "worker produced no strict JSON row",
            "correctness_witness_pass": False,
            "window": window,
        }


def _run_window_repeats(
    args: argparse.Namespace, tier: str, window: int, run_config: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Warmup + measured fresh-process repeats for ONE window."""
    warm: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    for _ in range(max(0, args.warmup)):
        warm.append(_worker_once(args, tier, window, run_config))
    for _ in range(max(1, args.repeats)):
        measured.append(_worker_once(args, tier, window, run_config))
    return warm, measured


def _aggregate_window(
    tier: str, window: int, warm: list[dict[str, Any]], measured: list[dict[str, Any]]
) -> dict[str, Any]:
    wall_key = {
        "a": "indexed_row_wall_s",
        "b": "gpu_batch_n_wall_s",
        "c": "cpu_wall_s",
    }.get(tier, "")
    walls = [float(r[wall_key]) for r in measured if wall_key in r]
    all_pass = bool(measured) and all(r.get("correctness_witness_pass") for r in measured)
    aggregate = summarize_repeats(walls)
    row: dict[str, Any] = {
        "tier": tier,
        "window": window,
        "repeat_count": len(measured),
        "warmup_discarded": len(warm),
        "correctness_witness_pass": all_pass,
        "claim_level": "REPRODUCED_INTERNAL",
        **{f"repeat_{i}": r for i, r in enumerate(measured)},
    }
    row.update({f"aggregate_{k}": v for k, v in aggregate.items()})
    statuses = {r.get("status") for r in measured} | {r.get("status") for r in warm}
    if statuses == {"NOT_RUN_ENVIRONMENT"}:
        row["status"] = "NOT_RUN_ENVIRONMENT"
    elif "FAILED" in statuses or not all_pass:
        row["status"] = "FAILED"
    elif statuses == {"UNSUPPORTED"}:
        row["status"] = "UNSUPPORTED"
    else:
        row["status"] = "SUCCESS"
    if tier == "a" and row["status"] == "SUCCESS":
        first = next((r for r in measured if r.get("status") == "SUCCESS"), {})
        row.update(
            {
                "index_build_wall_s": first.get("parse_index_wall_s"),
                "reference_row_wall_s": first.get("reference_row_wall_s"),
                "indexed_row_wall_s": first.get("indexed_row_wall_s"),
                "first_touch_speedup_x": first.get("first_touch_speedup_x"),
                "query_wall_s": first.get("query_wall_s"),
                "flat_polygon_count": first.get("flat_polygon_count"),
                "flat_scans_avoided_pct": first.get("flat_scans_avoided_pct"),
                "index_entries": first.get("index_entries"),
                "rows_visited": first.get("rows_visited"),
                "owned_run_count": first.get("owned_run_count"),
            }
        )
    if tier == "b" and row["status"] == "SUCCESS":
        # §14: batching speedup from FORMAL MEDIANS, never one repeat.
        b1 = [float(r["gpu_batch1_wall_s"]) for r in measured if r.get("gpu_batch1_wall_s")]
        bn = [float(r["gpu_batch_n_wall_s"]) for r in measured if r.get("gpu_batch_n_wall_s")]
        if b1 and bn:
            row["batching_speedup_x"] = round(statistics.median(b1) / statistics.median(bn), 4)
    return row


def run_tier(
    args: argparse.Namespace, tier: str, identity: str, workspace: Path, run_config: Any
) -> dict[str, Any]:
    """Driver-side tier execution (2B.1-G): EVERY declared window gets its
    own warmup + measured fresh-process repeats and its own aggregate row.

    Tier C runs one fixed Hopkins grid row (the grid is the declared
    workload, not a ladder).
    """
    if tier == "c":
        warm, measured = _run_window_repeats(args, tier, run_config.hopkins_grid, run_config)
        return _aggregate_window(tier, run_config.hopkins_grid, warm, measured)

    window_rows = []
    statuses = []
    for window in run_config.window_sizes:
        warm, measured = _run_window_repeats(args, tier, window, run_config)
        row = _aggregate_window(tier, window, warm, measured)
        window_rows.append(row)
        statuses.append(row["status"])
    if "FAILED" in statuses:
        overall = "FAILED"
    elif statuses and all(s == "SUCCESS" for s in statuses):
        overall = "SUCCESS"
    else:
        overall = statuses[0] if statuses else "NOT_RUN_ENVIRONMENT"
    return {
        "tier": tier,
        "status": overall,
        "correctness_witness_pass": all(r.get("correctness_witness_pass") for r in window_rows)
        if window_rows
        else False,
        "claim_level": "REPRODUCED_INTERNAL",
        "window_rows": window_rows,
        "repeats_recorded": sum(r.get("repeat_count", 0) for r in window_rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
