"""Worker job functions for the Industrial Benchmark harness.

Extracted verbatim from the original single-file harness: every timed
worker re-invokes this module.  The physics scope is unchanged — the
ladder uses a deterministic 9x9 separable finite-support blur
(NOT_FOUNDRY_CALIBRATED), quality stages use built-in Hopkins SOCS with
identical parameters for every compared method.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess  # noqa: S404 - fixed-argv worker re-invocation only
import sys
import time
from pathlib import Path
from typing import Any

import torch

from openlithohub.benchmark.compliance.mrc import check_mrc
from openlithohub.benchmark.industrial import summarize
from openlithohub.benchmark.metrics.epe import compute_epe, compute_wafer_epe
from openlithohub.benchmark.metrics.l2_error import compute_l2_error
from openlithohub.benchmark.metrics.pvband import compute_pvband
from openlithohub.benchmark.metrics.shot_count import estimate_shot_count
from openlithohub.streaming.core_halo import plan_tile_requests
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.streaming.halo_policy import LegacyFixedHaloPolicy
from openlithohub.streaming.pipeline import run_streaming
from openlithohub.streaming.screening import ExactEmptyContextScreeningPolicy
from openlithohub.streaming.sinks import MetricOnlyTileSink, TensorTileSink
from openlithohub.streaming.sources import TensorTileSource
from openlithohub.streaming.vector_runs import KLayoutAlignedRunSource

FORWARD_RADIUS = 4
FORWARD_SUPPORT_PX = 2 * FORWARD_RADIUS + 1
LAYER = "66:44"  # sky130hd li1 layer convention used by B04 INC29
SEED = 0
MODELS = ["dummy-identity", "rule-based-opc", "levelset-ilt", "surrogate-ilt"]


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


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


def canonicalize_models(spec: str) -> list[str]:
    """Normalize a --models comma spec into the executed model list.

    Driver contract (audit P0.6): unknown models hard-fail, duplicates
    normalize to the first occurrence, empty entries hard-fail.  The
    result drives execution AND the run identity — a --models value that
    silently changed nothing would be a config-contract bug.
    """
    if not spec or not spec.strip():
        raise ValueError("--models is empty; at least one model is required")
    from openlithohub.models.registry import register_builtin_models, registry

    register_builtin_models()
    seen: dict[str, None] = {}
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            raise ValueError("--models contains an empty entry")
        if name not in registry.list_models():
            available = ", ".join(sorted(registry.list_models()))
            raise ValueError(f"unknown model {name!r}; available: [{available}]")
        seen.setdefault(name, None)
    return list(seen)


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
