#!/usr/bin/env python3
"""Reproducible performance benchmark for OpenLithoHub.

Measures wall-clock timing for forward models, mask-optimization models,
and metric computation at multiple grid sizes. Follows the methodology from
the README "Performance & Benchmarks" section.

Usage:
    python scripts/benchmark_performance.py
    python scripts/benchmark_performance.py --json out.json --samples 200
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class TimingResult:
    benchmark: str
    grid_size: str
    samples: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    notes: str = ""


@dataclass
class EnvInfo:
    python: str
    os: str
    machine: str
    torch_version: str
    cuda_available: bool
    device_name: str


def get_env() -> EnvInfo:
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU"
    return EnvInfo(
        python=f"{platform.python_implementation()} {platform.python_version()}",
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        torch_version=torch.__version__,
        cuda_available=cuda,
        device_name=device,
    )


def measure(
    fn: Callable[[], Any],
    label: str,
    grid_label: str,
    warmup: int = 10,
    samples: int = 100,
) -> TimingResult:
    for _ in range(warmup):
        fn()

    gc.collect()
    gc.disable()
    timings: list[int] = []
    try:
        for _ in range(samples):
            start = time.perf_counter_ns()
            fn()
            timings.append(time.perf_counter_ns() - start)
    finally:
        gc.enable()

    timings.sort()
    median_ns = statistics.median(timings)
    q = statistics.quantiles(timings, n=100)
    p95_ns, p99_ns = q[94], q[98]
    min_ns = timings[0]

    return TimingResult(
        benchmark=label,
        grid_size=grid_label,
        samples=samples,
        median_ms=median_ns / 1e6,
        p95_ms=p95_ns / 1e6,
        p99_ms=p99_ns / 1e6,
        min_ms=min_ns / 1e6,
    )


def make_mask(size: int) -> torch.Tensor:
    m = torch.zeros(size, size)
    m[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
    return m


def bench_forward_models(sizes: list[int], samples: int) -> list[TimingResult]:
    from openlithohub._utils.forward_model import simulate_aerial_image

    results: list[TimingResult] = []
    for size in sizes:
        mask = make_mask(size)
        results.append(
            measure(
                lambda m=mask: simulate_aerial_image(m, sigma_px=2.0),
                "forward_gaussian",
                f"{size}x{size}",
                warmup=max(3, 20 - size // 64),
                samples=samples,
            )
        )
    return results


def bench_hopkins(sizes: list[int], samples: int) -> list[TimingResult]:
    try:
        from openlithohub._utils.hopkins import (
            HopkinsParams,
            simulate_aerial_image_hopkins,
        )
    except ImportError:
        return []

    results: list[TimingResult] = []
    for size in sizes:
        mask = make_mask(size)
        params = HopkinsParams(
            wavelength_nm=193.0,
            na=1.35,
            sigma=0.7,
            num_kernels=24,
            pixel_size_nm=8.0,
        )
        results.append(
            measure(
                lambda m=mask, p=params: simulate_aerial_image_hopkins(m, p),
                "forward_hopkins",
                f"{size}x{size}",
                warmup=max(2, 10 - size // 128),
                samples=min(samples, 50),
            )
        )
    return results


def bench_metrics(sizes: list[int], samples: int) -> list[TimingResult]:
    from openlithohub.benchmark.metrics.epe import compute_epe
    from openlithohub.benchmark.metrics.pvband import compute_pvband

    results: list[TimingResult] = []
    for size in sizes:
        pred = make_mask(size)
        target = make_mask(size)
        target[size // 4 + 2 : 3 * size // 4 + 2, size // 4 + 2 : 3 * size // 4 + 2] = 1.0

        results.append(
            measure(
                lambda p=pred, t=target: compute_epe(p, t, pixel_size_nm=8.0),
                "metric_epe",
                f"{size}x{size}",
                warmup=5,
                samples=samples,
            )
        )
        results.append(
            measure(
                lambda p=pred: compute_pvband(p, pixel_size_nm=8.0),
                "metric_pvband",
                f"{size}x{size}",
                warmup=3,
                samples=min(samples, 30),
            )
        )
    return results


def bench_models(samples: int) -> list[TimingResult]:
    import openlithohub.models.examples.dummy_model  # noqa: F401
    import openlithohub.models.levelset_ilt  # noqa: F401
    import openlithohub.models.neural_ilt  # noqa: F401
    import openlithohub.models.rule_based_opc  # noqa: F401
    from openlithohub.models.registry import registry  # noqa: F401

    design = make_mask(64)
    configs = [
        ("dummy-identity", {}),
        ("rule-based-opc", {}),
        ("levelset-ilt", {"iterations": 10}),
        ("levelset-ilt-200", {"iterations": 200}),
    ]

    results: list[TimingResult] = []
    for name, kwargs in configs:
        try:
            model = registry.get(name, **kwargs)
            model.setup()
        except Exception:  # noqa: S112
            continue
        r = measure(
            lambda m=model: m.predict(design),
            f"model_{name}",
            "64x64",
            warmup=2,
            samples=min(samples, 20),
        )
        model.teardown()
        results.append(r)
    return results


def fmt_ms(v: float) -> str:
    if v < 1.0:
        return f"{v * 1000:.0f} µs"
    if v < 1000.0:
        return f"{v:.1f} ms"
    return f"{v / 1000:.2f} s"


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenLithoHub performance benchmarks")
    ap.add_argument("--json", help="Export results to JSON file")
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()

    if args.samples < 2:
        ap.error("--samples must be >= 2 (percentile computation requires at least 2 data points)")

    env = get_env()
    print("=" * 65)
    print(f"  Python : {env.python}")
    print(f"  OS     : {env.os}")
    print(f"  Machine: {env.machine}")
    print(f"  PyTorch: {env.torch_version}")
    print(f"  Device : {env.device_name}")
    print("=" * 65)

    all_results: list[TimingResult] = []
    sizes = [64, 256]

    print("\n── Forward models ──")
    print(f"{'Benchmark':<30} {'Grid':<10} {'Median':>12} {'P95':>12} {'P99':>12}")
    print("-" * 76)

    for r in bench_forward_models(sizes, args.samples):
        all_results.append(r)
        print(f"{r.benchmark:<30} {r.grid_size:<10} {fmt_ms(r.median_ms):>12} "
              f"{fmt_ms(r.p95_ms):>12} {fmt_ms(r.p99_ms):>12}")

    for r in bench_hopkins(sizes, args.samples):
        all_results.append(r)
        print(f"{r.benchmark:<30} {r.grid_size:<10} {fmt_ms(r.median_ms):>12} "
              f"{fmt_ms(r.p95_ms):>12} {fmt_ms(r.p99_ms):>12}")

    print("\n── Metrics ──")
    print(f"{'Benchmark':<30} {'Grid':<10} {'Median':>12} {'P95':>12} {'P99':>12}")
    print("-" * 76)

    for r in bench_metrics(sizes, args.samples):
        all_results.append(r)
        print(f"{r.benchmark:<30} {r.grid_size:<10} {fmt_ms(r.median_ms):>12} "
              f"{fmt_ms(r.p95_ms):>12} {fmt_ms(r.p99_ms):>12}")

    print("\n── Models ──")
    print(f"{'Benchmark':<30} {'Grid':<10} {'Median':>12} {'P95':>12} {'P99':>12}")
    print("-" * 76)

    for r in bench_models(args.samples):
        all_results.append(r)
        print(f"{r.benchmark:<30} {r.grid_size:<10} {fmt_ms(r.median_ms):>12} "
              f"{fmt_ms(r.p95_ms):>12} {fmt_ms(r.p99_ms):>12}")

    if args.json:
        payload = {
            "environment": asdict(env),
            "results": [asdict(r) for r in all_results],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults exported to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
