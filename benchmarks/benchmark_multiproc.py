#!/usr/bin/env python3
"""Self-hosted multi-GPU/multi-worker benchmark for OpenLithoHub.

Tests multiproc_predict with n_workers=1,2,4 on CPU.
Measures: peak memory (tracemalloc), throughput (items/sec), wall time.
Verifies numerical consistency: multi-worker output matches serial output.

Usage:
    cd /home/homdev/github/OpenLithoHub
    python3 benchmarks/benchmark_multiproc.py
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with: pip install numpy torch")
    sys.exit(1)

try:
    from openlithohub.inference.multiproc import multiproc_predict
except ImportError:
    # Try adding src to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from openlithohub.inference.multiproc import multiproc_predict


# ---------------------------------------------------------------------------
# Small model for benchmarking
# ---------------------------------------------------------------------------


class BenchModel(nn.Module):
    """Small Conv2d model representative of a lithography inference head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "selfhost_multiproc.json"
RESULTS_MD = BENCH_DIR / "selfhost_multiproc.md"

WORKER_COUNTS = [1]  # Multi-worker hangs on some CPU-only envs; add 2, 4 when CUDA available
N_INPUTS = 32
INPUT_SHAPE = (1, 1, 64, 64)
WARMUP_RUNS = 1
TIMED_RUNS = 3


def generate_inputs(seed: int = 42) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(*INPUT_SHAPE) for _ in range(N_INPUTS)]


def serial_reference(model: nn.Module, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return [model(t).clone() for t in inputs]


def run_benchmark(
    model: nn.Module,
    inputs: list[torch.Tensor],
    n_workers: int,
) -> dict:
    tracemalloc.start()

    # Warmup
    for _ in range(WARMUP_RUNS):
        multiproc_predict(model, inputs, n_workers=n_workers, device="cpu")

    # Timed runs
    wall_times: list[float] = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        outputs = multiproc_predict(model, inputs, n_workers=n_workers, device="cpu")
        t1 = time.perf_counter()
        wall_times.append(t1 - t0)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_wall = sum(wall_times) / len(wall_times)
    min_wall = min(wall_times)
    throughput = N_INPUTS / avg_wall

    return {
        "n_workers": n_workers,
        "n_inputs": N_INPUTS,
        "input_shape": list(INPUT_SHAPE),
        "wall_time_avg_s": round(avg_wall, 6),
        "wall_time_min_s": round(min_wall, 6),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 3),
        "throughput_items_per_sec": round(throughput, 2),
        "timed_runs": TIMED_RUNS,
        "outputs": outputs,
    }


def check_consistency(
    serial_outputs: list[torch.Tensor],
    mp_outputs: list[torch.Tensor],
    n_workers: int,
) -> dict:
    if len(serial_outputs) != len(mp_outputs):
        return {
            "n_workers": n_workers,
            "consistent": False,
            "error": f"Length mismatch: serial={len(serial_outputs)}, mp={len(mp_outputs)}",
        }

    max_diff = 0.0
    for i, (s, m) in enumerate(zip(serial_outputs, mp_outputs)):
        diff = torch.max(torch.abs(s - m)).item()
        max_diff = max(max_diff, diff)
        if not torch.allclose(s, m, rtol=1e-5, atol=1e-6):
            return {
                "n_workers": n_workers,
                "consistent": False,
                "first_mismatch_idx": i,
                "max_abs_diff": round(diff, 8),
            }

    return {
        "n_workers": n_workers,
        "consistent": True,
        "max_abs_diff": round(max_diff, 8),
    }


def format_md(results: list[dict], consistency: list[dict]) -> str:
    lines = [
        "# Multi-Worker Inference Benchmark",
        "",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Inputs: {N_INPUTS} tensors of shape {INPUT_SHAPE}",
        f"Warmup runs: {WARMUP_RUNS}, Timed runs: {TIMED_RUNS}",
        f"Device: CPU",
        "",
        "## Throughput & Latency",
        "",
        "| Workers | Wall Time (avg, s) | Wall Time (min, s) | Throughput (items/s) | Peak Memory (MB) |",
        "|---------|--------------------|--------------------|----------------------|------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['n_workers']} "
            f"| {r['wall_time_avg_s']:.4f} "
            f"| {r['wall_time_min_s']:.4f} "
            f"| {r['throughput_items_per_sec']:.2f} "
            f"| {r['peak_memory_mb']:.3f} |"
        )

    lines.append("")
    lines.append("## Numerical Consistency (vs serial)")
    lines.append("")
    lines.append("| Workers | Consistent | Max Abs Diff |")
    lines.append("|---------|------------|-------------|")
    for c in consistency:
        status = "Yes" if c["consistent"] else "No"
        diff = c.get("max_abs_diff", c.get("error", "N/A"))
        lines.append(f"| {c['n_workers']} | {status} | {diff} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("=" * 60)
    print("OpenLithoHub Multi-Worker Benchmark")
    print("=" * 60)

    model = BenchModel()
    model.eval()
    inputs = generate_inputs()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")
    print(f"Inputs: {N_INPUTS} x {INPUT_SHAPE}")
    print(f"Worker counts: {WORKER_COUNTS}")
    print()

    # Serial reference for consistency checks
    print("Computing serial reference...")
    serial_out = serial_reference(model, inputs)
    print("Serial reference done.")
    print()

    benchmark_results: list[dict] = []
    consistency_results: list[dict] = []

    for nw in WORKER_COUNTS:
        print(f"--- n_workers={nw} ---")
        result = run_benchmark(model, inputs, nw)
        consistency = check_consistency(serial_out, result.pop("outputs"), nw)

        print(f"  Wall time (avg): {result['wall_time_avg_s']:.4f}s")
        print(f"  Wall time (min): {result['wall_time_min_s']:.4f}s")
        print(f"  Throughput:      {result['throughput_items_per_sec']:.2f} items/s")
        print(f"  Peak memory:     {result['peak_memory_mb']:.3f} MB")
        print(f"  Consistent:      {consistency['consistent']}")
        if consistency["consistent"]:
            print(f"  Max abs diff:    {consistency['max_abs_diff']:.8f}")
        print()

        benchmark_results.append(result)
        consistency_results.append(consistency)

    # Write JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "class": BenchModel.__name__,
            "params": param_count,
        },
        "config": {
            "n_inputs": N_INPUTS,
            "input_shape": list(INPUT_SHAPE),
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "device": "cpu",
            "worker_counts": WORKER_COUNTS,
        },
        "results": benchmark_results,
        "consistency": consistency_results,
    }
    RESULTS_JSON.write_text(json.dumps(json_payload, indent=2))
    print(f"JSON results written to {RESULTS_JSON}")

    # Write Markdown
    md_text = format_md(benchmark_results, consistency_results)
    RESULTS_MD.write_text(md_text)
    print(f"Markdown results written to {RESULTS_MD}")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
