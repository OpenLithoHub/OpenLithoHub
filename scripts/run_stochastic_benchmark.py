#!/usr/bin/env python3
"""Multi-scenario closed-loop stochastic benchmark for BayesianStochasticModel.

Runs BayesianStochasticModel (Poisson-MC mode) across:
- Multiple pattern types (line/space, contact array, elbow, dense random)
- Multiple EUV dose conditions (10, 30, 60, 100 photons/nm²)
- Multiple process nodes (EUV N3, N7, ArF 45nm)
- Cross-validates against compute_stochastic_robustness and
  compute_stochastic_defect_classes

Outputs a benchmark table as markdown suitable for README / BENCHMARKS.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import torch

from openlithohub.benchmark.metrics.stochastic import (
    compute_stochastic_defect_classes,
    compute_stochastic_robustness,
)
from openlithohub.models.bayesian_stochastic import (
    BayesianStochasticModel,
    generate_synthetic_ground_truth,
)

# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------


def line_space(size: int = 64, pitch: int = 8) -> torch.Tensor:
    mask = torch.zeros(size, size)
    for x in range(0, size, pitch):
        mask[:, x : x + pitch // 2] = 1.0
    return mask


def contact_array(size: int = 64, pitch: int = 16, radius: int = 4) -> torch.Tensor:
    mask = torch.zeros(size, size)
    cy, cx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for y0 in range(pitch // 2, size, pitch):
        for x0 in range(pitch // 2, size, pitch):
            dist = ((cy - y0).float().pow(2) + (cx - x0).float().pow(2)).sqrt()
            mask[dist <= radius] = 1.0
    return mask


def elbow(size: int = 64, width: int = 4) -> torch.Tensor:
    mask = torch.zeros(size, size)
    mid = size // 2
    mask[mid - width : mid + width, :mid] = 1.0
    mask[mid:, mid - width : mid + width] = 1.0
    return mask


def dense_random(size: int = 64, fill: float = 0.4, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise = torch.rand(size, size, generator=gen)
    binary = (noise < fill).float()
    # Morphological close to make connected features
    from openlithohub._utils.morphology import binary_dilation

    closed = binary_dilation(binary, radius=2)
    return (closed > 0.5).float()


PATTERNS = {
    "line/space": line_space,
    "contact array": contact_array,
    "elbow": elbow,
    "dense random": dense_random,
}

# ---------------------------------------------------------------------------
# Node presets (wavelength, NA, sigma_px heuristic)
# ---------------------------------------------------------------------------

NODE_PRESETS: dict[str, dict[str, Any]] = {
    "EUV N3": {
        "dose_photons_per_nm2": 30.0,
        "sigma_px": 1.5,
        "pixel_size_nm": 1.0,
    },
    "EUV N7": {
        "dose_photons_per_nm2": 40.0,
        "sigma_px": 2.0,
        "pixel_size_nm": 1.0,
    },
    "ArF 45nm": {
        "dose_photons_per_nm2": 60.0,
        "sigma_px": 3.0,
        "pixel_size_nm": 1.0,
    },
}

DOSE_LEVELS = [10.0, 30.0, 60.0, 100.0]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRow:
    pattern: str
    node: str
    dose_photons_nm2: float
    mean_failure_prob: float
    max_failure_prob: float
    mean_ler_nm: float
    max_ler_nm: float
    robustness_score: float
    bridge_prob: float
    break_prob: float
    defect_total_cm2: float
    calibration_mae: float
    wall_time_s: float


def run_single(
    pattern_name: str,
    mask: torch.Tensor,
    node_name: str,
    dose: float,
    sigma_px: float,
    pixel_size_nm: float,
    n_mc: int = 64,
) -> BenchmarkRow:
    model = BayesianStochasticModel(
        n_mc_samples=n_mc,
        mode="poisson",
        dose_photons_per_nm2=dose,
        sigma_px=sigma_px,
        pixel_size_nm=pixel_size_nm,
        seed=0,
    )

    t0 = time.perf_counter()
    result = model.predict(mask)
    elapsed = time.perf_counter() - t0

    fp = result.metadata["failure_prob"]
    ler = result.metadata["ler_nm"]

    # Cross-validate with existing stochastic metrics
    robust = compute_stochastic_robustness(
        mask,
        num_trials=n_mc,
        dose_photons_per_nm2=dose,
        pixel_size_nm=pixel_size_nm,
        seed=0,
    )

    defects = compute_stochastic_defect_classes(
        mask,
        num_trials=n_mc,
        dose_photons_per_nm2=dose,
        pixel_size_nm=pixel_size_nm,
        seed=0,
    )

    # Calibration: compare model failure_prob vs large ground-truth MC
    gt = generate_synthetic_ground_truth(
        mask,
        n_mc=256,
        dose_photons_per_nm2=dose,
        pixel_size_nm=pixel_size_nm,
        sigma_px=sigma_px,
        seed=0,
    )
    calibration_mae = (fp - gt["failure_prob"]).abs().mean().item()

    return BenchmarkRow(
        pattern=pattern_name,
        node=node_name,
        dose_photons_nm2=dose,
        mean_failure_prob=fp.mean().item(),
        max_failure_prob=fp.max().item(),
        mean_ler_nm=ler.mean().item(),
        max_ler_nm=ler.max().item(),
        robustness_score=robust["robustness_score"],
        bridge_prob=robust["bridge_probability"],
        break_prob=robust["break_probability"],
        defect_total_cm2=defects.total_per_cm2,
        calibration_mae=calibration_mae,
        wall_time_s=elapsed,
    )


def run_full_benchmark() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    total = len(PATTERNS) * len(NODE_PRESETS)
    i = 0
    for pat_name, pat_fn in PATTERNS.items():
        mask = pat_fn()
        for node_name, preset in NODE_PRESETS.items():
            i += 1
            print(f"[{i}/{total}] {pat_name} @ {node_name}", flush=True)
            row = run_single(
                pat_name,
                mask,
                node_name,
                dose=preset["dose_photons_per_nm2"],
                sigma_px=preset["sigma_px"],
                pixel_size_nm=preset["pixel_size_nm"],
            )
            rows.append(row)
    return rows


def run_dose_sweep() -> list[BenchmarkRow]:
    """Sweep dose on line/space at EUV N7 to show monotonic response."""
    mask = line_space()
    rows: list[BenchmarkRow] = []
    for dose in DOSE_LEVELS:
        print(f"  dose sweep: {dose} ph/nm²", flush=True)
        row = run_single(
            "line/space",
            mask,
            "EUV N7",
            dose=dose,
            sigma_px=2.0,
            pixel_size_nm=1.0,
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def rows_to_markdown(
    rows: list[BenchmarkRow],
    title: str = "Stochastic Benchmark",
) -> str:
    lines = [
        f"## {title}",
        "",
        "Generated with `python scripts/run_stochastic_benchmark.py`. "
        "All runs use `BayesianStochasticModel(mode='poisson')` with "
        "`n_mc_samples=64`, cross-validated against "
        "`compute_stochastic_robustness` (bridge/break) and "
        "`compute_stochastic_defect_classes` (imec per-class rates). "
        "Calibration MAE = mean |predicted_failure_prob − ground_truth| "
        "from 256-sample Poisson MC.",
        "",
        "| Pattern | Node | Dose (ph/nm²) | Mean FP | Max FP | "
        "Mean LER (nm) | Robustness | Bridge P | Break P | "
        "Defects (cm⁻²) | Cal. MAE | Time (s) |",
        "|---------|------|---------------|---------|--------|"
        "--------------|------------|----------|---------|"
        "---------------|----------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.pattern} | {r.node} | {r.dose_photons_nm2:.0f} "
            f"| {r.mean_failure_prob:.4f} | {r.max_failure_prob:.4f} "
            f"| {r.mean_ler_nm:.2f} | {r.robustness_score:.4f} "
            f"| {r.bridge_prob:.4f} | {r.break_prob:.4f} "
            f"| {r.defect_total_cm2:.1f} "
            f"| {r.calibration_mae:.4f} | {r.wall_time_s:.2f} |"
        )
    return "\n".join(lines)


def rows_to_json(rows: list[BenchmarkRow]) -> str:
    data = []
    for r in rows:
        data.append(
            {
                "pattern": r.pattern,
                "node": r.node,
                "dose_photons_nm2": r.dose_photons_nm2,
                "mean_failure_prob": r.mean_failure_prob,
                "max_failure_prob": r.max_failure_prob,
                "mean_ler_nm": r.mean_ler_nm,
                "max_ler_nm": r.max_ler_nm,
                "robustness_score": r.robustness_score,
                "bridge_probability": r.bridge_prob,
                "break_probability": r.break_prob,
                "defect_total_per_cm2": r.defect_total_cm2,
                "calibration_mae": r.calibration_mae,
                "wall_time_s": r.wall_time_s,
            }
        )
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    print("=== Multi-pattern × Multi-node benchmark ===")
    rows = run_full_benchmark()

    print("\n=== Dose sweep (line/space @ EUV N7) ===")
    dose_rows = run_dose_sweep()

    all_rows = rows + dose_rows

    md = rows_to_markdown(rows, "Cross-Pattern Stochastic Benchmark") + "\n\n"
    md += rows_to_markdown(dose_rows, "Dose-Response Sweep (line/space, EUV N7)")

    with open("benchmark_stochastic_results.md", "w") as f:
        f.write(md)
    print("\nResults written to benchmark_stochastic_results.md")

    jdata = rows_to_json(all_rows)
    with open("benchmark_stochastic_results.json", "w") as f:
        f.write(jdata)
    print("JSON data written to benchmark_stochastic_results.json")
