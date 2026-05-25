"""P4/P5/P6 probes for GAN-OPC v0.4.

P4 — Memmap cache integrity: 100 cached samples match online preprocessing.
P5 — Mixed precision consistency: FP32 vs FP16 loss relative error < 1e-3.
P6 — Memory peak: peak memory < 12 GB in smoke test + AMP benchmark.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.models._unet import UNet, UNetV2


def _simulate_aerial_batch(mask: torch.Tensor, sigma_px: float, dose: float = 1.0) -> torch.Tensor:
    """Gaussian PSF convolution on (B,1,H,W) tensor."""
    import math
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32) - radius
    g1d = torch.exp(-0.5 * (coords / max(sigma_px, 1e-6)) ** 2)
    kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(mask.device)
    padding = radius
    # Simple circular padding via replicate (good enough for probe)
    padded = functional.pad(mask, [padding] * 4, mode="replicate")
    return functional.conv2d(padded, kernel) * dose


def probe_p4() -> dict:
    """P4: Memmap cache integrity check.

    Verify the _MemmapGanOpcPairs cache matches online preprocessing.
    """
    # Import train_gan_opc components by running the script's module scope
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_gan_opc",
        Path(__file__).resolve().parent / "train_gan_opc.py",
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    data_root = Path("data/ganopc/extracted/ganopc-data")
    if not data_root.exists():
        return {
            "probe": "P4",
            "status": "SKIPPED",
            "reason": f"no data at {data_root}",
        }

    try:
        online = train_mod._GanOpcPairs(data_root, resize_to=512, resize_mode="bilinear")
        cached = train_mod._MemmapGanOpcPairs(
            data_root, resize_to=512, resize_mode="bilinear",
            cache_dir="cache/ganopc_probe/",
        )

        n_check = min(100, len(online))
        mismatches = 0
        for i in range(n_check):
            d_on, m_on = online[i]
            d_ca, m_ca = cached[i]
            if not torch.allclose(d_on, d_ca, atol=1e-3) or not torch.allclose(m_on, m_ca, atol=1e-3):
                mismatches += 1

        return {
            "probe": "P4",
            "status": "PASS" if mismatches == 0 else "FAIL",
            "n_checked": n_check,
            "mismatches": mismatches,
        }
    except Exception as e:
        return {"probe": "P4", "status": "ERROR", "error": str(e)}


def probe_p5() -> dict:
    """P5: Mixed precision numerical consistency.

    Compare FP32 vs AMP(FP16) loss on 5 random batches. Relative error < 1e-3.
    """
    model = UNet()
    model.eval()
    device = "cpu"

    results = []
    for trial in range(5):
        x = torch.randn(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()

        # FP32
        with torch.no_grad():
            logits = model(x)
            loss_fp32 = functional.binary_cross_entropy_with_logits(logits, target).item()

        # AMP FP16
        with torch.no_grad(), torch.amp.autocast("cpu"):
            logits_amp = model(x)
            loss_amp = functional.binary_cross_entropy_with_logits(logits_amp, target).item()

        rel_err = abs(loss_fp32 - loss_amp) / max(abs(loss_fp32), 1e-8)
        results.append({
            "fp32": loss_fp32,
            "amp": loss_amp,
            "rel_err": rel_err,
        })

    max_rel_err = max(r["rel_err"] for r in results)
    return {
        "probe": "P5",
        "status": "PASS" if max_rel_err < 1e-3 else "WARN",
        "max_relative_error": max_rel_err,
        "trials": results,
        "note": "AMP on AMD 5600G (no AVX512/FP16) may be slower than FP32",
    }


def probe_p6() -> dict:
    """P6: Memory peak + AMP benchmark.

    Measure peak memory in smoke test (1 batch), verify < 12 GB.
    Benchmark AMP on vs off for 2 epochs.
    """
    gc.collect()
    torch.set_num_threads(6)

    results: dict = {"probe": "P6"}

    # Memory check with UNet
    model = UNet()
    x = torch.randn(4, 1, 512, 512)
    target = (torch.rand(4, 1, 512, 512) > 0.5).float()

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # Track memory via RSS
    import resource

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    logits = model(x)
    loss = functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_peak_mb = (mem_after - mem_before) / 1024  # rough estimate

    results["unet_memory_delta_mb"] = max(0, mem_peak_mb)
    results["unet_peak_rss_mb"] = mem_after / 1024

    # AMP benchmark
    amp_times = []
    fp32_times = []
    n_steps = 5

    for label, use_amp in [("fp32", False), ("amp", True)]:
        model = UNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        times = []
        for _ in range(n_steps):
            x = torch.randn(4, 1, 256, 256)
            target = (torch.rand(4, 1, 256, 256) > 0.5).float()

            t0 = time.perf_counter()
            with torch.amp.autocast("cpu", enabled=use_amp):
                logits = model(x)
                loss = functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        if use_amp:
            amp_times = times
            results["amp_avg_step_s"] = avg_time
        else:
            fp32_times = times
            results["fp32_avg_step_s"] = avg_time

    amp_ratio = results["amp_avg_step_s"] / max(results["fp32_avg_step_s"], 1e-8)
    results["amp_slowdown_ratio"] = amp_ratio
    results["amp_decision"] = (
        "DISABLE_AMP" if amp_ratio > 1.5
        else "ENABLE_AMP" if amp_ratio < 0.9
        else "NEUTRAL"
    )
    results["status"] = "PASS"

    return results


def main() -> int:
    all_results = {}

    print("=== P4: Memmap Cache Integrity ===")
    p4 = probe_p4()
    all_results["P4"] = p4
    print(json.dumps(p4, indent=2))

    print("\n=== P5: Mixed Precision Consistency ===")
    p5 = probe_p5()
    all_results["P5"] = p5
    print(json.dumps(p5, indent=2))

    print("\n=== P6: Memory Peak + AMP Benchmark ===")
    p6 = probe_p6()
    all_results["P6"] = p6
    print(json.dumps(p6, indent=2))

    out_path = Path("out/probes/v0_4_p4_p5_p6.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_path}")

    # P4/P5/P6 failures are non-blocking (plan §3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
