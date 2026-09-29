"""Industrial Benchmark v2 formal-measurement preflight (PR-G §34).

Reports every formal blocker for a planned v2 measurement run and exits
nonzero when formal publication would be impossible.  On a CPU-only host
this exits nonzero with::

    FORMAL_PUBLICATION_BLOCKED: CUDA measurement environment unavailable

which is the documented, honest terminal state of Phase 2A — never a
benchmark failure and never emulated.

Usage::

    python scripts/preflight_industrial_v2.py --gds /path/to/ibex.gds --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_v2_harness() -> object:
    """Import the v2 harness module by path.  The Tier C probe MUST
    construct ``HopkinsParams`` through the worker's own constructor
    (GPU Authority Repair §6: no preflight-only code path that can drift
    away from the benchmark worker)."""
    harness_path = REPO / "benchmarks" / "industrial-v2" / "run_v2_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_v2_benchmark_harness", harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the v2 harness at {harness_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tier_b_preflight_probe(device: str) -> tuple[bool, str]:
    """Bounded Tier B probe through the SHARED worker-path helpers
    (GPU Authority Repair §6): CUDA initialization → peak-stat reset (the
    exact ordering that failed as issue #56 defect 1) → one actual
    finite-support CUDA forward → input/output device witness."""
    import torch

    from openlithohub.benchmark.industrial_v2 import BatchedFiniteSupportBlur
    from openlithohub.benchmark.measurement_support import (
        ForwardExecutionWitness,
        initialize_cuda_measurement_device,
    )

    initialize_cuda_measurement_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    blur = BatchedFiniteSupportBlur(radius=4, sigma=1.6)
    blur.to(device)
    tile = torch.rand((64, 64), dtype=torch.float32)
    gpu_in = tile.to(device)
    out = blur.window_forward(gpu_in)
    witness = ForwardExecutionWitness(device)
    witness.record(gpu_in, out)
    summary = witness.summary()
    ok = bool(summary["cuda_execution_witness_pass"] and torch.isfinite(out).all())
    detail = (
        f"input={summary['forward_input_device']} "
        f"output_before_d2h={summary['forward_output_device_before_d2h']}"
    )
    return ok, detail


def tier_c_preflight_probe(device: str) -> tuple[bool, str]:
    """Bounded Tier C probe through the SHARED worker-path helpers
    (GPU Authority Repair §6): the EXACT worker ``HopkinsParams``
    constructor (issue #56 defect 2) + one bounded CUDA Hopkins
    execution with a finite output."""
    import torch

    from openlithohub._utils.hopkins import simulate_aerial_image_hopkins

    harness = load_v2_harness()
    params = harness.tier_c_hopkins_params(
        {
            "hopkins_wavelength_nm": 13.5,
            "hopkins_na": 0.33,
            "hopkins_sigma_outer": 0.9,
            "hopkins_sigma_inner": 0.6,
            "hopkins_defocus_nm": 0.0,
            "pixel_nm": 1.0,
        }
    )
    grid = 128
    mask = torch.ones((grid, grid), device=device)
    aerial = simulate_aerial_image_hopkins(mask, params=params)
    torch.cuda.synchronize(device)
    ok = bool(torch.isfinite(aerial).all() and str(aerial.device).startswith("cuda"))
    detail = f"grid={grid} sigma=({params.sigma_inner},{params.sigma}) device={aerial.device}"
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", required=True, help="real routed GDS fixture path")
    parser.add_argument("--device", default="cuda:0", help="planned device (cuda:0)")
    parser.add_argument("--tiers", default="a,b,c")
    args = parser.parse_args()

    import torch

    from openlithohub.server.config import resolve_execution_device

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # git state — GPU Authority Repair §8: the shared cross-platform
    # helper (shutil.which + fixed argv); no hard-coded /usr/bin/git and
    # no Windows operator shim.
    import subprocess

    try:
        from openlithohub.benchmark.measurement_support import measurement_git_state

        commit, clean = measurement_git_state(REPO)
        check("measurement commit", True, commit)
        check("tracked tree clean", clean, "dirty files block formal runs (B2-A)")
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        check("git state", False, str(exc)[-200:])

    # fixture
    gds = Path(args.gds)
    check("fixture exists", gds.is_file(), str(gds))
    if gds.is_file():
        from openlithohub.benchmark.industrial_v2 import sha256_file

        check("fixture sha256", True, sha256_file(gds))

    # CUDA environment
    cuda_available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if cuda_available else 0
    check("CUDA available", cuda_available, f"count={device_count}")
    try:
        selected = resolve_execution_device(
            args.device, cuda_available=cuda_available, device_count=device_count
        )
        check("selected device", True, selected)
    except ValueError as exc:
        selected = "unavailable"
        check("selected device", False, str(exc))
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        check("GPU model", True, props.name)
        check("VRAM bytes", True, str(int(props.total_memory)))
        check("compute capability", True, f"{props.major}.{props.minor}")
        check("torch CUDA build", True, torch.version.cuda or "none")
        cudnn = torch.backends.cudnn.version()
        check("cuDNN version", cudnn is not None, str(cudnn))
    check("dtype fp32 support", True, "fp32 is always supported")
    # 2B.1-I/§10 + GPU Authority Repair §6: bounded probes through the
    # SAME shared helpers as the real workers — a failed probe blocks the
    # formal run BEFORE expensive Tier A/B work starts, and the preflight
    # can never pass while a real worker path is broken.
    if cuda_available:
        try:
            ok, detail = tier_b_preflight_probe(args.device)
            check("Tier B actual-CUDA forward probe", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            check("Tier B actual-CUDA forward probe", False, repr(exc)[:200])
        try:
            ok, detail = tier_c_preflight_probe(args.device)
            check("Tier C GPU Hopkins smoke", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            check("Tier C GPU Hopkins smoke", False, repr(exc)[:200])
    check(
        "TF32 policy",
        True,
        f"matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32}",
    )

    # disk + writability.  16 GiB free disk is the AUTHORITATIVE documented
    # host requirement (docs/industrial-v2-measurement.md): the 32768² fp32
    # window alone spans ~4 GiB per tensor and the formal run keeps fixture,
    # workspaces and results — the preflight enforces exactly that value.
    free = shutil.disk_usage(REPO).free
    check("disk free >= 16 GiB", free >= 16 * 1024**3, f"{free} bytes free")
    probe = REPO / "benchmarks" / "results" / "industrial-v2" / ".preflight-probe"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        check("v2 results root writable", True, str(probe.parent))
    except OSError as exc:
        check("v2 results root writable", False, str(exc))

    # tooling
    check("KLayout available", importlib.util.find_spec("klayout") is not None)
    check("benchmark capabilities", True, "finite-support blur + hopkins sim present")

    cuda_required = any(tier.strip() in ("b", "c") for tier in args.tiers.split(","))
    formal_blockers = [name for name, ok, _ in checks if not ok]
    if cuda_required and not cuda_available:
        # exact machine-readable marker (§16): automation keys on this text;
        # printed once by the summary loop below.
        formal_blockers.append(
            "FORMAL_PUBLICATION_BLOCKED: CUDA measurement environment unavailable"
        )

    print("== Industrial Benchmark v2 preflight ==")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    for blocker in formal_blockers:
        if blocker.startswith("FORMAL"):
            print(f"  {blocker}")
    if formal_blockers:
        print(
            f"PREFLIGHT: FAIL — {len(formal_blockers)} formal blocker(s); "
            "canonical v2 publication is blocked"
        )
        return 1
    print("PREFLIGHT: PASS — formal v2 measurement may proceed on this host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
