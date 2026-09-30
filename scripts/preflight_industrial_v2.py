"""Industrial Benchmark v2 formal-measurement preflight (PR-G §34; v3 §14).

Reports every formal blocker for a planned v2 measurement run and exits
nonzero when formal publication would be impossible.  On a CPU-only host
this exits nonzero with::

    FORMAL_PUBLICATION_BLOCKED: CUDA measurement environment unavailable

which is the documented, honest terminal state of Phase 2A — never a
benchmark failure and never emulated.

GPU Authority Repair v3 §14: the preflight has TWO distinct Tier-C
gates, and a small-grid smoke NEVER implies formal-grid capacity again:

* Gate A — the ANALYTICAL FORMAL-GRID memory plan, computed from the
  actual frozen ``RunConfigV2`` grid (no full-H allocation), proving the
  formal problem is feasible under the frozen memory policy.
* Gate B — a BOUNDED real-CUDA block-path probe executing a
  representative H block through the EXACT worker implementation
  (generation → Gram update → finite checks → device witness), plus a
  small full bounded-SOCS execution as an end-to-end smoke.

It also spawns at least one fresh worker through the EXACT worker
mechanism and verifies the environment fingerprint matches this process
(v3 §22).

Usage::

    python scripts/preflight_industrial_v2.py --gds /path/to/ibex.gds --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess  # noqa: S404 — fixed-argv worker re-invocation only
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO / "benchmarks" / "industrial-v2" / "run_v2_benchmark.py"


def load_v2_harness() -> object:
    """Import the v2 harness module by path.  The Tier C probe MUST
    construct ``HopkinsParams`` through the worker's own constructor
    (GPU Authority Repair §6: no preflight-only code path that can drift
    away from the benchmark worker)."""
    spec = importlib.util.spec_from_file_location("run_v2_benchmark_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the v2 harness at {HARNESS_PATH}")
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


def frozen_tier_c_params(harness: object, hopkins_grid: int) -> tuple[object, int]:
    """The EXACT worker ``HopkinsParams`` (GPU Authority Repair §5) for the
    ACTUAL frozen formal grid — the preflight reasons about the same
    problem the workers will execute, never a stand-in grid."""
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
    return params, int(hopkins_grid)


def formal_memory_plan_gate(
    harness: object, device: str, hopkins_grid: int
) -> tuple[bool, str, object]:
    """Gate A (v3 §14A/§15): the analytical FORMAL-GRID memory plan.

    Uses the actual frozen grid's real dimensions (n_src from the real
    source sampling), computes the frozen-policy plan, prints the
    diagnostic block, and PASSES only when the plan is feasible BEFORE any
    allocation.  No full H is ever allocated here."""

    from openlithohub._utils.hopkins import socs_problem_dimensions
    from openlithohub._utils.socs_memory_plan import (
        collect_cuda_memory_facts,
        plan_socs_decomposition,
    )

    params, grid = frozen_tier_c_params(harness, hopkins_grid)
    dims = socs_problem_dimensions(params, grid, "cpu")
    facts = collect_cuda_memory_facts(device)
    plan = plan_socs_decomposition(
        grid_size=grid,
        n_src=dims.n_src,
        n_freq=dims.n_freq,
        K=dims.K,
        dtype="complex64",
        complex_bytes=8,
        device=device,
        facts=facts,
    )
    legacy_dense_bytes = dims.n_src * dims.n_freq * 8
    print("Tier C formal memory plan:")
    print(f"  strategy: {plan.strategy}")
    print(f"  grid: {grid}")
    print(f"  n_src: {dims.n_src}")
    print(f"  n_freq: {dims.n_freq}")
    print(f"  K: {dims.K}")
    print(f"  GPU total: {plan.device_total_bytes}")
    print(f"  GPU free at plan: {plan.device_free_bytes_at_plan}")
    print(f"  safety reserve: {plan.absolute_headroom_bytes + plan.fractional_headroom_bytes}")
    print(f"  workspace reserve: {plan.workspace_reserve_bytes}")
    print(f"  legacy dense-H bytes: {legacy_dense_bytes}")
    print("  legacy dense strategy: NOT A RUNTIME OPTION")
    print(f"  selected chunk columns: {plan.selected_chunk_columns}")
    print(f"  chunk count: {plan.chunk_count}")
    print(f"  estimated peak: {plan.estimated_peak_bytes}")
    print(f"  estimated headroom: {plan.estimated_headroom_bytes}")
    verdict = "PASS" if plan.memory_feasible else "FAIL"
    print(f"  FORMAL TIER-C MEMORY PLAN: {verdict}")
    detail = f"chunk={plan.selected_chunk_columns} plan={plan.plan_sha256[:16]}…"
    return plan.memory_feasible, detail, plan


def _formal_chunk_probe_child(
    q: Any, params: Any, grid: int, device: str, block_columns: int
) -> None:
    """Spawned-subprocess body (v3.1): run the PLANNED-CHUNK-SIZED block
    through the exact worker path and exit — the probe allocates the real
    chunk footprint, so it must not hold allocations in the preflight
    process afterwards."""
    try:
        from openlithohub._utils.hopkins import run_bounded_block_probe

        q.put(("ok", run_bounded_block_probe(params, grid, device, block_columns=block_columns)))
    except Exception as exc:  # noqa: BLE001 — probe failure is the diagnostic
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def formal_chunk_probe_gate(
    harness: object, device: str, hopkins_grid: int, plan: Any
) -> tuple[bool, str]:
    """Gate C (v3.1 closure): execute ONE block at the PLANNED chunk
    width through the exact worker block path, inside a short-lived
    spawned subprocess so the preflight process never retains the
    multi-GiB footprint.  This proves the analytical plan's actual chunk
    really fits and executes — not just that a small block does."""
    import multiprocessing

    params, grid = frozen_tier_c_params(harness, hopkins_grid)
    if not getattr(plan, "memory_feasible", False):
        return False, "plan infeasible — chunk probe not attempted (gate A must pass first)"
    ctx = multiprocessing.get_context("spawn")
    queue: Any = ctx.Queue()
    proc = ctx.Process(
        target=_formal_chunk_probe_child,
        args=(queue, params, grid, device, int(plan.selected_chunk_columns)),
    )
    proc.start()
    proc.join(timeout=1800)
    if proc.exitcode != 0 or queue.empty():
        return (
            False,
            f"formal chunk probe crashed (exit={proc.exitcode}) at "
            f"chunk={plan.selected_chunk_columns} columns",
        )
    kind, payload = queue.get()
    if kind != "ok":
        return False, f"formal chunk probe failed at chunk={plan.selected_chunk_columns}: {payload}"
    probe = payload
    ok = bool(probe.get("pass"))
    detail = (
        f"chunk={probe.get('block_columns')} n_src={probe.get('n_src')} "
        f"gram={probe.get('gram_update_device')} finite={probe.get('finite')}"
    )
    print(f"  FORMAL CHUNK CUDA PROBE: {'PASS' if ok else 'FAIL'} ({detail})")
    return ok, detail


def bounded_block_probe_gate(harness: object, device: str, hopkins_grid: int) -> tuple[bool, str]:
    """Gate B (v3 §14B): the bounded real-CUDA block-path probe — one
    representative H block through the EXACT worker implementation
    (bounded generation → CUDA Gram update → finite checks), then a small
    full bounded-SOCS execution as an end-to-end smoke with a device
    witness.  Same implementation as the worker, by construction."""
    import torch

    from openlithohub._utils.hopkins import (
        run_bounded_block_probe,
        simulate_aerial_image_hopkins,
    )
    from openlithohub.benchmark.measurement_support import (
        ForwardExecutionWitness,
        initialize_cuda_measurement_device,
    )

    params, grid = frozen_tier_c_params(harness, hopkins_grid)
    probe = run_bounded_block_probe(params, grid, device, block_columns=4096)
    if not bool(probe["pass"]):
        return False, f"formal-grid block probe failed: {probe}"

    smoke_grid = 64
    initialize_cuda_measurement_device(device)
    witness = ForwardExecutionWitness(device)
    mask = torch.ones((smoke_grid, smoke_grid), device=device)
    aerial = simulate_aerial_image_hopkins(mask, params=params)
    witness.record(mask, aerial)
    torch.cuda.synchronize(device)
    summary = witness.summary()
    ok = bool(
        torch.isfinite(aerial).all()
        and str(aerial.device).startswith("cuda")
        and summary["cuda_execution_witness_pass"]
    )
    detail = (
        f"formal-grid block columns={probe['block_columns']} gram={probe['gram_update_device']} | "
        f"smoke grid={smoke_grid} input={summary['forward_input_device']} "
        f"output_before_d2h={summary['forward_output_device_before_d2h']}"
    )
    return ok, detail


def worker_environment_probe(device: str, commit: str) -> tuple[bool, str]:
    """v3 §22: spawn at least ONE fresh worker through the EXACT
    fresh-worker mechanism and verify its environment fingerprint matches
    this parent process field-for-field."""
    from openlithohub.benchmark.measurement_support import (
        compare_worker_environment_fingerprints,
        worker_environment_fingerprint,
    )

    parent = worker_environment_fingerprint(commit)
    cmd = [
        sys.executable,
        str(HARNESS_PATH),
        "--worker",
        "--worker-tier",
        "envprobe",
        "--measurement-commit",
        commit,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return False, f"envprobe worker failed: {(proc.stderr or '')[-200:]}"
    try:
        child = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "envprobe worker produced no strict JSON row"
    ok, mismatches = compare_worker_environment_fingerprints(parent, child)
    return ok, "environment parity" if ok else f"mismatched: {mismatches[:6]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", required=True, help="real routed GDS fixture path")
    parser.add_argument("--device", default="cuda:0", help="planned device (cuda:0)")
    parser.add_argument("--tiers", default="a,b,c")
    parser.add_argument(
        "--hopkins-grid",
        type=int,
        default=1024,
        help="the ACTUAL frozen formal Tier-C grid (v3 §14 — never a stand-in)",
    )
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

    measurement_commit = ""
    try:
        from openlithohub.benchmark.measurement_support import measurement_git_state

        measurement_commit, clean = measurement_git_state(REPO)
        check("measurement commit", True, measurement_commit)
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

    # 2B.1-I/§10 + GPU Authority Repair §6/§14/§22: bounded probes through
    # the SAME shared helpers as the real workers — a failed probe blocks
    # the formal run BEFORE expensive Tier A/B work starts, and the
    # preflight can never pass while a real worker path is broken.
    if cuda_available:
        try:
            ok, detail = tier_b_preflight_probe(args.device)
            check("Tier B actual-CUDA forward probe", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            check("Tier B actual-CUDA forward probe", False, repr(exc)[:200])
        harness = load_v2_harness()
        # Gate A (v3 §14A): analytical FORMAL-GRID memory plan.
        try:
            ok, detail, tier_c_plan = formal_memory_plan_gate(
                harness, args.device, args.hopkins_grid
            )
            check("Tier C formal-grid memory plan (gate A)", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            tier_c_plan = None
            check("Tier C formal-grid memory plan (gate A)", False, repr(exc)[:200])
        try:
            ok, detail = formal_chunk_probe_gate(
                harness, args.device, args.hopkins_grid, tier_c_plan
            )
            check("Tier C formal-chunk CUDA probe (gate C, planned chunk)", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            check("Tier C formal-chunk CUDA probe (gate C, planned chunk)", False, repr(exc)[:200])
        # Gate B (v3 §14B): bounded real-CUDA block-path probe.
        try:
            ok, detail = bounded_block_probe_gate(harness, args.device, args.hopkins_grid)
            check("Tier C bounded CUDA block probe (gate B)", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            check("Tier C bounded CUDA block probe (gate B)", False, repr(exc)[:200])
        # v3 §22: fresh-worker environment parity through the exact mechanism.
        try:
            ok, detail = worker_environment_probe(args.device, measurement_commit)
            check("fresh-worker environment parity", ok, detail)
        except Exception as exc:  # noqa: BLE001 — capability probe
            check("fresh-worker environment parity", False, repr(exc)[:200])
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
    check("benchmark capabilities", True, "finite-support blur + bounded SOCS present")

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
