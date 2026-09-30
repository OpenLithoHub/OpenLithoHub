"""SocsMemoryPlan hostile tests (GPU Authority Repair v3 §20).

The planner must be monotone under free-memory changes, aligned,
fail-closed BEFORE allocation, hash-bound, and worker-enforced.  CUDA
paths are exercised through injected :class:`DeviceMemoryFacts` and
monkeypatched CUDA queries so the whole matrix runs on CPU-only CI.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from openlithohub._utils.socs_memory_plan import (
    CHUNK_ALIGNMENT,
    MIN_CHUNK_COLUMNS,
    PEAK_WITNESS_TOLERANCE,
    DeviceMemoryFacts,
    MemoryPlanContractViolation,
    MemoryPlanHeadroomViolation,
    SocsMemoryPlan,
    assert_headroom,
    plan_socs_decomposition,
    validate_worker_plan,
)

GRID = 1024
N_SRC = 1609
N_FREQ = GRID * GRID
K = 24
TOTAL_16GIB = 16 * 1024**3


def facts(free_bytes: int) -> DeviceMemoryFacts:
    return DeviceMemoryFacts(
        device="cuda:0",
        device_total_bytes=TOTAL_16GIB,
        device_free_bytes=free_bytes,
        allocated_bytes=0,
        reserved_bytes=0,
    )


def plan_for(free_bytes: int) -> SocsMemoryPlan:
    return plan_socs_decomposition(
        grid_size=GRID,
        n_src=N_SRC,
        n_freq=N_FREQ,
        K=K,
        dtype="complex64",
        complex_bytes=8,
        device="cuda:0",
        facts=facts(free_bytes),
    )


# ---- monotonicity / alignment / bounds --------------------------------------------


def test_less_free_vram_never_increases_selected_chunk() -> None:
    big = plan_for(12 * 1024**3)
    small = plan_for(6 * 1024**3)
    assert big.memory_feasible and small.memory_feasible
    assert small.selected_chunk_columns <= big.selected_chunk_columns


def test_more_free_vram_never_decreases_selected_chunk_unless_capped() -> None:
    from openlithohub._utils.socs_memory_plan import MAX_OPERATIONAL_H_CHUNK_BYTES

    mid = plan_for(10 * 1024**3)
    huge = plan_for(15 * 1024**3)
    assert huge.selected_chunk_columns >= mid.selected_chunk_columns
    # both respect the frozen operational cap (§8)
    cap_columns = MAX_OPERATIONAL_H_CHUNK_BYTES // (N_SRC * 8)
    assert mid.selected_chunk_columns <= cap_columns
    assert huge.selected_chunk_columns <= cap_columns


def test_selected_chunk_is_aligned_and_within_n_freq() -> None:
    plan = plan_for(12 * 1024**3)
    assert plan.selected_chunk_columns % CHUNK_ALIGNMENT == 0
    assert plan.selected_chunk_columns <= N_FREQ


def test_estimated_peak_within_conservative_budget() -> None:
    free = 12 * 1024**3
    plan = plan_for(free)
    budget = free - plan.absolute_headroom_bytes - plan.fractional_headroom_bytes
    assert plan.estimated_peak_bytes <= budget
    assert plan.estimated_headroom_bytes >= 0


def test_plan_hash_changes_with_semantic_inputs() -> None:
    base = plan_for(12 * 1024**3)
    other_k = plan_socs_decomposition(
        grid_size=GRID,
        n_src=N_SRC,
        n_freq=N_FREQ,
        K=K + 1,
        dtype="complex64",
        complex_bytes=8,
        device="cuda:0",
        facts=facts(12 * 1024**3),
    )
    other_free = plan_for(11 * 1024**3)
    assert base.plan_sha256 != other_k.plan_sha256
    assert base.plan_sha256 != other_free.plan_sha256
    # identical inputs → identical hash
    again = plan_for(12 * 1024**3)
    assert base.plan_sha256 == again.plan_sha256


# ---- infeasibility is decided BEFORE allocation -------------------------------------


def test_minimum_legal_chunk_cannot_fit_is_infeasible_before_allocation() -> None:
    plan = plan_for(1 * 1024**3)
    assert not plan.memory_feasible
    assert plan.reason.startswith("MEMORY_PLAN_INFEASIBLE")
    assert plan.selected_chunk_columns == 0
    assert plan.chunk_count == 0


def test_minimum_legal_chunk_fits_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the frozen MIN_CHUNK_COLUMNS requirement via a tiny problem:
    a feasible plan passes with positive chunks."""
    plan = plan_for(12 * 1024**3)
    assert plan.memory_feasible
    assert plan.selected_chunk_columns >= MIN_CHUNK_COLUMNS


# ---- §11 runtime guard ---------------------------------------------------------------


def _patch_cuda(
    monkeypatch: pytest.MonkeyPatch,
    free_bytes: int,
    *,
    allocated_bytes: int = 0,
    reserved_bytes: int = 0,
) -> list[int]:
    calls: list[int] = []

    monkeypatch.setattr(
        "openlithohub.benchmark.measurement_support.initialize_cuda_measurement_device",
        lambda device: device,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: allocated_bytes)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: reserved_bytes)

    def fake_mem_get_info(index: int) -> tuple[int, int]:
        calls.append(index)
        return free_bytes, TOTAL_16GIB

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)  # type: ignore[attr-defined]
    return calls


def test_headroom_violation_stops_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = plan_for(12 * 1024**3)
    assert plan.memory_feasible
    _patch_cuda(monkeypatch, free_bytes=plan.required_free_floor_bytes - 1)
    with pytest.raises(MemoryPlanHeadroomViolation, match="MEMORY_PLAN_HEADROOM_VIOLATION"):
        assert_headroom(plan, "cuda:0")


def test_headroom_guard_passes_at_floor_and_reports_free(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = plan_for(12 * 1024**3)
    _patch_cuda(monkeypatch, free_bytes=plan.required_free_floor_bytes)
    observation = assert_headroom(plan, "cuda:0")
    assert observation.physical_free_bytes == plan.required_free_floor_bytes
    assert observation.allocator_reusable_bytes == 0
    assert observation.effective_reusable_bytes == plan.required_free_floor_bytes


def test_headroom_guard_noop_off_cuda() -> None:
    plan = plan_socs_decomposition(
        grid_size=GRID,
        n_src=N_SRC,
        n_freq=N_FREQ,
        K=K,
        dtype="complex64",
        complex_bytes=8,
        device="cpu",
        facts=None,
    )
    observation = assert_headroom(plan, "cpu")
    assert observation.physical_free_bytes == -1
    assert observation.effective_reusable_bytes == -1


# ---- V3.1 allocator-aware double budget ---------------------------------------------


def test_reusable_allocator_cache_prevents_false_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """V3.1 core scenario: after the first chunk, PyTorch holds most freed
    VRAM as reserved cache the driver no longer reports as free.  The
    guard must credit REUSABLE cache against the modeled PyTorch need —
    physical free alone would wrongly STOP a safe run."""
    plan = plan_for(12 * 1024**3)
    physical = 3 * 1024**3  # driver sees far less than the modeled need
    cache = 8 * 1024**3  # but the allocator holds reusable cache
    _patch_cuda(
        monkeypatch,
        free_bytes=physical,
        allocated_bytes=1 * 1024**3,
        reserved_bytes=1 * 1024**3 + cache,
    )
    observation = assert_headroom(plan, "cuda:0")
    assert observation.physical_free_bytes == physical
    assert observation.allocator_reusable_bytes == cache
    assert observation.effective_reusable_bytes == physical + cache


def test_physical_floor_still_stops_even_with_large_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache must NEVER back the hard physical floor (workspace +
    emergency): no allocator reuse can serve driver/OS memory."""
    plan = plan_for(12 * 1024**3)
    physical = plan.physical_free_floor_bytes - 1
    _patch_cuda(
        monkeypatch,
        free_bytes=physical,
        allocated_bytes=0,
        reserved_bytes=16 * 1024**3,  # absurd cache cannot help the physical budget
    )
    with pytest.raises(MemoryPlanHeadroomViolation, match="hard floor"):
        assert_headroom(plan, "cuda:0")


def test_effective_budget_below_modeled_need_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Physical above its floor but modeled need uncovered even after
    crediting reusable cache -> STOP before allocation."""
    plan = plan_for(12 * 1024**3)
    physical = plan.physical_free_floor_bytes + 1024**3
    _patch_cuda(monkeypatch, free_bytes=physical, allocated_bytes=0, reserved_bytes=0)
    assert physical + 0 < plan.required_free_floor_bytes
    with pytest.raises(MemoryPlanHeadroomViolation, match="effective reusable"):
        assert_headroom(plan, "cuda:0")


def test_chunk_independent_phase_over_budget_fails_closed() -> None:
    """V3.1.2: when a chunk-INDEPENDENT phase (FFT/fftshift or
    eigendecomposition overlap) cannot fit the frozen budget — large K,
    large n_src — the plan is infeasible with the FIXED_PHASE reason no
    matter what chunk selection says; no allocation is ever attempted."""
    from openlithohub._utils.socs_memory_plan import REASON_INFEASIBLE_FIXED_PHASE

    # FFT phase dominates: 3*K*n_freq*8 must exceed the budget.  With
    # K = 420 the FFT/fftshift coexistence (~10.6 GiB) alone breaks the
    # ~10.4 GiB frozen budget regardless of chunk size.
    k_big = 420
    plan = plan_socs_decomposition(
        grid_size=GRID,
        n_src=1609,
        n_freq=N_FREQ,
        K=k_big,
        dtype="complex64",
        complex_bytes=8,
        device="cuda:0",
        facts=facts(12 * 1024**3),
    )
    budget = (
        12 * 1024**3
        - plan.absolute_headroom_bytes
        - plan.fractional_headroom_bytes
        - plan.workspace_reserve_bytes
    )
    assert plan.phase_peak_new_bytes_fft > budget
    assert not plan.memory_feasible
    assert plan.reason.startswith(REASON_INFEASIBLE_FIXED_PHASE)
    assert "fft" in plan.reason
    assert plan.selected_chunk_columns == 0


def test_physical_floor_fields_are_frozen_plan_facts() -> None:
    plan = plan_for(12 * 1024**3)
    from openlithohub._utils.socs_memory_plan import EMERGENCY_HEADROOM_BYTES

    assert plan.emergency_headroom_bytes == EMERGENCY_HEADROOM_BYTES
    assert plan.physical_free_floor_bytes == (
        plan.workspace_reserve_bytes + EMERGENCY_HEADROOM_BYTES
    )
    assert plan.required_free_floor_bytes > plan.physical_free_floor_bytes


def test_per_chunk_model_matches_the_v311_phase_accounting() -> None:
    """V3.1.1 phase-specific conservative model: the WORST-phase slope
    (Pass 2) is 24*n_src + 24*K + 64 B/column — block c64 + its c128
    copy + all index/gather temporaries + the complex128 v_block + the
    complex64 cast temporary — and the per-phase peaks are recorded on
    the plan for audit."""
    plan = plan_for(12 * 1024**3)
    assert plan.bytes_per_chunk_column == 24 * N_SRC + 24 * K + 64
    # exact per-phase formulas, independently recomputed from the frozen
    # policy and the audited allocation enumeration:
    n_freq = GRID * GRID
    common = 3 * n_freq * 8 + K * 4
    fixed_pass1 = N_SRC * N_SRC * 16
    fixed_pass2 = K * N_SRC * 16 + 2 * K * 8 + K * n_freq * 8
    chunk = plan.selected_chunk_columns
    assert plan.phase_peak_new_bytes_pass1 == (common + fixed_pass1 + (24 * N_SRC + 64) * chunk)
    assert plan.phase_peak_new_bytes_pass2 == (
        common + fixed_pass2 + (24 * N_SRC + 24 * K + 64) * chunk
    )
    assert plan.phase_peak_new_bytes_eigendecomposition == common + 6 * fixed_pass1
    assert plan.phase_peak_new_bytes_fft == common + fixed_pass2 + 2 * K * n_freq * 8
    assert plan.phase_peak_new_bytes_pass2 == max(
        plan.phase_peak_new_bytes_pass1,
        plan.phase_peak_new_bytes_eigendecomposition,
        plan.phase_peak_new_bytes_pass2,
        plan.phase_peak_new_bytes_fft,
    )
    # planning budget semantics UNIFIED with the runtime guard: the floor
    # covers worst-phase NEW allocations only (the reserved pool is
    # credited guard-side, never double-counted)
    assert plan.required_free_floor_bytes == (
        plan.absolute_headroom_bytes
        + plan.fractional_headroom_bytes
        + plan.workspace_reserve_bytes
        + plan.phase_peak_new_bytes_pass2
    )
    assert plan.estimated_peak_bytes == (
        plan.reserved_bytes_at_plan + plan.phase_peak_new_bytes_pass2
    )


# ---- §10 worker enforcement -----------------------------------------------------------


def _valid_plan() -> SocsMemoryPlan:
    return plan_for(12 * 1024**3)


def test_worker_accepts_the_exact_frozen_plan() -> None:
    plan = _valid_plan()
    validate_worker_plan(
        plan,
        grid_size=GRID,
        n_src=N_SRC,
        n_freq=N_FREQ,
        K=K,
        dtype="complex64",
        device="cuda:0",
    )


def test_worker_refuses_hash_tampering() -> None:
    plan = _valid_plan()
    tampered = dataclasses.replace(plan, selected_chunk_columns=plan.selected_chunk_columns + 1)
    with pytest.raises(MemoryPlanContractViolation, match="hash mismatch"):
        validate_worker_plan(
            tampered,
            grid_size=GRID,
            n_src=N_SRC,
            n_freq=N_FREQ,
            K=K,
            dtype="complex64",
            device="cuda:0",
        )


def test_worker_refuses_semantic_mismatch() -> None:
    plan = _valid_plan()
    with pytest.raises(MemoryPlanContractViolation, match="grid_size"):
        validate_worker_plan(
            plan,
            grid_size=512,
            n_src=N_SRC,
            n_freq=N_FREQ,
            K=K,
            dtype="complex64",
            device="cuda:0",
        )
    with pytest.raises(MemoryPlanContractViolation, match="replan"):
        validate_worker_plan(
            plan,
            grid_size=GRID,
            n_src=N_SRC,
            n_freq=N_FREQ,
            K=K,
            dtype="complex64",
            device="cuda:1",
        )


def test_worker_refuses_infeasible_plan() -> None:
    plan = plan_for(512 * 1024**2)
    assert not plan.memory_feasible
    with pytest.raises(MemoryPlanContractViolation, match="MEMORY_PLAN_INFEASIBLE"):
        validate_worker_plan(
            plan,
            grid_size=GRID,
            n_src=N_SRC,
            n_freq=N_FREQ,
            K=K,
            dtype="complex64",
            device="cuda:0",
        )


# ---- §11 OOM is never control flow ------------------------------------------------------


def test_unexpected_oom_is_planner_contract_violation_not_retry() -> None:
    import sys
    from pathlib import Path

    harness_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "industrial-v2" / "run_v2_benchmark.py"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_v2_benchmark_oom", harness_path)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    oom = torch.cuda.OutOfMemoryError()
    failure_class, reason = harness._classify_worker_exception(oom)
    assert failure_class == "PLANNER_CONTRACT_VIOLATION"
    assert "shrink-and-retry" in reason
    other = RuntimeError("something else")
    failure_class2, _ = harness._classify_worker_exception(other)
    assert failure_class2 == "UNEXPECTED_EXCEPTION"
    assert sys.modules  # keep import used


def test_peak_witness_tolerance_is_frozen() -> None:
    assert PEAK_WITNESS_TOLERANCE == 1.05


# ---- same plan across warmup/repeats (§32, driver-side drift gate) ----------------------


def test_aggregate_row_fails_on_plan_drift_across_repeats() -> None:
    import importlib.util
    from pathlib import Path

    harness_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "industrial-v2" / "run_v2_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("run_v2_benchmark_drift", harness_path)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    def repeat(sha: str) -> dict:
        return {
            "status": "SUCCESS",
            "correctness_witness_pass": True,
            "gpu_warm_wall_s": 0.02,
            "memory_plan_sha256": sha,
            "worker_environment_witness_pass": True,
        }

    good = [repeat("a" * 64) for _ in range(5)]
    row = harness._aggregate_window("c", 1024, warm=[], measured=good)
    assert row["status"] == "SUCCESS"

    drifted = [repeat("a" * 64) for _ in range(4)] + [repeat("b" * 64)]
    row_bad = harness._aggregate_window("c", 1024, warm=[], measured=drifted)
    assert row_bad["status"] == "FAILED"
    assert "replanned" in row_bad["reason"]

    missing = [repeat("a" * 64) for _ in range(4)] + [repeat("")]
    row_missing = harness._aggregate_window("c", 1024, warm=[], measured=missing)
    assert row_missing["status"] == "FAILED"
