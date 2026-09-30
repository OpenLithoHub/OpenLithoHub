"""Source-owned VRAM planner for the bounded SOCS decomposition.

GPU Authority Repair v3 (§6-§9): the formal Tier-C Hopkins tier must
predict peak GPU memory BEFORE any large allocation, select the largest
safe aligned chunk column count under a frozen memory policy, and record
that decision as an authority artifact
(``OpenLithoHub.socs-memory-plan.v1``) bound into the run identity.

The planner never measures by failing: an infeasible problem is
reported as ``memory_feasible=false`` with reason
``MEMORY_PLAN_INFEASIBLE`` before any allocation is attempted, and the
runtime guard raises ``MEMORY_PLAN_HEADROOM_VIOLATION`` the moment free
memory falls below the plan floor.  CUDA OOM is never normal control
flow; an OOM after a planner PASS is a ``PLANNER_CONTRACT_VIOLATION``
by the caller's contract (§11), never a shrink-and-retry trigger.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

SOCS_MEMORY_PLAN_SCHEMA = "OpenLithoHub.socs-memory-plan.v1"

"""The frozen bounded strategy identity. The legacy dense full-H SVD
runtime has no strategy version: its absence from this table is the
point (§2 — no flag, no env var, no fallback re-enables it)."""
SOCS_STRATEGY = "exact_block_gram_topk_v1"
SOCS_STRATEGY_VERSION = 1

# ---- frozen planner policy constants (§7; frozen BEFORE measurement) ----------

ABSOLUTE_HEADROOM_BYTES = 512 * 1024 * 1024
"""Flat allocator-safety reserve held back from the device."""

FRACTIONAL_HEADROOM = 0.10
"""Fraction of TOTAL device memory additionally held back."""

BACKEND_WORKSPACE_RESERVE_BYTES = 256 * 1024 * 1024
"""Reserve for cuSOLVER/cuBLAS/cuDNN workspaces the allocator reports
only after the fact."""

EMERGENCY_HEADROOM_BYTES = 256 * 1024 * 1024
"""Hard PHYSICAL floor (V3.1): driver/OS/display reserve that must be
backed by real free VRAM regardless of what the PyTorch caching
allocator could reuse.  Checked separately from the modeled PyTorch
need (v3.1 closure — allocator-aware double-budget guard)."""


MIN_CHUNK_COLUMNS = 1024
"""Smallest legal frequency-column chunk; below this the plan is
infeasible and must fail BEFORE allocation."""

CHUNK_ALIGNMENT = 1024
"""Every selected chunk column count is a multiple of this."""

MAX_OPERATIONAL_H_CHUNK_BYTES = 2 * 1024**3
"""Operational cap on one H block (§8: frozen before measurement so
chunk selection stays capacity-optimal under the frozen policy rather
than unbounded on large cards)."""

CPU_DEFAULT_CHUNK_COLUMNS = 65536
"""Bounded default chunk on non-CUDA devices (no free-memory gate
exists there; the boundedness of the algorithm itself is the policy)."""

# §16 planned-vs-observed frozen relation: observed peak allocation may
# exceed the conservative model by at most this factor before the run's
# memory-plan peak witness fails.
PEAK_WITNESS_TOLERANCE = 1.05

REASON_INFEASIBLE_PREFIX = "MEMORY_PLAN_INFEASIBLE"
REASON_INFEASIBLE_FIXED_PHASE = "MEMORY_PLAN_INFEASIBLE_FIXED_PHASE"
"""A phase whose peak is INDEPENDENT of the chunk (eigendecomposition
or FFT/fftshift overlap) cannot fit the frozen budget at all — no chunk
choice can fix it, so the plan fails closed before any allocation."""


class MemoryPlanContractViolation(RuntimeError):  # noqa: N818 — violation is the protocol term
    """A runtime participant refused (or could not honor) the frozen plan.

    Raised on plan-hash or semantic-input mismatch, on free memory below
    the plan floor (§11 guard), and by the harness when a CUDA OOM occurs
    after a planner PASS (§11: OOM is never adaptive control flow)."""


class MemoryPlanHeadroomViolation(MemoryPlanContractViolation):
    """Free memory fell below ``required_free_floor_bytes`` before a major
    allocation (§11: STOP BEFORE ALLOCATION — never attempt it)."""


@dataclass(frozen=True)
class DeviceMemoryFacts:
    """Point-in-time device memory facts (§6 planner inputs), collected
    AFTER CUDA initialization/synchronization via ``torch.cuda.mem_get_info``."""

    device: str
    device_total_bytes: int
    device_free_bytes: int
    allocated_bytes: int
    reserved_bytes: int


def collect_cuda_memory_facts(device: str) -> DeviceMemoryFacts:
    """Collect :class:`DeviceMemoryFacts` for a CUDA device.  Initializes
    and synchronizes the device first (§7) so the numbers are not racing
    pending kernels."""
    import torch

    from openlithohub.benchmark.measurement_support import initialize_cuda_measurement_device

    initialize_cuda_measurement_device(device)
    torch.cuda.synchronize(device)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    free, total = torch.cuda.mem_get_info(index)  # type: ignore[no-untyped-call]
    return DeviceMemoryFacts(
        device=device,
        device_total_bytes=int(total),
        device_free_bytes=int(free),
        allocated_bytes=int(torch.cuda.memory_allocated(index)),
        reserved_bytes=int(torch.cuda.memory_reserved(index)),
    )


@dataclass(frozen=True)
class SocsMemoryPlan:
    """The frozen memory plan (§9 artifact, strict-JSON serializable)."""

    strategy: str
    strategy_version: int
    grid_size: int
    n_src: int
    n_freq: int
    K: int
    dtype: str
    complex_bytes: int
    device: str
    device_total_bytes: int
    device_free_bytes_at_plan: int
    allocated_bytes_at_plan: int
    reserved_bytes_at_plan: int
    absolute_headroom_bytes: int
    fractional_headroom_bytes: int
    workspace_reserve_bytes: int
    emergency_headroom_bytes: int
    physical_free_floor_bytes: int
    fixed_peak_estimate_bytes: int
    """V3.1.1: the worst NON-chunk-scaled phase allocation (conservative,
    not exact)."""
    phase_peak_new_bytes_pass1: int
    phase_peak_new_bytes_eigendecomposition: int
    phase_peak_new_bytes_pass2: int
    phase_peak_new_bytes_fft: int
    """V3.1.1 per-phase NEW-allocation peaks (reserved excluded — the
    runtime guard credits the reserved pool separately)."""
    bytes_per_chunk_column: int
    """V3.1.1: the WORST-phase per-column slope (Pass 2)."""
    selected_chunk_columns: int
    chunk_alignment: int
    chunk_count: int
    estimated_peak_bytes: int
    estimated_headroom_bytes: int
    required_free_floor_bytes: int
    """V3.1: the ALLOCATOR-EFFECTIVE floor (physical free + reusable
    PyTorch cache) the guard enforces before every chunk allocation."""
    memory_feasible: bool
    reason: str
    plan_sha256: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def plan_sha256_excluding_self(self) -> str:
        """SHA-256 over the canonical strict-JSON payload with the
        ``plan_sha256`` field excluded — the identity of the plan."""
        payload = self.to_payload()
        payload.pop("plan_sha256", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


def plan_socs_decomposition(
    *,
    grid_size: int,
    n_src: int,
    n_freq: int,
    K: int,  # noqa: N803 — K is the frozen SOCS truncation order
    dtype: str,
    complex_bytes: int,
    device: str,
    facts: DeviceMemoryFacts | None = None,
) -> SocsMemoryPlan:
    """Compute the frozen bounded-memory plan for one SOCS computation.

    V3.1.1: a PHASE-SPECIFIC CONSERVATIVE peak model (Pass 1 /
    eigendecomposition / Pass 2 / FFT+fftshift, each enumerated from the
    implementation's real allocations) — deliberately not called exact;
    audited against the source and locked by tests.  Pure arithmetic over
    the frozen policy constants and the measured
    :class:`DeviceMemoryFacts` — never over benchmark results (§8).  On a
    non-CUDA device (``facts is None``) the plan falls back to the frozen
    bounded default chunk with no free-memory gate; feasibility there is
    structural (a legal chunk exists), not a VRAM claim.

    Returns the plan even when infeasible: callers must treat
    ``memory_feasible=false`` (reason prefixed ``MEMORY_PLAN_INFEASIBLE``)
    as a stop-before-allocation condition, never as a hint to retry
    smaller ad hoc.
    """
    total = facts.device_total_bytes if facts is not None else 0
    free = facts.device_free_bytes if facts is not None else 0
    reserved = facts.reserved_bytes if facts is not None else 0
    allocated = facts.allocated_bytes if facts is not None else 0
    is_cuda = device.startswith("cuda") and facts is not None

    absolute = ABSOLUTE_HEADROOM_BYTES if is_cuda else 0
    fractional = int(FRACTIONAL_HEADROOM * total) if is_cuda else 0
    workspace = BACKEND_WORKSPACE_RESERVE_BYTES if is_cuda else 0

    # ---- V3.1.1 phase-specific conservative peak model (§6) -------------
    # NOT claimed as exact: a conservative per-phase peak of what the
    # implementation ACTUALLY materializes, audited against the source.
    pupil_state = 3 * n_freq * complex_bytes
    gram = n_src * n_src * 16  # complex128 accumulator (§4: stronger accumulator)
    topk = K * n_src * 16 + K * 8 + K * 8  # U_K complex128, sigma_K, lambda_K
    final_freq = K * n_freq * complex_bytes
    weights = K * 4
    common = pupil_state + weights  # resident through every phase

    # ---- per-phase chunk slopes (§6; V3.1.1 enumeration) ----------------
    # Every temporary the phase creates, per frequency column:
    #   h_block_c64   n_src × 8    the complex64 H block (both passes)
    #   h_block_c128  n_src × 16   its complex128 copy for the matmul
    #   cols/y/x      3 × 8        chunk-shaped int64 index vectors
    #   iy/ix/idx     3 × 8        per-source-row int64 index temporaries
    #   gathered      8            complex64 gathered row
    #   weighted      8            complex64 weighted row
    #   v_block       K × 16       Pass-2 reconstruction, complex128 form
    #   recon_cast    K × 8        the complex64 cast temporary that
    #                              exists while v_block is still alive
    index_vectors = 6 * 8  # cols, y, x (chunk) + iy, ix, linear idx (per row)
    gather_temps = 2 * 8  # gathered row + weighted row, complex64
    slope_pass1 = 24 * n_src + index_vectors + gather_temps
    slope_pass2 = slope_pass1 + 16 * K + 8 * K  # + v_block c128 + c64 cast temp

    # ---- phase fixed parts (on top of `common`) --------------------------
    fixed_pass1 = gram
    # symmetrize copy + eigh outputs + flipped U/lam overlap (conservative)
    fixed_eig = 6 * gram
    fixed_pass2 = topk + final_freq
    # kernels_freq + spatial + shifted spatial coexist at fftshift time
    fixed_fft = topk + 3 * final_freq

    def phase_new_peak(selected_columns: int) -> dict[str, int]:
        return {
            "pass1": common + fixed_pass1 + slope_pass1 * selected_columns,
            "eigendecomposition": common + fixed_eig,
            "pass2": common + fixed_pass2 + slope_pass2 * selected_columns,
            "fft": common + fixed_fft,
        }

    reason = ""
    feasible = True
    selected = 0
    if is_cuda:
        # V3.1.1 budget semantics UNIFIED with the runtime guard: NEW torch
        # pages are charged to physical free (the already-reserved pool is
        # credited by the guard's effective budget, never double-counted
        # here — physical_free never contained reserved pages anyway).
        budget_new = free - absolute - fractional - workspace
        limits = []
        for slope, fixed in (
            (slope_pass1, common + fixed_pass1),
            (slope_pass2, common + fixed_pass2),
        ):
            if budget_new > fixed:
                limits.append(int((budget_new - fixed) // slope))
            else:
                limits.append(0)
        max_by_budget = min(limits)
        cap_columns = max(0, MAX_OPERATIONAL_H_CHUNK_BYTES // max(1, n_src * complex_bytes))
        upper = min(max_by_budget, cap_columns, n_freq)
        aligned = (upper // CHUNK_ALIGNMENT) * CHUNK_ALIGNMENT
        if aligned < min(MIN_CHUNK_COLUMNS, n_freq):
            feasible = False
            selected = 0
            reason = (
                f"{REASON_INFEASIBLE_PREFIX}: physical budget {budget_new} B cannot hold "
                f"the minimum legal chunk ({min(MIN_CHUNK_COLUMNS, n_freq)} aligned "
                f"columns) at {slope_pass2} B/column (worst phase) under the frozen policy"
            )
        else:
            selected = aligned
            reason = (
                f"capacity-optimal chunk under frozen policy "
                f"(strategy {SOCS_STRATEGY} v{SOCS_STRATEGY_VERSION})"
            )
    else:
        selected = min(CPU_DEFAULT_CHUNK_COLUMNS, n_freq)
        reason = (
            f"non-CUDA device: frozen bounded default chunk policy "
            f"(strategy {SOCS_STRATEGY} v{SOCS_STRATEGY_VERSION})"
        )

    phases = phase_new_peak(selected)
    peak_new_max = max(phases.values())

    # v3.1.2 generic fixed-phase fail-closed: the chunk-scaled phases are
    # already bounded by the chunk selection, but the chunk-INDEPENDENT
    # phases (eigendecomposition overlap, FFT/fftshift coexistence) can
    # exceed the frozen budget no matter how small the chunk is (large K
    # or large n_src).  Refuse BEFORE any allocation — chunk selection
    # alone would otherwise report a feasible plan that cannot execute.
    if is_cuda and feasible:
        budget_new = free - absolute - fractional - workspace
        for phase_name, fixed_only in (
            ("eigendecomposition", phases["eigendecomposition"]),
            ("fft", phases["fft"]),
        ):
            if fixed_only > budget_new:
                feasible = False
                selected = 0
                reason = (
                    f"{REASON_INFEASIBLE_FIXED_PHASE}: {phase_name} phase peak "
                    f"{fixed_only} B exceeds the frozen physical budget {budget_new} B "
                    "independently of chunk size — reduce n_src/K or use a larger "
                    "device; this is a separate protocol decision, never a runtime retry"
                )
                break

    chunk_count = int(math.ceil(n_freq / selected)) if selected > 0 else 0
    estimated_peak = reserved + peak_new_max
    estimated_headroom = free - absolute - fractional - workspace - peak_new_max if is_cuda else 0
    emergency = EMERGENCY_HEADROOM_BYTES if is_cuda else 0
    physical_floor = workspace + emergency
    # V3.1.1: the runtime guard's EFFECTIVE floor = absolute + fractional
    # + workspace + worst-phase NEW allocations.  The reserved pool is
    # credited on the guard side (physical + reserved - allocated), so it
    # is deliberately NOT baked in here again.
    required_floor = absolute + fractional + workspace + peak_new_max

    plan = SocsMemoryPlan(
        strategy=SOCS_STRATEGY,
        strategy_version=SOCS_STRATEGY_VERSION,
        grid_size=grid_size,
        n_src=n_src,
        n_freq=n_freq,
        K=K,
        dtype=dtype,
        complex_bytes=complex_bytes,
        device=device,
        device_total_bytes=total,
        device_free_bytes_at_plan=free,
        allocated_bytes_at_plan=allocated,
        reserved_bytes_at_plan=reserved,
        absolute_headroom_bytes=absolute,
        fractional_headroom_bytes=fractional,
        workspace_reserve_bytes=workspace,
        emergency_headroom_bytes=emergency,
        physical_free_floor_bytes=physical_floor,
        fixed_peak_estimate_bytes=max(
            common + fixed_eig, common + fixed_fft
        ),  # worst phase WITHOUT chunk-scaling terms (conservative)
        phase_peak_new_bytes_pass1=phases["pass1"],
        phase_peak_new_bytes_eigendecomposition=phases["eigendecomposition"],
        phase_peak_new_bytes_pass2=phases["pass2"],
        phase_peak_new_bytes_fft=phases["fft"],
        bytes_per_chunk_column=slope_pass2,
        selected_chunk_columns=selected,
        chunk_alignment=CHUNK_ALIGNMENT,
        chunk_count=chunk_count,
        estimated_peak_bytes=estimated_peak,
        estimated_headroom_bytes=max(0, estimated_headroom),
        required_free_floor_bytes=required_floor,
        memory_feasible=feasible,
        reason=reason,
    )
    return SocsMemoryPlan(**{**plan.to_payload(), "plan_sha256": plan.plan_sha256_excluding_self()})


@dataclass(frozen=True)
class HeadroomObservation:
    """V3.1 allocator-aware headroom facts (§11 guard, v3.1 closure).

    ``physical_free_bytes`` is what the CUDA driver reports
    (``torch.cuda.mem_get_info``).  After the first chunk, PyTorch's
    caching allocator holds most freed VRAM as RESERVED cache, which the
    driver no longer reports as free but the next same-shaped PyTorch
    allocation can reuse directly.  Treating that cache as spent would
    wrongly STOP a safe run; treating it as driver-usable would be a
    fiction.  The guard therefore budgets both:

    * ``allocator_reusable_bytes`` = max(reserved - allocated, 0)
    * ``effective_reusable_bytes`` = physical_free + allocator_reusable

    against two separate floors (see :func:`assert_headroom`)."""

    physical_free_bytes: int
    allocator_reusable_bytes: int
    effective_reusable_bytes: int


def observe_headroom(device: str) -> HeadroomObservation:
    """Synchronize and collect the double-budget headroom facts."""
    import torch

    from openlithohub.benchmark.measurement_support import initialize_cuda_measurement_device

    initialize_cuda_measurement_device(device)
    torch.cuda.synchronize(device)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    free, _ = torch.cuda.mem_get_info(index)  # type: ignore[no-untyped-call]
    physical_free = int(free)
    allocated = int(torch.cuda.memory_allocated(index))
    reserved = int(torch.cuda.memory_reserved(index))
    reusable = max(reserved - allocated, 0)
    return HeadroomObservation(
        physical_free_bytes=physical_free,
        allocator_reusable_bytes=reusable,
        effective_reusable_bytes=physical_free + reusable,
    )


def assert_headroom(plan: SocsMemoryPlan, device: str) -> HeadroomObservation:
    """§11 pre-allocation guard (v3.1 allocator-aware double budget):
    CUDA-synchronize, collect the headroom facts, and raise
    :class:`MemoryPlanHeadroomViolation` BEFORE allocating when either
    budget is violated.

    Physical budget: ``physical_free >= workspace_reserve + emergency``
    — the part NO allocator cache can back.  Effective budget:
    ``physical_free + allocator_reusable >= required_free_floor_bytes``
    — the full modeled PyTorch need, crediting cache the allocator can
    actually hand back.  Returns the observation so the caller can
    accumulate the §16 minimum witnesses.

    No-op on non-CUDA devices (the plan itself carries no floors there)."""
    if not device.startswith("cuda") or plan.device_total_bytes == 0:
        return HeadroomObservation(
            physical_free_bytes=-1, allocator_reusable_bytes=-1, effective_reusable_bytes=-1
        )
    observation = observe_headroom(device)
    physical_floor = plan.workspace_reserve_bytes + EMERGENCY_HEADROOM_BYTES
    if observation.physical_free_bytes < physical_floor:
        raise MemoryPlanHeadroomViolation(
            "MEMORY_PLAN_HEADROOM_VIOLATION: physical free VRAM "
            f"{observation.physical_free_bytes} B < hard floor {physical_floor} B "
            "(workspace + emergency; no allocator cache can back this) "
            f"(plan {plan.plan_sha256[:16]}…) — STOPPING BEFORE ALLOCATION (§11)"
        )
    if observation.effective_reusable_bytes < plan.required_free_floor_bytes:
        raise MemoryPlanHeadroomViolation(
            "MEMORY_PLAN_HEADROOM_VIOLATION: effective reusable VRAM "
            f"{observation.effective_reusable_bytes} B "
            f"(physical {observation.physical_free_bytes} + allocator cache "
            f"{observation.allocator_reusable_bytes}) < modeled need "
            f"{plan.required_free_floor_bytes} B "
            f"(plan {plan.plan_sha256[:16]}…) — STOPPING BEFORE ALLOCATION (§11)"
        )
    return observation


def validate_worker_plan(
    plan: SocsMemoryPlan,
    *,
    grid_size: int,
    n_src: int,
    n_freq: int,
    K: int,  # noqa: N803 — K is the frozen SOCS truncation order
    dtype: str,
    device: str,
) -> None:
    """§10: a fresh worker may not replan.  Validate that the received plan
    matches THIS process's semantic inputs exactly, and that its hash is
    self-consistent.  Any mismatch is a
    :class:`MemoryPlanContractViolation` — not an adaptive retry."""
    expected_sha = plan.plan_sha256_excluding_self()
    if expected_sha != plan.plan_sha256:
        raise MemoryPlanContractViolation(
            "MEMORY_PLAN_CONTRACT_VIOLATION: plan hash mismatch — payload "
            f"hashes to {expected_sha[:16]}… but records {plan.plan_sha256[:16]}…"
        )
    for field, actual, expected in (
        ("grid_size", plan.grid_size, grid_size),
        ("n_src", plan.n_src, n_src),
        ("n_freq", plan.n_freq, n_freq),
        ("K", plan.K, K),
        ("dtype", plan.dtype, dtype),
        ("device", plan.device, str(device)),
    ):
        if actual != expected:
            raise MemoryPlanContractViolation(
                f"MEMORY_PLAN_CONTRACT_VIOLATION: plan {field}={actual!r} does not "
                f"match this worker's frozen input {expected!r} — replanning is forbidden"
            )
    if not plan.memory_feasible:
        raise MemoryPlanContractViolation(f"{plan.reason} — refusing to execute an infeasible plan")
    if plan.selected_chunk_columns <= 0 or plan.chunk_count <= 0:
        raise MemoryPlanContractViolation(
            "MEMORY_PLAN_CONTRACT_VIOLATION: plan selects no chunk columns"
        )
