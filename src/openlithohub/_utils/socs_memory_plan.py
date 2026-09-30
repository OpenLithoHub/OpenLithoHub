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
    fixed_peak_estimate_bytes: int
    bytes_per_chunk_column: int
    selected_chunk_columns: int
    chunk_alignment: int
    chunk_count: int
    estimated_peak_bytes: int
    estimated_headroom_bytes: int
    required_free_floor_bytes: int
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

    Pure arithmetic over the frozen policy constants and the measured
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

    # ---- fixed allocations (§6), bytes, conservative ---------------------
    pupil_state = 3 * n_freq * complex_bytes
    gram = n_src * n_src * 16  # complex128 accumulator (§4: stronger accumulator)
    topk = K * n_src * 16 + K * 8 + K * 8  # U_K complex128, sigma_K, lambda_K
    final_freq = K * n_freq * complex_bytes
    spatial_fft = K * n_freq * complex_bytes
    weights = K * 4
    fixed_peak = pupil_state + gram + topk + final_freq + spatial_fft + weights

    # ---- per-chunk allocations (§6), bytes per frequency column ----------
    h_block_c64 = n_src * complex_bytes
    h_block_c128 = n_src * 16  # complex128 copy for the Gram matmul
    v_block = K * complex_bytes
    index_vectors = 5 * 8  # y/x chunk vectors + per-row iy/ix/idx int64 temporaries
    block_temp = 8  # per-source-row gather temporary
    per_column = h_block_c64 + h_block_c128 + v_block + index_vectors + block_temp

    reason = ""
    feasible = True
    selected = 0
    if is_cuda:
        usable = free - absolute - fractional - workspace - reserved - fixed_peak
        max_by_budget = int(usable // per_column) if usable > 0 and per_column > 0 else 0
        cap_columns = max(0, MAX_OPERATIONAL_H_CHUNK_BYTES // max(1, n_src * complex_bytes))
        upper = min(max_by_budget, cap_columns, n_freq)
        aligned = (upper // CHUNK_ALIGNMENT) * CHUNK_ALIGNMENT
        if aligned < min(MIN_CHUNK_COLUMNS, n_freq):
            feasible = False
            selected = 0
            reason = (
                f"{REASON_INFEASIBLE_PREFIX}: budget {usable} B cannot hold the minimum "
                f"legal chunk ({min(MIN_CHUNK_COLUMNS, n_freq)} aligned columns) at "
                f"{per_column} B/column under the frozen policy"
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

    chunk_count = int(math.ceil(n_freq / selected)) if selected > 0 else 0
    estimated_peak = reserved + fixed_peak + per_column * selected
    estimated_headroom = free - absolute - fractional - workspace - estimated_peak if is_cuda else 0
    required_floor = absolute + fractional + workspace + fixed_peak + per_column * selected

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
        fixed_peak_estimate_bytes=fixed_peak,
        bytes_per_chunk_column=per_column,
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


def assert_headroom(plan: SocsMemoryPlan, device: str) -> int:
    """§11 pre-allocation guard: CUDA-synchronize, query free memory, and
    raise :class:`MemoryPlanHeadroomViolation` BEFORE allocating when free
    memory is below the plan floor.  Returns the observed free bytes so
    the caller can accumulate ``minimum_free_bytes_observed``.

    No-op on non-CUDA devices (the plan itself carries no floor there)."""
    if not device.startswith("cuda") or plan.device_total_bytes == 0:
        return -1
    import torch

    from openlithohub.benchmark.measurement_support import initialize_cuda_measurement_device

    initialize_cuda_measurement_device(device)
    torch.cuda.synchronize(device)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    free, _ = torch.cuda.mem_get_info(index)  # type: ignore[no-untyped-call]
    free = int(free)
    if free < plan.required_free_floor_bytes:
        raise MemoryPlanHeadroomViolation(
            "MEMORY_PLAN_HEADROOM_VIOLATION: free VRAM "
            f"{free} B < required floor {plan.required_free_floor_bytes} B "
            f"(plan {plan.plan_sha256[:16]}…) — STOPPING BEFORE ALLOCATION (§11)"
        )
    return free


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
