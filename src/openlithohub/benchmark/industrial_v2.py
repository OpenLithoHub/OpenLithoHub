"""Industrial Benchmark v2 — protocol authority (PR-G Phase 2A).

**Protocol first, measurement second.** This module freezes the v2
namespace, status vocabulary, run-identity computation, GPU environment
lock, canonical family membership, headline admission firewall and the
formal-publication guard. It deliberately contains NO measured numbers:
canonical v2 publication requires a controlled CUDA measurement run
(roadmap PR-G Phase 2B/2C); a CPU-only host cannot publish it.

Firewall (§50): benchmark code exists != measurement exists != claim
exists != headline exists. Phase 2A closes only the first item.

Independent namespace (§5) — v1.1 is frozen and never touched:

* schema: ``OpenLithoHub.industrial-benchmark.v2``
* run config: ``OpenLithoHub.industrial-run-config.v2``
* results root: ``benchmarks/results/industrial-v2/``
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from openlithohub._constants import WAVELENGTH_EUV_NM as DEFAULT_HOPKINS_WAVELENGTH_NM

SCHEMA_NAME = "OpenLithoHub.industrial-benchmark.v2"
RUN_CONFIG_SCHEMA = "OpenLithoHub.industrial-run-config.v2"
RESULTS_ROOT = "benchmarks/results/industrial-v2"

CANONICAL_FAMILY: frozenset[str] = frozenset(
    {
        "industrial-v2-index.json",
        "industrial-v2-gpu-runtime.json",
        "industrial-v2-hopkins.json",
        "industrial-v2-run-config.json",
        "industrial-v2-distribution-freeze.txt",
        "manifest.json",
        "SHA256SUMS.txt",
    }
)
"""The complete canonical family (PR-G §6). Publication is all-or-nothing:
a partial canonical root must never exist (verifier rejects it)."""

TIMING_HOST = "host_perf_counter"
TIMING_CUDA_SYNC = "cuda_synchronized"

STATUS_NOT_RUN_ENVIRONMENT = "NOT_RUN_ENVIRONMENT"
STATUS_UNSUPPORTED = "UNSUPPORTED"

# §18: frozen BEFORE measurement. Discrete semantics use exact equality;
# float output tolerance is per-dtype and never merges fp32/bf16 claims.
FLOAT_TOLERANCE_BY_DTYPE: dict[str, float] = {"fp32": 1e-5}

# §26: headline admission thresholds.
HEADLINE_MIN_RUNTIME_SPEEDUP = 1.10
HEADLINE_MIN_MEMORY_REDUCTION = 0.20
HEADLINE_MIN_REPEATS = 5


class StatusV2(str, Enum):
    """V2 status vocabulary (§23)."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_RUN_ENVIRONMENT = "NOT_RUN_ENVIRONMENT"
    NOT_RUN_MEMORY_POLICY = "NOT_RUN_MEMORY_POLICY"
    INFEASIBLE_ON_REFERENCE_MACHINE = "INFEASIBLE_ON_REFERENCE_MACHINE"


class ClaimLevelV2(str, Enum):
    """Claim provenance levels (§24). Initial v2 is REPRODUCED_INTERNAL."""

    REPRODUCED_INTERNAL = "REPRODUCED_INTERNAL"
    PUBLIC_BENCHMARK = "PUBLIC_BENCHMARK"
    THIRD_PARTY_REPRODUCED = "THIRD_PARTY_REPRODUCED"
    FOUNDRY_CALIBRATED = "FOUNDRY_CALIBRATED"


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic strict JSON: sorted keys, tight separators, NaN/Inf
    rejected at write time (never enters an artifact)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_strict_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write canonical strict JSON; NaN/Infinity raise instead of landing
    in an artifact (B2 gate: artifacts are strict JSON, always)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(path).write_text(text)


# ---- run identity (§8) ------------------------------------------------------


@dataclass(frozen=True)
class RunConfigV2:
    """Every performance-relevant semantic knob of a v2 run.

    Changing ANY field changes the run identity (hostile-tested).  These
    are the semantic CLI args of Tier A/B/C — nothing non-semantic may
    enter, nothing semantic may be omitted.
    """

    tiers: tuple[str, ...] = ("a",)
    device: str = "cuda:0"
    dtype: str = "fp32"
    tf32_matmul: bool = False
    tf32_cudnn: bool = False
    deterministic_algorithms: bool = True
    compile_mode: str = "off"
    batch_size: int = 8
    tile_size: int = 1024
    halo_px: int = 64
    pixel_nm: float = 1.0
    forward_radius: int = 4
    forward_sigma_nm: float = 1.6
    window_sizes: tuple[int, ...] = (4096, 8192, 16384, 32768)
    hopkins_wavelength_nm: float = DEFAULT_HOPKINS_WAVELENGTH_NM
    hopkins_na: float = 0.33
    hopkins_sigma_outer: float = 0.9
    hopkins_sigma_inner: float = 0.6
    hopkins_defocus_nm: float = 0.0
    hopkins_grid: int = 1024
    warmup_count: int = 2
    repeat_count: int = 5
    layer: str = "66:44"
    """GDS layer selection (sky130hd li1). Changing the layer changes the
    measured workload, so it MUST change the run identity (2B.1-B)."""
    fixture_sha256: str = ""
    """SHA-256 of the exact GDS bytes (2B.1-A). Identical args over
    different fixture bytes must produce a different run identity."""

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tiers"] = list(self.tiers)
        payload["window_sizes"] = list(self.window_sizes)
        return payload

    def with_changes(self, **changes: Any) -> RunConfigV2:
        return replace(self, **changes)


def compute_run_identity_v2(
    run_config: RunConfigV2,
    *,
    measurement_commit: str,
    harness_sha256: str,
    core_sha256: str,
    claim_generator_sha256: str,
    verifier_sha256: str,
    environment_lock_sha256: str = "",
) -> str:
    """Content-addressed v2 run identity (§8, 2B.1-D).

    Binds ONE sha256 to the measurement commit, the exact bytes of the
    harness/core/generator/verifier, the GPU environment lock hash
    (model/driver/CUDA build/Torch build/VRAM/topology/TF32), the fixture
    identity and EVERY semantic knob in the run config.  Any
    performance-relevant change — code bytes, GPU model, driver, CUDA
    build, dtype, TF32, determinism, batch, tile, halo, compile mode,
    Hopkins params, repeat/warmup counts, GDS bytes or layer — changes
    the identity.
    """
    payload = {
        "schema": RUN_CONFIG_SCHEMA,
        "measurement_commit": measurement_commit,
        "source_hashes": {
            "harness": harness_sha256,
            "core": core_sha256,
            "claim_generator": claim_generator_sha256,
            "verifier": verifier_sha256,
        },
        "environment_lock_sha256": environment_lock_sha256,
        "run_config": run_config.to_payload(),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


# ---- GPU environment lock (§9) ------------------------------------------------


def gpu_environment_lock_sha256(lock: Mapping[str, Any]) -> str:
    """Hash the canonicalized environment lock so it can enter the run
    identity (2B.1-D)."""
    return hashlib.sha256(canonical_json(dict(lock)).encode()).hexdigest()


def nvidia_driver_version() -> str:
    """Driver version from fixed-argv nvidia-smi, or "" when unavailable
    (empty driver is acceptable ONLY on CPU-only hosts, where canonical
    GPU publication is already blocked)."""
    import shutil
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return ""
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no user input
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_environment_lock() -> dict[str, Any]:
    """Collect the complete GPU environment lock object.

    Records driver/CUDA/cuDNN/device facts — never just a model name, and
    NEVER infers a GPU claim from ``torch.cuda.is_available()`` alone. On
    a CPU-only host the lock records ``available: false`` with empty
    devices; every GPU-required tier then resolves to
    ``NOT_RUN_ENVIRONMENT`` and canonical publication stays blocked.
    """
    import torch

    cuda_available = bool(torch.cuda.is_available())
    lock: dict[str, Any] = {
        "available": cuda_available,
        "count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
        "driver_version": nvidia_driver_version() if cuda_available else "",
        "torch_cuda_version": torch.version.cuda or "",
        # torch.backends.cudnn.version() is untyped upstream; the value is
        # recorded verbatim (or None) and never used numerically.
        "cudnn_version": (
            torch.backends.cudnn.version() if cuda_available else None  # type: ignore[no-untyped-call]
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "torch_version": torch.__version__,
    }
    if cuda_available:
        for index in range(lock["count"]):
            props = torch.cuda.get_device_properties(index)
            lock["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return lock


# ---- formal publication guard (§7/§35) ------------------------------------------


def formal_publication_blockers(
    *,
    env_lock: Mapping[str, Any],
    tier_rows: Mapping[str, Mapping[str, Any]],
    git_clean: bool,
    provisional: bool,
) -> list[str]:
    """Every reason the canonical v2 family cannot be promoted yet.

    Ordered, human-readable, and fail-closed: an empty list is the ONLY
    way ``promote_canonical_family`` promotes.  On a CPU-only host the
    first blocker is always the missing CUDA measurement environment
    (§35: do not emulate CUDA and call it formal measurement).
    """
    blockers: list[str] = []
    if not env_lock.get("available"):
        blockers.append("FORMAL_PUBLICATION_BLOCKED: CUDA measurement environment unavailable")
    if provisional:
        blockers.append("provisional run: canonical publication requires a formal run")
    if not git_clean:
        blockers.append("dirty tracked tree cannot publish authority (B2-A)")
    for tier in ("a", "b", "c"):
        row = tier_rows.get(tier)
        if row is None:
            blockers.append(f"tier {tier.upper()} has no measurement row")
        elif row.get("status") != StatusV2.SUCCESS.value:
            blockers.append(f"tier {tier.upper()} status {row.get('status')!r} is not SUCCESS")
        elif not row.get("correctness_witness_pass"):
            blockers.append(f"tier {tier.upper()} correctness witness did not pass (B2-G)")
        if row is not None and row.get("device_requires_cuda") and not env_lock.get("available"):
            blockers.append(f"tier {tier.upper()} requires CUDA; environment lacks it")
    return blockers


def build_canonical_family_in_workspace(
    *,
    workspace_dir: str | Path,
    run_config: RunConfigV2,
    run_identity: str,
    measurement_commit: str,
    source_hashes: Mapping[str, str],
    environment_lock: Mapping[str, Any],
    tier_rows: Mapping[str, Mapping[str, Any]],
    tracked_tree_clean: bool,
    provisional: bool,
) -> list[str]:
    """Map a formal run workspace onto the exact seven-member canonical
    family, with manifest + SHA256SUMS closure — fail-closed (2B.1-J/§12).

    Refuses (returns blockers, writes NOTHING into the family) on: dirty
    tree, provisional mode, non-SUCCESS tier, failed correctness,
    insufficient repeats, incomplete environment lock, missing fixture
    hash or layer, incomplete window ladder, unsynchronized GPU timing,
    or missing GPU memory facts.  The only sanctioned path from a formal
    workspace to ``promote_canonical_family()``.
    """
    blockers: list[str] = []
    workspace = Path(workspace_dir)
    if not tracked_tree_clean:
        blockers.append("dirty tracked tree cannot build the canonical family (B2-A)")
    if provisional:
        blockers.append("provisional run cannot build the canonical family")
    if not run_config.fixture_sha256:
        blockers.append("run config has no fixture sha256 (B2-A)")
    if not run_config.layer:
        blockers.append("run config has no GDS layer selection (2B.1-B)")

    env_lock = dict(environment_lock)
    lock_hash = gpu_environment_lock_sha256(env_lock)
    lock_required = (
        "available",
        "count",
        "devices",
        "driver_version",
        "torch_cuda_version",
        "torch_version",
        "tf32_matmul",
        "tf32_cudnn",
    )
    for key in lock_required:
        if key not in env_lock:
            blockers.append(f"environment lock incomplete: missing {key!r}")
    if env_lock.get("available") and not env_lock.get("driver_version"):
        blockers.append("formal CUDA run requires a nonempty driver_version (2B.1-E)")

    identity = compute_run_identity_v2(
        run_config,
        measurement_commit=measurement_commit,
        harness_sha256=source_hashes.get("harness", ""),
        core_sha256=source_hashes.get("core", ""),
        claim_generator_sha256=source_hashes.get("claim_generator", ""),
        verifier_sha256=source_hashes.get("verifier", ""),
        environment_lock_sha256=lock_hash,
    )
    if identity != run_identity:
        blockers.append(
            f"workspace identity {run_identity[:16]}… does not match recomputed "
            f"{identity[:16]}… (stale workspace or drifted config)"
        )

    expected_windows = sorted(run_config.window_sizes)
    tier_payloads: dict[str, dict[str, Any]] = {}
    tier_members = {
        "a": ("tier-a.json", "industrial-v2-index.json"),
        "b": ("tier-b.json", "industrial-v2-gpu-runtime.json"),
        "c": ("tier-c.json", "industrial-v2-hopkins.json"),
    }
    for tier, (_workspace_name, canonical_name) in tier_members.items():
        row = tier_rows.get(tier)
        if row is None:
            blockers.append(f"tier {tier.upper()} has no workspace row")
            continue
        if row.get("status") != StatusV2.SUCCESS.value:
            blockers.append(f"tier {tier.upper()} status {row.get('status')!r} is not SUCCESS")
        if not row.get("correctness_witness_pass"):
            blockers.append(f"tier {tier.upper()} correctness witness did not pass (B2-G)")
        if tier in ("a", "b"):
            windows_seen = sorted(w.get("window", 0) for w in row.get("window_rows", []))
            if windows_seen != expected_windows:
                blockers.append(
                    f"tier {tier.upper()} window ladder incomplete: "
                    f"{windows_seen} != {expected_windows} (2B.1-G)"
                )
        if tier in ("b", "c"):
            for wrow in row.get("window_rows", [row]):
                label = f"tier {tier.upper()} window {wrow.get('window')}"
                if wrow.get("timing_method") != TIMING_CUDA_SYNC:
                    blockers.append(f"{label}: unsynchronized GPU timing (B2-E)")
                for key in ("max_memory_allocated", "max_memory_reserved", "host_peak_rss_bytes"):
                    if key not in wrow:
                        blockers.append(f"{label}: missing {key!r} (§15)")
                if int(wrow.get("repeat_count") or 0) < HEADLINE_MIN_REPEATS:
                    blockers.append(f"{label}: insufficient repeats (B2)")
        if tier == "c":
            # 2B.2-B: the claim-bearing Tier C repeat statistics must be
            # complete (n == repeat_count) and sufficient (n >= 5) — a
            # zero-length aggregate from a stale timing key can never
            # enter the canonical family.
            aggregate_n = int(row.get("aggregate_n") or 0)
            if aggregate_n != int(row.get("repeat_count") or -1):
                blockers.append(
                    f"tier C timing aggregate n={aggregate_n} != repeat_count "
                    f"{row.get('repeat_count')!r} (stale timing key or partial repeats)"
                )
            if aggregate_n < HEADLINE_MIN_REPEATS:
                blockers.append(f"tier C has insufficient claim-bearing repeats: {aggregate_n}")
        repeat_level = row.get("repeat_level")
        if repeat_level is not None and int(repeat_level) < HEADLINE_MIN_REPEATS:
            blockers.append(f"tier {tier.upper()} has insufficient repeats")
        tier_payloads[canonical_name] = {
            "schema": SCHEMA_NAME,
            "run_identity": run_identity,
            "tier": tier,
            "correctness_witness_pass": bool(row.get("correctness_witness_pass")),
            "claim_level": ClaimLevelV2.REPRODUCED_INTERNAL.value,
            "rows": row.get("window_rows", [row]),
            "aggregate": {
                k: v
                for k, v in row.items()
                if k.startswith("aggregate_") or k in ("status", "repeats_recorded")
            },
        }

    if blockers:
        return blockers

    # ---- write the family (all gates closed) --------------------------
    workspace = Path(workspace_dir)
    write_strict_json(
        workspace / "industrial-v2-run-config.json",
        {
            "schema": RUN_CONFIG_SCHEMA,
            "measurement_commit": measurement_commit,
            "tracked_tree_clean": tracked_tree_clean,
            "provisional": False,
            "source_hashes": dict(source_hashes),
            "environment_lock_sha256": lock_hash,
            "run_identity": run_identity,
            "run_config": run_config.to_payload(),
        },
    )
    write_strict_json(
        workspace / "industrial-v2-distribution-freeze.txt",
        {"gpu": dict(env_lock), "environment_lock_sha256": lock_hash},
    )
    for canonical_name, payload in tier_payloads.items():
        write_strict_json(workspace / canonical_name, payload)

    members = sorted(CANONICAL_FAMILY - {"manifest.json", "SHA256SUMS.txt"})
    manifest = {
        "run_identity": run_identity,
        "members": [{"name": name, "bytes": (workspace / name).stat().st_size} for name in members],
    }
    write_strict_json(workspace / "manifest.json", manifest)
    lines = [
        f"{sha256_file(workspace / name)}  {name}"
        for name in sorted(CANONICAL_FAMILY - {"SHA256SUMS.txt"})
    ]
    (workspace / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    return []


def promote_canonical_family(
    *,
    workspace_dir: str | Path,
    canonical_root: str | Path,
    env_lock: Mapping[str, Any],
    tier_rows: Mapping[str, Mapping[str, Any]],
    git_clean: bool,
    provisional: bool,
) -> list[str]:
    """Promote the verified workspace family to the canonical root — or
    return the blockers and promote NOTHING (all-or-nothing, §6/§7)."""
    blockers = formal_publication_blockers(
        env_lock=env_lock,
        tier_rows=tier_rows,
        git_clean=git_clean,
        provisional=provisional,
    )
    if blockers:
        return blockers
    import shutil

    canonical_root = Path(canonical_root)
    canonical_root.mkdir(parents=True, exist_ok=True)
    for member in sorted(CANONICAL_FAMILY):
        source = Path(workspace_dir) / member
        if not source.is_file():
            return [f"canonical member missing from workspace: {member}"]
        shutil.copy2(source, canonical_root / member)
    return []


# ---- headline admission firewall (§26) ------------------------------------------


def admit_headline(
    *,
    claim_id: str,
    correctness_pass: bool,
    repeat_count: int,
    runtime_speedup: float | None = None,
    memory_reduction: float | None = None,
    scope: str = "",
    status: str = StatusV2.SUCCESS.value,
) -> tuple[bool, str]:
    """Headline firewall (§26): a v2 claim can headline only when the
    artifact is valid, correctness passed, repeats are sufficient, the
    scope is explicit, and the effect threshold is exceeded.  Quality
    interpretation is not an input — Phase 2A publishes no quality
    headlines at all (§27)."""
    if status != StatusV2.SUCCESS.value:
        return False, f"status {status!r} is not SUCCESS"
    if not correctness_pass:
        return False, "correctness witness did not pass (B2-G)"
    if repeat_count < HEADLINE_MIN_REPEATS:
        return False, f"repeat count {repeat_count} < {HEADLINE_MIN_REPEATS}"
    if not scope:
        return False, "claim scope is not explicit"
    if runtime_speedup is None and memory_reduction is None:
        return False, "no measured effect"
    if runtime_speedup is not None and runtime_speedup < HEADLINE_MIN_RUNTIME_SPEEDUP:
        return False, f"runtime speedup {runtime_speedup:.3f}x < {HEADLINE_MIN_RUNTIME_SPEEDUP}x"
    if memory_reduction is not None and memory_reduction < HEADLINE_MIN_MEMORY_REDUCTION:
        return (
            False,
            f"memory reduction {memory_reduction:.3f} < {HEADLINE_MIN_MEMORY_REDUCTION}",
        )
    return True, f"{claim_id} admitted"


# ---- CUDA timing / memory discipline (§16/§17) -----------------------------------


def cuda_synchronized_wall(fn: Callable[[], Any], device: str) -> tuple[Any, float]:
    """Wall-time a CUDA region with mandatory synchronization (§16).

    All claim-bearing GPU timings MUST come through this helper — the
    verifier rejects GPU rows whose ``timing_method`` is not
    ``cuda_synchronized`` (B2-E), so unsynchronized kernel-launch time can
    never masquerade as elapsed GPU compute.
    """
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def host_peak_rss_bytes() -> int:
    """Platform-normalized host peak RSS in bytes (§15)."""
    import resource
    import sys

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def reset_gpu_peak_stats(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_memory(device: str) -> dict[str, int]:
    """Allocated and reserved peak memory are DISTINCT facts (§17)."""
    import torch

    if not device.startswith("cuda"):
        return {"max_memory_allocated": 0, "max_memory_reserved": 0}
    return {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


class BatchedFiniteSupportBlur:
    """Tier B forward model (§14): deterministic finite-support blur,
    batch-safe, CPU/CUDA matched, NO learned weights.

    The same mathematical operator on every device: fixed 9x9 (radius 4)
    separable Gaussian-support kernel with frozen parameters.  Architecture
    claim only — never neural-ILT quality.
    """

    def __init__(self, radius: int = 4, sigma: float = 1.6) -> None:
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.radius = radius
        self.sigma = sigma
        axis = torch.arange(2 * radius + 1, dtype=torch.float32) - radius
        profile = torch.exp(-(axis**2) / (2.0 * sigma**2))
        profile = profile / profile.sum()
        self._kx = profile.reshape(1, 1, 1, -1)
        self._ky = profile.reshape(1, 1, -1, 1)
        self._kernel_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def to(self, device: str) -> BatchedFiniteSupportBlur:
        """Pre-position kernels on a device BEFORE the timed region
        (2B.1-H: no uncontrolled per-tile kernel reconstruction inside the
        timed path)."""
        self._kernels_for(torch.device(device))
        return self

    def _kernels_for(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen coefficients per device, built once and cached.

        Single authority for the window and batched paths: the same
        kernels execute on whatever device the input occupies — a CUDA
        input can never meet a CPU kernel (2B.1-H).
        """
        key = str(device)
        cached = self._kernel_cache.get(key)
        if cached is None:
            cached = (self._kx.to(device), self._ky.to(device))
            self._kernel_cache[key] = cached
        return cached

    def window_forward(self, tile: torch.Tensor) -> torch.Tensor:
        kx, ky = self._kernels_for(tile.device)
        x = tile.float().unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.conv2d(x, kx, padding=(0, self.radius))
        x = torch.nn.functional.conv2d(x, ky, padding=(self.radius, 0))
        return x.squeeze(0).squeeze(0)

    def batch_forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() != 4 or batch.shape[1] != 1:
            raise ValueError(f"batch_forward expects (B, 1, H, W), got {tuple(batch.shape)}")
        kx, ky = self._kernels_for(batch.device)
        x = batch.float()
        x = torch.nn.functional.conv2d(x, kx, padding=(0, self.radius))
        x = torch.nn.functional.conv2d(x, ky, padding=(self.radius, 0))
        return x


__all__ = [
    "BatchedFiniteSupportBlur",
    "build_canonical_family_in_workspace",
    "gpu_environment_lock_sha256",
    "host_peak_rss_bytes",
    "nvidia_driver_version",
    "CANONICAL_FAMILY",
    "ClaimLevelV2",
    "FLOAT_TOLERANCE_BY_DTYPE",
    "RESULTS_ROOT",
    "RUN_CONFIG_SCHEMA",
    "SCHEMA_NAME",
    "RunConfigV2",
    "StatusV2",
    "TIMING_CUDA_SYNC",
    "TIMING_HOST",
    "admit_headline",
    "canonical_json",
    "compute_run_identity_v2",
    "cuda_synchronized_wall",
    "formal_publication_blockers",
    "gpu_environment_lock",
    "gpu_peak_memory",
    "promote_canonical_family",
    "reset_gpu_peak_stats",
    "sha256_bytes",
    "sha256_file",
    "write_strict_json",
]
