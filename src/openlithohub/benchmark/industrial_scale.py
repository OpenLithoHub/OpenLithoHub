"""Industrial Scale Benchmark — authority core (PR-G scale track, S1).

An INDEPENDENT namespace: it is neither Industrial Benchmark v1.1 nor
Industrial Benchmark v2, and it must never touch their artifacts:

* schema:            ``OpenLithoHub.industrial-scale-benchmark.v1``
* run config:        ``OpenLithoHub.industrial-scale-run-config.v1``
* fixture manifest:  ``OpenLithoHub.scale-fixture.v1``
* results root:      ``benchmarks/results/industrial-scale/``

Every scale artifact carries an explicit authority scope:
``SCALE_CHARACTERIZATION`` — never v1.1 authority, never v2 authority,
never foundry calibrated, never a commercial-tool comparison.  Until the
formal scale protocol is frozen on a CUDA host, every result is
PROVISIONAL / DEVELOPMENT DRY RUN / NOT PERFORMANCE AUTHORITY.

This module deliberately contains NO measured numbers and NO execution
logic: schemas, validation, run identity, environment lock, status
vocabulary, fixture-manifest authority, the scale ladder, forward
profiles, canonical family membership and formal blockers only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

SCHEMA_NAME = "OpenLithoHub.industrial-scale-benchmark.v1"
RUN_CONFIG_SCHEMA = "OpenLithoHub.industrial-scale-run-config.v1"
FIXTURE_SCHEMA = "OpenLithoHub.scale-fixture.v1"
RESULTS_ROOT = "benchmarks/results/industrial-scale"

AUTHORITY_SCOPE = "SCALE_CHARACTERIZATION"
NOT_V1_1_AUTHORITY = True
NOT_V2_AUTHORITY = True
NOT_FOUNDRY_CALIBRATED = True
COMMERCIAL_TOOL_COMPARISON = "NONE"

# Frozen PDB (Physical Design Database) lineage shared with v1.1 / v2.
PDB_REPOSITORY = "SJTU-YONGFU-RESEARCH-GRP/PDB-Physical-Design-Database"
PDB_COMMIT = "9e1e3399b1b707f26fee853bce1ff91ab466ce24"

# Development scale ladder (§S7 of the charter).  A host never has to run
# every rung; statuses make partial ladders honest.
SCALE_LADDER: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072, 262144)

FORWARD_PROFILES: tuple[str, ...] = (
    "P0_IDENTITY",
    "P1_FINITE_SUPPORT",
    "P2_HOPKINS_BOUNDED",
)
"""Forward profiles (charter §S5).  P0 is plumbing-only and may NEVER be
a performance headline; P1 (deterministic finite-support kernel) is the
main large-layout / multi-GPU profile; P2 is bounded Hopkins compute
stress and is never a first-acceptance requirement."""

DEVICE_BACKENDS: tuple[str, ...] = ("cpu-worker-emulation", "cuda")
SINK_KINDS: tuple[str, ...] = ("memmap_npy", "manhattan")
OUTPUT_SEMANTICS = "core_raster_fp32"

TIMING_CUDA_SYNC = "cuda_synchronized"
TIMING_HOST = "host_perf_counter"

# Lanes (charter §S4.2): large-layout scale and multi-GPU scaling are
# SEPARATE claims — one number must never mix them.
LANE_A = "A_LARGE_LAYOUT_STREAMING"
LANE_B = "B_MULTI_GPU_SCALING"
LANE_C = "C_SINGLE_GPU_SATURATION"

FROZEN_MICROBATCH_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
"""Lane C's frozen microbatch ladder (1×RTX4090 migration).  Frozen
BEFORE any GPU observation; changing it is a protocol change."""


class ScaleStatus(str, Enum):
    """Scale status vocabulary — silent skips are impossible (§S7.3)."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    FAILED_CORRECTNESS = "FAILED_CORRECTNESS"
    NOT_RUN_MEMORY_POLICY = "NOT_RUN_MEMORY_POLICY"
    NOT_RUN_TIME_POLICY = "NOT_RUN_TIME_POLICY"
    NOT_RUN_ENVIRONMENT = "NOT_RUN_ENVIRONMENT"
    UNSUPPORTED = "UNSUPPORTED"


SCALE_CANONICAL_FAMILY: frozenset[str] = frozenset(
    {
        "industrial-scale-index.json",
        "industrial-scale-runtime.json",
        "industrial-scale-saturation.json",
        "industrial-scale-fixture.json",
        "industrial-scale-run-config.json",
        "industrial-scale-environment.json",
        "industrial-scale-distribution-freeze.txt",
        "manifest.json",
        "SHA256SUMS.txt",
    }
)
"""The scale canonical family (nine members since the 1×RTX4090
migration added the Lane C saturation member).  Family membership is
fail-closed: a partial or augmented root must never pass the verifier
(S10)."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic strict JSON: sorted keys, tight separators, NaN/Inf
    rejected at write time."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_strict_json(path: str, payload: Mapping[str, Any]) -> None:
    """Strict-JSON write: NaN/Infinity raise instead of landing in an
    artifact."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dense_float32_equivalent_bytes(width_px: int, height_px: int) -> int:
    """The hypothetical full float32 raster size a die-area bbox WOULD
    occupy at the declared pixel size.  This is a DERIVED EQUIVALENT, not
    processed bytes — claims must never say the raster was materialized."""
    if width_px < 0 or height_px < 0:
        raise ValueError("pixel dimensions must be non-negative")
    return width_px * height_px * 4


# ---- fixture manifest (§S2 / charter) ------------------------------------------


@dataclass(frozen=True)
class ScaleFixtureManifest:
    """Authority record for ONE prepared scale fixture.

    Produced only by ``scripts/prepare_industrial_scale_fixture.py``; the
    benchmark refuses to run without one.  Every field is re-validated on
    load — the harness never trusts a historical artifact.
    """

    source_repository: str
    source_commit: str
    design: str
    gds_sha256: str
    gds_bytes: int
    top_cell: str
    dbu_nm: float
    bbox_dbu: tuple[int, int, int, int]
    pixel_nm: float
    die_size_px: tuple[int, int]
    equivalent_pixels: int
    dense_float32_equivalent_bytes: int
    layers: tuple[str, ...]
    selected_layer: str
    preparation_script_sha256: str
    split_chunks: tuple[str, ...] = ()
    schema: str = FIXTURE_SCHEMA

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_dbu"] = list(self.bbox_dbu)
        payload["die_size_px"] = list(self.die_size_px)
        payload["layers"] = list(self.layers)
        payload["split_chunks"] = list(self.split_chunks)
        return payload

    def validate(self) -> list[str]:
        """Fail-closed consistency checks; empty list == valid."""
        problems: list[str] = []
        required = (
            "source_repository",
            "source_commit",
            "design",
            "gds_sha256",
            "top_cell",
            "selected_layer",
            "preparation_script_sha256",
        )
        payload = self.to_payload()
        for key in required:
            if not payload.get(key):
                problems.append(f"fixture manifest missing {key!r}")
        if len(self.gds_sha256) != 64:
            problems.append("gds_sha256 is not a 64-hex digest")
        if self.gds_bytes <= 0:
            problems.append("gds_bytes must be positive")
        if self.dbu_nm <= 0 or self.pixel_nm <= 0:
            problems.append("dbu_nm and pixel_nm must be positive")
        x0, y0, x1, y1 = self.bbox_dbu
        if x1 <= x0 or y1 <= y0:
            problems.append("bbox_dbu is not a positive-area box")
        if not self.layers:
            problems.append("no layers recorded")
        if self.selected_layer not in self.layers:
            problems.append(
                f"selected layer {self.selected_layer!r} is not among the enumerated "
                f"layers {list(self.layers)}"
            )
        width_px, height_px = self.die_size_px
        expected_w = round((x1 - x0) * self.dbu_nm / self.pixel_nm)
        expected_h = round((y1 - y0) * self.dbu_nm / self.pixel_nm)
        if (width_px, height_px) != (expected_w, expected_h):
            problems.append(
                f"die_size_px {(width_px, height_px)} inconsistent with bbox/dbu/pixel "
                f"(expected {(expected_w, expected_h)})"
            )
        if self.equivalent_pixels != width_px * height_px:
            problems.append("equivalent_pixels inconsistent with die_size_px")
        if self.dense_float32_equivalent_bytes != dense_float32_equivalent_bytes(
            width_px, height_px
        ):
            problems.append("dense_float32_equivalent_bytes inconsistent with die_size_px")
        return problems

    def with_changes(self, **changes: Any) -> ScaleFixtureManifest:
        return replace(self, **changes)


def load_scale_fixture_manifest(path: str) -> ScaleFixtureManifest:
    """Load and fully re-validate a fixture manifest (fail-closed)."""
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"fixture manifest schema is not {FIXTURE_SCHEMA!r}")
    if "equivalent_pixels" not in payload:
        w, h = (int(v) for v in payload["die_size_px"])
        payload["equivalent_pixels"] = w * h  # legacy manifest backfill
    manifest = ScaleFixtureManifest(
        source_repository=str(payload["source_repository"]),
        source_commit=str(payload["source_commit"]),
        design=str(payload["design"]),
        gds_sha256=str(payload["gds_sha256"]),
        gds_bytes=int(payload["gds_bytes"]),
        top_cell=str(payload["top_cell"]),
        dbu_nm=float(payload["dbu_nm"]),
        bbox_dbu=(
            int(payload["bbox_dbu"][0]),
            int(payload["bbox_dbu"][1]),
            int(payload["bbox_dbu"][2]),
            int(payload["bbox_dbu"][3]),
        ),
        pixel_nm=float(payload["pixel_nm"]),
        die_size_px=(int(payload["die_size_px"][0]), int(payload["die_size_px"][1])),
        equivalent_pixels=int(payload["equivalent_pixels"]),
        dense_float32_equivalent_bytes=int(payload["dense_float32_equivalent_bytes"]),
        layers=tuple(str(v) for v in payload["layers"]),
        selected_layer=str(payload["selected_layer"]),
        preparation_script_sha256=str(payload["preparation_script_sha256"]),
        split_chunks=tuple(str(v) for v in payload.get("split_chunks", [])),
    )
    problems = manifest.validate()
    if problems:
        raise ValueError("invalid fixture manifest: " + "; ".join(problems))
    return manifest


# ---- run config + identity (§S3) ------------------------------------------------


@dataclass(frozen=True)
class ScaleRunConfig:
    """Every performance-relevant semantic knob of a scale run.  ANY
    change — fixture, layer, pixel, topology request, dtype, TF32,
    determinism, tile, halo, microbatch, queue depth, forward profile or
    parameters, sink, output semantics, warmup, repeats, lanes — changes
    the run identity (hostile-tested)."""

    lanes: tuple[str, ...] = (LANE_A,)
    device_backend: str = "cpu-worker-emulation"
    gpu_count: int = 0
    worker_count: int = 1
    dtype: str = "fp32"
    tf32_matmul: bool = False
    tf32_cudnn: bool = False
    deterministic_algorithms: bool = True
    tile_size: int = 1024
    halo_px: int = 64
    microbatch: int = 8
    microbatch_ladder: tuple[int, ...] = ()
    queue_depth: int = 64
    forward_profile: str = "P1_FINITE_SUPPORT"
    forward_radius: int = 4
    forward_sigma_nm: float = 1.6
    hopkins_grid: int = 1024
    sink_kind: str = "memmap_npy"
    output_semantics: str = OUTPUT_SEMANTICS
    warmup_count: int = 1
    repeat_count: int = 3
    window_sizes: tuple[int, ...] = (4096, 8192)
    fixture_manifest_sha256: str = ""
    fixture_gds_sha256: str = ""
    selected_layer: str = ""
    pixel_nm: float = 1.0

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lanes"] = list(self.lanes)
        payload["window_sizes"] = list(self.window_sizes)
        payload["microbatch_ladder"] = list(self.microbatch_ladder)
        return payload

    def with_changes(self, **changes: Any) -> ScaleRunConfig:
        return replace(self, **changes)


def compute_scale_run_identity(
    run_config: ScaleRunConfig,
    *,
    measurement_commit: str,
    harness_sha256: str,
    core_sha256: str,
    verifier_sha256: str,
    claim_generator_sha256: str,
    fixture_manifest_sha256: str,
    environment_lock_sha256: str,
) -> str:
    """Content-addressed scale run identity (§S3.1).  Binds one sha256 to
    the measurement commit, the exact bytes of harness/core/verifier/
    claim generator, the fixture manifest AND GDS digests, the GPU
    environment lock and every semantic knob."""
    payload = {
        "schema": RUN_CONFIG_SCHEMA,
        "measurement_commit": measurement_commit,
        "source_hashes": {
            "harness": harness_sha256,
            "core": core_sha256,
            "verifier": verifier_sha256,
            "claim_generator": claim_generator_sha256,
        },
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "fixture_gds_sha256": run_config.fixture_gds_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "run_config": run_config.to_payload(),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


# ---- environment lock (§S3.2) ----------------------------------------------------


def parse_nvidia_smi_topology(text: str) -> list[dict[str, str]]:
    """Parse ``nvidia-smi topo -m`` output into per-GPU PCI facts.

    Tolerant, test-fixture-driven parser: only rows whose first column is
    ``GPU<i>`` are interpreted; the PCI bus id is the LAST whitespace
    token on the row (the legend/footer is ignored).  CPU-only hosts have
    no such output — the lock records an empty list."""
    devices: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("GPU"):
            continue
        tokens = stripped.split()
        if len(tokens) < 2 or not tokens[0][3:].isdigit():
            continue
        if tokens[1].startswith("GPU"):
            continue  # header row (GPU0 GPU1 ... CPU Affinity ...)
        devices.append({"gpu_index": tokens[0][3:], "pci_bus_id": tokens[-1]})
    return devices


def scale_environment_lock(
    *,
    topology_text: str = "",
    requested_gpu_count: int = 0,
) -> dict[str, Any]:
    """Collect the scale environment lock.  CPU-only development hosts
    record ``available: false`` with an empty device list — multi-worker
    CPU emulation still gets a complete, hashable lock.  GPU hosts must
    additionally record per-device identity (name/UUID/PCI bus/VRAM/
    compute capability), driver, CUDA/cuDNN builds, TF32 state and the
    nvidia-smi topology facts."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    lock: dict[str, Any] = {
        "available": cuda_available,
        "count": torch.cuda.device_count() if cuda_available else 0,
        "requested_gpu_count": requested_gpu_count,
        "devices": [],
        "driver_version": nvidia_driver_version() if cuda_available else "",
        "torch_cuda_version": torch.version.cuda or "",
        "torch_version": torch.__version__,
        "cudnn_version": (
            torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
            if cuda_available
            else None
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "topology": parse_nvidia_smi_topology(topology_text) if cuda_available else [],
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
                    "uuid": device_uuid(index),
                }
            )
    return lock


def device_uuid(index: int) -> str:
    """GPU UUID from fixed-argv nvidia-smi, or "" when unavailable (the
    empty string is acceptable only where canonical publication is
    already blocked)."""
    import shutil
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return ""
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no user input
            [
                nvidia_smi,
                f"--id={index}",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def nvidia_driver_version() -> str:
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


def scale_environment_lock_sha256(lock: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(lock)).encode()).hexdigest()


# ---- formal blockers (§S3) -------------------------------------------------------


def formal_scale_blockers(
    *,
    env_lock: Mapping[str, Any],
    run_config: ScaleRunConfig,
    lane_rows: Mapping[str, Mapping[str, Any]],
    git_clean: bool,
    provisional: bool,
) -> list[str]:
    """Every reason the scale canonical family cannot be promoted yet —
    fail-closed; an empty list is the ONLY promotion license.  On a CPU
    host the first blocker is the missing CUDA environment; CPU emulation
    can NEVER satisfy a formal cuda-backend run (charter §S6.3)."""
    blockers: list[str] = []
    if run_config.device_backend != "cuda":
        blockers.append(
            "formal scale runs require the cuda backend; "
            "cpu-worker-emulation is development-only and never authority"
        )
    if run_config.device_backend == "cuda" and not env_lock.get("available"):
        blockers.append("FORMAL_SCALE_BLOCKED: CUDA measurement environment unavailable")
    if (
        run_config.device_backend == "cuda"
        and env_lock.get("available")
        and int(env_lock.get("count") or 0) < run_config.gpu_count
    ):
        blockers.append(
            f"environment has {env_lock.get('count')} GPU(s); "
            f"run config requests {run_config.gpu_count}"
        )
    if provisional:
        blockers.append("provisional run: canonical publication requires a formal run")
    if not git_clean:
        blockers.append("dirty tracked tree cannot publish scale authority (B2-A)")
    if not run_config.fixture_gds_sha256:
        blockers.append("run config has no fixture gds sha256")
    if not run_config.selected_layer:
        blockers.append("run config has no selected layer")
    for lane in run_config.lanes:
        row = lane_rows.get(lane)
        if row is None:
            blockers.append(f"lane {lane} has no measurement row")
        elif row.get("status") != ScaleStatus.SUCCESS.value:
            blockers.append(f"lane {lane} status {row.get('status')!r} is not SUCCESS")
        elif not row.get("correctness_witness_pass"):
            blockers.append(f"lane {lane} correctness witness did not pass")
        if row is not None and int(row.get("repeat_count") or 0) < 5:
            blockers.append(f"lane {lane} has insufficient repeats for formal claims")
    if LANE_C in run_config.lanes:
        # single-GPU saturation is single-GPU BY DEFINITION (1×RTX4090
        # migration): multi-worker emulation can never satisfy it.
        if run_config.gpu_count != 1:
            blockers.append(f"lane C requires gpu_count == 1, got {run_config.gpu_count}")
        if tuple(run_config.microbatch_ladder) != FROZEN_MICROBATCH_LADDER:
            blockers.append(
                f"lane C microbatch ladder {list(run_config.microbatch_ladder)} != "
                f"frozen {list(FROZEN_MICROBATCH_LADDER)}"
            )
    if LANE_B in run_config.lanes and run_config.gpu_count < 2:
        # multi-GPU scaling remains a multi-GPU claim: it keeps requiring
        # multiple physical GPUs (deferred on the 1×RTX4090 host)
        blockers.append(
            "lane B is a multi-GPU claim and requires gpu_count >= 2 — "
            "deferred on the current 1×RTX4090 host"
        )
    return blockers


__all__ = [
    "AUTHORITY_SCOPE",
    "DEVICE_BACKENDS",
    "FROZEN_MICROBATCH_LADDER",
    "FIXTURE_SCHEMA",
    "FORWARD_PROFILES",
    "LANE_A",
    "LANE_B",
    "LANE_C",
    "PDB_COMMIT",
    "PDB_REPOSITORY",
    "RESULTS_ROOT",
    "RUN_CONFIG_SCHEMA",
    "SCALE_CANONICAL_FAMILY",
    "SCALE_LADDER",
    "SCHEMA_NAME",
    "SINK_KINDS",
    "OUTPUT_SEMANTICS",
    "ScaleFixtureManifest",
    "ScaleRunConfig",
    "ScaleStatus",
    "TIMING_CUDA_SYNC",
    "TIMING_HOST",
    "canonical_json",
    "compute_scale_run_identity",
    "dense_float32_equivalent_bytes",
    "device_uuid",
    "formal_scale_blockers",
    "load_scale_fixture_manifest",
    "nvidia_driver_version",
    "parse_nvidia_smi_topology",
    "scale_environment_lock",
    "scale_environment_lock_sha256",
    "sha256_bytes",
    "sha256_file",
    "write_strict_json",
]
