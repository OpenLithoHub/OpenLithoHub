"""GPU Authority Repair (PR-G2B.3) — V2 GPU worker / Windows host repair.

Covers issue #56's two frozen-source defects and the Windows
portability/authority repairs: CUDA-init-before-peak-stat-reset ordering,
the exact Tier-C ``HopkinsParams`` mapping, the actual-CUDA execution
witness, the shared worker-path preflight probes, the cross-platform
git/RSS helpers and fail-closed driver/device identity.

CPU CI proves structure and routing with mocks — it never claims real
GPU validation; the real-GPU proof belongs to the qualification issue
executed on the actual CUDA host.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from openlithohub.benchmark import measurement_support as ms
from openlithohub.benchmark.industrial_v2 import (
    RunConfigV2,
    build_canonical_family_in_workspace,
    compute_run_identity_v2,
    gpu_environment_lock_sha256,
    write_strict_json,
)
from openlithohub.simulators.hopkins_sim import HopkinsParams

REPO = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO / "benchmarks" / "industrial-v2" / "run_v2_benchmark.py"
PREFLIGHT_PATH = REPO / "scripts" / "preflight_industrial_v2.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- GPU Authority Repair §5: the exact Tier-C HopkinsParams mapping ----------


def test_tier_c_hopkins_params_constructs_exact_worker_object() -> None:
    harness = _load_module("run_v2_benchmark_g2b3_params", HARNESS_PATH)
    cfg = {
        "hopkins_wavelength_nm": 13.5,
        "hopkins_na": 0.33,
        "hopkins_sigma_outer": 0.9,
        "hopkins_sigma_inner": 0.6,
        "hopkins_defocus_nm": 0.0,
        "pixel_nm": 1.0,
    }
    params = harness.tier_c_hopkins_params(cfg)
    assert isinstance(params, HopkinsParams)
    # the declared run-config knob maps onto the dataclass field `sigma`
    assert params.sigma == cfg["hopkins_sigma_outer"]
    assert params.sigma_inner == cfg["hopkins_sigma_inner"]
    assert params.wavelength_nm == cfg["hopkins_wavelength_nm"]


def test_invalid_sigma_outer_keyword_regression_impossible() -> None:
    # the historical defect: HopkinsParams(sigma_outer=...) raised TypeError
    with pytest.raises(TypeError):
        HopkinsParams(sigma_outer=0.9)  # type: ignore[call-arg]
    field_names = {f.name for f in dataclasses.fields(HopkinsParams)}
    assert "sigma" in field_names and "sigma_outer" not in field_names


# ---- GPU Authority Repair §4: CUDA init before peak-stat reset ----------------


def _fake_cuda(state: dict) -> types.SimpleNamespace:
    """A torch.cuda stand-in whose reset_peak_memory_stats reproduces the
    historical ``RuntimeError: Invalid device argument`` until the
    initialization helper has actually run."""

    def reset(device: str) -> None:
        if not state["initialized"]:
            raise RuntimeError("Invalid device argument")

    return types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        set_device=lambda index: state.__setitem__("initialized", True),
        synchronize=lambda device: None,
        reset_peak_memory_stats=reset,
        max_memory_allocated=lambda device: 0,
        max_memory_reserved=lambda device: 0,
    )


def test_cuda_init_helper_initializes_context_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    state: dict = {"initialized": False}
    cuda = _fake_cuda(state)
    cuda.set_device = lambda index: calls.append(("set_device", index))  # type: ignore[method-assign]
    cuda.synchronize = lambda device: calls.append(("synchronize", device))  # type: ignore[method-assign]
    fake_tensor = types.SimpleNamespace(device="cuda:0", add_=lambda *_: None)
    monkeypatch.setattr(torch, "cuda", cuda)
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: fake_tensor)
    resolved = ms.initialize_cuda_measurement_device("cuda:0")
    assert resolved == "cuda:0"
    assert [name for name, _ in calls] == ["set_device", "synchronize"]


def test_peak_reset_before_cuda_init_reproduces_historical_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict = {"initialized": False}
    cuda = _fake_cuda(state)
    monkeypatch.setattr(torch, "cuda", cuda)
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **k: types.SimpleNamespace(add_=lambda *_: None, device="cuda:0")
    )
    # the historical failure: reset on an uninitialized context raises
    with pytest.raises(RuntimeError, match="Invalid device argument"):
        torch.cuda.reset_peak_memory_stats("cuda:0")
    # the repair: the init helper makes the SAME reset succeed
    ms.initialize_cuda_measurement_device("cuda:0")
    torch.cuda.reset_peak_memory_stats("cuda:0")


def test_cuda_init_helper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    monkeypatch.setattr(torch, "cuda", cuda)
    with pytest.raises(RuntimeError, match="CUDA measurement environment unavailable"):
        ms.initialize_cuda_measurement_device("cuda:0")
    with pytest.raises(ValueError, match="requires a cuda device"):
        ms.initialize_cuda_measurement_device("cpu")


# ---- GPU Authority Repair §7: the actual-CUDA execution witness ---------------


def test_forward_witness_requires_cuda_tensor_path() -> None:
    witness = ms.ForwardExecutionWitness("cuda:0")
    cpu_in = torch.zeros((2, 2))
    witness.record(cpu_in, cpu_in)
    summary = witness.summary()
    assert summary["requested_device"] == "cuda:0"
    assert summary["cuda_execution_witness_pass"] is False
    assert summary["cpu_forward_executions"] == 1

    cuda_like = types.SimpleNamespace(device="cuda:0")
    witness.record(cuda_like, cuda_like)
    summary = witness.summary()
    assert summary["forward_input_device"] == "cuda:0"
    assert summary["forward_output_device_before_d2h"] == "cuda:0"
    assert summary["cuda_forward_executions"] == 1


def test_forward_witness_rejects_mixed_cpu_execution() -> None:
    witness = ms.ForwardExecutionWitness("cuda:0")
    cuda_like = types.SimpleNamespace(device="cuda:0")
    cpu_like = types.SimpleNamespace(device="cpu")
    witness.record(cuda_like, cuda_like)
    witness.record(cpu_like, cpu_like)  # one forward stayed on CPU
    summary = witness.summary()
    assert summary["cuda_execution_witness_pass"] is False

    witness_all_cuda = ms.ForwardExecutionWitness("cuda:0")
    witness_all_cuda.record(cuda_like, cuda_like)
    witness_all_cuda.record(cuda_like, cuda_like)
    assert witness_all_cuda.summary()["cuda_execution_witness_pass"] is True


def test_forward_witness_rejects_cpu_request() -> None:
    witness = ms.ForwardExecutionWitness("cpu")
    witness.record(torch.zeros((2, 2)), torch.zeros((2, 2)))
    assert witness.summary()["cuda_execution_witness_pass"] is False


def test_tier_b_worker_initializes_cuda_before_peak_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #56 defect 1, end-to-end at the harness level: the Tier B
    worker must initialize CUDA before its first peak-stat reset, and the
    timed passes must record witness facts (status stays FAILED on a
    CPU-only CI host — mocks prove routing/structure, never GPU)."""
    harness = _load_module("run_v2_benchmark_g2b3_tierb", HARNESS_PATH)
    init_calls: list[str] = []
    monkeypatch.setattr(
        ms,
        "initialize_cuda_measurement_device",
        lambda device: init_calls.append(device) or "cuda:0",
    )

    def reset(device: str) -> None:
        if not init_calls:
            raise RuntimeError("Invalid device argument")

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        set_device=lambda index: None,
        synchronize=lambda device: None,
        reset_peak_memory_stats=reset,
        max_memory_allocated=lambda device: 0,
        max_memory_reserved=lambda device: 0,
    )
    monkeypatch.setattr(torch, "cuda", cuda)

    monkeypatch.setattr(
        harness,
        "_tier_b_execution_source",
        lambda cfg: types.SimpleNamespace(
            shape=(256, 256),
            layout_hash="fake-layout-hash",
            pixel_size_nm=1.0,
            backend_kind="fake",
        ),
    )

    class FakeBlur:
        """A blur stand-in: no real CUDA tensor ops under CI mocks."""

        def __init__(self, radius: int, sigma: float) -> None:
            self.radius = radius
            self.sigma = sigma

        def to(self, device: str) -> FakeBlur:
            return self

        def window_forward(self, tile):
            return tile

        def batch_forward(self, batch):
            return batch

    import openlithohub.benchmark.industrial_v2 as industrial_v2_pkg

    monkeypatch.setattr(industrial_v2_pkg, "BatchedFiniteSupportBlur", FakeBlur)

    class FakeSink:
        def __init__(self, shape: tuple) -> None:
            self.shape = shape

        def finalize(self) -> torch.Tensor:
            return torch.from_numpy(np.zeros((4, 4), dtype=np.float32))

    import openlithohub.streaming as streaming_pkg
    import openlithohub.streaming.sinks as sinks_pkg

    monkeypatch.setattr(sinks_pkg, "TensorTileSink", FakeSink)

    stream_calls = {"n": 0}

    def fake_run_streaming(source, sink, forward_fn, *, core_size, batch_size=1, **_):
        stream_calls["n"] += 1
        if stream_calls["n"] == 1:  # the CPU reference pass
            forward_fn(torch.from_numpy(np.zeros((8, 8), dtype=np.float32)))
        return types.SimpleNamespace(n_tiles=1, forward_batches=1)

    monkeypatch.setattr(streaming_pkg, "run_streaming", fake_run_streaming)

    cfg = {
        "gds": "",
        "device": "cuda:0",
        "dtype": "fp32",
        "layer": "66:44",
        "window": 256,
        "tile": 64,
        "batch": 4,
        "pixel_nm": 1.0,
        "forward_radius": 4,
        "forward_sigma_nm": 1.6,
        "tf32_matmul": False,
    }
    row = harness.tier_b_worker_once(cfg)
    # the reset calls never raised (the test would have failed) and the
    # shared init helper ran before the first one
    assert init_calls == ["cuda:0"]
    assert row["requested_device"] == "cuda:0"
    for key in (
        "forward_input_device",
        "forward_output_device_before_d2h",
        "cuda_execution_witness_pass",
    ):
        assert key in row
    # no real GPU executions happened under CI mocks — the witness must
    # NOT claim CUDA authority (the row stays FAILED, never faked SUCCESS)
    assert row["cuda_execution_witness_pass"] is False
    assert row["status"] == "FAILED"


def test_tier_b_worker_not_run_environment_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_module("run_v2_benchmark_g2b3_tierb_norun", HARNESS_PATH)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    row = harness.tier_b_worker_once({"device": "cuda:0", "dtype": "fp32"})
    assert row["status"] == "NOT_RUN_ENVIRONMENT"


# ---- GPU Authority Repair §6: preflight shares the worker path -----------------


def test_preflight_tier_c_probe_shares_repaired_worker_constructor() -> None:
    preflight = _load_module("preflight_v2_g2b3", PREFLIGHT_PATH)
    harness = preflight.load_v2_harness()
    params = harness.tier_c_hopkins_params(
        {
            "hopkins_wavelength_nm": 13.5,
            "hopkins_na": 0.33,
            "hopkins_sigma_outer": 0.7,
            "hopkins_sigma_inner": 0.6,
            "hopkins_defocus_nm": 0.0,
            "pixel_nm": 1.0,
        }
    )
    # the probe's constructor is the REPAIRED worker mapping (sigma), so a
    # preflight can never pass while the worker path is broken
    assert params.sigma == 0.7 and params.sigma_inner == 0.6


def test_preflight_tier_c_probe_routes_through_worker_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _load_module("preflight_v2_g2b3_probe", PREFLIGHT_PATH)

    def broken_params(cfg):
        raise RuntimeError("shared-constructor sentinel")

    monkeypatch.setattr(
        preflight,
        "load_v2_harness",
        lambda: types.SimpleNamespace(tier_c_hopkins_params=broken_params),
    )
    # v3 §14: the probe routes through the bounded block-path gate, which
    # still constructs params via the SHARED worker constructor.
    with pytest.raises(RuntimeError, match="shared-constructor sentinel"):
        preflight.bounded_block_probe_gate(preflight.load_v2_harness(), "cuda:0", hopkins_grid=1024)


def test_preflight_tier_b_probe_routes_through_shared_cuda_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _load_module("preflight_v2_g2b3_probeb", PREFLIGHT_PATH)
    monkeypatch.setattr(
        ms,
        "initialize_cuda_measurement_device",
        lambda device: (_ for _ in ()).throw(RuntimeError("shared-init sentinel")),
    )
    with pytest.raises(RuntimeError, match="shared-init sentinel"):
        preflight.tier_b_preflight_probe("cuda:0")


# ---- GPU Authority Repair §8: cross-platform git-state authority ---------------


def test_measurement_git_state_matches_git_rev_parse() -> None:
    commit, clean = ms.measurement_git_state(REPO)
    expected = subprocess.run(  # noqa: S603 — fixed-argv git query
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert commit == expected
    assert isinstance(clean, bool)


def test_measurement_git_state_detects_dirty_tree(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(  # noqa: S603 — test fixture git setup
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (tmp_path / "tracked.txt").write_text("hello")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "init")
    commit, clean = ms.measurement_git_state(tmp_path)
    assert len(commit) == 40 and clean is True
    (tmp_path / "tracked.txt").write_text("dirty")
    _commit, dirty = ms.measurement_git_state(tmp_path)
    assert dirty is False


def test_measurement_git_state_fails_closed_without_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ms.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="git executable not found"):
        ms.measurement_git_state(REPO)


# ---- GPU Authority Repair §9: cross-platform host peak RSS ---------------------


def test_host_peak_rss_posix_dispatch_and_units(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resource = types.ModuleType("resource")
    fake_resource.RUSAGE_SELF = 0
    fake_resource.getrusage = lambda who: types.SimpleNamespace(ru_maxrss=1234)
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(sys, "platform", "linux")
    assert ms.host_peak_rss_bytes() == 1234 * 1024  # Linux ru_maxrss is KiB
    monkeypatch.setattr(sys, "platform", "darwin")
    assert ms.host_peak_rss_bytes() == 1234  # macOS ru_maxrss is bytes


def test_host_peak_rss_windows_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_windows_peak_rss_bytes", lambda: 1700_000_000)
    monkeypatch.setattr(sys, "platform", "win32")
    assert ms.host_peak_rss_bytes() == 1700_000_000


def test_host_peak_rss_positive() -> None:
    assert ms.host_peak_rss_bytes() > 0


# ---- GPU Authority Repair §10: driver/device identity fails closed --------------


def test_gpu_driver_identity_unavailable_when_no_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ms.shutil, "which", lambda _: None)
    monkeypatch.setattr(sys, "platform", "linux")
    identity = ms.gpu_driver_identity()
    assert identity["driver_version"] == ""
    assert identity["driver_identity_source"] == "unavailable"


def test_gpu_driver_identity_nvidia_smi_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_nvidia_smi_driver_version", lambda: "550.54")
    monkeypatch.setattr(
        ms,
        "_stable_device_identifier",
        lambda index, fallback="": {
            "device_identifier": "GPU-abc",
            "device_identifier_type": "cuda-uuid",
        },
    )
    identity = ms.gpu_driver_identity()
    assert identity["driver_identity_source"] == "nvidia-smi"
    assert identity["driver_version"] == "550.54"
    assert identity["device_identifier_type"] == "cuda-uuid"


def test_gpu_driver_identity_windows_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #74/#75: nvidia-smi exists but NVML fails — the source-owned
    Windows registry fallback identifies the driver, never presence."""
    monkeypatch.setattr(ms, "_nvidia_smi_driver_version", lambda: "")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        ms,
        "_windows_display_driver_identity",
        lambda: {
            "driver_description": "NVIDIA GeForce RTX 4090 Laptop GPU",
            "driver_version": "32.0.16.1088",
            "device_identifier": r"PCI\VEN_10DE&DEV_2704",
        },
    )
    identity = ms.gpu_driver_identity()
    assert identity["driver_identity_source"] == "windows-fallback"
    assert identity["driver_version"] == "32.0.16.1088"
    assert identity["device_identifier_type"] == "windows-pnp-matching-device-id"


def test_gpu_driver_identity_windows_fails_closed_when_registry_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ms, "_nvidia_smi_driver_version", lambda: "")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        ms,
        "_windows_display_driver_identity",
        lambda: {"driver_description": "", "driver_version": "", "device_identifier": ""},
    )
    identity = ms.gpu_driver_identity()
    assert identity["driver_identity_source"] == "unavailable"
    assert identity["driver_version"] == ""


# ---- fail-closed formal authority: empty identity / missing witness ------------

_FULL_LOCK = {
    "available": True,
    "count": 1,
    "devices": [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090 Laptop GPU",
            "total_memory_bytes": 17_170_956_288,
            "compute_capability": "8.9",
        }
    ],
    "platform": "Windows-11-10.0.26200-SP0",
    "driver_version": "32.0.16.1088",
    "driver_identity_source": "windows-fallback",
    "device_identifier": "PCI-VEN_10DE-DEV_2704",
    "device_identifier_type": "windows-pnp-matching-device-id",
    "torch_cuda_version": "13.0",
    "cudnn_version": 92400,
    "torch_version": "2.14.0+cu130",
    "tf32_matmul": False,
    "tf32_cudnn": False,
}

_WITNESS = {
    "requested_device": "cuda:0",
    "forward_input_device": "cuda:0",
    "forward_output_device_before_d2h": "cuda:0",
    "cuda_forward_executions": 3,
    "cpu_forward_executions": 0,
    "cuda_execution_witness_pass": True,
}


def _window_row(window: int) -> dict:
    return {
        "window": window,
        "status": "SUCCESS",
        "repeat_count": 5,
        "correctness_witness_pass": True,
        "timing_method": "cuda_synchronized",
        "device": "cuda:0",
        "dtype": "fp32",
        "max_memory_allocated": 1000,
        "max_memory_reserved": 2000,
        "host_peak_rss_bytes": 3000,
        "gpu_batch1_wall_s": 1.0,
        "gpu_batch_n_wall_s": 0.4,
        "aggregate_median_s": 0.4,
        **{k: _WITNESS[k] for k in _WITNESS if not k.endswith("executions")},
    }


def _tier_c_row() -> dict:
    return {
        "tier": "c",
        "window": 1024,
        "status": "SUCCESS",
        "repeat_count": 5,
        "aggregate_n": 5,
        "aggregate_median_s": 0.02,
        "aggregate_p10_s": 0.018,
        "aggregate_p90_s": 0.024,
        "correctness_witness_pass": True,
        "timing_method": "cuda_synchronized",
        "device": "cuda:0",
        "dtype": "fp32",
        "device_requires_cuda": True,
        "max_memory_allocated": 500,
        "max_memory_reserved": 800,
        "host_peak_rss_bytes": 3000,
        "warmup_executions": 2,
        "timing_observations": 1,
        "grid": 1024,
        "requested_device": "cuda:0",
        "forward_input_device": "cuda:0",
        "forward_output_device_before_d2h": "cuda:0",
        "cuda_execution_witness_pass": True,
        "strategy": "exact_block_gram_topk_v1",
        "strategy_version": 1,
        "memory_plan_sha256": "e" * 64,
        "memory_feasible": True,
        "selected_chunk_columns": 65536,
        "chunk_count": 16,
        "n_src": 1609,
        "kernel_count": 24,
        "memory_plan_peak_witness_pass": True,
        "worker_environment_witness_pass": True,
    }


def _tier_a_row(window: int) -> dict:
    return {
        "window": window,
        "status": "SUCCESS",
        "repeat_count": 5,
        "correctness_witness_pass": True,
        "timing_method": "host_perf_counter",
        "device_requires_cuda": False,
        "aggregate_median_s": 0.1,
    }


def _g2b3_workspace(tmp_path: Path, *, env_lock: dict, tier_b_witness: bool = True):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    run_config = RunConfigV2(
        tiers=("a", "b", "c"),
        window_sizes=(4096, 8192),
        repeat_count=5,
        fixture_sha256="d" * 64,
        layer="66:44",
        socs_memory_plan_sha256="e" * 64,
    )
    write_strict_json(
        workspace / "socs-memory-plan.json",
        {
            "schema": "OpenLithoHub.socs-memory-plan.v1",
            "strategy": "exact_block_gram_topk_v1",
            "strategy_version": 1,
            "plan_sha256": "e" * 64,
            "memory_feasible": True,
            "selected_chunk_columns": 65536,
            "chunk_count": 16,
        },
    )
    source = {"harness": "h", "core": "c", "claim_generator": "g", "verifier": "v"}
    lock_sha = gpu_environment_lock_sha256(env_lock)
    identity = compute_run_identity_v2(
        run_config,
        measurement_commit="a" * 40,
        harness_sha256="h",
        core_sha256="c",
        claim_generator_sha256="g",
        verifier_sha256="v",
        environment_lock_sha256=lock_sha,
    )
    b_row = _window_row(4096)
    if not tier_b_witness:
        for key in (
            "requested_device",
            "forward_input_device",
            "forward_output_device_before_d2h",
            "cuda_execution_witness_pass",
        ):
            b_row.pop(key, None)
    write_strict_json(
        workspace / "tier-a.json",
        {
            "tier": "a",
            "status": "SUCCESS",
            "correctness_witness_pass": True,
            "repeats_recorded": 10,
            "window_rows": [_tier_a_row(4096), _tier_a_row(8192)],
        },
    )
    write_strict_json(
        workspace / "tier-b.json",
        {
            "tier": "b",
            "status": "SUCCESS",
            "correctness_witness_pass": True,
            "repeats_recorded": 10,
            "window_rows": [b_row, _window_row(8192)],
        },
    )
    write_strict_json(
        workspace / "tier-c.json",
        {
            "tier": "c",
            "status": "SUCCESS",
            "correctness_witness_pass": True,
            "repeats_recorded": 5,
            **_tier_c_row(),
        },
    )
    rows = {k: json.loads((workspace / f"tier-{k}.json").read_text()) for k in ("a", "b", "c")}
    return workspace, run_config, identity, source, env_lock, rows


def _build_family(tmp_path: Path, *, env_lock: dict, tier_b_witness: bool = True):
    workspace, run_config, identity, source, env_lock, rows = _g2b3_workspace(
        tmp_path, env_lock=env_lock, tier_b_witness=tier_b_witness
    )
    blockers = build_canonical_family_in_workspace(
        workspace_dir=workspace,
        run_config=run_config,
        run_identity=identity,
        measurement_commit="a" * 40,
        source_hashes=source,
        environment_lock=env_lock,
        tier_rows=rows,
        tracked_tree_clean=True,
        provisional=False,
        socs_memory_plan={
            "schema": "OpenLithoHub.socs-memory-plan.v1",
            "strategy": "exact_block_gram_topk_v1",
            "strategy_version": 1,
            "plan_sha256": "e" * 64,
            "memory_feasible": True,
            "selected_chunk_columns": 65536,
            "chunk_count": 16,
        },
    )
    return workspace, identity, blockers


def test_builder_accepts_witnessed_rows_with_source_owned_identity(tmp_path: Path) -> None:
    _workspace, _identity, blockers = _build_family(tmp_path, env_lock=dict(_FULL_LOCK))
    assert blockers == [], blockers


def test_builder_rejects_unavailable_driver_identity(tmp_path: Path) -> None:
    lock = {**_FULL_LOCK, "driver_identity_source": "unavailable", "driver_version": ""}
    _workspace, _identity, blockers = _build_family(tmp_path, env_lock=lock)
    assert any("nonempty driver_version" in b for b in blockers)
    assert any("source-owned driver" in b for b in blockers)


def test_builder_rejects_missing_cuda_execution_witness(tmp_path: Path) -> None:
    _workspace, _identity, blockers = _build_family(
        tmp_path, env_lock=dict(_FULL_LOCK), tier_b_witness=False
    )
    assert any("CUDA execution witness" in b for b in blockers), blockers


# ---- the formal script verifier rejects a tampered (witness-stripped) family ----


def _reseal_family(root: Path, run_identity: str) -> None:
    """Recompute manifest + SHA256SUMS after a deliberate tamper."""
    members = sorted(p.name for p in root.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    write_strict_json(
        root / "manifest.json",
        {
            "run_identity": run_identity,
            "members": [
                {"name": name, "bytes": (root / name).stat().st_size}
                for name in members
                if name != "manifest.json"
            ],
        },
    )
    lines = [
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}" for name in members
    ]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def test_v2_verifier_rejects_missing_cuda_execution_witness(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "verify_v2_g2b3", REPO / "scripts" / "verify_industrial_v2_artifacts.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    from openlithohub.benchmark.industrial_v2 import CANONICAL_FAMILY

    workspace, identity, blockers = _build_family(tmp_path, env_lock=dict(_FULL_LOCK))
    assert blockers == [], blockers
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    for member in sorted(CANONICAL_FAMILY):
        (canonical / member).write_text((workspace / member).read_text())
    verifier.verify(canonical)  # the witnessed family passes

    # tamper: strip the CUDA execution witness from every locked GPU row
    witness_keys = (
        "requested_device",
        "forward_input_device",
        "forward_output_device_before_d2h",
        "cuda_execution_witness_pass",
    )
    payload = json.loads((canonical / "industrial-v2-gpu-runtime.json").read_text())
    for row in payload["rows"]:
        for key in witness_keys:
            row.pop(key, None)
        for repeat_row in row.get("window_rows", []):
            for key in witness_keys:
                repeat_row.pop(key, None)
    write_strict_json(canonical / "industrial-v2-gpu-runtime.json", payload)
    _reseal_family(canonical, identity)
    with pytest.raises(verifier.VerifyError, match="CUDA execution witness|witness key"):
        verifier.verify(canonical)
