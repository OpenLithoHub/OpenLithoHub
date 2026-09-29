"""PR-S7 hostile tests — Scale actual-CUDA / verifier / campaign repair.

Covers the GPU Authority Repair §14-§26 Scale repairs:

* Lane A/C route the tile/batch tensor to the target CUDA device and
  record the actual-CUDA execution witness (§14/§15);
* the formal verifier rejects missing/false CUDA witnesses and CPU
  actual-forwards under a cuda backend (§15);
* Lane B workers forward on their ASSIGNED device and an
  assignment/execution mismatch is a hard failure (§16);
* the verifier's Lane-C ladder semantics are scoped to the run config —
  a Lane A-only family with an EMPTY saturation member PASSES (§18);
* the Lane-A campaign binds exactly the four frozen windows with one
  source/fixture/environment/protocol — mixed anything FAILS (§19);
* evidence packaging is deterministic with sink buffers recorded by
  SHA-256 + byte count only (§25).

CPU CI proves structure and routing with mocks — it never claims real
GPU validation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest
import torch

from openlithohub.benchmark.industrial_scale import (
    FROZEN_LANE_A_WINDOWS,
    FROZEN_MICROBATCH_LADDER,
    LANE_A,
    LANE_B,
    LANE_C,
    ScaleRunConfig,
    compute_scale_run_identity,
    formal_scale_blockers,
    scale_environment_lock_sha256,
    scale_host_policy_blockers,
    write_strict_json,
)
from openlithohub.benchmark.industrial_scale_campaign import (
    LANE_A_CAMPAIGN_SCHEMA,
    build_lane_a_campaign,
    campaign_manifest_sha256,
    verify_lane_a_campaign,
)

REPO = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO / "benchmarks" / "industrial-scale" / "run_scale_benchmark.py"
MULTI_WORKER_PATH = REPO / "benchmarks" / "industrial-scale" / "multi_worker.py"
SCALE_VERIFIER_PATH = REPO / "scripts" / "verify_industrial_scale_artifacts.py"
CAMPAIGN_BUILDER = REPO / "scripts" / "build_industrial_scale_lane_a_campaign.py"
CAMPAIGN_VERIFIER = REPO / "scripts" / "verify_industrial_scale_lane_a_campaign.py"
PACKAGING_SCRIPT = REPO / "scripts" / "package_gpu_authority_evidence.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass resolution needs the module registered under its name
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_scale_verifier():
    return _load_module("verify_scale_s7", SCALE_VERIFIER_PATH)


def _load_harness():
    return _load_module("run_scale_s7_harness", HARNESS_PATH)


# ---- §14/§15: the measured path moves the tile/batch to CUDA + witness ---------


class FakeTile:
    """A tensor stand-in: ``.to(device)`` records the movement, so the
    test proves the wrapper moves the tile to the target device and that
    the witness sees the output BEFORE the D2H copy."""

    def __init__(self, device: str) -> None:
        self.device = device

    def to(self, device: str) -> FakeTile:
        return FakeTile(device)

    def cpu(self) -> FakeTile:
        return FakeTile("cpu")


def test_stream_once_moves_tile_to_target_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§14: the tile is EXPLICITLY moved to cuda:0 before the forward —
    caching kernels without moving the input (the #75 defect) is gone."""
    harness = _load_harness()
    moved_to: list[str] = []

    def recording_forward(tile):
        moved_to.append(tile.device)
        return tile

    def fake_run_streaming(
        source,
        sink,
        forward_fn,
        *,
        core_size,
        pixel_nm=1.0,
        batch_size=1,
        batched_forward_fn=None,
        **_,
    ):
        forward_fn(FakeTile("cpu"))  # the spine hands the window path a CPU tile
        return types.SimpleNamespace(n_tiles=1, forward_batches=1)

    import openlithohub.benchmark.measurement_support as ms
    import openlithohub.streaming as streaming_pkg

    monkeypatch.setattr(streaming_pkg, "run_streaming", fake_run_streaming)
    monkeypatch.setattr(ms, "initialize_cuda_measurement_device", lambda device: device)
    monkeypatch.setattr(
        torch,
        "cuda",
        types.SimpleNamespace(
            reset_peak_memory_stats=lambda device: None,
            synchronize=lambda device: None,
            max_memory_allocated=lambda device: 1024,
            max_memory_reserved=lambda device: 2048,
        ),
    )

    one = harness.stream_once(
        source=types.SimpleNamespace(shape=(64, 64)),
        forward_fn=recording_forward,
        batched_forward_fn=None,
        sink=types.SimpleNamespace(),
        tile=32,
        microbatch=1,
        queue_depth=4,
        device="cuda:0",
        pixel_nm=1.0,
    )
    # the tile ACTUALLY reached the target device (moved via .to(device))
    assert moved_to == ["cuda:0"]
    assert one["requested_device"] == "cuda:0"
    assert one["forward_input_device"] == "cuda:0"
    assert one["forward_output_device_before_d2h"] == "cuda:0"
    assert one["cuda_execution_witness_pass"] is True
    assert one["forward_witness"]["cuda_forward_executions"] == 1


def test_lane_c_batched_path_moves_batch_to_target_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§14: the BATCHED path moves the batch tensor to cuda:0 too."""
    harness = _load_harness()
    moved_to: list[str] = []

    def recording_forward(batch):
        moved_to.append(batch.device)
        return batch

    def fake_run_streaming(
        source,
        sink,
        forward_fn,
        *,
        core_size,
        pixel_nm=1.0,
        batch_size=1,
        batched_forward_fn=None,
        **_,
    ):
        assert batched_forward_fn is not None
        batched_forward_fn(FakeTile("cpu"))
        return types.SimpleNamespace(n_tiles=2, forward_batches=1)

    import openlithohub.benchmark.measurement_support as ms
    import openlithohub.streaming as streaming_pkg

    monkeypatch.setattr(streaming_pkg, "run_streaming", fake_run_streaming)
    monkeypatch.setattr(ms, "initialize_cuda_measurement_device", lambda device: device)
    monkeypatch.setattr(
        torch,
        "cuda",
        types.SimpleNamespace(
            reset_peak_memory_stats=lambda device: None,
            synchronize=lambda device: None,
            max_memory_allocated=lambda device: 1024,
            max_memory_reserved=lambda device: 2048,
        ),
    )

    one = harness.stream_once(
        source=types.SimpleNamespace(shape=(64, 64)),
        forward_fn=recording_forward,
        batched_forward_fn=recording_forward,
        sink=types.SimpleNamespace(),
        tile=32,
        microbatch=4,
        queue_depth=4,
        device="cuda:0",
        pixel_nm=1.0,
    )
    assert moved_to == ["cuda:0"]
    assert one["cuda_execution_witness_pass"] is True


def test_cpu_request_records_honest_non_cuda_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cpu request records requested_device=cpu and witness pass False —
    a synchronized timer around CPU work is never GPU authority."""
    harness = _load_harness()

    def fake_run_streaming(
        source,
        sink,
        forward_fn,
        *,
        core_size,
        pixel_nm=1.0,
        batch_size=1,
        batched_forward_fn=None,
        **_,
    ):
        forward_fn(FakeTile("cpu"))
        return types.SimpleNamespace(n_tiles=1, forward_batches=0)

    import openlithohub.streaming as streaming_pkg

    monkeypatch.setattr(streaming_pkg, "run_streaming", fake_run_streaming)
    one = harness.stream_once(
        source=types.SimpleNamespace(shape=(64, 64)),
        forward_fn=lambda tile: tile,
        batched_forward_fn=None,
        sink=types.SimpleNamespace(),
        tile=32,
        microbatch=1,
        queue_depth=4,
        device="cpu",
        pixel_nm=1.0,
    )
    assert one["requested_device"] == "cpu"
    assert one["cuda_execution_witness_pass"] is False


# ---- §16: Lane B uses the assigned device; mismatch is a hard failure ----------


class FakeSource:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape

    def read_window(self, bbox):
        return torch.zeros((bbox.height, bbox.width))


class NullSink:
    def write_core(self, tile_id, bbox, core, meta):
        pass


def test_lane_b_worker_forwards_on_assigned_device() -> None:
    multi_worker = _load_module("multi_worker_s7", MULTI_WORKER_PATH)
    forward_inputs: list[str] = []
    factory_devices: list[tuple[int, str]] = []

    def forward_factory(worker_index: int, device: str):
        factory_devices.append((worker_index, device))

        def forward(tile):
            forward_inputs.append(str(tile.device))
            return tile

        return forward

    report = multi_worker.run_multi_worker_stream(
        source=FakeSource((32, 32)),
        sink=NullSink(),
        forward_fn=lambda tile: tile,
        core_size=16,
        halo_px=2,
        worker_count=2,
        queue_depth=4,
        device_for_worker=lambda index: "cpu",
        forward_factory=forward_factory,
    )
    assert report.n_tiles > 0
    assert report.worker_assigned_devices == {"worker0": "cpu", "worker1": "cpu"}
    assert report.worker_forward_devices == {"worker0": "cpu", "worker1": "cpu"}
    assert factory_devices == [(0, "cpu"), (1, "cpu")]
    assert set(forward_inputs) == {"cpu"}
    assert report.device_binding_pass is True


def test_lane_b_assignment_execution_mismatch_propagates_failure() -> None:
    """A worker ASSIGNED cuda but executing elsewhere (here: no CUDA on
    the CI host) is a hard failure — never a silent success."""
    multi_worker = _load_module("multi_worker_s7_mismatch", MULTI_WORKER_PATH)
    with pytest.raises(multi_worker.WorkerFailureError):
        multi_worker.run_multi_worker_stream(
            source=FakeSource((32, 32)),
            sink=NullSink(),
            forward_fn=lambda tile: tile,
            core_size=16,
            halo_px=2,
            worker_count=1,
            queue_depth=4,
            device_for_worker=lambda index: "cuda:0",
        )


# ---- synthetic formal family factory (verifier + campaign hostile tests) -------


def _cuda_env_lock() -> dict:
    return {
        "available": True,
        "count": 1,
        "requested_gpu_count": 1,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090 Laptop GPU",
                "total_memory_bytes": 17_170_956_288,
                "compute_capability": "8.9",
                "uuid": "",
            }
        ],
        "platform": "Windows-11-10.0.26200-SP0",
        "platform_system": "Windows",
        "driver_version": "32.0.16.1088",
        "driver_identity_source": "windows-fallback",
        "device_identifier": r"PCI\VEN_10DE&DEV_2704",
        "device_identifier_type": "windows-pnp-matching-device-id",
        "torch_cuda_version": "13.0",
        "torch_version": "2.14.0+cu130",
        "cudnn_version": 92400,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "topology": [],
    }


def _lane_row(
    *,
    window: int,
    lane: str,
    microbatch: int = 8,
    witness: bool = True,
    input_device: str = "cuda:0",
) -> dict:
    return {
        "lane": lane,
        "window": window,
        "worker_count": 1,
        "gpu_count": 1,
        "device_backend": "cuda",
        "microbatch": microbatch,
        "forward_profile": "P1_FINITE_SUPPORT",
        "sink_kind": "memmap_npy",
        "device": "cuda:0",
        "device_requires_cuda": True,
        "timing_method": "cuda_synchronized",
        "headline_eligible": True,
        "warmup_discarded": 2,
        "repeat_count": 5,
        "correctness_witness_pass": True,
        "finite_witness": True,
        "aggregate_n": 5,
        "aggregate_median_s": 0.5,
        "aggregate_p10_s": 0.4,
        "aggregate_p90_s": 0.6,
        "parse_wall_s": 1.0,
        "executed_window": window,
        "full_die": False,
        "layout_equivalent_pixels": window * window,
        "forwarded_pixels": window * window * 5,
        "n_tiles": max(1, (window // 1024) ** 2),
        "n_forward_batches": 1,
        "output_bytes": window * window * 4,
        "peak_host_rss_bytes": 2_000_000_000,
        "max_memory_allocated": 1024,
        "max_memory_reserved": 2048,
        "throughput_gpx_s": 0.001,
        "per_repeat": [],
        "status": "SUCCESS",
        "requested_device": "cuda:0",
        "forward_input_device": input_device,
        "forward_output_device_before_d2h": "cuda:0",
        "cuda_execution_witness_pass": witness,
    }


def _fixture_manifest_payload(gds_sha256: str) -> dict:
    return {
        "schema": "OpenLithoHub.scale-fixture.v1",
        "source_repository": "SJTU-YONGFU-RESEARCH-GRP/PDB-Physical-Design-Database",
        "source_commit": "9e1e3399b1b707f26fee853bce1ff91ab466ce24",
        "design": "ibex",
        "gds_sha256": gds_sha256,
        "gds_bytes": 25_251_340,
        "top_cell": "ibex_core",
        "dbu_nm": 1.0,
        "bbox_dbu": [0, 0, 4096, 4096],
        "pixel_nm": 1.0,
        "die_size_px": [4096, 4096],
        "equivalent_pixels": 4096 * 4096,
        "dense_float32_equivalent_bytes": 4096 * 4096 * 4,
        "layers": ["66:44"],
        "selected_layer": "66:44",
        "preparation_script_sha256": "p" * 64,
        "split_chunks": [],
    }


def _build_formal_family(
    tmp_path: Path,
    *,
    window: int,
    lanes: tuple[str, ...] = (LANE_A,),
    lane_c_rungs: tuple[int, ...] | None = None,
    row_witness: bool = True,
    row_input_device: str = "cuda:0",
    measurement_commit: str = "a" * 40,
    fixture_gds_sha256: str = "d" * 64,
    env_lock: dict | None = None,
    tile_size: int = 1024,
    microbatch: int = 8,
    label: str = "",
) -> Path:
    """One formal verifier-closed family via the REAL fail-closed builder."""
    harness = _load_harness()
    workspace = tmp_path / f"ws-{window}{label}"
    family_dir = workspace / "family"
    family_dir.mkdir(parents=True)
    env_lock = env_lock if env_lock is not None else _cuda_env_lock()
    env_lock_sha = scale_environment_lock_sha256(env_lock)
    run_config = ScaleRunConfig(
        lanes=lanes,
        device_backend="cuda",
        gpu_count=1,
        worker_count=1,
        tile_size=tile_size,
        halo_px=64,
        microbatch=microbatch,
        microbatch_ladder=tuple(lane_c_rungs) if lane_c_rungs else (),
        queue_depth=64,
        forward_profile="P1_FINITE_SUPPORT",
        sink_kind="memmap_npy",
        warmup_count=2,
        repeat_count=5,
        window_sizes=(window,),
        fixture_manifest_sha256="f" * 64,
        fixture_gds_sha256=fixture_gds_sha256,
        selected_layer="66:44",
        pixel_nm=1.0,
    )
    identity = compute_scale_run_identity(
        run_config,
        measurement_commit=measurement_commit,
        harness_sha256="h" * 64,
        core_sha256="c" * 64,
        verifier_sha256="v" * 64,
        claim_generator_sha256="g" * 64,
        fixture_manifest_sha256="f" * 64,
        environment_lock_sha256=env_lock_sha,
    )
    source_hashes = {
        "harness": "h" * 64,
        "core": "c" * 64,
        "verifier": "v" * 64,
        "claim_generator": "g" * 64,
    }
    lane_rows: dict[str, list[dict]] = {}
    if LANE_A in lanes:
        lane_rows[LANE_A] = [
            _lane_row(
                window=window, lane=LANE_A, witness=row_witness, input_device=row_input_device
            )
        ]
    if LANE_C in lanes:
        rungs = lane_c_rungs or FROZEN_MICROBATCH_LADDER
        lane_rows[LANE_C] = [
            _lane_row(
                window=window,
                lane=LANE_C,
                microbatch=rung,
                witness=row_witness,
                input_device=row_input_device,
            )
            for rung in rungs
        ]
    if LANE_B in lanes:
        lane_rows[LANE_B] = [
            _lane_row(
                window=window, lane=LANE_B, witness=row_witness, input_device=row_input_device
            )
        ]
    blockers = harness.build_scale_family_in_workspace(
        workspace_dir=workspace,
        family_dir=family_dir,
        identity=identity,
        measurement_commit=measurement_commit,
        tracked_tree_clean=True,
        provisional=False,
        source_hashes=source_hashes,
        env_lock=env_lock,
        env_lock_sha=env_lock_sha,
        run_config_payload=run_config.to_payload(),
        fixture_manifest_payload=_fixture_manifest_payload(fixture_gds_sha256),
        lane_rows=lane_rows,
    )
    assert blockers == [], blockers
    return family_dir


def _reseal_family(family_dir: Path) -> None:
    """Recompute manifest + SHA256SUMS after a deliberate tamper."""
    members = sorted(
        p.name for p in family_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"
    )
    run_identity = json.loads((family_dir / "industrial-scale-run-config.json").read_text())[
        "run_identity"
    ]
    write_strict_json(
        family_dir / "manifest.json",
        {
            "run_identity": run_identity,
            "members": [
                {"name": name, "bytes": (family_dir / name).stat().st_size}
                for name in members
                if name != "manifest.json"
            ],
        },
    )
    lines = [
        f"{hashlib.sha256((family_dir / name).read_bytes()).hexdigest()}  {name}"
        for name in members
    ]
    (family_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def _tamper_lane_member(family_dir: Path, member: str, mutate) -> None:
    """Apply `mutate` to a lane member's rows, then reseal the family."""
    path = family_dir / member
    payload = json.loads(path.read_text())
    mutate(payload)
    write_strict_json(path, payload)
    _reseal_family(family_dir)


# ---- §18: verifier Lane-C scope -------------------------------------------------


def test_formal_lane_a_only_family_passes_with_empty_lane_c(tmp_path: Path) -> None:
    """THE historical defect (§18): a formal Lane A-only family has an
    empty saturation member and MUST pass — the verifier must not demand
    the frozen ladder from a run that never declared Lane C."""
    verifier = _load_scale_verifier()
    family = _build_formal_family(tmp_path, window=4096, lanes=(LANE_A,))
    claims = verifier.verify(family, require_formal=True)
    assert any(LANE_A in c for c in claims), claims


def test_formal_lane_c_full_ladder_passes(tmp_path: Path) -> None:
    verifier = _load_scale_verifier()
    family = _build_formal_family(
        tmp_path, window=8192, lanes=(LANE_C,), lane_c_rungs=FROZEN_MICROBATCH_LADDER
    )
    claims = verifier.verify(family, require_formal=True)
    assert any("6 rungs" in c for c in claims), claims


def test_builder_refuses_lane_c_ladder_drift(tmp_path: Path) -> None:
    """The fail-closed builder itself refuses a Lane C run whose ladder
    drifted from the frozen one — the violation cannot even be written."""
    harness = _load_harness()
    workspace = tmp_path / "ws-drift"
    family_dir = workspace / "family"
    family_dir.mkdir(parents=True)
    env_lock = _cuda_env_lock()
    env_lock_sha = scale_environment_lock_sha256(env_lock)
    run_config = ScaleRunConfig(
        lanes=(LANE_C,),
        device_backend="cuda",
        gpu_count=1,
        window_sizes=(8192,),
        repeat_count=5,
        warmup_count=2,
        microbatch_ladder=(1, 2, 4, 8, 16, 64),
        fixture_manifest_sha256="f" * 64,
        fixture_gds_sha256="d" * 64,
        selected_layer="66:44",
    )
    identity = compute_scale_run_identity(
        run_config,
        measurement_commit="a" * 40,
        harness_sha256="h" * 64,
        core_sha256="c" * 64,
        verifier_sha256="v" * 64,
        claim_generator_sha256="g" * 64,
        fixture_manifest_sha256="f" * 64,
        environment_lock_sha256=env_lock_sha,
    )
    blockers = harness.build_scale_family_in_workspace(
        workspace_dir=workspace,
        family_dir=family_dir,
        identity=identity,
        measurement_commit="a" * 40,
        tracked_tree_clean=True,
        provisional=False,
        source_hashes={
            "harness": "h" * 64,
            "core": "c" * 64,
            "verifier": "v" * 64,
            "claim_generator": "g" * 64,
        },
        env_lock=env_lock,
        env_lock_sha=env_lock_sha,
        run_config_payload=run_config.to_payload(),
        fixture_manifest_payload=_fixture_manifest_payload("d" * 64),
        lane_rows={
            LANE_C: [
                _lane_row(window=8192, lane=LANE_C, microbatch=rung)
                for rung in (1, 2, 4, 8, 16, 64)
            ]
        },
    )
    assert any("frozen" in b for b in blockers), blockers


def test_formal_lane_c_missing_rung_fails(tmp_path: Path) -> None:
    verifier = _load_scale_verifier()
    family = _build_formal_family(
        tmp_path, window=8192, lanes=(LANE_C,), lane_c_rungs=FROZEN_MICROBATCH_LADDER
    )

    def drop_rung(payload):
        payload["rows"] = [r for r in payload["rows"] if int(r["microbatch"]) != 16]

    _tamper_lane_member(family, "industrial-scale-saturation.json", drop_rung)
    with pytest.raises(verifier.VerifyError, match="ladder|rungs"):
        verifier.verify(family, require_formal=True)


def test_formal_lane_c_duplicate_rung_fails(tmp_path: Path) -> None:
    verifier = _load_scale_verifier()
    family = _build_formal_family(
        tmp_path, window=8192, lanes=(LANE_C,), lane_c_rungs=FROZEN_MICROBATCH_LADDER
    )

    def duplicate_rung(payload):
        payload["rows"] = payload["rows"] + [dict(payload["rows"][-1])]

    _tamper_lane_member(family, "industrial-scale-saturation.json", duplicate_rung)
    with pytest.raises(verifier.VerifyError, match="ladder|rungs"):
        verifier.verify(family, require_formal=True)


def test_formal_lane_c_drifted_ladder_fails(tmp_path: Path) -> None:
    verifier = _load_scale_verifier()
    family = _build_formal_family(
        tmp_path, window=8192, lanes=(LANE_C,), lane_c_rungs=FROZEN_MICROBATCH_LADDER
    )

    def drift_rung(payload):
        payload["rows"][-1]["microbatch"] = 64

    _tamper_lane_member(family, "industrial-scale-saturation.json", drift_rung)
    with pytest.raises(verifier.VerifyError, match="ladder|rungs"):
        verifier.verify(family, require_formal=True)


def test_structural_lane_c_rows_without_declared_lane_c_fail(tmp_path: Path) -> None:
    """Lane-C rows in a family whose run config never declared Lane C are
    an inconsistent family — fail closed on the structural tier too."""
    verifier = _load_scale_verifier()
    family = _build_formal_family(tmp_path, window=4096, lanes=(LANE_A,))

    def inject_lane_c(payload):
        payload["lane"] = LANE_C
        payload["rows"] = [_lane_row(window=4096, lane=LANE_C, microbatch=1)]

    _tamper_lane_member(family, "industrial-scale-saturation.json", inject_lane_c)
    with pytest.raises(verifier.VerifyError, match="not in the.*run config lanes"):
        verifier.verify(family)


# ---- §15: the formal verifier rejects missing/false CUDA witnesses --------------


def test_formal_verifier_rejects_missing_cuda_witness(tmp_path: Path) -> None:
    verifier = _load_scale_verifier()
    family = _build_formal_family(tmp_path, window=4096, lanes=(LANE_A,))

    def strip_witness(payload):
        for row in payload["rows"]:
            for key in (
                "requested_device",
                "forward_input_device",
                "forward_output_device_before_d2h",
                "cuda_execution_witness_pass",
            ):
                row.pop(key, None)

    _tamper_lane_member(family, "industrial-scale-index.json", strip_witness)
    with pytest.raises(verifier.VerifyError, match="witness"):
        verifier.verify(family, require_formal=True)


def test_formal_verifier_rejects_false_cuda_witness(tmp_path: Path) -> None:
    """A CPU actual-forward under a cuda backend (the #75 defect) is
    rejected: the recorded input device is cpu, so the witness is false."""
    verifier = _load_scale_verifier()
    family = _build_formal_family(tmp_path, window=4096, lanes=(LANE_A,))

    def cpu_forward(payload):
        for row in payload["rows"]:
            row["forward_input_device"] = "cpu"
            row["cuda_execution_witness_pass"] = False

    _tamper_lane_member(family, "industrial-scale-index.json", cpu_forward)
    with pytest.raises(verifier.VerifyError, match="witness"):
        verifier.verify(family, require_formal=True)


def test_core_formal_blockers_reject_missing_witness() -> None:
    run_config = ScaleRunConfig(
        lanes=(LANE_A,),
        device_backend="cuda",
        gpu_count=1,
        repeat_count=5,
        window_sizes=(4096,),
        fixture_gds_sha256="d" * 64,
        selected_layer="66:44",
    )
    row = _lane_row(window=4096, lane=LANE_A, witness=False)
    blockers = formal_scale_blockers(
        env_lock=_cuda_env_lock(),
        run_config=run_config,
        lane_rows={LANE_A: row},
        git_clean=True,
        provisional=False,
    )
    assert any("CUDA execution witness" in b for b in blockers), blockers


# ---- §22: the frozen formal Scale host policy ------------------------------------


def test_host_policy_accepts_reference_4090_laptop_host() -> None:
    blockers = scale_host_policy_blockers(_cuda_env_lock(), gpu_count=1)
    assert blockers == [], blockers


@pytest.mark.parametrize(
    "mutation,fragment",
    [
        ({"platform_system": "Darwin"}, "supported formal Scale host"),
        (
            {
                "devices": [
                    {
                        "name": "NVIDIA GeForce RTX 3080",
                        "total_memory_bytes": 10_737_418_240,
                        "compute_capability": "8.6",
                    }
                ]
            },
            "supported formal Scale GPU",
        ),
        (
            {
                "devices": [
                    {
                        "name": "NVIDIA GeForce RTX 4090 Laptop GPU",
                        "total_memory_bytes": 8 * 1024**3,
                        "compute_capability": "8.9",
                    }
                ]
            },
            "VRAM",
        ),
        (
            {
                "devices": [
                    {
                        "name": "NVIDIA GeForce RTX 4090 Laptop GPU",
                        "total_memory_bytes": 17_170_956_288,
                        "compute_capability": "8.6",
                    }
                ]
            },
            "compute capability",
        ),
        ({"devices": []}, "no CUDA devices"),
    ],
)
def test_host_policy_rejects_unsupported_hosts(mutation: dict, fragment: str) -> None:
    lock = {**_cuda_env_lock(), **mutation}
    blockers = scale_host_policy_blockers(lock, gpu_count=1)
    assert any(fragment in b for b in blockers), blockers


def test_formal_blockers_reject_unavailable_driver_identity() -> None:
    run_config = ScaleRunConfig(
        lanes=(LANE_A,),
        device_backend="cuda",
        gpu_count=1,
        repeat_count=5,
        window_sizes=(4096,),
        fixture_gds_sha256="d" * 64,
        selected_layer="66:44",
    )
    lock = {
        **_cuda_env_lock(),
        "driver_version": "",
        "driver_identity_source": "unavailable",
    }
    blockers = formal_scale_blockers(
        env_lock=lock,
        run_config=run_config,
        lane_rows={LANE_A: _lane_row(window=4096, lane=LANE_A)},
        git_clean=True,
        provisional=False,
    )
    assert any("driver_version" in b for b in blockers)
    assert any("identity" in b for b in blockers)


# ---- §19: the Lane-A window-scoped campaign ---------------------------------------


def _four_families(tmp_path: Path, **overrides) -> dict[int, Path]:
    families: dict[int, Path] = {}
    for window in FROZEN_LANE_A_WINDOWS:
        families[window] = _build_formal_family(
            tmp_path, window=window, lanes=(LANE_A,), **overrides
        )
    return families


def test_campaign_build_and_verify_roundtrip(tmp_path: Path) -> None:
    """Each frozen window is its own family; the campaign binds exactly
    the four windows under one authority and re-verifies them."""
    families = _four_families(tmp_path)
    manifest = build_lane_a_campaign(families)
    assert manifest["schema"] == LANE_A_CAMPAIGN_SCHEMA
    assert manifest["frozen_windows"] == list(FROZEN_LANE_A_WINDOWS)
    assert manifest["protocol"]["forward_profile"] == "P1_FINITE_SUPPORT"
    assert manifest["protocol"]["tile_size"] == 1024
    assert manifest["protocol"]["selected_layer"] == "66:44"
    assert [m["window"] for m in manifest["members"]] == list(FROZEN_LANE_A_WINDOWS)
    assert all(m["verifier_pass"] for m in manifest["members"])
    blockers = verify_lane_a_campaign(manifest, families)
    assert blockers == [], blockers
    sha = campaign_manifest_sha256(manifest)
    assert len(sha) == 64


def test_campaign_rejects_missing_window(tmp_path: Path) -> None:
    families = _four_families(tmp_path)
    families.pop(16384)
    with pytest.raises(ValueError, match="frozen Lane-A ladder"):
        build_lane_a_campaign(families)


def test_campaign_rejects_duplicate_window(tmp_path: Path) -> None:
    families = _four_families(tmp_path)
    manifest = dict(build_lane_a_campaign(families))
    manifest["members"] = manifest["members"] + [manifest["members"][0]]
    blockers = verify_lane_a_campaign(manifest, families)
    assert any("exactly once" in b for b in blockers), blockers


def test_campaign_rejects_mixed_source(tmp_path: Path) -> None:
    families = _four_families(tmp_path)
    manifest = build_lane_a_campaign(families)
    # swap ONE window's family for one built under a different source SHA
    other = _build_formal_family(
        tmp_path,
        window=8192,
        lanes=(LANE_A,),
        measurement_commit="b" * 40,
        label="-other",
    )
    swapped = dict(families)
    swapped[8192] = other
    blockers = verify_lane_a_campaign(manifest, swapped)
    assert any("mixed measurement source" in b for b in blockers), blockers


def test_campaign_rejects_mixed_env_lock(tmp_path: Path) -> None:
    families = _four_families(tmp_path)
    manifest = build_lane_a_campaign(families)
    other_lock = {**_cuda_env_lock(), "driver_version": "999.99"}
    other = _build_formal_family(
        tmp_path, window=32768, lanes=(LANE_A,), env_lock=other_lock, label="-other"
    )
    swapped = dict(families)
    swapped[32768] = other
    blockers = verify_lane_a_campaign(manifest, swapped)
    assert any("environment-lock" in b for b in blockers), blockers


def test_campaign_rejects_mixed_fixture(tmp_path: Path) -> None:
    families = _four_families(tmp_path)
    manifest = build_lane_a_campaign(families)
    other = _build_formal_family(
        tmp_path, window=16384, lanes=(LANE_A,), fixture_gds_sha256="e" * 64, label="-other"
    )
    swapped = dict(families)
    swapped[16384] = other
    blockers = verify_lane_a_campaign(manifest, swapped)
    assert any("fixture" in b for b in blockers), blockers


def test_campaign_rejects_mixed_protocol(tmp_path: Path) -> None:
    families = _four_families(tmp_path)
    manifest = build_lane_a_campaign(families)
    other = _build_formal_family(
        tmp_path, window=4096, lanes=(LANE_A,), tile_size=512, label="-other"
    )
    swapped = dict(families)
    swapped[4096] = other
    blockers = verify_lane_a_campaign(manifest, swapped)
    assert any("protocol" in b for b in blockers), blockers


def test_campaign_rejects_non_pass_family(tmp_path: Path) -> None:
    families = _four_families(tmp_path)
    # tamper one family so the formal verifier fails (flip a witness)
    family = families[4096]

    def flip_witness(payload):
        payload["rows"][0]["cuda_execution_witness_pass"] = False

    _tamper_lane_member(family, "industrial-scale-index.json", flip_witness)
    with pytest.raises(ValueError, match="family verifier PASS missing"):
        build_lane_a_campaign(families)


def test_campaign_scripts_roundtrip(tmp_path: Path) -> None:
    """The source-owned scripts build + verify the campaign end to end."""
    families = _four_families(tmp_path)
    out = tmp_path / "campaign.json"
    cmd = [sys.executable, str(CAMPAIGN_BUILDER), "--out", str(out)]
    for window, family in families.items():
        cmd += ["--window", f"{window}={family}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr
    assert "LANE-A CAMPAIGN: BUILT" in proc.stdout
    vcmd = [sys.executable, str(CAMPAIGN_VERIFIER), "--campaign", str(out)]
    for window, family in families.items():
        vcmd += ["--window", f"{window}={family}"]
    vproc = subprocess.run(vcmd, capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert vproc.returncode == 0, vproc.stderr
    assert "LANE-A CAMPAIGN VERIFIER: PASS" in vproc.stdout


# ---- §25: deterministic evidence packaging ----------------------------------------


def _packaging_workspace(tmp_path: Path) -> tuple[Path, Path]:
    family = _build_formal_family(tmp_path, window=4096, lanes=(LANE_A,))
    workspace = tmp_path / "ws-pack"
    (workspace / "family").mkdir(parents=True)
    for path in family.iterdir():
        (workspace / "family" / path.name).write_bytes(path.read_bytes())
    write_strict_json(workspace / "row-A-4096.json", _lane_row(window=4096, lane=LANE_A))
    (workspace / "w4096-r0.npy").write_bytes(b"\x00" * 128)
    return workspace, workspace / "family"


def test_evidence_packaging_deterministic_and_sink_local(tmp_path: Path) -> None:
    packaging = _load_module("pkg_evidence_s7", PACKAGING_SCRIPT)

    workspace, family = _packaging_workspace(tmp_path)
    members = packaging.collect_evidence_members(
        workspace=workspace,
        family_dir=family,
        preflight_log=None,
        verifier_log=None,
        campaign_manifest=None,
    )
    run_id = "0" * 16
    archive_one, sha_one = packaging.package_evidence(dict(members), run_id, tmp_path / "ev1")
    _archive_two, sha_two = packaging.package_evidence(dict(members), run_id, tmp_path / "ev2")
    # deterministic: identical inputs -> identical archive bytes + sha
    assert sha_one == sha_two
    # authority bundle members present
    assert "family/SHA256SUMS.txt" in members
    assert "family/industrial-scale-run-config.json" in members
    assert "workspace/row-A-4096.json" in members
    assert "evidence-manifest.json" in members
    manifest = json.loads(members["evidence-manifest.json"])
    # git authority + environment identity capture recorded
    assert manifest["git_authority"]["git_head"]
    assert manifest["environment_identity"]["driver_identity_source"] == "windows-fallback"
    # raw sink buffers retained LOCALLY, recorded by sha256 + byte count
    assert manifest["sink_buffers"] == [
        {
            "name": "w4096-r0.npy",
            "bytes": 128,
            "sha256": hashlib.sha256(b"\x00" * 128).hexdigest(),
            "retention": "local-only",
        }
    ]
    with tarfile.open(archive_one, "r:gz") as tar:
        names = tar.getnames()
    assert any(name.endswith("family/SHA256SUMS.txt") for name in names)
