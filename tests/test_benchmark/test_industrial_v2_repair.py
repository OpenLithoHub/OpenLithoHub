"""PR-G Phase 2B.1 hostile protocol tests: measurement-authority repair.

Covers the 2B.1 blockers: fixture/layer/env-lock identity sensitivity
through the ACTUAL harness config builder, per-window ladder rows,
Tier B device/kernel placement contract, fail-closed canonical family
builder over a REAL formal workspace, driver/cuDNN enforcement, and the
claim generator's honest metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from openlithohub.benchmark.industrial_v2 import (
    CANONICAL_FAMILY,
    BatchedFiniteSupportBlur,
    RunConfigV2,
    build_canonical_family_in_workspace,
    compute_run_identity_v2,
    gpu_environment_lock_sha256,
    host_peak_rss_bytes,
)

REPO = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO / "benchmarks" / "industrial-v2" / "run_v2_benchmark.py"
PROMOTER_PATH = REPO / "scripts" / "promote_industrial_v2_artifacts.py"


def _load_module(name: str, path: Path):
    """Load a repo script (dash-named dirs are not importable packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(cfg: RunConfigV2, env_lock: dict | None = None, **overrides: str) -> str:
    kwargs = dict(
        measurement_commit="a" * 40,
        harness_sha256="h",
        core_sha256="c",
        claim_generator_sha256="g",
        verifier_sha256="v",
    )
    if env_lock is not None:
        kwargs["environment_lock_sha256"] = gpu_environment_lock_sha256(env_lock)
    kwargs.update(overrides)
    return compute_run_identity_v2(cfg, **kwargs)


# ---- 2B.1-A/B: fixture + layer enter identity through the harness builder ----


def _harness_run_config(gds_bytes: bytes, layer: str = "66:44") -> tuple[RunConfigV2, str]:
    """The harness's ACTUAL config construction: hash the fixture bytes
    exactly as run_v2_benchmark.main does, then bind into RunConfigV2."""
    fixture_sha256 = hashlib.sha256(gds_bytes).hexdigest()
    run_config = RunConfigV2(
        tiers=("a",),
        layer=layer,
        fixture_sha256=fixture_sha256,
    )
    identity = compute_run_identity_v2(
        run_config,
        measurement_commit="a" * 40,
        harness_sha256="h",
        core_sha256="c",
        claim_generator_sha256="g",
        verifier_sha256="v",
    )
    return run_config, identity


def test_different_fixture_bytes_change_identity_via_harness_builder() -> None:
    _, id_a = _harness_run_config(b"ibex-gds-bytes-v1")
    _, id_b = _harness_run_config(b"ibex-gds-bytes-v2")
    assert id_a != id_b, "different GDS bytes must change the run identity"


def test_layer_change_changes_identity_via_harness_builder() -> None:
    _, id_6644 = _harness_run_config(b"same-bytes", layer="66:44")
    _, id_6720 = _harness_run_config(b"same-bytes", layer="67:20")
    assert id_6644 != id_6720, "layer selection must change the run identity"


def test_run_config_carries_fixture_and_layer() -> None:
    cfg, _ = _harness_run_config(b"x", layer="67:20")
    assert cfg.fixture_sha256 == hashlib.sha256(b"x").hexdigest()
    assert cfg.layer == "67:20"


# ---- 2B.1-D: GPU environment lock hash enters identity ------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        {"available": True},
        {"count": 2},
        {"driver_version": "999.99"},
        {"torch_cuda_version": "12.9"},
        {"torch_version": "9.9.9+cu129"},
        {"tf32_matmul": True},
        {"tf32_cudnn": True},
    ],
)
def test_env_lock_mutations_change_identity(mutation: dict) -> None:
    base_lock = {
        "available": False,
        "count": 0,
        "devices": [],
        "driver_version": "",
        "torch_cuda_version": "",
        "torch_version": "2.14.0",
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }
    mutated = {**base_lock, **mutation}
    assert _identity(RunConfigV2(), base_lock) != _identity(RunConfigV2(), mutated)


def test_gpu_device_topology_change_changes_identity() -> None:
    base_lock = {
        "available": True,
        "count": 1,
        "devices": [
            {
                "index": 0,
                "name": "TestGPU-A",
                "total_memory_bytes": 24_000_000_000,
                "compute_capability": "8.9",
            }
        ],
        "driver_version": "550",
        "torch_cuda_version": "12.4",
        "torch_version": "2.14.0",
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }
    vram = {**base_lock, "devices": [{**base_lock["devices"][0], "total_memory_bytes": 1}]}
    name = {**base_lock, "devices": [{**base_lock["devices"][0], "name": "TestGPU-B"}]}
    cc = {**base_lock, "devices": [{**base_lock["devices"][0], "compute_capability": "9.0"}]}
    base = _identity(RunConfigV2(), base_lock)
    assert base != _identity(RunConfigV2(), vram), "VRAM must alter identity"
    assert base != _identity(RunConfigV2(), name), "GPU model must alter identity"
    assert base != _identity(RunConfigV2(), cc), "compute capability must alter identity"


# ---- 2B.1-H: blur kernels follow input device, shared authority ----------------


def test_blur_kernels_follow_input_device_single_authority() -> None:
    blur = BatchedFiniteSupportBlur()
    tile = torch.rand(16, 16)
    window_out = blur.window_forward(tile)
    batch_out = blur.batch_forward(tile.unsqueeze(0).unsqueeze(0))[0, 0]
    assert torch.allclose(window_out, batch_out, atol=1e-6), (
        "window and batched paths must share one kernel authority"
    )
    assert set(blur._kernel_cache) == {"cpu"}
    kx, ky = blur._kernels_for(torch.device("cpu"))
    assert kx.device.type == "cpu" and ky.device.type == "cpu"
    # to() pre-positions without changing the frozen coefficients
    frozen = kx.clone()
    blur.to("cpu")
    kx2, _ = blur._kernels_for(torch.device("cpu"))
    assert torch.equal(kx2, frozen)


def test_blur_batch_shape_contract() -> None:
    blur = BatchedFiniteSupportBlur()
    with pytest.raises(ValueError, match=r"\(B, 1, H, W\)"):
        blur.batch_forward(torch.rand(4, 16))


# ---- host RSS (§15) -------------------------------------------------------------


def test_host_peak_rss_positive() -> None:
    assert host_peak_rss_bytes() > 0


# ---- 2B.1-J: canonical family builder over a REAL formal workspace ---------------

_FULL_LOCK = {
    "available": True,
    "count": 1,
    "devices": [
        {
            "index": 0,
            "name": "TestGPU",
            "total_memory_bytes": 24_000_000_000,
            "compute_capability": "8.9",
        }
    ],
    "driver_version": "550.54",
    "torch_cuda_version": "12.4",
    "cudnn_version": 90100,
    "torch_version": "2.14.0+cu124",
    "tf32_matmul": False,
    "tf32_cudnn": False,
}


def _window_row(window: int, status: str = "SUCCESS") -> dict:
    return {
        "window": window,
        "status": status,
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
        "flat_scans_avoided_pct": 99.5,
        "aggregate_median_s": 0.4,
    }


def _tier_c_repeat(warm_wall: float, status: str = "SUCCESS") -> dict:
    """One fresh-worker Tier C row EXACTLY as the executed worker emits it
    post-2B.2: one synchronized steady-state observation, cold start as a
    separate diagnostic, no best-of-N list."""
    return {
        "grid": 1024,
        "cpu_reference_wall_s": None,
        "gpu_cold_wall_s": 0.9,
        "gpu_warm_wall_s": warm_wall,
        "warmup_executions": 2,
        "timing_observations": 1,
        "finite_witness": status == "SUCCESS",
        "cpu_deterministic_witness": status == "SUCCESS",
        "correctness_witness_pass": status == "SUCCESS",
        "host_peak_rss_bytes": 3000,
        "max_memory_allocated": 500,
        "max_memory_reserved": 800,
        "timing_method": "cuda_synchronized",
        "dtype": "fp32",
        "device": "cuda:0",
        "device_requires_cuda": True,
        "status": status,
    }


def _tier_c_aggregate_row(status: str = "SUCCESS") -> dict:
    """The Tier C row EXACTLY as the driver's _aggregate_window emits it:
    the claim-bearing median/p10/p90/n aggregate over fresh-worker
    observations, with the GPU facts the canonical family locks."""
    measured = [_tier_c_repeat(0.020 + 0.001 * i, status=status) for i in range(5)]
    row: dict = {
        "tier": "c",
        "window": 1024,
        "repeat_count": len(measured),
        "warmup_discarded": 2,
        "correctness_witness_pass": status == "SUCCESS",
        "claim_level": "REPRODUCED_INTERNAL",
        "status": status,
    }
    for i, repeat in enumerate(measured):
        row[f"repeat_{i}"] = repeat
    if status == "SUCCESS":
        walls = [r["gpu_warm_wall_s"] for r in measured]
        row.update(
            {
                "aggregate_n": len(walls),
                "aggregate_median_s": statistics.median(walls),
                "aggregate_p10_s": 0.020,
                "aggregate_p90_s": 0.024,
                "grid": 1024,
                "timing_method": "cuda_synchronized",
                "device": "cuda:0",
                "dtype": "fp32",
                "device_requires_cuda": True,
                "max_memory_allocated": 500,
                "max_memory_reserved": 800,
                "host_peak_rss_bytes": 3000,
                "warmup_executions": 2,
                "timing_observations": 1,
                "gpu_cold_wall_s": 0.9,
            }
        )
    else:
        row.update(
            {
                "aggregate_n": 0,
                "aggregate_median_s": 0.0,
                "aggregate_p10_s": 0.0,
                "aggregate_p90_s": 0.0,
            }
        )
    return row


def _formal_workspace(
    tmp_path: Path,
    *,
    windows: tuple[int, ...] = (4096, 8192),
    tier_b_status: str = "SUCCESS",
    tier_c_status: str = "SUCCESS",
    fixture_sha256: str = "d" * 64,
    layer: str = "66:44",
    clean: bool = True,
    provisional: bool = False,
    driver: str = "550.54",
) -> tuple[Path, RunConfigV2, str, dict[str, str], dict]:
    from openlithohub.benchmark.industrial_v2 import write_strict_json

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    run_config = RunConfigV2(
        tiers=("a", "b", "c"),
        layer=layer,
        fixture_sha256=fixture_sha256,
        window_sizes=windows,
        repeat_count=5,
    )
    source = {
        "harness": "h",
        "core": "c",
        "claim_generator": "g",
        "verifier": "v",
    }
    env_lock = {**_FULL_LOCK, "driver_version": driver}
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
    write_strict_json(
        workspace / "environment-lock.json",
        env_lock,
    )
    write_strict_json(
        workspace / "run-config.json",
        {
            "schema": "OpenLithoHub.industrial-run-config.v2",
            "measurement_commit": "a" * 40,
            "tracked_tree_clean": clean,
            "provisional": provisional,
            "source_hashes": source,
            "environment_lock_sha256": lock_sha,
            "run_identity": identity,
            "run_config": run_config.to_payload(),
        },
    )
    write_strict_json(
        workspace / "tier-a.json",
        {
            "tier": "a",
            "status": "SUCCESS",
            "correctness_witness_pass": True,
            "repeats_recorded": 5 * len(windows),
            "window_rows": [_window_row(w) for w in windows],
        },
    )
    write_strict_json(
        workspace / "tier-b.json",
        {
            "tier": "b",
            "status": tier_b_status,
            "correctness_witness_pass": tier_b_status == "SUCCESS",
            "repeats_recorded": 5 * len(windows),
            "window_rows": [_window_row(w, status=tier_b_status) for w in windows],
        },
    )
    write_strict_json(
        workspace / "tier-c.json",
        {**_tier_c_aggregate_row(status=tier_c_status), "repeats_recorded": 5},
    )
    return workspace, run_config, identity, source, env_lock


def test_builder_builds_exact_seven_member_family(tmp_path: Path) -> None:
    workspace, run_config, identity, source, env_lock = _formal_workspace(tmp_path)
    rows = {
        "a": json.loads((workspace / "tier-a.json").read_text()),
        "b": json.loads((workspace / "tier-b.json").read_text()),
        "c": json.loads((workspace / "tier-c.json").read_text()),
    }
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
    )
    assert blockers == [], blockers
    members = {p.name for p in workspace.iterdir()}
    assert members >= CANONICAL_FAMILY, "canonical family must be complete"
    # promotion now succeeds with zero blockers
    canonical = tmp_path / "canonical"
    from openlithohub.benchmark.industrial_v2 import promote_canonical_family

    left = promote_canonical_family(
        workspace_dir=workspace,
        canonical_root=canonical,
        env_lock=env_lock,
        tier_rows={
            "a": {"status": "SUCCESS", "correctness_witness_pass": True},
            "b": {"status": "SUCCESS", "correctness_witness_pass": True},
            "c": {"status": "SUCCESS", "correctness_witness_pass": True},
        },
        git_clean=True,
        provisional=False,
    )
    assert left == []
    assert {p.name for p in canonical.iterdir()} == CANONICAL_FAMILY


def test_builder_fails_closed_on_dirty_or_provisional(tmp_path: Path) -> None:
    workspace, run_config, identity, source, env_lock = _formal_workspace(tmp_path)
    rows = {
        "a": json.loads((workspace / "tier-a.json").read_text()),
        "b": json.loads((workspace / "tier-b.json").read_text()),
        "c": json.loads((workspace / "tier-c.json").read_text()),
    }
    for kwargs in (
        {"tracked_tree_clean": False, "provisional": False},
        {"tracked_tree_clean": True, "provisional": True},
    ):
        blockers = build_canonical_family_in_workspace(
            workspace_dir=workspace,
            run_config=run_config,
            run_identity=identity,
            measurement_commit="a" * 40,
            source_hashes=source,
            environment_lock=env_lock,
            tier_rows=rows,
            **kwargs,
        )
        assert blockers, "must refuse dirty/provisional runs"


def test_builder_fails_closed_on_incomplete_window_ladder(tmp_path: Path) -> None:
    workspace, run_config, identity, source, env_lock = _formal_workspace(
        tmp_path, windows=(4096, 8192)
    )
    rows = {
        "a": json.loads((workspace / "tier-a.json").read_text()),
        "b": json.loads((workspace / "tier-b.json").read_text()),
        "c": json.loads((workspace / "tier-c.json").read_text()),
    }
    # declared ladder expects 4096+8192 but rows only carry 4096
    rows["a"]["window_rows"] = rows["a"]["window_rows"][:1]
    rows["b"]["window_rows"] = rows["b"]["window_rows"][:1]
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
    )
    assert any("window ladder incomplete" in b for b in blockers)


def test_builder_fails_closed_on_identity_mismatch(tmp_path: Path) -> None:
    workspace, run_config, identity, source, env_lock = _formal_workspace(tmp_path)
    rows = {
        "a": json.loads((workspace / "tier-a.json").read_text()),
        "b": json.loads((workspace / "tier-b.json").read_text()),
        "c": json.loads((workspace / "tier-c.json").read_text()),
    }
    wrong_identity = "f" * 64
    blockers = build_canonical_family_in_workspace(
        workspace_dir=workspace,
        run_config=run_config,
        run_identity=wrong_identity,
        measurement_commit="a" * 40,
        source_hashes=source,
        environment_lock=env_lock,
        tier_rows=rows,
        tracked_tree_clean=True,
        provisional=False,
    )
    assert any("does not match recomputed" in b for b in blockers)
    assert not (workspace / "manifest.json").exists(), "nothing written on refusal"


def test_builder_fails_closed_on_empty_fixture_or_driver(tmp_path: Path) -> None:
    workspace, run_config, identity, source, env_lock = _formal_workspace(tmp_path)
    rows = {
        "a": json.loads((workspace / "tier-a.json").read_text()),
        "b": json.loads((workspace / "tier-b.json").read_text()),
        "c": json.loads((workspace / "tier-c.json").read_text()),
    }
    cfg_no_fixture = run_config.with_changes(fixture_sha256="")
    blockers = build_canonical_family_in_workspace(
        workspace_dir=workspace,
        run_config=cfg_no_fixture,
        run_identity=identity,
        measurement_commit="a" * 40,
        source_hashes=source,
        environment_lock=env_lock,
        tier_rows=rows,
        tracked_tree_clean=True,
        provisional=False,
    )
    assert any("fixture sha256" in b for b in blockers)

    lock_no_driver = {**env_lock, "driver_version": ""}
    blockers2 = build_canonical_family_in_workspace(
        workspace_dir=workspace,
        run_config=run_config,
        run_identity=identity,
        measurement_commit="a" * 40,
        source_hashes=source,
        environment_lock=lock_no_driver,
        tier_rows=rows,
        tracked_tree_clean=True,
        provisional=False,
    )
    assert any("driver_version" in b for b in blockers2)


def test_builder_fails_closed_on_unsynced_timing_or_missing_rss(
    tmp_path: Path,
) -> None:
    workspace, run_config, identity, source, env_lock = _formal_workspace(tmp_path)
    rows = {
        "a": json.loads((workspace / "tier-a.json").read_text()),
        "b": json.loads((workspace / "tier-b.json").read_text()),
        "c": json.loads((workspace / "tier-c.json").read_text()),
    }
    for wrow in rows["b"]["window_rows"]:
        wrow["timing_method"] = "host_perf_counter"
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
    )
    assert any("unsynchronized" in b for b in blockers)

    for wrow in rows["b"]["window_rows"]:
        wrow["timing_method"] = "cuda_synchronized"
        del wrow["host_peak_rss_bytes"]
    blockers2 = build_canonical_family_in_workspace(
        workspace_dir=workspace,
        run_config=run_config,
        run_identity=identity,
        measurement_commit="a" * 40,
        source_hashes=source,
        environment_lock=env_lock,
        tier_rows=rows,
        tracked_tree_clean=True,
        provisional=False,
    )
    assert any("host_peak_rss_bytes" in b for b in blockers2)


# ---- §13/§14: claim generator honest metrics + medians ---------------------------


def test_claim_generator_uses_flat_scans_and_medians(tmp_path: Path, monkeypatch) -> None:
    import scripts.generate_industrial_v2_claims as generator

    canonical = tmp_path / "v2"
    canonical.mkdir()
    family_written = {
        "run_config": {
            "run_identity": "e" * 64,
            "measurement_commit": "a" * 40,
            "tracked_tree_clean": True,
            "provisional": False,
            "source_hashes": {"harness": "h", "core": "c", "claim_generator": "g", "verifier": "v"},
            "environment_lock_sha256": "e" * 64,
            "run_config": {
                "tiers": ["a", "b", "c"],
                "device": "cuda:0",
                "dtype": "fp32",
                "tf32_matmul": False,
                "tf32_cudnn": False,
                "deterministic_algorithms": True,
                "compile_mode": "off",
                "batch_size": 8,
                "tile_size": 1024,
                "halo_px": 64,
                "pixel_nm": 1.0,
                "forward_radius": 4,
                "forward_sigma_nm": 1.6,
                "window_sizes": [4096],
                "hopkins_wavelength_nm": 13.5,
                "hopkins_na": 0.33,
                "hopkins_sigma_outer": 0.9,
                "hopkins_sigma_inner": 0.6,
                "hopkins_defocus_nm": 0.0,
                "hopkins_grid": 1024,
                "warmup_count": 2,
                "repeat_count": 5,
                "layer": "66:44",
                "fixture_sha256": "d" * 64,
            },
        },
        "tiers": {
            "a": {
                "schema": "OpenLithoHub.industrial-benchmark.v2",
                "run_identity": "e" * 64,
                "correctness_witness_pass": True,
                "claim_level": "REPRODUCED_INTERNAL",
                "window_rows": [
                    {
                        "window": 4096,
                        "status": "SUCCESS",
                        "repeat_count": 5,
                        "correctness_witness_pass": True,
                        "flat_scans_avoided_pct": 99.6,
                    }
                ],
            },
            "b": {
                "schema": "OpenLithoHub.industrial-benchmark.v2",
                "run_identity": "e" * 64,
                "correctness_witness_pass": True,
                "claim_level": "REPRODUCED_INTERNAL",
                "window_rows": [
                    {
                        "window": 4096,
                        "status": "SUCCESS",
                        "repeat_count": 5,
                        "correctness_witness_pass": True,
                        "timing_method": "cuda_synchronized",
                        "device": "cuda:0",
                        "dtype": "fp32",
                        "max_memory_allocated": 1,
                        "max_memory_reserved": 2,
                        "host_peak_rss_bytes": 3,
                        "aggregate_batch1_median_s": 2.0,
                        "aggregate_batch_n_median_s": 1.0,
                    }
                ],
            },
            "c": {
                "schema": "OpenLithoHub.industrial-benchmark.v2",
                "run_identity": "e" * 64,
                "correctness_witness_pass": True,
                "claim_level": "REPRODUCED_INTERNAL",
                "rows": [
                    {
                        "grid": 1024,
                        "status": "SUCCESS",
                        "repeat_count": 5,
                        "correctness_witness_pass": True,
                        "timing_method": "cuda_synchronized",
                        "device": "cuda:0",
                        "dtype": "fp32",
                        "max_memory_allocated": 1,
                        "max_memory_reserved": 2,
                    }
                ],
            },
        },
    }
    monkeypatch.setattr(generator, "load_verified_family", lambda root: family_written)
    claims = generator.build_claims(family_written)

    by_id = {c["claim_id"]: c for c in claims["claims"]}
    scans = by_id["IB2-INDEX-SCANS-4096"]
    assert scans["unit"] == "flat_scans_avoided_pct"
    assert scans["value"] == 99.6
    assert "candidate_reduction_pct" not in json.dumps(claims)
    batch = by_id["IB2-GPU-BATCH-4096"]
    assert batch["value"] == 2.0, "batching speedup must come from medians 2.0/1.0"


# ---- 2B.2-A: Tier B EXECUTES the declared layer (not just declares it) ----


def test_tier_b_execution_source_receives_declared_layer(monkeypatch) -> None:
    """Execution-level: changing 66:44 → 67:20 changes the layer actually
    passed into KLayoutAlignedRunSource.from_file() by the harness's real
    source-construction step (the constructor is spied, everything else
    executes)."""
    harness = _load_module("run_v2_benchmark", HARNESS_PATH)
    import openlithohub.streaming.vector_runs as vector_runs

    calls: list[dict] = []

    class SpySource:
        def __init__(self, shape: tuple[int, int] = (8, 8)) -> None:
            self.shape = shape

        @classmethod
        def from_file(
            cls,
            path: str,
            *,
            pixel_size_nm: float,
            layer: str | None = None,
            top_cell: str | None = None,
        ) -> SpySource:
            calls.append({"path": str(path), "pixel_size_nm": pixel_size_nm, "layer": layer})
            return cls()

    monkeypatch.setattr(vector_runs, "KLayoutAlignedRunSource", SpySource)

    base = {"gds": "ibex.gds", "pixel_nm": 1.0}
    harness._tier_b_execution_source({**base, "layer": "66:44"})
    harness._tier_b_execution_source({**base, "layer": "67:20"})
    assert [call["layer"] for call in calls] == ["66:44", "67:20"], (
        "the executed source constructor must receive the declared layer"
    )
    assert all(call["pixel_size_nm"] == 1.0 for call in calls)
    # the worker body must go through this constructor (never bypass it)
    assert "_tier_b_execution_source(cfg)" in inspect.getsource(harness.tier_b_worker_once)
    assert 'from_file(cfg["gds"], pixel_size_nm=cfg["pixel_nm"])' not in inspect.getsource(
        harness.tier_b_worker_once
    ), "the layer-less constructor call must stay dead"


def test_worker_entry_carries_declared_layer_into_tier_b_cfg(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The declared --layer CLI arg reaches the executed Tier B cfg."""
    harness = _load_module("run_v2_benchmark", HARNESS_PATH)
    captured: dict = {}

    def fake_tier_b(cfg: dict) -> dict:
        captured.update(cfg)
        return {"status": "NOT_RUN_ENVIRONMENT", "correctness_witness_pass": False}

    monkeypatch.setattr(harness, "tier_b_worker_once", fake_tier_b)
    args = argparse.Namespace(
        worker_tier="b",
        gds="ibex.gds",
        device="cuda:0",
        dtype="fp32",
        layer="67:20",
        window=64,
        tile=256,
        batch=2,
        pixel_nm=1.0,
        forward_radius=4,
        forward_sigma=1.6,
        hopkins_wavelength_nm=13.5,
        hopkins_na=0.33,
        hopkins_sigma_outer=0.9,
        hopkins_sigma_inner=0.6,
        hopkins_grid=256,
    )
    assert harness.worker_entry(args) == 0
    assert captured["layer"] == "67:20"
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "NOT_RUN_ENVIRONMENT"


# ---- 2B.2-B: Tier C aggregates the ACTUAL GPU warm timing field -----------


def test_tier_c_aggregate_uses_gpu_warm_timing_field() -> None:
    """Blocker B: the claim-bearing Tier C statistic is derived from the
    executed synchronized GPU steady-state field (gpu_warm_wall_s), not a
    stale key."""
    harness = _load_module("run_v2_benchmark", HARNESS_PATH)
    walls = [0.020, 0.021, 0.022, 0.023, 0.025]
    measured = [_tier_c_repeat(w) for w in walls]
    row = harness._aggregate_window("c", 1024, warm=[], measured=measured)
    assert row["status"] == "SUCCESS"
    assert row["aggregate_n"] == len(walls)
    assert row["aggregate_median_s"] == statistics.median(walls)
    assert row["aggregate_p10_s"] == pytest.approx(harness.percentile(walls, 0.10))
    assert row["aggregate_p90_s"] == pytest.approx(harness.percentile(walls, 0.90))
    # GPU row facts the canonical family locks propagate onto the row
    assert row["timing_method"] == "cuda_synchronized"
    assert row["device"] == "cuda:0"
    assert row["timing_observations"] == 1
    # raw per-repeat observations stay recorded
    assert [row[f"repeat_{i}"]["gpu_warm_wall_s"] for i in range(len(walls))] == walls


def test_tier_c_zero_length_aggregate_cannot_be_success() -> None:
    """A stale/missing timing key must FAIL the row — a successful formal
    Tier C artifact can never contain a zero-length timing aggregate."""
    harness = _load_module("run_v2_benchmark", HARNESS_PATH)
    measured = [_tier_c_repeat(0.02) for _ in range(5)]
    for repeat in measured:
        del repeat["gpu_warm_wall_s"]  # simulate the stale-key worker
    row = harness._aggregate_window("c", 1024, warm=[], measured=measured)
    assert row["status"] == "FAILED"
    assert row["aggregate_n"] == 0
    assert "stale" in row["reason"]
    assert "gpu_warm_wall_s" in row["reason"]


def test_tier_c_aggregate_n_equals_repeat_count() -> None:
    """Hostile #3: formal Tier C aggregate has n == repeat_count; a
    partial aggregate fails the row instead of publishing silently."""
    harness = _load_module("run_v2_benchmark", HARNESS_PATH)
    walls = [0.020, 0.021, 0.022, 0.023, 0.024]
    measured = [_tier_c_repeat(w) for w in walls]
    row = harness._aggregate_window("c", 1024, warm=[], measured=measured)
    assert row["repeat_count"] == 5
    assert row["aggregate_n"] == row["repeat_count"]

    partial = [_tier_c_repeat(w) for w in walls]
    del partial[4]["gpu_warm_wall_s"]
    row_partial = harness._aggregate_window("c", 1024, warm=[], measured=partial)
    assert row_partial["status"] == "FAILED"
    assert row_partial["aggregate_n"] == 4 != row_partial["repeat_count"]


def test_no_stale_or_best_of_n_keys_anywhere_in_harness() -> None:
    """The stale Tier C key and the best-of-N emission are gone from the
    executed harness source."""
    harness = _load_module("run_v2_benchmark", HARNESS_PATH)
    module_src = inspect.getsource(harness)
    assert "cpu_wall_s" not in module_src, "stale Tier C timing key must not remain"
    assert '"warm_walls_s"' not in module_src, "best-of-N wall list must not be emitted"
    worker_src = inspect.getsource(harness.tier_c_worker_once)
    assert "min(" not in worker_src, "no best-of-N over in-worker repetitions"
    assert "cuda_synchronized_wall" in worker_src, (
        "claim-bearing GPU timing must come through the synchronized helper"
    )


def test_no_best_of_n_enters_tier_c_claim_statistic() -> None:
    """Hostile #4: the claim-bearing statistic is the median over one
    observation per fresh worker — a fast outlier minimum can never
    become the headline number."""
    harness = _load_module("run_v2_benchmark", HARNESS_PATH)
    walls = [0.050, 0.010, 0.090, 0.020, 0.080]  # min = 0.010 traps the fast outlier
    measured = [_tier_c_repeat(w) for w in walls]
    row = harness._aggregate_window("c", 1024, warm=[], measured=measured)
    assert row["status"] == "SUCCESS"
    assert row["aggregate_median_s"] == statistics.median(walls)
    assert row["aggregate_median_s"] != min(walls)
    assert row["timing_observations"] == 1, "exactly one observation per fresh worker"


def test_builder_refuses_tier_c_aggregate_n_mismatch(tmp_path: Path) -> None:
    """Authority-level: the canonical family builder refuses a Tier C row
    whose claim-bearing aggregate is incomplete or insufficient."""
    workspace, run_config, identity, source, env_lock = _formal_workspace(tmp_path)
    rows = {
        "a": json.loads((workspace / "tier-a.json").read_text()),
        "b": json.loads((workspace / "tier-b.json").read_text()),
        "c": json.loads((workspace / "tier-c.json").read_text()),
    }
    for tampered, why in ((3, "partial aggregate"), (0, "zero-length aggregate")):
        rows["c"]["aggregate_n"] = tampered
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
        )
        assert any("timing aggregate n=" in b for b in blockers), why
        assert any("insufficient claim-bearing repeats" in b for b in blockers), why


# ---- 2B.2-E: operator promotion CLI ----------------------------------------


def _workspace_rows(workspace: Path) -> dict[str, dict]:
    return {
        tier: json.loads((workspace / f"tier-{tier}.json").read_text()) for tier in ("a", "b", "c")
    }


def _built_formal_workspace(tmp_path: Path) -> tuple[Path, str, dict, dict]:
    """A formal workspace whose seven-member canonical family was built
    through the fail-closed builder (the only sanctioned source)."""
    workspace, run_config, identity, source, env_lock = _formal_workspace(tmp_path)
    blockers = build_canonical_family_in_workspace(
        workspace_dir=workspace,
        run_config=run_config,
        run_identity=identity,
        measurement_commit="a" * 40,
        source_hashes=source,
        environment_lock=env_lock,
        tier_rows=_workspace_rows(workspace),
        tracked_tree_clean=True,
        provisional=False,
    )
    assert blockers == [], blockers
    return workspace, identity, env_lock, run_config.to_payload()


def test_promotion_cli_produces_exactly_seven_members(tmp_path: Path) -> None:
    """Hostile #7: the operator path promotes EXACTLY the seven canonical
    members and the promoted root closes under the real verifier."""
    workspace, identity, env_lock, _ = _built_formal_workspace(tmp_path)
    promoter = _load_module("promote_industrial_v2_artifacts", PROMOTER_PATH)
    canonical = tmp_path / "canonical-staging"
    promoted, blockers, info = promoter.promote(
        workspace, canonical, promoter.REPO, tree_clean=True
    )
    assert promoted, blockers
    assert {p.name for p in canonical.iterdir()} == set(CANONICAL_FAMILY)
    assert info["run_identity"] == identity
    # idempotent: re-promoting the byte-identical family is allowed
    promoted2, blockers2, _ = promoter.promote(workspace, canonical, promoter.REPO, tree_clean=True)
    assert promoted2, blockers2
    assert {p.name for p in canonical.iterdir()} == set(CANONICAL_FAMILY)


def test_promotion_cli_refuses_incomplete_family(tmp_path: Path) -> None:
    """Hostile #5: a workspace whose canonical family is not complete is
    refused — nothing is written to the canonical root."""
    workspace, _, _, _ = _built_formal_workspace(tmp_path)
    (workspace / "SHA256SUMS.txt").unlink()
    promoter = _load_module("promote_industrial_v2_artifacts", PROMOTER_PATH)
    canonical = tmp_path / "canonical"
    promoted, blockers, _ = promoter.promote(workspace, canonical, promoter.REPO, tree_clean=True)
    assert not promoted
    assert any("canonical family" in blocker for blocker in blockers)
    assert any("SHA256SUMS.txt" in blocker for blocker in blockers)
    assert not canonical.exists(), "refusal must leave the canonical root untouched"


def test_promotion_cli_refuses_provisional_run(tmp_path: Path) -> None:
    """Hostile #6: a provisional run is refused even with a built family."""
    workspace, _, _, _ = _built_formal_workspace(tmp_path)
    run_config_path = workspace / "run-config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["provisional"] = True
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True))
    promoter = _load_module("promote_industrial_v2_artifacts", PROMOTER_PATH)
    canonical = tmp_path / "canonical"
    promoted, blockers, _ = promoter.promote(workspace, canonical, promoter.REPO, tree_clean=True)
    assert not promoted
    assert any("provisional" in blocker for blocker in blockers)
    assert not canonical.exists()


def test_promotion_cli_refuses_dirty_tree(tmp_path: Path) -> None:
    workspace, _, _, _ = _built_formal_workspace(tmp_path)
    promoter = _load_module("promote_industrial_v2_artifacts", PROMOTER_PATH)
    promoted, blockers, _ = promoter.promote(
        workspace, tmp_path / "canonical", promoter.REPO, tree_clean=False
    )
    assert not promoted
    assert any("dirty" in blocker for blocker in blockers)


def test_promotion_cli_refuses_overwriting_different_authority(tmp_path: Path) -> None:
    """An existing unrelated canonical authority is never silently
    overwritten; the byte-identical family re-promotes idempotently."""
    promoter = _load_module("promote_industrial_v2_artifacts", PROMOTER_PATH)
    first, _, _, _ = _built_formal_workspace(tmp_path / "run-a")
    canonical = tmp_path / "canonical"
    promoted, blockers, _ = promoter.promote(first, canonical, promoter.REPO, tree_clean=True)
    assert promoted, blockers

    # a DIFFERENT run (different fixture bytes → different run identity)
    workspace_b, run_config_b, identity_b, source_b, env_lock_b = _formal_workspace(
        tmp_path / "run-b", fixture_sha256="e" * 64
    )
    assert identity_b != json.loads((first / "run-config.json").read_text())["run_identity"]
    blockers_b = build_canonical_family_in_workspace(
        workspace_dir=workspace_b,
        run_config=run_config_b,
        run_identity=identity_b,
        measurement_commit="a" * 40,
        source_hashes=source_b,
        environment_lock=env_lock_b,
        tier_rows=_workspace_rows(workspace_b),
        tracked_tree_clean=True,
        provisional=False,
    )
    assert blockers_b == [], blockers_b
    promoted_b, blockers_b2, _ = promoter.promote(
        workspace_b, canonical, promoter.REPO, tree_clean=True
    )
    assert not promoted_b
    assert any("DIFFERENT" in blocker for blocker in blockers_b2)
    # the FIRST authority is still intact, byte for byte
    assert (
        first.joinpath("SHA256SUMS.txt").read_bytes() == (canonical / "SHA256SUMS.txt").read_bytes()
    )


def test_promotion_cli_refuses_frozen_v11_root(tmp_path: Path) -> None:
    """v1.1 is never a promotion target — neither a synthetic copy nor the
    real frozen root (read-only guard check)."""
    promoter = _load_module("promote_industrial_v2_artifacts", PROMOTER_PATH)
    fake_repo = tmp_path / "repo"
    blockers = promoter._frozen_v11_blockers(
        fake_repo / "benchmarks" / "results" / "industrial",
        tmp_path / "workspace",
        fake_repo,
    )
    assert blockers and "v1.1" in blockers[0]
    real = promoter._frozen_v11_blockers(
        REPO / "benchmarks" / "results" / "industrial", tmp_path / "workspace", REPO
    )
    assert real, "the real frozen v1.1 root must be refused as a target"
    assert not promoter._frozen_v11_blockers(tmp_path / "staging", tmp_path / "workspace", REPO), (
        "an unrelated staging root stays promotable"
    )


def test_promotion_cli_subprocess_refuses_incomplete_family(tmp_path: Path) -> None:
    """Operator executability: the documented CLI invocation refuses
    fail-closed with a nonzero exit code."""
    workspace, _, _, _ = _built_formal_workspace(tmp_path)
    (workspace / "manifest.json").unlink()
    canonical = tmp_path / "canonical"
    proc = subprocess.run(  # noqa: S603 — fixed-argv repo script
        [
            sys.executable,
            str(PROMOTER_PATH),
            "--workspace",
            str(workspace),
            "--canonical-root",
            str(canonical),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
    )
    assert proc.returncode == 1
    assert "V2 PROMOTION: REFUSED" in proc.stderr
    assert not canonical.exists()


def test_promotion_cli_subprocess_happy_path(tmp_path: Path) -> None:
    """Operator executability: the documented CLI invocation promotes end
    to end in a real subprocess, with the live git-clean gate actually
    evaluated.  Runs against a THROWAWAY --shared clone pinned to the
    checkout HEAD so the gate is deterministic and can never race with
    parallel xdist workers touching the outer checkout — the CLI's
    dirty-tree refusal is the feature under test, never weakened."""
    head = subprocess.run(  # noqa: S603 — fixed-argv git query
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    clone = tmp_path / "repo-clone"
    subprocess.run(  # noqa: S603 — local --shared clone, no network
        ["git", "clone", "--shared", "--quiet", "--no-checkout", str(REPO), str(clone)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    subprocess.run(  # noqa: S603 — fixed-argv git checkout
        ["git", "-C", str(clone), "checkout", "--quiet", "--force", head],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    workspace, _, _, _ = _built_formal_workspace(tmp_path)
    canonical = tmp_path / "canonical"
    proc = subprocess.run(  # noqa: S603 — fixed-argv repo script
        [
            sys.executable,
            str(clone / "scripts" / "promote_industrial_v2_artifacts.py"),
            "--workspace",
            str(workspace),
            "--canonical-root",
            str(canonical),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(clone),
    )
    assert proc.returncode == 0, proc.stderr
    assert "V2 PROMOTION: PASS" in proc.stdout
    assert {p.name for p in canonical.iterdir()} == set(CANONICAL_FAMILY)


# ---- frozen-artifact safety ------------------------------------------------


def _frozen_artifact_snapshot() -> dict[str, str]:
    tracked = subprocess.run(  # noqa: S603 — fixed-argv git query
        ["git", "-C", str(REPO), "ls-files", "benchmarks/results/industrial", "proof_artifacts"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.splitlines()
    assert tracked, "frozen v1.1 + P-054/B04 artifacts must be git-tracked"
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in tracked}


def test_promotion_never_touches_v11_or_p054_bytes(tmp_path: Path) -> None:
    """Hostile #8: a full operator promotion leaves every tracked v1.1 and
    P-054/B04 artifact byte-identical."""
    before = _frozen_artifact_snapshot()
    workspace, _, _, _ = _built_formal_workspace(tmp_path)
    promoter = _load_module("promote_industrial_v2_artifacts", PROMOTER_PATH)
    promoted, blockers, _ = promoter.promote(
        workspace, tmp_path / "canonical", promoter.REPO, tree_clean=True
    )
    assert promoted, blockers
    assert _frozen_artifact_snapshot() == before


def test_harness_refuses_v11_out_root_before_writing(tmp_path: Path) -> None:
    """The harness driver refuses a v1.1 output root (and any subpath of
    it) BEFORE creating or writing anything."""
    v11_like = tmp_path / "benchmarks" / "results" / "industrial"
    proc = subprocess.run(  # noqa: S603 — fixed-argv repo script
        [
            sys.executable,
            str(HARNESS_PATH),
            "--tiers",
            "a",
            "--out-root",
            str(v11_like),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
    )
    assert proc.returncode == 1
    assert "refusing to write v2 results into the frozen v1.1 root" in proc.stdout + proc.stderr
    assert not v11_like.exists(), "nothing may be created inside the frozen root"
