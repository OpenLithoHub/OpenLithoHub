"""PR-G Phase 2B.1 hostile protocol tests: measurement-authority repair.

Covers the 2B.1 blockers: fixture/layer/env-lock identity sensitivity
through the ACTUAL harness config builder, per-window ladder rows,
Tier B device/kernel placement contract, fail-closed canonical family
builder over a REAL formal workspace, driver/cuDNN enforcement, and the
claim generator's honest metrics.
"""

from __future__ import annotations

import hashlib
import json
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
        {
            "tier": "c",
            "status": tier_c_status,
            "correctness_witness_pass": tier_c_status == "SUCCESS",
            "repeats_recorded": 5,
            "window_rows": [
                {
                    "window": 1024,
                    "status": tier_c_status,
                    "repeat_count": 5,
                    "correctness_witness_pass": tier_c_status == "SUCCESS",
                    "timing_method": "cuda_synchronized",
                    "device": "cuda:0",
                    "dtype": "fp32",
                    "max_memory_allocated": 500,
                    "max_memory_reserved": 800,
                    "host_peak_rss_bytes": 3000,
                }
            ],
        },
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
