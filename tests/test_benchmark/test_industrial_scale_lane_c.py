"""Lane C — single-GPU microbatch saturation hostile tests (S6-migration).

CPU emulation covers the STRUCTURAL Lane C contract (schema, frozen
ladder closure, repeat statistics, output parity across microbatch
rungs, verifier refusal rules).  CPU runs are never GPU validation —
formal Lane C requires the cuda backend on real hardware.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from openlithohub.benchmark.industrial_scale import (
    FROZEN_MICROBATCH_LADDER,
    LANE_C,
    ScaleRunConfig,
    compute_scale_run_identity,
    formal_scale_blockers,
)

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "benchmarks" / "industrial-scale" / "run_scale_benchmark.py"
PREPARE_SCRIPT = REPO / "scripts" / "prepare_industrial_scale_fixture.py"
VERIFIER = REPO / "scripts" / "verify_industrial_scale_artifacts.py"
FROZEN_MICROWATT_GDS_SHA = "b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    source_root = tmp_path / "PDB"
    (source_root / "layout" / "sky130hd" / "ibex").mkdir(parents=True)

    import klayout.db as db

    ly = db.Layout()
    ly.dbu = 0.001
    li = ly.layer(66, 44)
    cell = ly.create_cell("ibex_core")
    cell.shapes(li).insert(db.Box(0, 0, 256, 256))
    ly.write(str(source_root / "layout" / "sky130hd" / "ibex" / "ibex.gds"))

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(source_root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "f@e.com")
    git("config", "user.name", "f")
    git("add", "-A")
    git("commit", "-q", "-m", "pdb")
    commit = git("rev-parse", "HEAD")

    prepare = _load("prepare_lane_c", PREPARE_SCRIPT)
    out = tmp_path / "fixtures" / "ibex"
    prepare.prepare_fixture(
        source_root=source_root,
        design="ibex",
        selected_layer="66:44",
        pixel_nm=1.0,
        output_dir=out,
        expected_commit=commit,
    )
    return out / "fixture-manifest.json", out / "ibex.gds", commit


def _run_harness(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(HARNESS), *args],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO),
    )


# ---- schema + identity ------------------------------------------------------------


def test_lane_c_accepted_by_schema_and_identity_binds_ladder() -> None:
    base = ScaleRunConfig(lanes=(LANE_C,), microbatch_ladder=FROZEN_MICROBATCH_LADDER)
    drifted = base.with_changes(microbatch_ladder=(1, 2, 4, 8, 16))
    kwargs = dict(
        measurement_commit="a" * 40,
        harness_sha256="h",
        core_sha256="c",
        verifier_sha256="v",
        claim_generator_sha256="g",
        fixture_manifest_sha256="f" * 64,
        environment_lock_sha256="e" * 64,
    )
    assert compute_scale_run_identity(base, **kwargs) != compute_scale_run_identity(
        drifted, **kwargs
    ), "changing the microbatch ladder must change the run identity"


def test_lane_c_formal_blockers() -> None:
    env = {"available": True, "count": 1}
    good = {
        "status": "SUCCESS",
        "correctness_witness_pass": True,
        "repeat_count": 5,
    }
    cfg = ScaleRunConfig(
        lanes=(LANE_C,),
        device_backend="cuda",
        gpu_count=1,
        fixture_gds_sha256="9" * 64,
        selected_layer="66:44",
        microbatch_ladder=FROZEN_MICROBATCH_LADDER,
    )
    assert (
        formal_scale_blockers(
            env_lock=env,
            run_config=cfg,
            lane_rows={LANE_C: good},
            git_clean=True,
            provisional=False,
        )
        == []
    )

    # wrong ladder refused
    bad_ladder = cfg.with_changes(microbatch_ladder=(1, 2, 4, 8, 16))
    blockers = formal_scale_blockers(
        env_lock=env,
        run_config=bad_ladder,
        lane_rows={LANE_C: good},
        git_clean=True,
        provisional=False,
    )
    assert any("frozen [1, 2, 4, 8, 16, 32]" in b for b in blockers)

    # multi-GPU lane C refused — single-GPU by definition
    multi = cfg.with_changes(gpu_count=3)
    blockers = formal_scale_blockers(
        env_lock={"available": True, "count": 3},
        run_config=multi,
        lane_rows={LANE_C: good},
        git_clean=True,
        provisional=False,
    )
    assert any("gpu_count == 1" in b for b in blockers)

    # Lane B keeps requiring multiple GPUs when actually requested
    lane_b = ScaleRunConfig(
        lanes=("B_MULTI_GPU_SCALING",),
        device_backend="cuda",
        gpu_count=1,
        fixture_gds_sha256="9" * 64,
        selected_layer="66:44",
    )
    blockers = formal_scale_blockers(
        env_lock={"available": True, "count": 1},
        run_config=lane_b,
        lane_rows={"B_MULTI_GPU_SCALING": good},
        git_clean=True,
        provisional=False,
    )
    assert any("multi-GPU claim" in b for b in blockers)


# ---- harness execution (CPU structural emulation) ----------------------------------


def test_lane_c_full_ladder_execution_and_parity(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale-lane-c"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "c",
        "--windows",
        "128",
        "--tile",
        "32",
        "--repeats",
        "2",
        "--warmup",
        "0",
        "--microbatch-ladder",
        "1,2,4",
        "--device",
        "cpu",
        "--gpu-count",
        "1",
        "--out-root",
        str(out_root),
    )
    assert proc.returncode == 0, proc.stderr
    workspace = Path(json.loads(proc.stdout)["workspace"])
    family = workspace / "family"

    # exact rung closure, canonical order, one row per rung
    saturation = json.loads((family / "industrial-scale-saturation.json").read_text())
    rungs = [r["microbatch"] for r in saturation["rows"]]
    assert rungs == [1, 2, 4], "every frozen rung exactly once, in order"
    for row in saturation["rows"]:
        assert row["status"] == "SUCCESS"
        assert row["correctness_witness_pass"] is True
        assert row["aggregate_n"] == row["repeat_count"] == 2
        assert row["gpu_count"] == 1

    # MB=1 vs MB=4 outputs byte-identical (batching only groups the forward)
    out_mb1 = (workspace / "w128-mb1-r1.npy").read_bytes()
    out_mb4 = (workspace / "w128-mb4-r1.npy").read_bytes()
    assert hashlib.sha256(out_mb1).digest() == hashlib.sha256(out_mb4).digest()

    # no best-of-N: the aggregate is the median over per-repeat observations
    row_mb2 = json.loads((workspace / "row-C_SINGLE_GPU_SATURATION-128-mb2.json").read_text())
    walls = [r["wall_s"] for r in row_mb2["per_repeat"]]
    import statistics

    assert row_mb2["aggregate_median_s"] == statistics.median(walls)
    assert row_mb2["aggregate_n"] == len(walls)


def test_lane_c_family_verifies_structurally_and_refuses_missing_rung(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale-lane-c-v"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "c",
        "--windows",
        "128",
        "--tile",
        "32",
        "--repeats",
        "5",
        "--warmup",
        "0",
        "--microbatch-ladder",
        "1,2",
        "--device",
        "cpu",
        "--gpu-count",
        "1",
        "--out-root",
        str(out_root),
    )
    assert proc.returncode == 0, proc.stderr
    family = Path(json.loads(proc.stdout)["workspace"]) / "family"

    ok = subprocess.run(  # noqa: S603
        [sys.executable, str(VERIFIER), "--root", str(family)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
    )
    assert ok.returncode == 0, ok.stderr

    # drop the mb=2 rung row and refresh closures — the verifier must
    # catch the missing rung via the frozen-ladder closure
    member = family / "industrial-scale-saturation.json"
    payload = json.loads(member.read_text())
    payload["rows"] = [r for r in payload["rows"] if r["microbatch"] != 2]
    member.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((family / "manifest.json").read_text())
    for entry in manifest["members"]:
        entry["bytes"] = (family / entry["name"]).stat().st_size
    (family / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    members = sorted(p.name for p in family.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (family / "SHA256SUMS.txt").write_text(
        "\n".join(f"{hashlib.sha256((family / n).read_bytes()).hexdigest()}  {n}" for n in members)
        + "\n"
    )
    bad = subprocess.run(  # noqa: S603
        [sys.executable, str(VERIFIER), "--root", str(family)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
    )
    assert bad.returncode == 1
    assert "silent missing rung" in bad.stderr


def test_lane_c_formal_refused_on_cpu(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale-lane-c-formal"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "c",
        "--windows",
        "128",
        "--repeats",
        "5",
        "--device",
        "cpu",
        "--formal",
        "--out-root",
        str(out_root),
    )
    assert proc.returncode == 1
    summary = json.loads(proc.stdout)
    assert any("cuda backend" in b for b in summary["family_blockers"])
    assert not (Path(summary["workspace"]) / "family" / "manifest.json").exists()


# ---- §9: CUDA timing contract -------------------------------------------------------


def test_cuda_claim_timing_uses_the_audited_synchronized_helper() -> None:
    """Claim-bearing CUDA wall timing in the scale harness must come from
    the core cuda_synchronized_wall authority (synchronize → timer →
    operation → synchronize → stop) — never an incidental D2H transfer."""

    import openlithohub.benchmark.industrial_v2 as core

    harness_src = (REPO / "benchmarks" / "industrial-scale" / "run_scale_benchmark.py").read_text()
    stream_once_src = harness_src.split("def stream_once(", 1)[1].split("\ndef ", 1)[0]
    assert "cuda_synchronized_wall" in stream_once_src, (
        "stream_once must time CUDA through the audited synchronized helper"
    )
    assert core.cuda_synchronized_wall.__module__ == "openlithohub.benchmark.industrial_v2"


# ---- frozen authority invariance -----------------------------------------------------


def test_microwatt_authority_unchanged_by_migration() -> None:
    manifest = json.loads(
        (
            REPO / "benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json"
        ).read_text()
    )
    assert manifest["gds_sha256"] == FROZEN_MICROWATT_GDS_SHA
    assert manifest["gds_bytes"] == 554770926
    assert manifest["top_cell"] == "microwatt"
    assert manifest["selected_layer"] == "66:44"
    assert manifest["source_commit"] == "9e1e3399b1b707f26fee853bce1ff91ab466ce24"
    assert manifest["die_size_px"] == [3020000, 3610000]
    assert manifest["dense_float32_equivalent_bytes"] == 43608800000000
    assert manifest["equivalent_pixels"] == 10902200000000
    assert len(manifest["split_chunks"]) == 11
