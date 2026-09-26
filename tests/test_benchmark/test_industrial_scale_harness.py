"""Industrial Scale streaming harness tests (S2).

Exercises the real harness end to end on tiny synthetic routed-GDS
fixtures: honest statuses, the full-layout firewall, claim-bearing
repeat statistics, headline eligibility, fixture-bytes binding, the
8-member workspace family, and formal refusal on a CPU host.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openlithohub.benchmark.industrial_scale import (
    SCALE_CANONICAL_FAMILY,
    compute_scale_run_identity,
    load_scale_fixture_manifest,
    scale_environment_lock,
    scale_environment_lock_sha256,
)

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "benchmarks" / "industrial-scale" / "run_scale_benchmark.py"
PREPARE_SCRIPT = REPO / "scripts" / "prepare_industrial_scale_fixture.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gds(path: Path, width: int = 256, height: int = 256, top: str = "ibex_core") -> None:
    import klayout.db as db

    ly = db.Layout()
    ly.dbu = 0.001  # 1 nm per DBU
    li = ly.layer(66, 44)
    cell = ly.create_cell(top)
    cell.shapes(li).insert(db.Box(0, 0, width, height))
    ly.write(str(path))


@pytest.fixture
def prepared_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    """A synthetic PDB checkout, git-committed, with a prepared Ibex
    fixture manifest — the harness's only sanctioned input shape."""
    source_root = tmp_path / "PDB"
    (source_root / "layout" / "sky130hd" / "ibex").mkdir(parents=True)
    _write_gds(source_root / "layout" / "sky130hd" / "ibex" / "ibex.gds")

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

    prepare = _load("prepare_industrial_scale_fixture", PREPARE_SCRIPT)
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


def test_harness_end_to_end_lane_a(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "a",
        "--windows",
        "128",
        "--tile",
        "64",
        "--microbatch",
        "4",
        "--repeats",
        "2",
        "--warmup",
        "1",
        "--device",
        "cpu",
        "--out-root",
        str(out_root),
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["provisional"] is True
    assert "NOT PERFORMANCE AUTHORITY" in summary["status"]

    workspace = Path(summary["workspace"])
    row = json.loads((workspace / "row-A_LARGE_LAYOUT_STREAMING-128.json").read_text())
    assert row["status"] == "SUCCESS"
    assert row["correctness_witness_pass"] is True
    assert row["finite_witness"] is True
    assert row["headline_eligible"] is True, "P1 rows are headline-eligible"
    assert row["aggregate_n"] == row["repeat_count"] == 2
    assert row["aggregate_median_s"] > 0
    assert row["executed_window"] == 128
    assert row["full_die"] is False
    assert row["n_tiles"] > 0
    assert row["output_bytes"] > 0
    output = np.load(workspace / "w128-r1.npy", mmap_mode="r")
    assert output.shape == (128, 128)
    assert np.isfinite(np.asarray(output)).all()

    # the exact 8-member family is closed inside workspace/family
    family = workspace / "family"
    assert {p.name for p in family.iterdir()} == SCALE_CANONICAL_FAMILY
    sums = (family / "SHA256SUMS.txt").read_text().strip().splitlines()
    assert len(sums) == len(SCALE_CANONICAL_FAMILY) - 1
    manifest = load_scale_fixture_manifest(manifest_path)
    fixture_member = json.loads((family / "industrial-scale-fixture.json").read_text())
    assert fixture_member["fixture"]["gds_sha256"] == manifest.gds_sha256

    # identity recomputes from the written run-config member
    run_config_member = json.loads((family / "industrial-scale-run-config.json").read_text())
    payload = run_config_member["run_config"]
    recomputed = compute_scale_run_identity(
        __import__(
            "openlithohub.benchmark.industrial_scale", fromlist=["ScaleRunConfig"]
        ).ScaleRunConfig(
            **{
                **payload,
                "lanes": tuple(payload["lanes"]),
                "window_sizes": tuple(payload["window_sizes"]),
                "microbatch_ladder": tuple(payload.get("microbatch_ladder", [])),
            }
        ),
        measurement_commit=run_config_member["measurement_commit"],
        harness_sha256=run_config_member["source_hashes"]["harness"],
        core_sha256=run_config_member["source_hashes"]["core"],
        verifier_sha256=run_config_member["source_hashes"]["verifier"],
        claim_generator_sha256=run_config_member["source_hashes"]["claim_generator"],
        fixture_manifest_sha256=run_config_member["run_config"]["fixture_manifest_sha256"],
        environment_lock_sha256=run_config_member["environment_lock_sha256"],
    )
    assert recomputed == run_config_member["run_identity"]


def test_p0_identity_is_headline_ineligible(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale-p0"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "a",
        "--windows",
        "128",
        "--tile",
        "64",
        "--forward-profile",
        "P0_IDENTITY",
        "--repeats",
        "2",
        "--device",
        "cpu",
        "--out-root",
        str(out_root),
    )
    assert proc.returncode == 0, proc.stderr
    workspace = Path(json.loads(proc.stdout)["workspace"])
    row = json.loads((workspace / "row-A_LARGE_LAYOUT_STREAMING-128.json").read_text())
    assert row["status"] == "SUCCESS"
    assert row["headline_eligible"] is False, "P0 plumbing can never be a headline"


def test_full_layout_firewall_poisoned_sink() -> None:
    harness = _load("run_scale_benchmark", HARNESS)
    with pytest.raises(harness.FirewallViolationError, match="full-layout"):
        harness.enforce_full_layout_firewall(
            full_die=True, sink_kind="tensor", window=555355, die_px=(555355, 555355)
        )
    # out-of-core sinks are the sanctioned full-die path
    harness.enforce_full_layout_firewall(
        full_die=True, sink_kind="memmap_npy", window=555355, die_px=(555355, 555355)
    )
    harness.enforce_full_layout_firewall(
        full_die=False, sink_kind="memmap_npy", window=4096, die_px=(555355, 555355)
    )


def test_full_die_row_runs_out_of_core(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale-fulldie"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "a",
        "--windows",
        "1024",
        "--full-die",
        "--tile",
        "64",
        "--repeats",
        "1",
        "--device",
        "cpu",
        "--out-root",
        str(out_root),
    )
    assert proc.returncode == 0, proc.stderr
    workspace = Path(json.loads(proc.stdout)["workspace"])
    row = json.loads((workspace / "row-A_LARGE_LAYOUT_STREAMING-1024.json").read_text())
    assert row["status"] == "SUCCESS"
    assert row["full_die"] is True, "1024 window covers the whole 256px die"


def test_lane_b_multi_worker_executes_with_topology(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """Lane B with 3 CPU-emulated workers executes the sharded scheduler;
    the row records the worker topology and bounded-queue facts.  (This
    proves scheduler/ownership semantics ONLY — never GPU validation.)"""
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale-b3"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "b",
        "--worker-count",
        "3",
        "--windows",
        "128",
        "--repeats",
        "1",
        "--tile",
        "64",
        "--device",
        "cpu",
        "--out-root",
        str(out_root),
    )
    assert proc.returncode == 0, proc.stderr
    workspace = Path(json.loads(proc.stdout)["workspace"])
    row = json.loads((workspace / "row-B_MULTI_GPU_SCALING-128.json").read_text())
    assert row["status"] == "SUCCESS"
    assert row["worker_count"] == 3
    first = row["per_repeat"][0]
    assert sum(first["worker_counts"].values()) == first["n_tiles"]
    assert all(count > 0 for count in first["worker_counts"].values())
    assert first["max_resident_results"] <= 3 * 64


def test_harness_refuses_gds_manifest_mismatch(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    tampered = gds.with_name("tampered.gds")
    tampered.write_bytes(gds.read_bytes() + b"extra")
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(tampered),
        "--windows",
        "128",
        "--out-root",
        str(tmp_path / "scale-bad"),
    )
    assert proc.returncode != 0
    assert "must match the frozen manifest" in (proc.stderr + proc.stdout)


def test_formal_run_refused_on_cpu_host(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    manifest_path, gds, _ = prepared_fixture
    out_root = tmp_path / "scale-formal"
    proc = _run_harness(
        "--fixture-manifest",
        str(manifest_path),
        "--gds",
        str(gds),
        "--lanes",
        "a",
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
    assert proc.returncode == 1, proc.stdout
    summary = json.loads(proc.stdout)
    assert summary["provisional"] is False
    assert summary["family_blockers"], "CPU host must carry formal blockers"
    assert any("cuda backend" in b for b in summary["family_blockers"])
    workspace = Path(summary["workspace"])
    assert not (workspace / "family" / "manifest.json").exists(), (
        "refused formal runs write NO family"
    )


def test_environment_lock_hash_is_stable_and_sensitive() -> None:
    lock = scale_environment_lock()
    sha = scale_environment_lock_sha256(lock)
    assert sha == scale_environment_lock_sha256(dict(lock))
    assert sha != scale_environment_lock_sha256({**lock, "count": (lock["count"] or 0) + 1})


def test_lane_b_worker_count_parity(
    prepared_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """Charter §10.4 in CI: lane B at 1/2/3 CPU-emulated workers must
    produce byte-identical out-of-core outputs for the same run config
    (same fixture, same tile plan, same forward)."""
    manifest_path, gds, _ = prepared_fixture
    digests = []
    for workers in (1, 2, 3):
        out_root = tmp_path / f"scale-parity-{workers}"
        proc = _run_harness(
            "--fixture-manifest",
            str(manifest_path),
            "--gds",
            str(gds),
            "--lanes",
            "b",
            "--worker-count",
            str(workers),
            "--windows",
            "128",
            "--tile",
            "32",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--device",
            "cpu",
            "--out-root",
            str(out_root),
        )
        assert proc.returncode == 0, proc.stderr
        workspace = Path(json.loads(proc.stdout)["workspace"])
        row = json.loads((workspace / "row-B_MULTI_GPU_SCALING-128.json").read_text())
        assert row["status"] == "SUCCESS"
        import hashlib

        output = workspace / "w128-r0.npy"
        digests.append(hashlib.sha256(output.read_bytes()).hexdigest())
    assert digests[0] == digests[1] == digests[2], (
        "1/2/3-worker runs must produce byte-identical out-of-core outputs"
    )
