"""Industrial Scale benchmark protocol + fixture authority tests (S1).

Covers the frozen namespace, run-identity sensitivity, fixture-manifest
authority, the nvidia-smi topology parser, formal blockers, and the
fixture preparation script end to end on synthetic PDB-shaped trees.
No real PDB data and no GPU required.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from openlithohub.benchmark.industrial_scale import (
    RESULTS_ROOT,
    SCALE_CANONICAL_FAMILY,
    ScaleFixtureManifest,
    ScaleRunConfig,
    ScaleStatus,
    compute_scale_run_identity,
    dense_float32_equivalent_bytes,
    formal_scale_blockers,
    load_scale_fixture_manifest,
    parse_nvidia_smi_topology,
    scale_environment_lock,
    scale_environment_lock_sha256,
)

REPO = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = REPO / "scripts" / "prepare_industrial_scale_fixture.py"


def _load_prepare():
    spec = importlib.util.spec_from_file_location(
        "prepare_industrial_scale_fixture", PREPARE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- namespace isolation + frozen vocabulary ------------------------------------


def test_scale_namespace_is_independent() -> None:
    assert RESULTS_ROOT == "benchmarks/results/industrial-scale"
    assert "results/industrial/" not in RESULTS_ROOT
    assert "results/industrial-v2/" not in RESULTS_ROOT


def test_status_vocabulary_is_frozen() -> None:
    assert {s.value for s in ScaleStatus} == {
        "SUCCESS",
        "FAILED",
        "FAILED_CORRECTNESS",
        "NOT_RUN_MEMORY_POLICY",
        "NOT_RUN_TIME_POLICY",
        "NOT_RUN_ENVIRONMENT",
        "UNSUPPORTED",
    }


def test_canonical_family_has_exactly_eight_members() -> None:
    assert len(SCALE_CANONICAL_FAMILY) == 8
    assert "manifest.json" in SCALE_CANONICAL_FAMILY
    assert "SHA256SUMS.txt" in SCALE_CANONICAL_FAMILY


def test_dense_equivalent_math() -> None:
    assert dense_float32_equivalent_bytes(0, 0) == 0
    # 555355² die @1nm/px ≈ 1.12 TiB (the v1.1 Ibex full-die equivalent)
    assert dense_float32_equivalent_bytes(555355, 555355) == 1_233_676_704_100
    # Microwatt die area ≈ 10.9022 mm² → ~43.6 TB dense equivalent
    side = 3_301_848  # px per side from area at 1 nm/px
    total = dense_float32_equivalent_bytes(side, side)
    assert 4.3e13 < total < 4.4e13
    with pytest.raises(ValueError):
        dense_float32_equivalent_bytes(-1, 1)


# ---- run identity sensitivity ----------------------------------------------------


def _identity(cfg: ScaleRunConfig, **overrides: str) -> str:
    kwargs = dict(
        measurement_commit="a" * 40,
        harness_sha256="h",
        core_sha256="c",
        verifier_sha256="v",
        claim_generator_sha256="g",
        fixture_manifest_sha256="f" * 64,
        environment_lock_sha256="e" * 64,
    )
    kwargs.update(overrides)
    return compute_scale_run_identity(cfg, **kwargs)


@pytest.mark.parametrize(
    "mutation",
    [
        {"lanes": ("B_MULTI_GPU_SCALING",)},
        {"device_backend": "cuda"},
        {"gpu_count": 3},
        {"worker_count": 3},
        {"dtype": "bf16"},
        {"tf32_matmul": True},
        {"deterministic_algorithms": False},
        {"tile_size": 512},
        {"halo_px": 32},
        {"microbatch": 4},
        {"queue_depth": 8},
        {"forward_profile": "P0_IDENTITY"},
        {"forward_radius": 6},
        {"forward_sigma_nm": 2.0},
        {"sink_kind": "manhattan"},
        {"output_semantics": "other"},
        {"warmup_count": 2},
        {"repeat_count": 7},
        {"window_sizes": (16384,)},
        {"fixture_gds_sha256": "9" * 64},
        {"selected_layer": "67:20"},
        {"pixel_nm": 0.5},
    ],
)
def test_identity_sensitive_to_every_semantic_knob(mutation: dict) -> None:
    base = _identity(ScaleRunConfig())
    mutated = _identity(ScaleRunConfig().with_changes(**mutation))
    assert base != mutated, f"identity ignored semantic change {mutation}"


def test_identity_sensitive_to_commit_and_sources_and_fixture() -> None:
    base = _identity(ScaleRunConfig())
    assert base != _identity(ScaleRunConfig(), measurement_commit="b" * 40)
    assert base != _identity(ScaleRunConfig(), harness_sha256="h2")
    assert base != _identity(ScaleRunConfig(), core_sha256="c2")
    assert base != _identity(ScaleRunConfig(), verifier_sha256="v2")
    assert base != _identity(ScaleRunConfig(), claim_generator_sha256="g2")
    assert base != _identity(ScaleRunConfig(), fixture_manifest_sha256="a" * 64)
    assert base != _identity(ScaleRunConfig(), environment_lock_sha256="1" * 64)


# ---- environment lock + topology parser -----------------------------------------


TOPOLOGY_TEXT = """# GPU-GPU direct peer detections are noted with 'PIX 0'
GPU0	GPU1	GPU2	CPU Affinity	NUMA Affinity
GPU0	 X 	PIX	SYS	0-7	0
GPU1	PIX	 X 	SYS	0-7	0
GPU2	SYS	SYS	 X 	8-15	1
Legend:

  X    = self"""


def test_topology_parser_extracts_pci_bus_ids() -> None:
    devices = parse_nvidia_smi_topology(TOPOLOGY_TEXT)
    assert [d["gpu_index"] for d in devices] == ["0", "1", "2"]
    assert devices[0]["pci_bus_id"] == "0"
    assert devices[2]["pci_bus_id"] == "1"
    assert parse_nvidia_smi_topology("") == []
    assert parse_nvidia_smi_topology("Legend:\n  X = self") == []


def test_environment_lock_completeness_on_cpu_host() -> None:
    lock = scale_environment_lock()
    for key in (
        "available",
        "count",
        "requested_gpu_count",
        "devices",
        "driver_version",
        "torch_cuda_version",
        "torch_version",
        "cudnn_version",
        "tf32_matmul",
        "tf32_cudnn",
        "topology",
    ):
        assert key in lock, f"scale environment lock missing {key!r}"
    if not lock["available"]:
        assert lock["devices"] == [] and lock["count"] == 0
        assert lock["topology"] == []
    # hashable and mutation-sensitive
    assert scale_environment_lock_sha256(lock) != scale_environment_lock_sha256(
        {**lock, "requested_gpu_count": 3}
    )


# ---- formal blockers --------------------------------------------------------------


def _cuda_lock(count: int = 3) -> dict:
    return {
        "available": True,
        "count": count,
        "requested_gpu_count": count,
        "devices": [],
        "driver_version": "550",
        "torch_cuda_version": "12.4",
        "torch_version": "2.14",
        "cudnn_version": 90100,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "topology": [],
    }


def _good_rows(lanes: tuple[str, ...]) -> dict:
    return {
        lane: {
            "status": "SUCCESS",
            "correctness_witness_pass": True,
            "repeat_count": 5,
        }
        for lane in lanes
    }


def test_formal_blockers_empty_on_complete_formal_run() -> None:
    cfg = ScaleRunConfig(
        device_backend="cuda", gpu_count=3, fixture_gds_sha256="9" * 64, selected_layer="66:44"
    )
    assert (
        formal_scale_blockers(
            env_lock=_cuda_lock(),
            run_config=cfg,
            lane_rows=_good_rows(cfg.lanes),
            git_clean=True,
            provisional=False,
        )
        == []
    )


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"env_lock": {"available": False, "count": 0}}, "CUDA measurement environment"),
        ({"git_clean": False}, "dirty tracked tree"),
        ({"provisional": True}, "provisional"),
        ({"lane_rows": {}}, "has no measurement row"),
    ],
)
def test_formal_blockers_fail_closed(kwargs: dict, fragment: str) -> None:
    cfg = ScaleRunConfig(
        device_backend="cuda", gpu_count=3, fixture_gds_sha256="9" * 64, selected_layer="66:44"
    )
    values = {
        "env_lock": _cuda_lock(),
        "run_config": cfg,
        "lane_rows": _good_rows(cfg.lanes),
        "git_clean": True,
        "provisional": False,
    }
    values.update(kwargs)
    blockers = formal_scale_blockers(**values)
    assert any(fragment in b for b in blockers)


def test_formal_blockers_refuse_cpu_emulation_and_layer_and_fixture_gaps() -> None:
    cfg = ScaleRunConfig(device_backend="cuda", gpu_count=3)
    blockers = formal_scale_blockers(
        env_lock=_cuda_lock(count=1),
        run_config=cfg,
        lane_rows=_good_rows(cfg.lanes),
        git_clean=True,
        provisional=False,
    )
    assert any("no selected layer" in b for b in blockers)
    assert any("no fixture gds sha256" in b for b in blockers)
    assert any("environment has 1 GPU(s)" in b for b in blockers)
    # CPU emulation can never satisfy a formal cuda-backend run
    blockers_cpu = formal_scale_blockers(
        env_lock={"available": False, "count": 0},
        run_config=cfg,
        lane_rows=_good_rows(cfg.lanes),
        git_clean=True,
        provisional=False,
    )
    assert any("CUDA measurement environment" in b for b in blockers_cpu)


# ---- fixture manifest authority ----------------------------------------------------


def _manifest(**overrides) -> ScaleFixtureManifest:
    values = dict(
        source_repository="org/PDB",
        source_commit="a" * 40,
        design="ibex",
        gds_sha256="b" * 64,
        gds_bytes=1000,
        top_cell="ibex_core",
        dbu_nm=1.0,
        bbox_dbu=(0, 0, 1000, 2000),
        pixel_nm=1.0,
        die_size_px=(1000, 2000),
        dense_float32_equivalent_bytes=dense_float32_equivalent_bytes(1000, 2000),
        layers=("66:44", "67:20"),
        selected_layer="66:44",
        preparation_script_sha256="c" * 64,
    )
    values.update(overrides)
    return ScaleFixtureManifest(**values)


def test_valid_manifest_roundtrip(tmp_path: Path) -> None:
    from openlithohub.benchmark.industrial_scale import write_strict_json

    path = tmp_path / "fixture-manifest.json"
    write_strict_json(path, _manifest().to_payload())
    loaded = load_scale_fixture_manifest(path)
    assert loaded.selected_layer == "66:44"
    assert loaded.die_size_px == (1000, 2000)


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"selected_layer": "99:99"}, "not among the enumerated layers"),
        ({"layers": ()}, "no layers recorded"),
        ({"gds_sha256": "short"}, "64-hex"),
        ({"gds_bytes": 0}, "positive"),
        ({"die_size_px": (5, 5)}, "inconsistent with bbox"),
        (
            {"dense_float32_equivalent_bytes": 1},
            "dense_float32_equivalent_bytes inconsistent",
        ),
        ({"bbox_dbu": (0, 0, 0, 0)}, "positive-area box"),
        ({"source_commit": ""}, "missing"),
    ],
)
def test_manifest_validation_failures(overrides: dict, fragment: str) -> None:
    problems = _manifest(**overrides).validate()
    assert any(fragment in p for p in problems)


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    from openlithohub.benchmark.industrial_scale import write_strict_json

    path = tmp_path / "m.json"
    write_strict_json(path, {"schema": "something.else"})
    with pytest.raises(ValueError, match="schema"):
        load_scale_fixture_manifest(path)


# ---- fixture preparation script (synthetic PDB trees) -------------------------------


def _git_commit_repo(source_root: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(source_root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.com")
    git("config", "user.name", "fixture")
    git("add", "-A")
    git("commit", "-q", "-m", "pdb fixture")
    return git("rev-parse", "HEAD")


def _write_ibex_gds(path: Path, width_px: int = 256, height_px: int = 192) -> None:
    import klayout.db as db

    ly = db.Layout()
    ly.dbu = 0.001  # 1 nm per DBU
    li = ly.layer(66, 44)
    extra = ly.layer(67, 20)
    top = ly.create_cell("ibex_core")
    top.shapes(li).insert(db.Box(0, 0, width_px, height_px))  # nm (px @1nm)
    top.shapes(extra).insert(db.Box(10, 10, 20, 20))
    ly.write(str(path))


def _split_into_chunks(gds: Path, chunk_dir: Path, chunk_size: int = 64) -> list[Path]:
    import string

    data = gds.read_bytes()
    chunk_dir.mkdir(parents=True, exist_ok=True)
    letters = string.ascii_lowercase
    paths = []
    for index in range(0, max(1, (len(data) + chunk_size - 1) // chunk_size)):
        part = data[index * chunk_size : (index + 1) * chunk_size]
        name = f"microwatt.gds.part_a{letters[index // 26]}{letters[index % 26]}"
        target = chunk_dir / name
        target.write_bytes(part)
        paths.append(target)
    return paths


@pytest.fixture
def pdb_tree(tmp_path: Path) -> tuple[Path, str]:
    source_root = tmp_path / "PDB-Physical-Design-Database"
    (source_root / "layout" / "sky130hd" / "ibex").mkdir(parents=True)
    (source_root / "layout" / "sky130hd" / "microwatt").mkdir(parents=True)
    _write_ibex_gds(source_root / "layout" / "sky130hd" / "ibex" / "ibex.gds")
    return source_root, _git_commit_repo(source_root)


def test_prepare_ibex_fixture_end_to_end(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, commit = pdb_tree
    payload = prepare.prepare_fixture(
        source_root=source_root,
        design="ibex",
        selected_layer="66:44",
        pixel_nm=1.0,
        output_dir=tmp_path / "fixtures" / "ibex",
        expected_commit=commit,
    )
    assert payload["top_cell"] == "ibex_core"
    assert payload["selected_layer"] == "66:44"
    assert tuple(payload["die_size_px"]) == (256, 192)
    assert payload["dense_float32_equivalent_bytes"] == dense_float32_equivalent_bytes(256, 192)
    assert set(payload["layers"]) == {"66:44", "67:20"}
    manifest_path = tmp_path / "fixtures" / "ibex" / "fixture-manifest.json"
    loaded = load_scale_fixture_manifest(manifest_path)
    assert loaded.gds_sha256 == payload["gds_sha256"]


def test_prepare_microwatt_from_split_chunks(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, commit = pdb_tree
    import hashlib

    import klayout.db as db

    microwatt_dir = source_root / "layout" / "sky130hd" / "microwatt"
    full = microwatt_dir / "microwatt_full.gds"
    ly = db.Layout()
    ly.dbu = 0.001
    li = ly.layer(66, 44)
    top = ly.create_cell("microwatt")
    top.shapes(li).insert(db.Box(0, 0, 128, 128))
    ly.write(str(full))
    original_sha = hashlib.sha256(full.read_bytes()).hexdigest()

    chunk_dir = microwatt_dir
    _split_into_chunks(full, chunk_dir)
    full.unlink()  # only the split chunks remain, as in the real PDB tree

    payload = prepare.prepare_fixture(
        source_root=source_root,
        design="microwatt",
        selected_layer="66:44",
        pixel_nm=1.0,
        output_dir=tmp_path / "fixtures" / "microwatt",
        expected_commit=commit,
    )
    assert payload["top_cell"] == "microwatt"
    assert len(payload["split_chunks"]) >= 2
    # the canonical-order concatenation reproduces the ORIGINAL single-file bytes
    assert payload["gds_sha256"] == original_sha
    # re-preparing from the chunks yields the identical digest (deterministic)
    again = prepare.prepare_fixture(
        source_root=source_root,
        design="microwatt",
        selected_layer="66:44",
        pixel_nm=1.0,
        output_dir=tmp_path / "fixtures" / "microwatt-2",
        expected_commit=commit,
    )
    assert again["gds_sha256"] == payload["gds_sha256"]
    assert tuple(again["die_size_px"]) == (128, 128)


def test_prepare_refuses_wrong_commit(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, _ = pdb_tree
    with pytest.raises(prepare.FixtureError, match="expected frozen PDB commit"):
        prepare.prepare_fixture(
            source_root=source_root,
            design="ibex",
            selected_layer="66:44",
            pixel_nm=1.0,
            output_dir=tmp_path / "out",
            expected_commit="f" * 40,
        )


def test_prepare_refuses_layer_not_in_file(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, commit = pdb_tree
    with pytest.raises(prepare.FixtureError, match="not present"):
        prepare.prepare_fixture(
            source_root=source_root,
            design="ibex",
            selected_layer="99:99",
            pixel_nm=1.0,
            output_dir=tmp_path / "out",
            expected_commit=commit,
        )


def test_prepare_refuses_wrong_top_cell(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, commit = pdb_tree
    with pytest.raises(prepare.FixtureError, match="top cell"):
        prepare.prepare_fixture(
            source_root=source_root,
            design="ibex",
            selected_layer="66:44",
            pixel_nm=1.0,
            output_dir=tmp_path / "out",
            expected_commit=commit,
            expected_top_cell="not_ibex",
        )


def test_prepare_refuses_truncated_chunk(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, commit = pdb_tree
    import klayout.db as db

    microwatt_dir = source_root / "layout" / "sky130hd" / "microwatt"
    full = microwatt_dir / "microwatt_full.gds"
    ly = db.Layout()
    ly.dbu = 0.001
    li = ly.layer(66, 44)
    top = ly.create_cell("microwatt")
    top.shapes(li).insert(db.Box(0, 0, 128, 128))
    ly.write(str(full))
    _split_into_chunks(full, microwatt_dir)
    full.unlink()
    # truncate one chunk in place — assembly must fail KLayout validation
    victim = sorted(p for p in microwatt_dir.iterdir() if ".part_" in p.name)[1]
    victim.write_bytes(victim.read_bytes()[:4])

    with pytest.raises(prepare.FixtureError, match="KLayout cannot read"):
        prepare.prepare_fixture(
            source_root=source_root,
            design="microwatt",
            selected_layer="66:44",
            pixel_nm=1.0,
            output_dir=tmp_path / "out",
            expected_commit=commit,
        )


def test_prepare_refuses_expected_sha_mismatch(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, commit = pdb_tree
    with pytest.raises(prepare.FixtureError, match="sha256"):
        prepare.prepare_fixture(
            source_root=source_root,
            design="ibex",
            selected_layer="66:44",
            pixel_nm=1.0,
            output_dir=tmp_path / "out",
            expected_commit=commit,
            expected_gds_sha256="e" * 64,
        )


def test_prepare_refuses_implicit_layer_string(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    prepare = _load_prepare()
    source_root, commit = pdb_tree
    with pytest.raises(prepare.FixtureError, match="LAYER:DTYPE"):
        prepare.prepare_fixture(
            source_root=source_root,
            design="ibex",
            selected_layer="6644",
            pixel_nm=1.0,
            output_dir=tmp_path / "out",
            expected_commit=commit,
        )


def test_cli_end_to_end(pdb_tree: tuple[Path, str], tmp_path: Path) -> None:
    source_root, commit = pdb_tree
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--source-root",
            str(source_root),
            "--design",
            "ibex",
            "--selected-layer",
            "66:44",
            "--output",
            str(tmp_path / "cli-out"),
            "--expected-commit",
            commit,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "FIXTURE PREP: PASS" in proc.stdout
    assert load_scale_fixture_manifest(tmp_path / "cli-out" / "fixture-manifest.json")

    bad = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--source-root",
            str(source_root),
            "--design",
            "ibex",
            "--selected-layer",
            "66:44",
            "--output",
            str(tmp_path / "cli-bad"),
            "--expected-commit",
            "f" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert bad.returncode == 1
    assert "FIXTURE PREP: FAIL" in bad.stderr
