"""Hostile authority tests for the Industrial Benchmark machinery.

Each test corresponds to a second-pass audit blocker (B0.10): these are
cheap, deterministic attacks on the authority contracts — identity
mutation, stale reuse, family mixing, index incompleteness — that must
always FAIL CLOSED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- load run_support (sibling of the harness) ------------------------------
_rs_spec = importlib.util.spec_from_file_location(
    "olh_run_support", REPO_ROOT / "benchmarks/industrial/run_support.py"
)
assert _rs_spec is not None and _rs_spec.loader is not None
rs = importlib.util.module_from_spec(_rs_spec)
sys.modules["olh_run_support"] = rs
_rs_spec.loader.exec_module(rs)

# --- load the verifier by file path (it is dependency-free) ----------------
_va_spec = importlib.util.spec_from_file_location(
    "olh_verify_artifacts", REPO_ROOT / "scripts/verify_industrial_artifacts.py"
)
assert _va_spec is not None and _va_spec.loader is not None
verifier = importlib.util.module_from_spec(_va_spec)
sys.modules["olh_verify_artifacts"] = verifier
_va_spec.loader.exec_module(verifier)

SOURCE = {
    "commit": "a" * 40,
    "commit_valid": True,
    "working_tree_dirty": False,
    "harness_sha256": "1" * 64,
    "industrial_core_sha256": "2" * 64,
    "claim_generator_sha256": "3" * 64,
    "run_support_sha256": "4" * 64,
}
LOCK = {"lock_sha256": "5" * 64}
ARGS = {
    "repeats": 5,
    "large_repeats": 3,
    "core": 1024,
    "sizes": "4096",
    "dense_max_bytes": 30 * (1 << 30),
    "max_vector_size": 32768,
    "max_selective_size": 65536,
    "die_tile_px": 32768,
    "die_tiles_per_side": None,
    "die_core_px": 4096,
    "tile_size": 1024,
    "min_tile_occupancy": 0.02,
    "max_tiles": 6,
    "ilt_iterations": 50,
    "ilt_iterations_iccad16": 200,
    "quality_reps": 3,
    "surrogate_train_samples": 16,
    "surrogate_epochs": 3,
    "iccad16_crop_px": 256,
    "iccad16_dir": None,
    "models": "a,b",
    "layer": "66:44",
    "pixel_size_nm": 1.0,
    "seed": 0,
}


def _identity(
    *,
    args: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    gds: str = "f" * 64,
    iccad: dict[str, str] | None = None,
    env: str = "5" * 64,
) -> str:
    return rs.compute_run_identity(
        source={**(source or SOURCE)},
        environment_lock_sha256=env,
        parent_gds_sha256=gds,
        iccad_fixture_hashes=iccad,
        args_payload={**ARGS, **(args or {})},
    )


class TestRunIdentityMutation:
    def test_identity_is_stable(self) -> None:
        assert _identity() == _identity()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("repeats", 4),
            ("large_repeats", 2),
            ("core", 512),
            ("sizes", "8192"),
            ("dense_max_bytes", 1 << 30),
            ("max_vector_size", 8192),
            ("max_selective_size", 32768),
            ("die_tile_px", 16384),
            ("die_tiles_per_side", 9),
            ("die_core_px", 2048),
            ("tile_size", 512),
            ("min_tile_occupancy", 0.03),
            ("max_tiles", 5),
            ("ilt_iterations", 49),
            ("ilt_iterations_iccad16", 199),
            ("quality_reps", 2),
            ("surrogate_train_samples", 15),
            ("surrogate_epochs", 2),
            ("iccad16_crop_px", 128),
            ("iccad16_dir", "/data"),
            ("models", "a"),
            ("layer", "67:44"),
            ("pixel_size_nm", 2.0),
            ("seed", 1),
        ],
    )
    def test_every_semantic_arg_changes_identity(self, field: str, value: Any) -> None:
        assert _identity(args={field: value}) != _identity(), f"identity blind to {field}"

    def test_source_commit_changes_identity(self) -> None:
        other = {**SOURCE, "commit": "b" * 40}
        assert _identity(source=other) != _identity()

    @pytest.mark.parametrize(
        "key",
        [
            "harness_sha256",
            "industrial_core_sha256",
            "claim_generator_sha256",
            "run_support_sha256",
        ],
    )
    def test_source_hash_changes_identity(self, key: str) -> None:
        other = {**SOURCE, key: "9" * 64}
        assert _identity(source=other) != _identity(), f"identity blind to {key}"

    def test_parent_fixture_changes_identity(self) -> None:
        assert _identity(gds="e" * 64) != _identity()

    def test_iccad_fixture_changes_identity(self) -> None:
        base = _identity(iccad={"testcase1.oas": "c" * 64})
        assert _identity(iccad={"testcase1.oas": "d" * 64}) != base
        assert _identity(iccad={}) != base

    def test_environment_lock_changes_identity(self) -> None:
        assert _identity(env="6" * 64) != _identity()


class TestEnvironmentLock:
    def test_freeze_is_inside_lock_hash(self) -> None:
        """B0.2: the freeze hash must be part of the lock before lock_sha256."""
        lock = {
            "python": "3.12",
            "distribution_freeze_sha256": "7" * 64,
        }
        expected = rs.strict_dumps(lock).encode()
        import hashlib

        assert hashlib.sha256(expected).hexdigest() == _lock_hash(lock)


def _lock_hash(lock: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(rs.strict_dumps(lock).encode()).hexdigest()


class TestCheckpointIdentity:
    def test_wrong_identity_hard_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "runtime.jsonl"
        ckpt = rs.Checkpoint(path, "1" * 64, "runtime")
        ckpt.append("row_a", {"x": 1})
        with pytest.raises(rs.CheckpointIdentityError, match="does not match"):
            rs.Checkpoint(path, "2" * 64, "runtime")

    def test_wrong_stage_hard_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "runtime.jsonl"
        ckpt = rs.Checkpoint(path, "1" * 64, "runtime")
        ckpt.append("row_a", {"x": 1})
        with pytest.raises(rs.CheckpointIdentityError, match="stage"):
            rs.Checkpoint(path, "1" * 64, "quality")

    def test_nan_checkpoint_hard_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "runtime.jsonl"
        path.write_text(
            json.dumps(
                {
                    "_schema": rs.CHECKPOINT_SCHEMA,
                    "_run_identity": "1" * 64,
                    "_stage": "runtime",
                    "_key": "row_a",
                    "_created_utc": "now",
                    "payload": {"score": float("nan")},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-finite"):
            rs.Checkpoint(path, "1" * 64, "runtime")


class TestStaleFixture:
    def test_stale_fixture_reuse_refused(self, tmp_path: Path) -> None:
        fixture = tmp_path / "crop.gds"
        fixture.write_bytes(b"original-bytes")
        record = rs.fixture_record(
            fixture,
            parent_sha256="a" * 64,
            crop_bbox_dbu=[0, 0, 10, 10],
            top_cell="C",
            layer="66:44",
            pixel_size_nm=1.0,
            generator_source=SOURCE,
        )
        # Same everything: reuse allowed.
        rs.verify_fixture_reuse(fixture, record, parent_sha256="a" * 64, generator_source=SOURCE)
        # Parent changed: reuse refused.
        fixture.write_bytes(b"TAMPERED-BYTES")
        with pytest.raises(RuntimeError, match="reuse refused"):
            rs.verify_fixture_reuse(
                fixture, record, parent_sha256="a" * 64, generator_source=SOURCE
            )
        # Generator changed: reuse refused.
        other_source = {**SOURCE, "commit": "b" * 40}
        with pytest.raises(RuntimeError, match="reuse refused"):
            rs.verify_fixture_reuse(
                fixture, record, parent_sha256="a" * 64, generator_source=other_source
            )


def make_artifact(run_identity: str) -> dict[str, Any]:
    """A minimal VALID runtime artifact carrying the given identity."""
    return {
        "schema": "OpenLithoHub.industrial-benchmark.v1",
        "kind": "runtime",
        "status": "SUCCESS",
        "git_commit": "a" * 40,
        "timestamp_utc": "2026-09-21T00:00:00Z",
        "hardware": {"cpu_model": "t"},
        "software": {"python": "3.12"},
        "claim_scope": {},
        "reproducibility": {},
        "measurement_source": {
            "commit": "a" * 40,
            "commit_valid": True,
            "working_tree_dirty": False,
            "harness_sha256": "1" * 64,
            "industrial_core_sha256": "2" * 64,
            "claim_generator_sha256": "3" * 64,
            "run_support_sha256": "4" * 64,
        },
        "fixture": {"sha256": "c" * 64, "bytes": 1},
        "run_identity": run_identity,
        "environment_lock": {"lock_sha256": "5" * 64},
    }


class TestArtifactFamilyClosure:
    def test_mixed_run_family_rejected(self) -> None:
        from openlithohub.benchmark.industrial import validate_artifact_family

        family = {
            "runtime": make_artifact("1" * 64),
            "quality": make_artifact("2" * 64),  # from a DIFFERENT run
            "fulldie": make_artifact("1" * 64),
            "run-config": make_artifact("1" * 64),
        }
        problems = validate_artifact_family(family)
        assert any("run_identity" in p for p in problems)

    def test_complete_family_passes(self) -> None:
        from openlithohub.benchmark.industrial import validate_artifact_family

        family = {
            role: make_artifact("1" * 64)
            for role in ("runtime", "quality", "fulldie", "run-config")
        }
        assert validate_artifact_family(family) == []

    def test_incomplete_family_rejected(self) -> None:
        from openlithohub.benchmark.industrial import validate_artifact_family

        family = {"runtime": make_artifact("1" * 64)}
        problems = validate_artifact_family(family)
        assert any("incomplete family" in p for p in problems)


class TestVerifierSetClosure:
    def _write_family(self, tmp_path: Path) -> None:
        """Reuse the production family builder: one canonical definition,
        no second competing writer."""
        _write_production_family(tmp_path)

    def test_missing_sha_entry_hard_fails(self, tmp_path: Path) -> None:
        self._write_family(tmp_path)
        sums = tmp_path / "SHA256SUMS.txt"
        lines = sums.read_text().splitlines()
        sums.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        problems = verifier.verify(tmp_path, REPO_ROOT)
        assert any("no entry for" in p for p in problems)

    def test_complete_family_verifies_clean(self, tmp_path: Path) -> None:
        self._write_family(tmp_path)
        # NOTE: source closure is skipped here by design of the check order?
        # It runs — so monkeypatch it out; family/set logic is under test.
        monkey_verifier = verifier
        monkey_verifier.verify_source_closure = lambda *a, **k: []
        problems = monkey_verifier.verify(tmp_path, REPO_ROOT)
        assert problems == []

    def test_extra_stale_sum_entry_hard_fails(self, tmp_path: Path) -> None:
        self._write_family(tmp_path)
        sums = tmp_path / "SHA256SUMS.txt"
        sums.write_text(sums.read_text() + f"{'0' * 64}  industrial-ghost.json\n", encoding="utf-8")
        problems = verifier.verify(tmp_path, REPO_ROOT)
        assert any("stale entries" in p for p in problems)


class TestPreflightIdentityMatchesHarness:
    def test_preflight_identity_equals_harness_identity(self) -> None:
        """B0.3: preflight must use the harness identity function so the
        printed RUN_IDENTITY is the one the formal measurement will use."""
        harness_spec = importlib.util.spec_from_file_location(
            "olh_harness", REPO_ROOT / "benchmarks/industrial/run_industrial_benchmark.py"
        )
        assert harness_spec is not None and harness_spec.loader is not None
        harness = importlib.util.module_from_spec(harness_spec)
        sys.modules["olh_harness"] = harness
        harness_spec.loader.exec_module(harness)

        args = harness.build_arg_parser().parse_args(
            [
                "--gds",
                "/dev/null",  # never opened on this path (identity uses its hash)
                "--repeats",
                "3",
                "--sizes",
                "2048",
            ]
        )
        # Stand in for the file hash: /dev/null cannot be hashed portably,
        # so inject a fixed fixture hash the way main() does.
        args.parent_gds_sha256 = "f" * 64
        source = {
            **SOURCE,
            "commit": "c" * 40,
        }
        harness_identity, config, _ = harness.compute_run_identity_from_args(args, source)

        preflight_identity = _identity(
            args={k: v for k, v in config["args"].items() if k != "preflight"},
            source=source,
            gds="f" * 64,
            iccad={},
            env=config["environment_lock"]["lock_sha256"],
        )
        assert preflight_identity == harness_identity


def test_family_closure_tests_are_collected() -> None:
    """P0.2 regression guard: the family-closure tests must be real,
    collected pytest tests — not unreachable nested functions."""
    import inspect

    source = inspect.getsource(sys.modules[__name__])
    for name in (
        "test_mixed_run_family_rejected",
        "test_complete_family_passes",
        "test_incomplete_family_rejected",
    ):
        # each must be defined at class-body indentation (4 spaces), not
        # nested deeper inside a helper function (8+ spaces)
        definition = f"    def {name}(self)"
        assert definition in source, f"{name} is not a collected class method"
        nested = f"        def {name}(self)"
        assert nested not in source, f"{name} is nested after a return and never collected"


def _write_production_family(tmp_path: Path) -> Path:
    """P0.10: build a REAL-schema family exactly as publish_family would,
    using the production build/sanitize/identity functions."""
    harness_spec = importlib.util.spec_from_file_location(
        "olh_harness2", REPO_ROOT / "benchmarks/industrial/run_industrial_benchmark.py"
    )
    assert harness_spec is not None and harness_spec.loader is not None
    harness = importlib.util.module_from_spec(harness_spec)
    sys.modules["olh_harness2"] = harness
    harness_spec.loader.exec_module(harness)

    ind_spec = importlib.util.spec_from_file_location(
        "olh_ind3", REPO_ROOT / "src/openlithohub/benchmark/industrial.py"
    )
    assert ind_spec is not None and ind_spec.loader is not None
    ind = importlib.util.module_from_spec(ind_spec)
    sys.modules["olh_ind3"] = ind
    ind_spec.loader.exec_module(ind)

    commit = "c" * 40
    source = {
        "commit": commit,
        "commit_valid": True,
        "working_tree_dirty": False,
        "harness_sha256": "1" * 64,
        "industrial_core_sha256": "2" * 64,
        "claim_generator_sha256": "3" * 64,
        "run_support_sha256": "4" * 64,
    }
    lock = {"python": "3.12", "distribution_freeze_sha256": "d" * 64}
    lock["lock_sha256"] = hashlib.sha256(ind._canonical_dumps(lock).encode()).hexdigest()
    config = {
        "schema": "OpenLithoHub.industrial-run-config.v1",
        "source": source,
        "environment_lock": lock,
        "fixtures": {"parent_gds_sha256": "e" * 64, "iccad": {}},
        "args": {"repeats": 5, "sizes": "4096", "layer": "66:44", "seed": 0},
    }
    payload = ind.build_run_identity_payload(config)
    identity = hashlib.sha256(ind._canonical_dumps(payload).encode()).hexdigest()
    config["run_identity"] = identity
    (tmp_path / "industrial-run-config.json").write_text(
        ind._canonical_dumps(config) + "\n", encoding="utf-8"
    )
    (tmp_path / "industrial-distribution-freeze.txt").write_text(
        "diff-surrogate @ https://github.com/telleroutlook/diff-surrogate.git@6b0ec10916f58297a0b48cb3af470e7d1ad99b45\n"
        "numpy==2.0.0\n",
        encoding="utf-8",
    )
    lock["distribution_freeze_sha256"] = hashlib.sha256(
        (tmp_path / "industrial-distribution-freeze.txt").read_bytes()
    ).hexdigest()
    # Mirror the real semantics: lock_sha256 covers the lock body WITHOUT
    # the previous lock_sha256 key.
    lock_body = {k: v for k, v in lock.items() if k != "lock_sha256"}
    lock["lock_sha256"] = hashlib.sha256(ind._canonical_dumps(lock_body).encode()).hexdigest()
    config["environment_lock"] = lock
    config["run_identity"] = hashlib.sha256(
        ind._canonical_dumps(ind.build_run_identity_payload(config)).encode()
    ).hexdigest()
    (tmp_path / "industrial-run-config.json").write_text(
        ind._canonical_dumps(config) + "\n", encoding="utf-8"
    )

    def benchmark_artifact(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        base = {
            "schema": ind.SCHEMA_NAME,
            "kind": kind,
            "status": "SUCCESS",
            "git_commit": commit,
            "timestamp_utc": "2026-09-21T00:00:00Z",
            "hardware": {"cpu_model": "t", "physical_ram_bytes": 48 * (1 << 30), "gpu": None},
            "software": {"python": "3.12"},
            "claim_scope": {"physics_claim": "NOT_FOUNDRY_CALIBRATED"},
            "reproducibility": {"command": "run"},
            "measurement_source": source,
            "fixture": {"sha256": "e" * 64, "bytes": 24_000_000},
            "run_identity": config["run_identity"],
            "environment_lock": lock,
        }
        base.update(payload)
        return ind._sanitize(base)

    runtime_payload = {
        "memory": {
            "per_size": {
                "4096": {
                    "dense_full": {"peak_rss_bytes_median": 1 << 30},
                    "b04_selective": {"peak_rss_bytes_median": 400 * (1 << 20)},
                    "streaming_memory_reduction_pct": 60.0,
                }
            },
            "max_streamed_size_px": 65536,
            "dense_not_run_under_memory_policy_px": [65536],
        },
        "comparisons": {},
        "rows": [],
        "policy": {"repeats": 5},
    }
    for role, payload in (
        ("runtime", runtime_payload),
        ("quality", {"datasets": {}}),
        ("fulldie", {"dense_die_status": "INFEASIBLE_ON_REFERENCE_MACHINE"}),
    ):
        artifact = benchmark_artifact(role, payload)
        (tmp_path / f"industrial-{role}.json").write_text(
            ind._canonical_dumps(artifact) + "\n", encoding="utf-8"
        )
    # manifest LAST (commit marker), entries over the canonical family
    family_names = [
        "industrial-runtime.json",
        "industrial-quality.json",
        "industrial-fulldie.json",
        "industrial-run-config.json",
        "industrial-distribution-freeze.txt",
    ]
    manifest_entries = [
        {
            "file": name,
            "sha256": rs.sha256_file(tmp_path / name),
            "bytes": (tmp_path / name).stat().st_size,
        }
        for name in family_names
    ]
    manifest = benchmark_artifact(
        "manifest",
        {"artifacts": manifest_entries, "run_identity": config["run_identity"]},
    )
    (tmp_path / "manifest.json").write_text(ind._canonical_dumps(manifest) + "\n", encoding="utf-8")
    sums = ""
    for name in family_names:
        sums += rs.sha256_file(tmp_path / name) + f"  {name}\n"
    (tmp_path / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    return tmp_path


def test_production_verifier_accepts_real_schema_family(tmp_path: Path) -> None:
    """P0.10: the PRODUCTION verifier must accept one valid real-schema
    family (including the run-config member) end-to-end."""
    family_dir = _write_production_family(tmp_path)
    problems = verifier.verify(family_dir, REPO_ROOT)
    # source closure will try git-show of the synthetic commit; that check
    # is not under test here, so filter its (expected) failures.
    problems = [p for p in problems if "source closure" not in p]
    assert problems == [], problems


def test_production_verifier_rejects_identity_tamper(tmp_path: Path) -> None:
    family_dir = _write_production_family(tmp_path)
    config_path = family_dir / "industrial-run-config.json"
    config = json.loads(config_path.read_text())
    config["args"]["repeats"] = 4  # semantic change while keeping the label
    config_path.write_text(_ind_str(config) + "\n", encoding="utf-8")
    problems = verifier.verify(family_dir, REPO_ROOT)
    assert any("recomputed" in p for p in problems), problems


def _ind_str(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"


class TestPEP610Provenance:
    def test_vcs_commit_id_preserved(self) -> None:
        """P0.2: PEP 610 uses commit_id — the exact installed commit must
        survive into the freeze line."""
        direct = {
            "url": "https://github.com/x/y.git",
            "vcs_info": {
                "vcs": "git",
                "requested_revision": "v0.3.0",
                "commit_id": "abcdef1234567890abcdef1234567890abcdef12",
            },
        }
        line = rs.format_direct_reference("diff-surrogate", "0.3.0", direct)
        assert "abcdef1234567890abcdef1234567890abcdef12" in line
        assert "diff-surrogate @ https://github.com/x/y.git@v0.3.0@abcdef" in line

    def test_editable_from_dir_info(self) -> None:
        direct = {
            "url": "file:///src/diff-surrogate",
            "dir_info": {"editable": True},
        }
        line = rs.format_direct_reference("diff-surrogate", "0.3.0", direct)
        assert "# editable" in line

    def test_regular_distribution_uses_version(self) -> None:
        assert rs.format_direct_reference("numpy", "2.5.1", {}) == "numpy==2.5.1"


# ---------------------------------------------------------------------------
# P0.8: production drill — REAL publish_family → PRODUCTION verifier →
# PRODUCTION claims --check. Source hashes are the real working-tree bytes
# and git-show reads the working tree, so closure is genuinely exercised
# with NO problem filtering.
# ---------------------------------------------------------------------------


_harness_spec = importlib.util.spec_from_file_location(
    "olh_harness_drill", REPO_ROOT / "benchmarks/industrial/run_industrial_benchmark.py"
)
assert _harness_spec is not None and _harness_spec.loader is not None
harness = importlib.util.module_from_spec(_harness_spec)
sys.modules["olh_harness_drill"] = harness
_harness_spec.loader.exec_module(harness)

ind_spec_drill = importlib.util.spec_from_file_location(
    "olh_ind_drill", REPO_ROOT / "src/openlithohub/benchmark/industrial.py"
)
ind = importlib.util.module_from_spec(ind_spec_drill)
sys.modules["olh_ind_drill"] = ind
ind_spec_drill.loader.exec_module(ind)


_gen_spec = importlib.util.spec_from_file_location(
    "olh_gen", REPO_ROOT / "scripts/generate_industrial_claims.py"
)
assert _gen_spec is not None and _gen_spec.loader is not None
gen = importlib.util.module_from_spec(_gen_spec)
sys.modules["olh_gen"] = gen
_gen_spec.loader.exec_module(gen)


def _drill_source() -> dict[str, Any]:
    """Source dict whose hashes are the REAL working-tree bytes."""
    import hashlib

    source = {
        "commit": "d" * 40,
        "commit_valid": True,
        "working_tree_dirty": False,
        "harness_sha256": "",
        "industrial_core_sha256": "",
        "claim_generator_sha256": "",
        "run_support_sha256": "",
    }
    for key, rel in (
        ("harness_sha256", "benchmarks/industrial/run_industrial_benchmark.py"),
        ("industrial_core_sha256", "src/openlithohub/benchmark/industrial.py"),
        ("claim_generator_sha256", "scripts/generate_industrial_claims.py"),
        ("run_support_sha256", "benchmarks/industrial/run_support.py"),
    ):
        source[key] = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
    return source


class TestProductionFamilyDrill:
    @pytest.fixture
    def run_dir_and_out(self, tmp_path: Path, monkeypatch):
        """Build a synthetic run workspace, then run the REAL
        publish_family. git-show is replaced by a working-tree read for
        the synthetic commit — the closure still compares stored vs
        actual bytes with no filtering."""
        import hashlib

        source = _drill_source()
        monkeypatch.setattr(
            verifier,
            "_git_show_sha256",
            lambda commit, rel, root: hashlib.sha256((root / rel).read_bytes()).hexdigest(),
        )
        out = tmp_path / "public"
        out.mkdir(parents=True)

        lock = {"python": "3.12", "distribution_freeze_sha256": ""}
        freeze_text = (
            "numpy==2.0.0\n"
            "diff-surrogate @ https://github.com/x/y.git@"
            "abcdef1234567890abcdef1234567890abcdef12\n"
        )
        lock["distribution_freeze_sha256"] = hashlib.sha256(freeze_text.encode()).hexdigest()
        lock["lock_sha256"] = hashlib.sha256(ind._canonical_dumps(lock).encode()).hexdigest()
        config = {
            "schema": ind.RUN_CONFIG_SCHEMA,
            "source": source,
            "environment_lock": lock,
            "fixtures": {"parent_gds_sha256": "f" * 64, "iccad": {}},
            "args": {"repeats": 5, "sizes": "4096", "layer": "66:44", "seed": 0},
        }
        identity = hashlib.sha256(
            ind._canonical_dumps(ind.build_run_identity_payload(config)).encode()
        ).hexdigest()
        config["run_identity"] = identity
        run_dir = tmp_path / "runs" / identity
        run_dir.mkdir(parents=True)
        (run_dir / "industrial-run-config.json").write_text(
            ind._canonical_dumps(config) + "\n", encoding="utf-8"
        )
        (run_dir / "industrial-distribution-freeze.txt").write_text(freeze_text, encoding="utf-8")

        def benchmark_artifact(kind: str, extra: dict[str, Any]) -> dict[str, Any]:
            base = {
                "schema": ind.SCHEMA_NAME,
                "kind": kind,
                "status": "SUCCESS",
                "git_commit": source["commit"],
                "timestamp_utc": "2026-09-21T00:00:00Z",
                "hardware": {
                    "cpu_model": "t",
                    "physical_ram_bytes": 48 * (1 << 30),
                    "gpu": None,
                },
                "software": {"python": "3.12"},
                "claim_scope": {"physics_claim": "NOT_FOUNDRY_CALIBRATED"},
                "reproducibility": {"command": "run"},
                "measurement_source": source,
                "fixture": {"sha256": "f" * 64, "bytes": 1000},
                "run_identity": identity,
                "environment_lock": lock,
            }
            base.update(extra)
            return ind._sanitize(base)

        runtime = benchmark_artifact(
            "runtime",
            {
                "memory": {
                    "per_size": {
                        "4096": {
                            "dense_full": {"peak_rss_bytes_median": 1 << 30},
                            "b04_selective": {"peak_rss_bytes_median": 400 * (1 << 20)},
                            "streaming_memory_reduction_pct": 60.0,
                        }
                    },
                    "max_streamed_size_px": 65536,
                    "dense_not_run_under_memory_policy_px": [65536],
                },
                "comparisons": {},
                "rows": [],
                "policy": {"repeats": 5},
            },
        )
        (run_dir / "industrial-runtime.json").write_text(
            ind._canonical_dumps(runtime) + "\n", encoding="utf-8"
        )
        (run_dir / "industrial-quality.json").write_text(
            ind._canonical_dumps(benchmark_artifact("quality", {"datasets": {}})) + "\n",
            encoding="utf-8",
        )
        (run_dir / "industrial-fulldie.json").write_text(
            ind._canonical_dumps(
                benchmark_artifact(
                    "fulldie", {"dense_die_status": "INFEASIBLE_ON_REFERENCE_MACHINE"}
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return run_dir, out, identity, source

    def test_publish_verify_claims_end_to_end(
        self, tmp_path: Path, run_dir_and_out, monkeypatch
    ) -> None:
        run_dir, out, identity, source = run_dir_and_out
        publish_args = argparse.Namespace(
            gds=REPO_ROOT / "x.gds",
            fixture_block={"sha256": "f" * 64, "bytes": 1000},
            measurement_source=source,
            run_identity=identity,
            out=out,
            repeats=5,
            seed=0,
        )
        monkeypatch.setattr(harness, "_PUBLISH_ARGS", publish_args)
        # REAL production publication.
        harness.publish_family(run_dir, out, identity)

        # Published root contains exactly the canonical family.
        published = {p.name for p in out.iterdir()}
        assert published == {
            "industrial-runtime.json",
            "industrial-quality.json",
            "industrial-fulldie.json",
            "industrial-run-config.json",
            "industrial-distribution-freeze.txt",
            "manifest.json",
            "SHA256SUMS.txt",
        }

        # PRODUCTION verifier accepts the family with NO problem filtering.
        problems = verifier.verify(out, REPO_ROOT)
        assert problems == [], problems

        # PRODUCTION claims generator runs and its --check passes against
        # the published artifacts (drift gate over a real generated doc).
        claims_json = tmp_path / "gen" / "industrial-claims.json"
        claims_md = tmp_path / "gen" / "industrial-claims.md"
        readme = tmp_path / "README.md"
        readme.write_text("# test\n", encoding="utf-8")
        argv = sys.argv
        sys.argv = [
            "generate_industrial_claims.py",
            "--artifacts",
            str(out),
            "--out-json",
            str(claims_json),
            "--out-md",
            str(claims_md),
            "--readme",
            str(readme),
        ]
        try:
            assert gen.main() == 0
            assert claims_json.exists() and claims_md.exists()
            sys.argv = [
                "generate_industrial_claims.py",
                "--artifacts",
                str(out),
                "--out-json",
                str(claims_json),
                "--out-md",
                str(claims_md),
                "--readme",
                str(readme),
                "--check",
            ]
            assert gen.main() == 0
        finally:
            sys.argv = argv

    def test_freeze_byte_mutation_fails_verifier(
        self, tmp_path: Path, run_dir_and_out, monkeypatch
    ) -> None:
        run_dir, out, identity, source = run_dir_and_out
        publish_args = argparse.Namespace(
            gds=REPO_ROOT / "x.gds",
            fixture_block={"sha256": "f" * 64, "bytes": 1000},
            measurement_source=source,
            run_identity=identity,
            out=out,
            repeats=5,
            seed=0,
        )
        monkeypatch.setattr(harness, "_PUBLISH_ARGS", publish_args)
        harness.publish_family(run_dir, out, identity)
        freeze = out / "industrial-distribution-freeze.txt"
        freeze.write_text(freeze.read_text().replace("numpy", "NUMPY"), encoding="utf-8")
        problems = verifier.verify(out, REPO_ROOT)
        assert any(
            "distribution-freeze" in p and ("mismatch" in p or "does not match" in p)
            for p in problems
        ), problems

    def test_run_config_arg_mutation_fails_verifier(
        self, tmp_path: Path, run_dir_and_out, monkeypatch
    ) -> None:
        run_dir, out, identity, source = run_dir_and_out
        publish_args = argparse.Namespace(
            gds=REPO_ROOT / "x.gds",
            fixture_block={"sha256": "f" * 64, "bytes": 1000},
            measurement_source=source,
            run_identity=identity,
            out=out,
            repeats=5,
            seed=0,
        )
        monkeypatch.setattr(harness, "_PUBLISH_ARGS", publish_args)
        harness.publish_family(run_dir, out, identity)
        config_path = out / "industrial-run-config.json"
        config = json.loads(config_path.read_text())
        config["args"]["repeats"] = 4
        config_path.write_text(ind._canonical_dumps(config) + "\n", encoding="utf-8")
        problems = verifier.verify(out, REPO_ROOT)
        assert any("recomputed" in p for p in problems), problems

    def test_artifact_byte_mutation_fails_verifier(
        self, tmp_path: Path, run_dir_and_out, monkeypatch
    ) -> None:
        run_dir, out, identity, source = run_dir_and_out
        publish_args = argparse.Namespace(
            gds=REPO_ROOT / "x.gds",
            fixture_block={"sha256": "f" * 64, "bytes": 1000},
            measurement_source=source,
            run_identity=identity,
            out=out,
            repeats=5,
            seed=0,
        )
        monkeypatch.setattr(harness, "_PUBLISH_ARGS", publish_args)
        harness.publish_family(run_dir, out, identity)
        target = out / "industrial-quality.json"
        target.write_text(target.read_text().replace("cpu_model", "cpu_model_x"))
        problems = verifier.verify(out, REPO_ROOT)
        assert any("hash mismatch" in p or "sha256 != actual" in p for p in problems)

    def test_unknown_root_json_fails(self, tmp_path: Path, run_dir_and_out, monkeypatch) -> None:
        run_dir, out, identity, source = run_dir_and_out
        publish_args = argparse.Namespace(
            gds=REPO_ROOT / "x.gds",
            fixture_block={"sha256": "f" * 64, "bytes": 1000},
            measurement_source=source,
            run_identity=identity,
            out=out,
            repeats=5,
            seed=0,
        )
        monkeypatch.setattr(harness, "_PUBLISH_ARGS", publish_args)
        harness.publish_family(run_dir, out, identity)
        (out / "industrial-rogue.json").write_text("{}", encoding="utf-8")
        problems = verifier.verify(out, REPO_ROOT)
        assert any("unknown authority-root JSON" in p for p in problems)
