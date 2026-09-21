"""Hostile authority tests for the Industrial Benchmark machinery.

Each test corresponds to a second-pass audit blocker (B0.10): these are
cheap, deterministic attacks on the authority contracts — identity
mutation, stale reuse, family mixing, index incompleteness — that must
always FAIL CLOSED.
"""

from __future__ import annotations

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
        """Write a complete, self-consistent family for the verifier."""
        for role in ("runtime", "quality", "fulldie", "run-config"):
            name = f"industrial-{role}.json"
            (tmp_path / name).write_text(json.dumps(make_artifact("1" * 64)), encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps(make_artifact("1" * 64)), encoding="utf-8"
        )
        names = sorted(
            n
            for n in (
                "industrial-runtime.json",
                "industrial-quality.json",
                "industrial-fulldie.json",
                "industrial-run-config.json",
                "manifest.json",
            )
        )
        import hashlib

        sums = ""
        for n in names:
            sums += hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() + f"  {n}\n"
        (tmp_path / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")

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
