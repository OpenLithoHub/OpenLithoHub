"""Tests for the Industrial Benchmark v1 schema/stats/claim core."""

from __future__ import annotations

import json

import pytest

from openlithohub.benchmark.industrial import (
    REPRODUCED_INTERNAL,
    SCHEMA_NAME,
    STATUS_SUCCESS,
    build_claim,
    compare_runtime,
    environment_snapshot,
    git_commit,
    load_artifact,
    memory_reduction_pct,
    percentile,
    relative_reduction_pct,
    speedup,
    summarize,
    validate_artifact,
)


class TestPercentile:
    def test_known_values_odd(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 50.0) == 3.0
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 100.0) == 5.0

    def test_interpolation_even(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5

    def test_p10_p90_bracket_median(self) -> None:
        values = list(range(100))
        s = summarize(values)
        assert s["p10"] <= s["median"] <= s["p90"]
        assert s["n"] == 100
        assert s["min"] == 0
        assert s["max"] == 99

    def test_single_element(self) -> None:
        assert percentile([7.0], 90.0) == 7.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            percentile([], 50.0)


class TestClaimMath:
    def test_speedup(self) -> None:
        assert speedup(10.0, 2.0) == 5.0

    def test_speedup_rejects_zero_candidate(self) -> None:
        with pytest.raises(ValueError):
            speedup(10.0, 0.0)

    def test_memory_reduction(self) -> None:
        assert memory_reduction_pct(1000, 250) == pytest.approx(75.0)

    def test_memory_reduction_rejects_bad_baseline(self) -> None:
        with pytest.raises(ValueError):
            memory_reduction_pct(0, 100)

    def test_relative_reduction(self) -> None:
        assert relative_reduction_pct(20.0, 15.0) == pytest.approx(25.0)

    def test_relative_reduction_zero_baseline(self) -> None:
        assert relative_reduction_pct(0.0, 5.0) == 0.0


class TestClaimBuilder:
    def test_build_claim_ok(self) -> None:
        claim = build_claim(
            claim_id="IB-T",
            metric="m",
            value=1.5,
            unit="x",
            baseline="b",
            dataset="d",
            hardware="h",
            scope="s",
            artifact="a.json",
        )
        assert claim.level == REPRODUCED_INTERNAL
        assert claim.to_dict()["claim_id"] == "IB-T"

    def test_rejects_unknown_level(self) -> None:
        with pytest.raises(ValueError, match="claim level"):
            build_claim(
                claim_id="IB-T",
                metric="m",
                value=1,
                unit="",
                baseline="b",
                dataset="d",
                hardware="h",
                scope="s",
                artifact="a.json",
                level="FOUNDRY_BLESSED",
            )

    def test_rejects_missing_provenance(self) -> None:
        with pytest.raises(ValueError, match="needs dataset"):
            build_claim(
                claim_id="IB-T",
                metric="m",
                value=1,
                unit="",
                baseline="b",
                dataset="",
                hardware="h",
                scope="s",
                artifact="a.json",
            )


COMMIT = "a" * 40
SHA = "b" * 64


def _valid_artifact() -> dict:
    return {
        "schema": SCHEMA_NAME,
        "kind": "runtime",
        "status": STATUS_SUCCESS,
        "git_commit": COMMIT,
        "timestamp_utc": "2026-09-20T00:00:00Z",
        "hardware": {"cpu_model": "test"},
        "software": {"python": "3.12"},
        "claim_scope": {"physics_claim": "NOT_FOUNDRY_CALIBRATED"},
        "reproducibility": {"command": "run.sh"},
        "measurement_source": {
            "commit": COMMIT,
            "commit_valid": True,
            "working_tree_dirty": False,
            "harness_sha256": SHA,
            "industrial_core_sha256": SHA,
        },
        "fixture": {"sha256": SHA, "bytes": 1234, "top_cell": "ibex_core"},
    }


class TestValidateArtifact:
    def test_valid(self) -> None:
        assert validate_artifact(_valid_artifact()) == []

    def test_missing_section(self) -> None:
        artifact = _valid_artifact()
        del artifact["hardware"]
        problems = validate_artifact(artifact)
        assert any("hardware" in p for p in problems)

    def test_wrong_schema(self) -> None:
        artifact = _valid_artifact()
        artifact["schema"] = "bogus"
        assert any("schema" in p for p in validate_artifact(artifact))

    def test_unknown_status(self) -> None:
        artifact = _valid_artifact()
        artifact["status"] = "SUCCESSFUL_MAYBE"
        assert any("status" in p for p in validate_artifact(artifact))

    def test_short_commit(self) -> None:
        artifact = _valid_artifact()
        artifact["git_commit"] = "x"
        assert any("git_commit" in p for p in validate_artifact(artifact))

    def test_placeholder_commit_rejected(self) -> None:
        artifact = _valid_artifact()
        artifact["git_commit"] = "UNKNOWN"
        assert any("git_commit" in p for p in validate_artifact(artifact))

    def test_dirty_measurement_source_rejected(self) -> None:
        artifact = _valid_artifact()
        artifact["measurement_source"]["working_tree_dirty"] = True
        assert any("working_tree_dirty" in p for p in validate_artifact(artifact))

    def test_non_finite_rejected(self) -> None:
        artifact = _valid_artifact()
        artifact["rows"] = [{"score": float("inf")}]
        assert any("non-finite" in p for p in validate_artifact(artifact))

    def test_missing_fixture_rejected(self) -> None:
        artifact = _valid_artifact()
        del artifact["fixture"]
        assert any("fixture" in p for p in validate_artifact(artifact))

    def test_bad_fixture_hash_rejected(self) -> None:
        artifact = _valid_artifact()
        artifact["fixture"]["sha256"] = "nope"
        assert any("fixture.sha256" in p for p in validate_artifact(artifact))

    def test_claim_scope_must_be_object(self) -> None:
        artifact = _valid_artifact()
        artifact["claim_scope"] = "nope"
        assert any("claim_scope" in p for p in validate_artifact(artifact))

    def test_load_artifact_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "a.json"
        path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
        assert load_artifact(path)["kind"] == "runtime"

    def test_load_artifact_invalid_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema": "bogus"}), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid industrial benchmark artifact"):
            load_artifact(path)


class TestCompareRuntime:
    def test_ratio(self) -> None:
        baseline = {"execution_wall_s": {"median": 4.0}}
        candidate = {"execution_wall_s": {"median": 2.0}}
        result = compare_runtime(
            baseline=baseline,
            candidate=candidate,
            baseline_name="dense",
            candidate_name="streaming",
        )
        assert result["speedup"] == 2.0
        assert result["baseline"] == "dense"


class TestEnvironment:
    def test_snapshot_sections(self) -> None:
        snap = environment_snapshot()
        assert snap["hardware"]["cpu_model"]
        assert snap["hardware"]["cpu_count"]
        assert snap["software"]["python"]
        assert "torch" in snap["software"]

    def test_git_commit_nonempty(self) -> None:
        from openlithohub.benchmark.industrial import is_full_commit

        assert is_full_commit(git_commit())
