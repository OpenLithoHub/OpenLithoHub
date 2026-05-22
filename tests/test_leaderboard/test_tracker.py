"""Tests for leaderboard tracker (submit/retrieve)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openlithohub.leaderboard.schema import BenchmarkResult, MaskTopology, ProcessNode
from openlithohub.leaderboard.tracker import LeaderboardStore, get_leaderboard, submit_result


@pytest.fixture
def tmp_store(tmp_path: Path) -> LeaderboardStore:
    return LeaderboardStore(tmp_path / "test_leaderboard.json")


@pytest.fixture
def sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        model_name="test-model",
        dataset="lithobench",
        process_node=ProcessNode.N7,
        mask_topology=MaskTopology.MANHATTAN,
        epe_mean_nm=2.0,
        epe_max_nm=6.5,
        l2_error_pixels=42.0,
    )


def test_submit_creates_file(tmp_store: LeaderboardStore, sample_result: BenchmarkResult) -> None:
    sub_id = tmp_store.submit(sample_result)
    assert sub_id.startswith("test-model-")
    assert tmp_store.path.exists()


def test_submit_and_retrieve(tmp_store: LeaderboardStore, sample_result: BenchmarkResult) -> None:
    tmp_store.submit(sample_result)
    results = tmp_store.query()
    assert len(results) == 1
    assert results[0].model_name == "test-model"
    assert results[0].epe_mean_nm == 2.0


def test_multiple_submissions_sorted_by_l2_error(tmp_store: LeaderboardStore) -> None:
    """Leaderboard ranks by ``l2_error_pixels`` (Neural-ILT printability),
    not mask-level EPE — the old ``epe_mean_nm`` key let an Identity model
    score 0 and tie the table for first place. See
    ``leaderboard/tracker.py::_ranking_key``.
    """
    for l2, name in [(50.0, "bad"), (10.0, "best"), (30.0, "mid")]:
        r = BenchmarkResult(
            model_name=name,
            dataset="lithobench",
            process_node=ProcessNode.N7,
            mask_topology=MaskTopology.MANHATTAN,
            epe_mean_nm=2.0,
            epe_max_nm=4.0,
            l2_error_pixels=l2,
        )
        tmp_store.submit(r)

    results = tmp_store.query()
    assert [r.model_name for r in results] == ["best", "mid", "bad"]


def test_filter_by_dataset(tmp_store: LeaderboardStore) -> None:
    for ds in ["lithobench", "lithosim", "lithobench"]:
        r = BenchmarkResult(
            model_name=f"model-{ds}",
            dataset=ds,
            process_node=ProcessNode.N7,
            mask_topology=MaskTopology.MANHATTAN,
            epe_mean_nm=2.0,
            epe_max_nm=5.0,
            l2_error_pixels=10.0,
        )
        tmp_store.submit(r)

    results = tmp_store.query(dataset="lithosim")
    assert len(results) == 1
    assert results[0].dataset == "lithosim"


def test_filter_by_process_node(tmp_store: LeaderboardStore) -> None:
    for node in [ProcessNode.N7, ProcessNode.N3_EUV, ProcessNode.N7]:
        r = BenchmarkResult(
            model_name="m",
            dataset="lithobench",
            process_node=node,
            mask_topology=MaskTopology.MANHATTAN,
            epe_mean_nm=2.0,
            epe_max_nm=5.0,
            l2_error_pixels=10.0,
        )
        tmp_store.submit(r)

    results = tmp_store.query(process_node="3nm-euv")
    assert len(results) == 1


def test_empty_leaderboard(tmp_store: LeaderboardStore) -> None:
    results = tmp_store.query()
    assert results == []


def test_leaderboard_file_is_valid_json(
    tmp_store: LeaderboardStore, sample_result: BenchmarkResult
) -> None:
    tmp_store.submit(sample_result)
    data = json.loads(tmp_store.path.read_text())
    assert "entries" in data
    assert len(data["entries"]) == 1
    assert data["schema_version"] == 3


def test_legacy_unversioned_file_still_loads(
    tmp_store: LeaderboardStore, sample_result: BenchmarkResult
) -> None:
    """Files written by older versions had no schema_version key."""
    legacy_entry = sample_result.model_dump(mode="json")
    legacy_entry["submission_id"] = "legacy-deadbeef"
    tmp_store.path.parent.mkdir(parents=True, exist_ok=True)
    tmp_store.path.write_text(json.dumps({"entries": [legacy_entry]}))

    results = tmp_store.query()
    assert len(results) == 1
    assert results[0].model_name == sample_result.model_name


def test_future_schema_version_rejected(
    tmp_store: LeaderboardStore, sample_result: BenchmarkResult
) -> None:
    tmp_store.path.parent.mkdir(parents=True, exist_ok=True)
    tmp_store.path.write_text(json.dumps({"schema_version": 999, "entries": []}))
    with pytest.raises(ValueError, match="newer schema"):
        tmp_store.query()


def test_v2_l2_error_nulled_on_load_without_num_samples(tmp_store: LeaderboardStore) -> None:
    """v2-and-older entries had l2_error_pixels as a cross-sample SUM with no
    num_samples recorded. The v3 migration cannot reliably divide back into a
    mean, so it nulls out the field rather than corrupting the ranking with
    apples-to-oranges scalars."""
    legacy_entry = {
        "model_name": "legacy-summed",
        "dataset": "lithobench",
        "process_node": "7nm",
        "mask_topology": "curvilinear",
        "epe_mean_nm": 1.5,
        "epe_max_nm": 3.0,
        "l2_error_pixels": 12345.0,
        "l2_error_nm2": 67890.0,
        "submission_id": "legacy-deadbeef",
    }
    tmp_store.path.parent.mkdir(parents=True, exist_ok=True)
    tmp_store.path.write_text(json.dumps({"schema_version": 2, "entries": [legacy_entry]}))

    results = tmp_store.query()
    assert len(results) == 1
    assert results[0].l2_error_pixels is None
    assert results[0].l2_error_nm2 is None


def test_v3_l2_error_preserved_when_num_samples_present(tmp_store: LeaderboardStore) -> None:
    """v3 entries record num_samples alongside the (mean) l2_error_pixels;
    those entries must round-trip through load unchanged."""
    entry = {
        "model_name": "modern",
        "dataset": "lithobench",
        "process_node": "7nm",
        "mask_topology": "curvilinear",
        "epe_mean_nm": 1.5,
        "epe_max_nm": 3.0,
        "l2_error_pixels": 299.875,
        "l2_error_nm2": 19192.0,
        "num_samples": 8,
        "submission_id": "modern-cafebabe",
    }
    tmp_store.path.parent.mkdir(parents=True, exist_ok=True)
    tmp_store.path.write_text(json.dumps({"schema_version": 3, "entries": [entry]}))

    results = tmp_store.query()
    assert len(results) == 1
    assert results[0].l2_error_pixels == 299.875
    assert results[0].num_samples == 8


def test_module_level_functions(tmp_path: Path, sample_result: BenchmarkResult) -> None:
    store = LeaderboardStore(tmp_path / "lb.json")
    sub_id = submit_result(sample_result, store=store)
    assert sub_id
    results = get_leaderboard(store=store)
    assert len(results) == 1


def test_submission_id_format(tmp_store: LeaderboardStore, sample_result: BenchmarkResult) -> None:
    sub_id = tmp_store.submit(sample_result)
    parts = sub_id.rsplit("-", 1)
    assert len(parts) == 2
    assert parts[0] == "test-model"
    assert len(parts[1]) == 8


def test_submission_without_l2_error_is_rejected(tmp_store: LeaderboardStore) -> None:
    """Forward-sim gate: a submission missing l2_error_pixels (i.e. one that
    skipped forward simulation) is refused at submit time. See
    ``_require_forward_simulation``.
    """
    bare_mask_only = BenchmarkResult(
        model_name="cheater",
        dataset="lithobench",
        process_node=ProcessNode.N7,
        mask_topology=MaskTopology.MANHATTAN,
        epe_mean_nm=0.0,
        epe_max_nm=0.0,
        # l2_error_pixels deliberately omitted — proves no forward-sim ran.
    )
    with pytest.raises(ValueError, match="l2_error_pixels is required"):
        tmp_store.submit(bare_mask_only)


def test_combined_filters(tmp_store: LeaderboardStore) -> None:
    entries = [
        ("a", "lithobench", ProcessNode.N7),
        ("b", "lithosim", ProcessNode.N7),
        ("c", "lithobench", ProcessNode.N3_EUV),
        ("d", "lithosim", ProcessNode.N3_EUV),
    ]
    for name, ds, node in entries:
        r = BenchmarkResult(
            model_name=name,
            dataset=ds,
            process_node=node,
            mask_topology=MaskTopology.MANHATTAN,
            epe_mean_nm=2.0,
            epe_max_nm=5.0,
            l2_error_pixels=10.0,
        )
        tmp_store.submit(r)

    results = tmp_store.query(dataset="lithobench", process_node="3nm-euv")
    assert len(results) == 1
    assert results[0].model_name == "c"
