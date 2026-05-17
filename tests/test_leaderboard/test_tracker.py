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


def test_multiple_submissions_sorted_by_epe(tmp_store: LeaderboardStore) -> None:
    for epe, name in [(5.0, "bad"), (1.0, "best"), (3.0, "mid")]:
        r = BenchmarkResult(
            model_name=name,
            dataset="lithobench",
            process_node=ProcessNode.N7,
            mask_topology=MaskTopology.MANHATTAN,
            epe_mean_nm=epe,
            epe_max_nm=epe * 2,
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
        )
        tmp_store.submit(r)

    results = tmp_store.query(dataset="lithobench", process_node="3nm-euv")
    assert len(results) == 1
    assert results[0].model_name == "c"
