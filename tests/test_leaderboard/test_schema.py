"""Tests for leaderboard schema."""

from datetime import datetime

from openlithohub.leaderboard.schema import (
    BenchmarkResult,
    MaskTopology,
    ProcessNode,
)


def test_benchmark_result_creation():
    result = BenchmarkResult(
        model_name="test-model",
        dataset="lithobench",
        process_node=ProcessNode.N45,
        mask_topology=MaskTopology.MANHATTAN,
        epe_mean_nm=2.5,
        epe_max_nm=8.0,
    )
    assert result.model_name == "test-model"
    assert result.process_node == ProcessNode.N45
    assert result.epe_mean_nm == 2.5
    assert isinstance(result.submitted_at, datetime)


def test_benchmark_result_with_all_metrics():
    result = BenchmarkResult(
        model_name="curvy-ilt",
        dataset="lithosim",
        process_node=ProcessNode.N3_EUV,
        mask_topology=MaskTopology.CURVILINEAR,
        epe_mean_nm=1.2,
        epe_max_nm=3.5,
        pvband_nm=4.0,
        mrc_violation_rate=0.02,
        drc_pass=True,
        shot_count=150000,
        stochastic_robustness=0.95,
        paper_url="https://arxiv.org/abs/2024.xxxxx",
    )
    assert result.stochastic_robustness == 0.95
    assert result.drc_pass is True


def test_process_node_values():
    assert ProcessNode.N3_EUV.value == "3nm-euv"
    assert ProcessNode.N2_EUV.value == "2nm-euv"


def test_mask_topology_values():
    assert MaskTopology.MANHATTAN.value == "manhattan"
    assert MaskTopology.CURVILINEAR.value == "curvilinear"
