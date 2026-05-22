"""Tests for leaderboard schema."""

from datetime import datetime

import pytest
from pydantic import ValidationError

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
        pvband_mean_nm=4.0,
        pvband_max_nm=6.5,
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


def _valid_payload(**overrides):
    base = dict(
        model_name="test-model",
        dataset="lithobench",
        process_node="45nm",
        mask_topology="manhattan",
        epe_mean_nm=1.0,
        epe_max_nm=2.0,
    )
    base.update(overrides)
    return base


def test_extra_fields_are_rejected():
    """The schema is the firewall in front of `auto-leaderboard.yml` —
    submitters must not be able to inject undeclared keys that round-trip
    into the canonical store."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        BenchmarkResult.model_validate(_valid_payload(injected_field="payload"))


def test_url_must_be_http_or_https():
    with pytest.raises(ValidationError, match="http:// or https://"):
        BenchmarkResult.model_validate(_valid_payload(paper_url="javascript:alert(1)"))
    with pytest.raises(ValidationError, match="http:// or https://"):
        BenchmarkResult.model_validate(_valid_payload(code_url="file:///etc/passwd"))


def test_url_rejects_credentials_and_whitespace():
    with pytest.raises(ValidationError, match="user:password"):
        BenchmarkResult.model_validate(
            _valid_payload(paper_url="https://user:pass@evil.example.com/")
        )
    with pytest.raises(ValidationError, match="whitespace"):
        BenchmarkResult.model_validate(_valid_payload(code_url="https://example.com/ foo"))
    with pytest.raises(ValidationError, match="host"):
        BenchmarkResult.model_validate(_valid_payload(paper_url="https://"))


def test_submission_id_charset_enforced():
    with pytest.raises(ValidationError, match="alphanumeric"):
        BenchmarkResult.model_validate(_valid_payload(submission_id="../../../etc/passwd"))
    # Allowed charset round-trips.
    result = BenchmarkResult.model_validate(_valid_payload(submission_id="abc-123_DEF"))
    assert result.submission_id == "abc-123_DEF"


def test_string_length_bounded():
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(_valid_payload(model_name="x" * 200))
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(_valid_payload(notes="x" * 5000))


def test_negative_metrics_rejected():
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(_valid_payload(epe_mean_nm=-1.0))


def test_unknown_enum_rejected():
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(_valid_payload(process_node="999nm"))
