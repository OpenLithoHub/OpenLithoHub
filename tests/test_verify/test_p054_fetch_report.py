"""P-054 PR-5E5 — canonical fetch report: writer, validator, provenance handoff.

The fetch report is the durable provenance of the effective transport URL.
It is written by the fetch only after host/size/hash validation succeeds,
validated by the producer against the verified artifact, and consumed
into the replay report — so the effective URL survives the fetch and
lands in the canonical evidence.
"""

import json

import pytest
from scripts.fetch_proof_artifacts import write_fetch_report
from scripts.replay_p054_artifact import validate_fetch_report

SHA = "a" * 64


def _report(**overrides) -> dict:
    report = {
        "schema": "P054.fetch-report.v1",
        "profile": "p054-arf37",
        "effective_url": "https://zenodo.org/records/22843141/files/p054.zip",
        "artifact_sha256": SHA,
        "artifact_bytes": 4096,
    }
    report.update(overrides)
    return report


def test_write_fetch_report_is_atomic_and_complete(tmp_path):
    out = tmp_path / "p054-arf37-frozen-artifact.fetch-report.json"
    write_fetch_report(
        out,
        profile="p054-arf37",
        effective_url="https://zenodo.org/records/22843141/files/p054.zip",
        artifact_sha256=SHA,
        artifact_bytes=4096,
    )
    written = json.loads(out.read_text())
    assert written["schema"] == "P054.fetch-report.v1"
    assert written["profile"] == "p054-arf37"
    assert written["effective_url"].startswith("https://zenodo.org/")
    assert written["artifact_sha256"] == SHA
    assert written["artifact_bytes"] == 4096
    assert not (tmp_path / "p054-arf37-frozen-artifact.fetch-report.json.part").exists()


def test_validator_accepts_matching_report():
    report = _report()
    assert (
        validate_fetch_report(report, artifact_sha256=SHA, artifact_bytes=4096)
        == report["effective_url"]
    )


def test_validator_rejects_sha_mismatch():
    with pytest.raises(ValueError, match="does not match the verified artifact"):
        validate_fetch_report(
            _report(artifact_sha256="f" * 64), artifact_sha256=SHA, artifact_bytes=4096
        )


def test_validator_rejects_bytes_mismatch():
    with pytest.raises(ValueError, match="artifact_bytes does not match"):
        validate_fetch_report(_report(artifact_bytes=1), artifact_sha256=SHA, artifact_bytes=4096)


def test_validator_rejects_empty_or_off_https_url():
    with pytest.raises(ValueError, match="https URL"):
        validate_fetch_report(_report(effective_url=""), artifact_sha256=SHA, artifact_bytes=4096)
    with pytest.raises(ValueError, match="https URL"):
        validate_fetch_report(
            _report(effective_url="http://zenodo.org/files/p054.zip"),
            artifact_sha256=SHA,
            artifact_bytes=4096,
        )


def test_validator_rejects_wrong_schema(tmp_path):
    with pytest.raises(ValueError, match="fetch report schema"):
        validate_fetch_report(
            _report(schema="P054.fetch-report.v99"),
            artifact_sha256=SHA,
            artifact_bytes=4096,
        )


def test_producer_report_carries_effective_url():
    # the wiring check: the producer's report template includes the fetched
    # URL — verified by importing the constants/flow end to end is
    # artifact-gated, so assert the template source instead.
    import inspect

    from scripts import replay_p054_artifact as producer

    source = inspect.getsource(producer)
    assert "validate_fetch_report" in source
    assert '"effective_url": effective_url' in source


# ---------------------------------------------------------------------------
# PR-5E6 — provenance binding: profile + Zenodo host
# ---------------------------------------------------------------------------


def test_l6_profile_mismatch_rejected():
    with pytest.raises(ValueError, match="!= expected"):
        validate_fetch_report(
            _report(),
            artifact_sha256=SHA,
            artifact_bytes=4096,
            expected_profile="other-profile",
        )


def test_l7_off_zenodo_host_rejected_for_zenodo_provenance():
    with pytest.raises(ValueError, match="outside the expected provenance"):
        validate_fetch_report(
            _report(effective_url="https://mirror.example.org/files/p054.zip"),
            artifact_sha256=SHA,
            artifact_bytes=4096,
        )


def test_l8_www_zenodo_host_accepted():
    url = validate_fetch_report(
        _report(effective_url="https://www.zenodo.org/records/22843141/files/p054.zip"),
        artifact_sha256=SHA,
        artifact_bytes=4096,
    )
    assert url.startswith("https://www.zenodo.org/")
