"""Tests for the hackathon manifest loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from openlithohub.hackathon import HackathonManifest, load_manifest
from openlithohub.hackathon.manifest import DEFAULT_MANIFEST


def test_default_manifest_loads() -> None:
    m = load_manifest()
    assert isinstance(m, HackathonManifest)
    assert m.track == "hackathon-2026q3"
    assert m.dataset_tag == "hackathon-2026q3-test-v1"
    assert m.mrc_violation_rate_max == 0.0
    assert m.drc_pass_required is True
    assert m.ranking_primary == "epe_mean_nm"
    assert "epe_max_nm" in m.ranking_tiebreakers


def test_charter_status_has_no_calibrated_target() -> None:
    """While `status: charter`, target EPE and dataset SHA stay TBD."""
    m = load_manifest()
    assert m.status == "charter"
    assert m.is_open is False
    assert m.has_calibrated_target is False
    assert m.target_epe_mean_nm is None
    assert m.dataset_commit_sha is None
    assert m.dataset_sample_count is None


def test_track_matches_leaderboard_enum() -> None:
    """Manifest track must be a registered LeaderboardTrack value."""
    from openlithohub.leaderboard.schema import LeaderboardTrack

    m = load_manifest()
    LeaderboardTrack(m.track)


def test_load_from_explicit_path(tmp_path: Path) -> None:
    src = DEFAULT_MANIFEST.read_text(encoding="utf-8")
    target = tmp_path / "copy.yaml"
    target.write_text(src, encoding="utf-8")
    m = load_manifest(target)
    assert m.track == "hackathon-2026q3"


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.yaml")


def test_filled_target_parses_as_float(tmp_path: Path) -> None:
    """When organisers fill in target_epe_mean_nm, it parses as float."""
    src = DEFAULT_MANIFEST.read_text(encoding="utf-8")
    src = src.replace(
        "target_epe_mean_nm: TBD",
        "target_epe_mean_nm: 2.4",
    )
    target = tmp_path / "open.yaml"
    target.write_text(src, encoding="utf-8")
    m = load_manifest(target)
    assert m.target_epe_mean_nm == pytest.approx(2.4)
    assert m.has_calibrated_target is True
