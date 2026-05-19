"""Hackathon manifest loader.

Reads the YAML contract that pins the hackathon: tag, sample count,
gates, target. Used by the CLI (`openlithohub hackathon info`), the
leaderboard CI (verify-against-manifest step), and the website build
script that sources the page from this single artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "hackathon" / "2026q3.yaml"


@dataclass(frozen=True)
class HackathonManifest:
    """Frozen view of the hackathon contract."""

    track: str
    status: str
    process_node: str
    dataset_tag: str
    dataset_commit_sha: str | None
    dataset_sample_count: int | None
    pixel_nm: float
    mrc_violation_rate_max: float
    drc_pass_required: bool
    target_epe_mean_nm: float | None
    ranking_primary: str
    ranking_tiebreakers: tuple[str, ...]

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def has_calibrated_target(self) -> bool:
        return self.target_epe_mean_nm is not None


def _opt(value: Any) -> Any:
    """Map the YAML literal ``TBD`` to None."""
    if isinstance(value, str) and value.strip() == "TBD":
        return None
    return value


def load_manifest(path: Path | None = None) -> HackathonManifest:
    """Load and validate the hackathon manifest YAML."""
    src = Path(path) if path is not None else DEFAULT_MANIFEST
    if not src.exists():
        raise FileNotFoundError(f"Hackathon manifest not found: {src}")
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    dataset = raw["dataset"]
    gates = raw["gates"]
    target = raw["target"]
    ranking = raw["ranking"]

    sample_count = _opt(dataset.get("sample_count"))
    if sample_count is not None:
        sample_count = int(sample_count)
    target_epe = _opt(target.get("target_epe_mean_nm"))
    if target_epe is not None:
        target_epe = float(target_epe)

    return HackathonManifest(
        track=str(raw["track"]),
        status=str(raw["status"]),
        process_node=str(raw["process_node"]),
        dataset_tag=str(dataset["tag"]),
        dataset_commit_sha=_opt(dataset.get("commit_sha")),
        dataset_sample_count=sample_count,
        pixel_nm=float(dataset.get("pixel_nm", 1.0)),
        mrc_violation_rate_max=float(gates["mrc_violation_rate_max"]),
        drc_pass_required=bool(gates["drc_pass_required"]),
        target_epe_mean_nm=target_epe,
        ranking_primary=str(ranking["primary"]),
        ranking_tiebreakers=tuple(ranking.get("tiebreakers", [])),
    )
