"""Pydantic schemas for leaderboard entries and submissions."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProcessNode(str, Enum):
    """Supported process technology nodes."""

    N45 = "45nm"
    N28 = "28nm"
    N7 = "7nm"
    N5_EUV = "5nm-euv"
    N3_EUV = "3nm-euv"
    N2_EUV = "2nm-euv"


class MaskTopology(str, Enum):
    """Mask shape classification."""

    MANHATTAN = "manhattan"
    CURVILINEAR = "curvilinear"


class LeaderboardTrack(str, Enum):
    """Leaderboard track. Default is the open ongoing competition.

    Hackathon tracks scope a fixed dataset + node + frozen test split for
    a bounded period. Entries marked with a hackathon track are
    displayed in their own ranked table on the website and never mix
    with the open leaderboard. See ``docs/hackathon.md``.
    """

    OPEN = "open"
    HACKATHON_2026Q3 = "hackathon-2026q3"


class BenchmarkResult(BaseModel):
    """A single benchmark submission for the leaderboard.

    The leaderboard ingests this schema from community pull requests via the
    ``auto-leaderboard`` workflow. The schema is the only firewall between
    PR-supplied YAML and the canonical store, so it forbids extra fields,
    bounds string lengths, and validates URL fields.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(
        ..., min_length=1, max_length=120, description="Name of the evaluated model"
    )
    dataset: str = Field(
        ..., min_length=1, max_length=120, description="Dataset used (lithobench/lithosim)"
    )
    process_node: ProcessNode
    mask_topology: MaskTopology
    track: LeaderboardTrack = Field(
        LeaderboardTrack.OPEN,
        description="Leaderboard track (open or a specific hackathon round).",
    )

    epe_mean_nm: float = Field(..., ge=0, description="Mean EPE in nanometers")
    epe_max_nm: float = Field(..., ge=0)
    pvband_mean_nm: float | None = Field(None, ge=0, description="Mean PV band width (nm)")
    pvband_max_nm: float | None = Field(None, ge=0, description="Max PV band width (nm)")
    mrc_violation_rate: float | None = Field(None, ge=0, le=1)
    drc_pass: bool | None = None
    shot_count: int | None = Field(None, ge=0)
    stochastic_robustness: float | None = Field(None, ge=0, le=1)

    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    submission_id: str | None = Field(
        None, max_length=64, description="Auto-assigned submission ID (read-only)."
    )
    paper_url: str | None = Field(None, max_length=2048)
    code_url: str | None = Field(None, max_length=2048)
    notes: str | None = Field(None, max_length=2000)

    @field_validator("paper_url", "code_url")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (v.startswith("https://") or v.startswith("http://")):
            raise ValueError("URL must start with http:// or https://")
        return v
