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
        ...,
        min_length=1,
        max_length=120,
        description=("Dataset used (lithobench / lithosim / asap7 / freepdk45 / orfs)."),
    )
    process_node: ProcessNode
    mask_topology: MaskTopology
    track: LeaderboardTrack = Field(
        LeaderboardTrack.OPEN,
        description="Leaderboard track (open or a specific hackathon round).",
    )

    # Mask-level EPE — kept as a sanity baseline. NOT canonical: an Identity
    # model scores 0 here by construction. The leaderboard ranks on
    # ``l2_error_pixels`` (Neural-ILT contract), with ``pvband_mean_nm`` as
    # the secondary key. See ``benchmark/metrics/l2_error.py``.
    epe_mean_nm: float = Field(
        ..., ge=0, description="Mean mask-level EPE in nm (sanity, not the ranking key)."
    )
    epe_max_nm: float = Field(..., ge=0, description="Max mask-level EPE in nm.")

    # Wafer-level metrics — printability after forward simulation. These are
    # the physically meaningful figures and feed the leaderboard ranking.
    epe_wafer_mean_nm: float | None = Field(
        None, ge=0, description="Mean wafer-level EPE in nm (post forward-sim)."
    )
    epe_wafer_max_nm: float | None = Field(
        None, ge=0, description="Max wafer-level EPE in nm (post forward-sim)."
    )
    l2_error_pixels: float | None = Field(
        None,
        ge=0,
        description=(
            "Neural-ILT canonical printability scalar: |wafer - target| summed "
            "over pixels (technically L1; named per the Neural-ILT paper). "
            "Primary leaderboard ranking key."
        ),
    )
    l2_error_nm2: float | None = Field(
        None, ge=0, description="``l2_error_pixels`` converted to nm² area."
    )

    pvband_mean_nm: float | None = Field(None, ge=0, description="Mean PV band width (nm)")
    pvband_max_nm: float | None = Field(None, ge=0, description="Max PV band width (nm)")
    mrc_violation_rate: float | None = Field(None, ge=0, le=1)
    drc_pass: bool | None = None
    shot_count: int | None = Field(None, ge=0)
    stochastic_robustness: float | None = Field(None, ge=0, le=1)
    resist_diffusion_nm: float | None = Field(
        None,
        ge=0,
        description=(
            "Acid diffusion length used during evaluation. Must be 0.0 (or "
            "None) for leaderboard-eligible submissions; positive values are "
            "non-comparable with the canonical CTR baseline."
        ),
    )

    # Number of samples behind the aggregated metrics. Recorded so future
    # migrations can detect (and if needed re-normalize) entries written
    # under different aggregation conventions — see schema v3 migration.
    num_samples: int | None = Field(None, ge=0)

    # Per-metric counts of samples whose value was non-finite (``nan`` /
    # ``inf``) and was therefore excluded from the aggregate. Surfaces
    # eval-time dataset noise on the leaderboard so a quietly-broken run
    # doesn't sit next to a clean one with no indication. Keys match the
    # aggregated metric names (e.g. ``epe_wafer_mean_nm``).
    dropped_nonfinite: dict[str, int] | None = Field(default=None)

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
        # Strict URL validation: parse and require https/http scheme, a
        # non-empty network location, no embedded user:password, and no
        # whitespace anywhere. The previous prefix-only check accepted
        # malformed strings like "http://", "http:// foo", or
        # "https://user:pass@evil/" — fine as text, but unusable as a
        # link and a phishing vector when surfaced in the leaderboard UI.
        from urllib.parse import urlparse

        if any(ch.isspace() for ch in v):
            raise ValueError("URL must not contain whitespace")
        try:
            parsed = urlparse(v)
        except ValueError as exc:
            raise ValueError(f"Invalid URL: {exc}") from exc
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must use http:// or https:// scheme")
        if not parsed.netloc:
            raise ValueError("URL must include a host")
        if "@" in parsed.netloc:
            raise ValueError("URL must not include user:password credentials")
        hostname = parsed.hostname
        if hostname is None or "." not in hostname:
            raise ValueError("URL hostname must contain at least one '.'")
        return v

    @field_validator("submission_id")
    @classmethod
    def _validate_submission_id(cls, v: str | None) -> str | None:
        # Submission IDs are surfaced in URLs and filesystem paths; constrain
        # to a safe charset (alnum + dash + underscore) so a hostile ID can't
        # path-traverse out of the submissions/ directory or break URL routing.
        if v is None:
            return v
        if not v:
            raise ValueError("submission_id must not be empty")
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                "submission_id must contain only alphanumeric characters, dashes, and underscores"
            )
        return v
