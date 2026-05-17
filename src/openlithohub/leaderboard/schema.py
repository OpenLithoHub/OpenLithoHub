"""Pydantic schemas for leaderboard entries and submissions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class ProcessNode(str, Enum):
    """Supported process technology nodes."""

    N45 = "45nm"
    N28 = "28nm"
    N7 = "7nm"
    N5 = "5nm"
    N3_EUV = "3nm-euv"
    N2_EUV = "2nm-euv"


class MaskTopology(str, Enum):
    """Mask shape classification."""

    MANHATTAN = "manhattan"
    CURVILINEAR = "curvilinear"


class BenchmarkResult(BaseModel):
    """A single benchmark submission for the leaderboard."""

    model_name: str = Field(..., description="Name of the evaluated model")
    dataset: str = Field(..., description="Dataset used (lithobench/lithosim)")
    process_node: ProcessNode
    mask_topology: MaskTopology

    epe_mean_nm: float = Field(..., ge=0, description="Mean EPE in nanometers")
    epe_max_nm: float = Field(..., ge=0)
    pvband_nm: float | None = Field(None, ge=0)
    mrc_violation_rate: float | None = Field(None, ge=0, le=1)
    drc_pass: bool | None = None
    shot_count: int | None = Field(None, ge=0)
    stochastic_robustness: float | None = Field(None, ge=0, le=1)

    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    paper_url: str | None = None
    code_url: str | None = None
    notes: str | None = None
