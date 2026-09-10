"""End-to-end replay provenance and enclosure acceptance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolchainProvenance:
    git_commit: str
    python_version: str
    numpy_version: str
    torch_version: str
    mpmath_version: str
    klayout_version: str
    platform: str
    interval_rounding_model: str
    compiler: str | None = None
    blas: str | None = None


@dataclass(frozen=True)
class HaloDecisionProvenance:
    core_size_px: int
    halo_px: int
    selection_method: str
    optical_tail_upper: float | None
    required_error_budget: float
    kernel_snapshot_hash: str
    source_snapshot_hash: str
    boundary_model: str

    def __post_init__(self) -> None:
        if self.core_size_px <= 0 or self.halo_px < 0:
            raise ValueError("invalid core/halo geometry")
        if self.required_error_budget < 0:
            raise ValueError("negative error budget")
        if self.optical_tail_upper is not None and self.optical_tail_upper < 0:
            raise ValueError("negative optical tail bound")

    @property
    def theorem_ready(self) -> bool:
        return (
            self.optical_tail_upper is not None
            and self.optical_tail_upper <= self.required_error_budget
        )


@dataclass(frozen=True)
class ReplayProvenance:
    toolchain: ToolchainProvenance
    halo: HaloDecisionProvenance
    layout_hash: str
    layout_format: str
    layer_datatype: str
    dbu_nm: str
    quantization_policy: str
    global_origin_xy: tuple[int, int]
    physical_identity_scheme: str
    reducer_hash: str
    assumptions: tuple[str, ...]
    guarantees: tuple[str, ...]
    does_not_prove: tuple[str, ...]

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class Interval1D:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("invalid interval")


def one_way_reference_enclosed(
    *,
    certified: Interval1D,
    reference: Interval1D,
    certified_bridge_error: float,
) -> bool:
    """Strong practical acceptance criterion.

    If an independently certified bridge error eps says the physical target
    lies within reference +/- eps, then the theorem interval must contain that
    expanded reference interval:

        reference ⊕ [-eps, eps] ⊆ certified.

    This is deliberately one-way.  Mutual inclusion is unnecessary.
    """
    if certified_bridge_error < 0:
        raise ValueError("bridge error must be non-negative")
    return (
        certified.lo <= reference.lo - certified_bridge_error
        and reference.hi + certified_bridge_error <= certified.hi
    )
