"""`Report` — flat view over the existing metric / compliance outputs.

No new math. The flat fields are projections of fields already computed
by ``benchmark.metrics`` and ``benchmark.compliance``; the raw underlying
results are kept on the dataclass for power users who want every field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openlithohub.benchmark.compliance.drc import DRCResult
    from openlithohub.benchmark.compliance.mrc import CurvilinearMRCResult, MRCResult
    from openlithohub.benchmark.metrics.epe import EPEResult


@dataclass(frozen=True)
class Report:
    """Aggregated mask-quality report produced by ``LitheEngine.evaluate``."""

    epe_mean_nm: float
    epe_max_nm: float
    epe_std_nm: float
    pvband_mean_nm: float
    pvband_max_nm: float

    drc_violations: int
    drc_passed: bool
    mrc_violations: int
    mrc_passed: bool

    shot_count: int
    estimated_write_time_s: float

    model_name: str
    pixel_size_nm: float

    raw_epe: EPEResult
    raw_drc: DRCResult
    raw_mrc: MRCResult
    raw_pvband: dict[str, float]
    raw_shot_count: dict[str, int | float]
    raw_curvilinear_mrc: CurvilinearMRCResult | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view. Raw payloads are recursively flattened."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Report(model={self.model_name!r} "
            f"epe_mean={self.epe_mean_nm:.3f} nm "
            f"pvband_mean={self.pvband_mean_nm:.3f} nm "
            f"drc={self.drc_violations} mrc={self.mrc_violations} "
            f"shots={self.shot_count})"
        )
