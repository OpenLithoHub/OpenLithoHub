"""Benchmark metrics for computational lithography evaluation."""

from openlithohub.benchmark.metrics.epe import compute_epe, compute_wafer_epe
from openlithohub.benchmark.metrics.euv_3d import (
    Mask3DParams,
    apply_3d_shadow,
    compute_3d_mask_residual,
)
from openlithohub.benchmark.metrics.hotspot import compute_hotspot_detection
from openlithohub.benchmark.metrics.l2_error import compute_l2_error
from openlithohub.benchmark.metrics.manhattanization import (
    curvilinear_to_manhattan,
    manhattanization_degradation,
)
from openlithohub.benchmark.metrics.monte_carlo import (
    MonteCarloFailureResult,
    monte_carlo_failure_probability,
)
from openlithohub.benchmark.metrics.mrc_loss import curvilinear_mrc_loss
from openlithohub.benchmark.metrics.pvband import compute_pvband
from openlithohub.benchmark.metrics.shot_count import estimate_shot_count
from openlithohub.benchmark.metrics.sraf import sraf_print_penalty
from openlithohub.benchmark.metrics.stochastic import (
    StochasticDefectRates,
    compute_stochastic_defect_classes,
    compute_stochastic_robustness,
)
from openlithohub.benchmark.metrics.tiling_consistency import (
    cross_tile_sraf_consistency,
    tile_boundary_consistency,
)

__all__ = [
    "Mask3DParams",
    "MonteCarloFailureResult",
    "StochasticDefectRates",
    "apply_3d_shadow",
    "compute_3d_mask_residual",
    "compute_epe",
    "compute_hotspot_detection",
    "compute_l2_error",
    "compute_pvband",
    "compute_stochastic_defect_classes",
    "compute_stochastic_robustness",
    "compute_wafer_epe",
    "cross_tile_sraf_consistency",
    "curvilinear_mrc_loss",
    "curvilinear_to_manhattan",
    "estimate_shot_count",
    "manhattanization_degradation",
    "monte_carlo_failure_probability",
    "sraf_print_penalty",
    "tile_boundary_consistency",
]
