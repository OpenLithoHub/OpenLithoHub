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
from openlithohub.benchmark.metrics.stochastic_loss import (
    StochasticAwareLoss,
    StochasticProcessWindow,
    StochasticProcessWindowResult,
    cvar_loss,
    differentiable_edge_error,
    differentiable_lcdu,
    quantile_loss,
)
from openlithohub.benchmark.metrics.coverage_gate import (
    CoverageResult,
    ProcessWindowPlotter,
    StochasticAcceptanceGate,
    StochasticSampler,
    ThroughFocusCoverageCalibrator,
)
from openlithohub.benchmark.metrics.tiling_consistency import (
    cross_tile_contour_residual,
    cross_tile_epe_residual,
    cross_tile_sraf_consistency,
    schwarz_vs_naive_comparison,
    sweep_overlap_convergence,
    tile_boundary_consistency,
)

__all__ = [
    "CoverageResult",
    "Mask3DParams",
    "MonteCarloFailureResult",
    "ProcessWindowPlotter",
    "StochasticAcceptanceGate",
    "StochasticAwareLoss",
    "StochasticDefectRates",
    "StochasticProcessWindow",
    "StochasticProcessWindowResult",
    "StochasticSampler",
    "ThroughFocusCoverageCalibrator",
    "apply_3d_shadow",
    "cvar_loss",
    "compute_3d_mask_residual",
    "compute_epe",
    "compute_hotspot_detection",
    "compute_l2_error",
    "compute_pvband",
    "compute_stochastic_defect_classes",
    "compute_stochastic_robustness",
    "compute_wafer_epe",
    "cross_tile_contour_residual",
    "cross_tile_epe_residual",
    "cross_tile_sraf_consistency",
    "curvilinear_mrc_loss",
    "curvilinear_to_manhattan",
    "differentiable_edge_error",
    "differentiable_lcdu",
    "estimate_shot_count",
    "manhattanization_degradation",
    "monte_carlo_failure_probability",
    "quantile_loss",
    "schwarz_vs_naive_comparison",
    "sraf_print_penalty",
    "sweep_overlap_convergence",
    "tile_boundary_consistency",
]
