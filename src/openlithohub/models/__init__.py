"""Layer 3: Model Integration — abstract interface and registry for lithography models."""

from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.bayesian_stochastic import (
    BayesianStochasticModel,
    StochasticUNet,
    generate_synthetic_ground_truth,
)
from openlithohub.models.diffusion_mask import (
    DiffusionMaskBenchmark,
    DiffusionMaskConfig,
    DiffusionMaskSynthesis,
    LithoGuidance,
    MaskDiffusionUNet,
    MaskLatentDecoder,
    MaskLatentEncoder,
)
from openlithohub.models.generative_sraf import (
    LithographicImprovementScorer,
    SRAFConfig,
    SRAFGenerator,
    SRAFPipeline,
    SRAFRLTrainer,
)
from openlithohub.models.grpo_warm_start import GRPOConfig, GRPOWarmStart, StyleConditioning
from openlithohub.models.hub import ModelHub
from openlithohub.models.layout_mae import LayoutMAE, LayoutMAEConfig
from openlithohub.models.registry import ModelRegistry
from openlithohub.models.resist_pinn import (
    ResistBenchmark,
    ResistCalibrator,
    ResistPhysicsConstraints,
    ResistPINN,
)
from openlithohub.models.vae_benchmark import VAEBenchmark

__all__ = [
    "LithographyModel",
    "PredictionResult",
    "ModelRegistry",
    "ModelHub",
    "LayoutMAE",
    "LayoutMAEConfig",
    "VAEBenchmark",
    "BayesianStochasticModel",
    "StochasticUNet",
    "generate_synthetic_ground_truth",
    "DiffusionMaskBenchmark",
    "DiffusionMaskConfig",
    "DiffusionMaskSynthesis",
    "LithoGuidance",
    "MaskDiffusionUNet",
    "MaskLatentDecoder",
    "MaskLatentEncoder",
    "GRPOWarmStart",
    "GRPOConfig",
    "StyleConditioning",
    "ResistPINN",
    "ResistCalibrator",
    "ResistBenchmark",
    "ResistPhysicsConstraints",
    "SRAFGenerator",
    "SRAFRLTrainer",
    "SRAFPipeline",
    "LithographicImprovementScorer",
    "SRAFConfig",
]
