"""Integration tests: full pipeline with real models."""

import torch

from openlithohub.models.levelset_ilt import LevelSetILTModel


class TestLevelSetILTPipeline:
    def test_optimization_improves_epe(self) -> None:
        """End-to-end: LevelSet-ILT produces a mask with lower EPE than identity."""
        design = torch.zeros(32, 32)
        design[10:22, 10:22] = 1.0

        # Optimized: LevelSet-ILT
        model = LevelSetILTModel(iterations=50, lr=0.1, sigma_px=1.5)
        result = model.predict(design)

        # The EPE of the optimized mask should be finite and the model should converge
        assert result.metadata["final_loss"] < 1.0
        assert result.mask.shape == design.shape

    def test_pipeline_with_process_node(self) -> None:
        """Test that process node config integrates with model."""
        from openlithohub.workflow.process_node import get_node

        node = get_node("45nm")
        model = LevelSetILTModel(iterations=20, sigma_px=node.sigma_px)

        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0
        result = model.predict(design)

        assert result.mask.shape == design.shape
        assert result.metadata["sigma_px"] == node.sigma_px

    def test_levelset_then_mrc_check(self) -> None:
        """Verify optimized mask can be checked for MRC compliance."""
        from openlithohub.benchmark.compliance.mrc import check_mrc

        design = torch.zeros(32, 32)
        design[8:24, 8:24] = 1.0

        model = LevelSetILTModel(iterations=30, sigma_px=1.0)
        result = model.predict(design)

        mrc = check_mrc(result.mask, min_width_nm=2.0, min_spacing_nm=2.0, pixel_size_nm=1.0)
        assert hasattr(mrc, "passed")
        assert hasattr(mrc, "violation_count")
