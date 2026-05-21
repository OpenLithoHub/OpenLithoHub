"""End-to-end ranking test: Identity-like vs a real ILT model.

This pins the *headline* contract of the wafer-L2 leaderboard rebase:
a model that does no mask correction must rank below a real ILT model
whose output prints closer to the target after Hopkins diffraction. The
unit test in ``test_tracker.py`` only exercises the sort key against
pre-baked records — this one runs each candidate through the real
``compute_l2_error`` / ``compute_wafer_epe`` pipeline before submitting,
so a regression in the simulator wiring, the metric, or the schema
rank-ordering surfaces here.

Why ``levelset-ilt`` and not ``openilt`` / ``neural-ilt``: at the small
synthetic scale used by ``scripts/generate_baselines.py`` the latter
two converge to ≈Identity (their internal forward models match the
target so closely that no improvement is found). LevelSet-ILT does
shift its mask off the target on these patterns and lands at a lower
wafer L2 — see ``baselines/results.json``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from openlithohub.benchmark.metrics.epe import compute_wafer_epe
from openlithohub.benchmark.metrics.l2_error import compute_l2_error
from openlithohub.leaderboard.schema import BenchmarkResult, MaskTopology, ProcessNode
from openlithohub.leaderboard.tracker import LeaderboardStore
from openlithohub.models.registry import registry
from openlithohub.simulators.base import SimulatorConfig
from openlithohub.simulators.hopkins_sim import HopkinsSimulator


def _evaluate(
    model_name: str,
    targets: list[torch.Tensor],
    *,
    pixel_size_nm: float,
    simulator: HopkinsSimulator,
) -> BenchmarkResult:
    """Run ``model_name`` over ``targets`` and aggregate to a leaderboard entry."""
    # Late-import keeps test collection cheap when only schema tests run.
    import openlithohub.models.examples.dummy_model  # noqa: F401
    import openlithohub.models.levelset_ilt  # noqa: F401

    model = registry.get(model_name)
    model.setup()
    try:
        l2_pixels = []
        l2_nm2 = []
        wafer_means = []
        wafer_maxes = []
        for target in targets:
            pred = model.predict(target)
            l2 = compute_l2_error(
                pred.mask, target, pixel_size_nm=pixel_size_nm, simulator=simulator
            )
            wepe = compute_wafer_epe(
                pred.mask, target, pixel_size_nm=pixel_size_nm, simulator=simulator
            )
            l2_pixels.append(float(l2["l2_error_pixels"]))
            l2_nm2.append(float(l2["l2_error_nm2"]))
            if wepe.get("valid"):
                wafer_means.append(float(wepe["epe_mean_nm"]))
                wafer_maxes.append(float(wepe["epe_max_nm"]))
    finally:
        model.teardown()

    return BenchmarkResult(
        model_name=model_name,
        dataset="lithobench",
        process_node=ProcessNode.N7,
        mask_topology=MaskTopology.MANHATTAN,
        epe_mean_nm=0.0,
        epe_max_nm=0.0,
        epe_wafer_mean_nm=(sum(wafer_means) / len(wafer_means)) if wafer_means else None,
        epe_wafer_max_nm=(max(wafer_maxes)) if wafer_maxes else None,
        l2_error_pixels=sum(l2_pixels) / len(l2_pixels),
        l2_error_nm2=sum(l2_nm2) / len(l2_nm2),
    )


def test_levelset_ilt_outranks_identity_after_full_pipeline(tmp_path: Path) -> None:
    """``levelset-ilt`` (200-iteration SGD) must rank below ``dummy-identity``
    on wafer L2. This is the wafer-L2 leaderboard contract end-to-end:
    real models → ``compute_l2_error`` (shared simulator) → ``BenchmarkResult``
    → ``LeaderboardStore.submit`` → ``query``.

    Uses the same synthetic pattern set as ``scripts/generate_baselines.py``
    (8 patterns, 64×64 each at 8 nm/px) so the ranking inversion that
    appears in the published baselines also appears here.
    """
    from scripts.generate_baselines import build_synthetic_patterns

    targets = [p.target_mask for p in build_synthetic_patterns(grid=64)]

    pixel_size_nm = 8.0
    simulator = HopkinsSimulator(SimulatorConfig(pixel_size_nm=pixel_size_nm))

    identity = _evaluate(
        "dummy-identity", targets, pixel_size_nm=pixel_size_nm, simulator=simulator
    )
    ilt = _evaluate("levelset-ilt", targets, pixel_size_nm=pixel_size_nm, simulator=simulator)

    # If this guard fires the test is no longer measuring ranking — the
    # ILT model has regressed on these patterns, which is itself a signal
    # worth surfacing here rather than letting the ranking assert mask it.
    assert ilt.l2_error_pixels < identity.l2_error_pixels, (
        f"levelset-ilt L2 {ilt.l2_error_pixels} not better than identity "
        f"{identity.l2_error_pixels}; ILT regressed on the test patterns"
    )

    store = LeaderboardStore(tmp_path / "leaderboard.json")
    store.submit(identity)
    store.submit(ilt)

    ranked = store.query()
    assert [r.model_name for r in ranked] == ["levelset-ilt", "dummy-identity"], (
        f"ranking did not put ILT above Identity: {[r.model_name for r in ranked]}"
    )
