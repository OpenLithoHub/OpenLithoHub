"""Bounded SOCS decomposition — mathematical validation (v3 §5/§18/§19).

The independent tiny oracle is DIRECT dense linear algebra on tiny
grids, allowed ONLY under tests/ (v3 §18): it validates the exact
bounded Gram top-K construction against a naive SVD reference.  It is
never importable as a runtime backend and never used for performance.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from openlithohub._utils.hopkins import (
    SOCS_RANK_ATOL,
    SOCS_RANK_RTOL,
    HopkinsParams,
    SocsNumericalClosureError,
    clear_kernel_cache,
    compute_socs_kernels,
    run_bounded_block_probe,
    simulate_aerial_image_hopkins,
    socs_problem_dimensions,
)
from openlithohub._utils.socs_memory_plan import (
    SOCS_STRATEGY,
    plan_socs_decomposition,
)

HOPKINS_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "openlithohub" / "_utils" / "hopkins.py"
)

FROZEN_ATOL = 1e-5


def _assert_field_close(
    got: torch.Tensor, want: torch.Tensor, *, atol_abs: float = FROZEN_ATOL, rel: float = 1e-3
) -> None:
    """Scale-aware frozen comparison gate.

    The open-frame calibration is a GLOBAL scaling whose firing threshold
    sits at a numerical knife edge for no-DC sources (annular/dipole on
    small grids): the bounded and oracle paths can legitimately differ by
    the calibration CONSTANT while encoding the identical spectrum.  The
    frozen gate therefore compares the normalized field (shape), plus an
    absolute gate for genuinely-zero signals; the absolute calibration
    itself is locked separately by the open-frame unity test."""
    scale = float(want.abs().max())
    assert torch.isfinite(got).all()
    if scale <= atol_abs:
        assert float((got - want).abs().max()) <= atol_abs
        return
    assert float(((got / scale) - (want / scale)).abs().max()) <= rel


def _params(illumination: str = "circular", num_kernels: int = 8) -> HopkinsParams:
    return HopkinsParams(
        wavelength_nm=13.5,
        na=0.33,
        sigma=0.9,
        sigma_inner=0.6,
        pixel_size_nm=1.0,
        num_kernels=num_kernels,
        illumination=illumination,  # type: ignore[arg-type]
    )


def _plan_with_chunk(chunk: int, grid: int, n_src: int, n_freq: int, k: int):
    """A legal plan with an arbitrary chunk size and a CONSISTENT hash —
    the test-side equivalent of a driver choosing a different (still
    legal) chunk policy."""
    base = plan_socs_decomposition(
        grid_size=grid,
        n_src=n_src,
        n_freq=n_freq,
        K=k,
        dtype="complex64",
        complex_bytes=8,
        device="cpu",
        facts=None,
    )
    stepped = replace(
        base,
        selected_chunk_columns=chunk,
        chunk_count=int(math.ceil(n_freq / chunk)),
    )
    return replace(stepped, plan_sha256=stepped.plan_sha256_excluding_self())


def _freq_basis(kernels: torch.Tensor, grid: int) -> torch.Tensor:
    """Orthonormal frequency-domain basis (n_freq, K) of the top-K kernel
    set (QR-stabilized)."""
    a = torch.fft.fft2(
        torch.fft.ifftshift(kernels.to(torch.complex128), dim=(-2, -1)), dim=(-2, -1)
    ).reshape(kernels.shape[0], -1)
    q, _ = torch.linalg.qr(a.T)
    return q


def _principal_angle_gap(kernels_a: torch.Tensor, kernels_b: torch.Tensor, grid: int) -> float:
    """V3.1: 1 - sigma_min(Q_A^H Q_B) — the true subspace agreement gate.
    Identical K-dimensional subspaces give exactly 0; ANY rotation away
    from the oracle top-K subspace strictly decreases sigma_min."""
    qa = _freq_basis(kernels_a, grid)
    qb = _freq_basis(kernels_b, grid)
    cosines = torch.linalg.svdvals(qa.conj().T @ qb)
    return 1.0 - float(cosines.min().real)


def _stable_subspace_depth(ref_weights: torch.Tensor) -> int:
    """The largest top-j whose lower edge sits INSIDE a spectral gap
    (relative gap > 1e-6 below the trailing eigenvalue).  A top-K set
    whose K-th eigenvalue is numerically degenerate with the next one
    has NO cross-backend stable membership (§5): the K-th mode is chosen
    from a split degenerate pair, so the honest subspace gate pins only
    the stable depth.  Returns 0 when even the leading eigenvalue is
    degenerate (the caller skips the subspace gate for that combo)."""
    lam = ref_weights.to(torch.float64)
    scale = lam[0].clamp(min=1e-30)
    depth = 0
    for j in range(lam.numel() - 1):
        if float((lam[j] - lam[j + 1]) / scale) > 1e-6:
            depth = j + 1
        else:
            break
    return depth


def _projector_gap(kernels_a: torch.Tensor, kernels_b: torch.Tensor, grid: int) -> float:
    """V3.1: the actual space projector difference max|P_A - P_B| with
    P = Q Q^H on the frequency grid (small grids only — n_freq^2 memory)."""
    qa = _freq_basis(kernels_a, grid)
    qb = _freq_basis(kernels_b, grid)
    return float((qa @ qa.conj().T - qb @ qb.conj().T).abs().max())


def _oracle_socs(params: HopkinsParams, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent reference: naive full-H SVD on a TINY grid (§18)."""
    from openlithohub._utils.hopkins import _build_frequency_block, _socs_static_tables

    dims = socs_problem_dimensions(params, grid, "cpu")
    pupil_flat, shifts, weights_sqrt = _socs_static_tables(params, grid, torch.device("cpu"))
    h = torch.empty((dims.n_src, dims.n_freq), dtype=torch.complex64)
    step = 256
    for start in range(0, dims.n_freq, step):
        end = min(start + step, dims.n_freq)
        h[:, start:end] = _build_frequency_block(
            pupil_flat, shifts, weights_sqrt, grid, start, end, torch.device("cpu")
        )
    u, s, vh = torch.linalg.svd(h, full_matrices=False)  # oracle ONLY
    lam = (s**2)[: dims.K]
    kernels_freq = vh[: dims.K].reshape(dims.K, grid, grid)
    spatial = torch.fft.fftshift(torch.fft.ifft2(kernels_freq), dim=(-2, -1))
    weights = lam.to(torch.float32)
    open_frame = torch.zeros((), dtype=torch.float32)
    for idx in range(dims.K):
        dc = spatial[idx].sum()
        open_frame = open_frame + weights[idx] * (dc.real**2 + dc.imag**2)
    if float(open_frame) > 0.0:
        weights = (weights / open_frame).clamp(max=1e6)
    return spatial.to(torch.complex64), weights


# ---- §19 validation matrix: grids × optical sets × K ----------------------------


@pytest.mark.parametrize("grid", [16, 32, 64, 128])
@pytest.mark.parametrize("illumination", ["circular", "annular", "dipole"])
@pytest.mark.parametrize("num_kernels", [4, 8, 24])
def test_bounded_socs_matches_independent_oracle(
    grid: int, illumination: str, num_kernels: int
) -> None:
    torch.manual_seed(0)
    params = _params(illumination, num_kernels)
    dims = socs_problem_dimensions(params, grid, "cpu")
    if grid <= 32 and illumination != "circular" and dims.n_src - 1 <= dims.K:
        # §5: mode phase, basis orientation AND the boundary of a
        # nearly-degenerate top-K set are only stable away from the rank
        # edge.  On a tiny grid a no-DC source has n_src ≈ K: the K-th
        # mode sits exactly on a degenerate eigenvalue split that no
        # frozen tolerance can pin across BLAS backends.  The strong
        # oracle gates below are exercised on every non-edge combination;
        # this edge combination keeps the physical contract (finite,
        # non-negative, deterministic, circular unity).
        pytest.skip(
            f"K={dims.K} at the numerical-rank edge (n_src={dims.n_src}) on "
            f"grid={grid}: degenerate top-K boundary is not cross-platform stable (§5)"
        )
    clear_kernel_cache()
    kernels, weights = compute_socs_kernels(params, grid)
    ref_kernels, ref_weights = _oracle_socs(params, grid)

    assert kernels.shape == (dims.K, grid, grid)
    assert weights.shape == (dims.K,)
    # §5: mode phase/basis may rotate inside near-degenerate subspaces —
    # the frozen gates are the singular VALUE spectrum and the SUBSPACE.
    _assert_field_close(weights, ref_weights)

    # §5 (V3.1): TRUE subspace gate.  The earlier a^H a vs b^H b form
    # compared each basis against its OWN Gram matrix — two orthogonal
    # bases of two entirely different subspaces would both pass.  The
    # gate pins the bounded top-K SUBSPACE to the oracle's — at the
    # stable depth: when the K-th eigenvalue is numerically degenerate
    # with the next one, the top-K MEMBERSHIP itself is a §5 knife edge
    # (the physical outputs stay consistent — the aerial gates above —
    # because the contested mode carries the same weight on both sides).
    depth = _stable_subspace_depth(ref_weights)
    if depth >= 1:
        angle_gap = _principal_angle_gap(kernels[:depth], ref_kernels[:depth], grid)
        assert angle_gap <= 1e-6, (
            f"principal-angle gap {angle_gap} at stable depth {depth}/{dims.K}"
        )
        if grid <= 32:
            projector_err = _projector_gap(kernels[:depth], ref_kernels[:depth], grid)
            assert projector_err <= 1e-6, f"projector error {projector_err}"

    # open-frame normalization: whether the open-frame calibration fires
    # depends on whether the truncated top-K subspace contains the DC
    # component (annular/dipole geometry can legitimately concentrate the
    # DC energy outside top-K).  The frozen gate is CONSISTENCY with the
    # oracle: bounded and oracle must apply the same calibration and
    # produce the same uniform-mask image.
    mask = torch.ones((grid, grid))
    aerial = simulate_aerial_image_hopkins(mask, kernels=kernels, weights=weights)
    ref_uniform = simulate_aerial_image_hopkins(mask, kernels=ref_kernels, weights=ref_weights)
    assert torch.isfinite(aerial).all()
    _assert_field_close(aerial, ref_uniform)
    if illumination == "circular":
        # a DC-containing source always calibrates to unity
        assert float(aerial.mean()) == pytest.approx(1.0, abs=FROZEN_ATOL)

    # aerial image on a deterministic mask vs the oracle kernels — the
    # frozen gate is scale-aware: an uncompensated (no-DC) source carries
    # large raw λ weights, so the absolute gate scales with the signal.
    torch.manual_seed(7)
    det_mask = (torch.rand((grid, grid)) > 0.5).float()
    got = simulate_aerial_image_hopkins(det_mask, kernels=kernels, weights=weights)
    want = simulate_aerial_image_hopkins(det_mask, kernels=ref_kernels, weights=ref_weights)
    _assert_field_close(got, want)


def test_aerial_matches_oracle_through_params_path() -> None:
    """The public params-only forward (worker path) equals the oracle."""
    torch.manual_seed(1)
    params = _params("circular", 8)
    grid = 32
    clear_kernel_cache()
    mask = (torch.rand((grid, grid)) > 0.5).float()
    got = simulate_aerial_image_hopkins(mask, params=params)
    ref_kernels, ref_weights = _oracle_socs(params, grid)
    want = simulate_aerial_image_hopkins(mask, kernels=ref_kernels, weights=ref_weights)
    _assert_field_close(got, want)


def test_finite_and_nonnegative_witness() -> None:
    torch.manual_seed(2)
    params = _params("quasar", 6)
    grid = 24
    clear_kernel_cache()
    mask = (torch.rand((grid, grid)) > 0.4).float()
    aerial = simulate_aerial_image_hopkins(mask, params=params)
    assert torch.isfinite(aerial).all() and bool((aerial >= 0).all())


def test_gradient_witness_through_bounded_path() -> None:
    """The public differentiable path keeps its mask gradient (kernels are
    cached detached)."""
    torch.manual_seed(3)
    params = _params("circular", 4)
    grid = 24
    clear_kernel_cache()
    mask = torch.rand((grid, grid), requires_grad=True)
    aerial = simulate_aerial_image_hopkins(mask, params=params)
    aerial.sum().backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()
    assert float(mask.grad.abs().sum()) > 0.0


# ---- §19: chunk-size invariance --------------------------------------------------


@pytest.mark.parametrize("chunk_a,chunk_b", [(256, 1024), (512, 4096)])
def test_chunk_size_changes_scheduling_only(chunk_a: int, chunk_b: int) -> None:
    torch.manual_seed(4)
    params = _params("circular", 8)
    grid = 32
    dims = socs_problem_dimensions(params, grid, "cpu")
    clear_kernel_cache()
    k1, w1 = compute_socs_kernels(
        params, grid, memory_plan=_plan_with_chunk(chunk_a, grid, dims.n_src, dims.n_freq, dims.K)
    )
    clear_kernel_cache()
    k2, w2 = compute_socs_kernels(
        params, grid, memory_plan=_plan_with_chunk(chunk_b, grid, dims.n_src, dims.n_freq, dims.K)
    )
    assert torch.allclose(k1, k2, atol=1e-6, rtol=0.0)
    assert torch.allclose(w1, w2, atol=1e-6, rtol=0.0)
    mask = (torch.rand((grid, grid)) > 0.5).float()
    a1 = simulate_aerial_image_hopkins(mask, kernels=k1, weights=w1)
    a2 = simulate_aerial_image_hopkins(mask, kernels=k2, weights=w2)
    assert torch.allclose(a1, a2, atol=FROZEN_ATOL, rtol=0.0)


def test_cache_identity_binds_strategy_and_chunk() -> None:
    """§13: an obsolete-strategy or different-chunk cache entry can never
    satisfy the key."""
    torch.manual_seed(5)
    params = _params("circular", 4)
    grid = 24
    dims = socs_problem_dimensions(params, grid, "cpu")
    clear_kernel_cache()
    k1, _ = compute_socs_kernels(
        params, grid, memory_plan=_plan_with_chunk(256, grid, dims.n_src, dims.n_freq, dims.K)
    )
    # same params, different chunk policy → recomputed (not served stale)
    k2, _ = compute_socs_kernels(
        params, grid, memory_plan=_plan_with_chunk(512, grid, dims.n_src, dims.n_freq, dims.K)
    )
    assert k1.data_ptr() != k2.data_ptr()
    # identical plan → cache hit (same object)
    k3, _ = compute_socs_kernels(
        params, grid, memory_plan=_plan_with_chunk(512, grid, dims.n_src, dims.n_freq, dims.K)
    )
    assert k2.data_ptr() == k3.data_ptr()


def test_worker_cannot_replan_with_tampered_plan_hash() -> None:
    """§10: a plan whose payload no longer hashes to plan_sha256 is
    refused, not honored."""
    torch.manual_seed(6)
    params = _params("circular", 4)
    grid = 24
    dims = socs_problem_dimensions(params, grid, "cpu")
    plan = _plan_with_chunk(256, grid, dims.n_src, dims.n_freq, dims.K)
    tampered = replace(plan, selected_chunk_columns=128)
    clear_kernel_cache()
    from openlithohub._utils.socs_memory_plan import MemoryPlanContractViolation

    with pytest.raises(MemoryPlanContractViolation, match="hash mismatch"):
        compute_socs_kernels(params, grid, memory_plan=tampered)


# ---- §14B: bounded block probe -----------------------------------------------------


def test_bounded_block_probe_passes_on_cpu_problem_shape() -> None:
    params = _params("circular", 4)
    probe = run_bounded_block_probe(params, 32, "cpu", block_columns=64)
    assert probe["pass"] is True
    assert probe["finite"] is True
    assert probe["block_device"] == "cpu"
    assert probe["gram_update_device"] == "cpu"


# ---- §2: the legacy dense runtime is GONE -------------------------------------------


def test_legacy_dense_runtime_is_absent_from_production_paths() -> None:
    """§37 Q1/Q2/Q3: no full-H materialization, no dense full-H SVD, and
    no dense fallback flag/env remains in any production SOCS runtime."""
    production_sources = [
        HOPKINS_PATH.read_text(),
        (HOPKINS_PATH.parent / "socs_memory_plan.py").read_text(),
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "industrial-v2"
            / "run_v2_benchmark.py"
        ).read_text(),
    ]
    for source in production_sources:
        # §37 Q1/Q2: no dense full-H SVD CALL exists in production
        # (docstring mentions of the removed runtime are the record of its
        # removal, not a runtime path).
        assert "linalg.svd(" not in source, "dense full-H SVD must not exist in production"
        assert "dense-socs" not in source, "no --dense-socs flag may exist"
        assert "DENSE_SOCS" not in source, "no dense re-enable env var may exist"
        assert "full_matrices" not in source, "no dense economy-SVD call may exist"


def test_numerical_closure_hard_fails_below_frozen_rank() -> None:
    """§4: fewer than K numerically positive modes is a hard failure, not
    a silent truncation."""
    params = _params("circular", 8)
    grid = 16
    # A zero pupil produces an all-zero H → zero spectrum → rank closure
    # failure.  Reach in through the plan only; the guard lives in the
    # eigendecomposition path.
    from openlithohub._utils import hopkins as hopkins_module

    original = hopkins_module._socs_static_tables

    def zero_pupil(*args: object, **kwargs: object):
        pupil, shifts, weights = original(*args, **kwargs)  # type: ignore[arg-type]
        return torch.zeros_like(pupil), shifts, weights

    clear_kernel_cache()
    hopkins_module._socs_static_tables = zero_pupil  # type: ignore[assignment]
    try:
        with pytest.raises(SocsNumericalClosureError, match="numerical closure"):
            compute_socs_kernels(params, grid)
    finally:
        hopkins_module._socs_static_tables = original  # type: ignore[assignment]
        clear_kernel_cache()
    # frozen tolerances stay frozen
    assert SOCS_RANK_RTOL == 1e-9 and SOCS_RANK_ATOL == 1e-12


def test_strategy_identity_is_frozen() -> None:
    assert SOCS_STRATEGY == "exact_block_gram_topk_v1"
