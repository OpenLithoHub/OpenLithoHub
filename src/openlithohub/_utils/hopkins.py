"""Differentiable Hopkins partial-coherence aerial image model via SOCS.

Hopkins formulation describes partial-coherent imaging through a
Transmission Cross Coefficient (TCC):

    I(x) = ∫∫ TCC(f1, f2) * M~(f1) * conj(M~(f2)) * exp(2πi (f1-f2) x) df1 df2

where M~ is the mask Fourier transform, J is the source intensity, P is the
pupil function, and

    TCC(f1, f2) = ∫ J(f) * P(f + f1) * conj(P(f + f2)) df

The Sum Of Coherent Systems (SOCS) decomposition is the eigendecomposition
of TCC viewed as a Hermitian operator:

    TCC = Σ_k w_k * φ_k(f1) * conj(φ_k(f2))           with w_k descending

Truncating at K kernels yields the standard fast OPC forward model:

    I(x) ≈ Σ_k w_k * | (mask * φ_k)(x) |^2

This module implements both steps in pure PyTorch so the entire chain is
auto-differentiable (the mask is the optimization variable in ILT).
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import torch

from openlithohub._constants import (
    DEFOCUS_NM_DEFAULT,
    NA_IMMERSION,
    NUM_KERNELS_DEFAULT,
    PIXEL_SIZE_NM_DEFAULT,
    POLE_OPENING_DEG_DEFAULT,
    SIGMA_INNER_DEFAULT,
    SIGMA_OUTER_DEFAULT,
    WAVELENGTH_ARF_NM,
)
from openlithohub._utils.socs_memory_plan import (
    SOCS_STRATEGY,
    SOCS_STRATEGY_VERSION,
    DeviceMemoryFacts,
    MemoryPlanContractViolation,
    SocsMemoryPlan,
    assert_headroom,
    collect_cuda_memory_facts,
    plan_socs_decomposition,
    validate_worker_plan,
)

IlluminationKind = Literal["circular", "annular", "dipole", "quasar"]


@dataclass(frozen=True)
class HopkinsParams:
    """Optical parameters for Hopkins partial-coherence imaging.

    Attributes:
        wavelength_nm: Exposure wavelength (193 nm = ArF, 13.5 nm = EUV).
        na: Numerical aperture (image-side). 1.35 = ArF immersion, 0.33 = EUV NXE.
        sigma: Partial-coherence factor for circular illumination, or
            outer sigma for annular/dipole/quasar.
        sigma_inner: Inner sigma for annular/dipole/quasar (ignored for circular).
        pixel_size_nm: Physical size of one mask pixel.
        num_kernels: SOCS truncation order. Defaults to 24 to match the
            ``Yang2023_LithoBench`` Table II benchmark; that paper does
            not publish a truncation-error vs K curve, so the underlying
            defensibility chain is Cobb 1995, §IV (the original SOCS
            construction) plus accumulated practice. Production
            deployments at a different node should re-sweep K against
            their own EPE noise floor before pinning this value.
        illumination: Source shape — circular, annular, dipole (X-direction
            poles), or quasar (4-pole, CQuad).
        dipole_angle_deg: Pole-pair orientation for dipole/quasar (degrees).
        pole_opening_deg: Half-angle of each pole wedge for dipole/quasar
            (degrees). 30° is a common production CQuad value.
        defocus_nm: Defocus offset; affects the pupil phase only.
    """

    wavelength_nm: float = WAVELENGTH_ARF_NM
    na: float = NA_IMMERSION
    sigma: float = SIGMA_OUTER_DEFAULT
    sigma_inner: float = SIGMA_INNER_DEFAULT
    pixel_size_nm: float = PIXEL_SIZE_NM_DEFAULT
    num_kernels: int = NUM_KERNELS_DEFAULT
    illumination: IlluminationKind = "circular"
    dipole_angle_deg: float = 0.0
    pole_opening_deg: float = POLE_OPENING_DEG_DEFAULT
    defocus_nm: float = DEFOCUS_NM_DEFAULT

    def cache_key(self, grid_size: int, device: str, kernel_dtype: str) -> tuple[Hashable, ...]:
        return (
            self.wavelength_nm,
            self.na,
            self.sigma,
            self.sigma_inner,
            self.pixel_size_nm,
            self.num_kernels,
            self.illumination,
            self.dipole_angle_deg,
            self.pole_opening_deg,
            self.defocus_nm,
            grid_size,
            device,
            kernel_dtype,
        )


_KERNEL_CACHE: OrderedDict[Hashable, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_KERNEL_CACHE_MAXSIZE = 8


def _wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap an angle tensor to the interval (-pi, pi]."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _frequency_grid(grid_size: int, pixel_size_nm: float, device: torch.device) -> torch.Tensor:
    """Cycles per nm grid for an N×N mask, ordered as fftfreq.

    Returns shape (grid_size,).
    """
    grid: torch.Tensor = torch.fft.fftfreq(grid_size, d=pixel_size_nm, device=device).to(
        torch.float32
    )
    return grid


def _illumination_samples(
    params: HopkinsParams,
    grid_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the source plane on a dense polar grid and map each source point
    to the nearest mask-frequency-grid index (dy, dx).

    Decoupling the source sampling resolution from the mask-frequency grid is
    what lets SOCS produce many distinct kernels even on small benchmark
    grids — without it, a 64×64 mask only captures a single dc source point.

    Returns:
        shifts: int64 tensor of shape (S, 2), each row (dy, dx) modulo grid_size.
        weights: float32 tensor of shape (S,), source intensities, sum to 1.
    """
    f_grid_step = 1.0 / (grid_size * params.pixel_size_nm)
    f_pupil = params.na / params.wavelength_nm

    # Pick source resolution finer than the mask-frequency grid so we resolve
    # the source shape; cap at a sensible maximum.
    n_radial = max(8, int(math.ceil(8.0 * f_pupil / max(f_grid_step, 1e-12))))
    n_radial = min(n_radial, 64)
    n_angular = 2 * n_radial

    radii = torch.linspace(0.0, 1.0, n_radial, device=device)
    angles = torch.linspace(0.0, 2.0 * math.pi, n_angular + 1, device=device)[:-1]
    rr, aa = torch.meshgrid(radii, angles, indexing="ij")
    rr = rr.reshape(-1)
    aa = aa.reshape(-1)

    # Drop the duplicated origin samples
    keep = (rr > 0) | (aa == 0)
    rr = rr[keep]
    aa = aa[keep]

    sigma_outer = params.sigma
    sigma_inner = params.sigma_inner
    if params.illumination == "circular":
        in_src = rr <= sigma_outer
    elif params.illumination == "annular":
        in_src = (rr <= sigma_outer) & (rr >= sigma_inner)
    elif params.illumination in ("dipole", "quasar"):
        angle_rad = math.radians(params.dipole_angle_deg)
        opening_rad = math.radians(max(1.0, params.pole_opening_deg))
        in_ring = rr <= sigma_outer
        if sigma_inner > 0:
            in_ring = in_ring & (rr >= sigma_inner)
        if params.illumination == "dipole":
            # Two poles on the dipole axis; angular distance to ±x-axis
            # (after rotation by dipole_angle) must be within opening_rad.
            theta = aa - angle_rad
            d_axis = torch.minimum(
                torch.abs(_wrap_to_pi(theta)),
                torch.abs(_wrap_to_pi(theta - math.pi)),
            )
            in_src = in_ring & (d_axis <= opening_rad)
        else:
            # Quasar / CQuad: 4 poles at ±dipole_angle and ±dipole_angle + 90°.
            theta = aa - angle_rad
            d_pole = torch.full_like(theta, math.pi)
            for offset in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
                d_pole = torch.minimum(d_pole, torch.abs(_wrap_to_pi(theta - offset)))
            in_src = in_ring & (d_pole <= opening_rad)
    else:
        raise ValueError(f"Unknown illumination kind: {params.illumination!r}")

    rr = rr[in_src]
    aa = aa[in_src]
    if rr.numel() == 0:
        raise ValueError(
            f"Illumination {params.illumination!r} with sigma=({sigma_inner},{sigma_outer}) "
            "yields zero source samples."
        )

    # Convert (rr, aa) on normalized pupil coords to physical cycles/nm
    fx = rr * f_pupil * torch.cos(aa)
    fy = rr * f_pupil * torch.sin(aa)

    # Map each source point to the nearest fft frequency bin (signed shift)
    sx = torch.round(fx / f_grid_step).to(torch.int64) % grid_size
    sy = torch.round(fy / f_grid_step).to(torch.int64) % grid_size

    shifts = torch.stack([sy, sx], dim=1)
    # Merge identical bins by accumulating weights
    flat_key = shifts[:, 0] * grid_size + shifts[:, 1]
    unique_keys, inverse = torch.unique(flat_key, return_inverse=True)
    n_unique = unique_keys.numel()
    src_weights = torch.zeros(n_unique, dtype=torch.float32, device=device)
    # Polar-grid Jacobian: an (r, θ) bin covers physical area r·dr·dθ, so
    # each sample contributes its radius — not a flat 1 — to the source
    # intensity. Without this the centre is over-weighted (issue #29:
    # circular illumination measured ~38% of intensity in the inner third
    # instead of the ~11% an equal-area source predicts).
    sample_jac = rr.to(torch.float32)
    src_weights.scatter_add_(0, inverse, sample_jac)
    src_weights = src_weights / src_weights.sum()
    unique_shifts = torch.stack([unique_keys // grid_size, unique_keys % grid_size], dim=1)
    return unique_shifts, src_weights


def _pupil(
    fx: torch.Tensor,
    fy: torch.Tensor,
    params: HopkinsParams,
) -> torch.Tensor:
    """Complex pupil P(f) — amplitude inside NA cutoff, defocus phase.

    Defocus phase follows the scalar parabolic approximation
    φ = π * defocus * λ * (f^2) (small-angle), which is sufficient for
    benchmarking; full vector defocus can be added later without changing API.
    """
    f_pupil = params.na / params.wavelength_nm
    r2 = fx**2 + fy**2
    aperture = (r2 <= f_pupil**2).to(torch.float32)
    if params.defocus_nm == 0.0:
        return aperture.to(torch.complex64)
    phase = math.pi * params.defocus_nm * params.wavelength_nm * r2
    real = aperture * torch.cos(phase)
    imag = aperture * torch.sin(phase)
    return torch.complex(real, imag)


# ---- bounded SOCS decomposition (GPU Authority Repair v3, §3-§4) ----------------

SOCS_EIG_NEG_TOLERANCE_RTOL = 1e-6
"""Frozen §4 policy: an eigenvalue below ``-rtol * lambda_max`` is a
material negative (Hermitian violation), not roundoff — hard failure."""

SOCS_RANK_RTOL = 1e-9
SOCS_RANK_ATOL = 1e-12
"""Frozen §4 numerical-rank tolerance: eigenmodes at or below
``atol + rtol * lambda_max`` are not numerically positive."""


class SocsNumericalClosureError(RuntimeError):
    """The exact bounded Gram route failed numerical closure (§4/§30).

    Raised on materially negative eigenvalues or fewer than K numerically
    positive modes.  Per protocol this STOPs the track and moves to a
    memory-bounded matrix-free top-K solver — it NEVER restores the old
    dense runtime, and it is never resolved by lowering the grid."""


@dataclass(frozen=True)
class SocsProblemDimensions:
    """The exact formal SOCS problem dimensions (§14: preflight and
    planner must reason about the ACTUAL formal grid)."""

    n_src: int
    n_freq: int
    K: int


def socs_problem_dimensions(
    params: HopkinsParams,
    grid_size: int,
    device: torch.device | str = "cpu",
) -> SocsProblemDimensions:
    """Compute the exact ``(n_src, n_freq, K)`` for the frozen optical
    configuration (§37 Q5: n_src comes from the REAL source sampling of
    the frozen params, never from a nominal estimate)."""
    dev = torch.device(device)
    shifts, _ = _illumination_samples(params, grid_size, dev)
    n_src = int(shifts.shape[0])
    return SocsProblemDimensions(
        n_src=n_src,
        n_freq=grid_size * grid_size,
        K=max(1, min(params.num_kernels, n_src)),
    )


def _build_frequency_block(
    pupil_flat: torch.Tensor,
    src_shifts_cpu: list[tuple[int, int]],
    src_weights_sqrt: list[float],
    grid_size: int,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate ``H[:, start:end]`` — the (n_src, chunk) complex64 block of
    shifted-pupil columns — WITHOUT materializing full-grid index tensors
    (§12): only the chunk's flat indices exist, and each source row is
    filled by one bounded gather."""
    cols = torch.arange(start, end, device=device, dtype=torch.int64)
    y = torch.div(cols, grid_size, rounding_mode="floor")
    x = cols - y * grid_size
    block = torch.empty((len(src_shifts_cpu), end - start), dtype=torch.complex64, device=device)
    for k, ((sy, sx), weight_sqrt) in enumerate(zip(src_shifts_cpu, src_weights_sqrt, strict=True)):
        iy = (y - sy) % grid_size
        ix = (x - sx) % grid_size
        gathered = pupil_flat.index_select(0, iy * grid_size + ix)
        block[k] = gathered * weight_sqrt
    return block


def _socs_static_tables(
    params: HopkinsParams,
    grid_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[float]]:
    """Pupil (flattened) + CPU-resident source tables shared by both
    streamed passes."""
    f = _frequency_grid(grid_size, params.pixel_size_nm, device)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    pupil_flat = _pupil(fy, fx, params).reshape(-1)
    src_shifts, src_weights = _illumination_samples(params, grid_size, device)
    src_shifts_cpu = [(int(a), int(b)) for a, b in src_shifts.cpu().tolist()]
    weights_sqrt = [float(w) ** 0.5 for w in src_weights.cpu().tolist()]
    return pupil_flat, src_shifts_cpu, weights_sqrt


def run_bounded_block_probe(
    params: HopkinsParams,
    grid_size: int,
    device: torch.device | str,
    block_columns: int = 4096,
) -> dict[str, object]:
    """Preflight Gate B (§14B): push ONE representative H block through the
    EXACT worker block path — bounded generation, CUDA Gram update, finite
    checks, actual device witness.  Shares ``_build_frequency_block`` with
    the production decomposition, so a probe can never pass while the real
    worker path is broken."""
    dev = torch.device(device)
    pupil_flat, shifts, weights_sqrt = _socs_static_tables(params, grid_size, dev)
    end = min(int(block_columns), grid_size * grid_size)
    block = _build_frequency_block(pupil_flat, shifts, weights_sqrt, grid_size, 0, end, dev)
    gram_part = block.to(torch.complex128) @ block.to(torch.complex128).mH
    del block
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    finite = bool(torch.isfinite(gram_part.real).all() and torch.isfinite(gram_part.imag).all())
    return {
        "requested_device": str(device),
        "block_device": str(pupil_flat.device),
        "gram_update_device": str(gram_part.device),
        "block_columns": end,
        "n_src": len(shifts),
        "finite": finite,
        "pass": bool(
            finite
            and str(pupil_flat.device).startswith(str(dev))
            and str(gram_part.device).startswith(str(dev))
        ),
    }


def _plan_for_device(
    params: HopkinsParams,
    grid_size: int,
    device: torch.device,
    memory_plan: SocsMemoryPlan | None,
) -> SocsMemoryPlan:
    """Resolve the plan for this computation: an explicitly supplied plan
    is VALIDATED (a worker may never replan, §10); otherwise a fresh plan
    is computed from live device facts."""
    dims = socs_problem_dimensions(params, grid_size, device)
    if memory_plan is not None:
        validate_worker_plan(
            memory_plan,
            grid_size=grid_size,
            n_src=dims.n_src,
            n_freq=dims.n_freq,
            K=dims.K,
            dtype="complex64",
            device=str(device),
        )
        return memory_plan
    facts: DeviceMemoryFacts | None = None
    if device.type == "cuda":
        facts = collect_cuda_memory_facts(str(device))
    return plan_socs_decomposition(
        grid_size=grid_size,
        n_src=dims.n_src,
        n_freq=dims.n_freq,
        K=dims.K,
        dtype="complex64",
        complex_bytes=8,
        device=str(device),
        facts=facts,
    )


def compute_socs_kernels(
    params: HopkinsParams,
    grid_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    memory_plan: SocsMemoryPlan | None = None,
    memory_witness: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SOCS kernels and their weights for a square grid.

    The implementation is the EXACT two-pass bounded-memory Gram
    construction (GPU Authority Repair v3 §3): a streamed Gram
    accumulation over frequency-column blocks, one eigendecomposition of
    the (n_src x n_src) Gram matrix, then a streamed top-K
    reconstruction writing each block directly into the output slice.
    Full H and full Vh are NEVER resident (§2); the legacy dense
    full-H ``torch.linalg.svd`` runtime has been removed with no
    fallback (§2: no flag, no env var, no re-enable).

    Args:
        params: Optical parameters.
        grid_size: Square grid edge length (pixels).
        device: PyTorch device.
        dtype: Complex dtype of the returned kernels (``complex64`` or
            ``complex128``). The internal decomposition runs in
            ``complex64`` blocks with a ``complex128`` Gram accumulator.
        memory_plan: Optional frozen :class:`SocsMemoryPlan` (the formal
            worker path passes the driver-owned plan; supplying one means
            this call VALIDATES instead of replans, §10).
        memory_witness: Optional dict filled with the planned-vs-observed
            memory facts (§16) for the caller's authority artifact.

    Returns:
        kernels: complex tensor of shape (K, H, W) with the requested dtype.
            Each kernel is in the spatial domain, ready for FFT-based
            convolution.
        weights: real float32 tensor of shape (K,). Sorted descending.

    The returned kernels are zero-centered (fftshift-style) with the kernel
    support concentrated near the origin, which is what
    `simulate_aerial_image_hopkins` expects.
    """
    dev = torch.device(device)
    dims = socs_problem_dimensions(params, grid_size, dev)
    n_src, n_freq = dims.n_src, dims.n_freq
    K: int = dims.K  # noqa: N806 — K is the frozen SOCS truncation order

    plan = _plan_for_device(params, grid_size, dev, memory_plan)
    if not plan.memory_feasible:
        raise MemoryPlanContractViolation(
            f"{plan.reason} — refusing to execute an infeasible memory plan"
        )
    chunk = plan.selected_chunk_columns

    # §13: cache identity binds optical params, grid, device, dtype, K,
    # the strategy version and the chunk policy.  A cache produced by an
    # obsolete strategy can never satisfy this key.
    cache_key = tuple(params.cache_key(grid_size, str(dev), str(dtype))) + (
        SOCS_STRATEGY,
        SOCS_STRATEGY_VERSION,
        chunk,
    )
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        _KERNEL_CACHE.move_to_end(cache_key)
        return cached

    pupil_flat, shifts, weights_sqrt = _socs_static_tables(params, grid_size, dev)
    min_free_observed = 2**62
    headroom_checks = 0

    def guard() -> None:
        nonlocal min_free_observed, headroom_checks
        if dev.type == "cuda":
            free = assert_headroom(plan, str(dev))
            if free >= 0:
                min_free_observed = min(min_free_observed, free)
            headroom_checks += 1

    # ---- Pass 1: streamed Gram accumulation, complex128 (§3/§4) -------
    gram = torch.zeros((n_src, n_src), dtype=torch.complex128, device=dev)
    for start in range(0, n_freq, chunk):
        end = min(start + chunk, n_freq)
        guard()
        block = _build_frequency_block(pupil_flat, shifts, weights_sqrt, grid_size, start, end, dev)
        block128 = block.to(torch.complex128)
        gram.addmm_(block128, block128.mH)
        del block, block128

    # §4: symmetrize, descending sort, clamp only tiny roundoff, hard-fail
    # on material negatives, enforce the frozen numerical rank.
    gram = (gram + gram.mH) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(gram)
    del gram
    lam = torch.flip(eigvals, dims=(0,))
    u = torch.flip(eigvecs, dims=(1,))
    del eigvals, eigvecs
    lam_max = float(lam[0].real) if lam.numel() else 0.0
    neg_tolerance = SOCS_EIG_NEG_TOLERANCE_RTOL * max(lam_max, 0.0)
    if lam.numel() and float(lam[-1].real) < -neg_tolerance:
        raise SocsNumericalClosureError(
            f"bounded Gram route failed numerical closure: most negative "
            f"eigenvalue {float(lam[-1].real):.3e} < -{neg_tolerance:.3e} (§4 hard fail)"
        )
    lam = lam.clamp(min=0.0)
    rank_tolerance = SOCS_RANK_ATOL + SOCS_RANK_RTOL * max(lam_max, 0.0)
    positive_modes = int((lam.real > rank_tolerance).sum()) if lam.numel() else 0
    if positive_modes < K:
        raise SocsNumericalClosureError(
            f"bounded Gram route failed numerical closure: {positive_modes} "
            f"numerically positive modes < K={K} at frozen rank tolerance "
            f"(atol={SOCS_RANK_ATOL}, rtol={SOCS_RANK_RTOL})"
        )
    lam_k = lam[:K]
    u_k = u[:, :K]
    del lam, u
    sigma_inv = torch.rsqrt(lam_k.clamp(min=1e-300)).to(torch.complex128)

    # ---- Pass 2: streamed top-K reconstruction, direct slice writes ----
    kernels_freq = torch.empty((K, n_freq), dtype=torch.complex64, device=dev)
    for start in range(0, n_freq, chunk):
        end = min(start + chunk, n_freq)
        guard()
        block = _build_frequency_block(pupil_flat, shifts, weights_sqrt, grid_size, start, end, dev)
        v_block = (block.to(torch.complex128).mH @ u_k) * sigma_inv.unsqueeze(0)
        kernels_freq[:, start:end] = v_block.conj().T.to(torch.complex64)
        del block, v_block
    del u_k, sigma_inv

    kernels_spatial = torch.fft.ifft2(kernels_freq.reshape(K, grid_size, grid_size))
    kernels_spatial = torch.fft.fftshift(kernels_spatial, dim=(-2, -1))
    del kernels_freq

    # Calibrate so that an open-frame (all-ones) mask produces aerial ≈ 1.
    # For a constant mask, coherent_k = sum(kernel_k); aerial_open = Σ_k w_k |sum(k_k)|².
    weights = lam_k.real.to(torch.float32)
    open_frame = torch.zeros((), dtype=torch.float32, device=dev)
    for k_idx in range(K):
        coherent_dc = kernels_spatial[k_idx].sum()
        open_frame = open_frame + weights[k_idx] * (coherent_dc.real**2 + coherent_dc.imag**2)
    if float(open_frame) > 0.0:
        weights = weights / open_frame
        # Clamp rescaled weights to prevent overflow in downstream aerial
        # accumulation (extremely small open_frame values can produce huge weights).
        weights = weights.clamp(max=1e6)

    kernels_spatial = kernels_spatial.to(dtype)
    if memory_witness is not None:
        memory_witness.update(
            {
                "strategy": plan.strategy,
                "strategy_version": plan.strategy_version,
                "memory_plan_sha256": plan.plan_sha256,
                "memory_feasible": plan.memory_feasible,
                "selected_chunk_columns": plan.selected_chunk_columns,
                "chunk_count": plan.chunk_count,
                "planner_estimated_peak_bytes": plan.estimated_peak_bytes,
                "required_free_floor_bytes": plan.required_free_floor_bytes,
                "minimum_free_bytes_observed": (min_free_observed if headroom_checks else -1),
                "headroom_checks": headroom_checks,
            }
        )
    _KERNEL_CACHE[cache_key] = (kernels_spatial.detach(), weights.detach())
    while len(_KERNEL_CACHE) > _KERNEL_CACHE_MAXSIZE:
        _KERNEL_CACHE.popitem(last=False)
    return _KERNEL_CACHE[cache_key]


def _fft_conv2d_complex(
    image: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Circular 2D convolution of a real image with a complex kernel.

    image: (H, W) real, kernel: (H, W) complex on the same grid (kernel must
    already be fftshift-centered). Output: (H, W) complex.

    Circular (periodic) padding matches the standard OPC convention where the
    mask is treated as a tile of an infinite layout — eliminates the open-frame
    border artifacts that zero-padding would introduce.
    """
    H, W = image.shape  # noqa: N806
    if kernel.shape != image.shape:
        raise ValueError(
            f"kernel shape {tuple(kernel.shape)} must match image shape {tuple(image.shape)}"
        )
    image_c = image.to(torch.complex64)
    kernel_shifted = torch.fft.ifftshift(kernel, dim=(-2, -1))
    image_f = torch.fft.fft2(image_c)
    kernel_f = torch.fft.fft2(kernel_shifted)
    out: torch.Tensor = torch.fft.ifft2(image_f * kernel_f)
    return out


def simulate_aerial_image_hopkins(
    mask: torch.Tensor,
    params: HopkinsParams | None = None,
    kernels: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    dose: float = 1.0,
    dtype: torch.dtype = torch.float32,
    precomputed_kernels_f: torch.Tensor | None = None,
    memory_plan: SocsMemoryPlan | None = None,
    memory_witness: dict[str, object] | None = None,
) -> torch.Tensor:
    """Simulate aerial image via SOCS-truncated Hopkins imaging.

    Args:
        mask: Real-valued mask (H, W) or (B, 1, H, W), values in [0, 1].
            Differentiable w.r.t. the mask. The SOCS kernels are cached
            detached, so gradients do *not* flow into kernel parameters.
        params: Optical parameters. Required if `kernels`/`weights` are None.
        kernels: Pre-computed complex SOCS kernels (K, H, W). If provided,
            `params` is only used for `dose` (and may be None).
        weights: Pre-computed real weights (K,). Must accompany `kernels`.
        dose: Linear dose multiplier on the resulting intensity.
        dtype: Real dtype of the returned aerial image (``float32`` or
            ``bfloat16``). The internal FFT is always done in
            ``complex64`` because PyTorch's ``fft2`` does not support
            ``bfloat16``-complex; the cast happens before squaring and at
            the output.
        precomputed_kernels_f: Optional pre-FFT'd kernels of shape
            (K, H, W), complex64. When provided, the inner loop skips the
            per-kernel ``ifftshift + fft2`` cost. Must be the FFT of
            ``ifftshift(kernels, dim=(-2,-1))`` for numerical equivalence.
            Coerced to complex64 if a different complex dtype is passed.
        memory_plan: Optional frozen plan forwarded to the bounded SOCS
            construction when kernels must be built (v3 §10 — a formal
            worker passes the driver-owned plan here; it is never
            replanned).
        memory_witness: Optional dict forwarded to the bounded SOCS
            construction for the §16 planned-vs-observed facts.

    Returns:
        Real-valued aerial image with the same spatial shape as `mask`.
    """
    squeezed = False
    if mask.ndim == 2:
        mask4d = mask.unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask4d = mask
    else:
        raise ValueError(f"Expected mask shape (H,W) or (B,1,H,W); got {tuple(mask.shape)}")

    B, _, H, W = mask4d.shape  # noqa: N806
    if H != W:
        raise ValueError(f"Hopkins forward model expects a square grid; got {H}x{W}")

    if kernels is None or weights is None:
        if params is None:
            raise ValueError("Provide either (params) or (kernels and weights).")
        kernels, weights = compute_socs_kernels(
            params, H, mask4d.device, memory_plan=memory_plan, memory_witness=memory_witness
        )

    if params is not None:
        # Tile must be wider than a few Rayleigh units, otherwise the
        # circular FFT convolution wraps optical energy from one edge to
        # the other and contaminates the interior. ~4 Rayleigh
        # (lambda/NA) gives the kernel room to decay before wrapping.
        rayleigh_nm = params.wavelength_nm / max(params.na, 1e-6)
        tile_extent_nm = H * params.pixel_size_nm
        if tile_extent_nm < 4.0 * rayleigh_nm:
            warnings.warn(
                f"Hopkins forward: tile extent {tile_extent_nm:.0f} nm "
                f"({H} px x {params.pixel_size_nm} nm) is smaller than "
                f"4*lambda/NA={4 * rayleigh_nm:.0f} nm; circular FFT "
                f"wraparound will pollute tile edges. Use larger tiles "
                f"or pad before calling.",
                UserWarning,
                stacklevel=2,
            )

    image = mask4d.to(torch.float32).squeeze(1)  # (B, H, W)
    K = kernels.shape[0]  # noqa: N806
    aerial = torch.zeros_like(image)
    kernels_c64 = kernels.to(torch.complex64) if kernels.dtype != torch.complex64 else kernels

    if precomputed_kernels_f is not None:
        if precomputed_kernels_f.dtype != torch.complex64:
            precomputed_kernels_f = precomputed_kernels_f.to(torch.complex64)
        if precomputed_kernels_f.shape[0] != K:
            raise ValueError(
                f"precomputed_kernels_f has K={precomputed_kernels_f.shape[0]}; "
                f"expected {K} matching kernels."
            )

    image_c = image.to(torch.complex64)
    image_f = torch.fft.fft2(image_c)  # (B, H, W)
    for k in range(K):
        if precomputed_kernels_f is not None:
            kernel_f = precomputed_kernels_f[k]
        else:
            kernel_shifted = torch.fft.ifftshift(kernels_c64[k], dim=(-2, -1))
            kernel_f = torch.fft.fft2(kernel_shifted)  # (H, W)
        coherent = torch.fft.ifft2(image_f * kernel_f.unsqueeze(0))  # (B, H, W)
        aerial = aerial + weights[k] * (coherent.real**2 + coherent.imag**2)

    aerial = aerial * dose
    if dtype != torch.float32:
        aerial = aerial.to(dtype)
    if squeezed:
        return aerial.squeeze(0)
    return aerial.unsqueeze(1)


def clear_kernel_cache() -> None:
    """Drop all cached SOCS kernels. Useful in tests and long-running services."""
    _KERNEL_CACHE.clear()
