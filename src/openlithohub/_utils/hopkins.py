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

    def cache_key(self, grid_size: int, device: str, kernel_dtype: str) -> Hashable:
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


def compute_socs_kernels(
    params: HopkinsParams,
    grid_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SOCS kernels and their weights for a square grid.

    Args:
        params: Optical parameters.
        grid_size: Square grid edge length (pixels).
        device: PyTorch device.
        dtype: Complex dtype of the returned kernels (``complex64`` or
            ``complex128``). The internal FFT/SVD always runs in
            ``complex64`` — this dtype only controls the cached output.

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
    cache_key = params.cache_key(grid_size, str(dev), str(dtype))
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        _KERNEL_CACHE.move_to_end(cache_key)
        return cached

    f = _frequency_grid(grid_size, params.pixel_size_nm, dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")

    pupil = _pupil(fx, fy, params)

    src_shifts, src_weights = _illumination_samples(params, grid_size, dev)
    n_src = src_shifts.shape[0]

    n_freq = grid_size * grid_size
    # SVD matrix is (n_src, n_freq) complex64 — warn on large grids where
    # memory and compute time grow as O(n_src * grid_size^2).
    mem_gb = n_src * n_freq * 8 / 1e9
    if mem_gb > 4.0:
        warnings.warn(
            f"SOCS kernel computation for grid_size={grid_size} allocates ~{mem_gb:.1f} GB. "
            f"Consider using a smaller grid or reducing num_kernels.",
            stacklevel=2,
        )

    yy, xx = torch.meshgrid(
        torch.arange(grid_size, device=dev),
        torch.arange(grid_size, device=dev),
        indexing="ij",
    )
    H = torch.zeros((n_src, n_freq), dtype=torch.complex64, device=dev)  # noqa: N806
    for k in range(n_src):
        sy = int(src_shifts[k, 0].item())
        sx = int(src_shifts[k, 1].item())
        idx_y = (yy - sy) % grid_size
        idx_x = (xx - sx) % grid_size
        shifted = pupil[idx_y, idx_x]
        weight = torch.sqrt(src_weights[k])
        H[k] = (shifted * weight).reshape(-1)

    K = max(1, min(params.num_kernels, n_src))  # noqa: N806
    u, s, vh = torch.linalg.svd(H, full_matrices=False)
    s2 = (s**2)[:K]
    eigvecs = vh[:K]

    kernels_freq = eigvecs.reshape(K, grid_size, grid_size)
    weights = s2.to(torch.float32)

    kernels_spatial = torch.fft.ifft2(kernels_freq, norm="backward")
    kernels_spatial = torch.fft.fftshift(kernels_spatial, dim=(-2, -1))

    # Calibrate so that an open-frame (all-ones) mask produces aerial ≈ 1.
    # For a constant mask, coherent_k = sum(kernel_k); aerial_open = Σ_k w_k |sum(k_k)|².
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
) -> torch.Tensor:
    """Simulate aerial image via SOCS-truncated Hopkins imaging.

    Args:
        mask: Real-valued mask (H, W) or (B, 1, H, W), values in [0, 1].
            Differentiable: gradients flow back through the kernels.
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
        kernels, weights = compute_socs_kernels(params, H, mask4d.device)

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
