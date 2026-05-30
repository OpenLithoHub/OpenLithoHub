"""Simplified aerial image forward model using Gaussian PSF convolution.

Padding contract
----------------
All convolutions in this module MUST use circular (periodic) padding, not
zero-padding. This is not stylistic — the Hopkins source-mask formulation
treats the mask as a tile in a periodic illumination, and zero-padding the
input introduces spurious dim-aerial fringes near the frame edge that
silently degrade EPE / PV-band metrics on layouts with features close to
the boundary. If you add a new conv-based simulator here, route it through
``_circular_pad_clamped`` (or an equivalent ``mode="circular"`` call) — do
not switch to ``F.conv2d``'s default zero-pad as a "simplification".
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from openlithohub._constants import THRESHOLD_GENERIC


def _build_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    g1d = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    kernel = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def simulate_aerial_image(
    mask: torch.Tensor,
    sigma_px: float,
    dose: float = 1.0,
) -> torch.Tensor:
    """Simulate aerial image via Gaussian PSF convolution.

    Approximates Hopkins diffraction with a single Gaussian point spread function.

    Accepts ``(H, W)`` for single-image use and ``(B, 1, H, W)`` for batched
    forward passes — the output preserves the input rank.

    Uses circular (periodic) padding to match the Hopkins forward model's
    convention. OPC treats the mask as a tile of an infinite layout, so
    zero-padding at the border would introduce spurious dim-aerial fringes
    that the Hopkins path does not have.
    """
    if sigma_px < 1e-6:
        return mask.float() * dose

    kernel = _build_gaussian_kernel(sigma_px, mask.device)
    padding = kernel.shape[-1] // 2

    squeezed = False
    if mask.ndim == 2:
        inp = mask.float().unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif mask.ndim == 4 and mask.shape[1] == 1:
        inp = mask.float()
    else:
        raise ValueError(f"Expected mask shape (H,W) or (B,1,H,W); got {tuple(mask.shape)}")

    inp_padded = _circular_pad_clamped(inp, padding)
    aerial = functional.conv2d(inp_padded, kernel)
    if squeezed:
        aerial = aerial.squeeze(0).squeeze(0)
    return aerial * dose


def _circular_pad_clamped(inp: torch.Tensor, padding: int) -> torch.Tensor:
    """Circular pad an (N, C, H, W) tensor by ``padding`` on every side.

    PyTorch's circular pad refuses pad sizes >= the corresponding image
    dimension. For very small masks (typical of unit tests) we tile the
    padding in steps that respect that constraint.
    """
    if padding == 0:
        return inp
    out = inp
    remaining_h = padding
    remaining_w = padding
    while remaining_h > 0 or remaining_w > 0:
        cur_h = out.shape[-2]
        cur_w = out.shape[-1]
        step_h = min(remaining_h, cur_h - 1) if remaining_h > 0 else 0
        step_w = min(remaining_w, cur_w - 1) if remaining_w > 0 else 0
        if step_h == 0 and step_w == 0:
            # Image is 1 px wide/tall in some axis — circular pad cannot extend
            # it (PyTorch refuses pad sizes >= dim). The previous behaviour was
            # to fall back to replicate padding with a RuntimeWarning, but
            # `warnings` defaults to "default" filtering (once-per-location) so
            # downstream metrics could pick up replicate-padded edge fringes
            # silently after the first call. Raise instead — every production
            # caller (pvband / stochastic / openilt / levelset_ilt /
            # process_window) feeds layouts orders of magnitude larger than
            # 1 px, so this only triggers on misconfigured inputs that should
            # surface loudly. Issue #10.
            raise ValueError(
                f"_circular_pad_clamped: input shape {tuple(out.shape)} has a "
                "1-pixel-wide axis; circular padding requires every spatial "
                "dim >= 2. Resize the input or pad it to >=2 px before calling "
                "the forward model. See forward_model.py module docstring."
            )
        out = functional.pad(out, (step_w, step_w, step_h, step_h), mode="circular")
        remaining_h -= step_h
        remaining_w -= step_w
    return out


def _gaussian_diffuse(image: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Apply Gaussian diffusion blur with circular padding.

    Used by both the aerial-image forward model and the resist diffusion
    step — sharing this helper keeps the periodic-boundary contract
    consistent. Callers requiring strict bit-reproducibility should
    ensure ``set_deterministic()`` has been called.
    """
    kernel = _build_gaussian_kernel(sigma_px, image.device)
    padding = kernel.shape[-1] // 2
    inp = image.unsqueeze(0).unsqueeze(0)
    inp_padded = _circular_pad_clamped(inp, padding)
    return functional.conv2d(inp_padded, kernel).squeeze(0).squeeze(0)


def apply_resist_threshold(
    aerial_image: torch.Tensor,
    threshold: float = THRESHOLD_GENERIC,
    *,
    resist_diffusion_nm: float = 0.0,
    pixel_size_nm: float = 1.0,
    quencher: float = 0.0,
) -> torch.Tensor:
    """Apply a hard resist threshold to produce a binary resist pattern.

    The 0.5 default is a generic mid-intensity cutoff for ad-hoc use; the
    canonical ICCAD16 / LithoBench cutoff is 0.225 (see
    [Yang2023_LithoBench, §3.2, p.5] and ``SimulatorConfig.threshold``).
    Pass ``threshold=0.225`` when reproducing benchmark numbers.

    With ``resist_diffusion_nm=0.0`` and ``quencher=0.0`` (default) this
    is **constant threshold resist (CTR) without diffusion** — the
    sigmoid-on-aerial simplification documented in
    ``docs/architecture.md → Resist Model Simplification``. The output is
    bit-identical to the legacy ``(aerial >= threshold).float()`` path.

    With a positive ``resist_diffusion_nm`` the aerial image is first
    blurred by a Gaussian whose sigma matches the acid diffusion length,
    then quencher is subtracted, and finally the hard threshold is
    applied. This models a simplified chemically-amplified resist (CAR).

    Real per-node CTR parameters are foundry-confidential and cannot ship
    in an open-source repo; benchmark-relative comparison is unaffected,
    but absolute wafer prediction is not in scope.

    Returns a hard 0/1 tensor — gradients do **not** flow back through
    this function. The README's "end-to-end differentiable" claim refers
    to the ILT optimizer path, which uses
    :func:`openlithohub._utils.resist_model.differentiable_threshold`
    (a temperature-controlled sigmoid). Use that helper for any
    gradient-bearing forward; reserve this hard threshold for
    measurement / scoring code (PVB envelopes, stochastic comparisons,
    leaderboard pass/fail).
    """
    if resist_diffusion_nm <= 0.0 and quencher <= 0.0:
        return (aerial_image >= threshold).float()

    acid = aerial_image.clone()
    sigma_px = resist_diffusion_nm / max(pixel_size_nm, 1e-6)
    if sigma_px > 0.1:
        acid = _gaussian_diffuse(acid, sigma_px)
    acid = (acid - quencher).clamp(min=0.0)
    return (acid >= threshold).float()


def simulate_aerial_image_born(
    mask: torch.Tensor,
    sigma_px: float,
    dose: float = 1.0,
    n_born_terms: int = 2,
    reflectivity: float = 0.1,
) -> torch.Tensor:
    """Simulate aerial image with Born-series scattering correction.

    Extends the single-convolution Hopkins approximation with higher-order
    scattering terms that model thick-mask (3-D EM) effects.  Each Born term
    convolves the residual (difference between mask and previous-order aerial)
    with the same PSF, weighted by *reflectivity*^n:

        I ≈ PSF ⊛ mask + r · PSF ⊛ (mask − PSF ⊛ mask)
            + r² · PSF ⊛ (mask − PSF ⊛ mask − r · PSF ⊛ (…))

    Setting *n_born_terms=1* recovers :func:`simulate_aerial_image`.
    Typical values: *n_born_terms=2* for 193i, *n_born_terms=3* for EUV
    where thick-mask scattering is pronounced.

    Uses circular padding consistent with the rest of this module.

    Args:
        mask: Layout mask ``(H, W)`` or ``(B, 1, H, W)``.
        sigma_px: Gaussian PSF width in pixels.
        dose: Exposure dose multiplier.
        n_born_terms: Number of Born scattering terms (>= 1).
        reflectivity: Coupling strength per scattering order (0–1).

    Returns:
        Aerial image tensor matching input rank.
    """
    if n_born_terms < 1:
        raise ValueError(f"n_born_terms must be >= 1, got {n_born_terms}")

    if sigma_px < 1e-6:
        return mask.float() * dose

    kernel = _build_gaussian_kernel(sigma_px, mask.device)
    padding = kernel.shape[-1] // 2

    squeezed = False
    if mask.ndim == 2:
        inp = mask.float().unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif mask.ndim == 4 and mask.shape[1] == 1:
        inp = mask.float()
    else:
        raise ValueError(f"Expected mask shape (H,W) or (B,1,H,W); got {tuple(mask.shape)}")

    def _conv(x: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(_circular_pad_clamped(x, padding), kernel)

    # First Born term: standard convolution
    aerial = _conv(inp) * dose

    # Higher-order Born corrections
    residual = inp - aerial / dose
    r_power = reflectivity
    for _ in range(1, n_born_terms):
        correction = _conv(residual) * dose * r_power
        aerial = aerial + correction
        residual = inp - aerial / dose
        r_power *= reflectivity

    if squeezed:
        aerial = aerial.squeeze(0).squeeze(0)
    return aerial


def simulate_aerial_image_thick_mask(
    mask: torch.Tensor,
    sigma_px: float,
    dose: float = 1.0,
    *,
    thickness_nm: float = 70.0,
    refractive_index_n: float = 1.0,
    refractive_index_k: float = 0.5,
    wavelength_nm: float = 193.0,
) -> torch.Tensor:
    """Simulate aerial image with thick-mask amplitude/phase perturbation.

    Applies a complex-valued perturbation to the mask before Hopkins
    convolution. The perturbation models the amplitude attenuation and phase
    shift that light accumulates traversing mask material of finite thickness:

        phase_shift  = 2π · t · (n − 1) / λ
        amplitude    = exp(−4π · k · t / λ)
        mask_perturbed = amplitude · mask · exp(j · phase_shift · mask)

    The real part of the convolved complex aerial image is returned.
    Setting *thickness_nm = 0* recovers :func:`simulate_aerial_image`.

    Uses circular padding consistent with the rest of this module.

    Args:
        mask: Layout mask ``(H, W)`` or ``(B, 1, H, W)``.
        sigma_px: Gaussian PSF width in pixels.
        dose: Exposure dose multiplier.
        thickness_nm: Mask material thickness in nanometres.
        refractive_index_n: Real part of the refractive index (phase).
        refractive_index_k: Imaginary part of the refractive index (absorption).
        wavelength_nm: Illumination wavelength in nanometres.

    Returns:
        Aerial image tensor matching input rank.
    """
    if sigma_px < 1e-6:
        return mask.float() * dose

    kernel = _build_gaussian_kernel(sigma_px, mask.device)
    padding = kernel.shape[-1] // 2

    squeezed = False
    if mask.ndim == 2:
        inp = mask.float().unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif mask.ndim == 4 and mask.shape[1] == 1:
        inp = mask.float()
    else:
        raise ValueError(f"Expected mask shape (H,W) or (B,1,H,W); got {tuple(mask.shape)}")

    phase_shift = 2.0 * math.pi * thickness_nm * (refractive_index_n - 1.0) / wavelength_nm
    amplitude = math.exp(-4.0 * math.pi * refractive_index_k * thickness_nm / wavelength_nm)

    mask_complex = amplitude * inp * torch.exp(1j * phase_shift * inp)

    mask_real = mask_complex.real

    aerial = functional.conv2d(_circular_pad_clamped(mask_real, padding), kernel) * dose

    if squeezed:
        aerial = aerial.squeeze(0).squeeze(0)
    return aerial


def simulate_aerial_image_abbe(
    mask: torch.Tensor,
    sigma_px: float,
    dose: float = 1.0,
    *,
    n_source_points: int = 16,
    partial_coherence: float = 0.7,
) -> torch.Tensor:
    """Abbe partial-coherence forward model (reference implementation).

    Sums contributions from multiple coherent source points distributed
    over the illumination pupil. Each source point produces a shifted PSF
    that is convolved with the mask; the intensity contributions are summed.

    This is the high-accuracy reference for thick-mask simulation but is
    **not optimised for production** — it runs O(n_source_points)
    convolutions per forward pass.

    Uses circular padding consistent with the rest of this module.

    Args:
        mask: Layout mask ``(H, W)`` or ``(B, 1, H, W)``.
        sigma_px: Gaussian PSF width in pixels.
        dose: Exposure dose multiplier.
        n_source_points: Number of source points (powers of 2 recommended).
        partial_coherence: Partial coherence factor σ (0 = coherent, 1 =
            fully incoherent).

    Returns:
        Aerial image tensor matching input rank.
    """
    if sigma_px < 1e-6:
        return mask.float() * dose

    kernel_base = _build_gaussian_kernel(sigma_px, mask.device)

    squeezed = False
    if mask.ndim == 2:
        inp = mask.float().unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif mask.ndim == 4 and mask.shape[1] == 1:
        inp = mask.float()
    else:
        raise ValueError(f"Expected mask shape (H,W) or (B,1,H,W); got {tuple(mask.shape)}")

    h, w = inp.shape[-2], inp.shape[-1]
    aerial_total = torch.zeros_like(inp)

    angles = torch.linspace(0, 2 * math.pi, n_source_points + 1, device=inp.device)[:-1]
    for i in range(n_source_points):
        shift_r = partial_coherence * math.cos(angles[i].item())
        shift_c = partial_coherence * math.sin(angles[i].item())

        shift_px_r = shift_r * sigma_px * 0.5
        shift_px_c = shift_c * sigma_px * 0.5

        shifted_kernel = _shift_kernel(kernel_base, shift_px_r, shift_px_c, h, w)
        padding = shifted_kernel.shape[-1] // 2

        field = functional.conv2d(_circular_pad_clamped(inp, padding), shifted_kernel)
        aerial_total = aerial_total + field.pow(2)

    aerial_total = aerial_total / n_source_points * dose

    if squeezed:
        aerial_total = aerial_total.squeeze(0).squeeze(0)
    return aerial_total


def _shift_kernel(
    kernel: torch.Tensor,
    shift_r: float,
    shift_c: float,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Shift a PSF kernel by fractional pixel offsets via phase multiplication.

    Applies a linear phase ramp in Fourier space to shift the kernel
    sub-pixel, then crops back to the original support size.
    """
    k = kernel.squeeze(0).squeeze(0)
    kh, kw = k.shape

    padded = torch.zeros(target_h, target_w, device=kernel.device, dtype=kernel.dtype)
    r_center = target_h // 2
    c_center = target_w // 2
    half_h = kh // 2
    half_w = kw // 2
    r_lo = max(0, r_center - half_h)
    c_lo = max(0, c_center - half_w)
    r_hi = min(target_h, r_center + kh - half_h)
    c_hi = min(target_w, c_center + kw - half_w)
    kr_lo = r_lo - (r_center - half_h)
    kc_lo = c_lo - (c_center - half_w)
    kr_hi = kr_lo + (r_hi - r_lo)
    kc_hi = kc_lo + (c_hi - c_lo)
    padded[r_lo:r_hi, c_lo:c_hi] = k[kr_lo:kr_hi, kc_lo:kc_hi]

    padded_complex = torch.fft.fft2(padded)
    rows = torch.arange(target_h, device=kernel.device, dtype=torch.float32)
    cols = torch.arange(target_w, device=kernel.device, dtype=torch.float32)
    phase_r = torch.exp(-2j * math.pi * shift_r * rows / target_h)
    phase_c = torch.exp(-2j * math.pi * shift_c * cols / target_w)
    phase = phase_r.unsqueeze(1) * phase_c.unsqueeze(0)
    shifted_fft = padded_complex * phase

    shifted_full = torch.fft.ifft2(shifted_fft).real

    cropped = shifted_full[r_lo:r_hi, c_lo:c_hi]

    return cropped.unsqueeze(0).unsqueeze(0)
