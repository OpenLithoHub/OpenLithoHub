"""Standardized I/O for measured sources and pupil aberrations.

Hopkins / SOCS already accepts *parametric* sources (circular, annular,
dipole, quasar) and a clean defocus-only pupil. Real production OPC
needs more:

* **Measured / freeform source intensity** — a TIFF or PNG dumped from
  the scanner illuminator, or a custom shape designed by SMO. This is
  loaded as a 2-D ``[N, N]`` array on normalized pupil coordinates
  ``(σx, σy) ∈ [-1, 1]``.
* **Pupil aberrations as Zernike coefficients** — typically tens of
  Zernikes (Z2..Z37 or higher) measured on the scanner. We parse a
  small text/JSON/CSV file mapping Noll-indexed terms → coefficients
  in waves of OPD, and synthesize a phase map on the same normalized
  pupil grid.

Both outputs are plain ``torch.Tensor`` so they can be plugged into a
custom Hopkins / SOCS run without further conversion.

Coordinate convention: normalized pupil coordinates run from -1 to +1
across the diameter, with the unit circle being the NA-defined edge.
The center pixel is at (σx, σy) = (0, 0). For a square ``N x N``
grid this places the center at index ``(N-1) / 2``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

# Noll → (n, m) for Zernikes 1..37. Reference: Noll (1976), JOSA 66.
# The polynomials below are normalized so ∫ Z² dA / (π R²) = 1 over
# the unit disk (the standard scanner / OPC convention).
_NOLL_TO_NM: dict[int, tuple[int, int]] = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, -2),
    6: (2, 2),
    7: (3, -1),
    8: (3, 1),
    9: (3, -3),
    10: (3, 3),
    11: (4, 0),
    12: (4, 2),
    13: (4, -2),
    14: (4, 4),
    15: (4, -4),
    16: (5, 1),
    17: (5, -1),
    18: (5, 3),
    19: (5, -3),
    20: (5, 5),
    21: (5, -5),
    22: (6, 0),
    23: (6, -2),
    24: (6, 2),
    25: (6, -4),
    26: (6, 4),
    27: (6, -6),
    28: (6, 6),
    29: (7, -1),
    30: (7, 1),
    31: (7, -3),
    32: (7, 3),
    33: (7, -5),
    34: (7, 5),
    35: (7, -7),
    36: (7, 7),
    37: (8, 0),
}


def load_source_intensity(
    path: str | Path,
    *,
    grid_size: int | None = None,
    normalize: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load a measured source intensity image as a square ``[N, N]`` tensor.

    Supports any format Pillow can read: TIFF (16-bit OK), PNG, BMP, etc.
    The image is converted to float32 in the range ``[0, ∞)``, optionally
    resized to ``grid_size`` (bilinear), and optionally normalized so the
    sum is 1 (which is what Hopkins ``J(f)`` expects).

    Args:
        path: Image file.
        grid_size: If given, resize to ``[grid_size, grid_size]`` with
            bilinear interpolation. If ``None``, keep the native shape
            but require it to be square.
        normalize: If True (default), rescale so the source sums to 1.
        device: Torch device for the output tensor.

    Returns:
        Float32 tensor on ``device`` of shape ``[N, N]`` in normalized
        pupil-coordinate layout (origin at the geometric center).
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required to load source intensity images. Install with: pip install Pillow"
        ) from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source intensity image not found: {p}")

    with Image.open(p) as img:
        # 'F' = 32-bit float, preserves dynamic range from 16-bit TIFFs etc.
        arr = torch.from_numpy(_pil_to_float32(img).copy())

    if arr.ndim != 2:
        raise ValueError(f"Source intensity must be 2-D (grayscale); got shape {tuple(arr.shape)}")

    if grid_size is None:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"Source intensity must be square when grid_size is unset; got {tuple(arr.shape)}"
            )
    else:
        arr = (
            torch.nn.functional.interpolate(
                arr.unsqueeze(0).unsqueeze(0),
                size=(grid_size, grid_size),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    arr = arr.clamp(min=0.0)
    if normalize:
        total = arr.sum()
        if total <= 0:
            raise ValueError("Source intensity has total zero/negative — cannot normalize.")
        arr = arr / total

    return arr.to(device=device, dtype=torch.float32).contiguous()


def _pil_to_float32(img: object) -> np.ndarray:
    """PIL Image → numpy float32 array. Handles 16-bit and float TIFFs."""
    import numpy as np
    from PIL import Image

    assert isinstance(img, Image.Image)
    if img.mode in ("F", "I;16", "I"):
        return np.asarray(img, dtype=np.float32)
    return np.asarray(img.convert("F"), dtype=np.float32)


def load_zernike_coefficients(path: str | Path) -> dict[int, float]:
    """Parse a Zernike coefficient file → ``{noll_index: coeff_in_waves}``.

    Three formats are supported, dispatched by extension:

    * ``.json`` — flat object ``{"4": 0.05, "11": -0.02, ...}`` or a
      nested form ``{"zernikes": {"4": 0.05, ...}}``.
    * ``.csv`` — header row required, columns ``noll`` and ``coeff``
      (case-insensitive). Other columns are ignored.
    * ``.txt`` — whitespace-separated ``noll coeff`` pairs, ``#`` comments.

    Coefficients are interpreted as **waves of OPD** (the scanner /
    Synopsys convention). Z1 (piston) is silently dropped — it has no
    optical effect and many vendor tools include it as a sanity column.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Zernike file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        raw = _zernike_from_json(p)
    elif suffix == ".csv":
        raw = _zernike_from_csv(p)
    elif suffix in (".txt", ".dat", ".zer"):
        raw = _zernike_from_txt(p)
    else:
        raise ValueError(f"Unsupported Zernike format: {suffix!r}. Use .json, .csv, or .txt.")

    out: dict[int, float] = {}
    for noll, coeff in raw.items():
        if noll == 1:
            continue
        if noll < 1:
            raise ValueError(f"Noll indices are 1-based; got {noll}.")
        if noll not in _NOLL_TO_NM:
            raise ValueError(
                f"Noll index {noll} is beyond the supported range "
                f"(1..{max(_NOLL_TO_NM)}); add it to _NOLL_TO_NM if needed."
            )
        out[noll] = coeff
    return out


def _zernike_from_json(path: Path) -> dict[int, float]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "zernikes" in obj:
        obj = obj["zernikes"]
    if not isinstance(obj, dict):
        raise ValueError(f"Zernike JSON must be an object; got {type(obj).__name__}.")
    return {int(k): float(v) for k, v in obj.items()}


def _zernike_from_csv(path: Path) -> dict[int, float]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = [h.strip().lower() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"Zernike CSV {path.name} is empty.") from None
        try:
            i_noll = header.index("noll")
            i_coeff = header.index("coeff")
        except ValueError:
            raise ValueError(
                f"Zernike CSV must have 'noll' and 'coeff' columns; got {header}."
            ) from None
        out: dict[int, float] = {}
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            out[int(row[i_noll])] = float(row[i_coeff])
        return out


def _zernike_from_txt(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Zernike .txt line must have 'noll coeff'; got {raw!r}.")
        out[int(parts[0])] = float(parts[1])
    return out


def zernike_phase_map(
    coeffs: dict[int, float],
    grid_size: int,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Synthesize a pupil OPD map from Noll-indexed Zernike coefficients.

    Returns a real ``[N, N]`` tensor of optical path difference in waves
    (multiply by ``2π`` to get phase in radians, or by the wavelength to
    get OPD in nm). Outside the unit pupil the map is set to 0.

    Args:
        coeffs: ``{noll_index: coeff_in_waves}`` (e.g. from
            :func:`load_zernike_coefficients`).
        grid_size: Output grid edge ``N``.
        device: Torch device.
    """
    if grid_size < 2:
        raise ValueError(f"grid_size must be ≥ 2; got {grid_size}.")

    # Normalized pupil coords on [-1, 1].
    axis = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    rho = torch.sqrt(xx * xx + yy * yy)
    theta = torch.atan2(yy, xx)
    inside = rho <= 1.0

    opd = torch.zeros((grid_size, grid_size), device=device, dtype=torch.float32)
    for noll, c in coeffs.items():
        if c == 0.0:
            continue
        n, m = _NOLL_TO_NM[noll]
        z = _zernike_nm(n, m, rho, theta)
        opd = opd + float(c) * z
    return opd * inside.to(opd.dtype)


def _zernike_nm(n: int, m: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Noll-normalized Zernike Z_n^m on (rho, theta).

    Convention: Z is normalized so that ∫₀²π ∫₀¹ Z² ρ dρ dθ = π, which
    matches the scanner / metrology convention (the "engineering" Noll
    normalization).
    """
    import math

    radial = _radial_polynomial(n, abs(m), rho)
    if m > 0:
        ang = torch.cos(m * theta)
        norm = math.sqrt(2 * (n + 1))
    elif m < 0:
        ang = torch.sin(-m * theta)
        norm = math.sqrt(2 * (n + 1))
    else:
        ang = torch.ones_like(theta)
        norm = math.sqrt(n + 1)
    return norm * radial * ang


def _radial_polynomial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
    """R_n^m(ρ), the standard Zernike radial polynomial."""
    import math

    radial = torch.zeros_like(rho)
    if (n - m) % 2 != 0:
        return radial
    for k in range((n - m) // 2 + 1):
        coeff = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
        )
        radial = radial + coeff * rho ** (n - 2 * k)
    return radial
