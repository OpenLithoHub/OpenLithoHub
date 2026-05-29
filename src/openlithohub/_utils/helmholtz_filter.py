"""Differentiable Helmholtz PDE filter for manufacturing-aware ILT optimization.

Implements the PDE-based Helmholtz filter from topology optimization
(Lazarov & Sigmund 2016) as a pure-PyTorch Jacobi iteration so gradients
flow through the entire mask -> filter -> aerial image -> resist -> loss
chain in a single autograd graph.

The filter solves:

    (I - r^2 * laplacian) rho_filtered = rho

where *r* is the minimum length scale radius.  Features smaller than
approximately ``2 * radius`` are suppressed, giving the optimizer a
smooth manufacturing constraint without heuristic penalty terms.

Circular (periodic) padding is used throughout to match the Hopkins
forward model convention — the mask is treated as a tile in an infinite
periodic layout.  This avoids the spurious edge fringes that zero-padding
or Neumann BCs would introduce near tile boundaries.

Reference
---------
B. S. Lazarov and O. Sigmund, "Filters in topology optimization based on
Helmholtz-type differential equations," *International Journal for
Numerical Methods in Engineering*, vol. 86, no. 6, pp. 765–781, 2011.
Ported from DiffCFD ``diffcfd.geometry.filters.HelmholtzFilter``.
"""

from __future__ import annotations

from torch import Tensor

from openlithohub._utils.forward_model import _circular_pad_clamped


def apply_helmholtz_filter(
    rho: Tensor,
    radius: float = 0.05,
    n_iter: int = 50,
) -> Tensor:
    """Apply a differentiable Helmholtz PDE filter to a density field.

    Solves ``(I - r^2 * laplacian) rho_filtered = rho`` using Jacobi
    iteration with circular padding.  Every operation is a PyTorch tensor
    call so autograd tracks the full computational graph.

    Parameters
    ----------
    rho : torch.Tensor
        Unfiltered density field with shape ``(H, W)``.
    radius : float
        Minimum length scale radius *r*.  Features smaller than ~2*r
        are suppressed.  Assumed to be in the same normalised units as the
        pixel spacing (dx = dy = 1.0).  Default ``0.05``.
    n_iter : int
        Number of Jacobi iterations.  50 is sufficient for convergence
        on typical litho tile sizes (256x256 and above).  Default ``50``.

    Returns
    -------
    torch.Tensor
        Filtered density field with the same shape ``(H, W)`` as the input.

    Notes
    -----
    The Jacobi update for the discrete Helmholtz system with unit grid
    spacing is:

        rho_f = (rho + r^2 * (N_e + N_w + N_n + N_s)) / diag

    where ``diag = 1 + 4*r^2`` and ``N_*`` are the four nearest
    neighbours (obtained via circular padding).
    """
    if rho.ndim != 2:
        raise ValueError(
            f"apply_helmholtz_filter expects a 2-D (H, W) tensor; got shape {tuple(rho.shape)}"
        )

    if radius < 1e-12:
        # Radius effectively zero — identity filter.
        return rho

    r2 = radius**2
    # Unit grid spacing (dx = dy = 1.0) — matches OpenLithoHub's pixel grid.
    diag = 1.0 + 4.0 * r2

    # Prepare 4-D tensor for _circular_pad_clamped which expects (N, C, H, W).
    rho_f = rho.clone()
    dev = rho.device
    dt = rho.dtype

    for _ in range(n_iter):
        # Reshape to (1, 1, H, W) for _circular_pad_clamped.
        rho_f_4d = rho_f.unsqueeze(0).unsqueeze(0)
        padded = _circular_pad_clamped(rho_f_4d, padding=1)
        # padded is (1, 1, H+2, W+2); extract neighbours.
        n_east = padded[0, 0, 1:-1, 2:]  # shift left
        n_west = padded[0, 0, 1:-1, :-2]  # shift right
        n_north = padded[0, 0, 2:, 1:-1]  # shift down
        n_south = padded[0, 0, :-2, 1:-1]  # shift up

        # Jacobi update: rho_f^{k+1} = (rhs) / diag
        # where rhs = rho + r^2 * (sum of neighbours)
        rho_f = (rho + r2 * (n_east + n_west + n_north + n_south)) / diag

    return rho_f.to(dtype=dt, device=dev)
