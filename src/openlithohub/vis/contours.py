"""Paper-ready contour overlays for OPC / ILT / inverse-lithography results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from openlithohub._utils.tensor_ops import ensure_2d
from openlithohub.vis.style import PALETTE, paper_style


def _to_numpy(x: torch.Tensor | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_contours(
    target: torch.Tensor | np.ndarray[Any, Any],
    predicted: torch.Tensor | np.ndarray[Any, Any],
    *,
    pv_band: torch.Tensor | np.ndarray[Any, Any] | None = None,
    pixel_size_nm: float = 1.0,
    title: str | None = None,
    style: str = "ieee",
    save_path: str | Path | None = None,
    show_legend: bool = True,
    close: bool = False,
) -> Any:
    """Render a target / predicted contour overlay with optional PV band.

    Produces a single-panel figure suitable for direct inclusion in IEEE / SPIE
    papers. Contours are extracted with ``matplotlib.contour`` at the 0.5
    iso-level so the figure remains crisp at any DPI.

    Args:
        target: Target binary mask (H, W).
        predicted: Predicted / simulated binary mask (H, W).
        pv_band: Optional PV-band mask where ``> 0`` marks band region.
        pixel_size_nm: Physical pixel pitch (axes are labelled in nm).
        title: Optional figure title.
        style: ``"ieee"`` or ``"spie"``.
        save_path: If given, ``fig.savefig`` is called (vector format inferred
            from extension; ``.pdf`` recommended for camera-ready).
        show_legend: Toggle legend rendering.
        close: When True, the figure is closed via ``plt.close(fig)`` before
            returning ``None``. Use this in batch loops that only call
            ``save_path`` for I/O — without it, matplotlib retains every
            figure in its global registry and the worker leaks memory until
            the process exits.

    Returns:
        The matplotlib ``Figure`` object, or ``None`` if ``close=True``.
    """
    tgt = ensure_2d(torch.as_tensor(_to_numpy(target)))
    pred = ensure_2d(torch.as_tensor(_to_numpy(predicted)))
    tgt_arr = _to_numpy(tgt).astype(np.float32)
    pred_arr = _to_numpy(pred).astype(np.float32)

    h, w = tgt_arr.shape
    extent = (0.0, w * pixel_size_nm, 0.0, h * pixel_size_nm)

    with paper_style(style):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.imshow(
            np.ones_like(tgt_arr),
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=extent,
            origin="lower",
            interpolation="nearest",
        )

        if pv_band is not None:
            band_arr = _to_numpy(pv_band).astype(np.float32)
            ax.contourf(
                band_arr,
                levels=[0.5, 1.5],
                colors=[PALETTE["pv_outer"]],
                alpha=0.35,
                extent=extent,
                origin="lower",
            )

        ax.contour(
            tgt_arr,
            levels=[0.5],
            colors=[PALETTE["target"]],
            linewidths=1.0,
            extent=extent,
            origin="lower",
        )
        ax.contour(
            pred_arr,
            levels=[0.5],
            colors=[PALETTE["predicted"]],
            linewidths=1.0,
            linestyles="--",
            extent=extent,
            origin="lower",
        )

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_aspect("equal")
        if title:
            ax.set_title(title)

        if show_legend:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            handles: list[Any] = [
                Line2D([0], [0], color=PALETTE["target"], lw=1.0, label="Target"),
                Line2D([0], [0], color=PALETTE["predicted"], lw=1.0, ls="--", label="Predicted"),
            ]
            if pv_band is not None:
                handles.append(Patch(facecolor=PALETTE["pv_outer"], alpha=0.35, label="PV band"))
            ax.legend(handles=handles, loc="upper right", frameon=False)

        fig.tight_layout()

        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out)

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
        return None
    return fig


def plot_pv_band(
    nominal: torch.Tensor | np.ndarray[Any, Any],
    inner: torch.Tensor | np.ndarray[Any, Any],
    outer: torch.Tensor | np.ndarray[Any, Any],
    *,
    pixel_size_nm: float = 1.0,
    title: str | None = None,
    style: str = "ieee",
    save_path: str | Path | None = None,
    close: bool = False,
) -> Any:
    """Render an inner / nominal / outer PV-band envelope figure.

    Shows the inner (intersection) and outer (union) resist contours bracketing
    the nominal contour — the standard lithography PV-band figure.

    See :func:`plot_contours` for the ``close`` parameter — set it to True in
    batch loops to avoid leaking matplotlib figures.
    """
    nom_arr = _to_numpy(nominal).astype(np.float32)
    inn_arr = _to_numpy(inner).astype(np.float32)
    out_arr = _to_numpy(outer).astype(np.float32)

    h, w = nom_arr.shape
    extent = (0.0, w * pixel_size_nm, 0.0, h * pixel_size_nm)

    with paper_style(style):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.imshow(
            np.ones_like(nom_arr),
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=extent,
            origin="lower",
            interpolation="nearest",
        )

        band = np.clip(out_arr - inn_arr, 0.0, 1.0)
        ax.contourf(
            band,
            levels=[0.5, 1.5],
            colors=[PALETTE["pv_outer"]],
            alpha=0.35,
            extent=extent,
            origin="lower",
        )
        ax.contour(
            nom_arr,
            levels=[0.5],
            colors=["black"],
            linewidths=1.0,
            extent=extent,
            origin="lower",
        )
        ax.contour(
            inn_arr,
            levels=[0.5],
            colors=[PALETTE["pv_inner"]],
            linewidths=0.7,
            extent=extent,
            origin="lower",
        )
        ax.contour(
            out_arr,
            levels=[0.5],
            colors=[PALETTE["predicted"]],
            linewidths=0.7,
            extent=extent,
            origin="lower",
        )

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_aspect("equal")
        if title:
            ax.set_title(title)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Line2D([0], [0], color="black", lw=1.0, label="Nominal"),
            Line2D([0], [0], color=PALETTE["pv_inner"], lw=0.7, label="Inner"),
            Line2D([0], [0], color=PALETTE["predicted"], lw=0.7, label="Outer"),
            Patch(facecolor=PALETTE["pv_outer"], alpha=0.35, label="PV band"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False)

        fig.tight_layout()
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out)

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
        return None
    return fig
