"""Paper-publication matplotlib style presets.

Two presets are provided:

- ``IEEE_STYLE`` — IEEE two-column papers (column width ≈ 3.5 in), serif fonts,
  600 dpi raster fallback, vector PDF preferred.
- ``SPIE_STYLE`` — SPIE Advanced Lithography proceedings (single column ≈ 6.5 in),
  Helvetica-like sans, 600 dpi.

Use via the ``paper_style`` context manager so global rcParams are restored.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

IEEE_STYLE: dict[str, Any] = {
    "figure.figsize": (3.5, 2.6),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SPIE_STYLE: dict[str, Any] = {
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Colorblind-safe palette (Tol-bright variant) for target / predicted / PV band.
PALETTE = {
    "target": "#4477AA",  # blue
    "predicted": "#EE6677",  # red
    "pv_outer": "#CCBB44",  # yellow
    "pv_inner": "#228833",  # green
    "shade": "#BBBBBB",
}


@contextmanager
def paper_style(style: str | dict[str, Any] = "ieee") -> Iterator[None]:
    """Temporarily apply a paper-publication matplotlib style.

    Args:
        style: ``"ieee"``, ``"spie"``, or a custom rcParams dict.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for openlithohub.vis. "
            "Install with: pip install openlithohub[jupyter]"
        ) from exc

    if isinstance(style, str):
        if style.lower() == "ieee":
            rc = IEEE_STYLE
        elif style.lower() == "spie":
            rc = SPIE_STYLE
        else:
            raise ValueError(f"unknown style '{style}', expected 'ieee' or 'spie'")
    else:
        rc = style

    with plt.rc_context(rc):
        yield
