"""Paper-ready visualization for computational lithography results.

Designed for IEEE / SPIE Advanced Lithography publication style: muted
high-contrast palette, vector-friendly defaults, sensible DPI for camera-ready
PDFs. Requires the ``jupyter`` extra (matplotlib).
"""

from openlithohub.vis.contours import plot_contours, plot_pv_band
from openlithohub.vis.style import IEEE_STYLE, SPIE_STYLE, paper_style

__all__ = [
    "plot_contours",
    "plot_pv_band",
    "paper_style",
    "IEEE_STYLE",
    "SPIE_STYLE",
]
