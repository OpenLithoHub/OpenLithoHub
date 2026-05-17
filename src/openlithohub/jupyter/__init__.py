"""Jupyter/IPython integration for OpenLithoHub.

Usage in Jupyter notebooks:
    %load_ext openlithohub.jupyter

    # Display masks with rich formatting
    from openlithohub.jupyter import display_mask, display_comparison

    display_mask(predicted_mask, title="Optimized Mask")
    display_comparison(predicted, target, title="Pred vs Target")
"""

from __future__ import annotations

from typing import Any

from openlithohub.jupyter.display import display_comparison, display_mask


def load_ipython_extension(ipython: Any) -> None:
    """Called by IPython when running %load_ext openlithohub.jupyter."""
    from openlithohub.jupyter.magics import OpenLithoHubMagics

    ipython.register_magics(OpenLithoHubMagics)


__all__ = ["load_ipython_extension", "display_mask", "display_comparison"]
