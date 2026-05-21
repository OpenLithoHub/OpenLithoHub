"""Shared HTML rendering helpers for lithography result dataclasses.

The functions here are imported by ``_repr_html_`` methods on result
dataclasses (PredictionResult, MRCResult, CurvilinearMRCResult,
DRCResult, MonteCarloFailureResult, SimulatorResult). They produce
a self-contained HTML fragment — no external CSS, no JS — so the
output renders identically in classic Jupyter, JupyterLab, VS Code,
and Colab.

Design notes:
- matplotlib is an optional dep. Every helper degrades gracefully to
  text when matplotlib or numpy is missing — never raise from a repr.
- Images are inlined as base64 PNG so the HTML survives notebook export.
- Hotspot overlays draw red on top of a grey mask thumbnail so the
  reader can locate violations without leaving the cell.
"""

from __future__ import annotations

import base64
import html
import io
from typing import Any

import torch


def _to_numpy(t: Any) -> Any:
    """Best-effort tensor → numpy without forcing imports at module load."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'background:{color};color:white;font-weight:600;font-size:90%;">'
        f"{html.escape(text)}</span>"
    )


def pass_fail_badge(passed: bool) -> str:
    return _badge("PASS", "#1f9d55") if passed else _badge("FAIL", "#cc1f1a")


def mask_thumbnail_png_b64(
    mask: torch.Tensor | None,
    *,
    max_size: int = 256,
    hotspots: list[tuple[float, float]] | None = None,
) -> str | None:
    """Render a mask as a base64 PNG with optional red hotspot dots.

    Returns ``None`` if matplotlib is unavailable or the mask is empty —
    callers should fall back to text in that case.
    """
    if mask is None:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    arr = _to_numpy(mask)
    if arr is None or getattr(arr, "ndim", 0) < 2:
        return None
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    h, w = arr.shape
    scale = min(1.0, max_size / max(h, w))
    fig_w = max(2.0, w * scale / 64.0)
    fig_h = max(2.0, h * scale / 64.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=72)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if hotspots:
        ys = [p[0] for p in hotspots]
        xs = [p[1] for p in hotspots]
        ax.scatter(xs, ys, s=18, facecolors="none", edgecolors="#ff2d2d", linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, dpi=72)
    plt.close(fig)
    _ = np  # silence unused
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64_to_img_tag(b64: str | None, *, alt: str = "") -> str:
    if not b64:
        return ""
    return (
        f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" '
        'style="image-rendering:pixelated;max-width:100%;border:1px solid #ddd;'
        'border-radius:4px;" />'
    )


def kv_table(rows: list[tuple[str, str]]) -> str:
    """Build a small two-column key/value table."""
    body = "".join(
        f'<tr><td style="padding:2px 8px;color:#555;">{html.escape(k)}</td>'
        f'<td style="padding:2px 8px;font-family:monospace;">{html.escape(v)}</td></tr>'
        for k, v in rows
    )
    return f'<table style="border-collapse:collapse;font-size:90%;margin-top:6px;">{body}</table>'


def violation_table(violations: list[dict[str, Any]], *, max_rows: int = 5) -> str:
    """Render a violation list as a compact HTML table; truncates after max_rows.

    Columns are the union of keys across all violations (heterogeneous shapes
    from mixed rule types — e.g. DRC width vs area — render as ``—`` instead
    of disappearing because ``violations[0]`` happened to be a width row).
    Known columns are emitted in a stable order; unknown keys are appended.
    """
    if not violations:
        return ""
    preferred_order = (
        "rule",
        "type",
        "type_code",
        "x_nm",
        "y_nm",
        "actual_nm",
        "required_nm",
        "actual_nm2",
        "required_nm2",
        "threshold_nm",
    )
    seen: set[str] = set()
    for v in violations:
        seen.update(v.keys())
    columns = [c for c in preferred_order if c in seen]
    columns.extend(sorted(c for c in seen if c not in preferred_order))
    head = "".join(
        f'<th style="text-align:left;padding:2px 8px;border-bottom:1px solid #ccc;">'
        f"{html.escape(str(c))}</th>"
        for c in columns
    )
    body_rows = []
    for v in violations[:max_rows]:
        cells = "".join(
            f'<td style="padding:2px 8px;font-family:monospace;">'
            f"{html.escape(_fmt_cell(v[c]) if c in v else '—')}</td>"
            for c in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    truncated = ""
    if len(violations) > max_rows:
        truncated = (
            f'<div style="font-size:85%;color:#888;padding:2px 8px;">'
            f"… and {len(violations) - max_rows} more violations</div>"
        )
    return (
        '<table style="border-collapse:collapse;font-size:85%;margin-top:6px;">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        f"{truncated}"
    )


def _fmt_cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def panel(*, title: str, header_html: str, body_html: str) -> str:
    """Wrap a result in a labelled panel with consistent styling."""
    return (
        '<div style="border:1px solid #ddd;border-radius:6px;padding:10px;'
        "margin:4px 0;font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
        'max-width:680px;">'
        f'<div style="font-weight:600;font-size:95%;margin-bottom:4px;">'
        f"{html.escape(title)} {header_html}</div>"
        f"{body_html}</div>"
    )


def hotspot_pixels_to_points(
    violations: list[dict[str, Any]],
    *,
    pixel_size_nm: float = 1.0,
    max_points: int = 50,
) -> list[tuple[float, float]]:
    """Pull (y_px, x_px) from violation dicts. Accepts either pixel
    coords (``y_px``/``x_px``) or nm coords (``y_nm``/``x_nm``)."""
    points: list[tuple[float, float]] = []
    for v in violations[:max_points]:
        if "y_px" in v and "x_px" in v:
            points.append((float(v["y_px"]), float(v["x_px"])))
        elif "y_nm" in v and "x_nm" in v and pixel_size_nm > 0:
            points.append((float(v["y_nm"]) / pixel_size_nm, float(v["x_nm"]) / pixel_size_nm))
    return points
