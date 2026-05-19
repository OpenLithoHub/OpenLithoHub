"""OpenLithoHub Playground — Interactive web demo for computational lithography evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Upper bound on the longest side of an uploaded mask. EPE uses a distance
# transform on the GPU/CPU tensor, so memory grows with W*H. 1024 keeps a
# single evaluation comfortably under 1 GB on the HF free 16 GB container.
MAX_UPLOAD_DIM = int(os.environ.get("OPENLITHOHUB_MAX_UPLOAD_DIM", "1024"))

# Monkeypatch gradio_client.utils to handle bool schemas (Gradio 4.44 bug)
# https://github.com/gradio-app/gradio/issues/10662
# E402 is unavoidable here: the patch must run before `import gradio` so that
# gradio_client.utils is replaced before gradio caches its references.
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
_orig_get_type = _gc_utils.get_type


def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


def _patched_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
_gc_utils.get_type = _patched_get_type

import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from openlithohub.benchmark.compliance.mrc import check_mrc as _olh_check_mrc  # noqa: E402
from openlithohub.benchmark.metrics.epe import _extract_edges as _olh_extract_edges  # noqa: E402
from openlithohub.benchmark.metrics.epe import compute_epe as _olh_compute_epe  # noqa: E402

# ---------------------------------------------------------------------------
# Metric adapters — thin numpy → torch wrappers around the canonical
# openlithohub implementations so the Space and the CLI/leaderboard always
# report identical numbers.
# ---------------------------------------------------------------------------


def _extract_edges(binary: np.ndarray) -> np.ndarray:
    edges = _olh_extract_edges(torch.from_numpy(binary.astype(np.float32)))
    return edges.numpy().astype(np.float32)


def compute_epe(predicted: np.ndarray, target: np.ndarray, pixel_size_nm: float = 1.0) -> dict:
    return _olh_compute_epe(
        torch.from_numpy(predicted.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
        pixel_size_nm=pixel_size_nm,
    )


def check_mrc(
    mask: np.ndarray,
    min_width_nm: float = 40.0,
    min_spacing_nm: float = 40.0,
    pixel_size_nm: float = 1.0,
) -> dict:
    result = _olh_check_mrc(
        torch.from_numpy(mask.astype(np.float32)),
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
        pixel_size_nm=pixel_size_nm,
    )
    return {
        "passed": result.passed,
        "violation_count": result.violation_count,
        "violation_rate": result.violation_rate,
        "width_violations": result.width_violation_count,
        "spacing_violations": result.spacing_violation_count,
    }


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------


def generate_line_space(size: int = 256, pitch_px: int = 20, duty: float = 0.5) -> np.ndarray:
    """Generate a line/space pattern."""
    mask = np.zeros((size, size), dtype=np.float32)
    line_width = int(pitch_px * duty)
    for x in range(0, size, pitch_px):
        mask[:, x : x + line_width] = 1.0
    return mask


def generate_contact_holes(size: int = 256, hole_size: int = 10, pitch: int = 40) -> np.ndarray:
    """Generate a contact hole array pattern."""
    mask = np.ones((size, size), dtype=np.float32)
    for y in range(pitch // 2, size, pitch):
        for x in range(pitch // 2, size, pitch):
            y0, y1 = max(0, y - hole_size // 2), min(size, y + hole_size // 2)
            x0, x1 = max(0, x - hole_size // 2), min(size, x + hole_size // 2)
            mask[y0:y1, x0:x1] = 0.0
    return mask


def generate_sram(size: int = 256) -> np.ndarray:
    """Generate an SRAM-like pattern with varied features."""
    mask = np.zeros((size, size), dtype=np.float32)
    # Horizontal lines
    for y in range(20, size - 20, 40):
        mask[y : y + 8, 10 : size - 10] = 1.0
    # Vertical connections
    for x in range(30, size - 30, 60):
        for y in range(20, size - 40, 80):
            mask[y : y + 40, x : x + 6] = 1.0
    # Contact pads
    for y in range(40, size - 40, 80):
        for x in range(50, size - 50, 80):
            mask[y - 5 : y + 5, x - 5 : x + 5] = 1.0
    return mask


def generate_random_logic(size: int = 256, *, seed: int = 7) -> np.ndarray:
    """Manhattan random-logic routing on a coarse grid (back-end-of-line look)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=np.float32)
    grid = 16
    for gy in range(grid // 2, size, grid):
        for gx in range(grid // 2, size, grid):
            roll = rng.random()
            if roll < 0.35:
                length = rng.integers(8, 28)
                width = rng.integers(2, 5)
                x0 = max(0, gx - length // 2)
                x1 = min(size, gx + length // 2)
                y0 = max(0, gy - width // 2)
                y1 = min(size, gy + width // 2)
                mask[y0:y1, x0:x1] = 1.0
            elif roll < 0.65:
                length = rng.integers(8, 28)
                width = rng.integers(2, 5)
                y0 = max(0, gy - length // 2)
                y1 = min(size, gy + length // 2)
                x0 = max(0, gx - width // 2)
                x1 = min(size, gx + width // 2)
                mask[y0:y1, x0:x1] = 1.0
            elif roll < 0.72:
                via = 4
                y0 = max(0, gy - via // 2)
                y1 = min(size, gy + via // 2)
                x0 = max(0, gx - via // 2)
                x1 = min(size, gx + via // 2)
                mask[y0:y1, x0:x1] = 1.0
    return mask


PATTERN_GENERATORS = {
    "Line/Space": generate_line_space,
    "Contact Holes": generate_contact_holes,
    "SRAM-like": generate_sram,
    "Random Logic": generate_random_logic,
}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_masks(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    pixel_size_nm: float = 1.0,
    min_width_nm: float = 40.0,
    min_spacing_nm: float = 40.0,
) -> plt.Figure:
    """5-panel visualization: target, predicted, edge overlay, EPE heatmap, MRC overlay."""
    from openlithohub.vis import plot_epe_heatmap, plot_mrc_overlay

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))

    axes[0].imshow(target, cmap="gray", interpolation="nearest")
    axes[0].set_title("Target (Design)")
    axes[0].axis("off")

    axes[1].imshow(predicted, cmap="gray", interpolation="nearest")
    axes[1].set_title("Predicted (Mask)")
    axes[1].axis("off")

    # Edge overlay
    pred_edges = _extract_edges(predicted)
    tgt_edges = _extract_edges(target)
    overlay = np.zeros((*target.shape, 3), dtype=np.float32)
    overlay[tgt_edges > 0] = [0.0, 1.0, 0.0]  # green = target edges
    overlay[pred_edges > 0] = [1.0, 0.0, 0.0]  # red = predicted edges
    both = (pred_edges > 0) & (tgt_edges > 0)
    overlay[both] = [1.0, 1.0, 0.0]  # yellow = overlap

    axes[2].imshow(overlay, interpolation="nearest")
    axes[2].set_title("Edge Overlay (G=Tgt, R=Pred)")
    axes[2].axis("off")

    plot_epe_heatmap(predicted, target, pixel_size_nm=pixel_size_nm, ax=axes[3])
    plot_mrc_overlay(
        predicted,
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
        pixel_size_nm=pixel_size_nm,
        ax=axes[4],
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio interface functions
# ---------------------------------------------------------------------------


def evaluate_pattern(
    pattern_type: str,
    noise_level: float,
    pixel_size_nm: float,
    min_width_nm: float,
    min_spacing_nm: float,
):
    """Generate pattern, add noise as 'predicted', compute metrics."""
    generator = PATTERN_GENERATORS[pattern_type]
    target = generator(size=256)

    # Simulate an imperfect prediction by adding noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_level, target.shape).astype(np.float32)
    predicted = np.clip(target + noise, 0, 1)
    predicted = (predicted > 0.5).astype(np.float32)

    # Compute metrics
    epe = compute_epe(predicted, target, pixel_size_nm=pixel_size_nm)
    mrc = check_mrc(
        predicted,
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
        pixel_size_nm=pixel_size_nm,
    )

    # Visualization
    fig = visualize_masks(
        predicted,
        target,
        pixel_size_nm=pixel_size_nm,
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
    )

    metrics_text = (
        f"## Evaluation Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| EPE Mean | {epe['epe_mean_nm']:.3f} nm |\n"
        f"| EPE Max | {epe['epe_max_nm']:.3f} nm |\n"
        f"| EPE Std | {epe['epe_std_nm']:.3f} nm |\n"
        f"| MRC Passed | {'Yes' if mrc['passed'] else 'No'} |\n"
        f"| Width Violations | {mrc['width_violations']} |\n"
        f"| Spacing Violations | {mrc['spacing_violations']} |\n"
        f"| Violation Rate | {mrc['violation_rate']:.6f} |\n"
    )

    return fig, metrics_text


def evaluate_uploaded(
    pred_file,
    target_file,
    pixel_size_nm: float,
    min_width_nm: float,
    min_spacing_nm: float,
):
    """Evaluate uploaded mask images."""
    from PIL import Image

    from openlithohub._utils.auto_crop import auto_crop

    if pred_file is None or target_file is None:
        return None, "Please upload both predicted and target mask images."

    pred_img = Image.open(pred_file)
    tgt_img = Image.open(target_file)

    src_w, src_h = pred_img.size

    pred_img = pred_img.convert("L")
    tgt_img = tgt_img.convert("L")

    # Resize to match if different
    if pred_img.size != tgt_img.size:
        tgt_img = tgt_img.resize(pred_img.size, Image.NEAREST)

    predicted = (np.array(pred_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    target = (np.array(tgt_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

    # Auto-Crop: if either axis exceeds MAX_UPLOAD_DIM, locate the densest
    # MAX_UPLOAD_DIM-square window on the predicted mask and crop both tensors
    # at the same bbox. Keeps EPE on the user's actual area of interest
    # instead of bailing out, and stays within the HF free-tier memory budget.
    crop_notice = ""
    if max(predicted.shape) > MAX_UPLOAD_DIM:
        pred_t = torch.from_numpy(predicted)
        _, bbox = auto_crop(pred_t, target_size=MAX_UPLOAD_DIM)
        y0, x0, y1, x1 = bbox
        predicted = predicted[y0:y1, x0:x1]
        target = target[y0:y1, x0:x1]
        crop_notice = (
            f"\n\n*Auto-cropped from {src_w}×{src_h} to "
            f"{x1 - x0}×{y1 - y0} at bbox y={y0}..{y1}, x={x0}..{x1} "
            f"(densest window).*"
        )

    epe = compute_epe(predicted, target, pixel_size_nm=pixel_size_nm)
    mrc = check_mrc(
        predicted,
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
        pixel_size_nm=pixel_size_nm,
    )

    fig = visualize_masks(
        predicted,
        target,
        pixel_size_nm=pixel_size_nm,
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
    )

    metrics_text = (
        f"## Evaluation Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| EPE Mean | {epe['epe_mean_nm']:.3f} nm |\n"
        f"| EPE Max | {epe['epe_max_nm']:.3f} nm |\n"
        f"| EPE Std | {epe['epe_std_nm']:.3f} nm |\n"
        f"| MRC Passed | {'Yes' if mrc['passed'] else 'No'} |\n"
        f"| Width Violations | {mrc['width_violations']} |\n"
        f"| Spacing Violations | {mrc['spacing_violations']} |\n"
        f"| Violation Rate | {mrc['violation_rate']:.6f} |\n" + crop_notice
    )

    return fig, metrics_text


# ---------------------------------------------------------------------------
# Leaderboard view
# ---------------------------------------------------------------------------


def _leaderboard_path() -> Path:
    env = os.environ.get("OPENLITHOHUB_LEADERBOARD_PATH")
    if env:
        return Path(env)
    here = Path(__file__).parent
    candidates = [
        here / "leaderboard.json",
        Path.home() / ".openlithohub" / "leaderboard.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_leaderboard():
    """Read the JSON leaderboard. Returns ``(rows, status_md)``."""
    path = _leaderboard_path()
    if not path.exists():
        return [], (
            "_No leaderboard entries yet. Submit your model via "
            "`openlithohub submit` — see the [submission guide]"
            "(https://github.com/OpenLithoHub/OpenLithoHub#leaderboard)._"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], f"_Failed to parse leaderboard: {exc}_"

    entries = data.get("entries", [])
    rows = []
    for e in entries:
        rows.append(
            [
                e.get("model_name", ""),
                e.get("dataset", ""),
                e.get("process_node", ""),
                e.get("mask_topology", ""),
                e.get("epe_mean_nm"),
                e.get("epe_max_nm"),
                e.get("pvband_mean_nm"),
                e.get("pvband_max_nm"),
                e.get("shot_count"),
                e.get("paper_url") or e.get("code_url") or "",
            ]
        )
    rows.sort(key=lambda r: (r[4] is None, r[4]))
    status = f"_{len(rows)} submission(s) — sorted by EPE mean (lower is better)._"
    return rows, status


# ---------------------------------------------------------------------------
# Built-in preset examples (committed to spaces/examples/)
# ---------------------------------------------------------------------------

# Source of truth for the demo PNGs is ``scripts/generate_demo_samples.py``.
# Shipping them under spaces/examples/ avoids the prior tempdir-on-cold-start
# fragility on HF Space and gives users browseable inputs in the repo.
_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

_PRESET_SAMPLES: list[tuple[str, str, float, float, float]] = [
    ("line_space", "Line/Space", 1.0, 10.0, 10.0),
    ("contact_holes", "Contact Holes", 1.0, 10.0, 10.0),
    ("sram_like", "SRAM-like", 1.0, 10.0, 10.0),
    ("random_logic", "Random Logic", 1.0, 10.0, 10.0),
]


def _get_upload_examples() -> list[list[str | float]]:
    """Return Upload-tab examples as [pred, target, px_nm, mw_nm, ms_nm] rows.

    Missing PNGs (e.g., a checkout without scripts/generate_demo_samples.py
    output) are silently skipped — the Space stays up.
    """
    rows: list[list[str | float]] = []
    for slug, _label, px, mw, ms in _PRESET_SAMPLES:
        pred = _EXAMPLES_DIR / f"{slug}_pred.png"
        tgt = _EXAMPLES_DIR / f"{slug}_target.png"
        if pred.exists() and tgt.exists():
            rows.append([str(pred), str(tgt), px, mw, ms])
    return rows


def _get_pattern_examples() -> list[list[str | float]]:
    """Return Synthetic-tab examples as [pattern, noise, px_nm, mw_nm, ms_nm] rows."""
    return [[label, 0.10, px, mw, ms] for _slug, label, px, mw, ms in _PRESET_SAMPLES]


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------


# Tab bar contrast fix — Gradio Soft theme renders unselected tabs in a pale
# gray that fails WCAG AA on light backgrounds. Darken unselected labels and
# mark the selected tab with the OpenLithoHub brand blue used on the website.
_TAB_CSS = """
.tab-nav { border-bottom: 1px solid #c6c6cd; }
.tab-nav button {
    color: #45464d;
    font-weight: 600;
    opacity: 1;
}
.tab-nav button:hover { color: #0058be; }
.tab-nav button.selected {
    color: #0058be;
    border-bottom: 2px solid #0058be;
}
"""

with gr.Blocks(
    title="OpenLithoHub Playground",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan"),
    css=_TAB_CSS,
) as demo:
    gr.Markdown(
        """
        # OpenLithoHub Playground
        **Interactive evaluation for computational lithography models**

        Compute Edge Placement Error (EPE), MRC compliance, and visualize mask quality.
        """
    )

    with gr.Tabs():
        # Tab 1: Synthetic pattern evaluation
        with gr.TabItem("Synthetic Patterns"):
            gr.Markdown("Generate synthetic test patterns and evaluate with simulated noise.")
            with gr.Row():
                with gr.Column(scale=1):
                    pattern_type = gr.Dropdown(
                        choices=list(PATTERN_GENERATORS.keys()),
                        value="Line/Space",
                        label="Pattern Type",
                    )
                    noise_level = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="Noise Level")
                    pixel_size = gr.Number(value=1.0, label="Pixel Size (nm)")
                    min_width = gr.Number(value=10.0, label="Min Width (nm)")
                    min_spacing = gr.Number(value=10.0, label="Min Spacing (nm)")
                    eval_btn = gr.Button("Evaluate", variant="primary")

                with gr.Column(scale=2):
                    plot_output = gr.Plot(label="Visualization")
                    metrics_output = gr.Markdown()

            eval_btn.click(
                fn=evaluate_pattern,
                inputs=[pattern_type, noise_level, pixel_size, min_width, min_spacing],
                outputs=[plot_output, metrics_output],
            )

            gr.Examples(
                examples=_get_pattern_examples(),
                inputs=[pattern_type, noise_level, pixel_size, min_width, min_spacing],
                label="Try a preset",
                examples_per_page=4,
            )

        # Tab 2: Upload evaluation
        with gr.TabItem("Upload Masks"):
            gr.Markdown(
                "Upload your own predicted and target mask images (grayscale, thresholded at 50%)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    pred_upload = gr.Image(type="filepath", label="Predicted Mask")
                    tgt_upload = gr.Image(type="filepath", label="Target Mask")
                    px_size_upload = gr.Number(value=1.0, label="Pixel Size (nm)")
                    mw_upload = gr.Number(value=40.0, label="Min Width (nm)")
                    ms_upload = gr.Number(value=40.0, label="Min Spacing (nm)")
                    upload_btn = gr.Button("Evaluate", variant="primary")

                with gr.Column(scale=2):
                    upload_plot = gr.Plot(label="Visualization")
                    upload_metrics = gr.Markdown()

            upload_btn.click(
                fn=evaluate_uploaded,
                inputs=[pred_upload, tgt_upload, px_size_upload, mw_upload, ms_upload],
                outputs=[upload_plot, upload_metrics],
            )

            gr.Examples(
                examples=_get_upload_examples(),
                inputs=[pred_upload, tgt_upload, px_size_upload, mw_upload, ms_upload],
                label="Try a preset",
                examples_per_page=4,
            )

        # Tab 3: Leaderboard
        with gr.TabItem("Leaderboard"):
            gr.Markdown(
                """
                ## Community SOTA Leaderboard

                Snapshot of community-submitted benchmark results, sorted by mean EPE.
                Submissions go through `openlithohub submit` against the published
                LithoBench / LithoSim splits — see the
                [submission guide](https://github.com/OpenLithoHub/OpenLithoHub#leaderboard).
                """
            )
            lb_status = gr.Markdown()
            lb_table = gr.Dataframe(
                headers=[
                    "Model",
                    "Dataset",
                    "Node",
                    "Topology",
                    "EPE mean (nm)",
                    "EPE max (nm)",
                    "PV band mean (nm)",
                    "PV band max (nm)",
                    "Shot count",
                    "Reference",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "str",
                ],
                interactive=False,
                wrap=True,
            )
            refresh_btn = gr.Button("Refresh", variant="secondary")

            def _load():
                rows, status = load_leaderboard()
                return rows, status

            demo.load(fn=_load, inputs=None, outputs=[lb_table, lb_status])
            refresh_btn.click(fn=_load, inputs=None, outputs=[lb_table, lb_status])

    gr.Markdown(
        """
        ---
        **OpenLithoHub** | [GitHub](https://github.com/OpenLithoHub/OpenLithoHub) |
        [Docs](https://docs.openlithohub.com) |
        [Leaderboard](https://openlithohub.com/leaderboard) |
        Apache 2.0 License
        """
    )

if __name__ == "__main__":
    demo.launch()
