"""OpenLithoHub Playground — Interactive web demo for computational lithography evaluation."""

from __future__ import annotations

import json
import os
import tempfile
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
    width_violations = sum(1 for v in result.violations if v.get("type_code") == 0.0)
    spacing_violations = sum(1 for v in result.violations if v.get("type_code") == 1.0)
    return {
        "passed": result.passed,
        "violation_count": result.violation_count,
        "violation_rate": result.violation_rate,
        "width_violations": width_violations,
        "spacing_violations": spacing_violations,
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


PATTERN_GENERATORS = {
    "Line/Space": generate_line_space,
    "Contact Holes": generate_contact_holes,
    "SRAM-like": generate_sram,
}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_masks(predicted: np.ndarray, target: np.ndarray) -> plt.Figure:
    """Create side-by-side visualization of predicted vs target with edge overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

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
    axes[2].set_title("Edge Overlay (Green=Target, Red=Pred)")
    axes[2].axis("off")

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
    fig = visualize_masks(predicted, target)

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

    if pred_file is None or target_file is None:
        return None, "Please upload both predicted and target mask images."

    pred_img = Image.open(pred_file).convert("L")
    tgt_img = Image.open(target_file).convert("L")

    longest = max(pred_img.size[0], pred_img.size[1], tgt_img.size[0], tgt_img.size[1])
    if longest > MAX_UPLOAD_DIM:
        return None, (
            f"Uploaded mask is {longest}px on its longest side; the playground "
            f"caps inputs at {MAX_UPLOAD_DIM}px to keep evaluation under the free "
            f"HF Space memory budget. Downsample or crop your mask and retry."
        )

    # Resize to match if different
    if pred_img.size != tgt_img.size:
        tgt_img = tgt_img.resize(pred_img.size, Image.NEAREST)

    predicted = (np.array(pred_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    target = (np.array(tgt_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

    epe = compute_epe(predicted, target, pixel_size_nm=pixel_size_nm)
    mrc = check_mrc(
        predicted,
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
        pixel_size_nm=pixel_size_nm,
    )

    fig = visualize_masks(predicted, target)

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
# Upload-tab example generation
# ---------------------------------------------------------------------------


def _build_upload_examples() -> list[list[str]]:
    """Materialize the synthetic patterns as PNG (pred, target) pairs.

    Failures here must not crash app startup — on the HF Space cold path,
    tempdir or PIL hiccups would otherwise leave the Space dead before
    Gradio mounts any routes. Caller fans out to an empty list, which
    Gradio renders as "no examples available" rather than a 500.

    Returns rows shaped to match the Upload tab inputs:
    [pred_path, target_path, pixel_size_nm, min_width_nm, min_spacing_nm].
    """
    from PIL import Image

    out_dir = Path(tempfile.gettempdir()) / "olh_upload_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows: list[list[str]] = []
    for name, gen in PATTERN_GENERATORS.items():
        target = gen(size=256)
        noise = rng.normal(0, 0.15, target.shape).astype(np.float32)
        predicted = ((target + noise) > 0.5).astype(np.float32)

        slug = name.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        pred_path = out_dir / f"{slug}_pred.png"
        tgt_path = out_dir / f"{slug}_target.png"
        Image.fromarray((predicted * 255).astype(np.uint8)).save(pred_path)
        Image.fromarray((target * 255).astype(np.uint8)).save(tgt_path)
        rows.append([str(pred_path), str(tgt_path), 1.0, 40.0, 40.0])
    return rows


_upload_examples_cache: list[list[str]] | None = None


def _get_upload_examples() -> list[list[str]]:
    """Lazy, fault-tolerant accessor for the upload examples list."""
    global _upload_examples_cache
    if _upload_examples_cache is None:
        try:
            _upload_examples_cache = _build_upload_examples()
        except Exception as exc:  # noqa: BLE001 — don't crash the Space on cold-start
            print(f"OpenLithoHub: failed to build upload examples ({exc!r}); skipping.")
            _upload_examples_cache = []
    return _upload_examples_cache


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
                label="Try a synthetic example",
                examples_per_page=3,
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
