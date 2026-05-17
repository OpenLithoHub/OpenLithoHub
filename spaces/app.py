"""OpenLithoHub Playground — Interactive web demo for computational lithography evaluation."""

from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Metric computation (inlined from openlithohub to keep Spaces self-contained)
# ---------------------------------------------------------------------------


def _extract_edges(binary: np.ndarray) -> np.ndarray:
    """Extract edges via Sobel-like gradient magnitude."""
    t = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
    ).reshape(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(t, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(t, sobel_y, padding=1)
    mag = (gx**2 + gy**2).sqrt().squeeze().numpy()
    return (mag > 0).astype(np.float32)


def compute_epe(predicted: np.ndarray, target: np.ndarray, pixel_size_nm: float = 1.0) -> dict:
    """Compute EPE between two binary masks."""
    pred_edges = _extract_edges(predicted)
    tgt_edges = _extract_edges(target)

    pred_pts = np.argwhere(pred_edges > 0).astype(np.float32)
    tgt_pts = np.argwhere(tgt_edges > 0).astype(np.float32)

    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        return {"epe_mean_nm": 0.0, "epe_max_nm": 0.0, "epe_std_nm": 0.0}

    pred_t = torch.from_numpy(pred_pts)
    tgt_t = torch.from_numpy(tgt_pts)
    dists = torch.cdist(pred_t, tgt_t)
    min_dists = dists.min(dim=1).values * pixel_size_nm

    return {
        "epe_mean_nm": float(min_dists.mean().item()),
        "epe_max_nm": float(min_dists.max().item()),
        "epe_std_nm": float(min_dists.std().item()) if len(min_dists) > 1 else 0.0,
    }


def check_mrc(
    mask: np.ndarray, min_width_nm: float = 40.0, min_spacing_nm: float = 40.0, pixel_size_nm: float = 1.0
) -> dict:
    """Simplified MRC check using morphological opening."""
    import math

    from scipy.ndimage import binary_dilation, binary_erosion

    binary = mask > 0.5
    h, w = binary.shape
    total = h * w

    radius_w = int(math.floor(min_width_nm / (2.0 * pixel_size_nm)))
    radius_s = int(math.floor(min_spacing_nm / (2.0 * pixel_size_nm)))

    width_violations = 0
    spacing_violations = 0

    if binary.any() and radius_w >= 1:
        struct = np.ones((2 * radius_w + 1, 2 * radius_w + 1), dtype=bool)
        opened = binary_dilation(binary_erosion(binary, structure=struct), structure=struct)
        width_violations = int(np.sum(binary & ~opened))

    if binary.any() and (~binary).any() and radius_s >= 1:
        bg = ~binary
        struct = np.ones((2 * radius_s + 1, 2 * radius_s + 1), dtype=bool)
        opened_bg = binary_dilation(binary_erosion(bg, structure=struct), structure=struct)
        spacing_violations = int(np.sum(bg & ~opened_bg))

    total_violations = width_violations + spacing_violations
    return {
        "passed": total_violations == 0,
        "violation_count": total_violations,
        "violation_rate": total_violations / total if total > 0 else 0.0,
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
    mrc = check_mrc(predicted, min_width_nm=min_width_nm, min_spacing_nm=min_spacing_nm, pixel_size_nm=pixel_size_nm)

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

    # Resize to match if different
    if pred_img.size != tgt_img.size:
        tgt_img = tgt_img.resize(pred_img.size, Image.NEAREST)

    predicted = (np.array(pred_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    target = (np.array(tgt_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

    epe = compute_epe(predicted, target, pixel_size_nm=pixel_size_nm)
    mrc = check_mrc(predicted, min_width_nm=min_width_nm, min_spacing_nm=min_spacing_nm, pixel_size_nm=pixel_size_nm)

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
# Gradio App
# ---------------------------------------------------------------------------


with gr.Blocks(
    title="OpenLithoHub Playground",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan"),
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
            gr.Markdown("Upload your own predicted and target mask images (grayscale, thresholded at 50%).")
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
