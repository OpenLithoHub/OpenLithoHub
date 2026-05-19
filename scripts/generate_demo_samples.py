"""Generate the deterministic preset demo PNGs shipped to the HF Space.

Run from the repo root:

    .venv/bin/python scripts/generate_demo_samples.py

The output files are committed to ``spaces/examples/`` so that the HF Space
syncer (``.github/workflows/sync-hf-space.yml``) ships them with the app.
Keeping the PNGs under version control means cold-start on HF doesn't depend
on a runtime tempdir — that historically broke the upload-tab examples on
new container boots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "spaces" / "examples"

SIZE = 256


# The line/space, contact-holes, and SRAM-like generators are intentionally
# duplicated from ``spaces/app.py`` (kept identical — see PATTERN_GENERATORS).
# This script must run without gradio installed (CI / minimal venvs), so we
# don't import the Space module here.


def generate_line_space(size: int = SIZE, pitch_px: int = 20, duty: float = 0.5) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    line_width = int(pitch_px * duty)
    for x in range(0, size, pitch_px):
        mask[:, x : x + line_width] = 1.0
    return mask


def generate_contact_holes(size: int = SIZE, hole_size: int = 10, pitch: int = 40) -> np.ndarray:
    mask = np.ones((size, size), dtype=np.float32)
    for y in range(pitch // 2, size, pitch):
        for x in range(pitch // 2, size, pitch):
            y0, y1 = max(0, y - hole_size // 2), min(size, y + hole_size // 2)
            x0, x1 = max(0, x - hole_size // 2), min(size, x + hole_size // 2)
            mask[y0:y1, x0:x1] = 0.0
    return mask


def generate_sram(size: int = SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    for y in range(20, size - 20, 40):
        mask[y : y + 8, 10 : size - 10] = 1.0
    for x in range(30, size - 30, 60):
        for y in range(20, size - 40, 80):
            mask[y : y + 40, x : x + 6] = 1.0
    for y in range(40, size - 40, 80):
        for x in range(50, size - 50, 80):
            mask[y - 5 : y + 5, x - 5 : x + 5] = 1.0
    return mask


def generate_random_logic(size: int = SIZE, *, seed: int = 7) -> np.ndarray:
    """Manhattan random-logic routing on a coarse grid.

    Mimics the visual density of standard-cell back-end routing: short
    horizontal/vertical segments at varying widths, occasional vias.
    """
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


def perturb(target: np.ndarray, *, seed: int) -> np.ndarray:
    """Produce a non-trivial 'predicted' mask from a clean target.

    Combines additive Gaussian noise with a 1-pixel asymmetric bias so the
    EPE comes out > 0 and the MRC overlay has something to highlight.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.18, target.shape).astype(np.float32)
    biased = np.roll(target, shift=1, axis=0)
    return ((target * 0.7 + biased * 0.3 + noise) > 0.5).astype(np.float32)


PRESETS: dict[str, tuple[callable, int]] = {
    "line_space": (lambda: generate_line_space(SIZE), 1),
    "contact_holes": (lambda: generate_contact_holes(SIZE), 2),
    "sram_like": (lambda: generate_sram(SIZE), 3),
    "random_logic": (lambda: generate_random_logic(SIZE), 4),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, (gen, seed) in PRESETS.items():
        target = gen()
        predicted = perturb(target, seed=seed)
        Image.fromarray((target * 255).astype(np.uint8)).save(OUT_DIR / f"{slug}_target.png")
        Image.fromarray((predicted * 255).astype(np.uint8)).save(OUT_DIR / f"{slug}_pred.png")
        print(f"wrote {slug}_target.png + {slug}_pred.png")


if __name__ == "__main__":
    main()
