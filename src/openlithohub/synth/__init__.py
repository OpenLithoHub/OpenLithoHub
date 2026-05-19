"""Synthetic layout generator.

Why a separate module from :mod:`openlithohub.data.dummy`?
``data.dummy`` exists for hermetic CI and debugging — it splats random
rectangles and runs a morphological clean-up. That is not a substitute
for a real synthetic dataset.

This package is the dataset-grade synthetic generator referred to as
P4 #10 in the project roadmap. It produces layouts that are:

* **PDK-aware** — design rules pulled from FreePDK45 / ASAP7 presets so
  that minimum-width, minimum-spacing, and via-coverage rules are
  honoured by construction.
* **Pattern-typed** — SRAM cells, contact arrays, and random logic
  routing are first-class generators, not the same Gaussian pattern
  with different random seeds.
* **MRC-clean** — every emitted mask passes :func:`check_mrc` for the
  PDK's pitch.

A diffusion-based generator is out of scope for v0.1 and lives behind
:class:`DiffusionLayoutGenerator`, which raises :class:`NotImplementedError`
with a pointer to the design RFC.
"""

from __future__ import annotations

from openlithohub.synth.diffusion import DiffusionLayoutGenerator
from openlithohub.synth.pdk import PDK_PRESETS, PdkRules, get_pdk
from openlithohub.synth.rule_based import (
    PatternKind,
    SyntheticBatch,
    generate_layout,
    generate_synthetic_batch,
)

__all__ = [
    "PDK_PRESETS",
    "DiffusionLayoutGenerator",
    "PatternKind",
    "PdkRules",
    "SyntheticBatch",
    "generate_layout",
    "generate_synthetic_batch",
    "get_pdk",
]
