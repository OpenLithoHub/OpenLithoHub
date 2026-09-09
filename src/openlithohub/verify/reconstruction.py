"""Certified nominal contour reconstruction artifacts for B04 MVP-1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .replay import sha256_file


@dataclass(frozen=True)
class ReconstructionReplay:
    components: int
    reconstructed_cells: int
    polyline_vertices: int
    polyline_segments: int
    nominal_reconstruction_upper_nm: float
    level_set_focus_upper_nm: float
    continuous_focus_to_reconstruction_upper_nm: float


def replay_reconstruction_artifact(path: str | Path) -> ReconstructionReplay:
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "B04.nominal_contour_reconstruction.v1":
        raise ValueError("unsupported B04 reconstruction schema")
    summary = raw["summary"]
    cells = raw["cells"]
    polyline = raw["polyline"]

    if raw.get("status") != "PASS_UNDER_EXACT_SOURCE_INTERVAL_PLUS_CHORD_THEOREM":
        raise ValueError("reconstruction artifact is not proof-certified")
    if int(summary["reconstructed_arc_cells"]) != 148 or len(cells) != 148:
        raise ValueError("unexpected reconstructed-cell count")
    if int(summary["contour_components"]) != 1:
        raise ValueError("unexpected contour-component count")
    if not bool(polyline["closed"]):
        raise ValueError("nominal reconstruction is not closed")
    if len(polyline["vertices"]) != int(summary["polyline_vertices"]):
        raise ValueError("polyline vertex-count mismatch")
    if len(polyline["segments"]) != int(summary["polyline_segments"]):
        raise ValueError("polyline segment-count mismatch")
    if any(float(c["normal_derivative_lower_per_nm"]) <= 0.0 for c in cells):
        raise ValueError("reconstruction contains a non-monotone cell")

    eta = max(float(c["eta_cell_upper_nm"]) for c in cells)
    if abs(eta - float(summary["nominal_reconstruction_hausdorff_upper_nm"])) > 1e-12:
        raise ValueError("reconstruction Hausdorff summary mismatch")

    focus = float(summary["level_set_focus_upper_nm"])
    total = float(summary["continuous_focus_to_reconstruction_upper_nm"])
    if total + 1e-12 < focus + eta:
        raise ValueError("combined focus/reconstruction bound is too small")

    return ReconstructionReplay(
        components=int(summary["contour_components"]),
        reconstructed_cells=int(summary["reconstructed_arc_cells"]),
        polyline_vertices=int(summary["polyline_vertices"]),
        polyline_segments=int(summary["polyline_segments"]),
        nominal_reconstruction_upper_nm=eta,
        level_set_focus_upper_nm=focus,
        continuous_focus_to_reconstruction_upper_nm=total,
    )


def verify_reconstruction_artifact(path: str | Path, expected_sha256: str) -> ReconstructionReplay:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("reconstruction artifact hash mismatch")
    return replay_reconstruction_artifact(path)
