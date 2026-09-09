"""Replay/consume certified B04 boundary-coverage artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .replay import sha256_file
from .spatial import reduce_coverage
from .types import (
    BoundaryRootStatus,
    CellCertificate,
    CellDisposition,
    CoverageStatus,
    RootBracket,
)


@dataclass(frozen=True)
class CoverageReplay:
    status: CoverageStatus
    cells: tuple[CellCertificate, ...]
    seeded_cells: int
    curvature_certified_empty_cells: int
    root_face_incidences: int
    root_free_face_incidences: int


def replay_boundary_coverage(path: str | Path) -> CoverageReplay:
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "B04.boundary_coverage.v1":
        raise ValueError("unsupported B04 boundary-coverage schema")

    cells: list[CellCertificate] = []
    for item in raw["cells"]:
        disposition = CellDisposition(str(item["disposition"]))
        brackets = tuple(
            RootBracket(
                float(r["lo_nm"]),
                float(r["hi_nm"]),
                face=str(r["face"]),
            )
            for r in item["root_brackets"]
        )
        face_statuses = {BoundaryRootStatus(str(v["status"])) for v in item["faces"].values()}
        if BoundaryRootStatus.INCONCLUSIVE in face_statuses:
            boundary_status = BoundaryRootStatus.INCONCLUSIVE
        elif BoundaryRootStatus.ROOT_BRACKETS in face_statuses:
            boundary_status = BoundaryRootStatus.ROOT_BRACKETS
        else:
            boundary_status = BoundaryRootStatus.ROOT_FREE_CERTIFIED

        y, x = (int(v) for v in item["cell_yx"])
        cell = CellCertificate(
            cell_id=f"{y}:{x}",
            cell_bbox_nm=(
                (x - 0.5) * 8.0,
                (y - 0.5) * 8.0,
                (x + 0.5) * 8.0,
                (y + 0.5) * 8.0,
            ),
            kappa_lower_per_nm=float(item["kappa_lower_per_nm"]),
            hessian_upper_per_nm2=float(item["hessian_upper_per_nm2"]),
            cell_diameter_nm=float(item["cell_diameter_nm"]),
            curvature_scale_nm=float(item["curvature_scale_nm"]),
            boundary_root_status=boundary_status,
            hidden_loop_excluded=bool(item["hidden_loop_excluded"]),
            root_brackets=brackets,
            eta_sp_nm=None,
            disposition=disposition,
        )
        cells.append(cell)

    cells_t = tuple(cells)
    status = reduce_coverage(cells_t)
    summary = raw["summary"]
    if status.value != str(summary["coverage_status"]):
        raise ValueError("coverage summary does not match cell dispositions")
    if len(cells_t) != int(summary["active_cells"]):
        raise ValueError("active-cell count mismatch")
    if sum(c.disposition is CellDisposition.SEEDED for c in cells_t) != int(
        summary["seeded_cells"]
    ):
        raise ValueError("seeded-cell count mismatch")
    if sum(c.disposition is CellDisposition.CURVATURE_CERTIFIED_EMPTY for c in cells_t) != int(
        summary["curvature_certified_empty_cells"]
    ):
        raise ValueError("curvature-empty count mismatch")

    return CoverageReplay(
        status=status,
        cells=cells_t,
        seeded_cells=int(summary["seeded_cells"]),
        curvature_certified_empty_cells=int(summary["curvature_certified_empty_cells"]),
        root_face_incidences=int(summary["root_face_incidences"]),
        root_free_face_incidences=int(summary["root_free_face_incidences"]),
    )


def verify_boundary_artifact(path: str | Path, expected_sha256: str) -> CoverageReplay:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("boundary-coverage artifact hash mismatch")
    return replay_boundary_coverage(path)
