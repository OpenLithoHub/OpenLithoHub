"""B04 MVP-1 proof-artifact adapter.

This adapter verifies artifact hashes, replays the exact-source expanded-band
gate, and constructs a typed level-set-stability certificate.  It intentionally
does not claim extracted-contour EPE coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .certifier import assemble_continuous_focus_certificate
from .coverage import verify_boundary_artifact
from .reconstruction import verify_reconstruction_artifact
from .replay import replay_expanded_band, sha256_file
from .types import (
    CertificateTarget,
    ContinuousFocusCertificate,
    CoverageStatus,
    DependencyRecord,
    ParameterInterval,
    ProofLevel,
)


def _proof_level(value: str) -> ProofLevel:
    try:
        return ProofLevel(value)
    except ValueError as exc:
        raise ValueError(f"unknown proof level: {value!r}") from exc


def certify_mvp1_manifest(
    manifest_path: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    tolerance_nm: float = 1.0,
    target: CertificateTarget = CertificateTarget.LEVEL_SET_STABILITY,
) -> ContinuousFocusCertificate:
    manifest_path = Path(manifest_path)
    root = Path(artifact_dir) if artifact_dir is not None else manifest_path.parent
    raw: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    if raw.get("schema") != "B04.openlithohub.mvp1.manifest.v1":
        raise ValueError("unsupported B04 MVP-1 manifest schema")

    fixture = raw["fixture"]
    proof = raw["proof_artifacts"]
    bounds = raw["bounds"]

    mask_path = root / "B04_MVP1_Mask_72x72_uint8.bin"
    if sha256_file(mask_path) != raw["input_mask_sha256"]:
        raise ValueError("input mask hash mismatch")

    grid_path = root / proof["expanded_band_grid_file"]
    if sha256_file(grid_path) != proof["expanded_band_grid_sha256"]:
        raise ValueError("expanded-band proof grid hash mismatch")

    replay = replay_expanded_band(
        grid_path,
        threshold=float(fixture["threshold"]),
        pixel_nm=float(fixture["pixel_nm"]),
        spatial_third_derivative_upper_per_nm3=4.972184653511609e-05,
        delta_total=float(bounds["combined_nominal_intensity_radius"]),
    )
    if replay.active_cells != int(bounds["expanded_band_active_cells"]):
        raise ValueError("active-cell replay mismatch")
    if abs(replay.kappa_lower_per_nm - float(bounds["expanded_band_kappa_lower_per_nm"])) > 1e-15:
        raise ValueError("kappa replay mismatch")
    if abs(replay.hausdorff_upper_nm - float(bounds["continuous_focus_contour_upper_nm"])) > 1e-12:
        raise ValueError("Hausdorff replay mismatch")

    coverage_status = CoverageStatus.INCONCLUSIVE
    boundary_file = proof.get("boundary_coverage_file")
    boundary_sha = proof.get("boundary_coverage_sha256")
    if boundary_file is not None or boundary_sha is not None:
        if not boundary_file or not boundary_sha:
            raise ValueError("boundary coverage file/hash must be supplied together")
        coverage = verify_boundary_artifact(root / str(boundary_file), str(boundary_sha))
        coverage_status = coverage.status

    reconstruction = None
    if target is CertificateTarget.EXTRACTED_CONTOUR_EPE:
        if coverage_status is not CoverageStatus.ALL_COMPONENTS_COVERED:
            raise ValueError("EXTRACTED_CONTOUR_EPE requires complete component coverage")
        recon_file = proof.get("reconstruction_file")
        recon_sha = proof.get("reconstruction_sha256")
        if not recon_file or not recon_sha:
            raise ValueError("EXTRACTED_CONTOUR_EPE requires a certified reconstruction artifact")
        reconstruction = verify_reconstruction_artifact(root / str(recon_file), str(recon_sha))

    dependencies = tuple(
        DependencyRecord(
            name=str(d["name"]),
            level=_proof_level(str(d["level"])),
            satisfied=bool(d["satisfied"]),
            method=str(d["method"]),
        )
        for d in raw["provenance"]
    )

    if reconstruction is not None:
        dependencies = dependencies + (
            DependencyRecord(
                name="nominal contour reconstruction",
                level=ProofLevel.INTERVAL_CERTIFIED,
                satisfied=True,
                method="fixed-normal implicit graph + certified chord interpolation",
                artifact_sha256=str(proof["reconstruction_sha256"]),
            ),
        )

    lo, hi = fixture["focus_interval_nm"]
    contour_upper = (
        replay.hausdorff_upper_nm
        if reconstruction is None
        else reconstruction.continuous_focus_to_reconstruction_upper_nm
    )
    recon_upper = None if reconstruction is None else reconstruction.nominal_reconstruction_upper_nm
    return assemble_continuous_focus_certificate(
        target=target,
        model_id=str(raw["model_id"]),
        input_sha256=str(raw["input_mask_sha256"]),
        parameters=(ParameterInterval("focus_nm", float(lo), float(hi)),),
        continuous_focus_contour_upper_nm=contour_upper,
        continuous_focus_pairwise_diameter_upper_nm=(replay.pairwise_contour_diameter_upper_nm),
        tolerance_nm=tolerance_nm,
        coverage_status=coverage_status,
        dependencies=dependencies,
        proof_artifact_sha256=(
            sha256_file(grid_path)
            if reconstruction is None
            else str(proof["reconstruction_sha256"])
        ),
        nominal_reconstruction_upper_nm=recon_upper,
        note=(
            "B04 theorem-facing continuous contour certificate. "
            "EXTRACTED_CONTOUR_EPE here means distance to the certified nominal "
            "full-source reconstruction polyline; it is not design-target EPE "
            "and is not raster Sobel epe_max_nm."
        ),
    )
