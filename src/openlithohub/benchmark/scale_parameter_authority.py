"""Scale parameter authority — the per-design source of truth for
benchmark parameters that a fixture manifest alone cannot justify
(PR-V3.1.3).

The frozen fixture manifests record WHICH value each design uses
(``selected_layer``, pixel size, ...) but record nothing about WHY that
value is authoritative for THAT design.  The review that opened
PR-V3.1.3 found exactly the failure mode this module exists to prevent:
Microwatt's ``66:44`` had propagated from Ibex by analogy, and its
rationale lived only in prose with no regenerable evidence artifact.

The rule this module freeze-closes:

    NO parameter may be inherited from another design by analogy.  A
    parameter is authoritative for a design only through one of:

    * ``lineage``      — the same value was measured live for THIS
      design in the frozen v1.1/v2 authority lineage;
    * ``direct_audit`` — a committed layer-audit artifact
      (``OpenLithoHub.scale-fixture-layer-audit.v1``) re-derives the
      decision from the frozen GDS bytes.

    Any parameter without one of these bases is UNRESOLVED and must not
    be used in a formal run for that design.  ``analogy`` is not a basis
    and is rejected by the validation helper.

This module also freezes the per-design fixture-tracking semantics the
tracked-tree gate operates under (why the same ``git status --porcelain``
gate is a live tripwire for one design and structurally blind for
another) — the asymmetry is a protocol property, not an accident of the
``.gitignore``, and CI enforces that the declared semantics match
reality (``tests/test_benchmark/test_scale_parameter_authority.py``).
"""

from __future__ import annotations

from typing import Any, Final

from openlithohub.benchmark.industrial_scale import PDB_COMMIT, PDB_REPOSITORY

SCALE_PARAMETER_AUTHORITY_SCHEMA: Final = "OpenLithoHub.scale-parameter-authority.v1"

VALID_AUTHORITY_BASES: Final = ("lineage", "direct_audit")

#: Ibex GDS regenerated from the pinned PDB tree (SCALE-G1R2, issue
#: #94).  The committed ``fixtures/ibex/`` directory is host-local
#: (gitignored), so this frozen value is the only in-repo anchor; the
#: preflight/``--expected-gds-sha256`` gate rejects any local file that
#: does not match it — including the stale standin-era 144,566-byte
#: ``ibex.gds`` (``9b1790b9…``) known to linger on measurement hosts.
IBEX_FROZEN_GDS_SHA256: Final = (
    "5b706ac417f994d357ff78627a01baad8724b808b5e66ed7fe32a022f904664c"
)

SCALE_PARAMETER_AUTHORITY: Final[dict[str, Any]] = {
    "schema": SCALE_PARAMETER_AUTHORITY_SCHEMA,
    "source_repository": PDB_REPOSITORY,
    "source_commit": PDB_COMMIT,
    "rule": (
        "No parameter may be inherited from another design by analogy. "
        "Every design-specific parameter carries authority_basis "
        "'lineage' (live-measured for THIS design in the frozen v1.1/v2 "
        "authority lineage) or 'direct_audit' (committed layer-audit "
        "artifact re-derived from the frozen GDS bytes). Anything else "
        "is UNRESOLVED and blocks formal use for that design."
    ),
    "designs": {
        "ibex": {
            "frozen_gds_sha256": IBEX_FROZEN_GDS_SHA256,
            "parameters": {
                "selected_layer": {
                    "value": "66:44",
                    "authority_basis": "lineage",
                    "basis_reference": (
                        "v1.1/v2 industrial authority lineage: the same GDS "
                        "was live-measured on layer 66:44 (docs/"
                        "industrial-benchmarks.md); layer-decision audit "
                        "recorded at freeze (2026-09-25)"
                    ),
                    "status": "FROZEN",
                },
                "pixel_nm": {
                    "value": 1.0,
                    "authority_basis": "lineage",
                    "basis_reference": (
                        "v1.1/v2 measurement grid, carried unchanged into the "
                        "scale track"
                    ),
                    "status": "FROZEN",
                },
            },
            "fixture_tracking": {
                "tracked_paths": [],
                "host_local_paths": [
                    "benchmarks/results/industrial-scale/fixtures/ibex/",
                ],
                "tracked_tree_gate_visibility": (
                    "none — the whole ibex fixture directory is gitignored, "
                    "so the tracked-tree gate is structurally blind to it; "
                    "integrity is enforced solely by manifest SHA-256 "
                    "equality (preflight revalidation + "
                    "--expected-gds-sha256)"
                ),
            },
        },
        "microwatt": {
            "frozen_gds_sha256": (
                "b0253af06f35d1a8b11c2a47f70aac89f33be53e8f28da844b35c2ad6cc92a6d"
            ),
            "parameters": {
                "selected_layer": {
                    "value": "66:44",
                    "authority_basis": "direct_audit",
                    "basis_reference": (
                        "benchmarks/results/industrial-scale/audits/"
                        "microwatt-layer-audit.json (committed evidence "
                        "re-derived from the frozen GDS bytes)"
                    ),
                    "status": "FROZEN",
                },
                "pixel_nm": {
                    "value": 1.0,
                    "authority_basis": "lineage",
                    "basis_reference": (
                        "scale-track protocol grid, frozen identically for "
                        "both designs (docs/industrial-scale-benchmark.md)"
                    ),
                    "status": "FROZEN",
                },
            },
            "fixture_tracking": {
                "tracked_paths": [
                    "benchmarks/results/industrial-scale/fixtures/microwatt/"
                    "fixture-manifest.json",
                    "benchmarks/results/industrial-scale/fixtures/microwatt/"
                    "pdb-split-manifest.json",
                ],
                "host_local_paths": [
                    "benchmarks/results/industrial-scale/fixtures/microwatt/"
                    "microwatt.gds",
                ],
                "tracked_tree_gate_visibility": (
                    "live tripwire on the two committed manifests only — a "
                    "regenerated manifest that changes bytes fails the "
                    "tracked-tree gate; the 554,770,926-byte GDS itself is "
                    "gitignored and invisible to the gate, enforced by "
                    "SHA-256 equality instead"
                ),
            },
        },
    },
}


def parameter_authority(design: str, parameter: str) -> dict[str, Any]:
    """Return the authority record for one design parameter."""
    try:
        entry = SCALE_PARAMETER_AUTHORITY["designs"][design]
    except KeyError as exc:  # pragma: no cover - frozen namespace
        raise KeyError(f"unknown scale design: {design!r}") from exc
    try:
        return entry["parameters"][parameter]
    except KeyError as exc:
        raise KeyError(
            f"design {design!r} has no authority record for parameter "
            f"{parameter!r} — a parameter without a record is UNRESOLVED"
        ) from exc


def validate_scale_parameter_authority() -> list[str]:
    """Fail-closed validation of the frozen table itself.

    Returns a list of violations (empty means the table is sound): every
    design must carry fixture-tracking semantics, every parameter a
    known authority basis — ``analogy`` is rejected by construction —
    and every ``direct_audit`` basis must point at an existing committed
    audit artifact.
    """
    violations: list[str] = []
    table = SCALE_PARAMETER_AUTHORITY
    if table["schema"] != SCALE_PARAMETER_AUTHORITY_SCHEMA:
        violations.append(f"schema drift: {table['schema']!r}")
    designs = table["designs"]
    if set(designs) < {"ibex", "microwatt"}:
        violations.append("authority table must cover at least ibex and microwatt")
    repo = None
    try:
        from pathlib import Path

        repo = Path(__file__).resolve().parents[3]
    except Exception:  # pragma: no cover - defensive
        repo = None
    for design, entry in sorted(designs.items()):
        tracking = entry.get("fixture_tracking")
        if not tracking or "tracked_tree_gate_visibility" not in tracking:
            violations.append(
                f"{design}: fixture_tracking semantics missing — the "
                "tracked-vs-gitignored asymmetry must be declared, not implied"
            )
        if "frozen_gds_sha256" not in entry:
            violations.append(f"{design}: frozen_gds_sha256 missing")
        if not entry.get("parameters"):
            violations.append(f"{design}: no parameter records")
        for name, record in sorted(entry.get("parameters", {}).items()):
            basis = record.get("authority_basis")
            if basis not in VALID_AUTHORITY_BASES:
                violations.append(
                    f"{design}.{name}: authority_basis {basis!r} is not one of "
                    f"{VALID_AUTHORITY_BASES} — analogy is never a basis"
                )
            if record.get("status") != "FROZEN":
                violations.append(
                    f"{design}.{name}: status {record.get('status')!r} is not FROZEN"
                )
            if basis == "direct_audit" and repo is not None:
                reference = record.get("basis_reference", "").split(" (")[0].strip()
                if not (repo / reference).is_file():
                    violations.append(
                        f"{design}.{name}: direct_audit evidence artifact "
                        f"missing: {reference}"
                    )
    return violations
