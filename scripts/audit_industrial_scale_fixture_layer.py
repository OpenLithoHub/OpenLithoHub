"""Industrial Scale fixture layer audit — parameter-authority evidence.

Produces the committed per-layer audit evidence that backs a design's
``selected_layer`` parameter authority (PR-V3.1.3).  The frozen fixture
manifests record WHICH layer was selected but carry no per-layer
statistics, so the selection rationale ("non-empty, second densest of
41 layers, spans 98.8% x 99.4% of the die") lived only in prose with no
regenerable artifact.  This script closes that gap: it re-reads the
FROZEN GDS bytes (never a historical claim), re-derives the per-layer
statistics with KLayout, fail-closed cross-checks every field against
the committed fixture manifest, and writes a deterministic strict-JSON
audit evidence file.

This script does AUDIT ONLY — it never writes the GDS, never mutates a
manifest, and never runs a benchmark.

Fail-closed rules (any mismatch is a hard FAIL):

1. the GDS on disk must hash to the manifest's ``gds_sha256`` (a stale
   or substituted GDS can never be audited into authority);
2. ``--expected-gds-sha256`` (optional) must also match;
3. KLayout must reopen the file with exactly one top cell whose name,
   DBU and bbox equal the manifest's;
4. the full layer enumeration must equal the manifest's;
5. ``--selected-layer`` must equal the manifest's ``selected_layer``
   and exist in the file — the audit NEVER picks a layer.

Checkpointing: partial results are written to ``<out>.checkpoint.json``
(keyed by the GDS sha256) after every layer, so an interrupted audit
resumes or at least never loses completed-layer work; the checkpoint is
removed on success.

Usage::

    python scripts/audit_industrial_scale_fixture_layer.py \
        --gds benchmarks/results/industrial-scale/fixtures/microwatt/microwatt.gds \
        --manifest benchmarks/results/industrial-scale/fixtures/microwatt/fixture-manifest.json \
        --design microwatt \
        --selected-layer 66:44 \
        --out benchmarks/results/industrial-scale/audits/microwatt-layer-audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_scale import write_strict_json  # noqa: E402

AUDIT_SCHEMA = "OpenLithoHub.scale-fixture-layer-audit.v1"
CHECKPOINT_SCHEMA = "OpenLithoHub.scale-fixture-layer-audit-checkpoint.v1"
CHUNK = 1 << 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _fail(message: str) -> NoReturn:
    print(f"LAYER AUDIT: FAIL — {message}", file=sys.stderr)
    raise SystemExit(1)


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        _fail(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema") != "OpenLithoHub.scale-fixture.v1":
        _fail(f"manifest schema is {manifest.get('schema')!r}, expected scale-fixture.v1")
    return manifest


def _percent(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)


def audit_layer_statistics(
    gds_path: Path,
    *,
    design: str,
    selected_layer: str,
    manifest: dict,
    expected_gds_sha256: str | None,
    out_path: Path,
) -> dict:
    """Open the GDS with KLayout and derive per-layer statistics,
    cross-checked against the manifest at every step."""
    try:
        import klayout.db as kdb
    except ImportError as exc:  # pragma: no cover - environment gate
        _fail(f"KLayout Python API unavailable: {exc}")

    gds_bytes = gds_path.stat().st_size
    print(f"[1/5] sha256 {gds_path} ({gds_bytes} bytes) ...", flush=True)
    gds_sha256 = _sha256_file(gds_path)
    if gds_sha256 != manifest["gds_sha256"]:
        _fail(
            f"GDS sha256 {gds_sha256} != manifest gds_sha256 "
            f"{manifest['gds_sha256']} — stale or substituted artifact has no authority"
        )
    if expected_gds_sha256 and gds_sha256 != expected_gds_sha256:
        _fail(f"GDS sha256 {gds_sha256} != --expected-gds-sha256 {expected_gds_sha256}")
    if gds_bytes != manifest["gds_bytes"]:
        _fail(f"GDS byte count {gds_bytes} != manifest gds_bytes {manifest['gds_bytes']}")
    if manifest["design"] != design:
        _fail(f"manifest design {manifest['design']!r} != --design {design!r}")
    if manifest["selected_layer"] != selected_layer:
        _fail(
            f"manifest selected_layer {manifest['selected_layer']!r} != "
            f"--selected-layer {selected_layer!r} — the audit never picks a layer"
        )

    print("[2/5] KLayout open + structural cross-check ...", flush=True)
    layout = kdb.Layout()
    try:
        layout.read(str(gds_path))
    except Exception as exc:  # noqa: BLE001 — KLayout raises bare RuntimeError
        _fail(f"KLayout cannot read the GDS (truncated/corrupt?): {exc}")
    tops = list(layout.top_cells())
    if len(tops) != 1:
        names = [top.name for top in tops]
        _fail(f"expected exactly one top cell, found {names}")
    top = tops[0]
    if top.name != manifest["top_cell"]:
        _fail(f"top cell {top.name!r} != manifest top_cell {manifest['top_cell']!r}")
    dbu_nm = float(layout.dbu) * 1000.0
    if abs(dbu_nm - float(manifest["dbu_nm"])) > 1e-9:
        _fail(f"DBU {dbu_nm} nm != manifest dbu_nm {manifest['dbu_nm']}")
    bbox = top.bbox()
    bbox_dbu = (bbox.left, bbox.bottom, bbox.right, bbox.top)
    if list(bbox_dbu) != list(manifest["bbox_dbu"]):
        _fail(f"bbox {list(bbox_dbu)} != manifest bbox_dbu {manifest['bbox_dbu']}")
    layers = [
        f"{layout.get_info(idx).layer}:{layout.get_info(idx).datatype}"
        for idx in layout.layer_indices()
    ]
    if sorted(layers) != sorted(manifest["layers"]):
        _fail("layer enumeration differs from the manifest — the fixture bytes drifted")
    if selected_layer not in layers:
        _fail(f"selected layer {selected_layer!r} is not present in the GDS")
    die_w = bbox.width()
    die_h = bbox.height()

    checkpoint_path = out_path.with_suffix(out_path.suffix + ".checkpoint.json")
    per_layer: list[dict] = []
    print(
        f"[3/5] per-layer statistics over {len(layers)} layers (checkpoint per layer) ...",
        flush=True,
    )
    for position, layer_idx in enumerate(sorted(layout.layer_indices(), key=str), start=1):
        info = layout.get_info(layer_idx)
        layer_str = f"{info.layer}:{info.datatype}"
        shape_elements = 0
        for cell in layout.each_cell():
            shape_elements += cell.shapes(layer_idx).size()
        flattened_instances = 0
        it = top.begin_shapes_rec(layer_idx)
        while not it.at_end():
            flattened_instances += 1
            it.next()
        layer_bbox = top.bbox(layer_idx)
        if layer_bbox.empty():
            entry = {
                "layer": layer_str,
                "shape_elements": shape_elements,
                "flattened_instances": flattened_instances,
                "bbox_dbu": None,
                "coverage_x_pct": 0.0,
                "coverage_y_pct": 0.0,
            }
        else:
            entry = {
                "layer": layer_str,
                "shape_elements": shape_elements,
                "flattened_instances": flattened_instances,
                "bbox_dbu": (
                    layer_bbox.left,
                    layer_bbox.bottom,
                    layer_bbox.right,
                    layer_bbox.top,
                ),
                "coverage_x_pct": _percent(layer_bbox.width(), die_w),
                "coverage_y_pct": _percent(layer_bbox.height(), die_h),
            }
        per_layer.append(entry)
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "gds_sha256": gds_sha256,
            "design": design,
            "completed": position,
            "total_layers": len(layers),
            "per_layer": per_layer,
        }
        write_strict_json(str(checkpoint_path), checkpoint)
        print(
            f"    [{position}/{len(layers)}] {layer_str}: "
            f"{shape_elements} elements / {flattened_instances} flattened instances, "
            f"coverage {entry['coverage_x_pct']}% x {entry['coverage_y_pct']}%",
            flush=True,
        )

    print("[4/5] selected-layer verdict ...", flush=True)
    selected_entry = next(e for e in per_layer if e["layer"] == selected_layer)
    non_empty = [e for e in per_layer if e["flattened_instances"] > 0]
    density_rank = sorted(non_empty, key=lambda e: e["flattened_instances"], reverse=True)
    rank = next(i for i, e in enumerate(density_rank, start=1) if e["layer"] == selected_layer)
    selected_audit = {
        "layer": selected_layer,
        "present": True,
        "shape_elements": selected_entry["shape_elements"],
        "flattened_instances": selected_entry["flattened_instances"],
        "non_empty": selected_entry["flattened_instances"] > 0,
        "density_rank_by_flattened_instances": rank,
        "non_empty_layers": len(non_empty),
        "coverage_x_pct": selected_entry["coverage_x_pct"],
        "coverage_y_pct": selected_entry["coverage_y_pct"],
    }

    print("[5/5] strict-JSON evidence ...", flush=True)
    manifest_path = (
        REPO
        / "benchmarks"
        / "results"
        / "industrial-scale"
        / "fixtures"
        / design
        / "fixture-manifest.json"
    )
    evidence = {
        "schema": AUDIT_SCHEMA,
        "design": design,
        "gds_sha256": gds_sha256,
        "gds_bytes": gds_bytes,
        "source_commit": manifest["source_commit"],
        "source_repository": manifest["source_repository"],
        "top_cell": top.name,
        "dbu_nm": dbu_nm,
        "bbox_dbu": list(bbox_dbu),
        "die_size_px": [die_w, die_h],
        "pixel_nm": manifest["pixel_nm"],
        "layer_count": len(layers),
        "selected_layer": selected_layer,
        "selected_layer_audit": selected_audit,
        "per_layer": sorted(per_layer, key=lambda e: e["layer"]),
        "count_definition": (
            "shape_elements: cell-local GDS elements summed over every cell "
            "(placements NOT multiplied out); flattened_instances: "
            "placement-expanded shape instances under the top cell (the "
            "protocol's density/instance metric). Density ranks use "
            "flattened_instances."
        ),
        "fixture_manifest_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else None,
        "audit_script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(str(out_path), evidence)
    checkpoint_path.unlink(missing_ok=True)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gds", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--selected-layer", required=True)
    parser.add_argument("--expected-gds-sha256", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gds_path = Path(args.gds)
    if not gds_path.is_file():
        _fail(f"GDS not found: {gds_path}")
    manifest = _load_manifest(Path(args.manifest))

    evidence = audit_layer_statistics(
        gds_path,
        design=args.design,
        selected_layer=args.selected_layer,
        manifest=manifest,
        expected_gds_sha256=args.expected_gds_sha256,
        out_path=Path(args.out),
    )
    audit = evidence["selected_layer_audit"]
    print(
        "LAYER AUDIT: PASS — "
        f"{args.design} {args.selected_layer}: {audit['flattened_instances']} "
        f"flattened instances ({audit['shape_elements']} elements), "
        f"density rank {audit['density_rank_by_flattened_instances']} of "
        f"{audit['non_empty_layers']} non-empty layers, "
        f"coverage {audit['coverage_x_pct']}% x {audit['coverage_y_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
