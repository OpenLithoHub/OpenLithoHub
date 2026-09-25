"""Industrial Scale fixture preparation (scale track, S2 / charter §S2).

Prepares ONE benchmark fixture from a local checkout of the frozen PDB
(Physical Design Database) lineage and emits a fail-closed
``OpenLithoHub.scale-fixture.v1`` manifest.  This script does
FIXTURE PREPARATION ONLY — it never runs a benchmark and never measures.

What it does, in order (every step fail-closed):

1. verify the source checkout is at the frozen PDB commit (never trust a
   historical artifact — re-read everything);
2. locate the GDS:
   - ``ibex``:      ``layout/sky130hd/ibex/ibex.gds`` (single file);
   - ``microwatt``: split chunks in ``layout/sky130hd/microwatt/`` —
     canonically sorted (lexical) and byte-concatenated into ONE
     ``microwatt.gds``.  Chunk order is canonical-by-construction, and
     the result is re-opened with KLayout so a corrupt or mis-assembled
     concatenation fails here instead of downstream;
3. SHA-256 + byte size of the prepared GDS (optional
   ``--expected-gds-sha256`` re-check);
4. KLayout open: exactly one top cell, name verified against the design;
   DBU, exact bbox (DBU) and the FULL layer enumeration recorded;
5. the benchmark layer is an EXPLICIT ``LAYER:DTYPE`` argument — never a
   default, never assumed from another design (Microwatt must not be
   assumed to use Ibex's 66:44).  The layer must exist in the file;
6. die size in pixels and the DENSE FLOAT32 RASTER EQUIVALENT are
   derived from bbox/DBU/pixel — a derived equivalent only, never a
   claim that such a raster was materialized;
7. strict ``fixture-manifest.json`` written with this script's own
   SHA-256 for provenance.

Usage::

    python scripts/prepare_industrial_scale_fixture.py \\
        --source-root /path/to/PDB-Physical-Design-Database \\
        --design ibex \\
        --selected-layer 66:44 \\
        --pixel-nm 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess  # noqa: S404 — fixed-argv git verification only
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_scale import (  # noqa: E402
    PDB_COMMIT,
    PDB_REPOSITORY,
    ScaleFixtureManifest,
    dense_float32_equivalent_bytes,
    write_strict_json,
)

DESIGN_SPECS = {
    "ibex": {
        "relative": "layout/sky130hd/ibex/ibex.gds",
        "top_cell": "ibex_core",
        "chunks": False,
    },
    "microwatt": {
        "relative": "layout/sky130hd/microwatt/microwatt.gds",
        "top_cell": "microwatt",
        "chunks": True,
    },
}


class FixtureError(Exception):
    pass


def _fail(message: str) -> None:
    raise FixtureError(message)


def verify_source_commit(source_root: Path, expected_commit: str) -> str:
    git = shutil.which("git")
    if git is None:
        _fail("git is required to verify the source commit")
    try:
        out = subprocess.run(  # noqa: S603 — fixed-argv git query
            [git, "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(f"cannot read source commit: {exc}")
    head = out.stdout.strip()
    if head != expected_commit:
        _fail(
            f"source checkout is at {head!r}, expected frozen PDB commit "
            f"{expected_commit!r} (checkout the exact commit first)"
        )
    return head


def discover_chunks(microwatt_dir: Path) -> list[Path]:
    """Locate the split chunks and return them in CANONICAL lexical
    order (``.part_aa, .part_ab, …``).  Canonical ordering is the
    defense against reassembly-order nondeterminism: the same chunks
    always concatenate to the same bytes."""
    if not microwatt_dir.is_dir():
        _fail(f"microwatt chunk directory missing: {microwatt_dir}")
    chunks = sorted(p for p in microwatt_dir.iterdir() if p.is_file() and ".part_" in p.name)
    if not chunks:
        _fail(f"no split chunks ('*.part_*') found in {microwatt_dir}")
    return chunks


def assemble_gds(source_root: Path, design: str, output_gds: Path) -> list[str]:
    """Materialize the prepared GDS file; returns the recorded chunk
    list (empty for single-file designs)."""
    spec = DESIGN_SPECS[design]
    source_gds = source_root / spec["relative"]
    if spec["chunks"]:
        chunks = discover_chunks(source_gds.parent)
        output_gds.parent.mkdir(parents=True, exist_ok=True)
        with open(output_gds, "wb") as sink:
            for chunk in chunks:
                digest = hashlib.sha256()
                with open(chunk, "rb") as handle:
                    for line in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        sink.write(line)
                        digest.update(line)
                del digest
        return [p.name for p in chunks]
    if not source_gds.is_file():
        _fail(f"fixture GDS missing: {source_gds}")
    output_gds.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_gds, output_gds)
    return []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_with_klayout(
    gds_path: Path,
    design: str,
    expected_top_cell: str,
    selected_layer: str,
    pixel_nm: float,
) -> dict:
    """Open the prepared GDS and extract the physical facts.  Fail-closed
    on: multi-top-cell files, wrong top cell, missing/invalid layer."""
    try:
        import klayout.db as db
    except ImportError:
        _fail("KLayout Python API is required for fixture preparation")
    layout = db.Layout()
    try:
        layout.read(str(gds_path))
    except Exception as exc:  # noqa: BLE001 — corrupt GDS must fail preparation
        _fail(f"KLayout cannot read the assembled GDS (truncated/corrupt?): {exc}")
    tops = list(layout.top_cells())
    if len(tops) != 1:
        names = [top.name for top in tops]
        _fail(f"expected exactly one top cell, found {names}")
    top = tops[0]
    if top.name != expected_top_cell:
        _fail(f"top cell is {top.name!r}, expected {expected_top_cell!r} for {design}")
    dbu_nm = float(layout.dbu) * 1000.0
    bbox = top.bbox()
    layers = [
        f"{layout.get_info(idx).layer}:{layout.get_info(idx).datatype}"
        for idx in layout.layer_indices()
    ]
    if selected_layer not in layers:
        _fail(
            f"selected layer {selected_layer!r} is not present; enumerated layers: "
            f"{layers} — inspect and choose the benchmark layer EXPLICITLY"
        )
    return {
        "top_cell": top.name,
        "dbu_nm": dbu_nm,
        "bbox_dbu": (bbox.left, bbox.bottom, bbox.right, bbox.top),
        "layers": layers,
    }


def prepare_fixture(
    *,
    source_root: Path,
    design: str,
    selected_layer: str,
    pixel_nm: float,
    output_dir: Path,
    expected_commit: str = PDB_COMMIT,
    expected_repository: str = PDB_REPOSITORY,
    expected_top_cell: str | None = None,
    expected_gds_sha256: str | None = None,
    preparation_script_path: Path | None = None,
) -> dict:
    """Prepare one fixture and return its manifest payload (validated
    before returning).  Raises :class:`FixtureError` on any failure."""
    if design not in DESIGN_SPECS:
        _fail(f"unknown design {design!r} (expected one of {sorted(DESIGN_SPECS)})")
    if ":" not in selected_layer:
        _fail(
            f"--selected-layer must be an explicit LAYER:DTYPE string, got "
            f"{selected_layer!r} — never assume a layer across designs"
        )
    spec = DESIGN_SPECS[design]
    top_cell = expected_top_cell or str(spec["top_cell"])

    verify_source_commit(source_root, expected_commit)

    output_dir = Path(output_dir)
    output_gds = output_dir / f"{design}.gds"
    chunks = assemble_gds(source_root, design, output_gds)

    gds_sha = sha256_file(output_gds)
    gds_bytes = output_gds.stat().st_size
    if expected_gds_sha256 and gds_sha != expected_gds_sha256:
        _fail(
            f"prepared GDS sha256 {gds_sha!r} != expected {expected_gds_sha256!r} "
            "(truncated or corrupted assembly?)"
        )

    facts = inspect_with_klayout(output_gds, design, top_cell, selected_layer, pixel_nm)
    x0, y0, x1, y1 = facts["bbox_dbu"]
    width_px = round((x1 - x0) * facts["dbu_nm"] / pixel_nm)
    height_px = round((y1 - y0) * facts["dbu_nm"] / pixel_nm)

    script_path = preparation_script_path or Path(__file__).resolve()
    manifest = ScaleFixtureManifest(
        source_repository=expected_repository,
        source_commit=expected_commit,
        design=design,
        gds_sha256=gds_sha,
        gds_bytes=gds_bytes,
        top_cell=facts["top_cell"],
        dbu_nm=facts["dbu_nm"],
        bbox_dbu=tuple(facts["bbox_dbu"]),
        pixel_nm=pixel_nm,
        die_size_px=(width_px, height_px),
        dense_float32_equivalent_bytes=dense_float32_equivalent_bytes(width_px, height_px),
        layers=tuple(facts["layers"]),
        selected_layer=selected_layer,
        preparation_script_sha256=sha256_file(script_path),
        split_chunks=tuple(chunks),
    )
    problems = manifest.validate()
    if problems:
        _fail("prepared manifest failed validation: " + "; ".join(problems))
    write_strict_json(output_dir / "fixture-manifest.json", manifest.to_payload())
    return manifest.to_payload()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-root", required=True, help="local PDB checkout root")
    parser.add_argument("--design", required=True, choices=sorted(DESIGN_SPECS))
    parser.add_argument(
        "--selected-layer",
        required=True,
        help="benchmark layer as LAYER:DTYPE — explicit, never inherited from another design",
    )
    parser.add_argument("--pixel-nm", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default=None,
        help="output directory (default benchmarks/results/industrial-scale/fixtures/<design>)",
    )
    parser.add_argument("--expected-commit", default=PDB_COMMIT)
    parser.add_argument("--expected-top-cell", default=None)
    parser.add_argument("--expected-gds-sha256", default=None)
    args = parser.parse_args()

    output_dir = (
        Path(args.output)
        if args.output
        else REPO / "benchmarks" / "results" / "industrial-scale" / "fixtures" / args.design
    )
    try:
        payload = prepare_fixture(
            source_root=Path(args.source_root),
            design=args.design,
            selected_layer=args.selected_layer,
            pixel_nm=args.pixel_nm,
            output_dir=output_dir,
            expected_commit=args.expected_commit,
            expected_top_cell=args.expected_top_cell,
            expected_gds_sha256=args.expected_gds_sha256,
        )
    except FixtureError as exc:
        print(f"FIXTURE PREP: FAIL — {exc}", file=sys.stderr)
        return 1
    print("FIXTURE PREP: PASS")
    print(f"  design:            {payload['design']}")
    print(f"  gds_sha256:        {payload['gds_sha256']}")
    print(f"  gds_bytes:         {payload['gds_bytes']}")
    print(f"  top_cell:          {payload['top_cell']}")
    print(f"  die_size_px:       {payload['die_size_px']} @ {payload['pixel_nm']} nm/px")
    print(
        f"  dense equivalent:  {payload['dense_float32_equivalent_bytes']} bytes "
        "(hypothetical full float32 raster — never materialized)"
    )
    print(f"  layers:            {payload['layers']}")
    print(f"  selected_layer:    {payload['selected_layer']}")
    print(f"  manifest:          {output_dir / 'fixture-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
