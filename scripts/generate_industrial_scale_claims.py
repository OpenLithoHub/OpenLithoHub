"""Industrial Scale claim generator (scale track, S11).

Derives claims ONLY from a verified scale family.  During the
development phase every claim is:

    PROVISIONAL / INTERNAL — NOT PERFORMANCE AUTHORITY

and the ``--check`` mode enforces the headline firewall: the README may
not reference ANY ``ISC-*`` scale claim id until the formal scale
protocol closes on a CUDA host (making this generator's rule change a
protocol change, not a post-hoc decision).

Usage::

    python scripts/generate_industrial_scale_claims.py \\
        --root <family-dir> [--out-dir <dir>]
    python scripts/generate_industrial_scale_claims.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

README = REPO / "README.md"
README_ZH = REPO / "README_zh.md"
ISC_ID_RE = re.compile(r"ISC-[A-Z0-9-]+")

PROVISIONAL_NOTE = (
    "PROVISIONAL / INTERNAL — NOT PERFORMANCE AUTHORITY. "
    "Scale characterization only: not v1.1 authority, not v2 authority, "
    "not foundry calibrated, no commercial-tool comparison. "
    "Dense raster equivalents are derived from geometry; such rasters "
    "were never materialized. Multi-worker CPU emulation proves scheduler "
    "semantics only and is never multi-GPU performance."
)


def _load_verifier() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_industrial_scale_artifacts",
        Path(__file__).resolve().parent / "verify_industrial_scale_artifacts.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_industrial_scale_artifacts"] = module
    spec.loader.exec_module(module)
    return module


def build_claims(root: Path) -> dict:
    """Build the provisional claim set from a VERIFIED family (the
    verifier runs first; claims may only come from a closed family)."""
    verifier = _load_verifier()
    verifier.verify(root)
    run_config = json.loads((root / "industrial-scale-run-config.json").read_text())
    claims: list[dict] = []
    for member_name, lane in (
        ("industrial-scale-index.json", "A_LARGE_LAYOUT_STREAMING"),
        ("industrial-scale-runtime.json", "B_MULTI_GPU_SCALING"),
    ):
        payload = json.loads((root / member_name).read_text())
        for row in payload.get("rows", []):
            claim_id = f"ISC-{lane.split('_')[0]}-{row.get('window', 0)}"
            claims.append(
                {
                    "claim_id": claim_id,
                    "lane": lane,
                    "status": row.get("status"),
                    "level": "PROVISIONAL_INTERNAL",
                    "headline_eligible": False,
                    "admitted": False,
                    "value": None,
                    "reason": "scale formal protocol not executed on a CUDA host",
                }
            )
    return {
        "schema": "OpenLithoHub.industrial-scale-benchmark.v1",
        "level": "PROVISIONAL_INTERNAL",
        "run_identity": run_config.get("run_identity"),
        "measurement_commit": run_config.get("measurement_commit"),
        "provisional": run_config.get("provisional"),
        "claims": claims,
        "note": PROVISIONAL_NOTE,
    }


def render_markdown(claims: dict) -> str:
    lines = [
        "# Industrial Scale — generated claims (PROVISIONAL)",
        "",
        f"Run identity: `{str(claims['run_identity'])[:16]}…` · "
        f"commit `{str(claims['measurement_commit'])[:12]}…` · "
        f"level `{claims['level']}`",
        "",
        "| Claim | Lane | Status | Admitted |",
        "|---|---|---|---|",
    ]
    for claim in claims["claims"]:
        lines.append(
            f"| `{claim['claim_id']}` | {claim['lane']} | {claim['status']} | "
            f"{'yes' if claim['admitted'] else 'no (provisional)'} |"
        )
    lines += ["", claims["note"], ""]
    return "\n".join(lines)


def check_readme_firewall() -> int:
    """Headline firewall (S11): no ISC-* claim may appear in either
    README while the scale track is provisional."""
    failures = 0
    for readme in (README, README_ZH):
        if not readme.is_file():
            continue
        ids = sorted(set(ISC_ID_RE.findall(readme.read_text())))
        if ids:
            print(
                f"SCALE CLAIMS CHECK: FAIL — {readme.name} references provisional "
                f"scale claims {ids} (no ISC-* headline is allowed while the "
                "scale protocol is provisional)",
                file=sys.stderr,
            )
            failures += 1
    if not failures:
        print("SCALE CLAIMS CHECK: OK — no ISC-* references in any README")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="scale family root (verified before claiming)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <root>/derived — keeps the family exact)",
    )
    parser.add_argument("--check", action="store_true", help="README headline firewall only")
    args = parser.parse_args()

    if args.check:
        return check_readme_firewall()
    if not args.root:
        parser.error("--root is required unless --check")
    root = Path(args.root)
    try:
        claims = build_claims(root)
    except Exception as exc:  # noqa: BLE001 — verifier errors surface verbatim
        print(f"SCALE CLAIM GENERATOR: FAIL — {exc}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else root / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "industrial-scale-claims.json").write_text(
        json.dumps(claims, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "industrial-scale-claims.md").write_text(render_markdown(claims))
    print(
        f"SCALE CLAIM GENERATOR: wrote provisional claims to {out_dir} "
        "(level PROVISIONAL_INTERNAL, nothing admitted)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
