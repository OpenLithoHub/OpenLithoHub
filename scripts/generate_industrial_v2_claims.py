"""Industrial Benchmark v2 claim generator (PR-G §29).

Derives generated claims from a VERIFIED canonical v2 family:

    docs/generated/industrial-v2-claims.json
    docs/generated/industrial-v2-claims.md

README v2 numbers may only come from these outputs (§21/§48).  The
``--check`` mode fails CI when a README references an ``IB2-*`` claim id
that the generated claims do not carry (B2-J: a GPU number before
canonical publication fails CI), and vice versa: generated headline
claims must appear in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openlithohub.benchmark.industrial_v2 import (  # noqa: E402
    SCHEMA_NAME,
    admit_headline,
)

GENERATED_JSON = REPO / "docs" / "generated" / "industrial-v2-claims.json"
GENERATED_MD = REPO / "docs" / "generated" / "industrial-v2-claims.md"
README = REPO / "README.md"


def load_verified_family(canonical_root: Path) -> dict:
    """Run the v2 verifier first: claims may only be generated from a
    family that closes (§29)."""
    import scripts.verify_industrial_v2_artifacts as verifier

    verifier.verify(canonical_root)  # raises VerifyError on any drift
    run_config = json.loads((canonical_root / "industrial-v2-run-config.json").read_text())
    index = json.loads((canonical_root / "industrial-v2-index.json").read_text())
    gpu = json.loads((canonical_root / "industrial-v2-gpu-runtime.json").read_text())
    hopkins = json.loads((canonical_root / "industrial-v2-hopkins.json").read_text())
    return {"run_config": run_config, "tiers": {"a": index, "b": gpu, "c": hopkins}}


def build_claims(family: dict) -> dict:
    """Headline admission runs per claim (§26).  Phase 2A publishes only
    throughput/memory claims — never model quality (§27)."""
    claims: list[dict] = []
    run_config = family["run_config"]["run_config"]
    repeats = int(run_config.get("repeat_count", 0))

    index = family["tiers"]["a"]
    for row in index.get("rows", []):
        claim_id = f"IB2-INDEX-QUERY-{row.get('window', 0)}"
        reduction = row.get("candidate_reduction_pct")
        admitted, reason = admit_headline(
            claim_id=claim_id,
            correctness_pass=bool(row.get("correctness_witness_pass")),
            repeat_count=repeats,
            memory_reduction=(
                float(reduction) / 100.0 if reduction is not None else None
            ),
            scope="Tier A indexed exact-vector window discovery on the declared "
            "fixture/hardware; same exact run semantics",
            status=row.get("status", "FAILED"),
        )
        claims.append(
            {
                "claim_id": claim_id,
                "admitted": admitted,
                "reason": reason,
                "value": row.get("candidate_reduction_pct"),
                "unit": "candidate_reduction_pct",
                "scope": "architecture-only indexed geometry throughput",
            }
        )

    gpu = family["tiers"]["b"]
    for row in gpu.get("rows", []):
        wall1 = float(row.get("gpu_batch1_wall_s", 0) or 0)
        walln = float(row.get("gpu_batch_n_wall_s", 0) or 0)
        speedup = round(wall1 / walln, 4) if walln else None
        claim_id = f"IB2-GPU-BATCH-{row.get('window', 0)}"
        admitted, reason = admit_headline(
            claim_id=claim_id,
            correctness_pass=bool(row.get("correctness_witness_pass")),
            repeat_count=repeats,
            runtime_speedup=speedup,
            scope="Tier B single-GPU streaming execution (finite-support "
            "forward; architecture-only, no model-quality claim)",
            status=row.get("status", "FAILED"),
        )
        claims.append(
            {
                "claim_id": claim_id,
                "admitted": admitted,
                "reason": reason,
                "value": speedup,
                "unit": "batching_runtime_speedup_x",
                "scope": "GPU streaming architecture throughput",
            }
        )

    return {
        "schema": SCHEMA_NAME,
        "claim_level": "REPRODUCED_INTERNAL",
        "run_identity": family["run_config"]["run_identity"],
        "measurement_commit": family["run_config"]["measurement_commit"],
        "claims": claims,
        "note": "No commercial-tool comparison. Not foundry calibrated. "
        "Hardware-specific measurements. No model-quality claim.",
    }


def render_markdown(claims: dict) -> str:
    lines = [
        "# Industrial Benchmark v2 — generated claims",
        "",
        f"Run identity: `{claims['run_identity'][:16]}…` · "
        f"measurement commit `{claims['measurement_commit'][:12]}…` · "
        f"level `{claims['claim_level']}`",
        "",
        "| Claim | Admitted | Value | Scope |",
        "|---|---|---|---|",
    ]
    for claim in claims["claims"]:
        lines.append(
            f"| `{claim['claim_id']}` | {'yes' if claim['admitted'] else 'no'} "
            f"| {claim['value']} {claim['unit']} | {claim['scope']} |"
        )
    lines += ["", claims["note"], ""]
    return "\n".join(lines)


def check_readme_drift(claims: dict) -> int:
    """B2-J: README GPU numbers must be generated from the canonical v2
    family.  Any ``IB2-*`` id in the README that the generated claims do
    not carry (or an unadmitted one used as fact) fails."""
    readme_text = README.read_text()
    referenced = set(_extract_ib2_ids(readme_text))
    known = {claim["claim_id"] for claim in claims["claims"] if claim["admitted"]}
    drift = sorted(referenced - known)
    if drift:
        print(
            f"CLAIMS CHECK: FAIL — README references v2 claims not admitted "
            f"by the canonical family: {drift}",
            file=sys.stderr,
        )
        return 1
    print(
        f"CLAIMS CHECK: OK — README v2 references ({sorted(referenced) or 'none'}) "
        "match generated admitted claims"
    )
    return 0


def _extract_ib2_ids(text: str) -> list[str]:
    import re

    return re.findall(r"IB2-[A-Z0-9-]+", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root", default="benchmarks/results/industrial-v2"
    )
    parser.add_argument("--check", action="store_true", help="README drift gate only")
    args = parser.parse_args()

    if args.check:
        if not GENERATED_JSON.exists():
            print(
                "CLAIMS CHECK: OK — no generated v2 claims exist; a README that "
                "references none is consistent",
                file=sys.stderr,
            )
            readme_ids = _extract_ib2_ids(README.read_text())
            if readme_ids:
                print(
                    f"CLAIMS CHECK: FAIL — README references {sorted(set(readme_ids))} "
                    "without any generated v2 claims (B2-J)",
                    file=sys.stderr,
                )
                return 1
            return 0
        claims = json.loads(GENERATED_JSON.read_text())
        return check_readme_drift(claims)

    try:
        family = load_verified_family(Path(args.canonical_root))
    except Exception as exc:  # noqa: BLE001 — verifier errors surface verbatim
        print(f"CLAIM GENERATOR: FAIL — {exc}", file=sys.stderr)
        return 1
    claims = build_claims(family)
    GENERATED_JSON.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_JSON.write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n")
    GENERATED_MD.write_text(render_markdown(claims))
    print(f"CLAIM GENERATOR: wrote {GENERATED_JSON.name} / {GENERATED_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
