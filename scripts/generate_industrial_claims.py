#!/usr/bin/env python3
"""Generate public industrial claims from measured benchmark artifacts.

Reads ``OpenLithoHub.industrial-benchmark.v1`` artifacts (produced by
``benchmarks/industrial/run_industrial_benchmark.py``) and derives every
public-facing number in ``docs/generated/industrial-claims.{json,md}``.
Hand-written performance numbers in the README are not allowed: the
README quotes claim IDs, and ``--check`` fails if quoted values drift
from the artifacts.

Claim admission rules (fail-closed):

- Only measurements from validated artifacts can become claims.
- Runtime claims must be median-of-repeats (never best-of-N) and are
  headline-eligible only at speedup >= 1.1.
- Memory claims are headline-eligible only at >= 20% peak-RSS reduction.
- Quality claims are headline-eligible only at >= 5% reduction of the
  bad metric with no worse value on the reported trade-off metric.
- Surrogate-vs-levelset runtime is headline-eligible only when the
  quality metrics stay within a 10% degradation tolerance; otherwise
  the claim stays scoped as "matched-iteration runtime ratio".
- Everything produced here is REPRODUCED_INTERNAL: this repository
  cannot self-certify foundry calibration or third-party equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def _load_industrial_authority():
    """Load the stdlib-only industrial module by file path.

    The authority tooling must run in environments WITHOUT torch (the
    lint job): importing ``openlithohub.benchmark.industrial`` normally
    executes the package ``__init__`` and drags in the heavy stack.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "src/openlithohub/benchmark/industrial.py"
    spec = importlib.util.spec_from_file_location("olh_industrial_authority", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve __module__ during exec
    spec.loader.exec_module(mod)
    return mod


_ind = _load_industrial_authority()
SCHEMA_NAME = _ind.SCHEMA_NAME
REPRODUCED_INTERNAL = _ind.REPRODUCED_INTERNAL
_sanitize = _ind._sanitize
load_artifact = _ind.load_artifact
relative_reduction_pct = _ind.relative_reduction_pct


CLAIMS_SCHEMA = "OpenLithoHub.industrial-claims.v1"

SPEEDUP_HEADLINE_MIN = 1.1
MEMORY_HEADLINE_MIN_PCT = 20.0
QUALITY_HEADLINE_MIN_PCT = 5.0
SURROGATE_QUALITY_TOLERANCE_PCT = 10.0
# P0.8 (audit 2026-09-21): with 4 tiles + 1 crop, quality comparisons are
# published as scoped facts only — never as industrial headline claims —
# until the sample is spatially expanded with a paired-CI gate.
QUALITY_HEADLINE_ENABLED = False

FIREWALL = [
    "No foundry qualification without wafer/SEM calibration.",
    "No commercial-tool (Calibre/Tachyon/cuLitho) speedup or quality claim "
    "without a matched external benchmark.",
    "No neural-model quality claim from degenerate blank masks.",
    "No synthetic benchmark marketed as production wafer performance.",
    "No plugin research backend marketed as third-party validated.",
    "No speedup claim without matching quality/tolerance context.",
]


def fmt_speedup(x: float) -> str:
    return f"{x:.2f}x"


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def fmt_gib(bytes_val: float) -> str:
    return f"{bytes_val / (1 << 30):.2f} GiB"


def fmt_gb(bytes_val: float) -> str:
    return f"{bytes_val / 1e9:.2f} GB"


def hardware_label(artifact: dict[str, Any]) -> str:
    hw = artifact.get("hardware", {})
    cpu = hw.get("cpu_model", "unknown CPU")
    ram = hw.get("physical_ram_bytes")
    ram_label = f", {ram / (1 << 30):.0f} GB RAM" if ram else ""
    gpu = hw.get("gpu")
    device = "GPU" if gpu else "CPU"
    return f"{cpu} ({device}{ram_label})"


def _claim(
    *,
    claim_id: str,
    metric: str,
    value: str,
    baseline: str,
    dataset: str,
    hardware: str,
    scope: str,
    artifact: str,
    artifact_sha256: str,
    headline: bool,
    context: dict[str, Any],
    unit: str = "",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "metric": metric,
        "value": value,
        "unit": unit,
        "baseline": baseline,
        "dataset": dataset,
        "hardware": hardware,
        "scope": scope,
        "level": REPRODUCED_INTERNAL,
        "artifact": artifact,
        "artifact_sha256": artifact_sha256,
        "headline": headline,
        "context": context,
    }


def degenerate_model_static(q_aggregate: dict[str, Any], model: str) -> bool:
    return bool(q_aggregate.get(model, {}).get("degenerate_blank_any_rep", False))


def derive_claims(
    artifacts: dict[str, dict[str, Any]], hashes: dict[str, str]
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    runtime = artifacts.get("runtime")
    quality = artifacts.get("quality")
    fulldie = artifacts.get("fulldie")
    hw = hardware_label(runtime or quality or fulldie or {})
    dataset = "pdb-sky130hd-ibex (public routed Ibex RISC-V core, sky130hd)"

    if runtime is not None:
        memory = runtime.get("memory", {}).get("per_size", {})
        for size, per_size in sorted(memory.items(), key=lambda kv: int(kv[0])):
            red = per_size.get("streaming_memory_reduction_pct")
            dense = per_size.get("dense_full", {}).get("peak_rss_bytes_median")
            sel = per_size.get("b04_selective", {}).get("peak_rss_bytes_median")
            if red is None or dense is None or sel is None:
                continue
            claims.append(
                _claim(
                    claim_id=f"IB-MEM-{size}",
                    metric="peak_rss_memory_reduction",
                    value=fmt_pct(red),
                    unit="%",
                    baseline=f"dense full-raster ({fmt_gib(dense)} peak RSS)",
                    dataset=dataset,
                    hardware=hw,
                    scope=(
                        f"exact-vector selective streaming vs dense forward, {size}px center crop, "
                        "median over repeats, structural 9x9 benchmark forward model "
                        "(NOT_FOUNDRY_CALIBRATED physics)"
                    ),
                    artifact="industrial-runtime.json",
                    artifact_sha256=hashes.get("industrial-runtime.json", ""),
                    headline=red >= MEMORY_HEADLINE_MIN_PCT,
                    context={
                        "dense_peak_rss_bytes": dense,
                        "streaming_peak_rss_bytes": sel,
                        "repeats": runtime.get("policy", {}).get("repeats"),
                    },
                )
            )

        for size, comp in sorted(runtime.get("comparisons", {}).items(), key=lambda kv: int(kv[0])):
            pair = comp.get("dense_vs_selective")
            if not pair:
                continue
            speed = float(pair["speedup"])
            claims.append(
                _claim(
                    claim_id=f"IB-RT-{size}",
                    metric="wall_time_ratio_dense_to_streaming",
                    value=fmt_speedup(speed),
                    unit="x",
                    baseline="dense full-raster forward",
                    dataset=dataset,
                    hardware=hw,
                    scope=(
                        f"median execution wall time at {size}px, structural 9x9 benchmark "
                        "forward model (NOT_FOUNDRY_CALIBRATED); values < 1.0 mean dense "
                        "is faster at this size on this hardware"
                    ),
                    artifact="industrial-runtime.json",
                    artifact_sha256=hashes.get("industrial-runtime.json", ""),
                    headline=speed >= SPEEDUP_HEADLINE_MIN,
                    context=dict(pair),
                )
            )

        max_streamed = runtime.get("memory", {}).get("max_streamed_size_px")
        if max_streamed:
            sel_rows = [
                r
                for r in runtime.get("rows", [])
                if r.get("mode") == "b04_selective"
                and int(r.get("size_px", [0, 0])[0]) == int(max_streamed)
            ]
            rss = sel_rows[0]["peak_rss_bytes"]["median"] if sel_rows else None
            infeasible = runtime.get("memory", {}).get("dense_structural_infeasible_sizes_px", [])
            claims.append(
                _claim(
                    claim_id=f"IB-SCALE-{max_streamed}",
                    metric="max_layout_streamed_end_to_end",
                    value=f"{max_streamed}x{max_streamed} px",
                    baseline="dense not run at this size under the harness memory policy",
                    dataset=dataset,
                    hardware=hw,
                    scope=(
                        "exact-vector selective streaming completed end-to-end "
                        f"({max_streamed**2 / 1e9:.2f} GPx) with median peak RSS "
                        + (fmt_gib(rss) if rss else "n/a")
                        + "; dense input alone would be "
                        + fmt_gib(max_streamed**2 * 4)
                        + "; dense not run at sizes "
                        + ", ".join(f"{s}px" for s in infeasible)
                        + " under the harness memory policy (policy decision, "
                        "not a structural impossibility)"
                    ),
                    artifact="industrial-runtime.json",
                    artifact_sha256=hashes.get("industrial-runtime.json", ""),
                    headline=True,
                    context={
                        "size_px": max_streamed,
                        "streaming_peak_rss_bytes": rss,
                        "dense_not_run_under_memory_policy_px": infeasible,
                    },
                )
            )

    if quality is not None:
        datasets = quality.get("datasets", {})
        prefix_for = {"sky130hd-ibex-tiles": "IB-Q", "iccad16-testcase1": "IB-QC"}

        def candidate_degenerate(ds: dict[str, Any], model: str) -> bool:
            return bool(
                ds.get("aggregate", {}).get(model, {}).get("degenerate_blank_any_rep", False)
            )

        def pairwise_claims(
            ds: dict[str, Any], ds_id: str, prefix_id: str, candidate_model: str
        ) -> None:
            aggregate = ds.get("aggregate", {})
            base_agg = aggregate.get("dummy-identity", {})
            cand_agg = aggregate.get(candidate_model, {})
            if not base_agg or not cand_agg:
                return
            degenerate = candidate_degenerate(ds, candidate_model)
            for key, suffix, metric_label in (
                ("pvband_mean_nm", "PVB", "PV Band mean reduction (no-OPC -> ILT)"),
                (
                    "mrc_violation_rate",
                    "MRC",
                    "benchmark MRC violating-pixel fraction reduction (no-OPC -> ILT)",
                ),
                ("wafer_epe_mean_nm", "WEPE", "wafer EPE mean reduction (no-OPC -> ILT)"),
            ):
                base_med = float(base_agg.get(key, {}).get("median", 0.0) or 0.0)
                cand_med = float(cand_agg.get(key, {}).get("median", 0.0) or 0.0)
                red = relative_reduction_pct(base_med, cand_med)
                abs_delta = cand_med - base_med
                if degenerate or not math.isfinite(red) or not math.isfinite(abs_delta):
                    # A blank mask "wins" MRC/PVB trivially and makes wafer
                    # EPE infinite — record the fact, never the percentage.
                    value = "DEGENERATE (blank mask)" if degenerate else "NOT MEASURABLE"
                    headline = False
                else:
                    value = (
                        f"{fmt_pct(red)} ({base_med:.5f} -> {cand_med:.5f}, delta {abs_delta:+.5f})"
                    )
                    headline = QUALITY_HEADLINE_ENABLED and red >= QUALITY_HEADLINE_MIN_PCT
                trade_off = ""
                if key == "wafer_epe_mean_nm" and headline:
                    base_mrc = float(
                        base_agg.get("mrc_violation_rate", {}).get("median", 0.0) or 0.0
                    )
                    cand_mrc = float(
                        cand_agg.get("mrc_violation_rate", {}).get("median", 0.0) or 0.0
                    )
                    if base_mrc > 0 and (cand_mrc - base_mrc) / base_mrc > 0.20:
                        # A wafer-EPE win that substantially degrades the
                        # hard-fail MRC metric is a trade-off, not a headline.
                        headline = False
                        trade_off = (
                            f" trade-off: MRC violation rate rises {base_mrc:.5f} -> {cand_mrc:.5f}"
                        )
                claims.append(
                    _claim(
                        claim_id=f"{prefix_id}-{suffix}",
                        metric=f"{metric_label} ({candidate_model})",
                        value=value,
                        unit="%",
                        baseline="dummy-identity (design-as-mask, no OPC)",
                        dataset=ds_id,
                        hardware=hw,
                        scope=(
                            "same Hopkins SOCS optical model for every method, same resist "
                            "threshold, same pixel size, same real-layout tiles, same metric "
                            "implementations; benchmark-relative, NOT_FOUNDRY_CALIBRATED"
                            + (
                                f" — DEGENERATE OUTPUT: {candidate_model} produced a "
                                "(near-)blank mask on at least one tile; reduction "
                                "percentages are not meaningful and are never claims"
                                if degenerate
                                else ""
                            )
                            + trade_off
                        ),
                        artifact="industrial-quality.json",
                        artifact_sha256=hashes.get("industrial-quality.json", ""),
                        headline=headline,
                        context={
                            "metric": key,
                            "baseline_median": base_med,
                            "candidate_median": cand_med,
                            "reduction_pct": red,
                            "dataset_scope": ds.get("description", ds_id),
                            "policy": ds.get("policy", {}),
                            "candidate_degenerate_blank": degenerate,
                            "trade_off": trade_off,
                        },
                    )
                )

        for ds_name, ds in datasets.items():
            prefix = prefix_for.get(ds_name, "IB-QX")
            ds_id = ds_name
            pairwise_claims(ds, ds_id, f"{prefix}-ILT", "levelset-ilt")
            pairwise_claims(ds, ds_id, f"{prefix}-RB", "rule-based-opc")

            surr = ds.get("comparisons", {}).get("levelset_vs_surrogate_runtime")
            if surr and surr.get("candidate_median_s"):
                speed = (
                    float(surr["baseline_median_s"]) / float(surr["candidate_median_s"])
                    if float(surr["candidate_median_s"]) > 0
                    else 0.0
                )
                q_aggregate = ds.get("aggregate", {})
                lv_pvb = float(
                    q_aggregate.get("levelset-ilt", {}).get("pvband_mean_nm", {}).get("median", 0.0)
                    or 0.0
                )
                sg_pvb = float(
                    q_aggregate.get("surrogate-ilt", {})
                    .get("pvband_mean_nm", {})
                    .get("median", 0.0)
                    or 0.0
                )
                lv_wepe = float(
                    q_aggregate.get("levelset-ilt", {})
                    .get("wafer_epe_mean_nm", {})
                    .get("median", 0.0)
                    or 0.0
                )
                sg_wepe = float(
                    q_aggregate.get("surrogate-ilt", {})
                    .get("wafer_epe_mean_nm", {})
                    .get("median", 0.0)
                    or 0.0
                )

                lv_degenerate = degenerate_model_static(q_aggregate, "levelset-ilt")
                sg_degenerate = degenerate_model_static(q_aggregate, "surrogate-ilt")

                def degraded(base: float, cand: float) -> bool:
                    if base <= 0 or not math.isfinite(base) or not math.isfinite(cand):
                        return True
                    return (cand - base) / base * 100.0 > SURROGATE_QUALITY_TOLERANCE_PCT

                invalid = (
                    lv_degenerate
                    or sg_degenerate
                    or not math.isfinite(lv_pvb)
                    or not math.isfinite(sg_pvb)
                    or not math.isfinite(lv_wepe)
                    or not math.isfinite(sg_wepe)
                )
                if invalid:
                    quality_ok = False
                else:
                    quality_ok = not (degraded(lv_pvb, sg_pvb) or degraded(lv_wepe, sg_wepe))
                claims.append(
                    _claim(
                        claim_id=f"{prefix}-SURR",
                        metric="matched-iteration optimization runtime ratio",
                        value=fmt_speedup(speed),
                        unit="x",
                        baseline="levelset-ilt",
                        dataset=f"{ds_id}",
                        hardware=hw,
                        scope=(
                            "surrogate-ilt vs levelset-ilt at the same iteration budget on "
                            f"{ds_id} with the recorded reduced surrogate training budget; "
                            + (
                                "quality within tolerance at this budget"
                                if quality_ok
                                else (
                                    "QUALITY_COMPARISON_INVALID: degenerate or non-finite "
                                    "metrics on at least one side"
                                    if invalid
                                    else "NOT quality-normalized at this budget: surrogate "
                                    "quality degrades beyond tolerance, so this is NOT a "
                                    "matched-quality speedup"
                                )
                            )
                        ),
                        artifact="industrial-quality.json",
                        artifact_sha256=hashes.get("industrial-quality.json", ""),
                        headline=(
                            QUALITY_HEADLINE_ENABLED
                            and speed >= SPEEDUP_HEADLINE_MIN
                            and quality_ok
                        ),
                        context={
                            **surr,
                            "levelset_pvband_mean_nm": lv_pvb,
                            "surrogate_pvband_mean_nm": sg_pvb,
                            "levelset_wafer_epe_mean_nm": lv_wepe,
                            "surrogate_wafer_epe_mean_nm": sg_wepe,
                        },
                    )
                )

    if fulldie is not None:
        die_bytes = fulldie.get("dense_die_raster_bytes_structural", 0)
        status = fulldie.get("dense_die_status")
        if die_bytes and status == "INFEASIBLE_ON_REFERENCE_MACHINE":
            claims.append(
                _claim(
                    claim_id="IB-DIE-1",
                    metric="full-die dense raster infeasibility",
                    value=f"{die_bytes / 1e12:.2f} TB ({die_bytes / (1 << 40):.2f} TiB)",
                    baseline="physical RAM of the benchmark machine",
                    dataset=dataset,
                    hardware=hw,
                    scope=(
                        "machine-relative arithmetic (die pixels x 4 bytes vs physical RAM): "
                        "the routed ibex die cannot be dense-rasterized at 1nm/px on the "
                        "48 GiB reference machine, while per-tile streaming fixtures "
                        "complete with O(tile) memory"
                    ),
                    artifact="industrial-fulldie.json",
                    artifact_sha256=hashes.get("industrial-fulldie.json", ""),
                    headline=True,
                    context={
                        "die_size_px": fulldie.get("die_size_px"),
                        "physical_ram_bytes": fulldie.get("physical_ram_bytes"),
                    },
                )
            )
        rows = fulldie.get("sample_rows", [])
        if rows:
            lo, hi = fulldie.get("sampled_screened_fraction_range", [0.0, 0.0])
            claims.append(
                _claim(
                    claim_id="IB-DIE-SURVEY",
                    metric="certified empty-context screening on sampled die tiles",
                    value=f"{lo * 100:.1f}%-{hi * 100:.1f}% screened",
                    baseline="n/a (screen-only diagnostic)",
                    dataset=dataset,
                    hardware=hw,
                    scope=(
                        "screen-only diagnostic (forward_simulator_calls=0) on "
                        f"{fulldie.get('n_sample_tiles')} sampled die tiles; full-die survey "
                        "cost is an ESTIMATE in the artifact, not a measurement"
                    ),
                    artifact="industrial-fulldie.json",
                    artifact_sha256=hashes.get("industrial-fulldie.json", ""),
                    headline=False,
                    context={
                        "full_die_survey_wall_s_estimate": fulldie.get(
                            "full_die_survey_wall_s_estimate"
                        ),
                        "known_scaling_limit": fulldie.get("known_scaling_limit"),
                    },
                )
            )

    return claims


def render_markdown(claims: list[dict[str, Any]], artifacts: dict[str, Any], hw: str) -> str:
    lines: list[str] = []
    lines.append("<!-- GENERATED by scripts/generate_industrial_claims.py — DO NOT EDIT -->")
    lines.append("")
    lines.append("# Industrial Benchmark v1 — generated claims")
    lines.append("")
    lines.append(
        "Every number below is derived from checked-in benchmark artifacts "
        "(``benchmarks/results/industrial/``). The README may only quote "
        "claims marked **headline** by their ID; `scripts/generate_industrial_claims.py "
        "--check` enforces this."
    )
    lines.append("")
    lines.append(
        f"Reference hardware: {hw}. Claim level: `REPRODUCED_INTERNAL` unless stated otherwise."
    )
    lines.append("")
    headline = [c for c in claims if c["headline"]]
    if headline:
        lines.append("## Headline-eligible claims")
        lines.append("")
        lines.append("| ID | Metric | Value | Baseline | Scope summary |")
        lines.append("|---|---|---|---|---|")
        for c in headline:
            scope = c["scope"]
            scope_short = scope if len(scope) <= 160 else scope[:157] + "..."
            row = (
                f"| `{c['claim_id']}` | {c['metric']} | {c['value']} "
                f"| {c['baseline']} | {scope_short} |"
            )
            lines.append(row)
        lines.append("")
    else:
        lines.append("## Headline-eligible claims")
        lines.append("")
        lines.append("_None yet — benchmark in progress. No number is claimed._")
        lines.append("")
    lines.append("## All measured comparisons (facts, not marketing)")
    lines.append("")
    lines.append("| ID | Metric | Value | Headline |")
    lines.append("|---|---|---|---|")
    for c in claims:
        lines.append(f"| `{c['claim_id']}` | {c['metric']} | {c['value']} | {c['headline']} |")
    lines.append("")
    lines.append("## Claim firewall — still forbidden")
    lines.append("")
    for rule in FIREWALL:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name, sha in sorted(artifacts.items()):
        lines.append(f"- `{name}` sha256=`{sha}`")
    lines.append("")
    return "\n".join(lines)


def check_readme(claims: list[dict[str, Any]], readme_path: Path) -> list[str]:
    """Verify the README quotes only real claims with exact values."""
    problems: list[str] = []
    by_id = {c["claim_id"]: c for c in claims}
    text = readme_path.read_text(encoding="utf-8")
    quoted = re.findall(r"`?(IB-[A-Z0-9-]+)`?", text)
    for claim_id in quoted:
        claim = by_id.get(claim_id)
        if claim is None:
            problems.append(f"README quotes unknown claim id {claim_id}")
            continue
        if not claim["headline"]:
            # Non-headline facts may be referenced in prose; only headline
            # claims must quote the exact artifact value.
            continue
        for line in text.splitlines():
            if claim_id in line and claim["value"] in line:
                break
        else:
            problems.append(
                f"README mentions {claim_id} without its exact artifact value {claim['value']!r}"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--artifacts", type=Path, default=repo / "benchmarks/results/industrial")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=repo / "docs/generated/industrial-claims.json",
    )
    ap.add_argument("--out-md", type=Path, default=repo / "docs/generated/industrial-claims.md")
    ap.add_argument("--readme", type=Path, default=repo / "README.md")
    ap.add_argument("--check", action="store_true", help="validate README quotes and exit")
    args = ap.parse_args()

    manifest_path = args.artifacts / "manifest.json"
    if not manifest_path.exists():
        if args.check:
            # No artifacts yet → no headline claims exist → nothing to drift.
            print(
                f"no industrial artifacts at {args.artifacts} — "
                "claims check skipped (benchmark not yet measured)"
            )
            return 0
        raise SystemExit(f"missing {manifest_path} — run the industrial benchmark first")
    manifest = load_artifact(manifest_path)

    names = ["industrial-runtime.json", "industrial-quality.json", "industrial-fulldie.json"]
    artifacts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for entry in manifest.get("artifacts", []):
        name = entry["file"]
        if name in names:
            path = args.artifacts / name
            if sha256_file := entry.get("sha256"):
                actual = _sha(path)
                if actual != sha256_file:
                    raise SystemExit(
                        f"artifact hash mismatch for {name}: "
                        f"manifest {sha256_file} != actual {actual}"
                    )
            hashes[name] = entry["sha256"]
            artifacts[name.replace("industrial-", "").replace(".json", "")] = load_artifact(path)
    loaded_names = {
        entry["file"]
        for entry in manifest.get("artifacts", [])
        if entry["file"] in names
        and entry["file"].replace("industrial-", "").replace(".json", "") in artifacts
    }
    missing = set(names) - loaded_names
    if missing:
        print(
            f"warning: missing artifacts (claims limited to what exists): {sorted(missing)}",
            file=sys.stderr,
        )

    claims = derive_claims(artifacts, hashes)
    hw = hardware_label(
        artifacts.get("runtime") or artifacts.get("quality") or artifacts.get("fulldie") or {}
    )
    doc = {
        "schema": CLAIMS_SCHEMA,
        "generated_from": SCHEMA_NAME,
        "git_commit": manifest.get("git_commit"),
        "timestamp_utc": manifest.get("timestamp_utc"),
        "hardware": hw,
        "artifacts": hashes,
        "firewall": FIREWALL,
        "claims": claims,
    }

    expected_json = json.dumps(_sanitize(doc), indent=2, sort_keys=True, allow_nan=False) + "\n"
    expected_md = render_markdown(claims, hashes, hw)

    if args.check:
        # B0.6: a TRUE drift gate — deterministically re-render the claims
        # documents from the artifacts and byte-compare with the checked-in
        # files. Stale or hand-edited generated files fail even when no
        # README quote references them.
        problems: list[str] = []
        if not args.out_json.exists():
            problems.append(f"{args.out_json} missing — run without --check to generate")
        elif args.out_json.read_text(encoding="utf-8") != expected_json:
            problems.append(
                f"{args.out_json} drifted from the artifacts (regenerate without --check)"
            )
        if not args.out_md.exists():
            problems.append(f"{args.out_md} missing — run without --check to generate")
        elif args.out_md.read_text(encoding="utf-8") != expected_md:
            problems.append(
                f"{args.out_md} drifted from the artifacts (regenerate without --check)"
            )
        problems.extend(check_readme(claims, args.readme))
        for p in problems:
            print(f"CHECK FAIL: {p}", file=sys.stderr)
        return 1 if problems else 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(expected_json, encoding="utf-8")
    args.out_md.write_text(expected_md, encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md} ({len(claims)} claims)")
    return 0


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
