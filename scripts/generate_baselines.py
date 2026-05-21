"""Generate reproducible baseline benchmark numbers for the bundled ILT models.

Two modes:

- ``--synthetic`` (default): builds 8 synthetic 64×64 layouts (lines, T, L,
  squares) and runs every registered baseline model on them. No external
  dataset required; suitable for CI and for the published headline numbers
  in docs/benchmarks.md.
- ``--data-root <path>``: pulls samples from a real dataset adapter
  (lithobench by default). Use this when you have downloaded LithoBench
  locally and want production numbers.

Outputs:

- ``<output>/results.json``: structured per-model metric records.
- ``<output>/results.md``: a markdown table ready to paste into docs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Register built-in models on import.
import openlithohub.models.examples.dummy_model  # noqa: F401
import openlithohub.models.levelset_ilt  # noqa: F401
import openlithohub.models.neural_ilt  # noqa: F401
import openlithohub.models.openilt  # noqa: F401
import openlithohub.models.rule_based_opc  # noqa: F401
from openlithohub.benchmark.compliance.mrc import check_mrc
from openlithohub.benchmark.metrics.epe import compute_epe, compute_wafer_epe
from openlithohub.benchmark.metrics.l2_error import compute_l2_error
from openlithohub.benchmark.metrics.pvband import compute_pvband
from openlithohub.data.base import LithoSample
from openlithohub.models.registry import registry


@dataclass
class SyntheticPattern:
    name: str
    design: torch.Tensor
    target_mask: torch.Tensor


@dataclass
class BaselineRecord:
    model: str
    dataset: str
    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def build_synthetic_patterns(grid: int = 64) -> list[SyntheticPattern]:
    """Hand-rolled patterns covering common layout primitives."""
    patterns: list[SyntheticPattern] = []

    # 1. Centered square
    sq = torch.zeros(grid, grid)
    sq[grid // 4 : 3 * grid // 4, grid // 4 : 3 * grid // 4] = 1.0
    patterns.append(SyntheticPattern("square", sq, sq.clone()))

    # 2. Single horizontal line
    hl = torch.zeros(grid, grid)
    hl[grid // 2 - 4 : grid // 2 + 4, 8 : grid - 8] = 1.0
    patterns.append(SyntheticPattern("h-line", hl, hl.clone()))

    # 3. Pair of vertical lines (line/space)
    ls = torch.zeros(grid, grid)
    ls[8 : grid - 8, 16:24] = 1.0
    ls[8 : grid - 8, 40:48] = 1.0
    patterns.append(SyntheticPattern("line-space", ls, ls.clone()))

    # 4. T-junction
    tj = torch.zeros(grid, grid)
    tj[grid // 2 - 4 : grid // 2 + 4, 12 : grid - 12] = 1.0
    tj[grid // 2 - 4 : grid - 12, grid // 2 - 4 : grid // 2 + 4] = 1.0
    patterns.append(SyntheticPattern("T", tj, tj.clone()))

    # 5. L-corner
    lc = torch.zeros(grid, grid)
    lc[12 : grid - 12, 12:20] = 1.0
    lc[grid - 20 : grid - 12, 12 : grid - 12] = 1.0
    patterns.append(SyntheticPattern("L", lc, lc.clone()))

    # 6. Cross
    cr = torch.zeros(grid, grid)
    cr[grid // 2 - 4 : grid // 2 + 4, 8 : grid - 8] = 1.0
    cr[8 : grid - 8, grid // 2 - 4 : grid // 2 + 4] = 1.0
    patterns.append(SyntheticPattern("cross", cr, cr.clone()))

    # 7. Sparse contacts
    ct = torch.zeros(grid, grid)
    for cy in (16, 32, 48):
        for cx in (16, 32, 48):
            ct[cy - 3 : cy + 3, cx - 3 : cx + 3] = 1.0
    patterns.append(SyntheticPattern("contacts", ct, ct.clone()))

    # 8. Dense lines
    dl = torch.zeros(grid, grid)
    for col in range(8, grid - 8, 8):
        dl[8 : grid - 8, col : col + 4] = 1.0
    patterns.append(SyntheticPattern("dense-lines", dl, dl.clone()))

    return patterns


def patterns_to_samples(patterns: Iterable[SyntheticPattern]) -> list[LithoSample]:
    samples: list[LithoSample] = []
    for p in patterns:
        samples.append(
            LithoSample(
                design=p.design,
                mask=p.target_mask,
                resist=None,
                metadata={"sample_id": p.name, "pixel_nm": 1.0},
            )
        )
    return samples


def load_dataset_samples(data_root: Path, pixel_nm: float, limit: int) -> list[LithoSample]:
    from openlithohub.data import LithoBenchDataset

    adapter = LithoBenchDataset(root=data_root, pixel_nm=pixel_nm)
    n = min(len(adapter), limit) if limit else len(adapter)
    return [adapter[i] for i in range(n)]


def evaluate_model(
    model_name: str,
    samples: list[LithoSample],
    pixel_nm: float,
    *,
    run_pvband: bool,
    run_mrc: bool,
    run_wafer: bool,
    min_width_nm: float,
    min_spacing_nm: float,
    model_kwargs: dict[str, Any] | None = None,
    simulator: Any | None = None,
) -> BaselineRecord | None:
    try:
        model = registry.get(model_name, **(model_kwargs or {}))
    except KeyError:
        return None

    try:
        model.setup()
    except Exception as exc:  # noqa: BLE001 — baseline runner should be tolerant
        return BaselineRecord(
            model=model_name,
            dataset="",
            num_samples=0,
            notes=f"setup failed: {exc!r}",
        )

    per_sample: list[dict[str, float]] = []
    for sample in samples:
        try:
            result = model.predict(sample.design)
        except Exception:  # noqa: BLE001
            per_sample.append({"_error": 1.0})
            continue

        row: dict[str, float] = {}
        if sample.mask is not None:
            epe = compute_epe(result.mask, sample.mask, pixel_size_nm=pixel_nm)
            # ``compute_epe`` returns a ``valid`` flag describing edge-set
            # health — copy only the numeric fields so we don't try to
            # average a bool with the L2 / EPE scalars.
            for k in ("epe_mean_nm", "epe_max_nm", "epe_std_nm"):
                if k in epe:
                    row[k] = float(epe[k])
        if run_pvband:
            with contextlib.suppress(Exception):
                row.update(compute_pvband(result.mask, pixel_size_nm=pixel_nm))
        # Wafer-level metrics: simulate then score against the layout target.
        # These are what the leaderboard ranks on (Neural-ILT contract); a
        # mask-EPE-only scorecard ties Identity / OpenILT / Neural-ILT at 0.
        if run_wafer and sample.mask is not None:
            with contextlib.suppress(Exception):
                wafer_epe = compute_wafer_epe(
                    result.mask, sample.mask, pixel_size_nm=pixel_nm, simulator=simulator
                )
                row["epe_wafer_mean_nm"] = float(wafer_epe["epe_mean_nm"])
                row["epe_wafer_max_nm"] = float(wafer_epe["epe_max_nm"])
            with contextlib.suppress(Exception):
                l2 = compute_l2_error(
                    result.mask, sample.mask, pixel_size_nm=pixel_nm, simulator=simulator
                )
                row["l2_error_pixels"] = float(l2["l2_error_pixels"])
                row["l2_error_nm2"] = float(l2["l2_error_nm2"])
        if run_mrc:
            with contextlib.suppress(Exception):
                mrc = check_mrc(
                    result.mask,
                    min_width_nm=min_width_nm,
                    min_spacing_nm=min_spacing_nm,
                    pixel_size_nm=pixel_nm,
                )
                row["mrc_violation_rate"] = mrc.violation_rate
                row["mrc_passed"] = 1.0 if mrc.passed else 0.0
        if row:
            per_sample.append(row)

    model.teardown()

    if not per_sample:
        return BaselineRecord(model=model_name, dataset="", num_samples=0, notes="no metrics")

    keys: set[str] = set()
    for r in per_sample:
        keys.update(r.keys())

    # Wafer-level metrics share a sample row (one forward simulation feeds
    # both EPE and L2). When a row has a non-finite wafer-EPE — common for
    # synthetic patterns at 8 nm/px where one polarity of the edge set is
    # empty after diffraction blur — its L2 is also unreliable, so drop the
    # entire wafer block from that row instead of dropping wafer-EPE alone
    # while keeping its L2. Otherwise the two columns get computed over
    # different sample subsets and the published baselines stop being
    # comparable across rows.
    wafer_keys = {"epe_wafer_mean_nm", "epe_wafer_max_nm", "l2_error_pixels", "l2_error_nm2"}
    for r in per_sample:
        if any(
            k in r and not torch.isfinite(torch.tensor(r[k])) for k in wafer_keys & set(r.keys())
        ):
            for k in wafer_keys:
                r.pop(k, None)

    aggregated: dict[str, float] = {}
    for key in sorted(keys):
        if key.startswith("_"):
            continue
        # Drop non-finite per-sample contributions (e.g. mask-level EPE
        # returns inf when one polarity is empty). Wafer fields were
        # already gated above as a coherent row group.
        vals = [r[key] for r in per_sample if key in r and torch.isfinite(torch.tensor(r[key]))]
        if vals:
            aggregated[key] = float(torch.tensor(vals).mean().item())

    return BaselineRecord(
        model=model_name,
        dataset="",
        num_samples=len(samples),
        metrics=aggregated,
    )


SYNTHETIC_BEGIN = "<!-- AUTO-GENERATED: synthetic-baselines BEGIN -->"
SYNTHETIC_END = "<!-- AUTO-GENERATED: synthetic-baselines END -->"


def _merge_synthetic_section(md_path: Path, new_synthetic_block: str) -> str:
    """Replace only the synthetic-baseline section delimited by sentinels.

    Manually-curated sections (ASAP7, FreePDK45, ORFS) live outside the
    sentinels and are preserved verbatim. If the file has no sentinels yet,
    treat any leading "Baseline results" header block as the synthetic
    section and rewrite it in place — anything after the first hand-written
    "## " heading is preserved.
    """
    wrapped = f"{SYNTHETIC_BEGIN}\n{new_synthetic_block.rstrip()}\n{SYNTHETIC_END}\n"
    if not md_path.exists():
        return wrapped
    existing = md_path.read_text()
    if SYNTHETIC_BEGIN in existing and SYNTHETIC_END in existing:
        prefix, _, rest = existing.partition(SYNTHETIC_BEGIN)
        _, _, suffix = rest.partition(SYNTHETIC_END)
        return f"{prefix}{wrapped.rstrip()}{suffix}"
    second_h2 = existing.find("\n## ")
    if second_h2 != -1:
        return f"{wrapped.rstrip()}\n{existing[second_h2:].lstrip(chr(10))}"
    return wrapped


def render_markdown(records: list[BaselineRecord], dataset_label: str) -> str:
    lines = [
        f"# Baseline results — {dataset_label}",
        "",
        "Auto-generated by `scripts/generate_baselines.py`. Numbers reflect the",
        "default model configuration shipped with OpenLithoHub.",
        "",
        "Reproduce with:",
        "",
        "```",
        ".venv/bin/python scripts/generate_baselines.py",
        "```",
        "",
        "`neural-ilt` downloads its v0.1 seed weights from HuggingFace",
        "(`openlithohub/neural-ilt-v0.1`). To iterate on a freshly-trained",
        "checkpoint before publishing it, pass",
        "`--neural-ilt-weights <path/to/model.pt>`.",
        "",
        "Wafer-level scores (`epe_wafer_*`, `l2_error_pixels`) come from a",
        "single shared HopkinsSimulator so every model is graded against the",
        "same wavelength / NA / threshold — these are the leaderboard scalars",
        "(mask-EPE ties Identity ≈ OpenILT ≈ Neural-ILT at 0).",
        "",
        "| Model | Samples | EPE mean (nm) | Wafer EPE (nm) | L2 (px) | PVB (nm) | MRC pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        m = rec.metrics
        epe_mean = f"{m['epe_mean_nm']:.3f}" if "epe_mean_nm" in m else "—"
        wafer_epe = f"{m['epe_wafer_mean_nm']:.3f}" if "epe_wafer_mean_nm" in m else "—"
        l2 = f"{m['l2_error_pixels']:.1f}" if "l2_error_pixels" in m else "—"
        pvb = f"{m['pvband_mean_nm']:.3f}" if "pvband_mean_nm" in m else "—"
        mrc = "{:.0%}".format(m["mrc_passed"]) if "mrc_passed" in m else "—"
        notes = f" ({rec.notes})" if rec.notes else ""
        lines.append(
            f"| `{rec.model}`{notes} | {rec.num_samples} | {epe_mean} "
            f"| {wafer_epe} | {l2} | {pvb} | {mrc} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ILT baseline numbers.")
    parser.add_argument("--output", type=Path, default=Path("baselines"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="LithoBench root directory; if omitted, falls back to synthetic.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic mode.")
    parser.add_argument("--limit", type=int, default=8)
    # Default is left as None so we can tell whether the user supplied a
    # value vs picked up the fallback. In synthetic mode we have to pin
    # pixel_nm to 8.0 (see comment near the override below); a user who
    # explicitly passed ``--pixel-nm`` deserves a warning rather than
    # having their flag silently dropped.
    parser.add_argument("--pixel-nm", type=float, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "dummy-identity",
            "rule-based-opc",
            "levelset-ilt",
            "openilt",
            "neural-ilt",
        ],
        help="Model names to evaluate (must be registered).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help=(
            "Pass pretrained=True to models that accept it (e.g. neural-ilt). "
            "Defaults to True so the published baselines reproduce out of the box; "
            "pass --no-pretrained to override."
        ),
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable --pretrained (forces neural-ilt to use random weights).",
    )
    parser.add_argument(
        "--neural-ilt-weights",
        type=Path,
        default=None,
        help=(
            "Local path to a neural-ilt checkpoint (state_dict). Overrides "
            "--pretrained for neural-ilt only — useful when iterating on a "
            "fresh training run before publishing to HuggingFace."
        ),
    )
    parser.add_argument("--no-pvband", action="store_true")
    parser.add_argument("--no-mrc", action="store_true")
    parser.add_argument(
        "--no-wafer",
        action="store_true",
        help=(
            "Skip wafer-level metrics (epe_wafer_*, l2_error_*). These require a "
            "forward simulator and dominate runtime; disable for the fastest "
            "smoke-test of the script itself."
        ),
    )
    parser.add_argument("--min-width-nm", type=float, default=40.0)
    parser.add_argument("--min-spacing-nm", type=float, default=40.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed torch RNG so models with random initialisation (neural-ilt) "
            "produce reproducible numbers."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.synthetic or args.data_root is None:
        patterns = build_synthetic_patterns(grid=64)[: args.limit]
        samples = patterns_to_samples(patterns)
        dataset_label = f"synthetic-{len(samples)}"
        # 8 nm/px so a 64×64 grid covers a 512 nm window — large enough that
        # 193 nm ArF diffraction actually resolves edges. At pixel_size_nm=1
        # the simulator collapses every feature into a sub-resolution blur
        # and wafer-level metrics return inf / a constant. Mirrors the
        # tests/test_benchmark/test_metrics.py convention.
        if args.pixel_nm is not None and args.pixel_nm != 8.0:
            print(
                f"WARNING: --pixel-nm={args.pixel_nm} is ignored in synthetic mode; "
                "the synthetic 64×64 patterns are pinned to 8 nm/px to keep ArF "
                "diffraction resolvable. Pass a real --data-root to honor the flag.",
                file=sys.stderr,
            )
        pixel_nm = 8.0
    else:
        # Non-synthetic mode: fall back to 1 nm/px when the flag is absent.
        pixel_nm = args.pixel_nm if args.pixel_nm is not None else 1.0
        samples = load_dataset_samples(args.data_root, pixel_nm, args.limit)
        dataset_label = f"lithobench-{len(samples)}"

    args.output.mkdir(parents=True, exist_ok=True)

    # Build a single shared simulator so every model is graded against the
    # same wavelength / NA / threshold (the leaderboard wafer-L2 pipeline
    # does the same thing — a per-model fresh simulator would let dose
    # drift between rows).
    simulator = None
    if not args.no_wafer:
        from openlithohub.simulators.base import SimulatorConfig
        from openlithohub.simulators.hopkins_sim import HopkinsSimulator

        simulator = HopkinsSimulator(SimulatorConfig(pixel_size_nm=pixel_nm))

    records: list[BaselineRecord] = []
    for model_name in args.models:
        model_kwargs: dict[str, Any] = {}
        if args.pretrained:
            model_kwargs["pretrained"] = True
        if model_name == "neural-ilt" and args.neural_ilt_weights is not None:
            model_kwargs["weights"] = str(args.neural_ilt_weights)
            model_kwargs.pop("pretrained", None)
        rec = evaluate_model(
            model_name,
            samples,
            pixel_nm=pixel_nm,
            run_pvband=not args.no_pvband,
            run_mrc=not args.no_mrc,
            run_wafer=not args.no_wafer,
            min_width_nm=args.min_width_nm,
            min_spacing_nm=args.min_spacing_nm,
            model_kwargs=model_kwargs,
            simulator=simulator,
        )
        if rec is None:
            print(f"  ! {model_name} not registered, skipping")
            continue
        rec.dataset = dataset_label
        records.append(rec)
        print(
            f"  ✓ {model_name}: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(rec.metrics.items()))
        )

    json_path = args.output / "results.json"
    md_path = args.output / "results.md"
    serializable: list[dict[str, Any]] = [
        {
            "model": r.model,
            "dataset": r.dataset,
            "num_samples": r.num_samples,
            "metrics": r.metrics,
            "notes": r.notes,
        }
        for r in records
    ]
    json_path.write_text(json.dumps(serializable, indent=2))
    md_path.write_text(_merge_synthetic_section(md_path, render_markdown(records, dataset_label)))
    print(f"\nResults written to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
