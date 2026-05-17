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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Register built-in models on import.
import openlithohub.models.examples.dummy_model  # noqa: F401
import openlithohub.models.levelset_ilt  # noqa: F401
import openlithohub.models.neural_ilt  # noqa: F401
from openlithohub.benchmark.compliance.mrc import check_mrc
from openlithohub.benchmark.metrics.epe import compute_epe
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
    min_width_nm: float,
    min_spacing_nm: float,
) -> BaselineRecord | None:
    try:
        model = registry.get(model_name)
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
            row.update(compute_epe(result.mask, sample.mask, pixel_size_nm=pixel_nm))
        if run_pvband:
            with contextlib.suppress(Exception):
                row.update(compute_pvband(result.mask, pixel_size_nm=pixel_nm))
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
    aggregated: dict[str, float] = {}
    for key in sorted(keys):
        if key.startswith("_"):
            continue
        vals = [r[key] for r in per_sample if key in r]
        if vals:
            aggregated[key] = float(torch.tensor(vals).mean().item())

    return BaselineRecord(
        model=model_name,
        dataset="",
        num_samples=len(samples),
        metrics=aggregated,
    )


def render_markdown(records: list[BaselineRecord], dataset_label: str) -> str:
    lines = [
        f"# Baseline results — {dataset_label}",
        "",
        "Auto-generated by `scripts/generate_baselines.py`. Numbers reflect the",
        "default model configuration shipped with OpenLithoHub.",
        "",
        "| Model | Samples | EPE mean (nm) | EPE max (nm) | PVB mean (nm) | MRC pass |",
        "|---|---|---|---|---|---|",
    ]
    for rec in records:
        m = rec.metrics
        epe_mean = f"{m['epe_mean_nm']:.3f}" if "epe_mean_nm" in m else "—"
        epe_max = f"{m['epe_max_nm']:.3f}" if "epe_max_nm" in m else "—"
        pvb = f"{m['pvband_mean_nm']:.3f}" if "pvband_mean_nm" in m else "—"
        mrc = "{:.0%}".format(m["mrc_passed"]) if "mrc_passed" in m else "—"
        notes = f" ({rec.notes})" if rec.notes else ""
        lines.append(
            f"| `{rec.model}`{notes} | {rec.num_samples} | {epe_mean} | {epe_max} | {pvb} | {mrc} |"
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
    parser.add_argument("--pixel-nm", type=float, default=1.0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy-identity", "levelset-ilt"],
        help="Model names to evaluate (must be registered).",
    )
    parser.add_argument("--no-pvband", action="store_true")
    parser.add_argument("--no-mrc", action="store_true")
    parser.add_argument("--min-width-nm", type=float, default=40.0)
    parser.add_argument("--min-spacing-nm", type=float, default=40.0)
    args = parser.parse_args()

    if args.synthetic or args.data_root is None:
        patterns = build_synthetic_patterns(grid=64)[: args.limit]
        samples = patterns_to_samples(patterns)
        dataset_label = f"synthetic-{len(samples)}"
        pixel_nm = 1.0
    else:
        samples = load_dataset_samples(args.data_root, args.pixel_nm, args.limit)
        dataset_label = f"lithobench-{len(samples)}"
        pixel_nm = args.pixel_nm

    args.output.mkdir(parents=True, exist_ok=True)

    records: list[BaselineRecord] = []
    for model_name in args.models:
        rec = evaluate_model(
            model_name,
            samples,
            pixel_nm=pixel_nm,
            run_pvband=not args.no_pvband,
            run_mrc=not args.no_mrc,
            min_width_nm=args.min_width_nm,
            min_spacing_nm=args.min_spacing_nm,
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
    md_path.write_text(render_markdown(records, dataset_label))
    print(f"\nResults written to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
