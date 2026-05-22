"""Run hotspot-detection baselines on the ICCAD 2016 Problem C dataset.

End-to-end smoke test of the hotspot-detection stack: load OASIS files
via :class:`Iccad16Dataset`, run a small set of trivial point-based
predictors, and score each predictor with
:func:`compute_hotspot_detection`. The predictors here are sanity
baselines, not ML models — they pin down the empty-prediction lower
bound and the saturated-grid recall ceiling so that any real predictor
slotted in later has a meaningful comparison.

The optional ``--strategy batchal`` flag invokes the Yang2020 §III
batch-active-sampling algorithm
(:mod:`openlithohub.baselines.hotspot_batchal`) over the clip-site grid.
Because no labelled hotspot detector ships in OpenLithoHub yet, the
predictive probabilities required by the algorithm are approximated by
the local pattern density of each clip — see the ``predict_batchal``
docstring for the honesty caveat.

Outputs:

- ``<output>/hotspot_results.json`` — per-(model, case) metric records.
- ``<output>/hotspot_results.md`` — markdown table for docs.

Usage::

    python -m openlithohub.benchmark.run_hotspot_baseline \\
        --data-root data/iccad16 --output out/hotspot

The data root must contain ``testcase{N}.oas`` + ``test{N}.csv`` files
from https://github.com/phdyang007/ICCAD16-N7M2EUV.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from openlithohub.baselines.hotspot_batchal import batch_active_select, extract_clip_features
from openlithohub.benchmark.metrics.hotspot import compute_hotspot_detection
from openlithohub.data.base import LithoSample
from openlithohub.data.iccad16 import Iccad16Dataset


@dataclass
class HotspotRecord:
    model: str
    case_id: int
    num_gt: int
    num_pred: int
    metrics: dict[str, float]


PredictorFn = Callable[[LithoSample], torch.Tensor]


def predict_empty(_sample: LithoSample) -> torch.Tensor:
    """Predict no hotspots. Lower bound: recall=0, precision=1 (vacuous)."""
    return torch.zeros(0, 2)


def predict_grid(sample: LithoSample, *, step_nm: float = 200.0) -> torch.Tensor:
    """Predict every grid point on a regular lattice over the design bbox.

    Saturates the FP rate while guaranteeing high recall — useful as the
    "detect everything" upper bound for recall on the metric. Step is
    chosen so the lattice covers the design without exploding the FP
    count; ICCAD16 layouts are ~1.5–2 µm so step=200 nm gives ~50–100
    candidates per case.
    """
    design = sample.design
    h_px, w_px = int(design.shape[0]), int(design.shape[1])
    pixel_nm = float(sample.metadata.get("pixel_nm", 1.0))
    ox_nm, oy_nm = sample.metadata.get("origin_nm", [0.0, 0.0])
    # Inclusive-stop arange: arange(0, n*step, step) drops the right/bottom
    # edge sample (e.g. 2000/200 stops at 1800). Adding step/2 to the stop
    # bound gives an inclusive lattice without floating-point boundary games.
    x_stop = w_px * pixel_nm + step_nm / 2.0
    y_stop = h_px * pixel_nm + step_nm / 2.0
    xs = torch.arange(0, x_stop, step_nm, dtype=torch.float32) + float(ox_nm)
    ys = torch.arange(0, y_stop, step_nm, dtype=torch.float32) + float(oy_nm)
    if xs.numel() == 0 or ys.numel() == 0:
        return torch.zeros(0, 2)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)


def predict_clip_centers(sample: LithoSample) -> torch.Tensor:
    """Predict the centroid of each clip-site box.

    The ICCAD16 auxiliary layer (10000, 0) defines 16×16 nm inspection
    windows on a regular grid that does NOT align with the CSV hotspot
    locations (verified empirically: 70+ nm separation). Treating clip
    centers as predictions therefore measures how badly the
    "inspection-grid as predictor" strawman performs — recall should be
    near 0 at small match radii.
    """
    sites = sample.metadata.get("clip_sites", [])
    if not sites:
        return torch.zeros(0, 2)
    pts = [[(s["x0_nm"] + s["x1_nm"]) / 2.0, (s["y0_nm"] + s["y1_nm"]) / 2.0] for s in sites]
    return torch.tensor(pts, dtype=torch.float32)


PREDICTORS: dict[str, PredictorFn] = {
    "empty": predict_empty,
    "grid-200nm": predict_grid,
    "clip-centers": predict_clip_centers,
}


def make_batchal_predictor(*, budget: int, feature_dim: int = 16) -> PredictorFn:
    """Build a ``--strategy batchal`` predictor over the clip-site pool.

    The returned closure scores each clip with the
    [Yang2020_BatchAL] §III acquisition function and returns the
    selected clip-site centers as predicted hotspot locations.

    Honesty caveat: the paper assumes a trained hotspot detector
    supplies the per-clip predictive probability ``p(y=1 | x_i)``.
    OpenLithoHub does not ship such a detector yet, so this script
    substitutes the **local pattern density** of each clip (mean of the
    pooled feature) as a stand-in for ``p(y=1 | x_i)``. This is a
    reasonable lower-bound proxy on a single testcase but is
    **not** a faithful reproduction of the paper's experimental
    setup. The point of wiring the predictor here is to exercise the
    selection algorithm end-to-end, not to claim the paper's accuracy
    numbers. See ``out/plans/external-resource-utilization.md`` Task #1
    for the v0.2 follow-up that adds a real detector.
    """

    def predict_batchal(sample: LithoSample) -> torch.Tensor:
        sites = sample.metadata.get("clip_sites", [])
        if not sites:
            return torch.zeros(0, 2)
        pixel_nm = float(sample.metadata.get("pixel_nm", 1.0))
        ox_nm, oy_nm = sample.metadata.get("origin_nm", [0.0, 0.0])
        features = extract_clip_features(
            sample.design,
            sites,
            pixel_nm=pixel_nm,
            origin_nm=(float(ox_nm), float(oy_nm)),
            feature_dim=feature_dim,
        )
        # No learner → use clip pattern density as a stand-in for p(y=1).
        # `clamp` keeps the value in [0, 1] for the algorithm contract.
        density = features.mean(dim=1).clamp(0.0, 1.0)
        k = min(budget, features.shape[0])
        idx = batch_active_select(features, density, k=k)
        centers = [
            [
                (sites[i]["x0_nm"] + sites[i]["x1_nm"]) / 2.0,
                (sites[i]["y0_nm"] + sites[i]["y1_nm"]) / 2.0,
            ]
            for i in idx.tolist()
        ]
        return torch.tensor(centers, dtype=torch.float32)

    return predict_batchal


def gt_points(sample: LithoSample) -> torch.Tensor:
    hs = sample.metadata.get("hotspots", [])
    if not hs:
        return torch.zeros(0, 2)
    pts = [[h["x_nm"], h["y_nm"]] for h in hs]
    return torch.tensor(pts, dtype=torch.float32)


def evaluate(
    dataset: Iccad16Dataset,
    predictors: dict[str, PredictorFn],
    match_radius_nm: float,
) -> list[HotspotRecord]:
    out: list[HotspotRecord] = []
    for case_id in dataset.case_ids:
        idx = dataset.case_ids.index(case_id)
        sample = dataset[idx]
        gt = gt_points(sample)
        for name, fn in predictors.items():
            pred = fn(sample)
            metrics = compute_hotspot_detection(pred, gt, match_radius_nm=match_radius_nm)
            out.append(
                HotspotRecord(
                    model=name,
                    case_id=case_id,
                    num_gt=int(gt.shape[0]),
                    num_pred=int(pred.shape[0]),
                    metrics=metrics,
                )
            )
    return out


def render_markdown(records: list[HotspotRecord], match_radius_nm: float) -> str:
    strategies = sorted({r.model for r in records})
    lines = [
        f"# Hotspot detection baselines — ICCAD16 (match radius = {match_radius_nm} nm)",
        "",
        "Auto-generated by `scripts/run_hotspot_baseline.py`. Strategies in",
        f"this run: {', '.join(f'`{s}`' for s in strategies)}.",
        "",
        "| Model | Case | GT | Predicted | TP | FP | FN | Recall | Precision | F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        m = r.metrics
        lines.append(
            f"| `{r.model}` | {r.case_id} | {r.num_gt} | {r.num_pred} "
            f"| {int(m['num_tp'])} | {int(m['num_fp'])} | {int(m['num_fn'])} "
            f"| {m['recall']:.3f} | {m['precision']:.3f} | {m['f1']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hotspot-detection baselines on ICCAD16.")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Directory with testcase*.oas + test*.csv."
    )
    parser.add_argument("--output", type=Path, default=Path("out/hotspot"))
    parser.add_argument("--match-radius-nm", type=float, default=1.0)
    parser.add_argument("--pixel-nm", type=float, default=1.0)
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=None,
        help="Subset of case ids to evaluate; defaults to every case present.",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", "empty", "grid-200nm", "clip-centers", "batchal"],
        default="all",
        help=(
            "Which predictor(s) to run. 'all' runs the three sanity baselines; "
            "'batchal' runs only the Yang2020 batch-active-sampling predictor. "
            "Other values run that single sanity baseline."
        ),
    )
    parser.add_argument(
        "--batchal-budget",
        type=int,
        default=60,
        help="Number of clips BatchAL selects per case. Paper default is 60.",
    )
    args = parser.parse_args()

    predictors: dict[str, PredictorFn]
    if args.strategy == "all":
        predictors = dict(PREDICTORS)
    elif args.strategy == "batchal":
        predictors = {"batchal": make_batchal_predictor(budget=args.batchal_budget)}
    else:
        predictors = {args.strategy: PREDICTORS[args.strategy]}

    dataset = Iccad16Dataset(args.data_root, cases=args.cases, pixel_nm=args.pixel_nm)
    records = evaluate(dataset, predictors, match_radius_nm=args.match_radius_nm)

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "hotspot_results.json"
    md_path = args.output / "hotspot_results.md"
    json_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    md_path.write_text(render_markdown(records, args.match_radius_nm))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
