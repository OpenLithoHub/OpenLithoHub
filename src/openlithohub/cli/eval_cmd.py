"""The `openlithohub eval` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

eval_app = typer.Typer(no_args_is_help=True)


@eval_app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path to load."),
    dataset: str = typer.Option(
        "lithobench", "--dataset", "-d", help="Dataset to evaluate on (lithobench/lithosim)."
    ),
    data_root: Path = typer.Option(
        ..., "--data-root", "-r", help="Path to dataset root directory."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Path to save evaluation report."
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, or markdown."
    ),
    node: str = typer.Option("45nm", "--node", "-n", help="Process node for evaluation context."),
    pixel_nm: float = typer.Option(1.0, "--pixel-nm", help="Pixel size in nanometers."),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Max samples to evaluate (default: all)."
    ),
    mrc_check: bool = typer.Option(True, "--mrc/--no-mrc", help="Run MRC compliance check."),
    pvband_check: bool = typer.Option(
        True, "--pvband/--no-pvband", help="Compute Process Variation Band metrics."
    ),
    min_width_nm: float = typer.Option(
        40.0, "--min-width-nm", help="Minimum feature width for MRC (nm)."
    ),
    min_spacing_nm: float = typer.Option(
        40.0, "--min-spacing-nm", help="Minimum spacing for MRC (nm)."
    ),
    submit_to_leaderboard: bool = typer.Option(
        False, "--submit/--no-submit", help="Auto-submit results to leaderboard."
    ),
    topology: str = typer.Option(
        "manhattan", "--topology", help="Mask topology for leaderboard: manhattan or curvilinear."
    ),
    paper_url: str | None = typer.Option(None, "--paper-url", help="Paper URL for leaderboard."),
    code_url: str | None = typer.Option(None, "--code-url", help="Code URL for leaderboard."),
    device: str = typer.Option(
        "cpu", "--device", help="Torch device for the forward model (cpu, cuda, mps)."
    ),
    dtype: str = typer.Option(
        "fp32", "--dtype", help="Compute dtype for the forward model: fp32 or bf16."
    ),
    compile_forward: bool = typer.Option(
        False, "--compile/--no-compile", help="Wrap the Hopkins forward with torch.compile."
    ),
    pretrained: bool = typer.Option(
        False,
        "--pretrained/--no-pretrained",
        help="Load pretrained weights for the selected model (when supported).",
    ),
    sha256: str | None = typer.Option(
        None,
        "--sha256",
        help=(
            "Expected SHA256 hex digest for direct-URL weight downloads. "
            "Required when the model resolves weights via a raw HTTPS URL; "
            "ignored for HuggingFace Hub repos."
        ),
    ),
) -> None:
    """Run evaluation of a lithography model on a benchmark dataset."""
    console = Console()

    import openlithohub.models.examples.dummy_model  # noqa: F401 — register built-in models
    import openlithohub.models.levelset_ilt  # noqa: F401
    import openlithohub.models.neural_ilt  # noqa: F401
    import openlithohub.models.rule_based_opc  # noqa: F401
    from openlithohub.benchmark.compliance.mrc import check_mrc
    from openlithohub.benchmark.metrics.epe import compute_epe
    from openlithohub.benchmark.metrics.pvband import compute_pvband
    from openlithohub.benchmark.report import generate_report
    from openlithohub.models.registry import registry
    from openlithohub.workflow.process_node import PROCESS_NODES

    # Auto-configure from process node if available
    if node in PROCESS_NODES:
        from openlithohub.workflow.process_node import get_node

        node_config = get_node(node)
        if pixel_nm == 1.0:
            pixel_nm = node_config.pixel_size_nm
        if min_width_nm == 40.0:
            min_width_nm = node_config.min_feature_nm
        if min_spacing_nm == 40.0:
            min_spacing_nm = node_config.min_spacing_nm

    console.print(f"[bold]Evaluating[/bold] model={model} dataset={dataset} node={node}")

    requested_kwargs = _build_model_kwargs(pretrained, sha256)
    try:
        support = registry.supports_kwargs(model, requested_kwargs)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    if (pretrained or sha256 is not None) and not all(support.values()):
        console.print(
            f"[yellow]Warning:[/yellow] Model {model!r} does not support "
            "--pretrained / --sha256; ignoring."
        )

    litho_model = registry.get(model, **requested_kwargs)

    litho_model.setup()

    try:
        adapter = _load_dataset(dataset, data_root, pixel_nm)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        litho_model.teardown()
        raise typer.Exit(1) from None

    n_samples = min(len(adapter), limit) if limit else len(adapter)
    console.print(f"Running on {n_samples} samples...")

    all_metrics: list[dict[str, float]] = []
    perf_kwargs = _build_perf_kwargs(device, dtype, compile_forward)
    for i in range(n_samples):
        sample = adapter[i]
        result = litho_model.predict(sample.design, **perf_kwargs)

        sample_metrics: dict[str, float] = {}

        if sample.mask is not None:
            epe = compute_epe(result.mask, sample.mask, pixel_size_nm=pixel_nm)
            # `valid` is a non-numeric flag describing edge-set health for this
            # sample; drop it before aggregation so we don't average a bool.
            epe.pop("valid", None)
            sample_metrics.update(epe)

        if mrc_check:
            mrc_result = check_mrc(
                result.mask,
                min_width_nm=min_width_nm,
                min_spacing_nm=min_spacing_nm,
                pixel_size_nm=pixel_nm,
            )
            sample_metrics["mrc_violation_rate"] = mrc_result.violation_rate
            sample_metrics["mrc_passed"] = 1.0 if mrc_result.passed else 0.0

        if pvband_check:
            pv = compute_pvband(result.mask, pixel_size_nm=pixel_nm)
            sample_metrics.update(pv)

        if sample_metrics:
            all_metrics.append(sample_metrics)

    litho_model.teardown()

    if not all_metrics:
        console.print("[yellow]Warning:[/yellow] No metrics computed.")
        raise typer.Exit(1)

    aggregated = _aggregate_metrics(all_metrics)
    aggregated["model"] = model
    aggregated["dataset"] = dataset
    aggregated["node"] = node
    aggregated["num_samples"] = n_samples

    report = generate_report(aggregated, output_format=format)
    console.print(report, highlight=False)

    if output:
        output.write_text(report)
        console.print(f"Report saved to {output}")

    if submit_to_leaderboard:
        from openlithohub.leaderboard.schema import BenchmarkResult, MaskTopology, ProcessNode
        from openlithohub.leaderboard.tracker import submit_result as lb_submit

        try:
            result_entry = BenchmarkResult(
                model_name=model,
                dataset=dataset,
                process_node=ProcessNode(node),
                mask_topology=MaskTopology(topology),
                epe_mean_nm=aggregated.get("epe_mean_nm", 0.0),
                epe_max_nm=aggregated.get("epe_max_nm", 0.0),
                pvband_mean_nm=aggregated.get("pvband_mean_nm"),
                pvband_max_nm=aggregated.get("pvband_max_nm"),
                mrc_violation_rate=aggregated.get("mrc_violation_rate"),
                drc_pass=(
                    aggregated.get("mrc_passed", 0.0) == 1.0 if "mrc_passed" in aggregated else None
                ),
                paper_url=paper_url,
                code_url=code_url,
            )
            sub_id = lb_submit(result_entry)
            console.print(f"[green]Submitted to leaderboard![/green] ID: {sub_id}")
        except (ValueError, KeyError) as e:
            console.print(f"[yellow]Warning:[/yellow] Could not submit to leaderboard: {e}")


def _load_dataset(dataset: str, data_root: Path, pixel_nm: float) -> Any:
    from openlithohub.data import LithoBenchDataset, LithoSimDataset

    if dataset == "lithobench":
        return LithoBenchDataset(root=data_root, pixel_nm=pixel_nm)
    if dataset == "lithosim":
        return LithoSimDataset(cache_dir=str(data_root), pixel_nm=pixel_nm)
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: lithobench, lithosim")


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, Any]:
    """Average per-sample metrics into a single aggregate dict."""
    import torch

    if not metrics_list:
        return {}

    all_keys: set[str] = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    aggregated: dict[str, Any] = {}
    for key in sorted(all_keys):
        vals = [m[key] for m in metrics_list if key in m]
        if vals:
            aggregated[key] = float(torch.tensor(vals).mean().item())
    return aggregated


def _build_perf_kwargs(device: str, dtype: str, compile_forward: bool) -> dict[str, Any]:
    """Translate CLI perf flags into predict() kwargs.

    Models that don't consume these keys (e.g. dummy-identity) ignore them
    via their **kwargs catch-all; models that drive the Hopkins forward
    (LevelSetILTModel, AIModelOPC) read them as listed in their docstrings.
    """
    import torch

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype not in dtype_map:
        raise typer.BadParameter(f"--dtype must be 'fp32' or 'bf16'; got {dtype!r}")
    return {
        "device": device,
        "dtype": dtype_map[dtype],
        "compile_forward": compile_forward,
    }


def _build_model_kwargs(pretrained: bool, sha256: str | None) -> dict[str, Any]:
    """Construct registry.get() kwargs for opt-in remote weight loading."""
    kwargs: dict[str, Any] = {}
    if pretrained:
        kwargs["pretrained"] = True
    if sha256 is not None:
        kwargs["url_sha256"] = sha256
    return kwargs
