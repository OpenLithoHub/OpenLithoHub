"""End-to-end demo: design -> optimize -> export -> commercial sim interface.

Shows the complete chain from mask creation through co-design optimisation
to EDA bridge handoff.  Uses mock/stub simulators (Tachyon, Calibre) since
real commercial toolchains are not available in open-source CI.

Run standalone::

    python -m openlithohub.examples.eda_bridge_demo

Or from the repo root::

    python examples/eda_bridge_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from openlithohub.simulators import (
    CalibreSimulator,
    HopkinsSimulator,
    SimulatorConfig,
    TachyonSimulator,
)
from openlithohub.simulators.commercial import write_mask_gdsii
from openlithohub.workflow import BridgeRules, emit_bridge_bundle


# ---------------------------------------------------------------------------
# Step 1: Create a simple mask pattern
# ---------------------------------------------------------------------------

def make_test_mask(size: int = 64) -> torch.Tensor:
    """Create a simple test mask with a centered rectangle."""
    mask = torch.zeros(size, size)
    margin = size // 4
    mask[margin : size - margin, margin : size - margin] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Step 2: Run co-design optimisation (Hopkins gradients + Calibre oracle)
# ---------------------------------------------------------------------------

def run_co_design(
    mask: torch.Tensor,
    target: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    """Optimise *mask* using Hopkins for gradients and Calibre as oracle.

    Returns the optimised (clamped) mask tensor.
    """
    mask_var = mask.clone().detach().requires_grad_(True)

    grad_sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
    oracle = CalibreSimulator(
        SimulatorConfig(pixel_size_nm=4.0, dose=1.0, extra={"mock_mode": True}),
    )

    optimizer = torch.optim.Adam([mask_var], lr=0.1)

    print("Co-design optimisation (Hopkins grads + Calibre oracle):")
    for step in range(steps):
        optimizer.zero_grad()
        result = grad_sim.simulate(mask_var)
        loss = torch.nn.functional.mse_loss(result.aerial, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            oracle_result = oracle.simulate(mask_var.detach().clamp(0, 1))
            oracle_loss = torch.nn.functional.mse_loss(
                oracle_result.aerial, target,
            ).item()

        if step % 2 == 0 or step == steps - 1:
            print(
                f"  step {step}: hopkins_loss={loss.item():.6f}, "
                f"calibre_oracle_loss={oracle_loss:.6f}",
            )

    return mask_var.detach().clamp(0, 1)


# ---------------------------------------------------------------------------
# Step 3: Cross-check with Tachyon mock simulator
# ---------------------------------------------------------------------------

def cross_check_tachyon(mask: torch.Tensor, target: torch.Tensor) -> dict:
    """Validate optimised mask against Tachyon mock simulator."""
    sim = TachyonSimulator(
        SimulatorConfig(pixel_size_nm=4.0, dose=1.0, extra={"mock_mode": True}),
    )
    status = sim.preflight()
    result = sim.simulate(mask)
    mse = torch.nn.functional.mse_loss(result.aerial, target).item()
    info = {
        "preflight_ok": status.ok,
        "aerial_shape": tuple(result.aerial.shape),
        "aerial_range": (
            float(result.aerial.min()),
            float(result.aerial.max()),
        ),
        "resist_pixels": int(result.resist.sum().item()),
        "mse_vs_target": mse,
        "mock": result.metadata["mock"],
    }
    print(f"\nTachyon cross-check: {info}")
    return info


# ---------------------------------------------------------------------------
# Step 4: Export mask and emit EDA bridge bundle
# ---------------------------------------------------------------------------

def export_to_eda(
    mask: torch.Tensor,
    output_dir: Path,
    cell_name: str = "TOP",
    min_width_nm: float = 40.0,
    min_spacing_nm: float = 40.0,
) -> dict[str, Path]:
    """Write mask file and emit Calibre + ICV bridge decks."""
    mask_path = write_mask_gdsii(mask, output_dir / "mask.txt")
    oasis_placeholder = output_dir / "mask.oas"
    oasis_placeholder.write_bytes(b"")  # placeholder for demo

    rules = BridgeRules(
        min_width_nm=min_width_nm,
        min_spacing_nm=min_spacing_nm,
    )
    bundle = emit_bridge_bundle(
        oasis_placeholder,
        rules,
        cell_name=cell_name,
    )
    print(f"\nEDA bridge bundle written to {output_dir}:")
    for key, path in bundle.items():
        print(f"  {key}: {path.name}")
    return {"mask": mask_path, **bundle}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str | Path | None = None) -> dict:
    """Run the full end-to-end demo.

    Returns a dict with all outputs for programmatic inspection.
    """
    print("=" * 60)
    print("EDA Bridge End-to-End Demo")
    print("=" * 60)

    # Step 1
    mask = make_test_mask()
    target = make_test_mask()
    print(f"\nMask shape: {mask.shape}")

    # Step 2
    optimised = run_co_design(mask, target)

    # Step 3
    tachyon_info = cross_check_tachyon(optimised, target)

    # Step 4
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="eda_bridge_demo_")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = export_to_eda(optimised, out)

    print("\nDone.")
    return {"tachyon_info": tachyon_info, "files": files, "output_dir": str(out)}


if __name__ == "__main__":
    main()
