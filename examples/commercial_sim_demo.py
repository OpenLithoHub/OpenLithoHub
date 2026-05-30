"""Commercial simulator adapter demo.

Shows how to configure and use Tachyon/Calibre adapters in both real and
mock modes, and how to combine them with OpenLithoHub masks.
"""

from __future__ import annotations

import torch

from openlithohub.simulators import (
    CalibreSimulator,
    SimulatorConfig,
    TachyonSimulator,
    get_simulator,
)
from openlithohub.simulators.commercial import ToolchainError


def make_test_mask(size: int = 64) -> torch.Tensor:
    """Create a simple test mask with a centered rectangle."""
    mask = torch.zeros(size, size)
    margin = size // 4
    mask[margin : size - margin, margin : size - margin] = 1.0
    return mask


def demo_mock_mode() -> None:
    """Demonstrate mock mode for both adapters (no commercial tool needed)."""
    mask = make_test_mask()
    config = SimulatorConfig(pixel_size_nm=4.0, dose=1.0, threshold=0.225)

    for name, cls in [("Tachyon", TachyonSimulator), ("Calibre", CalibreSimulator)]:
        cfg = SimulatorConfig(**{**config.__dict__, "extra": {"mock_mode": True}})
        sim = cls(cfg)

        # Preflight check
        status = sim.preflight()
        print(f"\n{name} preflight: ok={status.ok}")

        # Run simulation
        result = sim.simulate(mask)
        print(f"{name} aerial shape: {result.aerial.shape}")
        print(f"{name} aerial range: [{result.aerial.min():.4f}, {result.aerial.max():.4f}]")
        print(f"{name} resist coverage: {result.resist.sum().item():.0f} pixels")
        print(f"{name} mock: {result.metadata['mock']}")


def demo_real_mode_error() -> None:
    """Show what happens when the toolchain is not available."""
    mask = make_test_mask()

    sim = TachyonSimulator(
        SimulatorConfig(extra={
            "tachyon_home": "/opt/asml/tachyon",
            "recipe": "/recipes/default.tcl",
        })
    )

    status = sim.preflight()
    print(f"\nReal Tachyon preflight: ok={status.ok}")
    if not status.ok:
        print(f"  Issues: {'; '.join(status.messages)}")

    try:
        sim.simulate(mask)
    except ToolchainError as e:
        print(f"  Expected error: {e}")


def demo_registry_usage() -> None:
    """Use the registry to construct adapters by name."""
    mask = make_test_mask()

    sim = get_simulator(
        "tachyon",
        SimulatorConfig(extra={"mock_mode": True}),
    )
    result = sim.simulate(mask)
    print(f"\nRegistry lookup: {sim.name} backend, aerial mean={result.aerial.mean():.4f}")


def demo_co_design_workflow() -> None:
    """Combine a commercial simulator mock with a simple co-design loop.

    Commercial simulators are not differentiable. The co-design pattern
    uses the Hopkins simulator (which IS differentiable) for gradient-based
    optimization, and the commercial simulator as a periodic validation
    oracle to check real-world fidelity.
    """
    from openlithohub.simulators import HopkinsSimulator

    mask = make_test_mask().requires_grad_(True)

    # Differentiable simulator for gradient-based optimization
    grad_sim = HopkinsSimulator(SimulatorConfig(pixel_size_nm=4.0))
    # Commercial simulator as evaluation oracle
    oracle = CalibreSimulator(
        SimulatorConfig(
            pixel_size_nm=4.0,
            dose=1.0,
            extra={"mock_mode": True},
        )
    )

    optimizer = torch.optim.Adam([mask], lr=0.1)
    target = make_test_mask()

    print("\nCo-design optimization loop (Hopkins grads + Calibre oracle):")
    for step in range(5):
        optimizer.zero_grad()
        result = grad_sim.simulate(mask)
        loss = torch.nn.functional.mse_loss(result.aerial, target)
        loss.backward()
        optimizer.step()

        # Periodic oracle evaluation with commercial simulator
        with torch.no_grad():
            oracle_result = oracle.simulate(mask.detach().clamp(0, 1))
            oracle_loss = torch.nn.functional.mse_loss(oracle_result.aerial, target).item()

        if step % 2 == 0:
            print(
                f"  step {step}: hopkins_loss={loss.item():.6f}, "
                f"calibre_oracle_loss={oracle_loss:.6f}"
            )


if __name__ == "__main__":
    print("=" * 60)
    print("Commercial Simulator Adapter Demo")
    print("=" * 60)

    demo_mock_mode()
    demo_real_mode_error()
    demo_registry_usage()
    demo_co_design_workflow()

    print("\nDone.")
