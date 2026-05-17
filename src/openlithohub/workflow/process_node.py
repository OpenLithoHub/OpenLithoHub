"""DTCO process node configuration for lithography simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessNodeConfig:
    """Physical parameters for a specific semiconductor process node.

    These parameters configure the forward lithography model, compliance checks,
    and optimization targets based on the target manufacturing technology.
    """

    name: str
    wavelength_nm: float
    numerical_aperture: float
    sigma_inner: float
    sigma_outer: float
    pixel_size_nm: float
    min_feature_nm: float
    min_spacing_nm: float
    resist_threshold: float = 0.5
    defocus_budget_nm: float = 20.0

    @property
    def sigma_px(self) -> float:
        """Compute Gaussian PSF sigma in pixels from optical parameters."""
        resolution_nm = 0.5 * self.wavelength_nm / self.numerical_aperture
        return resolution_nm / self.pixel_size_nm

    @property
    def k1_factor(self) -> float:
        """Rayleigh k1 factor for minimum half-pitch."""
        half_pitch_nm = self.min_feature_nm / 2.0
        return half_pitch_nm * self.numerical_aperture / self.wavelength_nm


PROCESS_NODES: dict[str, ProcessNodeConfig] = {
    "3nm-euv": ProcessNodeConfig(
        name="3nm-euv",
        wavelength_nm=13.5,
        numerical_aperture=0.55,
        sigma_inner=0.2,
        sigma_outer=0.9,
        pixel_size_nm=0.5,
        min_feature_nm=14.0,
        min_spacing_nm=14.0,
        defocus_budget_nm=30.0,
    ),
    "5nm-euv": ProcessNodeConfig(
        name="5nm-euv",
        wavelength_nm=13.5,
        numerical_aperture=0.33,
        sigma_inner=0.3,
        sigma_outer=0.8,
        pixel_size_nm=1.0,
        min_feature_nm=20.0,
        min_spacing_nm=20.0,
        defocus_budget_nm=40.0,
    ),
    "7nm": ProcessNodeConfig(
        name="7nm",
        wavelength_nm=13.5,
        numerical_aperture=0.33,
        sigma_inner=0.4,
        sigma_outer=0.8,
        pixel_size_nm=1.0,
        min_feature_nm=28.0,
        min_spacing_nm=28.0,
        defocus_budget_nm=50.0,
    ),
    "45nm": ProcessNodeConfig(
        name="45nm",
        wavelength_nm=193.0,
        numerical_aperture=1.35,
        sigma_inner=0.5,
        sigma_outer=0.8,
        pixel_size_nm=1.0,
        min_feature_nm=40.0,
        min_spacing_nm=40.0,
        defocus_budget_nm=80.0,
    ),
}


def get_node(name: str) -> ProcessNodeConfig:
    """Get a process node configuration by name.

    Raises:
        KeyError: If the node name is not found in presets.
    """
    if name not in PROCESS_NODES:
        available = ", ".join(sorted(PROCESS_NODES.keys()))
        raise KeyError(f"Unknown process node '{name}'. Available: {available}")
    return PROCESS_NODES[name]


def list_nodes() -> list[str]:
    """Return sorted list of available process node names."""
    return sorted(PROCESS_NODES.keys())
