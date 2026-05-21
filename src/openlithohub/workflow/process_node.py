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
    optical_radius_nm: float = 1500.0
    """Optical interaction radius — how far light at a tile boundary
    'sees' through the imaging kernel. Used by ``workflow.halo`` to size
    tile halos so the forward model is fed real layout context, not
    zero-padded boundaries. Conservative default of 1.5 µm matches DUV
    rule-of-thumb (~10 × λ / (2 × NA)); EUV nodes can run much tighter.
    """
    demag_scan: float = 4.0
    demag_slit: float = 4.0
    """Reticle-to-wafer demagnification ratios along the scan and slit axes.

    Standard low-NA EUV (NA=0.33) and DUV scanners are isotropic 4×, so
    both default to 4.0. High-NA EUV (NA=0.55, ASML EXE:5000 class) is
    *anamorphic*: 8× along the scan axis and 4× along the slit axis,
    halving the field on-wafer to ~26 × 16.5 mm. The Hopkins forward
    currently assumes an isotropic mask grid and ignores these flags;
    they are recorded here so downstream tooling (anamorphic imaging,
    half-field stitching, mask-side pixel sizing) can branch on
    ``demag_scan != demag_slit``. See the High-NA tracking issue.
    """

    @property
    def is_anamorphic(self) -> bool:
        """Whether the projection optics are anamorphic (High-NA EUV)."""
        return self.demag_scan != self.demag_slit

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
    "2nm-euv": ProcessNodeConfig(
        name="2nm-euv",
        wavelength_nm=13.5,
        numerical_aperture=0.55,
        sigma_inner=0.2,
        sigma_outer=0.9,
        pixel_size_nm=0.5,
        min_feature_nm=12.0,
        min_spacing_nm=12.0,
        defocus_budget_nm=25.0,
        optical_radius_nm=250.0,
        demag_scan=8.0,
        demag_slit=4.0,
    ),
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
        optical_radius_nm=250.0,
        demag_scan=8.0,
        demag_slit=4.0,
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
        optical_radius_nm=400.0,
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
        optical_radius_nm=400.0,
    ),
    "28nm": ProcessNodeConfig(
        name="28nm",
        wavelength_nm=193.0,
        numerical_aperture=1.35,
        sigma_inner=0.5,
        sigma_outer=0.8,
        pixel_size_nm=1.0,
        min_feature_nm=28.0,
        min_spacing_nm=28.0,
        defocus_budget_nm=60.0,
        optical_radius_nm=1500.0,
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
        optical_radius_nm=1500.0,
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
