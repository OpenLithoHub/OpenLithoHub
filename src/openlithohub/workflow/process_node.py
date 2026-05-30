"""DTCO process node configuration for lithography simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass

from openlithohub._constants import (
    NA_EUV_HIGH,
    NA_EUV_STANDARD,
    NA_IMMERSION,
    WAVELENGTH_ARF_NM,
    WAVELENGTH_EUV_NM,
)


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
    multi_patterning: str = "none"
    """Multi-patterning scheme assumed by ``min_feature_nm`` / ``min_spacing_nm``.

    The Rayleigh limit ``k1 × λ / NA`` caps the *half-pitch* a single
    exposure can resolve at k1 ≈ 0.25 (production manufacturable). When
    ``min_feature_nm`` falls below that limit, the layout assumes
    multiple exposures (LELE / LELELE / SADP / SAQP) — a single-shot
    forward simulation is then a coarse approximation, valid for
    *one* of the multi-patterning sub-layers but not the composite
    image.

    Values: ``"none"``, ``"lele"`` (litho-etch-litho-etch, 2-colour
    decomposition), ``"lelele"`` (3-colour), ``"sadp"`` (self-aligned
    double patterning), ``"saqp"`` (self-aligned quadruple). The flag
    is informational — the forward model does not yet decompose layouts
    by colour, so callers running below the single-exposure k1=0.25
    floor should split layouts manually before scoring.

    Worked examples:
        - 28nm at 193 nm DUV / NA 1.35: single-exposure k1 = 14 × 1.35 /
          193 ≈ 0.098, well below 0.25. Production 28nm is LELE
          immersion. Marked ``"lele"``.
        - 7nm at EUV NA 0.33: k1 = 14 × 0.33 / 13.5 ≈ 0.34. Single
          exposure, marked ``"none"``.
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
        wavelength_nm=WAVELENGTH_EUV_NM,
        numerical_aperture=NA_EUV_HIGH,
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
        wavelength_nm=WAVELENGTH_EUV_NM,
        numerical_aperture=NA_EUV_HIGH,
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
        wavelength_nm=WAVELENGTH_EUV_NM,
        numerical_aperture=NA_EUV_STANDARD,
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
        wavelength_nm=WAVELENGTH_EUV_NM,
        numerical_aperture=NA_EUV_STANDARD,
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
        wavelength_nm=WAVELENGTH_ARF_NM,
        numerical_aperture=NA_IMMERSION,
        sigma_inner=0.5,
        sigma_outer=0.8,
        pixel_size_nm=1.0,
        min_feature_nm=28.0,
        min_spacing_nm=28.0,
        defocus_budget_nm=60.0,
        optical_radius_nm=1500.0,
        # 193i single-shot k1 ≈ 0.098 — well below the 0.25 manufacturable
        # floor, so production 28nm uses LELE. The forward model does not
        # decompose layouts; callers scoring 28nm at single shot are
        # imaging an LELE *sub-layer*, not the composite mask.
        multi_patterning="lele",
    ),
    "45nm": ProcessNodeConfig(
        name="45nm",
        wavelength_nm=WAVELENGTH_ARF_NM,
        numerical_aperture=NA_IMMERSION,
        sigma_inner=0.5,
        sigma_outer=0.8,
        pixel_size_nm=1.0,
        min_feature_nm=40.0,
        min_spacing_nm=40.0,
        defocus_budget_nm=80.0,
        optical_radius_nm=1500.0,
        # k1 ≈ 0.140 at 40nm half-pitch — still single-exposure
        # uncomfortable at production-grade 0.25 floor, but historical
        # 45nm shipped with single-pass 193i at relaxed CDU; closer to
        # "none" than full LELE. Mark "none" with the caveat in mind.
        multi_patterning="none",
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
