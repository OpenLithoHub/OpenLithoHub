"""Single source of truth for physical constants used across OpenLithoHub.

All default values for simulator configurations, resist models, exposure
parameters, and EUV mask geometry are defined here.  Every other module
imports from this file rather than duplicating magic numbers.

The module is organized into sections:
  - Optical defaults (wavelength, NA, coherence)
  - Resist defaults (threshold, diffusion, quencher)
  - EUV 3D mask geometry defaults
  - DiffCFD plugin defaults (Dill/Mack exposure + development)
  - DiffNano plugin defaults (CAR/PEB resist)
  - DiffNano spin-coating defaults
  - Process-condition defaults
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optical / imaging defaults
# ---------------------------------------------------------------------------

WAVELENGTH_ARF_NM: float = 193.0
"""ArF excimer laser wavelength (DUV immersion scanners)."""

WAVELENGTH_EUV_NM: float = 13.5
"""EUV source wavelength (NXE/EXE-class scanners)."""

NA_IMMERSION: float = 1.35
"""Numerical aperture for ArF immersion scanners."""

NA_EUV_STANDARD: float = 0.33
"""Numerical aperture for standard EUV (NXE:3400 class)."""

NA_EUV_HIGH: float = 0.55
"""Numerical aperture for high-NA EUV (EXE:5000 class)."""

SIGMA_OUTER_DEFAULT: float = 0.7
"""Default outer partial-coherence factor (circular illumination)."""

SIGMA_INNER_DEFAULT: float = 0.0
"""Default inner sigma (0.0 = circular, not annular)."""

PIXEL_SIZE_NM_DEFAULT: float = 1.0
"""Default mask pixel pitch in nanometres."""

DEFOCUS_NM_DEFAULT: float = 0.0
"""Default defocus offset (zero = best focus)."""

DOSE_DEFAULT: float = 1.0
"""Default linear dose multiplier."""

NUM_KERNELS_DEFAULT: int = 24
"""Default SOCS truncation order (Yang2023_LithoBench Table II)."""

POLE_OPENING_DEG_DEFAULT: float = 30.0
"""Default half-angle of each dipole/quasar pole wedge (degrees)."""

# ---------------------------------------------------------------------------
# Resist defaults
# ---------------------------------------------------------------------------

THRESHOLD_ICCAD16: float = 0.225
"""Canonical ICCAD16 / LithoBench resist cutoff (Yang2023 §3.2, Table II).

This is the intensity threshold used to score ICCAD16 mask layouts on the
simulated wafer image.  Use this when reproducing benchmark numbers.
"""

THRESHOLD_GENERIC: float = 0.5
"""Generic mid-intensity cutoff for ad-hoc resist thresholding."""

RESIST_DIFFUSION_NM_DEFAULT: float = 0.0
"""Default acid diffusion length (0.0 = legacy CTR, no diffusion)."""

QUENCHER_DEFAULT: float = 0.0
"""Default quencher concentration (0.0 = disabled)."""

STEEPNESS_DEFAULT: float = 50.0
"""Default sigmoid steepness for differentiable resist threshold."""

# ---------------------------------------------------------------------------
# EUV 3D mask shadow-effect defaults
# ---------------------------------------------------------------------------

ABSORBER_THICKNESS_NM_DEFAULT: float = 70.0
"""Default absorber stack height in nm (Ta-based absorber)."""

CHIEF_RAY_ANGLE_DEG_DEFAULT: float = 6.0
"""Default chief-ray angle of incidence at mask (NXE:3400-class)."""

CHIEF_RAY_AZIMUTH_DEG_DEFAULT: float = 0.0
"""Default azimuth of the chief ray (0 = +x direction)."""

# ---------------------------------------------------------------------------
# DiffCFD plugin: Dill exposure + Mack development defaults
# ---------------------------------------------------------------------------

DILL_A_DEFAULT: float = 0.55
"""Bleachable absorption coefficient (cm^-1 / (mJ/cm^2)), typical 193nm CAR."""

DILL_B_DEFAULT: float = 0.05
"""Non-bleachable absorption coefficient (cm^-1), typical 193nm CAR."""

DILL_C_DEFAULT: float = 0.014
"""PAC quantum efficiency (cm^2 / mJ), typical 193nm CAR."""

MACK_R_MAX: float = 150.0
"""Maximum dissolution rate (nm/s) for Mack development model."""

MACK_R_MIN: float = 0.1
"""Minimum dissolution rate (nm/s) for Mack development model."""

MACK_N_DEFAULT: float = 5.0
"""Development selectivity (dimensionless) for Mack model."""

MACK_A_DEFAULT: float = 0.5
"""Mack threshold parameter (dimensionless)."""

GAMMA_SOLVENT_DEFAULT: float = 3.0
"""Solvent plasticization coefficient (dimensionless)."""

DIFFCFD_LITHO_DEFAULTS: dict[str, float] = {
    "dill_A": DILL_A_DEFAULT,
    "dill_B": DILL_B_DEFAULT,
    "dill_C": DILL_C_DEFAULT,
    "r_max": MACK_R_MAX,
    "r_min": MACK_R_MIN,
    "mack_n": MACK_N_DEFAULT,
    "mack_a": MACK_A_DEFAULT,
    "gamma_solvent": GAMMA_SOLVENT_DEFAULT,
}

# ---------------------------------------------------------------------------
# DiffCFD plugin: spin-coating defaults
# ---------------------------------------------------------------------------

SPIN_COAT_RHO: float = 1000.0
"""Resist solution density (kg/m^3)."""

SPIN_COAT_MU_SOLVENT: float = 1e-3
"""Solvent dynamic viscosity (Pa*s)."""

SPIN_COAT_ALPHA_VISC: float = 4.5
"""Viscosity-concentration exponent (dimensionless)."""

SPIN_COAT_BETA_VISC: float = 1.5
"""Viscosity-concentration exponent (dimensionless)."""

SPIN_COAT_C_EVAP: float = 1.2e-6
"""Evaporation rate constant (m/s)."""

SPIN_COAT_C_SOLID: float = 0.15
"""Solid content in resist (mass fraction)."""

DIFFCFD_SPIN_COAT_DEFAULTS: dict[str, float] = {
    "rho": SPIN_COAT_RHO,
    "mu_solvent": SPIN_COAT_MU_SOLVENT,
    "alpha_visc": SPIN_COAT_ALPHA_VISC,
    "beta_visc": SPIN_COAT_BETA_VISC,
    "c_evap": SPIN_COAT_C_EVAP,
    "c_solid": SPIN_COAT_C_SOLID,
}

# ---------------------------------------------------------------------------
# DiffCFD plugin: process-condition defaults
# ---------------------------------------------------------------------------

PROCESS_THICKNESS_M: float = 8e-6
"""Default dry film thickness (m)."""

PROCESS_RESIDUAL_SOLVENT: float = 0.15
"""Default residual solvent fraction after spin."""

PROCESS_DEV_TIME_S: float = 30.0
"""Default development time (s)."""

PROCESS_SPIN_DT: float = 0.001
"""Default spin time step (s)."""

PROCESS_H0_M: float = 8e-6
"""Default initial film thickness (m)."""

PROCESS_C0: float = 0.85
"""Default initial solvent concentration (fraction)."""

PROCESS_OMEGA_RPM: float = 2500.0
"""Default spin speed (RPM)."""

DIFFCFD_PROCESS_DEFAULTS: dict[str, float] = {
    "thickness_m": PROCESS_THICKNESS_M,
    "residual_solvent": PROCESS_RESIDUAL_SOLVENT,
    "dev_time_s": PROCESS_DEV_TIME_S,
    "spin_dt": PROCESS_SPIN_DT,
    "h0_m": PROCESS_H0_M,
    "c0": PROCESS_C0,
    "omega_rpm": PROCESS_OMEGA_RPM,
}

# ---------------------------------------------------------------------------
# DiffNano plugin: resist model defaults
# ---------------------------------------------------------------------------

ACID_DIFFUSION_LENGTH_NM: float = 20.0
"""Acid diffusion length during PEB (nm) — DiffNano resist plugin."""

DEVELOPMENT_CONTRAST: float = 10.0
"""Resist development contrast (dimensionless) — higher = sharper."""

THRESHOLD_DOSE_DIFFNANO: float = 0.5
"""Normalized clearing threshold for DiffNano resist."""

PEB_DIFFUSION_NM: float = 10.0
"""Post-exposure bake diffusion length (nm) — DiffNano resist."""

DIFFNANO_RESIST_DEFAULTS: dict[str, float] = {
    "acid_diffusion_length_nm": ACID_DIFFUSION_LENGTH_NM,
    "development_contrast": DEVELOPMENT_CONTRAST,
    "threshold_dose": THRESHOLD_DOSE_DIFFNANO,
    "peb_diffusion_nm": PEB_DIFFUSION_NM,
    "pixel_size_nm": PIXEL_SIZE_NM_DEFAULT,
}
