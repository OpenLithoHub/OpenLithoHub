"""PDK design-rule presets.

Numbers below are *public-domain approximations* of the corresponding
academic PDKs; they are not redistributed proprietary tables. For
real-fab work, use the official PDK with its own DRC deck.

Sources:

* FreePDK45 — North Carolina State University, public release.
* ASAP7 — Arizona State University, predictive 7nm PDK.

We expose the design rules in nanometers, plus a few derived integer
pixel-grid quantities the generators need.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PdkRules:
    """Design rules for a single metal layer of a synthetic PDK.

    All distances are in nanometers. Pitch == width + spacing.
    """

    name: str
    pixel_size_nm: float
    min_width_nm: float
    min_spacing_nm: float
    min_area_nm2: float
    via_size_nm: float
    via_spacing_nm: float

    @property
    def pitch_nm(self) -> float:
        return self.min_width_nm + self.min_spacing_nm

    @property
    def min_width_px(self) -> int:
        """Smallest pixel width that survives the MRC opening check.

        ``check_mrc`` opens with structuring element radius
        ``floor(min_width_nm / (2 * pixel_size_nm))``, killing features
        narrower than ``2*radius + 1`` pixels. We mirror that formula and
        add 1 pixel of safety so traces drawn at this width pass MRC even
        after our own DRC opening pass.
        """
        from math import floor

        radius = floor(self.min_width_nm / (2.0 * self.pixel_size_nm))
        return max(1, 2 * radius + 1)

    @property
    def min_spacing_px(self) -> int:
        from math import floor

        radius = floor(self.min_spacing_nm / (2.0 * self.pixel_size_nm))
        return max(1, 2 * radius + 1)

    @property
    def pitch_px(self) -> int:
        return max(2, round(self.pitch_nm / self.pixel_size_nm))


PDK_PRESETS: dict[str, PdkRules] = {
    "freepdk45": PdkRules(
        name="freepdk45",
        pixel_size_nm=2.5,
        min_width_nm=50.0,
        min_spacing_nm=50.0,
        min_area_nm2=2500.0,
        via_size_nm=65.0,
        via_spacing_nm=70.0,
    ),
    "asap7": PdkRules(
        name="asap7",
        pixel_size_nm=1.0,
        min_width_nm=18.0,
        min_spacing_nm=18.0,
        min_area_nm2=324.0,
        via_size_nm=18.0,
        via_spacing_nm=22.0,
    ),
}


def get_pdk(name: str) -> PdkRules:
    """Look up a PDK preset by name. Case-insensitive."""

    key = name.lower().strip()
    if key not in PDK_PRESETS:
        raise KeyError(f"Unknown PDK {name!r}; available: {sorted(PDK_PRESETS)}")
    return PDK_PRESETS[key]
