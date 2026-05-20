"""Shared test fixtures for OpenLithoHub."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture
def sample_design() -> torch.Tensor:
    """A simple 64x64 binary design pattern for testing."""
    t = torch.zeros(64, 64)
    t[16:48, 16:48] = 1.0  # Square feature
    return t


@pytest.fixture
def sample_mask() -> torch.Tensor:
    """A simple 64x64 binary mask (slightly larger than design for OPC bias)."""
    t = torch.zeros(64, 64)
    t[14:50, 14:50] = 1.0  # Biased square
    return t


_LEF_TEXT = """VERSION 5.7 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;

UNITS
  DATABASE MICRONS 1000 ;
END UNITS

MANUFACTURINGGRID 0.001 ;

LAYER metal1
  TYPE ROUTING ;
  WIDTH 0.1 ;
  PITCH 0.2 ;
  DIRECTION HORIZONTAL ;
END metal1

MACRO INV
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 0.5 BY 1.0 ;
  SYMMETRY X Y ;
  PIN A
    DIRECTION INPUT ;
    PORT
      LAYER metal1 ;
        RECT 0.1 0.1 0.2 0.2 ;
    END
  END A
  PIN Y
    DIRECTION OUTPUT ;
    PORT
      LAYER metal1 ;
        RECT 0.3 0.1 0.4 0.2 ;
    END
  END Y
  OBS
    LAYER metal1 ;
      RECT 0.05 0.05 0.45 0.95 ;
  END
END INV

END LIBRARY
"""

_DEF_TEXT = """VERSION 5.7 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;

DESIGN top ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 5000 5000 ) ;

COMPONENTS 1 ;
  - u1 INV + PLACED ( 1000 1000 ) N ;
END COMPONENTS

END DESIGN
"""


@pytest.fixture
def lefdef_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic minimal LEF + DEF pair for ingestion tests.

    Returns ``(def_path, lef_path)``. The DEF places one ``INV`` cell
    (defined in the LEF) at (1, 1) µm in a 5×5 µm die. KLayout reads
    this as a top cell ``top`` with metal1.OBS / metal1.PIN layers.
    """
    lef = tmp_path / "tech.lef"
    deff = tmp_path / "top.def"
    lef.write_text(_LEF_TEXT)
    deff.write_text(_DEF_TEXT)
    return deff, lef
