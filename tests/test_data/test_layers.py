"""Lock the PDK layer registry against silent drift.

Each adapter's exported ``DEFAULT_DESIGN_LAYER`` must match the
registry entry. If someone later edits an adapter to override the
default, this test forces them to update ``_layers.py`` too — that's
the whole point of having a single source of truth.
"""

from __future__ import annotations

from openlithohub.data import asap7, freepdk45, orfs
from openlithohub.data._layers import LAYERS


def test_asap7_default_design_layer_matches_registry() -> None:
    assert asap7.DEFAULT_DESIGN_LAYER == LAYERS["asap7"].metal1 == (10, 0)


def test_freepdk45_default_design_layer_matches_registry() -> None:
    assert freepdk45.DEFAULT_DESIGN_LAYER == LAYERS["freepdk45"].metal1 == (11, 0)


def test_orfs_default_design_layer_matches_registry() -> None:
    assert orfs.DEFAULT_DESIGN_LAYER == LAYERS["orfs_asap7"].metal1 == (20, 0)


def test_orfs_layer_distinct_from_asap7_source() -> None:
    # The whole reason orfs_asap7 is a separate registry entry: post-route
    # M1 is on a different layer number than the cell-library source.
    assert LAYERS["orfs_asap7"].metal1 != LAYERS["asap7"].metal1
