"""Lock the PDK layer registry against silent drift.

Each adapter's exported ``DEFAULT_DESIGN_LAYER`` must match the
registry entry. If someone later edits an adapter to override the
default, this test forces them to update ``_layers.py`` too — that's
the whole point of having a single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from openlithohub.data import asap7, freepdk45, orfs
from openlithohub.data._layers import (
    LAYERS,
    PdkLayers,
    load_layermap,
    list_pkds,
    register_layermap,
)


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


# --- Configurable layer mapping tests ---


class TestBundledLayermaps:
    def test_sky130_loaded(self) -> None:
        assert "sky130" in LAYERS
        assert LAYERS["sky130"].metal1 == (67, 20)

    def test_asap7_has_multi_layer(self) -> None:
        assert LAYERS["asap7"].metal2 is not None
        assert LAYERS["asap7"].via1 is not None

    def test_orfs_asap7_has_multi_layer(self) -> None:
        assert LAYERS["orfs_asap7"].metal2 is not None


class TestLoadLayermap:
    def test_load_from_file(self, tmp_path: Path) -> None:
        layermap = {"metal1": [5, 0], "metal2": [6, 0]}
        p = tmp_path / "custom.json"
        p.write_text(json.dumps(layermap))
        result = load_layermap(p)
        assert result.metal1 == (5, 0)
        assert result.metal2 == (6, 0)

    def test_load_metal1_only(self, tmp_path: Path) -> None:
        p = tmp_path / "minimal.json"
        p.write_text(json.dumps({"metal1": [1, 0]}))
        result = load_layermap(p)
        assert result.metal1 == (1, 0)
        assert result.metal2 is None


class TestRegisterLayermap:
    def test_register_custom_pdk(self) -> None:
        custom = PdkLayers(metal1=(99, 0))
        register_layermap("test_custom", custom)
        assert "test_custom" in LAYERS
        assert LAYERS["test_custom"].metal1 == (99, 0)
        del LAYERS["test_custom"]


class TestListPdks:
    def test_includes_all_bundled(self) -> None:
        names = list_pkds()
        assert "asap7" in names
        assert "freepdk45" in names
        assert "orfs_asap7" in names
        assert "sky130" in names

    def test_sorted(self) -> None:
        names = list_pkds()
        assert names == sorted(names)
