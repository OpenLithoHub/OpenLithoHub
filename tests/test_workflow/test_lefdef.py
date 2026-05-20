"""DEF (IEEE 1481) + LEF ingestion through the workflow / data / api layers.

We synthesize a minimal LEF + DEF pair in ``tmp_path`` (see the
``lefdef_pair`` conftest fixture) and verify that:

* ``workflow.parsing.parse_layout`` reads the DEF when handed companion
  LEF context, exposes layers / cells / bbox.
* ``data.io.load_layout`` rasterizes the DEF onto a layer the LEF
  defines (``1:3`` is metal1.OBS in KLayout's default LEF/DEF mapping).
* ``api.Mask.from_def`` and ``Mask.load`` honour ``lef_files`` and
  reject it for non-DEF/LEF inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("klayout.db")

from openlithohub import Mask
from openlithohub.data.io import load_layout
from openlithohub.workflow.parsing import parse_layout


class TestParseLayoutDef:
    def test_def_with_lef_context(self, lefdef_pair: tuple[Path, Path]) -> None:
        deff, lef = lefdef_pair
        info = parse_layout(deff, lef_files=[lef])
        assert info["bounding_box"]["x_max"] == 5000
        assert info["bounding_box"]["y_max"] == 5000
        layer_names = {layer["name"] for layer in info["layers"]}
        # KLayout's stock DEF reader emits one layer per LEF metal/purpose.
        assert any("metal1" in name for name in layer_names)
        cell_names = {c["name"] for c in info["cells"]}
        assert "top" in cell_names

    def test_lef_only(self, lefdef_pair: tuple[Path, Path]) -> None:
        _, lef = lefdef_pair
        # LEF on its own has no DESIGN block — KLayout should still
        # produce a layout (cell abstracts), no error.
        info = parse_layout(lef)
        assert info["cells"], "LEF read should expose at least one cell"

    def test_def_without_lef_warns_or_empties(self, lefdef_pair: tuple[Path, Path]) -> None:
        deff, _ = lefdef_pair
        # Without LEF context, KLayout still parses the DEF but the INV
        # macro resolves to an empty cell. We don't assert KLayout's
        # exact behaviour (it varies by version) — just that we don't
        # crash.
        info = parse_layout(deff)
        assert "bounding_box" in info


class TestLoadLayoutDef:
    def test_def_rasterizes_metal1_obs(self, lefdef_pair: tuple[Path, Path]) -> None:
        deff, lef = lefdef_pair
        # KLayout maps LEF "metal1" + OBS purpose to layer 1, datatype 3.
        out = load_layout(deff, pixel_nm=10.0, layer="1:3", lef_files=[lef])
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2
        # The OBS rect is 0.05–0.45 × 0.05–0.95 µm, placed at (1,1) µm.
        # At 10 nm/px the die is 500×500 px and the OBS covers ~40×90 px.
        assert out.shape == (500, 500)
        assert (out > 0).any(), "rasterized OBS should produce non-zero pixels"

    def test_def_multilayer_requires_layer(self, lefdef_pair: tuple[Path, Path]) -> None:
        deff, lef = lefdef_pair
        # The DEF + LEF resolves to >1 layer; loader must refuse to
        # collapse them.
        with pytest.raises(ValueError, match="--layer"):
            load_layout(deff, pixel_nm=10.0, lef_files=[lef])


class TestMaskDef:
    def test_from_def(self, lefdef_pair: tuple[Path, Path]) -> None:
        deff, lef = lefdef_pair
        m = Mask.from_def(deff, pixel_size_nm=10.0, layer="1:3", lef_files=[lef])
        assert m.shape == (500, 500)
        assert m.layer == "1:3"

    def test_load_dispatches_def(self, lefdef_pair: tuple[Path, Path]) -> None:
        deff, lef = lefdef_pair
        m = Mask.load(deff, pixel_size_nm=10.0, layer="1:3", lef_files=[lef])
        assert m.shape == (500, 500)

    def test_load_rejects_lef_files_for_pt(
        self, tmp_path: Path, sample_design: torch.Tensor
    ) -> None:
        pt = tmp_path / "x.pt"
        Mask.from_tensor(sample_design).to_pt(pt)
        with pytest.raises(ValueError, match="lef_files is meaningless"):
            Mask.load(pt, lef_files=[tmp_path / "fake.lef"])
