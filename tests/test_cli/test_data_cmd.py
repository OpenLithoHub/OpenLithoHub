"""Tests for ``openlithohub data list`` and ``openlithohub data show``."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

# Adapter rasterization needs klayout; same gate as the upstream adapter tests.
pytest.importorskip("klayout.db")

from openlithohub.cli.app import app  # noqa: E402
from openlithohub.data.asap7 import CANONICAL_CELLS as ASAP7_CELLS  # noqa: E402
from openlithohub.data.asap7 import DEFAULT_DESIGN_LAYER as ASAP7_LAYER  # noqa: E402
from openlithohub.data.freepdk45_sram import (  # noqa: E402
    CANONICAL_CELLS as SRAM_CELLS,
)
from openlithohub.data.freepdk45_sram import (  # noqa: E402
    DEFAULT_DESIGN_LAYER as SRAM_LAYER,
)

runner = CliRunner()


# ---------- helpers ----------


def _build_synthetic_asap7_gds(target: Path) -> None:
    """Tiny ASAP7-shaped GDS with the four canonical cell names."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.00025
    metal = layout.layer(*ASAP7_LAYER)
    sizes = {
        "INVx1_ASAP7_75t_R": (824, 1256),
        "NAND2x1_ASAP7_75t_R": (1452, 1256),
        "NOR2x1_ASAP7_75t_R": (1464, 1256),
        "DFFHQNx1_ASAP7_75t_R": (4464, 1256),
    }
    for name, (w, h) in sizes.items():
        cell = layout.create_cell(name)
        cell.shapes(metal).insert(kdb.Box(0, 0, w, 200))
        cell.shapes(metal).insert(kdb.Box(0, h - 200, w, h))
    target.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(target))


def _write_sram_cell_gds(target: Path, name: str, *, w_dbu: int, h_dbu: int) -> None:
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.0005
    metal = layout.layer(*SRAM_LAYER)
    cell = layout.create_cell(name)
    rail = max(1, h_dbu // 20)
    cell.shapes(metal).insert(kdb.Box(0, 0, w_dbu, rail))
    cell.shapes(metal).insert(kdb.Box(0, h_dbu - rail, w_dbu, h_dbu))
    target.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(target))


@pytest.fixture
def asap7_root(tmp_path) -> Path:
    gds = tmp_path / "asap7sc7p5t_27" / "GDS" / "asap7sc7p5t_27_R_999999.gds"
    _build_synthetic_asap7_gds(gds)
    return tmp_path


@pytest.fixture
def sram_lib(tmp_path) -> Path:
    lib = tmp_path / "gds_lib"
    _write_sram_cell_gds(lib / "cell_1rw.gds", "cell_1rw", w_dbu=1800, h_dbu=3000)
    return lib


# ---------- data list ----------


class TestDataList:
    def test_list_asap7_prints_canonical_cells(self):
        result = runner.invoke(app, ["data", "list", "asap7"])
        assert result.exit_code == 0, result.output
        for cell in ASAP7_CELLS:
            assert cell in result.output
        # license info goes to stderr, but typer's CliRunner merges into output
        assert "BSD-3-Clause" in result.output
        assert "asap7" in result.output

    def test_list_freepdk45_sram_prints_canonical_cells(self):
        result = runner.invoke(app, ["data", "list", "freepdk45-sram"])
        assert result.exit_code == 0, result.output
        for cell in SRAM_CELLS:
            assert cell in result.output
        assert "OpenRAM" in result.output

    def test_list_unknown_dataset_errors(self):
        result = runner.invoke(app, ["data", "list", "not-a-real-dataset"])
        assert result.exit_code != 0
        assert "not-a-real-dataset" in result.output


# ---------- data show ----------


class TestDataShow:
    def test_show_asap7_renders_png(self, asap7_root, tmp_path):
        out = tmp_path / "inv.png"
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--cell",
                "INV",  # shorthand → INVx1_ASAP7_75t_R
                "--data-root",
                str(asap7_root),
                "--accept-license",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists() and out.stat().st_size > 0
        assert "INVx1_ASAP7_75t_R" in result.output
        assert "BSD-3-Clause" in result.output

    def test_show_asap7_requires_accept_license(self, asap7_root, tmp_path):
        out = tmp_path / "inv.png"
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--cell",
                "INV",
                "--data-root",
                str(asap7_root),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "accept-license" in result.output
        assert not out.exists()

    def test_show_asap7_requires_data_root(self, tmp_path):
        out = tmp_path / "inv.png"
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--cell",
                "INV",
                "--accept-license",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "data-root" in result.output

    def test_show_freepdk45_sram_renders_png(self, sram_lib, tmp_path, monkeypatch):
        # The CLI's freepdk45-sram path auto-locates the OpenRAM bundle. We
        # don't have openram installed, so monkeypatch the locator to point
        # at the synthetic gds_lib. This exercises everything except the
        # importlib.resources lookup itself (covered in the adapter tests).
        from openlithohub.data import freepdk45_sram

        monkeypatch.setattr(freepdk45_sram, "_locate_openram_gds_lib", lambda: sram_lib)

        out = tmp_path / "cell_1rw.png"
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "freepdk45-sram",
                "--cell",
                "cell_1rw",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists() and out.stat().st_size > 0
        assert "cell_1rw" in result.output

    def test_show_default_out_path(self, asap7_root, tmp_path, monkeypatch):
        # Default --out is '<cell>.png' in cwd; cd to tmp so we don't litter.
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--cell",
                "INV",
                "--data-root",
                str(asap7_root),
                "--accept-license",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "INV.png").exists()

    def test_show_design_layer_override_parses(self, asap7_root, tmp_path):
        out = tmp_path / "inv.png"
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--cell",
                "INV",
                "--data-root",
                str(asap7_root),
                "--accept-license",
                "--design-layer",
                "10/0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_show_design_layer_bad_format_errors(self, asap7_root, tmp_path):
        out = tmp_path / "inv.png"
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--cell",
                "INV",
                "--data-root",
                str(asap7_root),
                "--accept-license",
                "--design-layer",
                "bogus",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "design-layer" in result.output.lower() or "bogus" in result.output


# ---------- data show --all ----------


class TestDataShowAll:
    def test_all_asap7_renders_every_canonical_cell(self, asap7_root, tmp_path):
        out_dir = tmp_path / "asap7-out"
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--all",
                "--data-root",
                str(asap7_root),
                "--accept-license",
                "--out",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        for cell in ASAP7_CELLS:
            assert (out_dir / f"{cell}.png").exists()
        # summary line goes to stderr (merged into output)
        assert f"{len(ASAP7_CELLS)} cells" in result.output

    def test_all_freepdk45_sram_renders_every_canonical_cell(self, tmp_path, monkeypatch):
        # Build a synthetic gds_lib with all 10 cells, monkeypatch the locator.
        lib = tmp_path / "gds_lib"
        sizes = {
            "cell_1rw": (1800, 3000),
            "cell_2rw": (2400, 3000),
            "dff": (4000, 3000),
            "sense_amp": (3000, 3000),
            "write_driver": (3000, 3000),
            "tri_gate": (2000, 3000),
            "replica_cell_1rw": (1800, 3000),
            "replica_cell_2rw": (2400, 3000),
            "dummy_cell_1rw": (1800, 3000),
            "dummy_cell_2rw": (2400, 3000),
        }
        for name, (w, h) in sizes.items():
            _write_sram_cell_gds(lib / f"{name}.gds", name, w_dbu=w, h_dbu=h)

        from openlithohub.data import freepdk45_sram

        monkeypatch.setattr(freepdk45_sram, "_locate_openram_gds_lib", lambda: lib)

        out_dir = tmp_path / "fp45-out"
        result = runner.invoke(
            app,
            ["data", "show", "freepdk45-sram", "--all", "--out", str(out_dir)],
        )
        assert result.exit_code == 0, result.output
        for cell in SRAM_CELLS:
            assert (out_dir / f"{cell}.png").exists()
        assert f"{len(SRAM_CELLS)} cells" in result.output

    def test_all_default_out_dir(self, asap7_root, tmp_path, monkeypatch):
        # No --out: directory defaults to '<dataset>-cells/' in cwd.
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--all",
                "--data-root",
                str(asap7_root),
                "--accept-license",
            ],
        )
        assert result.exit_code == 0, result.output
        default_dir = tmp_path / "asap7-cells"
        assert default_dir.is_dir()
        for cell in ASAP7_CELLS:
            assert (default_dir / f"{cell}.png").exists()

    def test_all_and_cell_mutually_exclusive(self, asap7_root, tmp_path):
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--all",
                "--cell",
                "INV",
                "--data-root",
                str(asap7_root),
                "--accept-license",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_no_cell_and_no_all_errors(self, asap7_root, tmp_path):
        result = runner.invoke(
            app,
            [
                "data",
                "show",
                "asap7",
                "--data-root",
                str(asap7_root),
                "--accept-license",
                "--out",
                str(tmp_path / "out.png"),
            ],
        )
        assert result.exit_code != 0
        assert "--cell" in result.output or "--all" in result.output


# ---------- data export ----------


class TestDataExport:
    def test_export_asap7_webdataset(self, asap7_root, tmp_path):
        out_dir = tmp_path / "shards-wds"
        result = runner.invoke(
            app,
            [
                "data",
                "export",
                "asap7",
                "--output",
                str(out_dir),
                "--format",
                "webdataset",
                "--data-root",
                str(asap7_root),
                "--accept-license",
            ],
        )
        assert result.exit_code == 0, result.output
        shards = list(out_dir.glob("shard-*.tar"))
        assert len(shards) == 1
        assert (out_dir / "croissant.json").exists()

    def test_export_asap7_parquet_with_shards(self, asap7_root, tmp_path):
        pytest.importorskip("pyarrow")
        out_dir = tmp_path / "shards-pq"
        result = runner.invoke(
            app,
            [
                "data",
                "export",
                "asap7",
                "--output",
                str(out_dir),
                "--format",
                "parquet",
                "--shards",
                "2",
                "--data-root",
                str(asap7_root),
                "--accept-license",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(list(out_dir.glob("shard-*.parquet"))) == 2

    def test_export_rejects_unknown_format(self, asap7_root, tmp_path):
        result = runner.invoke(
            app,
            [
                "data",
                "export",
                "asap7",
                "--output",
                str(tmp_path / "out"),
                "--format",
                "csv",
                "--data-root",
                str(asap7_root),
                "--accept-license",
            ],
        )
        assert result.exit_code != 0
        assert "format" in result.output.lower()

    def test_export_rejects_shards_and_shard_size_together(self, asap7_root, tmp_path):
        result = runner.invoke(
            app,
            [
                "data",
                "export",
                "asap7",
                "--output",
                str(tmp_path / "out"),
                "--shards",
                "2",
                "--shard-size",
                "1MB",
                "--data-root",
                str(asap7_root),
                "--accept-license",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
