"""Tests for the paper-ready visualization helpers."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import torch  # noqa: E402

from openlithohub.data import generate_dummy_layout  # noqa: E402
from openlithohub.vis import (  # noqa: E402
    IEEE_STYLE,
    SPIE_STYLE,
    paper_style,
    plot_contours,
    plot_pv_band,
)


@pytest.fixture
def masks() -> tuple[torch.Tensor, torch.Tensor]:
    target = generate_dummy_layout(size=64, seed=0)
    predicted = generate_dummy_layout(size=64, seed=1)
    return target, predicted


def test_paper_style_context_manager_restores_rcparams() -> None:
    import matplotlib.pyplot as plt

    before = plt.rcParams["font.size"]
    with paper_style("ieee"):
        assert plt.rcParams["font.size"] == IEEE_STYLE["font.size"]
    assert plt.rcParams["font.size"] == before


def test_paper_style_supports_spie() -> None:
    import matplotlib.pyplot as plt

    with paper_style("spie"):
        assert plt.rcParams["font.size"] == SPIE_STYLE["font.size"]


def test_paper_style_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="unknown style"), paper_style("nature"):  # type: ignore[arg-type]
        pass


def test_plot_contours_returns_figure(masks) -> None:
    target, predicted = masks
    fig = plot_contours(target, predicted, pixel_size_nm=2.0, title="t", style="ieee")
    assert fig is not None
    axes = fig.get_axes()
    assert len(axes) == 1


def test_plot_contours_saves_pdf(tmp_path, masks) -> None:
    target, predicted = masks
    out = tmp_path / "out.pdf"
    plot_contours(target, predicted, save_path=out, style="ieee")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pv_band_runs(masks) -> None:
    target, _ = masks
    inner = target
    outer = target  # degenerate but exercises the code path
    fig = plot_pv_band(target, inner, outer, style="spie")
    assert fig is not None
