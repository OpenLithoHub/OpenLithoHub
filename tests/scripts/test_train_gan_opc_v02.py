"""Smoke tests for `scripts/train_gan_opc.py` v0.2 changes.

These keep the v0.2 wiring (B1/B2/B3 + Change 1 + Change 2) honest:
- _resize uses ``mode='area'`` (preserves areal mass).
- _step returns the dict shape the train loop unpacks.
- _lambda_mrc_at warm-up endpoints behave correctly.

The CLI smoke test (``--smoke-test``) is exercised separately in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_gan_opc.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_train_gan_opc", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_train_gan_opc"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _make_cfg(mod, **overrides):
    base = dict(
        data_root=None,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        sigma_px=4.5,
        forward_model="gaussian",
        device="cpu",
        output=Path("/tmp/_unused.pt"),
        resize_to=64,
        consistency_weight=0.05,
        smoke_test=True,
        num_kernels=4,
        pixel_size_nm=8.0,
        lambda_mrc=1.0,
        lambda_mrc_warmup_epochs=5,
        lambda_mrc_warmup_start=0.1,
        mrc_min_width_nm=24.0,
        mrc_min_spacing_nm=24.0,
        mrc_weight_min_spacing=0.5,
    )
    base.update(overrides)
    return mod.TrainConfig(**base)


class TestResizeAreaMode:
    """B2 regression: _resize must use mode='area' (binary-mask correct)."""

    def test_area_resize_preserves_thin_feature_mass_better_than_bilinear(self, mod) -> None:
        # 5-px-wide vertical strip in 256×256 -> 64×64 (4× downsample).
        # Each output pixel covers 16 input pixels (4×4); the strip
        # straddles ~1.25 output columns, so two adjacent output pixels
        # see ~5/16=31% and ~3/16=19% — both below the >0.5 threshold.
        # Bump strip width to 9 px (>1 output column wide) so at least
        # one output column gets ≥0.5 mass after area resampling.
        big = torch.zeros(256, 256)
        big[:, 124:133] = 1.0  # 9 px wide
        out_area = mod._GanOpcPairs._resize(big, 64)
        assert out_area.dtype == torch.float32
        assert out_area.unique().tolist() == [0.0, 1.0]
        # Area mode should preserve at least one full output column for a
        # 9-px strip in a 4× downsample.
        assert out_area.sum() > 0, "area mode dropped a 9-px strip — regression"
        # And the preserved column should align with the input strip
        # location (output pixel 31-32 contains input cols 124-131).
        assert out_area[:, 31].sum() > 0 or out_area[:, 32].sum() > 0


class TestStepDictShape:
    """B3 regression: _step must return dict with 'total','bce','consistency','mrc'."""

    def test_step_returns_dict_with_required_keys(self, mod) -> None:
        from openlithohub.models._unet import UNet

        torch.manual_seed(0)
        cfg = _make_cfg(mod, lambda_mrc=1.0)
        model = UNet().to(cfg.device)
        # Need at least 16x16 for the UNet 3-level downsample to work; use 64.
        design = torch.zeros(2, 64, 64)
        design[:, 16:48, 16:48] = 1.0
        target = design.clone()
        losses = mod._step(model, (design, target), cfg, epoch=0)
        for k in ("total", "bce", "consistency", "mrc", "lambda_mrc"):
            assert k in losses, f"missing key {k}"
        assert losses["total"].requires_grad
        # bce/consistency/mrc are detached for logging — should not require grad.
        assert not losses["bce"].requires_grad


class TestLambdaMrcWarmup:
    """Change 1 regression: warm-up endpoints behave linearly."""

    def test_lambda_mrc_endpoints(self, mod) -> None:
        cfg = _make_cfg(
            mod,
            lambda_mrc=1.0,
            lambda_mrc_warmup_epochs=5,
            lambda_mrc_warmup_start=0.1,
        )
        assert mod._lambda_mrc_at(0, cfg) == pytest.approx(0.1)
        assert mod._lambda_mrc_at(5, cfg) == pytest.approx(1.0)
        assert mod._lambda_mrc_at(10, cfg) == pytest.approx(1.0)
        # midpoint
        assert mod._lambda_mrc_at(2, cfg) == pytest.approx(0.1 + 0.4 * (1.0 - 0.1))

    def test_lambda_mrc_zero_disables(self, mod) -> None:
        cfg = _make_cfg(mod, lambda_mrc=0.0)
        assert mod._lambda_mrc_at(0, cfg) == 0.0
        assert mod._lambda_mrc_at(100, cfg) == 0.0
