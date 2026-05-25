"""Smoke tests for `scripts/train_gan_opc.py` v0.3 changes.

Keep the v0.3 wiring (Changes 1-5 per gan-opc-v0.3-improvements.md §2.3)
honest:
- _resize selects bilinear|area mode (Change 1; default bilinear).
- _step returns dict with bce/consistency/mrc/pvb keys.
- _lambda_mrc_at warm-up endpoints behave correctly.
- _lambda_pvb_at warm-up endpoints behave correctly.
- _pvb_loss is non-negative and connects gradients (Change 5).

The CLI smoke test (``--smoke-test``) is exercised separately.
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
        resize_mode="bilinear",
        consistency_weight=0.1,
        smoke_test=True,
        num_kernels=4,
        pixel_size_nm=4.0,
        lambda_mrc=0.5,
        lambda_mrc_warmup_epochs=10,
        lambda_mrc_warmup_start=0.0,
        mrc_min_width_nm=20.0,
        mrc_min_spacing_nm=20.0,
        mrc_weight_min_spacing=0.5,
        lambda_pvb=0.1,
        lambda_pvb_warmup_epochs=15,
        lambda_pvb_dose_delta=0.05,
        lambda_pvb_defocus_range_nm=20.0,
        lambda_pvb_sigma_nominal=2.0,
        seed=0,
    )
    base.update(overrides)
    return mod.TrainConfig(**base)


class TestResizeMode:
    """Change 1 regression: _resize must select bilinear|area; v0.3 defaults to bilinear."""

    def test_bilinear_resize_preserves_binary_dtype(self, mod) -> None:
        big = torch.zeros(256, 256)
        big[:, 124:133] = 1.0
        out = mod._GanOpcPairs._resize(big, 64, "bilinear")
        assert out.dtype == torch.float32
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_area_resize_still_supported(self, mod) -> None:
        big = torch.zeros(256, 256)
        big[:, 124:133] = 1.0
        out_area = mod._GanOpcPairs._resize(big, 64, "area")
        assert out_area.dtype == torch.float32
        assert out_area.sum() > 0

    def test_bilinear_differs_from_area_for_thin_feature(self, mod) -> None:
        # On a real-world-like random binary pattern the resize modes don't
        # agree byte-for-byte (otherwise reverting v0.2's area mode would be a
        # no-op).
        torch.manual_seed(42)
        big = (torch.rand(256, 256) > 0.7).float()
        out_b = mod._GanOpcPairs._resize(big, 64, "bilinear")
        out_a = mod._GanOpcPairs._resize(big, 64, "area")
        assert not torch.equal(out_b, out_a)

    def test_invalid_mode_raises(self, mod) -> None:
        with pytest.raises(ValueError):
            mod._GanOpcPairs(root=Path("/nonexistent"), resize_to=64, resize_mode="bicubic")


class TestStepDictShape:
    """_step must return dict with 'total','bce','consistency','mrc','pvb'."""

    def test_step_returns_dict_with_required_keys(self, mod) -> None:
        from openlithohub.models._unet import UNet

        torch.manual_seed(0)
        cfg = _make_cfg(mod, lambda_mrc=0.5, lambda_pvb=0.1)
        model = UNet().to(cfg.device)
        design = torch.zeros(2, 64, 64)
        design[:, 16:48, 16:48] = 1.0
        target = design.clone()
        losses = mod._step(model, (design, target), cfg, epoch=0)
        for k in (
            "total",
            "bce",
            "consistency",
            "mrc",
            "pvb",
            "lambda_mrc",
            "lambda_pvb",
            "mask_mean",
            "target_mean",
        ):
            assert k in losses, f"missing key {k}"
        assert losses["total"].requires_grad
        assert not losses["bce"].requires_grad
        assert not losses["pvb"].requires_grad


class TestLambdaMrcWarmup:
    def test_lambda_mrc_endpoints(self, mod) -> None:
        cfg = _make_cfg(
            mod,
            lambda_mrc=0.5,
            lambda_mrc_warmup_epochs=10,
            lambda_mrc_warmup_start=0.0,
        )
        assert mod._lambda_mrc_at(0, cfg) == pytest.approx(0.0)
        assert mod._lambda_mrc_at(10, cfg) == pytest.approx(0.5)
        assert mod._lambda_mrc_at(20, cfg) == pytest.approx(0.5)
        assert mod._lambda_mrc_at(5, cfg) == pytest.approx(0.25)

    def test_lambda_mrc_zero_disables(self, mod) -> None:
        cfg = _make_cfg(mod, lambda_mrc=0.0)
        assert mod._lambda_mrc_at(0, cfg) == 0.0
        assert mod._lambda_mrc_at(100, cfg) == 0.0


class TestLambdaPvbWarmup:
    """Change 5 warmup: 0.0 → lambda_pvb over lambda_pvb_warmup_epochs."""

    def test_lambda_pvb_endpoints(self, mod) -> None:
        cfg = _make_cfg(mod, lambda_pvb=0.1, lambda_pvb_warmup_epochs=15)
        assert mod._lambda_pvb_at(0, cfg) == pytest.approx(0.0)
        assert mod._lambda_pvb_at(15, cfg) == pytest.approx(0.1)
        assert mod._lambda_pvb_at(30, cfg) == pytest.approx(0.1)
        # half-way: 7.5 -> 0.05
        # use exact integer to avoid fractional epoch values.
        assert mod._lambda_pvb_at(5, cfg) == pytest.approx(5.0 / 15.0 * 0.1)

    def test_lambda_pvb_zero_disables(self, mod) -> None:
        cfg = _make_cfg(mod, lambda_pvb=0.0)
        assert mod._lambda_pvb_at(0, cfg) == 0.0
        assert mod._lambda_pvb_at(50, cfg) == 0.0


class TestPvbLossPhysics:
    """Change 5: _pvb_loss must be non-negative and connect gradients."""

    def test_pvb_loss_nonnegative_and_grad_flows(self, mod) -> None:
        cfg = _make_cfg(mod, lambda_pvb=0.1, pixel_size_nm=4.0)
        mask = torch.zeros(2, 1, 64, 64, requires_grad=True)
        with torch.no_grad():
            mask.data[..., 16:48, 16:48] = 1.0
        target = torch.zeros(2, 1, 64, 64)
        target[..., 16:48, 16:48] = 1.0
        loss = mod._pvb_loss(mask, target, cfg)
        assert float(loss.item()) >= 0.0
        loss.backward()
        assert mask.grad is not None
        assert float(mask.grad.abs().sum().item()) > 0.0

    def test_pvb_loss_smaller_on_sharp_than_blurred(self, mod) -> None:
        # P2(c)-style invariant baked into the test suite: loss is smaller
        # on the sharp mask than on a deliberately blurred copy.
        import torch.nn.functional as functional

        cfg = _make_cfg(mod, lambda_pvb=0.1, pixel_size_nm=4.0)
        sharp = torch.zeros(1, 1, 64, 64)
        sharp[..., 16:48, 16:48] = 1.0

        kernel_size = 7
        half = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32) - half
        g = torch.exp(-0.5 * (coords / 1.5) ** 2)
        g = g / g.sum()
        k1 = g.view(1, 1, 1, -1)
        k2 = g.view(1, 1, -1, 1)
        x = functional.pad(sharp, (half, half, 0, 0), mode="replicate")
        x = functional.conv2d(x, k1)
        x = functional.pad(x, (0, 0, half, half), mode="replicate")
        blurred = functional.conv2d(x, k2)

        target = sharp
        l_sharp = float(mod._pvb_loss(sharp, target, cfg).item())
        l_blur = float(mod._pvb_loss(blurred, target, cfg).item())
        assert l_sharp < l_blur, f"P2(c) violated in test: sharp={l_sharp} blurred={l_blur}"
