"""Tests for the opt-in deterministic torch backend helper."""

from __future__ import annotations

import torch

from openlithohub._utils.determinism import set_deterministic


def test_set_deterministic_toggles_backend_flags():
    """Idempotent: snapshot prior state, call, assert flags, restore."""
    prior = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    try:
        # Prove it actually flips: pre-set the opposite of what the helper
        # should produce.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        set_deterministic()

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
        ) = prior


def test_set_deterministic_strict_sets_workspace_env(monkeypatch):
    """``strict=True`` exports CUBLAS_WORKSPACE_CONFIG (cuBLAS requires it
    for deterministic matmul) and flips ``use_deterministic_algorithms``.

    Use ``setdefault`` semantics: if the user already set it, we don't
    clobber. Helper does ``os.environ.setdefault``.
    """
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    prior_flag = torch.are_deterministic_algorithms_enabled()
    try:
        set_deterministic(strict=True)
        import os

        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(prior_flag)


def test_set_deterministic_is_idempotent():
    """Calling twice should not raise nor change the result."""
    set_deterministic()
    set_deterministic()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
