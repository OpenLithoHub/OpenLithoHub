"""Opt-in deterministic torch backends for benchmark / Hackathon scoring.

Background
----------
Even with a fixed RNG seed, two runs of the same model on the same input
can disagree at the last few bits when:

* cuDNN picks a different convolution algorithm between runs
  (``cudnn.benchmark = True`` heuristic search).
* TF32 is enabled on Ampere+ GPUs — ``matmul`` and ``cudnn`` both quietly
  truncate to 19-bit mantissa.
* Atomic adds in scatter / gather kernels reorder summation across blocks.

For a Hackathon leaderboard where two participants submitting the same
model code must produce the same score, those last-bit drifts are
unacceptable. This helper centralises the flags used to suppress them.

Why opt-in
----------
The deterministic settings are slower (cuDNN can no longer pick the
fastest algorithm; TF32 is disabled). We do not want every user training
or running ``optimize`` to pay that cost — only the scoring paths need
bit-reproducibility. ``set_deterministic()`` is therefore a no-op unless
explicitly called.

``torch.use_deterministic_algorithms(True)`` is intentionally NOT set
here: a handful of common ops (some scatter variants, ``index_add``) are
not deterministic-implemented, and turning the strict mode on by default
would error out otherwise-fine models. Callers who want the strict mode
can pass ``strict=True``.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def set_deterministic(*, strict: bool = False) -> None:
    """Configure torch backends for bit-reproducible benchmark scoring.

    Sets ``cudnn.deterministic = True``, ``cudnn.benchmark = False``, and
    disables TF32 on both ``matmul`` and ``cudnn``. Idempotent and safe to
    call on CPU-only machines (the CUDA-specific knobs are still
    settable on the Python side; they just have no effect).

    ``strict=True`` additionally calls ``torch.use_deterministic_algorithms(True)``
    and exports ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (required by cuBLAS
    for deterministic matmul). Strict mode raises at runtime when an op
    has no deterministic implementation; opt in only when you have
    verified the model's op set supports it.

    Important: ``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS at the
    handle's first use. If CUDA has already been initialised before this
    function is called (any tensor on a CUDA device, any cublas-backed
    matmul), setting the env var here is a no-op and determinism is
    silently lost. Call ``set_deterministic(strict=True)`` *before* any
    CUDA work, or set ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the shell
    environment before launching Python. We log a warning when we detect
    CUDA was already initialised so the silent failure becomes a visible
    one.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    if strict:
        already_set = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # CUBLAS_WORKSPACE_CONFIG is consumed at cuBLAS handle creation,
        # which happens on first cuBLAS use. If CUDA has already been
        # initialised the env var write above is too late to take effect.
        if (
            already_set is None and torch.cuda.is_available() and torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        ):
            logger.warning(
                "set_deterministic(strict=True) called after CUDA was "
                "initialised; CUBLAS_WORKSPACE_CONFIG=:4096:8 was set but "
                "cuBLAS will not pick it up. Re-launch with the env var "
                "exported in the shell, or call set_deterministic before "
                "any CUDA tensor is created."
            )
        torch.use_deterministic_algorithms(True)

    logger.info(
        "deterministic mode enabled (strict=%s): cudnn.deterministic=True, "
        "cudnn.benchmark=False, allow_tf32=False on cudnn+matmul",
        strict,
    )
