"""LevelSet-ILT: Iterative mask optimization via gradient descent.

The level-set / continuous-mask formulation of Inverse Lithography
Technology dates to Pang, Liu & Abrams, *Inverse lithography technology
principles in practice: unintuitive patterns* (Proc. SPIE 5992, 2005)
and Poonawala & Milanfar, *Mask design for optical microlithography —
an inverse imaging problem* (IEEE TIP 16(3), 2007). This implementation
follows the SimpleILT-style L2 + total-variation formulation surveyed
in [Yang2023_LithoBench, §3.3, p.5] (open-access substitute for the
paywalled Granik / Pang journal write-ups).

Confidence **B** — algorithmic intent is matched against the
LithoBench narrative; specific hyperparameters here (``lr``,
``sigma_px``, ``tv_weight``) are this project's defaults, not literal
paper values.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch

from openlithohub._utils.forward_model import simulate_aerial_image
from openlithohub._utils.hopkins import (
    HopkinsParams,
    compute_socs_kernels,
    simulate_aerial_image_hopkins,
)
from openlithohub._utils.resist_model import differentiable_threshold
from openlithohub.models.base import LithographyModel, PredictionResult
from openlithohub.models.registry import registry
from openlithohub.workflow.process_window import (
    DEFAULT_PW_CORNERS,
    pw_fidelity_loss,
)

ForwardModelKind = Literal["gaussian", "hopkins"]


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Compute isotropic total variation for a 2D tensor."""
    diff_h = (x[1:, :] - x[:-1, :]).pow(2)
    diff_w = (x[:, 1:] - x[:, :-1]).pow(2)
    return diff_h.sum() + diff_w.sum()


@registry.register
class LevelSetILTModel(LithographyModel):
    """Inverse Lithography Technology via level-set gradient descent.

    Optimizes a continuous mask representation to minimize the difference
    between the simulated resist image and the target design pattern.
    Supports two forward models:

    - ``gaussian`` (default): a single Gaussian PSF — fast, used in tests.
    - ``hopkins``: SOCS-truncated partial-coherence Hopkins imaging — physically
      faithful, suitable for end-to-end AI-OPC research.
    """

    NAME = "levelset-ilt"
    SUPPORTS_CURVILINEAR = True
    # Issue #75: see openilt.py — iterative ILT gradient flow propagates
    # across many OIR's worth of pixels, so 0 px under-shoots the seam-free
    # halo at tile boundaries. 64 px is the same conservative bound used by
    # the U-Net-based Neural-ILT / GAN-OPC.
    RECEPTIVE_FIELD_PX = 64

    def __init__(
        self,
        iterations: int = 200,
        lr: float = 0.1,
        sigma_px: float = 2.0,
        tv_weight: float = 0.01,
        dose: float = 1.0,
        resist_steepness: float = 50.0,
        forward_model: ForwardModelKind = "gaussian",
        hopkins_params: HopkinsParams | None = None,
    ) -> None:
        self._iterations = iterations
        self._lr = lr
        self._sigma_px = sigma_px
        self._tv_weight = tv_weight
        self._dose = dose
        self._resist_steepness = resist_steepness
        self._forward_model = forward_model
        self._hopkins_params = hopkins_params or HopkinsParams()
        self._cached_kernels: torch.Tensor | None = None
        self._cached_weights: torch.Tensor | None = None
        self._cached_grid: int | None = None
        self._compiled_hopkins_cache: dict[tuple[Any, ...], Any] = {}

    def _ensure_hopkins_kernels(
        self, grid_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_kernels is None
            or self._cached_weights is None
            or self._cached_grid != grid_size
            or self._cached_kernels.device != device
        ):
            kernels, weights = compute_socs_kernels(self._hopkins_params, grid_size, device)
            self._cached_kernels = kernels
            self._cached_weights = weights
            self._cached_grid = grid_size
        return self._cached_kernels, self._cached_weights

    def predict(self, design: torch.Tensor, **kwargs: Any) -> PredictionResult:
        """Optimize a mask to reproduce the target design under lithography simulation.

        Args:
            design: Target design pattern (H, W), binary.
            **kwargs: Optional overrides — iterations, lr, sigma_px, tv_weight,
                forward_model, hopkins_params, device, dtype, compile_forward,
                process_window, pw_corners, checkpoint_dir, save_freq, resume_from.

                ``process_window=True`` swaps the nominal-only fidelity loss
                for ``workflow.process_window.pw_fidelity_loss`` evaluated
                across ``pw_corners`` (defaults to
                ``workflow.process_window.DEFAULT_PW_CORNERS``). Currently only
                supported on the ``gaussian`` forward model — pass
                ``forward_model="gaussian"`` (the default) when enabling PW.

                ``checkpoint_dir`` (``str | Path | None``) + ``save_freq``
                (int, default 0 = off) write the running ``mask_logit`` and
                Adam state to ``{checkpoint_dir}/ckpt_iter{N}.pt`` every
                ``save_freq`` iterations. ``resume_from`` (``str | Path``)
                loads such a file and continues from its iteration counter,
                so a SLURM preemption or CUDA crash does not erase prior
                progress.
        """
        target = design.detach().float()
        if target.ndim > 2:
            target = target.squeeze()

        iterations = kwargs.get("iterations", self._iterations)
        lr = kwargs.get("lr", self._lr)
        sigma_px = kwargs.get("sigma_px", self._sigma_px)
        tv_weight = kwargs.get("tv_weight", self._tv_weight)
        forward_model = kwargs.get("forward_model", self._forward_model)
        hopkins_params = kwargs.get("hopkins_params", self._hopkins_params)
        dtype = kwargs.get("dtype", torch.float32)
        compile_forward = kwargs.get("compile_forward", False)
        device = kwargs.get("device")
        process_window = kwargs.get("process_window", False)
        pw_corners = kwargs.get("pw_corners", DEFAULT_PW_CORNERS)
        checkpoint_dir = kwargs.get("checkpoint_dir")
        save_freq = int(kwargs.get("save_freq", 0))
        resume_from = kwargs.get("resume_from")
        if save_freq < 0:
            raise ValueError(f"save_freq must be >= 0, got {save_freq}")
        if save_freq > 0 and checkpoint_dir is None:
            raise ValueError("save_freq > 0 requires checkpoint_dir to be set.")
        ckpt_dir_path: Path | None = None
        if checkpoint_dir is not None:
            ckpt_dir_path = Path(checkpoint_dir)
            ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        if process_window and forward_model != "gaussian":
            raise ValueError(
                "process_window=True currently only supports forward_model='gaussian'; "
                "Hopkins corner sweeps will land in a follow-up. Set "
                "forward_model='gaussian' or process_window=False."
            )
        if device is not None:
            target = target.to(device)

        if forward_model == "hopkins":
            if hopkins_params != self._hopkins_params:
                self._hopkins_params = hopkins_params
                self._cached_kernels = None
                self._cached_weights = None
                self._cached_grid = None
            kernels, weights = self._ensure_hopkins_kernels(target.shape[0], target.device)
            hopkins_fn: Callable[..., torch.Tensor] | None = simulate_aerial_image_hopkins
            if compile_forward:
                cache_key = (
                    target.shape[0],
                    target.shape[-1],
                    str(target.device),
                    str(dtype),
                    forward_model,
                )
                compiled = self._compiled_hopkins_cache.get(cache_key)
                if compiled is None:
                    # Lift dynamo's per-function recompile ceiling so a run that
                    # legitimately encounters several tile sizes (e.g. ORFS with
                    # mixed-pitch designs) keeps cache hits instead of evicting.
                    try:
                        import torch._dynamo as _dynamo

                        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
                    except Exception:  # noqa: BLE001, S110 — best-effort tuning; old PyTorch lacks the symbol
                        pass
                    try:
                        compiled = torch.compile(hopkins_fn, mode="reduce-overhead", dynamic=False)
                    except Exception:
                        # torch.compile may be unavailable (Windows without Triton,
                        # some MPS configs, torch < 2.0). Falling back to eager
                        # keeps the run alive — caller can pass --no-compile to
                        # silence the attempt.
                        compiled = hopkins_fn
                    self._compiled_hopkins_cache[cache_key] = compiled
                hopkins_fn = compiled  # type: ignore[assignment]
        else:
            kernels = None
            weights = None
            hopkins_fn = None

        mask_logit = torch.zeros_like(target, requires_grad=True)
        with torch.no_grad():
            mask_logit.copy_(target * 4.0 - 2.0)
        mask_logit = mask_logit.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([mask_logit], lr=lr)

        best_loss = float("inf")
        best_mask: torch.Tensor = target.clone()
        start_iter = 0

        if resume_from is not None:
            # ``weights_only=False`` because we serialise an Adam state_dict
            # alongside the tensor — pickled Python dicts that torch refuses
            # to load under the 2.6+ default. Checkpoints are written by
            # this code path only, so the trust boundary is the user's own
            # filesystem.
            state = torch.load(str(resume_from), map_location=target.device, weights_only=False)
            with torch.no_grad():
                mask_logit.copy_(state["mask_logit"])
            optimizer.load_state_dict(state["optimizer"])
            start_iter = int(state.get("iteration", 0))
            best_loss = float(state.get("best_loss", best_loss))
            if "best_mask" in state:
                best_mask = state["best_mask"].to(target.device)

        for it in range(start_iter, iterations):
            optimizer.zero_grad()

            mask_continuous = torch.sigmoid(mask_logit)
            if forward_model == "hopkins":
                assert hopkins_fn is not None  # narrowed by forward_model == "hopkins" branch
                aerial = hopkins_fn(
                    mask_continuous,
                    kernels=kernels,
                    weights=weights,
                    dose=self._dose,
                    dtype=dtype,
                )
            else:
                aerial = simulate_aerial_image(mask_continuous, sigma_px=sigma_px, dose=self._dose)
            if aerial.dtype != torch.float32:
                aerial = aerial.float()

            resist = differentiable_threshold(
                aerial, threshold=0.5, steepness=self._resist_steepness
            )

            if process_window:
                fidelity_loss = pw_fidelity_loss(
                    mask_continuous,
                    target,
                    corners=pw_corners,
                    threshold=0.5,
                    steepness=self._resist_steepness,
                )
            else:
                fidelity_loss = torch.nn.functional.mse_loss(resist, target)
            tv_loss = _total_variation(mask_continuous)
            loss = fidelity_loss + tv_weight * tv_loss

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_mask = (mask_continuous > 0.5).float().detach()

            if save_freq > 0 and ckpt_dir_path is not None and (it + 1) % save_freq == 0:
                # ``it + 1`` is the count of *completed* steps, so a resume
                # from this file restarts iteration index at it+1 and runs
                # exactly ``iterations - (it+1)`` more steps — total work
                # equals an uninterrupted run.
                torch.save(
                    {
                        "iteration": it + 1,
                        "mask_logit": mask_logit.detach().clone(),
                        "optimizer": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "best_mask": best_mask.detach().clone(),
                    },
                    str(ckpt_dir_path / f"ckpt_iter{it + 1}.pt"),
                )

        return PredictionResult(
            mask=best_mask,
            metadata={
                "final_loss": best_loss,
                "iterations": iterations,
                "sigma_px": sigma_px,
                "forward_model": forward_model,
                "process_window": process_window,
                "pw_corner_count": len(pw_corners) if process_window else 0,
            },
        )
