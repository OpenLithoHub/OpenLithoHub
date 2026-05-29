"""VAE architecture benchmarking across lithography and nanophotonics domains.

Provides utilities to compare VAE architectures between OpenLithoHub (float32,
AdaptiveAvgPool -> FC encoder) and DiffNano (float64, stride-2 convolution
encoder).  The benchmark measures reconstruction quality, convergence speed,
and inference latency so that architecture choices are data-driven rather
than anecdotal.

Architecture reference
~~~~~~~~~~~~~~~~~~~~~~
- **DiffNano**: float64, Conv2d(stride=2) x2 -> FC(mu, logvar) encoder;
  ConvTranspose2d x2 + bilinear upsample decoder.
- **OpenLithoHub**: float32, same conv structure but with AdaptiveAvgPool2d(1)
  before FC for dimension-agnostic encoding.

The benchmark does not import DiffNano directly -- it accepts arbitrary
``nn.Module`` pairs (encoder, decoder) and measures their performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ReconstructionReport:
    """Result from a single-model reconstruction benchmark.

    Attributes
    ----------
    mse : float
        Mean squared error over the test set.
    time_ms : float
        Total inference time in milliseconds (encode + decode).
    n_samples : int
        Number of test samples evaluated.
    """

    mse: float
    time_ms: float
    n_samples: int


@dataclass
class ConvergenceReport:
    """Result from a VAE convergence benchmark.

    Attributes
    ----------
    losses : list[float]
        Per-epoch training losses.
    final_loss : float
        Loss after the last epoch.
    time_ms : float
        Total training time in milliseconds.
    """

    losses: list[float] = field(default_factory=list)
    final_loss: float = float("inf")
    time_ms: float = 0.0


@dataclass
class ComparisonReport:
    """Side-by-side comparison of two VAE architectures.

    Attributes
    ----------
    model_a_name : str
        Label for the first model.
    model_b_name : str
        Label for the second model.
    reconstruction_a : ReconstructionReport
    reconstruction_b : ReconstructionReport
    convergence_a : ConvergenceReport
    convergence_b : ConvergenceReport
    summary : dict[str, str]
        Human-readable comparison highlights (e.g. "model_a is 2.3x faster").
    """

    model_a_name: str = ""
    model_b_name: str = ""
    reconstruction_a: ReconstructionReport | None = None
    reconstruction_b: ReconstructionReport | None = None
    convergence_a: ConvergenceReport | None = None
    convergence_b: ConvergenceReport | None = None
    summary: dict[str, str] = field(default_factory=dict)


class VAEBenchmark:
    """Benchmark utility for comparing VAE architectures.

    Measures reconstruction quality (MSE), inference latency, and training
    convergence.  Works with any encoder/decoder pair that follows the
    standard VAE interface:

    - ``encoder(x) -> (mu, logvar)`` where x is ``(B, 1, H, W)``
    - ``decoder(z) -> recon`` where z is ``(B, latent_dim)``

    Parameters
    ----------
    device : str or torch.device
        Device for benchmarking.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def benchmark_reconstruction(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        test_data: torch.Tensor,
    ) -> ReconstructionReport:
        """Measure reconstruction MSE and inference latency.

        Parameters
        ----------
        encoder : nn.Module
            Encoder network mapping ``(B, 1, H, W) -> (mu, logvar)``.
        decoder : nn.Module
            Decoder network mapping ``(B, latent_dim) -> (B, 1, H, W)``.
        test_data : Tensor, shape ``(N, 1, H, W)``
            Test set of binary masks.

        Returns
        -------
        ReconstructionReport
        """
        encoder = encoder.to(self.device).eval()
        decoder = decoder.to(self.device).eval()
        data = test_data.to(self.device)

        # Warmup
        mu, logvar = encoder(data[:2])
        z = self._reparameterize(mu, logvar)
        _ = decoder(z)

        start = time.perf_counter()
        mu, logvar = encoder(data)
        z = self._reparameterize(mu, logvar)
        recon = decoder(z)
        elapsed = (time.perf_counter() - start) * 1000.0

        mse = nn.functional.mse_loss(recon, data).item()
        return ReconstructionReport(
            mse=mse,
            time_ms=elapsed,
            n_samples=data.shape[0],
        )

    def benchmark_convergence(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        target: torch.Tensor,
        n_steps: int = 200,
        lr: float = 0.05,
        latent_dim: int | None = None,
    ) -> ConvergenceReport:
        """Measure latent-space optimisation convergence speed.

        Trains a VAE on ``target``, then optimises a latent vector to
        reconstruct it, recording loss per step.

        Parameters
        ----------
        encoder : nn.Module
        decoder : nn.Module
        target : Tensor, shape ``(H, W)`` or ``(1, H, W)``
            Target mask to reconstruct.
        n_steps : int
            Number of latent optimisation steps.
        lr : float
            Adam learning rate for latent optimisation.
        latent_dim : int or None
            Latent dimension.  If None, inferred from encoder's final FC layer.

        Returns
        -------
        ConvergenceReport
        """
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        target = target.to(self.device).float()

        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3:
            target = target.unsqueeze(0)

        # Phase 1: quick VAE pre-training on random masks
        grid_size = target.shape[-1]
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        encoder.train()
        decoder.train()
        for _ in range(30):
            raw = torch.rand(64, 1, grid_size, grid_size, device=self.device)
            masks = (raw > 0.5).float()
            mu, logvar = encoder(masks)
            z = self._reparameterize(mu, logvar)
            recon = decoder(z)
            recon_loss = nn.functional.mse_loss(recon, masks, reduction="sum")
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
            loss = recon_loss + kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        encoder.eval()
        decoder.eval()

        # Phase 2: latent optimisation convergence
        with torch.no_grad():
            mu_init, _ = encoder(target)
        z = mu_init.squeeze(0).clone().detach().requires_grad_(True)

        for p in decoder.parameters():
            p.requires_grad_(False)

        latent_opt = torch.optim.Adam([z], lr=lr)
        losses: list[float] = []

        start = time.perf_counter()
        for _ in range(n_steps):
            latent_opt.zero_grad()
            recon = decoder(z.unsqueeze(0))
            loss = nn.functional.mse_loss(recon, target)
            loss.backward()
            latent_opt.step()
            losses.append(loss.item())
        elapsed = (time.perf_counter() - start) * 1000.0

        # Restore decoder grad
        for p in decoder.parameters():
            p.requires_grad_(True)

        return ConvergenceReport(
            losses=losses,
            final_loss=losses[-1] if losses else float("inf"),
            time_ms=elapsed,
        )

    def compare_architectures(
        self,
        model_a: tuple[nn.Module, nn.Module],
        model_b: tuple[nn.Module, nn.Module],
        test_data: torch.Tensor,
        name_a: str = "model_a",
        name_b: str = "model_b",
        convergence_target: torch.Tensor | None = None,
        convergence_steps: int = 200,
    ) -> ComparisonReport:
        """Compare two VAE architectures across reconstruction and convergence.

        Parameters
        ----------
        model_a : (encoder, decoder) tuple
        model_b : (encoder, decoder) tuple
        test_data : Tensor, shape ``(N, 1, H, W)``
            Test set for reconstruction benchmark.
        name_a, name_b : str
            Human-readable labels.
        convergence_target : Tensor or None
            Target mask for convergence benchmark.  If None, uses the first
            test sample.
        convergence_steps : int
            Number of latent optimisation steps.

        Returns
        -------
        ComparisonReport
        """
        enc_a, dec_a = model_a
        enc_b, dec_b = model_b

        recon_a = self.benchmark_reconstruction(enc_a, dec_a, test_data)
        recon_b = self.benchmark_reconstruction(enc_b, dec_b, test_data)

        target = convergence_target
        if target is None:
            target = test_data[0, 0]  # (H, W)

        conv_a = self.benchmark_convergence(enc_a, dec_a, target, n_steps=convergence_steps)
        conv_b = self.benchmark_convergence(enc_b, dec_b, target, n_steps=convergence_steps)

        # Build summary
        summary: dict[str, str] = {}

        if recon_a.mse < recon_b.mse:
            ratio = recon_b.mse / max(recon_a.mse, 1e-12)
            summary["reconstruction"] = (
                f"{name_a} has {ratio:.2f}x lower MSE "
                f"({recon_a.mse:.6f} vs {recon_b.mse:.6f})"
            )
        else:
            ratio = recon_a.mse / max(recon_b.mse, 1e-12)
            summary["reconstruction"] = (
                f"{name_b} has {ratio:.2f}x lower MSE "
                f"({recon_b.mse:.6f} vs {recon_a.mse:.6f})"
            )

        faster_name = name_a if recon_a.time_ms <= recon_b.time_ms else name_b
        slower_name = name_b if faster_name == name_a else name_a
        faster_time = min(recon_a.time_ms, recon_b.time_ms)
        slower_time = max(recon_a.time_ms, recon_b.time_ms)
        if slower_time > 0:
            summary["latency"] = (
                f"{faster_name} is {slower_time / max(faster_time, 1e-6):.2f}x faster "
                f"({faster_time:.1f} ms vs {slower_time:.1f} ms)"
            )

        if conv_a.final_loss < conv_b.final_loss:
            summary["convergence"] = (
                f"{name_a} converges to lower loss "
                f"({conv_a.final_loss:.6f} vs {conv_b.final_loss:.6f})"
            )
        else:
            summary["convergence"] = (
                f"{name_b} converges to lower loss "
                f"({conv_b.final_loss:.6f} vs {conv_a.final_loss:.6f})"
            )

        return ComparisonReport(
            model_a_name=name_a,
            model_b_name=name_b,
            reconstruction_a=recon_a,
            reconstruction_b=recon_b,
            convergence_a=conv_a,
            convergence_b=conv_b,
            summary=summary,
        )
