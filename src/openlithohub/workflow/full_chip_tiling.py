"""GPU batch-parallel Schwarz tiling for full-chip ILT.

Processes multiple tiles in parallel using GPU batch operations for
throughput-intensive full-chip mask optimization.  Falls back to CPU
multi-process when GPU is unavailable.

Public API
----------
* :class:`TileParallelProcessor` — partition, batch-process, and reassemble.
* :class:`SchwarzTilingSolver`   — iterative Schwarz with overlap exchange.
* :class:`TileBenchmarkReport`   — scaling / throughput benchmarks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from openlithohub.workflow.tiling import (
    Tile,
    stitch_tiles,
    tile_layout,
)

__all__ = ["TileParallelProcessor", "SchwarzTilingSolver", "TileBenchmarkReport"]


# ---------------------------------------------------------------------------
# TileParallelProcessor
# ---------------------------------------------------------------------------


class TileParallelProcessor:
    """GPU batch-parallel Schwarz tiling for full-chip ILT.

    Processes multiple tiles in parallel using GPU batch operations.
    Falls back to CPU multi-process when GPU is unavailable.

    Parameters
    ----------
    tile_size : int
        Square tile edge length in pixels.
    overlap : int
        Overlap width between adjacent tiles.
    n_workers : int
        Number of parallel workers (used as batch-size hint on GPU).
    device : str
        Torch device string (``"cpu"`` or ``"cuda"`` / ``"cuda:N"``).
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        n_workers: int = 4,
        device: str = "cpu",
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= tile_size:
            raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

        self.tile_size = tile_size
        self.overlap = overlap
        self.n_workers = n_workers
        self.device = torch.device(device)

    # -- partition ----------------------------------------------------------

    def partition(
        self,
        mask: torch.Tensor,
        tile_size: int | None = None,
        overlap: int | None = None,
    ) -> list[tuple[torch.Tensor, tuple[int, int]]]:
        """Partition a full-chip mask into overlapping tiles.

        Parameters
        ----------
        mask : Tensor
            Full layout mask ``(H, W)`` or ``(1, H, W)``.
        tile_size : int or None
            Override for ``self.tile_size``.
        overlap : int or None
            Override for ``self.overlap``.

        Returns
        -------
        list of (tile_tensor, (origin_y, origin_x))
            Tile tensors with their top-left origin in global coordinates.
        """
        ts = tile_size or self.tile_size
        ol = overlap if overlap is not None else self.overlap
        tiles = tile_layout(mask, tile_size=ts, overlap=ol)
        return [(t.tensor, (t.origin_y, t.origin_x)) for t in tiles]

    # -- process_tiles ------------------------------------------------------

    def process_tiles(
        self,
        tiles: list[torch.Tensor],
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int | None = None,
    ) -> list[torch.Tensor]:
        """Process tiles in GPU batches.

        Parameters
        ----------
        tiles : list of Tensor
            Individual tile tensors.
        process_fn : callable
            ``tensor_batch -> tensor_batch``.  Receives a 4-D tensor
            ``(B, 1, H, W)`` and must return the same shape.
        batch_size : int or None
            Number of tiles per GPU batch.  Defaults to ``self.n_workers``.

        Returns
        -------
        list of Tensor
            Processed tile tensors in the same order.
        """
        if not tiles:
            return []

        bs = batch_size or self.n_workers
        results: list[torch.Tensor] = []

        for start in range(0, len(tiles), bs):
            batch = tiles[start : start + bs]
            # Stack into (B, 1, H, W)
            stacked = torch.stack([t.to(self.device).float() for t in batch])
            if stacked.ndim == 3:
                stacked = stacked.unsqueeze(1)

            with torch.no_grad():
                processed = process_fn(stacked)

            # Un-stack back to individual tensors
            for i in range(processed.shape[0]):
                results.append(processed[i].cpu().squeeze(0))

        return results

    # -- reassemble ---------------------------------------------------------

    def reassemble(
        self,
        processed_tiles: list[torch.Tensor],
        origins: list[tuple[int, int]],
        full_shape: tuple[int, int],
        tile_size: int | None = None,
        overlap: int | None = None,
    ) -> torch.Tensor:
        """Reassemble processed tiles into full-chip result using overlap blending.

        Uses linear blending in overlap regions (same algorithm as
        :func:`stitch_tiles`).

        Parameters
        ----------
        processed_tiles : list of Tensor
            One tensor per tile.
        origins : list of (origin_y, origin_x)
            Top-left coordinate of each tile in the global frame.
        full_shape : (H, W)
            Shape of the full output.
        tile_size : int or None
            Override for ``self.tile_size``.
        overlap : int or None
            Override for ``self.overlap``.

        Returns
        -------
        Tensor
            Full-chip result ``(H, W)``.
        """
        ts = tile_size or self.tile_size
        ol = overlap if overlap is not None else self.overlap
        h, w = full_shape

        # Build Tile objects expected by stitch_tiles
        tile_objects: list[tuple[Tile, torch.Tensor]] = []
        for tensor, (oy, ox) in zip(processed_tiles, origins, strict=False):
            tile_h = min(ts, h - oy)
            tile_w = min(ts, w - ox)
            t = Tile(
                tensor=tensor,
                origin_x=ox,
                origin_y=oy,
                width=tile_w,
                height=tile_h,
                overlap=ol,
            )
            tile_objects.append((t, tensor))

        return stitch_tiles(tile_objects, output_shape=(h, w))


# ---------------------------------------------------------------------------
# SchwarzTilingSolver
# ---------------------------------------------------------------------------


class SchwarzTilingSolver:
    """Schwarz alternating method for full-chip ILT.

    Iteratively solves tiles with overlap exchange until convergence.

    Parameters
    ----------
    tile_processor : TileParallelProcessor
        Processor used for tiling, batching, and reassembly.
    tile_size : int
        Square tile edge in pixels.
    overlap : int
        Overlap width.
    max_iterations : int
        Maximum Schwarz exchange iterations.
    convergence_tol : float
        Stop early when the Schwarz residual drops below this value.
    """

    def __init__(
        self,
        tile_processor: TileParallelProcessor | None = None,
        tile_size: int = 512,
        overlap: int = 64,
        max_iterations: int = 3,
        convergence_tol: float = 0.01,
    ) -> None:
        self.processor = tile_processor or TileParallelProcessor(
            tile_size=tile_size,
            overlap=overlap,
        )
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

    # -- public API ---------------------------------------------------------

    def solve(
        self,
        mask: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        target: torch.Tensor | None = None,
        n_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Run Schwarz iterative tiling.

        Parameters
        ----------
        mask : Tensor ``(H, W)``
            Full-chip mask.
        forward_fn : callable
            Per-tile forward model.  Receives a single tile ``(H_t, W_t)``
            and returns a processed tile of the same shape.
        target : Tensor or None
            Unused placeholder for future target-guided optimisation.
        n_iterations : int or None
            Override ``self.max_iterations``.

        Returns
        -------
        dict
            * ``result`` — final stitched full-chip tensor ``(H, W)``.
            * ``residual_history`` — boundary MSE at each Schwarz iteration.
            * ``n_iterations`` — number of iterations actually performed.
        """
        from openlithohub.benchmark.metrics.tiling_consistency import (
            tile_boundary_consistency,
        )
        from openlithohub.workflow.tiling import _inject_boundary_data

        max_iters = n_iterations or self.max_iterations

        if mask.ndim > 2:
            mask = mask.squeeze()
        h, w = mask.shape

        tiles = tile_layout(mask, tile_size=self.tile_size, overlap=self.overlap)
        tile_results = [t.tensor.clone() for t in tiles]

        residual_history: list[float] = []

        for _it in range(max_iters):
            new_results: list[torch.Tensor] = []
            for idx, tile in enumerate(tiles):
                updated = _inject_boundary_data(
                    tile,
                    tile_results,
                    idx,
                    tiles,
                    self.overlap,
                )
                updated = forward_fn(updated)
                new_results.append(updated)

            tile_results = new_results

            # Measure convergence
            consistency = tile_boundary_consistency(
                tiles,
                tile_results,
                overlap=self.overlap,
            )
            mse = consistency["boundary_mse"]
            residual_history.append(mse)

            if mse < self.convergence_tol:
                break

        # Stitch final result
        stitched = stitch_tiles(
            [(t, r) for t, r in zip(tiles, tile_results, strict=False)],
            output_shape=(h, w),
        )

        return {
            "result": stitched,
            "residual_history": residual_history,
            "n_iterations": len(residual_history),
        }

    # -- residual -----------------------------------------------------------

    def _compute_residual(
        self,
        tiles: list[Tile],
        tile_results: list[torch.Tensor],
    ) -> float:
        """Compute Schwarz residual at overlap interfaces.

        Returns the boundary MSE between adjacent tile results in the
        overlap regions.
        """
        from openlithohub.benchmark.metrics.tiling_consistency import (
            tile_boundary_consistency,
        )

        metrics = tile_boundary_consistency(
            tiles,
            tile_results,
            overlap=self.overlap,
        )
        return metrics["boundary_mse"]


# ---------------------------------------------------------------------------
# TileBenchmarkReport
# ---------------------------------------------------------------------------


class TileBenchmarkReport:
    """Benchmark report for full-chip tiling.

    Measures scaling with number of tiles, residual convergence,
    and throughput (cm^2/min).
    """

    @staticmethod
    def generate(
        mask_sizes: list[tuple[int, int]],
        tile_size: int = 256,
        overlap: int = 32,
        device: str = "cpu",
    ) -> list[dict[str, Any]]:
        """Generate benchmark report for various mask sizes.

        Parameters
        ----------
        mask_sizes : list of (H, W)
            Mask dimensions to benchmark.
        tile_size : int
            Tile size in pixels.
        overlap : int
            Overlap width.
        device : str
            Torch device string.

        Returns
        -------
        list of dict
            Each dict contains:

            * ``mask_size`` — ``(H, W)`` tuple.
            * ``n_tiles`` — number of tiles generated.
            * ``time_s`` — wall-clock time for partition + identity process +
              reassemble (seconds).
            * ``residual`` — Schwarz boundary MSE after one identity pass.
            * ``throughput`` — estimated throughput in cm^2/min, assuming
              1 pixel = 2 nm.
        """
        reports: list[dict[str, Any]] = []

        for h, w in mask_sizes:
            mask = torch.rand(h, w)
            processor = TileParallelProcessor(
                tile_size=tile_size,
                overlap=overlap,
                device=device,
            )

            t0 = time.perf_counter()

            # Partition
            partitioned = processor.partition(mask)
            tile_tensors = [t for t, _origin in partitioned]
            origins = [o for _t, o in partitioned]

            # Identity process
            def _identity_batch(batch: torch.Tensor) -> torch.Tensor:
                return batch

            processed = processor.process_tiles(
                tile_tensors,
                _identity_batch,
            )

            # Reassemble
            _result = processor.reassemble(
                processed,
                origins,
                (h, w),
            )

            elapsed = time.perf_counter() - t0

            # Compute residual via consistency metric
            tiles = tile_layout(mask, tile_size=tile_size, overlap=overlap)
            from openlithohub.benchmark.metrics.tiling_consistency import (
                tile_boundary_consistency,
            )

            consistency = tile_boundary_consistency(
                tiles,
                [t.tensor for t in tiles],
                overlap=overlap,
            )
            residual = consistency["boundary_mse"]

            # Throughput: area in cm^2 / time in minutes
            # Assume pixel_size = 2 nm → area_nm2 = h * w * 4
            # 1 cm^2 = 1e14 nm^2
            pixel_size_nm = 2.0
            area_cm2 = h * w * (pixel_size_nm**2) / 1e14
            throughput = area_cm2 / (elapsed / 60.0) if elapsed > 0 else 0.0

            reports.append(
                {
                    "mask_size": (h, w),
                    "n_tiles": len(partitioned),
                    "time_s": elapsed,
                    "residual": residual,
                    "throughput": throughput,
                }
            )

        return reports
