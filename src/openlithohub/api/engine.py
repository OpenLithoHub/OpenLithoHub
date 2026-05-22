"""`LitheEngine` — thin wrapper over registry + tile/halo/stitch pipeline.

Mirrors the body of ``server.app._run_optimize`` minus filesystem I/O so
callers can drive the engine in-process without touching the HTTP server
or the CLI helpers.
"""

from __future__ import annotations

from typing import Any

import torch

from openlithohub.api.mask import Mask
from openlithohub.api.report import Report
from openlithohub.benchmark.compliance.drc import check_drc
from openlithohub.benchmark.compliance.mrc import check_curvilinear_mrc, check_mrc
from openlithohub.benchmark.metrics.epe import compute_epe, compute_wafer_epe
from openlithohub.benchmark.metrics.l2_error import compute_l2_error
from openlithohub.benchmark.metrics.pvband import compute_pvband
from openlithohub.benchmark.metrics.shot_count import estimate_shot_count
from openlithohub.models.base import LithographyModel
from openlithohub.models.registry import register_builtin_models, registry
from openlithohub.simulators.base import BaseSimulator, SimulatorConfig
from openlithohub.simulators.hopkins_sim import HopkinsSimulator
from openlithohub.workflow.halo import compute_halo_px
from openlithohub.workflow.process_node import ProcessNodeConfig, get_node
from openlithohub.workflow.tiling import stitch_tiles, tile_layout


class LitheEngine:
    """Object-oriented driver for the OpenLithoHub optimization pipeline.

    ``engine = LitheEngine(model="neural-ilt", node="3nm-euv")``
    ``optimized = engine.optimize(mask)``
    ``report = engine.evaluate(optimized, target=mask)``
    """

    def __init__(
        self,
        model: str | LithographyModel,
        *,
        node: str | None = None,
        tile_size: int = 2048,
        pretrained: bool = False,
        **model_kwargs: Any,
    ) -> None:
        register_builtin_models()

        if isinstance(model, str):
            kwargs: dict[str, Any] = dict(model_kwargs)
            if pretrained:
                kwargs.setdefault("pretrained", True)
            self._model: LithographyModel = registry.get(model, **kwargs)
            # Engine constructed the instance, so it owns the setup() call —
            # and, symmetrically, the teardown() in __exit__.
            self._model.setup()
            self._owns_model = True
        elif isinstance(model, LithographyModel):
            if model_kwargs or pretrained:
                raise ValueError(
                    "model_kwargs / pretrained only apply when `model` is a name; "
                    "pass a fully constructed LithographyModel without them."
                )
            # Caller-supplied instance: assume the caller has already called
            # setup(). Calling it again would re-load weights / re-init GPU
            # state in non-idempotent models like NeuralILTModel. The caller
            # also owns teardown — we must not close resources we did not open.
            self._model = model
            self._owns_model = False
        else:
            raise TypeError(
                f"`model` must be a name (str) or LithographyModel, got {type(model).__name__}"
            )

        # Let `get_node` raise KeyError on typos; silently coercing unknown
        # node names to None hides physics-affecting misconfiguration.
        self._node_config: ProcessNodeConfig | None = get_node(node) if node is not None else None
        self._tile_size = tile_size

    @property
    def model(self) -> LithographyModel:
        return self._model

    @property
    def node(self) -> ProcessNodeConfig | None:
        return self._node_config

    @staticmethod
    def list_models() -> list[str]:
        register_builtin_models()
        return registry.list_models()

    def close(self) -> None:
        """Tear down the underlying model if the engine constructed it.

        Safe to call multiple times. No-op for caller-supplied models
        (the caller owns those — closing them here would yank resources
        out from under code the engine never owned).
        """
        if self._owns_model and self._model is not None:
            self._model.teardown()
            self._owns_model = False

    def __enter__(self) -> LitheEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _resolve_pixel_size(self, supplied: float) -> float:
        # The Mask carries its pixel pitch; trust it. ``_coerce_to_mask``
        # already substitutes the node's native pitch when the caller
        # passed a bare tensor without a pitch annotation, so by the time
        # we reach here the value is authoritative.
        #
        # Earlier versions used ``supplied == 1.0`` as a "not set" sentinel,
        # but 1.0 nm/px is a legitimate pitch (e.g. ICCAD16 benchmarks),
        # which silently overrode the caller's value with the node's.
        return supplied

    def _build_simulator(self, pixel_nm: float) -> BaseSimulator | None:
        # Issue #72: when the engine is bound to a process node, the
        # wafer-level metrics (compute_wafer_epe, compute_l2_error) must
        # see *that* node's wavelength / NA / illumination. Otherwise
        # each metric internally constructs a fresh HopkinsSimulator at
        # the 193 nm DUV / NA 1.35 / 1.0 nm/px defaults and a 3 nm EUV
        # engine silently scores against DUV optics.
        #
        # Returning None falls back to each metric's default — preserves
        # existing behaviour when no node is bound (callers explicitly
        # picking an unconfigured engine were not expecting our defaults
        # to apply).
        if self._node_config is None:
            return None
        node = self._node_config
        config = SimulatorConfig(
            wavelength_nm=node.wavelength_nm,
            na=node.numerical_aperture,
            sigma=node.sigma_outer,
            sigma_inner=node.sigma_inner,
            pixel_size_nm=pixel_nm,
        )
        return HopkinsSimulator(config)

    def _coerce_to_mask(self, design: Mask | torch.Tensor) -> Mask:
        if isinstance(design, Mask):
            return design
        if isinstance(design, torch.Tensor):
            pixel_nm = self._node_config.pixel_size_nm if self._node_config is not None else 1.0
            return Mask.from_tensor(design, pixel_size_nm=pixel_nm)
        raise TypeError(f"expected Mask or torch.Tensor, got {type(design).__name__}")

    def optimize(self, design: Mask | torch.Tensor) -> Mask:
        """Run the model over ``design`` with tiling + halo + stitching.

        Returns a binarised ``Mask`` matching the input shape and pixel pitch.
        """
        in_mask = self._coerce_to_mask(design)
        pixel_nm = self._resolve_pixel_size(in_mask.pixel_size_nm)
        tensor = in_mask.tensor

        halo_px = compute_halo_px(
            node=self._node_config,
            model=self._model,
            pixel_nm=pixel_nm,
            tile_size=self._tile_size,
        )

        tiles = tile_layout(tensor, tile_size=self._tile_size, overlap=halo_px)
        tile_results: list[tuple[Any, torch.Tensor]] = []
        for tile in tiles:
            result = self._model.predict(tile.tensor)
            tile_results.append((tile, result.mask))

        h, w = tensor.shape
        stitched = stitch_tiles(tile_results, (int(h), int(w)))
        binarised = (stitched > 0.5).float()

        return Mask(tensor=binarised, pixel_size_nm=pixel_nm, layer=in_mask.layer)

    def evaluate(
        self,
        predicted: Mask | torch.Tensor,
        target: Mask | torch.Tensor,
    ) -> Report:
        """Compute the canonical metric / compliance battery on ``predicted`` vs ``target``."""
        pred = self._coerce_to_mask(predicted)
        tgt = self._coerce_to_mask(target)
        if pred.shape != tgt.shape:
            raise ValueError(f"shape mismatch: predicted {pred.shape} vs target {tgt.shape}")
        if pred.pixel_size_nm != tgt.pixel_size_nm:
            raise ValueError(
                f"pixel_size_nm mismatch: predicted {pred.pixel_size_nm} vs "
                f"target {tgt.pixel_size_nm}. EPE is reported in nanometers, so "
                f"masks at different pitches cannot be compared without resampling."
            )

        pixel_nm = self._resolve_pixel_size(pred.pixel_size_nm)

        epe = compute_epe(pred.tensor, tgt.tensor, pixel_size_nm=pixel_nm)
        # Wafer-level EPE: forward-simulate then compare. The mask-level
        # `epe` above is 0 for an Identity model by construction; this
        # one isn't, since diffraction and resist threshold reshape the
        # printed contour.
        #
        # Build the simulator from the engine's node so the wavelength /
        # NA / illumination match what `optimize()` would have run; metric
        # defaults (193 nm DUV / NA 1.35) would otherwise silently override
        # an EUV engine's bound node parameters (issue #72).
        simulator = self._build_simulator(pixel_nm)
        wafer_epe = compute_wafer_epe(
            pred.tensor, tgt.tensor, pixel_size_nm=pixel_nm, simulator=simulator
        )
        # L2 wafer error — Neural-ILT canonical printability scalar.
        # Same forward-sim path as wafer_epe, different aggregation:
        # |wafer - target|.sum() instead of edge-distance.
        l2 = compute_l2_error(pred.tensor, tgt.tensor, pixel_size_nm=pixel_nm, simulator=simulator)
        pvband = compute_pvband(pred.tensor, pixel_size_nm=pixel_nm)
        drc = check_drc(pred.tensor, pixel_size_nm=pixel_nm)
        mrc = check_mrc(pred.tensor, pixel_size_nm=pixel_nm)
        shots = estimate_shot_count(pred.tensor, pixel_size_nm=pixel_nm)
        curvilinear_mrc = (
            check_curvilinear_mrc(pred.tensor, pixel_size_nm=pixel_nm)
            if self._model.supports_curvilinear
            else None
        )

        # Recompute the same halo `optimize()` would have used at this pitch,
        # so the report documents the tile/halo configuration that produced
        # (or would produce) `predicted` — useful for reproducing a run.
        halo_px = compute_halo_px(
            node=self._node_config,
            model=self._model,
            pixel_nm=pixel_nm,
            tile_size=self._tile_size,
        )

        return Report(
            epe_mean_nm=float(epe["epe_mean_nm"]),
            epe_max_nm=float(epe["epe_max_nm"]),
            epe_std_nm=float(epe["epe_std_nm"]),
            epe_wafer_mean_nm=float(wafer_epe["epe_mean_nm"]),
            epe_wafer_max_nm=float(wafer_epe["epe_max_nm"]),
            epe_wafer_std_nm=float(wafer_epe["epe_std_nm"]),
            l2_error_pixels=float(l2["l2_error_pixels"]),
            l2_error_nm2=float(l2["l2_error_nm2"]),
            pvband_mean_nm=float(pvband["pvband_mean_nm"]),
            pvband_max_nm=float(pvband["pvband_max_nm"]),
            drc_violations=int(drc.violation_count),
            drc_passed=bool(drc.passed),
            mrc_violations=int(mrc.violation_count),
            mrc_passed=bool(mrc.passed),
            shot_count=int(shots["shot_count"]),
            estimated_write_time_s=float(shots["estimated_write_time_s"]),
            model_name=self._model.name,
            pixel_size_nm=pixel_nm,
            tile_size=int(self._tile_size),
            halo_px=int(halo_px),
            raw_epe=epe,
            raw_wafer_epe=wafer_epe,
            raw_l2=l2,
            raw_drc=drc,
            raw_mrc=mrc,
            raw_pvband=pvband,
            raw_shot_count=shots,
            raw_curvilinear_mrc=curvilinear_mrc,
        )
