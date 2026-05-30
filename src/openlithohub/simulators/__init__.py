"""Simulator backend interface and adapters.

The :class:`BaseSimulator` ABC defines a vendor-neutral interface for
lithography forward simulation. It exists so that downstream code
(metrics, ILT loops, ground-truth oracles in tests) can target a single
shape regardless of whether the backend is the bundled Hopkins/SOCS
model, a commercial simulator like Calibre nmOPC or Tachyon, or a
research prototype.

Two concrete adapters are provided:

* :class:`HopkinsSimulator` — wraps the bundled
  :func:`openlithohub._utils.hopkins.simulate_aerial_image_hopkins` and
  is the reference implementation. Differentiable end-to-end.
* :class:`CalibreSimulator`, :class:`TachyonSimulator` — vendor adapters
  that require the respective commercial toolchain on ``PATH`` and a
  license file. When the toolchain is unavailable, they support a
  ``mock_mode`` (via ``config.extra["mock_mode"] = True``) that produces
  synthetic aerial images for testing and CI. We deliberately do not
  bundle any vendor toolchain or license.

Use :func:`get_simulator` to construct one by string name; the registry
is open for users to register their own via :func:`register_simulator`.

This module is the answer to OpenLithoHub_NextSteps item #12.
"""

from __future__ import annotations

from openlithohub._utils.optics import (
    load_source_intensity,
    load_zernike_coefficients,
    zernike_phase_map,
)
from openlithohub.simulators.base import (
    BaseSimulator,
    SimulatorConfig,
    SimulatorResult,
)
from openlithohub.simulators.calibre import CalibreSimulator
from openlithohub.simulators.commercial import (
    CommercialSimulatorAdapter,
    PreflightStatus,
    ToolchainError,
)
from openlithohub.simulators.hopkins_sim import HopkinsSimulator
from openlithohub.simulators.registry import (
    describe_simulators,
    get_simulator,
    list_available_backends,
    list_simulators,
    register_simulator,
)
from openlithohub.simulators.tachyon import TachyonSimulator

__all__ = [
    "BaseSimulator",
    "CalibreSimulator",
    "CommercialSimulatorAdapter",
    "HopkinsSimulator",
    "PreflightStatus",
    "ToolchainError",
    "SimulatorConfig",
    "SimulatorResult",
    "TachyonSimulator",
    "describe_simulators",
    "get_simulator",
    "list_available_backends",
    "list_simulators",
    "load_source_intensity",
    "load_zernike_coefficients",
    "register_simulator",
    "zernike_phase_map",
]
