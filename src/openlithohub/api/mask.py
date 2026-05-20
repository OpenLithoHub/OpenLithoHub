"""`Mask` — frozen dataclass bundling (tensor, pixel_size_nm, layer).

Wraps OpenLithoHub's existing layout I/O so callers can write
``Mask.from_oasis("design.oas", layer="1:0")`` instead of going through
``load_layout`` and ``export_oasis`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

from openlithohub.data.io import load_layout
from openlithohub.workflow.export import export_gds, export_oasis


@dataclass(frozen=True)
class Mask:
    """A 2-D mask tensor with its physical pixel pitch and (optional) source layer.

    The fab/EDA-facing handle that ``LitheEngine`` consumes and produces.
    Construction is via classmethod constructors. The dataclass is frozen,
    so the three fields cannot be rebound after construction — but note
    that ``frozen=True`` does not protect against in-place mutation of the
    underlying tensor (``mask.tensor[i, j] = 0`` still mutates storage).
    Treat ``Mask`` as a structural binding of ``(tensor, pixel_size_nm,
    layer)``, not as a deep-immutable value.
    """

    tensor: torch.Tensor
    pixel_size_nm: float = 1.0
    layer: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.tensor.shape
        return int(h), int(w)

    def __array__(self, dtype: object = None) -> NDArray[np.float32]:
        arr = self.tensor.detach().cpu().numpy()
        if dtype is not None:
            # numpy hands `dtype` in via the protocol as an arbitrary object
            # (numpy's own type isn't exposed for this hook), so we narrow
            # it to what `astype` accepts before calling.
            arr = arr.astype(dtype)  # type: ignore[call-overload]
        return arr

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
    ) -> Mask:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 2:
            raise ValueError(f"Mask tensor must be 2-D (H, W), got ndim={tensor.ndim}")
        return cls(tensor=tensor.float(), pixel_size_nm=pixel_size_nm, layer=layer)

    @classmethod
    def from_pt(cls, path: str | Path, *, pixel_size_nm: float = 1.0) -> Mask:
        t = load_layout(Path(path), pixel_size_nm)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=None)

    @classmethod
    def from_npy(cls, path: str | Path, *, pixel_size_nm: float = 1.0) -> Mask:
        t = load_layout(Path(path), pixel_size_nm)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=None)

    @classmethod
    def from_oasis(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
    ) -> Mask:
        t = load_layout(Path(path), pixel_size_nm, layer=layer)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)

    @classmethod
    def from_gds(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
    ) -> Mask:
        t = load_layout(Path(path), pixel_size_nm, layer=layer)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)

    @classmethod
    def from_def(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
        lef_files: list[str | Path] | None = None,
    ) -> Mask:
        """Load a placed-and-routed DEF (IEEE 1481) layout.

        DEF carries placement + routing geometry but not cell internals;
        pass ``lef_files=[...]`` so KLayout can resolve cell abstracts.
        Without LEF context, the resulting raster contains only routing
        metal — std-cell internals are blank.
        """
        t = load_layout(Path(path), pixel_size_nm, layer=layer, lef_files=lef_files)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
        lef_files: list[str | Path] | None = None,
    ) -> Mask:
        """Suffix-sniffing constructor.

        ``.pt`` / ``.npy`` ignore ``layer``. ``.oas`` / ``.gds`` / ``.def``
        / ``.lef`` honour it. ``lef_files`` is only meaningful for
        ``.def`` / ``.lef`` inputs.
        """
        suffix = Path(path).suffix.lower()
        if suffix in {".pt", ".npy"}:
            if layer is not None:
                raise ValueError(f"layer is meaningless for {suffix} inputs")
            if lef_files is not None:
                raise ValueError(f"lef_files is meaningless for {suffix} inputs")
            t = load_layout(Path(path), pixel_size_nm)
            return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=None)
        if suffix in {".oas", ".gds"}:
            if lef_files is not None:
                raise ValueError(f"lef_files is meaningless for {suffix} inputs")
            t = load_layout(Path(path), pixel_size_nm, layer=layer)
            return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)
        if suffix in {".def", ".lef"}:
            t = load_layout(Path(path), pixel_size_nm, layer=layer, lef_files=lef_files)
            return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)
        raise ValueError(
            f"unsupported extension {suffix!r} — expected .pt / .npy / .oas / .gds / .def / .lef"
        )

    def to_pt(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.tensor, str(path))

    def to_npy(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.tensor.detach().cpu().numpy())

    def to_oasis(self, path: str | Path, *, mode: str = "curvilinear") -> None:
        export_oasis(self.tensor, path, mode=mode, pixel_size_nm=self.pixel_size_nm)

    def to_gds(self, path: str | Path, *, mode: str = "curvilinear") -> None:
        export_gds(self.tensor, path, mode=mode, pixel_size_nm=self.pixel_size_nm)
