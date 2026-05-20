"""`Mask` — frozen dataclass bundling (tensor, pixel_size_nm, layer).

Wraps OpenLithoHub's existing layout I/O so callers can write
``Mask.from_oasis("design.oas", layer="1:0")`` instead of going through
``_load_layout_as_tensor`` and ``export_oasis`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

# `_load_layout_as_tensor` is the canonical reader (handles .pt / .npy /
# .oas / .gds + the "NUM:DTYPE" layer selector). It currently lives under
# `cli/` and carries a leading underscore; the HTTP server already imports
# it the same way (see server/app.py). A future PR can promote it to
# `openlithohub.data.io.load_layout` — out of scope here.
from openlithohub.cli.optimize_cmd import _load_layout_as_tensor
from openlithohub.workflow.export import export_oasis


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
            arr = arr.astype(dtype)
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
        t = _load_layout_as_tensor(Path(path), pixel_size_nm)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=None)

    @classmethod
    def from_npy(cls, path: str | Path, *, pixel_size_nm: float = 1.0) -> Mask:
        t = _load_layout_as_tensor(Path(path), pixel_size_nm)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=None)

    @classmethod
    def from_oasis(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
    ) -> Mask:
        t = _load_layout_as_tensor(Path(path), pixel_size_nm, layer=layer)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)

    @classmethod
    def from_gds(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
    ) -> Mask:
        t = _load_layout_as_tensor(Path(path), pixel_size_nm, layer=layer)
        return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        pixel_size_nm: float = 1.0,
        layer: str | None = None,
    ) -> Mask:
        """Suffix-sniffing constructor.

        ``.pt`` / ``.npy`` ignore ``layer``. ``.oas`` / ``.gds`` honour it.
        """
        suffix = Path(path).suffix.lower()
        if suffix in {".pt", ".npy"}:
            if layer is not None:
                raise ValueError(f"layer is meaningless for {suffix} inputs")
            t = _load_layout_as_tensor(Path(path), pixel_size_nm)
            return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=None)
        if suffix in {".oas", ".gds"}:
            t = _load_layout_as_tensor(Path(path), pixel_size_nm, layer=layer)
            return cls(tensor=t, pixel_size_nm=pixel_size_nm, layer=layer)
        raise ValueError(f"unsupported extension {suffix!r} — expected .pt / .npy / .oas / .gds")

    def to_pt(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.tensor, str(path))

    def to_npy(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.tensor.detach().cpu().numpy())

    def to_oasis(self, path: str | Path, *, mode: str = "curvilinear") -> None:
        export_oasis(self.tensor, path, mode=mode, pixel_size_nm=self.pixel_size_nm)

    def to_gds(self, path: str | Path, *, mode: str = "curvilinear") -> None:
        # `export_oasis` writes via klayout, which sniffs the format from the
        # filename suffix — `.gds` produces GDSII, `.oas` produces OASIS.
        export_oasis(self.tensor, path, mode=mode, pixel_size_nm=self.pixel_size_nm)
