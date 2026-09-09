"""Per-PDK design-layer registry.

The post-route layer numbering for "metal1" (and friends) differs across
PDKs and even between source vs post-route variants of the same PDK:

- ASAP7 cell library (``asap7sc7p5t_27/GDS/...``): M1 = (10, 0)
- FreePDK45 + NanGate stdcells (``mflowgen/freepdk-45nm``): M1 = (11, 0)
- ORFS-routed ASAP7 platform (``flow/platforms/asap7``): M1 = (20, 0)

Carrying three module-level ``DEFAULT_DESIGN_LAYER`` constants invited
the (10, 0) / (20, 0) docstring drift caught in the May 2026 review.
This single registry is the source of truth; PDK adapters import from
it and re-export ``DEFAULT_DESIGN_LAYER`` so call sites stay stable.

Layer mappings are loaded from ``layermaps/*.json`` next to this module.
Each JSON file maps layer names (``metal1``, ``via1``, etc.) to
``[layer_number, datatype]`` arrays. Users can add custom PDKs by
dropping a new ``.json`` file into the ``layermaps/`` directory or
calling :func:`register_layermap` at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PdkLayers:
    """Layer numbers for a single PDK / variant.

    ``metal1`` is the only mandatory field because every adapter
    rasterizes it as the design tensor by default. All other layers
    are optional — populated when the PDK's JSON layermap includes them.
    """

    metal1: tuple[int, int]
    metal2: tuple[int, int] | None = None
    metal3: tuple[int, int] | None = None
    via1: tuple[int, int] | None = None
    via2: tuple[int, int] | None = None
    contact: tuple[int, int] | None = None


def _load_layermap_json(path: Path) -> PdkLayers:
    """Load a layermap JSON file into a :class:`PdkLayers` instance.

    Each key is a layer name (``metal1``, ``via1``, etc.) and the value
    is a ``[layer_number, datatype]`` array. Keys that do not correspond
    to a :class:`PdkLayers` field are ignored so third-party layermap
    files may carry extra metadata (``poly``, ``nwell``, ...).
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(PdkLayers)}
    kwargs: dict[str, tuple[int, int]] = {}
    for key, value in data.items():
        if key in known and isinstance(value, list) and len(value) == 2:
            kwargs[key] = (int(value[0]), int(value[1]))
    if "metal1" not in kwargs:
        raise ValueError(f"{path}: layermap is missing the mandatory 'metal1' entry")
    return PdkLayers(**kwargs)


def load_layermap(path: Path | str) -> PdkLayers:
    """Load a custom layermap from a JSON file.

    Args:
        path: Path to a JSON file with layer number mappings.

    Returns:
        A :class:`PdkLayers` instance.

    Raises:
        ValueError: If the file has no valid ``metal1`` entry.
    """
    return _load_layermap_json(Path(path))


def register_layermap(name: str, layers: PdkLayers) -> None:
    """Register a custom PDK layer mapping at runtime.

    Args:
        name: PDK identifier (e.g. ``"my_custom_pdk"``).
        layers: Layer number configuration.
    """
    LAYERS[name] = layers


def list_pkds() -> list[str]:
    """Return sorted list of available PDK names."""
    return sorted(LAYERS.keys())


# Auto-discover bundled JSON layermaps. Each file in the layermaps/
# directory becomes a key in LAYERS (filename stem = PDK name). Invalid
# files are skipped with a warning rather than breaking the package
# import — a user experimenting with a custom layermap must not lose
# the whole ``openlithohub.data`` module over it.
LAYERS: dict[str, PdkLayers] = {}
_layermap_dir = Path(__file__).parent / "layermaps"
if _layermap_dir.is_dir():
    import warnings

    for _p in sorted(_layermap_dir.glob("*.json")):
        try:
            LAYERS[_p.stem] = _load_layermap_json(_p)
        except (ValueError, TypeError, json.JSONDecodeError) as _exc:
            warnings.warn(
                f"Skipping invalid layermap {_p.name}: {_exc}",
                UserWarning,
                stacklevel=2,
            )
