"""Layer-purpose-pair (LPP) helpers — OpenAccess (Si2) + OASIS-compatible.

Industry layouts identify a polygon by **two** integers: the layer (the
mask layer it lives on, e.g. M1) and a *purpose* (what role the polygon
plays on that layer — a drawn shape, a pin, a routing blockage, a
boundary marker, ...). Tools encode the purpose differently:

* **GDSII / OASIS** — second integer is the *datatype*. Convention is
  per-foundry; ``0`` is universally "drawing". SEMI P39 (OASIS.MASK)
  inherits the same ``(layer, datatype)`` pair from OASIS itself.
* **OpenAccess (Si2)** — second integer is a registered *oaPurpose*
  (defined in ``oaLayer.h``). The set is small and stable, with names
  like ``drawing``, ``pin``, ``blockage``, ``net``, ``boundary``.

This module provides:

1. The canonical **purpose name → integer** map mirroring the OpenAccess
   default registry. Tools that emit OA-purpose numbers (Cadence /
   Synopsys / Si2 reference flow) round-trip cleanly.
2. ``classify_purpose(name)`` — a permissive alias resolver so common
   variants (``DRAWING``, ``drw``, ``pin1``) all fold to a canonical
   purpose.
3. ``LayerPurpose`` — a small frozen dataclass bundling
   ``(layer, datatype, purpose_name)`` so downstream code can branch on
   ``"pin"`` / ``"blockage"`` instead of hard-coding datatypes.

The mapping is **not** authoritative for any specific PDK — foundries
override datatypes freely. Use :func:`classify_purpose` as a default
when a layout file does not carry explicit purpose strings (most do not).
"""

from __future__ import annotations

from dataclasses import dataclass

# OpenAccess oaPurpose default registry (excerpt). The full list lives in
# Si2's ``oaLayer.h`` and is GPL-incompatible to redistribute, but the
# *names and numeric assignments* are de-facto public via published EDA
# textbooks and OpenROAD's LEF/DEF parsers (both Apache-2.0).
#
# These are the values an OpenAccess-aware tool will write into the GDS
# datatype column when `oaLayerPurposePair` semantics are enabled. For
# layouts produced by non-OA tools (most), the datatype is foundry-defined
# — see ``classify_purpose`` for permissive resolution.
OA_PURPOSE_TO_DATATYPE: dict[str, int] = {
    "drawing": 0,
    "net": 1,
    "pin": 2,
    "label": 3,
    "boundary": 4,
    "blockage": 5,
    "fill": 6,
    "fillopc": 7,
    "track": 8,
    "slot": 9,
    "annotation": 10,
    "warning": 11,
    "redundant": 12,
    "notch": 13,
    "cutsom": 14,  # OA spelling (sic)
}

DATATYPE_TO_OA_PURPOSE: dict[int, str] = {v: k for k, v in OA_PURPOSE_TO_DATATYPE.items()}

# Permissive aliases — case-insensitive variants seen in the wild.
# Maps lowercased input → canonical OA purpose name.
_PURPOSE_ALIASES: dict[str, str] = {
    # drawing / drawn shape
    "drawing": "drawing",
    "drw": "drawing",
    "draw": "drawing",
    "drawn": "drawing",
    "polygon": "drawing",
    "shape": "drawing",
    # pin
    "pin": "pin",
    "pin1": "pin",
    "pins": "pin",
    "terminal": "pin",
    "term": "pin",
    # blockage
    "blockage": "blockage",
    "block": "blockage",
    "obs": "blockage",
    "obstruction": "blockage",
    # boundary
    "boundary": "boundary",
    "bound": "boundary",
    "outline": "boundary",
    "prboundary": "boundary",
    "prbound": "boundary",
    # net (electrical net id)
    "net": "net",
    "netname": "net",
    # label / text annotation
    "label": "label",
    "text": "label",
    "txt": "label",
    # fill / dummy fill
    "fill": "fill",
    "dummy": "fill",
    "dummyfill": "fill",
    "filler": "fill",
    "fillopc": "fillopc",
    "opcfill": "fillopc",
    # track (preferred routing track)
    "track": "track",
    "tracks": "track",
    # slot (slotting for stress relief)
    "slot": "slot",
    "slotting": "slot",
    # annotation
    "annotation": "annotation",
    "anno": "annotation",
    # warning marker
    "warning": "warning",
    "warn": "warning",
    # redundant via / wire
    "redundant": "redundant",
    "redundancy": "redundant",
    "redun": "redundant",
    # notch marker
    "notch": "notch",
    # custom — preserve OA's typo and accept the corrected form
    "cutsom": "cutsom",
    "custom": "cutsom",
}


@dataclass(frozen=True)
class LayerPurpose:
    """One (layer, datatype, purpose-name) triple.

    ``purpose`` is the canonical OpenAccess name when known; ``None``
    when the input datatype has no registered purpose mapping.
    """

    layer: int
    datatype: int
    purpose: str | None

    @classmethod
    def from_pair(cls, layer: int, datatype: int) -> LayerPurpose:
        return cls(layer=layer, datatype=datatype, purpose=DATATYPE_TO_OA_PURPOSE.get(datatype))

    @classmethod
    def from_name(cls, layer: int, purpose_name: str) -> LayerPurpose:
        """Build from a layer number + a purpose alias.

        Raises ``KeyError`` when the alias is unknown — silently coercing
        unknown purpose names to ``drawing`` would mask layout bugs.
        """
        canonical = classify_purpose(purpose_name)
        if canonical is None:
            raise KeyError(
                f"Unknown OpenAccess purpose: {purpose_name!r}. "
                f"Known: {sorted(set(OA_PURPOSE_TO_DATATYPE))}"
            )
        return cls(
            layer=layer,
            datatype=OA_PURPOSE_TO_DATATYPE[canonical],
            purpose=canonical,
        )


def classify_purpose(name: str) -> str | None:
    """Resolve an arbitrary purpose alias to its canonical OA name.

    Returns ``None`` when ``name`` is unrecognized. The lookup is
    case-insensitive; whitespace and underscores are ignored so
    ``"DRAWING"``, ``"  drw  "``, and ``"draw_n"`` all resolve.
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace(" ", "").replace("_", "")
    return _PURPOSE_ALIASES.get(key)


def datatype_for_purpose(purpose_name: str) -> int:
    """Return the OA datatype number for a purpose alias.

    Raises ``KeyError`` for unknown aliases (same rationale as
    ``LayerPurpose.from_name``).
    """
    canonical = classify_purpose(purpose_name)
    if canonical is None:
        raise KeyError(f"Unknown OpenAccess purpose alias: {purpose_name!r}")
    return OA_PURPOSE_TO_DATATYPE[canonical]


def purpose_for_datatype(datatype: int) -> str | None:
    """Return the canonical OA purpose for a numeric datatype, or ``None``."""
    return DATATYPE_TO_OA_PURPOSE.get(datatype)
