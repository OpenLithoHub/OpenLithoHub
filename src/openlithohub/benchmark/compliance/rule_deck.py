"""Standard MRC (Mask Rule Check) rule-deck schema + loader.

Different fabs and process nodes ship MRC rules in different formats:
Calibre SVRF, Synopsys ICV runset, vendor-specific YAML, etc. None of
these are easy to read across teams. This module defines a single
JSON / TOML rule-deck format that:

* Captures every parameter the OpenLithoHub MRC checkers consume
  (Manhattan ``min_width_nm``/``min_spacing_nm``, curvilinear
  ``min_curvature_radius_nm``/``min_feature_area_nm2``).
* Carries enough metadata (process node, layer, mask polarity,
  pixel size) to be reproducible and auditable.
* Passes a JSON Schema validation step before being trusted.

The schema is intentionally narrow — it is **not** a full SVRF
replacement. It covers what the optical / OPC side needs to score a
mask. DRC-on-the-design side (transistor sizing, well rules, etc.)
stays in the existing process-node configs.

Usage::

    deck = load_rule_deck("rules/n5_metal1.json")
    result = check_mrc(mask, **deck.kwargs_manhattan())
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

if sys.version_info >= (3, 11):
    import tomllib as _tomllib  # noqa: F401  # re-exported via _load_toml
else:  # Python 3.10
    try:
        import tomli as _tomllib
    except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10 w/o tomli
        _tomllib = None  # type: ignore[assignment, unused-ignore]


def _load_toml(fh: IO[bytes]) -> dict[str, Any]:
    """Read a TOML file via stdlib (3.11+) or `tomli` fallback (3.10)."""
    if _tomllib is None:
        raise ImportError(
            "TOML rule decks on Python 3.10 require 'tomli'. "
            "Install with: pip install tomli — or use a .json deck instead."
        )
    data: dict[str, Any] = _tomllib.load(fh)
    return data


# Single source of truth: every rule-deck file must validate against this.
RULE_DECK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://openlithohub.com/schemas/mrc-rule-deck-v1.json",
    "title": "OpenLithoHub MRC Rule Deck v1",
    "type": "object",
    "required": ["schema_version", "process_node", "layer", "rules"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["1"]},
        "process_node": {
            "type": "string",
            "description": "Tech node tag (e.g. 'asap7', 'freepdk45', 'n5').",
        },
        "layer": {
            "type": "string",
            "description": "Mask layer tag (e.g. 'metal1', 'poly', 'active').",
        },
        "polarity": {
            "type": "string",
            "enum": ["clear", "dark"],
            "default": "clear",
            "description": "Foreground convention. 'clear' = features are 1.",
        },
        "pixel_size_nm": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Physical size of one mask pixel in the raster.",
        },
        "rules": {
            "type": "object",
            "additionalProperties": False,
            "minProperties": 1,
            "properties": {
                "min_width_nm": {"type": "number", "exclusiveMinimum": 0},
                "min_spacing_nm": {"type": "number", "exclusiveMinimum": 0},
                "min_curvature_radius_nm": {"type": "number", "exclusiveMinimum": 0},
                "min_feature_area_nm2": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "source": {
            "type": "object",
            "description": "Provenance — where this deck came from.",
            "additionalProperties": True,
            "properties": {
                "vendor": {"type": "string"},
                "translated_from": {
                    "type": "string",
                    "enum": ["svrf", "icv", "manual", "other"],
                },
                "reference": {"type": "string"},
            },
        },
        "notes": {"type": "string"},
    },
}


@dataclass(frozen=True)
class RuleDeck:
    """An MRC rule deck after validation."""

    schema_version: str
    process_node: str
    layer: str
    polarity: str
    pixel_size_nm: float | None
    rules: dict[str, float]
    source: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    path: Path | None = None

    def kwargs_manhattan(self) -> dict[str, float]:
        """Kwargs to forward to ``check_mrc()``.

        Only includes rules actually present in the deck plus the pixel
        size (so downstream code doesn't have to special-case partial
        decks).
        """
        out: dict[str, float] = {}
        if "min_width_nm" in self.rules:
            out["min_width_nm"] = self.rules["min_width_nm"]
        if "min_spacing_nm" in self.rules:
            out["min_spacing_nm"] = self.rules["min_spacing_nm"]
        if self.pixel_size_nm is not None:
            out["pixel_size_nm"] = self.pixel_size_nm
        return out

    def kwargs_curvilinear(self) -> dict[str, float]:
        """Kwargs to forward to ``check_curvilinear_mrc()``."""
        out: dict[str, float] = {}
        if "min_curvature_radius_nm" in self.rules:
            out["min_curvature_radius_nm"] = self.rules["min_curvature_radius_nm"]
        if "min_feature_area_nm2" in self.rules:
            out["min_feature_area_nm2"] = self.rules["min_feature_area_nm2"]
        if self.pixel_size_nm is not None:
            out["pixel_size_nm"] = self.pixel_size_nm
        return out


def load_rule_deck(path: str | Path) -> RuleDeck:
    """Load and validate an MRC rule-deck file.

    JSON is the canonical format (``.json``). TOML is supported via
    ``.toml`` for ergonomics — semantically identical. The file is
    validated against ``RULE_DECK_SCHEMA`` before instantiation; on
    failure, raises ``ValueError`` with a path-prefixed message.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rule deck not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text())
    elif suffix == ".toml":
        with p.open("rb") as fh:
            data = _load_toml(fh)
    else:
        raise ValueError(f"Unsupported rule-deck format: {suffix!r}. Use .json or .toml.")

    validate_rule_deck(data)

    return RuleDeck(
        schema_version=data["schema_version"],
        process_node=data["process_node"],
        layer=data["layer"],
        polarity=data.get("polarity", "clear"),
        pixel_size_nm=data.get("pixel_size_nm"),
        rules=dict(data["rules"]),
        source=dict(data.get("source", {})),
        notes=data.get("notes"),
        path=p,
    )


def validate_rule_deck(data: dict[str, Any]) -> None:
    """Lightweight in-tree validator for ``RULE_DECK_SCHEMA``.

    We don't pull in jsonschema as a hard dependency just for one
    schema — the surface is small enough that a hand-rolled validator
    keeps installation lean and error messages tailored. If a more
    sophisticated validator is needed downstream (e.g. integrated with
    CI tooling), the schema dict above is the single source of truth.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Rule deck must be a JSON object; got {type(data).__name__}.")

    required = ("schema_version", "process_node", "layer", "rules")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Rule deck is missing required keys: {missing}.")

    if data["schema_version"] != "1":
        raise ValueError(f"Unsupported schema_version {data['schema_version']!r}; expected '1'.")

    for k in ("process_node", "layer"):
        if not isinstance(data[k], str) or not data[k]:
            raise ValueError(f"Rule-deck '{k}' must be a non-empty string.")

    polarity = data.get("polarity", "clear")
    if polarity not in ("clear", "dark"):
        raise ValueError(f"Rule-deck polarity must be 'clear' or 'dark'; got {polarity!r}.")

    if "pixel_size_nm" in data and (
        not isinstance(data["pixel_size_nm"], (int, float)) or data["pixel_size_nm"] <= 0
    ):
        raise ValueError("Rule-deck pixel_size_nm must be a positive number.")

    rules = data["rules"]
    if not isinstance(rules, dict) or not rules:
        raise ValueError("Rule-deck 'rules' must be a non-empty object.")

    allowed_rules = {
        "min_width_nm",
        "min_spacing_nm",
        "min_curvature_radius_nm",
        "min_feature_area_nm2",
    }
    unknown = set(rules) - allowed_rules
    if unknown:
        raise ValueError(
            f"Unknown rule keys: {sorted(unknown)}. Supported: {sorted(allowed_rules)}."
        )
    for k, v in rules.items():
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"Rule '{k}' must be a positive number; got {v!r}.")

    extra = set(data) - {
        "schema_version",
        "process_node",
        "layer",
        "polarity",
        "pixel_size_nm",
        "rules",
        "source",
        "notes",
    }
    if extra:
        raise ValueError(f"Rule deck has unknown top-level keys: {sorted(extra)}.")
