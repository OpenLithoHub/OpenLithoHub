"""SEMI P39 layer-purpose-pair protocol for OASIS/GDS interoperability."""

from __future__ import annotations

from dataclasses import dataclass

from openlithohub.workflow.layer_purpose import LayerPurpose, classify_purpose

SEMI_P39_REGISTRY: dict[str, int] = {
    "drawing": 0,
    "pin": 1,
    "blockage": 2,
    "label": 3,
    "net": 4,
    "boundary": 5,
    "fill": 6,
    "sra Assist": 7,
    "opc Assist": 8,
    "text": 9,
    "custom": 10,
}

_NEXT_CUSTOM_CODE: int = 11


@dataclass(frozen=True)
class LayerPurposePair:
    layer: int
    datatype: int
    purpose_name: str
    is_derived: bool = False


class P39Mapper:
    def __init__(self, custom_codes: dict[str, int] | None = None) -> None:
        self._forward: dict[str, int] = dict(SEMI_P39_REGISTRY)
        self._reverse: dict[int, str] = {v: k for k, v in SEMI_P39_REGISTRY.items()}
        self._next_code: int = _NEXT_CUSTOM_CODE
        if custom_codes:
            for name, code in custom_codes.items():
                self._register_custom(name, code)

    def _register_custom(self, name: str, code: int) -> None:
        canonical = name.strip()
        self._forward[canonical] = code
        self._reverse[code] = canonical
        if code >= self._next_code:
            self._next_code = code + 1

    def _resolve_canonical(self, purpose_name: str) -> str | None:
        classified = classify_purpose(purpose_name)
        if classified is not None:
            key = classified.strip().lower().replace(" ", "").replace("_", "")
            for registered in self._forward:
                norm = registered.strip().lower().replace(" ", "").replace("_", "")
                if norm == key:
                    return registered
        key = purpose_name.strip().lower().replace(" ", "").replace("_", "")
        for registered in self._forward:
            norm = registered.strip().lower().replace(" ", "").replace("_", "")
            if norm == key:
                return registered
        return None

    def to_semi(self, layer_purpose: LayerPurpose) -> LayerPurposePair:
        if layer_purpose.purpose is None:
            return LayerPurposePair(
                layer=layer_purpose.layer,
                datatype=SEMI_P39_REGISTRY["custom"],
                purpose_name="custom",
                is_derived=True,
            )
        canonical = self._resolve_canonical(layer_purpose.purpose)
        if canonical is not None and canonical in self._forward:
            return LayerPurposePair(
                layer=layer_purpose.layer,
                datatype=self._forward[canonical],
                purpose_name=canonical,
            )
        code = self._forward.get("custom", 10)
        return LayerPurposePair(
            layer=layer_purpose.layer,
            datatype=code,
            purpose_name="custom",
            is_derived=True,
        )

    def from_semi(self, layer: int, datatype: int) -> LayerPurposePair:
        purpose_name = self._reverse.get(datatype, "custom")
        is_derived = datatype not in self._reverse
        return LayerPurposePair(
            layer=layer,
            datatype=datatype,
            purpose_name=purpose_name,
            is_derived=is_derived,
        )

    def validate_mapping(self, pairs: list[LayerPurposePair]) -> list[str]:
        warnings: list[str] = []
        seen: dict[tuple[int, int], str] = {}
        valid_codes = set(self._reverse.keys())
        for pair in pairs:
            key = (pair.layer, pair.datatype)
            if key in seen:
                warnings.append(
                    f"Duplicate (layer, datatype)=({pair.layer}, {pair.datatype}) "
                    f"for purposes '{seen[key]}' and '{pair.purpose_name}'"
                )
            else:
                seen[key] = pair.purpose_name
            if pair.datatype not in valid_codes:
                warnings.append(
                    f"Datatype {pair.datatype} on layer {pair.layer} "
                    f"not in SEMI P39 registry for purpose '{pair.purpose_name}'"
                )
        return warnings

    def export_table(self) -> dict[str, int]:
        return dict(sorted(self._forward.items(), key=lambda kv: kv[1]))
