"""Tests for SEMI P39 layer-purpose-pair plumbing."""

from __future__ import annotations

from openlithohub.workflow.layer_purpose import LayerPurpose
from openlithohub.workflow.semi_p39 import (
    SEMI_P39_REGISTRY,
    LayerPurposePair,
    P39Mapper,
)


class TestRegistry:
    def test_registry_has_expected_entries(self) -> None:
        expected = {
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
        for name, code in expected.items():
            assert SEMI_P39_REGISTRY[name] == code, f"{name} should map to {code}"

    def test_registry_codes_are_unique(self) -> None:
        codes = list(SEMI_P39_REGISTRY.values())
        assert len(codes) == len(set(codes))


class TestP39MapperToSemi:
    def test_maps_known_purposes(self) -> None:
        mapper = P39Mapper()
        for purpose_name, expected_dt in [
            ("drawing", 0),
            ("pin", 1),
            ("blockage", 2),
            ("net", 4),
            ("boundary", 5),
            ("fill", 6),
        ]:
            lp = LayerPurpose(layer=1, datatype=expected_dt, purpose=purpose_name)
            pair = mapper.to_semi(lp)
            assert pair.datatype == expected_dt, f"{purpose_name} -> {expected_dt}"
            assert pair.layer == 1

    def test_unknown_purpose_gets_custom(self) -> None:
        mapper = P39Mapper()
        lp = LayerPurpose(layer=5, datatype=99, purpose=None)
        pair = mapper.to_semi(lp)
        assert pair.datatype == SEMI_P39_REGISTRY["custom"]
        assert pair.purpose_name == "custom"
        assert pair.is_derived is True


class TestP39MapperFromSemi:
    def test_reverse_maps_correctly(self) -> None:
        mapper = P39Mapper()
        for purpose_name, code in SEMI_P39_REGISTRY.items():
            if purpose_name == "custom":
                continue
            pair = mapper.from_semi(layer=2, datatype=code)
            assert pair.purpose_name == purpose_name
            assert pair.layer == 2
            assert pair.is_derived is False

    def test_unknown_datatype_gets_custom(self) -> None:
        mapper = P39Mapper()
        pair = mapper.from_semi(layer=3, datatype=999)
        assert pair.purpose_name == "custom"
        assert pair.is_derived is True


class TestRoundTrip:
    def test_to_semi_then_from_semi_round_trips(self) -> None:
        mapper = P39Mapper()
        for purpose_name in ("drawing", "pin", "blockage", "label", "net", "fill"):
            lp = LayerPurpose(
                layer=7,
                datatype=SEMI_P39_REGISTRY.get(purpose_name, 10),
                purpose=purpose_name,
            )
            pair = mapper.to_semi(lp)
            back = mapper.from_semi(pair.layer, pair.datatype)
            assert back.purpose_name == pair.purpose_name
            assert back.layer == pair.layer
            assert back.datatype == pair.datatype


class TestValidateMapping:
    def test_catches_invalid_datatype(self) -> None:
        mapper = P39Mapper()
        pairs = [
            LayerPurposePair(layer=1, datatype=0, purpose_name="drawing"),
            LayerPurposePair(layer=1, datatype=999, purpose_name="weird"),
        ]
        warnings = mapper.validate_mapping(pairs)
        assert any("999" in w for w in warnings)

    def test_no_warnings_for_valid_pairs(self) -> None:
        mapper = P39Mapper()
        pairs = [
            LayerPurposePair(layer=1, datatype=0, purpose_name="drawing"),
            LayerPurposePair(layer=1, datatype=1, purpose_name="pin"),
        ]
        warnings = mapper.validate_mapping(pairs)
        assert len(warnings) == 0

    def test_catches_duplicate_layer_datatype(self) -> None:
        mapper = P39Mapper()
        pairs = [
            LayerPurposePair(layer=1, datatype=0, purpose_name="drawing"),
            LayerPurposePair(layer=1, datatype=0, purpose_name="pin"),
        ]
        warnings = mapper.validate_mapping(pairs)
        assert any("Duplicate" in w for w in warnings)


class TestExportTable:
    def test_returns_complete_mapping(self) -> None:
        mapper = P39Mapper()
        table = mapper.export_table()
        assert len(table) == len(SEMI_P39_REGISTRY)
        for name, code in SEMI_P39_REGISTRY.items():
            assert table[name] == code


class TestDeterministic:
    def test_same_input_same_output(self) -> None:
        mapper = P39Mapper()
        lp = LayerPurpose(layer=1, datatype=0, purpose="drawing")
        a = mapper.to_semi(lp)
        b = mapper.to_semi(lp)
        assert a == b
