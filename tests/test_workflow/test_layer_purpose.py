"""Tests for the OpenAccess (Si2) / OASIS layer-purpose-pair helpers."""

from __future__ import annotations

import pytest

from openlithohub.workflow.layer_purpose import (
    DATATYPE_TO_OA_PURPOSE,
    OA_PURPOSE_TO_DATATYPE,
    LayerPurpose,
    classify_purpose,
    datatype_for_purpose,
    purpose_for_datatype,
)


class TestClassifyPurpose:
    def test_canonical_names_round_trip(self) -> None:
        for name in OA_PURPOSE_TO_DATATYPE:
            assert classify_purpose(name) == name

    def test_case_insensitive(self) -> None:
        assert classify_purpose("DRAWING") == "drawing"
        assert classify_purpose("Pin") == "pin"
        assert classify_purpose("BLOCKAGE") == "blockage"

    def test_whitespace_and_underscore_stripped(self) -> None:
        assert classify_purpose("  drawing  ") == "drawing"
        assert classify_purpose("dummy_fill") == "fill"
        assert classify_purpose("draw n") == "drawing"

    def test_common_aliases(self) -> None:
        assert classify_purpose("drw") == "drawing"
        assert classify_purpose("polygon") == "drawing"
        assert classify_purpose("obs") == "blockage"
        assert classify_purpose("obstruction") == "blockage"
        assert classify_purpose("prboundary") == "boundary"
        assert classify_purpose("dummy") == "fill"
        assert classify_purpose("text") == "label"
        assert classify_purpose("term") == "pin"

    def test_unknown_returns_none(self) -> None:
        assert classify_purpose("not_a_purpose") is None
        assert classify_purpose("") is None

    def test_non_string_returns_none(self) -> None:
        # Defensive: caller passes an int or None — don't crash, just say
        # "unknown" so the caller can branch on the return value.
        assert classify_purpose(42) is None  # type: ignore[arg-type]
        assert classify_purpose(None) is None  # type: ignore[arg-type]

    def test_oa_typo_preserved(self) -> None:
        # OpenAccess has a typo "cutsom" in its registered purpose names;
        # we accept both the typo and the corrected spelling and resolve
        # them to the same canonical value (the typo, matching OA).
        assert classify_purpose("cutsom") == "cutsom"
        assert classify_purpose("custom") == "cutsom"


class TestDatatypeMapping:
    def test_purpose_to_datatype_known(self) -> None:
        assert datatype_for_purpose("drawing") == 0
        assert datatype_for_purpose("pin") == 2
        assert datatype_for_purpose("blockage") == 5

    def test_purpose_to_datatype_alias(self) -> None:
        assert datatype_for_purpose("drw") == 0
        assert datatype_for_purpose("dummy") == 6

    def test_purpose_to_datatype_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown OpenAccess purpose"):
            datatype_for_purpose("xyz_unknown")

    def test_datatype_to_purpose_known(self) -> None:
        assert purpose_for_datatype(0) == "drawing"
        assert purpose_for_datatype(2) == "pin"
        assert purpose_for_datatype(5) == "blockage"

    def test_datatype_to_purpose_unknown_returns_none(self) -> None:
        # Foundry-defined datatype outside the OA registry — return None,
        # don't raise. Callers fall back to literal datatype handling.
        assert purpose_for_datatype(99) is None

    def test_round_trip_consistency(self) -> None:
        for purpose, datatype in OA_PURPOSE_TO_DATATYPE.items():
            assert DATATYPE_TO_OA_PURPOSE[datatype] == purpose


class TestLayerPurpose:
    def test_from_pair_known_datatype(self) -> None:
        lp = LayerPurpose.from_pair(layer=10, datatype=2)
        assert lp.layer == 10
        assert lp.datatype == 2
        assert lp.purpose == "pin"

    def test_from_pair_unknown_datatype_keeps_purpose_none(self) -> None:
        lp = LayerPurpose.from_pair(layer=10, datatype=99)
        assert lp.purpose is None

    def test_from_name_resolves_alias(self) -> None:
        lp = LayerPurpose.from_name(layer=10, purpose_name="drw")
        assert lp.datatype == 0
        assert lp.purpose == "drawing"

    def test_from_name_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown OpenAccess purpose"):
            LayerPurpose.from_name(layer=10, purpose_name="bogus")

    def test_frozen(self) -> None:
        lp = LayerPurpose.from_pair(10, 0)
        with pytest.raises((AttributeError, Exception)):
            lp.layer = 20  # type: ignore[misc]


def test_reexport_from_workflow_namespace() -> None:
    import openlithohub.workflow as wf

    assert wf.LayerPurpose is LayerPurpose
    assert wf.classify_purpose is classify_purpose
    assert wf.OA_PURPOSE_TO_DATATYPE is OA_PURPOSE_TO_DATATYPE
