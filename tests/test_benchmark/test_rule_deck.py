"""Tests for `openlithohub.benchmark.compliance.rule_deck`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openlithohub.benchmark.compliance import (
    RULE_DECK_SCHEMA,
    RuleDeck,
    load_rule_deck,
    validate_rule_deck,
)

_MIN_DECK = {
    "schema_version": "1",
    "process_node": "freepdk45",
    "layer": "metal1",
    "rules": {"min_width_nm": 65.0, "min_spacing_nm": 65.0},
}


def test_load_minimal_json(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    p.write_text(json.dumps(_MIN_DECK))
    deck = load_rule_deck(p)
    assert isinstance(deck, RuleDeck)
    assert deck.process_node == "freepdk45"
    assert deck.layer == "metal1"
    assert deck.polarity == "clear"
    assert deck.rules == {"min_width_nm": 65.0, "min_spacing_nm": 65.0}
    assert deck.path == p


def test_kwargs_manhattan_only_includes_present_rules(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    deck_data = {**_MIN_DECK, "rules": {"min_width_nm": 65.0}}
    p.write_text(json.dumps(deck_data))
    deck = load_rule_deck(p)
    kwargs = deck.kwargs_manhattan()
    assert kwargs == {"min_width_nm": 65.0}


def test_kwargs_curvilinear(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    deck_data = {
        **_MIN_DECK,
        "pixel_size_nm": 0.5,
        "rules": {
            "min_width_nm": 65.0,
            "min_curvature_radius_nm": 25.0,
            "min_feature_area_nm2": 4225.0,
        },
    }
    p.write_text(json.dumps(deck_data))
    deck = load_rule_deck(p)
    assert deck.kwargs_curvilinear() == {
        "min_curvature_radius_nm": 25.0,
        "min_feature_area_nm2": 4225.0,
        "pixel_size_nm": 0.5,
    }


def test_load_toml(tmp_path: Path) -> None:
    p = tmp_path / "deck.toml"
    p.write_text(
        'schema_version = "1"\n'
        'process_node = "asap7"\n'
        'layer = "poly"\n'
        "[rules]\nmin_width_nm = 18.0\nmin_spacing_nm = 18.0\n"
    )
    deck = load_rule_deck(p)
    assert deck.process_node == "asap7"
    assert deck.rules["min_width_nm"] == 18.0


def test_missing_required_keys_raise(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    p.write_text(json.dumps({"schema_version": "1"}))
    with pytest.raises(ValueError, match="missing required keys"):
        load_rule_deck(p)


def test_unknown_rule_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    bad = {**_MIN_DECK, "rules": {"min_width_nm": 65.0, "min_height_nm": 9.0}}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Unknown rule keys"):
        load_rule_deck(p)


def test_unknown_top_level_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    bad = {**_MIN_DECK, "scanner": "ASML 2050i"}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_rule_deck(p)


def test_negative_rule_value_rejected(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    bad = {**_MIN_DECK, "rules": {"min_width_nm": -1.0}}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="must be a positive number"):
        load_rule_deck(p)


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    bad = {**_MIN_DECK, "schema_version": "2"}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema_version"):
        load_rule_deck(p)


def test_bad_polarity_rejected(tmp_path: Path) -> None:
    p = tmp_path / "deck.json"
    bad = {**_MIN_DECK, "polarity": "rainbow"}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="polarity"):
        load_rule_deck(p)


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    p = tmp_path / "deck.yaml"
    p.write_text("rules: {}")
    with pytest.raises(ValueError, match="Unsupported rule-deck format"):
        load_rule_deck(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rule_deck(tmp_path / "nope.json")


def test_validate_rule_deck_directly() -> None:
    validate_rule_deck(_MIN_DECK)  # should not raise


def test_schema_constant_is_self_consistent() -> None:
    """The schema dict is the source of truth for the validator's allowed
    keys; spot-check that they match.
    """
    schema_props = set(RULE_DECK_SCHEMA["properties"]["rules"]["properties"])
    assert schema_props == {
        "min_width_nm",
        "min_spacing_nm",
        "min_curvature_radius_nm",
        "min_feature_area_nm2",
    }


def test_shipped_example_deck_loads() -> None:
    """The shipped FreePDK45 example deck should load and produce sensible kwargs."""
    pkg = Path(__file__).resolve().parent.parent.parent
    deck_path = pkg / "src/openlithohub/benchmark/compliance/rule_decks/freepdk45_metal1.json"
    if not deck_path.exists():
        pytest.skip(f"Example deck not present in source tree: {deck_path}")
    deck = load_rule_deck(deck_path)
    assert deck.process_node == "freepdk45"
    assert deck.kwargs_manhattan()["min_width_nm"] == 65.0


def test_pipes_into_check_mrc(tmp_path: Path) -> None:
    """A loaded deck should feed cleanly into check_mrc()."""
    import torch

    from openlithohub.benchmark.compliance import check_mrc

    p = tmp_path / "deck.json"
    p.write_text(
        json.dumps(
            {
                **_MIN_DECK,
                "pixel_size_nm": 1.0,
                "rules": {"min_width_nm": 4.0, "min_spacing_nm": 4.0},
            }
        )
    )
    deck = load_rule_deck(p)

    # Wide block — passes both rules.
    mask = torch.zeros(64, 64)
    mask[16:48, 16:48] = 1.0
    result = check_mrc(mask, **deck.kwargs_manhattan())
    assert result.passed
