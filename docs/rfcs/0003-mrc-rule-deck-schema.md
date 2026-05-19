# RFC 0003 — Standard MRC Rule-Deck Schema (v1)

## Status

Accepted (May 2026). Implemented in
`openlithohub.benchmark.compliance.rule_deck`.

## Motivation

Different fabs and process nodes ship Mask Rule Check (MRC) decks in
incompatible formats — Mentor Calibre SVRF, Synopsys ICV runset,
vendor-specific YAML, ad-hoc Excel. None of them are easy to read or
exchange across teams. When OpenLithoHub scores a mask, every contributor
needs to know exactly which rules were applied and at which thresholds;
re-translating SVRF for every benchmark run is brittle and audit-hostile.

This RFC defines a **single JSON / TOML rule-deck format** that:

1. Captures every parameter the OpenLithoHub MRC checkers consume
   (`min_width_nm`, `min_spacing_nm`, `min_curvature_radius_nm`,
   `min_feature_area_nm2`).
2. Carries enough metadata (process node, layer, mask polarity,
   pixel size, provenance) to be reproducible and auditable.
3. Validates against a published JSON Schema (`RULE_DECK_SCHEMA`)
   before being trusted.

It is intentionally **not** a full SVRF replacement. It covers what the
optical / OPC side needs to score a mask. DRC-on-the-design side
(transistor sizing, well rules, etc.) stays in the existing process-node
configs.

## Schema (v1)

```json
{
  "schema_version": "1",
  "process_node": "freepdk45",
  "layer": "metal1",
  "polarity": "clear",
  "pixel_size_nm": 1.0,
  "rules": {
    "min_width_nm": 65.0,
    "min_spacing_nm": 65.0,
    "min_curvature_radius_nm": 25.0,
    "min_feature_area_nm2": 4225.0
  },
  "source": {
    "vendor": "NCSU FreePDK45",
    "translated_from": "manual",
    "reference": "https://eda.ncsu.edu/freepdk/freepdk45/"
  },
  "notes": "Optional free-text."
}
```

| Field | Required | Notes |
|---|---|---|
| `schema_version` | yes | Currently `"1"`. Bumped on breaking changes. |
| `process_node` | yes | Tech-node tag (`asap7`, `freepdk45`, `n5`, …). |
| `layer` | yes | Mask layer tag (`metal1`, `poly`, …). |
| `polarity` | no (`"clear"`) | `"clear"` = features are 1; `"dark"` = features are 0. |
| `pixel_size_nm` | no | Raster pixel size; required if any rule uses physical units. |
| `rules.min_width_nm` | no | Minimum allowed feature width. |
| `rules.min_spacing_nm` | no | Minimum allowed gap between features. |
| `rules.min_curvature_radius_nm` | no | Min local radius of curvature (curvilinear). |
| `rules.min_feature_area_nm2` | no | Minimum connected-component area (curvilinear). |
| `source` | no | Provenance — vendor, original tool, citation URL. |
| `notes` | no | Free-text. |

The validator rejects unknown top-level keys and unknown rule keys, so
typos surface immediately rather than silently dropping rules.

## SVRF → JSON translation

The Calibre SVRF lines below are the most commonly-encountered MRC
constructs. Mappings to v1 fields:

| SVRF | v1 JSON |
|---|---|
| `EXTERNAL "metal1" < 65 nm` | `"rules": { "min_spacing_nm": 65.0 }` |
| `INTERNAL "metal1" < 65 nm` | `"rules": { "min_width_nm": 65.0 }` |
| `WIDTH "metal1" < 65 nm` | `"rules": { "min_width_nm": 65.0 }` |
| `SPACE "metal1" < 65 nm` | `"rules": { "min_spacing_nm": 65.0 }` |
| `AREA "metal1" < 4225 nm2` | `"rules": { "min_feature_area_nm2": 4225.0 }` |
| `RADIUS "metal1" < 25 nm CONVEX EDGE` | `"rules": { "min_curvature_radius_nm": 25.0 }` |
| Layer derivation `LAYER metal1 ...` | `"layer": "metal1"` |
| Tech-node header comment | `"process_node": "..."` |

Rules SVRF expresses but v1 **does not yet** capture (use `notes` for
now, add to v2 if needed): enclosure, extension, end-of-line spacing,
notch, jog, off-grid alignment, density windows, antenna ratios.

## Synopsys ICV → JSON

ICV uses `width(...)`, `space(...)`, `area(...)`, and `radius(...)`
function-style operations on a layer expression; the parameters map
directly onto the same v1 fields. The `density` family does not map
to v1.

## Loading

```python
from openlithohub.benchmark.compliance import load_rule_deck, check_mrc

deck = load_rule_deck("rules/freepdk45_metal1.json")
result = check_mrc(mask, **deck.kwargs_manhattan())
```

`load_rule_deck` validates the file against the schema before
returning a `RuleDeck` dataclass. `kwargs_manhattan()` and
`kwargs_curvilinear()` produce kwargs for the existing `check_mrc` /
`check_curvilinear_mrc` functions — only rules actually present in the
deck are forwarded, so partial decks are safe.

## Versioning

Backwards-incompatible changes (renamed keys, narrowed enums) bump
`schema_version` to `"2"`; the loader rejects unknown major versions
explicitly. Backwards-compatible additions (new optional rules) do
**not** change `schema_version`; older decks continue to load.

## Open questions

- Should we accept multiple layers per file (an array of decks)?
  Currently no — one file = one (process_node, layer) pair, which keeps
  diffing cleaner. Multi-layer bundles can be a follow-up if demand
  emerges.
- Should `pixel_size_nm` be required when any rule is in nm? Currently
  optional — defaults to whatever the checker is called with.
