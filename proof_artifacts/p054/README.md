# P-054 — Certified Stratified Phase Diagrams for Finite Hopkins Lithography

**This directory is the canonical proof entry point for P-054.** The B04
Increment 11–29 artifacts one level up are the historical research trail;
P-054 is the frozen, citable result and the stable public surface.

- **Paper:** *Certified Stratified Phase Diagrams for Finite Hopkins
  Lithography*, Lin Tao, Zenodo,
  DOI [10.5281/zenodo.22843141](https://doi.org/10.5281/zenodo.22843141).
- **Frozen implementation basis:** `348fa5d86d5355465af98e2c4ce3deac60081a4c`.
  The current repository HEAD does **not** change the paper basis.
- **Canonical model:** `openlithohub.source_native_full.hopkins.discrete.p054-arf37`
  over the frozen `p054-arf37` fixture (37-source ArF, NA 1.35, σ 0.7,
  72×72 grid, 8 nm pixels, 49-bin pupil support, periodic finite tile).

## What P-054 certifies

Three **distinct** stratified phase diagrams of the declared finite
aerial model (see `docs/verification/phase-diagrams.md`):

1. **critical-set phase diagram** — where the spatial critical-point set
   undergoes fold / pitchfork bifurcations;
2. **critical-value ownership phase diagram** — where the lower/upper
   owners of neighbouring critical values change (fold cliffs, transverse
   crossings, reflection pitchforks);
3. **fixed-target topology phase diagram** — where a fixed-threshold
   contour's topology changes (governed by the target discriminant).

Replay/query API: `openlithohub.verify.phase_diagram.load_frozen_phase_diagram("p054-arf37")`.

## What P-054 does **not** prove

See `manifest.json → does_not_prove`: no continuous Hopkins equivalence,
no continuous source-plane integral certification, no foundry/process
qualification, no wafer-accuracy claim, and nothing about layouts outside
the declared frozen fixture.

## Files

| file | role |
|---|---|
| `manifest.json` | paper identity, frozen commit, fixture constants, declared proof levels, `does_not_prove` |
| `fixture.json` | the frozen fixture profile with explicit honesty flags |
| `event_catalog.json` | the three-layer event stratification (z1–z5, zV, zH, zP) |
| `chamber_catalog.json` | event-free chambers; target component sequence `1 → 3 → 5 → 4` |
| `external_artifacts.json` | registry of large frozen artifacts (SHA-256 + DOI, never copied into git) |
| `../verification-status.json` | machine-readable status consumed by CI (`scripts/check_verification_status.py`) |

## Replay

- **Offline (PR fast gate):** manifest/fixture/catalog schema and
  stratification-structure tests — `pytest tests/test_verify -q`.
- **Full:** requires the external artifact
  (`scripts/fetch_proof_artifacts.py` downloads and SHA-256-verifies it).
  Missing artifacts or hash/model-identity mismatches are **hard failures**
  in the proof-replay workflow — never silent skips.
