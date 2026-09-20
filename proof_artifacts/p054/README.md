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
| `frozen-basis.json` | immutable release anchor: content hashes of the profile files + external artifact identity |
| `../registry.json` | **single source of truth** for external artifact identity (DOI provenance + file-level download URL + SHA-256) |
| `*.fetch-report.json` | canonical fetch evidence: effective URL + artifact SHA/bytes, written after validation (PR-5E5) |
| `../verification-status.json` | machine-readable status consumed by CI (`scripts/check_verification_status.py`) |

## Activation evidence

The repository replay state is `IMPORTED_CERTIFICATE_VERIFIED`: the
committed canonical receipt (P054.replay-receipt.v2) was produced from a
fresh fetch of the exact published Zenodo bytes and validated by the
strict Contract-B shard and the whole-receipt equality gate.  See
`replay-receipt.json` and `scripts/check_verification_status.py`.

## Replay

- **Offline (PR fast gate):** manifest/fixture/catalog schema and
  stratification-structure tests — `pytest tests/test_verify -q`.
- **Frozen-certificate consistency replay** (Contract B semantics — this
  verifies the frozen artifact's declarations against the catalogs; it is
  **not** source-native recomputation):

  ```bash
  python scripts/fetch_proof_artifacts.py --profile p054-arf37 \
      --fetch-report-out proof_artifacts/p054/external/p054-arf37-frozen-artifact.fetch-report.json
  python scripts/fetch_proof_artifacts.py --profile p054-arf37 --verify-only
  B04_PROOF_REPLAY_STRICT=1 \
      pytest -q tests/test_verify/test_p054_full_replay.py -m proof_artifact_required
  ```

  Missing artifacts or hash/model-identity mismatches are **hard failures**
  in the proof-replay workflow — never silent skips.  The replay engine
  (`src/openlithohub/verify/phase_diagram.py`) is part of the frozen
  release contract; its hash is bound into the replay receipt.
