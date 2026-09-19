# P-054 Frozen Profile — `p054-arf37`

The canonical frozen fixture behind the P-054 phase-diagram certificate.

## Declared model

```text
scope            = FINITE_DECLARED_AERIAL_MODEL
boundary         = PERIODIC_FINITE_TILE
source           = FROZEN_37_SOURCE (37 samples)
wavelength_nm    = 193
na               = 1.35
sigma            = 0.7
grid             = 72 x 72
pixel_size_nm    = 8
pupil_support    = 49 bins
```

## Where the truth lives

- Structure (layers, events, owners, component sequence): this repository,
  `proof_artifacts/p054/`, replayed offline by
  `tests/test_verify/test_p054_phase_diagram.py`.
- Numeric content (exact focus intervals, chamber boundaries, coefficient
  tensor, content hashes): the frozen Zenodo artifact,
  DOI [10.5281/zenodo.22843141](https://doi.org/10.5281/zenodo.22843141),
  referenced by `proof_artifacts/registry.json` and SHA-256-verified before
  any replay (`scripts/fetch_proof_artifacts.py`).

## Replay

```bash
# offline structural replay (CI fast gate)
pytest -q tests/test_verify

# frozen-certificate consistency replay (Contract B; hard-fails on
# missing/mismatched artifacts; NOT source-native recomputation)
python scripts/fetch_proof_artifacts.py --profile p054-arf37
python scripts/fetch_proof_artifacts.py --profile p054-arf37 --verify-only
B04_PROOF_REPLAY_STRICT=1 \
  pytest -q tests/test_verify/test_p054_full_replay.py -m proof_artifact_required
```

## Limitations (explicit)

- `foundry_calibrated = false`
- `continuous_source_integral_certified = false`
- `wafer_process_qualified = false`
- `continuous_hopkins_equivalent = false`

The profile certifies the **declared finite model only**.  The frozen
implementation basis is `348fa5d86d5355465af98e2c4ce3deac60081a4c`; the
moving repository HEAD never rewrites it.
