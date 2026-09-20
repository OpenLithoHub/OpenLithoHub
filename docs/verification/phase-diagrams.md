# Phase Diagrams — the three frozen layers

P-054 certifies three **distinct** stratified phase diagrams of a declared
finite Hopkins aerial model.  They answer different questions and are
different mathematical objects; the repository encodes that distinction in
the type system (`openlithohub.verify.phase_diagram`).

## 1. Critical-set layer — `PhaseLayer.CRITICAL_SET`

Where does the set of spatial critical points itself bifurcate?

- `GENERIC_FOLD` — a pair of critical points is born/annihilated;
- `REFLECTION_PITCHFORK` — a symmetry-forced pitchfork, visible in the
  ownership layer;
- `OWNERSHIP_INVISIBLE_PITCHFORK` — a pitchfork that does **not** change
  the lower-owner envelope (the owner is preserved; replay tests enforce
  this).

## 2. Ownership layer — `PhaseLayer.OWNERSHIP`

Where do the lower/upper *owners* of neighbouring critical values change?

- `TRANSVERSE_OWNERSHIP_CROSSING` — two owner branches cross transversally
  (the `z3` envelope kink: owner `C → E`);
- `TERMINAL_FOLD_CLIFF` — a fold cliff terminates an ownership branch.

An ownership event is **not** a critical-set bifurcation and **not** a
contour-topology change.

## 3. Fixed-target topology layer — `PhaseLayer.TARGET_TOPOLOGY`

Where does the topology of a fixed-threshold contour change (governed by
the target discriminant)?

- `TARGET_MAX_EVENT`, `TARGET_SADDLE_EVENT` — component births/deaths via
  extrema and saddles.  On the frozen `p054-arf37` fixture the target
  component count follows the frozen sequence `1 → 3 → 5 → 4`.

## Query API (frozen proof replay)

```python
from openlithohub.verify.phase_diagram import load_frozen_phase_diagram, PhaseLayer

pd = load_frozen_phase_diagram("p054-arf37")
pd.events_by_layer(PhaseLayer.OWNERSHIP)  # typed ownership certificates
pd.event("z5")  # ownership-invisible pitchfork
pd.verify_manifest()  # structural replay (offline)
```

Numeric focus intervals and chamber boundaries live in the frozen external
artifact (see `proof_artifacts/p054/external_artifacts.json`).  Queries
that need them fail closed with `PhaseArtifactNotAvailableError` until the
artifact is fetched — offline replay verifies **structure**, never
fabricated numerics.
