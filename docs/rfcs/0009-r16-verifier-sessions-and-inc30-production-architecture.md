# RFC 0009 — R16 Verifier Sessions + Subdivide Routing, and the Inc30 Production Architecture

- Status: Proposed (design; implementation gated on the Inc29 verdict)
- Related: RFC 0007 (QDM verification), RFC 0008 (streaming core/halo),
  philosophy brief §5–§19 ("R5–R15"), B04 Increments 28–29
  (scale-first work avoidance, real-density external validity)

This RFC compresses two work streams into one commit-ready design:

1. **R16 — mandatory interface repairs** in the streaming verification
   pipeline, which must land *before* any Inc30 production integration:
   verifier-owned reduction semantics and true `subdivide` routing.
2. **Inc30 — the production architecture** that turns the R5–R15
   requirements into repo mechanisms rather than benchmark-side
   conventions, with a verdict-branch map keyed to the pending Inc29
   empty-screening verdict.

---

## Part I — R16 mandatory repairs

### R16.0 The two audited defects (verified at HEAD `d9d0f8a`)

**Defect A — plugin-owned reduction never runs.**
`VerificationPlugin` (streaming/verification.py) declares
`reduce(results) -> Any` and `finalize() -> Any`, but `run_streaming()`
unconditionally creates one shared `StreamingVerificationReducer()`,
feeds every verifier's `TileVerificationResult` into it with
`reducer.add(verdict)`, and calls `reducer.finalize()`. Consequences:

- the protocol surface `reduce`/`finalize` is dead code;
- the shared reducer is *latest-wins per `tile_id`*, so two verifiers
  inspecting the same tile silently overwrite each other — a verifier
  that says FAIL can be erased by a later verifier that says PASS.

**Defect B — `action="subdivide"` is silently mis-routed.**
`RefinementRequest` supports `action ∈ {"increase_halo", "subdivide"}`
with `subdivision: int = 2`, and `core_halo.subdivide_request()` is
implemented and tested — but `run_streaming()` re-plans refinements
exclusively through `_grown_request()`, which adds halo and keeps the
core box. A verifier that asks to subdivide an inconclusive tile gets
"the same core with more halo" instead. Any future sensitivity-core /
adaptive-subdivision mechanism (the expected Inc30 WEAK-verdict branch)
would be built on a broken primitive.

Both violate the same principle: *the interface must not promise
semantics the pipeline does not implement.* The repair is mandatory
before Inc30; the alternative of leaving subdivide rejected is allowed
only as an explicitly versioned interim.

### R16.1 Design: one verifier → one session, verifier-owned reduction

Principle: **the pipeline schedules and judges completeness; the
verifier owns reduction semantics.** The pipeline must never merge two
verifiers' per-tile results through a shared accumulator.

New dataclass in `streaming/verification.py`:

```python
class VerifierSession:
    """Per-verifier result store + reduction ownership.

    Latest result per tile_id wins *within* one verifier (refinement
    re-runs are re-verifications of the same core). Different verifiers
    never share this store.
    """

    def __init__(self, plugin: VerificationPlugin) -> None:
        self.plugin = plugin
        self._results: dict[str, TileVerificationResult] = {}
        self._superseded: set[str] = set()   # R16.2: subdivided parents

    def record(self, result: TileVerificationResult) -> None:
        self._results[result.tile_id] = result

    def supersede(self, parent_tile_id: str) -> None:
        self._superseded.add(parent_tile_id)

    def results(self) -> list[TileVerificationResult]:
        return [r for t, r in self._results.items() if t not in self._superseded]

    def reduce(self):
        """Delegate to the plugin; fall back to the stock reducer."""
        out = self.plugin.reduce(self.results())
        if isinstance(out, StreamingVerificationReducer):
            return out.finalize()
        return out  # plugin-owned object; pipeline folds it via finalize()

    def finalize(self):
        return self.plugin.finalize()
```

Pipeline changes (`streaming/pipeline.py`):

```python
sessions = [VerifierSession(v) for v in verifier_list]
...
for session, verifier in zip(sessions, verifier_list):
    verdict = verifier.verify_tile(tctx)
    session.record(verdict)
...
if verifier_list:
    report.verification = fold_global(sessions)
```

`fold_global(sessions)` (new, exported) applies the unchanged
PASS/FAIL/INCONCLUSIVE firewall *across* verifier outcomes:

- any verifier-level FAIL → global FAIL;
- no FAIL but any INCONCLUSIVE / PARTIAL_COVER / tolerance breach →
  global INCONCLUSIVE;
- otherwise PASS;
- `error_budget` composition stays additive over the per-verifier
  budgets (`E_process + E_spatial + E_halo + E_raster_bridge + E_other`),
  never over raw tile bounds of different verifiers.

Compatibility: `StreamingVerificationReducer` remains the default
reduction implementation (the dummy plugin's `reduce` already returns
one). `GlobalVerificationResult`, `VerificationResult`, and the
`verify_layout()` signature do not change. The only observable behavior
change is the removal of cross-verifier result clobbering — a bug fix,
covered by a new adversarial test (two verifiers, same tile, one FAIL
one PASS → global FAIL).

### R16.2 Design: route `subdivide` truthfully

Replace the tail of the per-tile `while True` loop with an explicit
work stack:

```python
pending: list[TileRequest] = [base]
while pending:
    current = pending.pop()
    ... screen / read / forward / verify (unchanged) ...
    if refinement is None:
        commit; continue
    refinements_left -= 1
    if refinement.action == "increase_halo":
        pending.append(_grown_request(current, refinement, source.shape))
    elif refinement.action == "subdivide":
        session.supersede_parent(current.tile_id)      # R16.1
        accounting.record_subdivision(current, ...)    # R16.3
        pending.extend(subdivide_request(current, refinement.subdivision))
```

Semantics that must be made explicit (and tested):

1. **Leaf identity.** `subdivide_request()` currently reuses the parent
   `tile_id`. Leaves get structured ids `"{parent}#q{k}"` (row-major k)
   so provenance, session keys, and certificate refs stay unambiguous.
   A parent id must never collide with a leaf id (`#` reserved in
   planner-generated ids).
2. **Completeness.** A subdivided parent is *superseded*, not
   inconclusive: the session removes the parent's unresolved result from
   the reduction once all leaves have concluded. Global PASS is then
   decided on leaves only. If any leaf itself ends inconclusive (or its
   own refinement budget runs out), the parent region remains
   unresolved → INCONCLUSIVE. Unresolved work is never promoted to PASS
   (philosophy §7).
3. **Termination.** `max_requeues` counts refinement *events* (both
   actions), so `subdivide` cannot loop unboundedly. A leaf inherits no
   extra budget.
4. **Halo.** Leaves inherit the parent's halo (`subdivide_request`
   already does this via `HaloSpec.uniform(max(...))`); verifiers that
   need more context per leaf must issue their own `increase_halo` for
   that leaf — the pipeline does not silently add halo to subdivision.

Interim alternative (acceptable for exactly one release): raise
`NotImplementedError("pipeline does not route subdivide yet; request "
"increase_halo or extend the pipeline")` when
`refinement.action == "subdivide"`. Fail-closed beats fake support. The
R16 patch implements the full routing, so this fallback is documentation
only.

### R16.3 Work accounting under subdivision

`WorkAccounting` uniqueness must become **region-keyed**, not
tile-id-keyed, or every subdivision would inflate `active_pixels` by the
parent area again. Concretely:

- `record_active_core(tile_id, core_bbox)` keys uniqueness on the
  *canonical core rectangle* (`(x0, y0, x1, y1)`), not the string id.
  Refinement re-runs and leaf re-verifications of the same region charge
  `reused_work_units` only.
- `record_subdivision(parent, leaves)`: `reused_work_units += parent
  core area` (the parent's verification work is discarded-but-spent),
  active area unchanged.
- A leaf later proven empty by the certified screen moves its own area
  from `active_pixels` to `screened_out_pixels` (new
  `record_screened_leaf(leaf)`), keeping `accounted_pct == 100`
  invariant intact.
- `summary()` gains `n_subdivisions` and
  `subdivided_reused_work_units`.

All Inc28 numbers remain stable for non-subdividing runs (the Inc28/29
benchmarks never emit `subdivide`), so the Inc28/29 returned evidence
stays reproducible bit-for-bit at the JSON level.

### R16.4 Patch plan and acceptance

Files:

- `src/openlithohub/streaming/verification.py`: `VerifierSession`,
  `fold_global`, protocol docstring stating ownership.
- `src/openlithohub/streaming/pipeline.py`: session wiring, work stack,
  subdivide branch, structured leaf ids.
- `src/openlithohub/streaming/core_halo.py`: `subdivide_request` gains
  the leaf-id parameter (default off → R16 passes explicit ids).
- `src/openlithohub/streaming/work_accounting.py`: region-keyed
  uniqueness + subdivision counters.
- tests: `tests/test_streaming/test_verifier_sessions_r16.py`,
  `tests/test_streaming/test_subdivide_routing_r16.py`,
  `tests/test_streaming/test_work_accounting_subdivision_r16.py`.

Acceptance criteria (all must hold):

1. Two verifiers, same tile, FAIL vs PASS → global FAIL (bug-fixed).
2. Plugin returning a custom object from `reduce()` reaches
   `report.verification` unchanged when it provides `finalize()`.
3. Inconclusive tile + `subdivide(subdivision=2)` → exactly 4 leaf
   verifications, parent superseded, `n_tiles` counts 4, not 5.
4. Leaf screened empty → `accounted_pct == 100`, `avoided_pct` increases
   by exactly the leaf fraction.
5. `max_requeues` bounds total refinement events across both actions.
6. Inc28/29 benchmark JSONs byte-identical on a non-subdividing rerun.
7. ruff + `mypy --strict` on `streaming/` stay clean; full
   `tests/test_streaming` + `tests/test_verify` suites pass.

---

## Part II — Inc30 production architecture (R5–R15 compressed)

Inc30's goal is to stop treating philosophy §5–§15 as benchmark-side
conventions and make them repo mechanisms. Each item below states:
requirement → mechanism (exists / to build) → commit-ready shape.

### M1. Scaling contract as code (§5)

`StreamingRunReport.work_accounting` already carries the counters;
Inc30 promotes the *report schema* to a typed, versioned contract:

```python
@dataclass(frozen=True)
class ScalingReport:            # stream report → verify_layout result
    schema_version: int          # 1
    full_chip_pixels: int
    active_pixels: int
    screened_out_pixels: int
    accounted_pct: float
    peak_rss_bytes: int          # sampled from resource in pipeline
    max_window_px: tuple[int, int]
    dense_allocation_events: int
```

`verify_layout()` exposes it; the benchmark JSONs (Inc28/29 spec files)
reference the same field names. A hidden full-chip materialization
fails the gate (`dense_allocation_events > 0` → status FAIL by
contract, not by benchmark inspection).

### M2. Staged active-set pipeline (§7)

The Stage 0→5 ladder becomes an explicit pipeline profile rather than
ad-hoc hooks:

```text
Stage 0  TileScreeningPolicy (exists, R16.2 adds subdivide-aware screen)
Stage 1  active-set construction = non-screened cores (exists)
Stage 2  exact-vector run/spectral compression per tile (Inc21/22/26 pieces)
Stage 3  continuous local certificate (RFC 0007 / Inc24 bridge — OPEN math)
Stage 4  refinement: increase_halo + subdivide (R16.2)
Stage 5  fold_global(sessions) (R16.1)
```

Inc30 delivers Stages 0/1/2/4/5 as production code and leaves Stage 3
explicitly INCONCLUSIVE-with-reason until the pupil/source quadrature
closes — the firewall is the feature.

### M3. Simulator independence + provenance (§11, §12)

`verify_layout(simulator=..., process_window=..., mode=...)` per §16:
the declared forward model, source, NA/λ, kernel count, focus/dose and
calibration provenance become required fields of
`VerificationResult.model_provenance` (structured dict, not a string).
Three modes (`BENCHMARK`, `CALIBRATED_ENGINEERING`, `PROOF_CARRYING`)
gate which claims the result is allowed to carry; `PROOF_CARRYING`
requires `representation_bridge_upper == 0` backends and refuses
diagnostic-only bridges.

### M4. Benchmark tiers + required baselines (§13, §14)

Formalize the Inc28/29 ladders into a Tier table runner:

- Tier A: deterministic correctness fixtures (existing test suite);
- Tier B: synthetic density/size ladder (Inc28 spec JSON, parameterized
  density — add density sweep, currently sparse-only);
- Tier C: pinned public layouts with hashes + licenses (Inc29 Ibex
  pattern; add sky130hd AEOLUS/ivregate or ORFS GDS as second design);
- Tier D: opt-in, never required.

Baseline modes A–D (dense / tiled / vector / selective) already exist in
both benchmarks; the runner reports the §14 crossover
`A_cross: T_B04 < T_dense` per layout class.

### M5. Real-layout semantics fail-closed gate (§8, §9, §10)

- Keep the Inc22 ownership invariant tests as the §9 firewall.
- Add an adapter conformance suite: every unsupported geometry class
  (PATH caps, OASIS repetitions beyond implemented ones, …) must either
  parse with correct semantics or raise — never silently flatten
  (test-per-semantic-class, mirroring the Inc22 klayout-gated tests).
- Finite-vs-periodic: existing tested firewall stays; no new work.

### M6. Terminology guard (§15)

A docs-lint rule (or test) asserting result-facing strings do not claim
wafer accuracy without `mode=CALIBRATED_ENGINEERING|` wafer validation
provenance. Cheap, prevents drift.

### M7. Hygiene + scoreboard (§17, §18)

Already practiced (per-increment tests, JSONs, SHA256 manifest, ruff,
mypy strict on streaming, live `docs/qdm-readiness.md`); Inc30 wires the
scoreboard update into CI (fail the docs job if `Current increment` /
HEAD lines lag the release tag).

---

## Part III — Inc29 verdict branch map (decision, not code)

| Inc29 verdict at 16384² selective | Inc30 mainline emphasis |
|---|---|
| `STRONG_ON_REALISTIC_DENSITY` (active ≤ 0.5) | R16 then M1–M4: survivor-only continuous verification on real crops; empty screen stays the primary mechanism. |
| `MODERATE_ON_REALISTIC_DENSITY` (0.5 < active ≤ 0.8) | R16 first; add **sensitivity-collar L1 screen** design (screen nonempty tiles by metric relevance) behind the same fail-closed `TileScreeningPolicy` contract — no contract change needed, only a second policy. |
| `WEAK_ON_REALISTIC_DENSITY` (active > 0.8) | Empty screen is demoted to L0; the collar becomes primary. R16.2 `subdivide` routing is the enabling primitive: sensitivity cores are found by subdividing nonempty-but-irrelevant tiles, so Inc30 lands R16 + collar together. |

In all three branches R16 lands first and unchanged — that is why it is
a mandatory repair rather than a branch option.

---

## Non-goals

- No new spectral/quadrature mathematics (Inc26/27 remain the frontier).
- No change to `epe_max_nm` raster metric semantics (§16 firewall).
- No GPU/multiprocessing/caching optimization until profiling evidence
  (philosophy §20.9) — expected *after* Inc30 M4 produces it.
- No trimming of the Inc29 default ladder; the gate result is the
  branch selector.
