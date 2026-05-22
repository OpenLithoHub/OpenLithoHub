# Citation Map

This page is the authoritative crosswalk between **OpenLithoHub design decisions** and the **published works** that justify them. If you find a constant, parameter, or schema rule with no matching row here, that's a documentation bug — please open an issue.

The BibTeX entries are stored in
[`docs/references.bib`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/docs/references.bib).
Citation keys below match that file verbatim.

## How to read this table

- **Decision** — what OpenLithoHub does (a constant, a default, a schema rule, a metric formula).
- **Where it lives** — the file:line that implements it. Search for the citation key in source comments to find the inline justification.
- **Citation key** — the BibTeX key in `references.bib`.
- **Section / claim** — the specific section, table, or figure of the paper that backs the choice.

## Forward-simulation and printability metrics

| Decision | Where it lives | Citation key | Section / claim |
|----------|----------------|--------------|------------------|
| Resist threshold defaults to `0.225` | `src/openlithohub/simulators/base.py` | `Yang2023_LithoBench` | §3.2 — calibrated against the ICCAD-16 reference resist model. |
| Forward-sim gate at submit-time (`l2_error_pixels` is required) | `src/openlithohub/leaderboard/tracker.py` | `Yang2023_LithoBench` | Table III — academic OPC printability = L2 + PVB on the simulated wafer image; an EPE-only score with no L2 is rejectable. |
| Hopkins SOCS uses **24 kernels** by default | `src/openlithohub/simulators/hopkins_sim.py` | `Yang2023_LithoBench`, `Cobb1995_FastSparse` | Yang §3.2 / Table II for the count; Cobb for the SOCS construction itself. |

## Datasets

| Adapter | Where it lives | Citation key | Notes |
|---------|----------------|--------------|-------|
| `LithoBenchDataset` | `src/openlithohub/data/lithobench.py` | `Yang2023_LithoBench` | NeurIPS'23 — paper introducing the benchmark consumed by this adapter. |
| `Iccad16Dataset` | `src/openlithohub/data/iccad16.py` | `Yang2016_ICCAD16Bench`, `Banerjee2013_ICCAD`, `Yang2020_BatchAL` | The 7nm-N7M2EUV release (Yang2016) extends the original ICCAD-2013 contest format (Banerjee2013). The N7M2EUV stack and per-layer mapping convention are documented in `Yang2020_BatchAL` §III-A. |
| `GanOpcDataset` | `src/openlithohub/data/ganopc.py` | `Yang2018_GANOPC` | DAC'18 — paper releasing the underlying mask-optimization dataset. |

## Models

| Component | Where it lives | Citation key | Notes |
|-----------|----------------|--------------|-------|
| `NeuralILTModel` (U-Net + L2/PVB co-loss) | `src/openlithohub/models/neural_ilt.py` | `Jiang2020_NeuralILT` | ICCAD'20 — architecture and loss formulation. Architecture audit is task 3.3. |

## Baselines

| Component | Where it lives | Citation key | Notes |
|-----------|----------------|--------------|-------|
| `batch_active_select` (uncertainty + diversity batch sampler) | `src/openlithohub/baselines/hotspot_batchal.py` | `Yang2020_BatchAL` | TCAD'20 §III — Eq. (8) uncertainty + Eq. (9) inner-product diversity. Greedy max-min selection replaces the paper's QP relaxation (Theorem 1 bounds the gap). The full active-learning loop (paper §3.4) is **not** shipped — see Candidate techniques table below. |

## Metadata format

| Surface | Where it lives | Citation key | Notes |
|---------|----------------|--------------|-------|
| `DatasetAdapter.to_croissant()` | `src/openlithohub/data/base.py` | `MLCommons2024_Croissant` | MLCommons Croissant 1.0 JSON-LD format — the de-facto ML metadata schema (HuggingFace, Kaggle, Google). |

## Tile / halo strategy

| Decision | Where it lives | Citation key | Notes |
|----------|----------------|--------------|-------|
| Process-node-aware halo sizing (`halo_px = max(ceil(OIR_nm/pixel_nm), receptive_field_px)`) | RFC 0005 (`docs/rfcs/0005-process-node-halo-sizing.md`), `src/openlithohub/workflow/halo.py` | — | Single-resolution physical optical-interaction-radius formula; no published-paper citation drives the formula itself (`OIR ≈ 10 × λ/(2·NA)` is textbook Hopkins/SOCS). |

## Candidate techniques (cited but not yet implemented)

The entries below are kept in `docs/references.bib` because they are
plausible techniques for a future v0.x performance pass, **not** because
the current code uses them. Adding a "Where it lives" pointer for any of
these requires implementing the technique first.

| Citation key | Technique | Where it would land if implemented | Status |
|--------------|-----------|------------------------------------|--------|
| `Yu2014_AccelerationOPC` | Coarse-to-fine multi-resolution SOCS forward-sim | `src/openlithohub/simulators/hopkins_sim.py` | Not implemented as of 2026-05-23. RFC 0005's halo pipeline uses a single-resolution OIR formula, not Yu2014's coarse-then-refine strategy. Verified against the actual code. |
| `Yang2020_BatchAL` | Full hotspot active-learning loop (detector training + lithography-simulation oracle alongside the §III sampler) | `src/openlithohub/baselines/hotspot_batchal.py` already ships the §III sampler; the loop would land beside it as `hotspot_al_loop.py`. | Sampler shipped 2026-05-23; full loop not implemented because OpenLithoHub does not ship a hotspot detector and the on-disk ICCAD16 corpus has only one testcase. See `out/plans/external-resource-utilization.md` Task #1 v0.2. **Note:** the same citation is also wired in for two unrelated purposes — the N7M2EUV stack / layer-mapping convention used by `Iccad16Dataset` (Datasets table) and the sampler itself (Baselines table). |

## Adding a new citation

1. Add the `@type{key, ...}` block to `docs/references.bib`. Use the
   `FirstAuthor<YEAR>_ShortTopic` key style.
2. Reference the key verbatim in the source comment / docstring at the point
   of use (so `grep` finds both sides).
3. Add a row above pointing at the file/section.
4. If the paper supersedes an existing citation, update the rows that pointed
   at the old key — don't leave stale pointers.

For the BibTeX file format and snapshotting policy, see
[References](references.md).
