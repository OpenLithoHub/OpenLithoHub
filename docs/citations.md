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
| `Iccad16Dataset` | `src/openlithohub/data/iccad16.py` | `Yang2016_ICCAD16Bench`, `Banerjee2013_ICCAD` | The 7nm-N7M2EUV release (Yang2016) extends the original ICCAD-2013 contest format (Banerjee2013). |
| `GanOpcDataset` | `src/openlithohub/data/ganopc.py` | `Yang2018_GANOPC` | DAC'18 — paper releasing the underlying mask-optimization dataset. |

## Models

| Component | Where it lives | Citation key | Notes |
|-----------|----------------|--------------|-------|
| `NeuralILTModel` (U-Net + L2/PVB co-loss) | `src/openlithohub/models/neural_ilt.py` | `Jiang2020_NeuralILT` | ICCAD'20 — architecture and loss formulation. Architecture audit is task 3.3. |

## Metadata format

| Surface | Where it lives | Citation key | Notes |
|---------|----------------|--------------|-------|
| `DatasetAdapter.to_croissant()` | `src/openlithohub/data/base.py` | `MLCommons2024_Croissant` | MLCommons Croissant 1.0 JSON-LD format — the de-facto ML metadata schema (HuggingFace, Kaggle, Google). |

## Tile / halo strategy

| Decision | Where it lives | Citation key | Notes |
|----------|----------------|--------------|-------|
| Multi-resolution forward-sim in halo pipeline | RFC 0005 (`docs/rfcs/0005-process-node-halo-sizing.md`) | `Yu2014_AccelerationOPC` | ICCAD'14 — original accelerator strategy. |

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
