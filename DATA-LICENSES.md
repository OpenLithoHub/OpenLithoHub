# Dataset Licenses

OpenLithoHub **does not redistribute raw datasets**. The dataset adapters in
`src/openlithohub/data/` download from each dataset's official source upon
user request, or expect the user to provide a local copy. Each dataset
retains its original license, and users are responsible for complying with
that license when downloading or using the data through OpenLithoHub.

This file lists the datasets with first-class adapter support in
OpenLithoHub. New adapters must add an entry to this file as part of the PR
that introduces them — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Supported Datasets

| Dataset | Adapter | Original License | Source | Citation |
|---------|---------|------------------|--------|----------|
| LithoBench | `openlithohub.data.lithobench` | See upstream repository (verify before redistribution) | https://github.com/phdyang007/lithobench | Yang et al., *LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing*, NeurIPS 2023 |
| LithoSim | `openlithohub.data.lithosim` | See upstream source (verify before redistribution) | (upstream source as documented in adapter) | (refer to upstream source) |

> **Note**: License entries above marked "verify before redistribution" mean
> the upstream project did not declare a single SPDX-identified license at
> the time this file was written. If you intend to redistribute or build a
> commercial product on top of these datasets, **independently confirm the
> upstream license terms** before doing so.

---

## What OpenLithoHub Does and Does Not Do

OpenLithoHub:

- ✅ Provides Python adapters that load datasets into a common
  `LithoSample` interface for benchmarking.
- ✅ Computes metrics (EPE, PV Band, shot count, etc.) from those samples.
- ❌ Does **not** include or redistribute the underlying dataset files in
  this repository.
- ❌ Does **not** alter, sublicense, or relicense any third-party dataset.

The OpenLithoHub adapter code itself is licensed under Apache 2.0 (per
[LICENSE](LICENSE)). The data those adapters load is governed exclusively
by the dataset's own license.

---

## User Responsibilities

Before using any dataset through OpenLithoHub, you must:

1. Visit the dataset's official source URL and read its license.
2. Comply with all terms of that license — including attribution, citation,
   non-commercial restrictions (if any), and redistribution constraints.
3. Provide the citation requested by the dataset authors in any
   publication, product documentation, or benchmark report that uses
   results derived from the dataset.
4. For any commercial use (e.g., training a production model, building a
   commercial benchmark service), independently confirm that the dataset
   license permits your intended use case.

OpenLithoHub maintainers make **no representation** as to the suitability
of any third-party dataset for any particular use, commercial or otherwise.

---

## Adding a New Dataset Adapter

A pull request that adds a new dataset adapter must also:

1. Add a row to the **Supported Datasets** table above with:
   - The Python module path of the adapter.
   - The dataset's original license (SPDX identifier where available).
   - The dataset's official source URL.
   - The citation key or full reference requested by the dataset authors.
2. If the dataset's license is unclear, add a note (as above) and link
   any correspondence with the upstream maintainers.
3. Implement `download()` to fetch from the official source — never
   embed dataset bytes in the repository.
4. Surface citation information to end users (e.g., printed when an
   adapter is first instantiated, or via a `citation` property on the
   adapter class).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution flow,
including the CLA requirement.

---

## Reporting a License Issue

If you believe a dataset listed here is misattributed, has changed
license, or has been removed from its upstream source, please email
**teller.lin@outlook.com** or open a GitHub issue.
