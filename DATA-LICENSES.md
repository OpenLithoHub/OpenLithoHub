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
| LithoBench | `openlithohub.data.lithobench` | See upstream release (verify before redistribution) | Refer to the NeurIPS 2023 LithoBench paper for the project's current data release page | Yang et al., *LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing*, NeurIPS 2023 |
| LithoSim | `openlithohub.data.lithosim` | See upstream source (verify before redistribution) | (upstream source as documented in adapter) | (refer to upstream source) |
| ICCAD16-N7M2EUV | `openlithohub.data.iccad16` | See upstream repository (verify before redistribution) | https://github.com/phdyang007/ICCAD16-N7M2EUV | ICCAD 2016 CAD Contest, Problem C — EUV Simulation; Yang et al., *Bridging the Gap Between Layout Pattern Sampling and Hotspot Detection via Batch Active Learning*, TCAD 2020 |
| GAN-OPC | `openlithohub.data.ganopc` | See upstream repository (verify before redistribution) | https://github.com/phdyang007/GAN-OPC | Yang et al., *GAN-OPC: Mask Optimization with Lithography-guided Generative Adversarial Nets*, TCAD 2020 |
| ASAP7 (predictive 7nm PDK) | `openlithohub.data.asap7` | BSD-3-Clause | https://github.com/The-OpenROAD-Project/asap7 | Clark et al., *ASAP7: A 7-nm finFET predictive process design kit*, Microelectronics Journal, 2016 |
| FreePDK45 + NanGate OCL (predictive 45nm PDK) | `openlithohub.data.freepdk45` | Stacked: FreePDK45 (NCSU) + NanGate Open Cell Library (Si2). Verify both upstream — the mflowgen mirror ships no LICENSE file. | https://github.com/mflowgen/freepdk-45nm (mirror); https://eda.ncsu.edu/freepdk/freepdk45/ (FreePDK45 terms); https://si2.org/open-cell-library/ (NanGate terms) | Stine et al., *FreePDK: An Open-Source Variation-Aware Design Kit*, MSE 2007 (FreePDK); Si2 NanGate Open Cell Library v1.3 |
| ORFS-routed ASAP7 layouts (mock-alu, riscv32i, …) | `openlithohub.data.orfs` | Same as ASAP7 (BSD-3-Clause); ORFS itself is BSD-3-Clause. The adapter never redistributes ORFS or ASAP7 bytes — GDS is produced locally via the `build-asap7-mock-alu` GitHub Actions workflow or `scripts/build_riscv_alu.sh`. | https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts (ORFS); https://github.com/The-OpenROAD-Project/asap7 (PDK) | Ajayi et al., *Toward an Open-Source Digital Flow*, DAC 2019 (ORFS); Clark et al., 2016 (ASAP7) |

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
**openlithohub@163.com** or open a GitHub issue.
