# OpenLithoHub Strategic Whitepaper (2026 Ultimate Edition)

**Positioning: Open-source computational lithography workflow, manufacturability benchmarking infrastructure, and foundation model data engine for advanced EUV/curvilinear mask processes**

> **Project Vision:** "We don't build lithography machines, nor do we write low-level physics engines. We pave the standardized highway connecting everyone who optimizes lithography — ensuring it leads to both the Manhattan present and the curvilinear future."

[中文版 / Chinese Version](docs/README_zh.md)

---

## I. Industry Landscape & Ecosystem Gaps (2026)

Strategy begins with a clear-eyed view of the battlefield. Computational lithography is caught between an "compute explosion" and "ecosystem fragmentation."

### 1.1 Industry: The Compute Revolution, Curvilinear Masks & EUV Stochastics

- **NVIDIA cuLitho in Production:** TSMC has deployed cuLitho, achieving 40–60× ILT speedup. The low-level compute war (CUDA/C++) is over — competition has shifted to **AI algorithm orchestration, evaluation, and productionization**.
- **MBMW & OASIS.MBW as Standard:** Multi-beam mask writers (MBMW) make curvilinear masks the absolute mainstream for sub-28nm. Data volumes explode; GDSII is obsolete. Industry has fully migrated to **OASIS** format (10× design data compression, 4×+ post-OPC compression). The **`OASIS.MBW 2.1` standard**, purpose-built for mask writers with native curve primitives, is the lifeline of advanced nodes.
- **EUV Stochastics as Core Pain Point:** Photon shot noise causes line-edge roughness (LER) and micro-bridging — the yield killers. At current EUV doses, sub-20nm stochastic LER can exceed **20%** of critical dimension, far beyond the ITRS **8%** safety threshold. Deterministic optical simulation cannot meet 2nm demands.

### 1.2 Academia: Thriving Islands & the Half-Life Crisis

Led by `OpenOPC` (CUHK), academia has produced exceptional open-source projects (`TorchLitho 2.0`, `LithoSim`, `curvyILT`), yet the ecosystem is severely fragmented:

- **Format & Metric Divergence:** Datasets (LithoBench vs LithoSim) are incompatible; EPE/PVB definitions vary; fair cross-paper comparison is impossible.
- **Engineering Pipeline Breaks:** Academic tools operate on raw tensors — no open-source tool handles real OASIS/GDSII full-chip end-to-end optimization.
- **Half-Life Crisis:** Papers go unmaintained post-publication; dependency conflicts (Python/PyTorch version hell) make industrial reproduction nearly impossible.

### 1.3 Strategic Conclusion: OpenLithoHub's Precise Niche

```text
[Compute Layer]     NVIDIA cuLitho / GPU Clusters (existing — we don't compete)
[Physics Engine]    TorchLitho 2.0 / OpenILT / curvyILT (existing — we integrate)
[Dataset Layer]     LithoBench / LithoSim / ICCAD (existing — we unify access)
                    ↑
[OpenLithoHub]      Cross-framework eval + OASIS workflow + EUV/MRC compliance + Data engine
                    ↑
[Users]             Researchers / Chip Engineers / EDA Foundation Model Developers
```

---

## II. Core Architecture (Five-Layer Industrial Model)

OpenLithoHub adopts a **plugin-first** modular design, bridging the gap from academic research to industrial manufacturing.

### Layer 1: Unified Data Adapter Layer

Abstracts format differences, providing a unified PyTorch Tensor output interface.

- **Backward Compatible:** One-click loading of LithoBench (`.npy`), LithoSim (HuggingFace Parquet).
- **Metadata Alignment:** Automatic alignment of pixel resolution, process window, and source parameters.

### Layer 2: Manufacturability & EUV Benchmark — *Core Differentiator*

Breaking academia's "EPE-only" mindset by introducing metrics the industry actually cares about:

- **EUV Stochastic Robustness:** Novel stochastic noise injection to quantify micro-bridging/open probability of AI-generated masks under photon shot noise — addressing the sub-20nm LER crisis.
- **MRC/DRC Violation Rate:** Integrated `EasyMRC` and `OpenDRC` for minimum width/spacing rule checks (hard-fail metric).
- **Standardized Precision & Cost:** Unified EPE and PV Band computation; shot count estimation for mask manufacturing cost.

### Layer 3: Model Integration Layer

Provides a minimal `LithographyModel` interface. Whether traditional heuristic OPC, U-Net deep learning, or state-of-the-art curvilinear ILT (`curvyILT`), implementing `predict()` is all that's needed to join the evaluation pipeline.

### Layer 4: OASIS.MBW Workflow Engine — *Largest Engineering Moat*

**Bridging the last mile from tensor to fab.**

- **Full-Chip Distributed Processing:** Rapid parsing and tiling of design layouts via `KLayout` Python API.
- **Dual-Track Contour Export (Bypassing KLayout Limitations):**
  - *Manhattan Mode (Legacy):* Staircase polygons for traditional VSB writers, leveraging KLayout's geometry engine.
  - *Curvilinear Mode (Modern):* **Bypasses traditional geometry engines** — directly fits AI-generated smooth contours to B-spline curves and **natively serializes to `OASIS.MBW 2.1` format** for multi-beam writers. True mathematical curve representation without quadratic data explosion.
- **Target CLI Experience:**
  ```bash
  openlithohub optimize --input chip.oas --model diffusion-ilt --writer mbmw --node 3nm-euv --drc-check --output optimized.oas
  ```

### Layer 5: Leaderboard & Foundation Model Data Engine

- **Public SOTA Tracking:** Building computational lithography's PapersWithCode — ranked by process node and mask topology.
- **EDA Foundation Model Data Engine:** The biggest bottleneck for EDA visual large models (LVMs) is **extreme scarcity of open circuit data**. OpenLithoHub is not just a benchmark — it's a **data generator**. Automated pipelines produce high-quality "layout → mask → resist contour" paired datasets across process conditions and compliance labels, fueling next-gen EDA foundation model pre-training.

---

## III. Execution Roadmap (12-Month Plan)

### Phase 1: Minimum Viable Benchmark MVP (Months 1–2)

- **Goal:** Establish the first "cross-dataset comparison ruler."
- **Action:** Unified DataLoader for LithoBench & LithoSim; basic EPE evaluation; `openlithohub eval` CLI.
- **Deliverable:** 30-second terminal GIF showing standardized evaluation report across two datasets.

### Phase 2: Manufacturability Compliance & Contour Extraction (Months 3–4)

- **Goal:** Dimensional superiority over existing academic benchmarks.
- **Action:** MRC check module; pixel-to-polygon/curve contour extraction (referencing `EasyMRC` & `curvyILT`).
- **Deliverable:** Evaluation report adds "MRC Violation Rate" metric. Blog post: *"Why 90% of AI Lithography Papers Are Waste Paper in the Fab."*

### Phase 3: OASIS Workflow & Web Demo (Months 5–6)

- **Goal:** Ignite the engineering community.
- **Action:** End-to-end `.oas/.gds` optimization pipeline; zero-config Web playground on HuggingFace Spaces (drag layout, one-click optimize).
- **Deliverable:** First 200+ GitHub Stars; attract industrial engineers.

### Phase 4: Academic Coalition & Leaderboard Launch (Months 7–9)

- **Goal:** Establish industry-standard status.
- **Action:** Official leaderboard website; proactively submit PRs to upstream projects, inviting authors to adopt OpenLithoHub as the recommended evaluation tool.
- **Deliverable:** At least 3 top university teams submit scores.

### Phase 5: Foundation Incubation & Commercialization (Months 10–12)

- **Goal:** Secure long-term funding and official endorsement.
- **Action:** Apply to **CHIPS Alliance** (Linux Foundation) as an incubation project; launch "Private Benchmark" commercial offering for fabless companies.

---

## IV. Business Model & Moat Analysis

### 4.1 The Real Moat: Standard Inertia & Ecosystem Lock-in

OpenLithoHub's ultimate barrier is not code complexity — it's **monopoly over the unit of measurement**. Once academia publishes with it, industry validates with it, and foundation model teams generate data with it, it becomes irreplaceable infrastructure.

### 4.2 Monetization Paths (Open Core, Commercial Add-ons)

1. **Private Benchmark Hosting:** Enable fabless companies to objectively evaluate Synopsys/Cadence/startup AI algorithms without exposing confidential OASIS data (currently a blank paid market).
2. **Enterprise Orchestration:** Commercial toolchain with Kubernetes cluster scheduling for full-chip scale compute and memory management.
3. **Cloud Mask Optimization SaaS:** Pay-per-use GPU-accelerated optimization for small/medium design houses.

---

## V. Day 1–30 Quick Start Guide (Action Items for Founders)

Don't be overwhelmed by the grand architecture — the first 30 days are pure engineering basics:

- **Day 1–3 (Claim Territory):** Register GitHub org `OpenLithoHub`, upload this whitepaper as `README.md`. Reserve PyPI package name.
- **Day 4–10 (Conquer Data):** Download minimal samples from LithoSim & LithoBench (100 images each). Write a Python script to load and visualize with `matplotlib`.
- **Day 11–20 (Conquer Metrics):** Write a simple `metrics.py` computing pixel-level differences (basic EPE).
- **Day 21–30 (Package CLI):** Use `Typer` to build a CLI tool wiring data loading and metrics. Record a slick terminal GIF for the README header.

**Your open-source startup journey officially begins the moment you type `git init`.**

---

## Appendix: Key Open-Source Projects & References

| Project | Venue | Description |
|---------|-------|-------------|
| **LithoSim** | NeurIPS'25 | Sub-28nm industrial benchmark & dataset |
| **LithoBench** | NeurIPS'23 | 45nm baseline evaluation framework |
| **TorchLitho 2.0** | ASICON'25 | State-of-the-art differentiable lithography simulator |
| **curvyILT** | NVIDIA arXiv'24 | GPU-accelerated curvilinear ILT with B-spline contours |
| **EasyMRC** | TODAES'25 | Manhattanization & MRC reference implementation |
| **IEEE DATC RDF-2025** | — | Authoritative description of AI-for-EDA reproducibility crisis |

---

## License

[Apache 2.0](LICENSE)
