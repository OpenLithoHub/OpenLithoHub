# OpenLithoHub

**Open-source computational lithography benchmarking and workflow toolkit for advanced EUV/curvilinear mask processes.**

---

## Overview

OpenLithoHub provides a unified evaluation and workflow framework for computational lithography research. It bridges the gap between academic tensor-based optimization and industrial mask manufacturing.

```text
┌─────────────────────────────────────────────────────────┐
│                    OpenLithoHub                          │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  Data Layer │  Benchmark   │   Workflow   │     CLI     │
│ LithoBench  │  EPE/PVBand  │ Tiling/Stitch│ eval        │
│ LithoSim    │  MRC/DRC     │ Contour Ext. │ optimize    │
│ Transforms  │  Stochastic  │ OASIS Export │ leaderboard │
│             │  Shot Count  │ B-spline Fit │             │
└─────────────┴──────────────┴──────────────┴─────────────┘
```

## Key Features

- **Unified dataset access** — single interface to LithoBench, LithoSim, and other lithography datasets
- **Standardized metrics** — EPE, PV Band, shot count, EUV stochastic robustness
- **Manufacturing compliance** — MRC/DRC rule checking as hard-fail gating
- **OASIS workflow** — end-to-end pipeline from tensor to fab-ready mask (manhattan & curvilinear)
- **Model-agnostic evaluation** — plug any OPC/ILT model into the benchmark via a minimal interface
- **Public leaderboard** — track SOTA results across models, datasets, and process nodes

## Quick Links

- [Getting Started](getting-started.md) — installation and first evaluation
- [Architecture](architecture.md) — system design and module overview
- [CLI Reference](cli-reference.md) — command-line usage
- [API Reference](api/data.md) — Python API documentation
- [Contributing](contributing.md) — how to contribute
