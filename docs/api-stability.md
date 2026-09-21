# API Stability

<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

OpenLithoHub is pre-1.0 and evolving, but the surface is **not uniformly
experimental**. Import according to this stability table (audit P2.9):

| Namespace | Stability | Contract |
|---|---|---|
| `openlithohub` (package root: `Mask`, `LitheEngine`, `Report`, `__version__`) | **Public** | Breaking changes only with a deprecation cycle and a CHANGELOG entry |
| `openlithohub.api.*` | **Public** | Same contract as the package root |
| `openlithohub.cli.*` command names and flags | **Public** | Renames/flag removals are breaking changes |
| `openlithohub.server` HTTP API (`/v1/*`) | **Public, versioned** | `/v1` responses carry `api_schema_version`; within `/v1` only additive changes |
| `openlithohub.benchmark.metrics` / `compliance` | **Public** | Metric definitions may gain fields; existing field semantics are stable |
| `openlithohub.benchmark.industrial` | **Public (v1)** | Artifact schema `OpenLithoHub.industrial-benchmark.v1` is frozen for v1.1; additive fields only |
| `openlithohub.models` / `simulators` registries | **Provisional** | New models/backends appear here; per-model constructor kwargs are each model's own surface |
| `openlithohub.streaming.*` | **Provisional** | RFC 0008 architecture; APIs may change with a migration note |
| `openlithohub.data.*` adapters | **Provisional** | Adapter behavior follows upstream dataset layouts |
| `openlithohub.verify.*` | **Specialized** | P-054-governed proof surface — governed separately, never changed casually |
| `openlithohub._utils.*` | **INTERNAL** | No stability guarantee. Code in `openlithohub._utils` may change at any time; examples that import it are illustrative, not a supported API. Long-lived consumers should use the public façade instead. |

Notes:

- `ModelRegistry.get(..., ignore_unsupported=True)` is the CLI default
  (unknown kwargs are dropped so optional flags work across models).
  Programmatic/server configuration should pass
  `ignore_unsupported=False` so typos raise `ValueError` instead of
  silently misconfiguring a model.
- Server payloads: `/v1/capabilities` reports `api_schema_version` and
  `capability_schema_version`; job records report the same
  `api_schema_version`. Clients should ignore unknown fields.
- `ModelHub` / pretrained weight URLs follow the model releases, not this
  package's version.
