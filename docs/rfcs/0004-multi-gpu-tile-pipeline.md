# RFC 0004 — Multi-GPU Tile Pipeline

| | |
|-|-|
| Status | Draft (research, no implementation) |
| Author | OpenLithoHub maintainers |
| Created | 2026-05-20 |
| Targets | v0.3 (provisional) |
| Related | `openlithohub.workflow.tiling`, `openlithohub.cli.optimize_cmd`, RFC 0001 |

## Summary

Today `openlithohub optimize run` processes a tiled layout **sequentially** —
one tile at a time, on a single device. For full-chip designs (M1 GDS in
the GB range) this is the dominant wall-clock cost. This RFC scopes how
to scale that loop across multiple GPUs (and eventually nodes) without
touching the model layer or breaking the ONNX/TorchScript export path.

**This is a research RFC.** No code lands from this document; it picks a
direction so a follow-up implementation RFC can be narrow.

## Current state (factual)

Verified against `main` at 2026-05-20:

- **Tile loop** is a sequential `for tile in tiles: model.predict(tile)`
  in `cli/optimize_cmd.py:142–145`. No batching, no async.
- **Tile geometry** already handles halo overlap and ramp-blended stitching
  (`workflow/tiling.py:11–181`). Tiles are **independent** at inference
  time — the only cross-tile coupling is the post-hoc stitch.
- **Model contract** (`models/base.py:46–56, 64–80`):
  - `predict(design, **kwargs) -> PredictionResult` accepts `(H, W)` or
    `(B, C, H, W)` and unsqueezes internally.
  - `to_torch_module() -> nn.Module` returns a **bare** `nn.Module` for
    export — not wrapped, not on any accelerator.
- **Device placement** is explicit, per-model: `.to(self._device)` /
  `device=` kwarg (`neural_ilt.py:47,97`, `levelset_ilt.py:105–107`).
  No `.cuda()`, no global device manager.
- **`torch.compile`** is applied conditionally to the Hopkins forward
  kernel only (`levelset_ilt.py:127`), with a cache key including
  `str(target.device)` so per-device compilation works.
- **Existing distributed infra**: none. No `torch.distributed`,
  `torch.multiprocessing`, `DataParallel`, `DistributedDataParallel`,
  `accelerate`, or `ray` in `src/`.
- **GPU-gated tests**: none. `tests/test_workflow/` has no
  `torch.cuda.is_available` skips and no distributed harness.

## Hard constraints

1. **Export must keep working.** `cli/export_cmd.py:105` calls
   `litho_model.to_torch_module()` and expects a bare `nn.Module`. Any
   solution that wraps the model itself (e.g. `accelerator.prepare(model)`,
   `DistributedDataParallel(model)`) breaks ONNX export. Therefore
   parallelism must wrap the **tile loop**, not the model.
2. **Single-GPU and CPU paths are the default** (Colab, laptops, CI).
   Multi-GPU is opt-in. No new required dependency.
3. **`torch.compile` cache should not be invalidated.** Cache is keyed
   by device string; one compile per worker is acceptable, recompiling
   per tile is not.
4. **The model layer stays untouched.** No `models/*.py` changes for
   parallelism.

## Options considered

### Option A — `torch.multiprocessing.spawn` over tile shards

Spawn N worker processes, shard `tiles` round-robin, each worker pins
itself to one GPU and runs the existing per-tile `predict` loop, results
go back via a `Queue` and the main process stitches.

- **+** Zero new runtime dependencies (stdlib + torch).
- **+** Model layer untouched; export path untouched.
- **+** `torch.compile` cache lives per-worker — natural fit.
- **+** Halo/stitch logic stays in the main process; workers see
  independent tiles.
- **−** We hand-roll worker lifecycle, error propagation, cancellation,
  KeyboardInterrupt forwarding. Not free, but ~150 lines bounded.
- **−** No multi-node story (need MPI / `torchrun` for that, future work).
- **−** Model weights are reloaded per worker — RAM cost ≈ N × weights.
  Acceptable for current model sizes (NeuralILT < 100 MB), revisit if
  a backbone lands from RFC 0001.

### Option B — Hugging Face Accelerate

Use `Accelerator` + `accelerator.prepare()` to manage devices, optionally
launch via `accelerate launch`.

- **+** Battle-tested, mature, multi-node out of the box.
- **+** Mixed-precision and gradient-accumulation primitives we may want
  later for training-time pipelines.
- **−** **Idiomatic Accelerate wraps the model** — exactly what our
  hard constraint #1 forbids. We could use Accelerate purely for its
  process-group bootstrap and skip `prepare(model)`, but at that point
  we are using ~10% of the library.
- **−** Adds a non-trivial dependency (`accelerate` pulls `huggingface_hub`,
  `psutil`, etc.) for what would, in our usage, be a thin process launcher.
- **−** Users now need to know whether to invoke `python -m
  openlithohub …` or `accelerate launch -m openlithohub …`. Two entry
  points is a documentation tax.
- **−** Export path testing burden: every release we have to verify
  `accelerate` did not silently start wrapping our model.

### Option C — Ray Core

Ray actors, one per GPU, `tiles.map(actor.predict)`.

- **+** Cleanest multi-node story; cluster-aware scheduling for free.
- **+** Decouples tile scheduling from model code completely.
- **−** Heaviest dependency by a wide margin (Ray + grpc + plasma store).
  An order of magnitude more install footprint than A or B.
- **−** Operational complexity (Ray head node, dashboard, etc.) is
  overkill for a single-node 8-GPU box, which is the realistic near-term
  target.
- **−** No current users have asked for cluster scheduling. Premature.

### Option D — Status quo + advisory docs

Document that single-GPU is the supported configuration; users who
want parallelism shard the input GDS themselves and run N CLIs.

- **+** Zero new code.
- **−** The whole point of the "industrial software" pitch is that we
  handle this for them.

## Recommendation

**Option A (`torch.multiprocessing.spawn`)** for v0.3, with **Option C
(Ray)** kept on the table for v0.4+ if multi-node demand materialises.
Reject **Option B (Accelerate)** for this use case — its core value-add
is model wrapping, which our export constraint forbids; using it as a
process launcher is paying a dependency for very little.

### Why not Accelerate

The Accelerate library shines when *training* is the workload: model
sharding, mixed-precision, gradient accumulation, FSDP. Our workload
is **inference over independent tiles** — embarrassingly parallel, with
a hard "do not wrap the model" constraint from the export path. Using
Accelerate here means importing the library and then opting out of its
main API. That is a smell.

### Why not Ray yet

Ray is the right answer when the unit of parallelism is a cluster.
Today the unit is "an 8-GPU workstation" or "an 8-GPU node in a fab
LSF queue." A 2 MB stdlib dependency is a much better fit than a
200 MB Ray cluster dependency for that target. Re-evaluate when a user
actually asks for multi-node — not before.

## Minimal landing path (for the implementation RFC)

The follow-up RFC should bound the work to:

1. **CLI surface**: add `--num-gpus N` to `cli/optimize_cmd.py`. `N=1`
   (default) keeps the current sequential path bit-for-bit identical.
   `N>1` enters the multiprocessing path.
2. **New module**: `src/openlithohub/workflow/parallel.py` (~150 lines)
   - `parallel_tile_inference(model_factory, tiles, num_gpus) ->
     list[(Tile, Tensor)]`
   - Workers receive a *factory* (model name + kwargs), not a live
     model object — keeps pickling sane and avoids CUDA-context-fork
     hazards.
   - `mp.spawn` with `start_method='spawn'` (not `fork` — CUDA + fork
     is undefined behaviour).
3. **No new runtime dependency.** Stdlib + torch only.
4. **Tests** (`tests/test_workflow/test_parallel.py`):
   - CPU-only smoke test using `num_gpus=2` mapped to `cpu` (validate
     the dispatch logic without needing GPUs in CI).
   - GPU test gated on `torch.cuda.device_count() >= 2`, marked
     `@pytest.mark.gpu`, skipped in default CI.
5. **Regression coverage** for the unchanged path:
   - Existing tiling tests must pass with `num_gpus=1` (the default).
   - Export path test (`test_cli/test_export.py`, if present) must pass
     unchanged — the model layer is not touched.
6. **Docs**: one new section in `docs/cli-reference.md` under
   `optimize run`, plus a one-paragraph mention in `docs/architecture.md`.
   No new top-level page.

## Out of scope for v0.3

- Multi-node (Ray / `torchrun` / Slurm).
- Mixed-precision inference.
- Streaming OASIS reader (the assumption is the layout fits in host
  RAM as a tensor; this is the same assumption today).
- Per-GPU dynamic load balancing (we shard tiles round-robin; if some
  tiles are slower than others, c'est la vie for v0.3).

## Open questions for the implementation RFC

1. Should workers each load their own model weights, or should the
   main process load once and pass via shared memory? (Probably "each
   loads" — shared memory + CUDA is fragile.)
2. How to surface per-worker `torch.compile` warmup time in the
   progress bar? (Probably ignore for v0.3 — first tile per worker is
   slow, subsequent ones aren't.)
3. KeyboardInterrupt propagation — `mp.spawn` swallows signals in
   subtle ways. Worth a dedicated test.
