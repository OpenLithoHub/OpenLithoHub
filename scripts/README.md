# scripts/

One-off and reference scripts that live outside the installable package.

| Script | Purpose |
|--------|---------|
| `generate_baselines.py` | Run all built-in models on a small dummy dataset and write `baselines/results.{json,md}`. |
| `generate_hero_figure.py` | Render the 3-panel design / no-OPC / with-ILT comparison published as the docs hero image. |
| `train_neural_ilt.py` | Reference trainer for the U-Net consumed by `openlithohub.models.neural_ilt`. |
| `train_gan_opc.py` | GAN-OPC v0.4 trainer with metric-aligned PVB bandwidth loss, UNetV2, memmap cache. |

## Training the Neural-ILT baseline

`scripts/train_neural_ilt.py` is a deliberately small reference trainer
suitable for your own data and configs. The published v0.1 seed weights
served at [`openlithohub/neural-ilt-v0.1`](https://huggingface.co/openlithohub/neural-ilt-v0.1)
were produced by `scripts/train_neural_ilt_seed.py`, which pins every
hyperparameter so the checkpoint is reproducible bit-for-bit (modulo
PyTorch minor-version drift).

### Smoke test (no real training)

Useful in CI or to confirm your environment is wired up. Trains for one
batch on the dummy generator (no dataset required), saves a checkpoint,
exits.

```bash
python scripts/train_neural_ilt.py --smoke-test \
    --output checkpoints/neural_ilt_smoke.pt
```

### Real training on LithoBench

```bash
python scripts/train_neural_ilt.py \
    --data-root /path/to/lithobench \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --forward-model gaussian \
    --device cuda \
    --output checkpoints/neural_ilt.pt
```

The loss combines BCE-with-logits against the ground-truth mask with an
imaging-consistency term:
`MSE(forward(sigmoid(logits)), design)`. The forward model is the same
Gaussian PSF used for the hero figure (`sigma_px=4.5`), so the U-Net
learns ILT-style mask inversion rather than just mask reconstruction.

Pass `--forward-model hopkins` to use the differentiable Hopkins SOCS
forward instead — slower per-step but physically faithful.

A `<output>.metadata.json` sidecar records hyperparameters and per-epoch
loss history.

### Uploading weights to HuggingFace Hub

```bash
huggingface-cli upload <user>/<repo> checkpoints/neural_ilt.pt
```

Then point `NeuralILTModel` at your repo via the `repo_id` constructor
argument — for example `NeuralILTModel(pretrained=True, repo_id="<you>/<repo>")`.
The default is `openlithohub/neural-ilt-v0.1`, which carries the
published v0.1 seed weights.

### Caveats

- **No data augmentation**: this is a baseline trainer. Add rotations /
  flips for production.
- **No held-out validation split**: the metadata sidecar reports training
  loss only. Wire `LithoBenchDataset(split="val")` into the loop if your
  data has one.
- **Single-GPU / CPU**: no DDP. For multi-node training, wrap the loop
  in `torch.nn.parallel.DistributedDataParallel` and replace the
  `DataLoader` with one that uses `DistributedSampler`.

## Training the GAN-OPC v0.4 model

`scripts/train_gan_opc.py` trains a U-Net mask predictor with BCE +
consistency + MRC + optional PVB bandwidth loss.

### Quick start

```bash
python scripts/train_gan_opc.py \
    --data-root data/ganopc/extracted/ganopc-data \
    --resize-to 256 --pixel-size-nm 8.0 \
    --epochs 50 --batch-size 4 \
    --lambda-mrc 0.5 --lambda-pvb 0.1 --pvb-steepness 20.0 \
    --cache-dir cache/ganopc/ \
    --output checkpoints/gan_opc_v0_4.pt
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda-pvb` | 0.1 | PVB bandwidth loss weight (0 = disabled) |
| `--pvb-steepness` | 20.0 | Sigmoid steepness for differentiable threshold |
| `--lambda-mrc` | 0.5 | MRC loss weight |
| `--gradient-accumulation` | 1 | Accumulate gradients over N steps |
| `--mixed-precision` | off | Enable AMP (CPU: `torch.amp.autocast('cpu')`) |
| `--arch` | unet | `unet` (3-level) or `unetv2` (4-level, 7.7M params) |
| `--plateau-patience` | 5 | Early-stop after N epochs with no loss decrease |
| `--cache-dir` | cache/ganopc/ | Memmap cache directory (auto-built on first run) |

### Probes and eval

- `scripts/_probe_p3_pvb_bandwidth.py` — PVB bandwidth loss correctness
- `scripts/_probe_p4_p5_p6.py` — Cache integrity, AMP consistency, memory peak
- `scripts/_eval_v04_iccad16.py` — ICCAD16 evaluation with greenlight rules

### Caveats

- **CPU training**: on AMD 5600G (6C/12T), expect ~45 min/epoch at 256² (no PVB)
  or ~90 min/epoch with 4-corner PVB bandwidth loss. GPU is ~30× faster.
- **Memmap cache**: first run builds a ~1.2 GB (256²) or ~5.1 GB (512²) cache file.
  Subsequent runs reuse it instantly.
