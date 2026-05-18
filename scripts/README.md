# scripts/

One-off and reference scripts that live outside the installable package.

| Script | Purpose |
|--------|---------|
| `generate_baselines.py` | Run all built-in models on a small dummy dataset and write `baselines/results.{json,md}`. |
| `generate_hero_figure.py` | Render the 3-panel design / no-OPC / with-ILT comparison published as the docs hero image. |
| `train_neural_ilt.py` | Reference trainer for the U-Net consumed by `openlithohub.models.neural_ilt`. |

## Training the Neural-ILT baseline

`scripts/train_neural_ilt.py` is a deliberately small reference trainer.
It is the script the project uses to produce the weights served at
`openlithohub/neural-ilt-v1` on HuggingFace; copy and adapt for your own
runs.

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

Then point `NeuralILTModel(pretrained=True)` at your repo. The repo path
is currently hardcoded at `openlithohub/neural-ilt-v1` in
`src/openlithohub/models/neural_ilt.py`; making it configurable is a
welcome follow-up PR.

### Caveats

- **No data augmentation**: this is a baseline trainer. Add rotations /
  flips for production.
- **No held-out validation split**: the metadata sidecar reports training
  loss only. Wire `LithoBenchDataset(split="val")` into the loop if your
  data has one.
- **Single-GPU / CPU**: no DDP. For multi-node training, wrap the loop
  in `torch.nn.parallel.DistributedDataParallel` and replace the
  `DataLoader` with one that uses `DistributedSampler`.
