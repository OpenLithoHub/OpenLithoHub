# Built-in demo samples

These PNGs power the HF Playground "Try a preset" examples in both the
Synthetic Patterns tab and the Upload Masks tab. They are deterministic
outputs of `scripts/generate_demo_samples.py` (run from the repo root) and
are committed so the HF Space cold-start has no runtime tempdir dependency.

| Slug            | Pattern         | Source                                 |
| --------------- | --------------- | -------------------------------------- |
| `line_space`    | Line/Space grid | Synthetic, Apache-2.0 (this repo)      |
| `contact_holes` | Contact array   | Synthetic, Apache-2.0 (this repo)      |
| `sram_like`     | SRAM-like cell  | Synthetic, Apache-2.0 (this repo)      |
| `random_logic`  | Manhattan logic | Synthetic, Apache-2.0 (this repo)      |

Each preset ships as two files: `<slug>_target.png` (clean design) and
`<slug>_pred.png` (a perturbed prediction with intentional EPE > 0).

To regenerate after changing the script:

```bash
.venv/bin/python scripts/generate_demo_samples.py
```
