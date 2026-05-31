## Cross-Pattern Stochastic Benchmark

Generated with `python scripts/run_stochastic_benchmark.py`. All runs use `BayesianStochasticModel(mode='poisson')` with `n_mc_samples=64`, cross-validated against `compute_stochastic_robustness` (bridge/break) and `compute_stochastic_defect_classes` (imec per-class rates). Calibration MAE = mean |predicted_failure_prob − ground_truth| from 256-sample Poisson MC.

| Pattern | Node | Dose (ph/nm²) | Mean FP | Max FP | Mean LER (nm) | Robustness | Bridge P | Break P | Defects (cm⁻²) | Cal. MAE | Time (s) |
|---------|------|---------------|---------|--------|--------------|------------|----------|---------|---------------|----------|----------|
| line/space | EUV N3 | 30 | 0.1139 | 0.5781 | 0.18 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0161 | 0.07 |
| line/space | EUV N7 | 40 | 0.0287 | 0.2500 | 0.10 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0089 | 0.02 |
| line/space | ArF 45nm | 60 | 0.0007 | 0.0312 | 0.01 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0009 | 0.01 |
| contact array | EUV N3 | 30 | 0.0406 | 0.5156 | 0.08 | 1.0000 | 0.0000 | 0.0000 | 2777481079101.6 | 0.0072 | 0.01 |
| contact array | EUV N7 | 40 | 0.0600 | 0.6875 | 0.10 | 1.0000 | 0.0000 | 0.0000 | 2822113037109.4 | 0.0090 | 0.01 |
| contact array | ArF 45nm | 60 | 0.0789 | 0.6094 | 0.15 | 1.0000 | 0.0000 | 0.0000 | 2495193481445.3 | 0.0135 | 0.02 |
| elbow | EUV N3 | 30 | 0.0100 | 0.5312 | 0.02 | 0.7344 | 0.0000 | 0.5312 | 777816772460.9 | 0.0020 | 0.01 |
| elbow | EUV N7 | 40 | 0.0188 | 0.6875 | 0.03 | 0.7578 | 0.0000 | 0.4844 | 775909423828.1 | 0.0025 | 0.01 |
| elbow | ArF 45nm | 60 | 0.0173 | 0.6094 | 0.03 | 0.7969 | 0.0000 | 0.4062 | 735473632812.5 | 0.0033 | 0.03 |
| dense random | EUV N3 | 30 | 0.0000 | 0.0156 | 0.00 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | 0.01 |
| dense random | EUV N7 | 40 | 0.0000 | 0.0000 | 0.00 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | 0.01 |
| dense random | ArF 45nm | 60 | 0.0000 | 0.0000 | 0.00 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | 0.01 |

## Dose-Response Sweep (line/space, EUV N7)

Generated with `python scripts/run_stochastic_benchmark.py`. All runs use `BayesianStochasticModel(mode='poisson')` with `n_mc_samples=64`, cross-validated against `compute_stochastic_robustness` (bridge/break) and `compute_stochastic_defect_classes` (imec per-class rates). Calibration MAE = mean |predicted_failure_prob − ground_truth| from 256-sample Poisson MC.

| Pattern | Node | Dose (ph/nm²) | Mean FP | Max FP | Mean LER (nm) | Robustness | Bridge P | Break P | Defects (cm⁻²) | Cal. MAE | Time (s) |
|---------|------|---------------|---------|--------|--------------|------------|----------|---------|---------------|----------|----------|
| line/space | EUV N7 | 10 | 0.1708 | 0.5625 | 0.33 | 0.8750 | 0.0000 | 0.2500 | 6103515625.0 | 0.0285 | 0.01 |
| line/space | EUV N7 | 30 | 0.0451 | 0.3125 | 0.13 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0116 | 0.02 |
| line/space | EUV N7 | 60 | 0.0220 | 0.2188 | 0.08 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0073 | 0.01 |
| line/space | EUV N7 | 100 | 0.0088 | 0.1406 | 0.04 | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0040 | 0.01 |