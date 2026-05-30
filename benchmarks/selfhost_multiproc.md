# Multi-Worker Inference Benchmark

Date: 2026-05-30 09:22 UTC
Inputs: 32 tensors of shape (1, 1, 64, 64)
Warmup runs: 1, Timed runs: 3
Device: CPU

## Throughput & Latency

| Workers | Wall Time (avg, s) | Wall Time (min, s) | Throughput (items/s) | Peak Memory (MB) |
|---------|--------------------|--------------------|----------------------|------------------|
| 1 | 0.0102 | 0.0098 | 3146.30 | 0.014 |

## Numerical Consistency (vs serial)

| Workers | Consistent | Max Abs Diff |
|---------|------------|-------------|
| 1 | Yes | 0.0 |
