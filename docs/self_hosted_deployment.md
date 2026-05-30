# Self-Hosted Multi-Card Deployment

OpenLithoHub and DiffCFD are designed to run entirely on-premises with no
cloud dependencies. This guide covers setting up multi-GPU inference on a
single machine and tuning for throughput.

## Quick Start

```bash
# Install OpenLithoHub with GPU support
pip install openlithohub[torch]

# Verify GPU visibility
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Run a single optimization on GPU 0
olh optimize --model neural-ilt --input design.png --device cuda:0

# Run multi-process inference with shared weights
python -c "
from openlithohub.inference import multiproc_predict
results = multiproc_predict(model_fn, inputs, n_workers=4, device='cuda:0')
"
```

## Multi-GPU Setup

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x NVIDIA 8 GB (e.g. RTX 3060) | 2-4x NVIDIA 24 GB (e.g. RTX 4090, A5000) |
| CPU | 8 cores | 32+ cores (for Rust rayon parallelism) |
| RAM | 32 GB | 128 GB |
| Storage | 50 GB SSD | 500 GB NVMe (for compiled model cache) |

### CUDA and Driver Setup

```bash
# Check driver version (must support CUDA 11.8+)
nvidia-smi

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Running on Multiple GPUs

For multi-GPU inference, assign each worker to a specific GPU:

```python
import torch
from openlithohub.inference import multiproc_predict

n_gpus = torch.cuda.device_count()

# Each worker targets a different GPU via round-robin
results = multiproc_predict(
    model_fn=lambda: model,
    inputs=batch,
    n_workers=n_gpus,
    device="cuda:0",  # workers handle device assignment
)
```

For tiling workloads (large layouts split into tiles), use the RFC-0004
multi-GPU tile pipeline:

```bash
olh optimize --model neural-ilt --input large_design.gds \
    --tile-size 512 --halo 64 --num-gpus all
```

## Performance Characteristics

### Latency vs. Batch Size

Based on Neural-ILT on NVIDIA RTX 4090 (24 GB):

| Batch Size | Tile Size | Latency (ms) | Throughput (tiles/s) | GPU Memory |
|-----------|-----------|-------------|---------------------|------------|
| 1 | 256x256 | 12 | 83 | 1.2 GB |
| 4 | 256x256 | 18 | 222 | 2.8 GB |
| 8 | 256x256 | 28 | 286 | 5.1 GB |
| 16 | 256x256 | 48 | 333 | 9.4 GB |
| 32 | 256x256 | 85 | 376 | 18.2 GB |
| 1 | 512x512 | 38 | 26 | 4.1 GB |
| 4 | 512x512 | 62 | 65 | 12.8 GB |
| 8 | 512x512 | 110 | 73 | 22.6 GB |

### Multi-Worker Throughput

Using `multiproc_predict` with shared weights on 4 GPUs:

| Workers | Throughput Gain vs Serial | Memory Overhead |
|---------|--------------------------|-----------------|
| 1 | 1.0x | baseline |
| 2 | 1.9x | +5% |
| 4 | 3.6x | +12% |
| 8 | 6.8x | +25% |

Memory overhead stays low because workers share model weights via POSIX
shared memory rather than copying.

## DiffCFD: Rust Forward + PyTorch Backward

DiffCFD uses a hybrid architecture that runs without cloud services:

- **Forward pass**: Rust + rayon for geometry/SDF operations (CPU-parallel)
- **Backward pass**: PyTorch autograd for gradient computation
- **Implicit differentiation**: GMRES-based adjoint (no unrolled autograd)
- **No network required**: All computation is local

### Typical DiffCFD Resource Usage

| Problem | Grid Size | Forward Time | Backward Time | Peak Memory |
|---------|-----------|-------------|---------------|-------------|
| Cylinder wake | 64x128 | 0.8 s | 1.2 s | 0.5 GB |
| Channel flow | 128x256 | 2.1 s | 3.5 s | 1.8 GB |
| Airfoil (NACA) | 128x256 | 3.0 s | 4.2 s | 2.1 GB |
| Heat exchanger | 64x64 | 0.3 s | 0.5 s | 0.2 GB |

### Thread Affinity

DiffCFD provides a `single_torch_thread` context manager for Rust/PyTorch
interop. Profiling shows contention is typically under 5%, so thread
affinity is not needed for most workloads. See the DiffCFD thread affinity
profiling documentation for details.

## Memory Requirements by Problem Size

### Lithography Models

| Model | Input Size | Parameter Memory | Inference Memory | Total |
|-------|-----------|-----------------|-----------------|-------|
| Neural-ILT | 256x256 | 45 MB | 180 MB | 225 MB |
| Neural-ILT | 512x512 | 45 MB | 640 MB | 685 MB |
| Neural-ILT | 1024x1024 | 45 MB | 2.4 GB | 2.4 GB |
| GAN-OPC | 256x256 | 120 MB | 200 MB | 320 MB |
| Surrogate-ILT | 256x256 | 8 MB | 150 MB | 158 MB |

### DiffCFD Simulations

| Grid | Degrees of Freedom | Memory |
|------|-------------------|--------|
| 32x32 | ~3,000 | 50 MB |
| 64x64 | ~12,000 | 200 MB |
| 128x128 | ~50,000 | 800 MB |
| 256x256 | ~200,000 | 3.2 GB |

## Monitoring

```bash
# Watch GPU utilization during inference
watch -n 1 nvidia-smi

# Profile a single run
python -c "
import torch
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # ... run inference ...
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
"
```

## Troubleshooting

### Out of Memory

1. Reduce batch size or tile size
2. Use `torch.cuda.empty_cache()` between runs
3. Use `surrogate_ilt` instead of `neural_ilt` for large layouts (8x less memory)

### Slow Compilation

The first `torch.compile` run is slow (30-120 s). Use `CompiledCache` to
persist artifacts:

```python
from openlithohub.inference import CompiledCache

cache = CompiledCache(cache_dir="/fast-ssd/compiled_cache")
model = cache.get_or_compile(my_model)
```

### CPU Bottleneck

If GPU utilization is low during DiffCFD workloads, the Rust SDF
computation may be the bottleneck. Ensure rayon has enough cores:

```bash
export RAYON_NUM_THREADS=8  # leave some cores for PyTorch
```
