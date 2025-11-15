# PufferLib Core

Core kernels and utilities for the PufferLib reinforcement-learning toolkit.
This package provides the Torch C++/CUDA extensions that power high-throughput
vectorized environments used across the Metta stack.

## Requirements

- Python 3.8+
- PyTorch 2.9.0+cu130 from https://download.pytorch.org/whl/cu130 (ensures CUDA 13.0 build)
- A valid `CUDA_HOME` that points at your installed toolkit, e.g.
  `/usr/local/cuda-13.0`

## Building

The package always attempts to compile its native extension. If `nvcc` is on
`PATH`, the CUDA version is built; otherwise a CPU-only build is produced.

You can trigger a local build with:

```bash
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130 \\
  PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130 \\
  uv build packages/pufferlib-core
```

If you only need the Python APIs and do not have CUDA installed, temporarily
remove `nvcc` from your `PATH` or unset `CUDA_HOME` before installing so the
build falls back to the CPU extension. Likewise, if you cannot use the CUDA
13.0 wheels, drop the extra index variables so the CPU build is selected.
