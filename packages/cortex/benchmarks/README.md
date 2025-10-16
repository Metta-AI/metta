# Cortex Kernel Benchmarks

Performance benchmarks comparing Triton-accelerated kernels against pure PyTorch implementations.

## CLI Usage

All benchmarks are now launched through a single entry point:

```bash
uv run ./run.py --list          # show available benchmarks
uv run ./run.py rtu             # run all RTU configs
uv run ./run.py slstm --config 0 --config 3  # run selected configs only
uv run ./run.py mlstm --device cpu           # force CPU mode (skips Triton)
uv run ./run.py conv1d --warmup 10 --iterations 50
```

The runner handles:

- Automatic CUDA vs CPU device selection (override with `--device`)
- Warmup/iteration overrides (`--warmup`, `--iterations`)
- Per-config selection via repeated `--config` flags (indices shown in output)
- Uniform table formatting for PyTorch/Triton timings, speedups, and error reporting

Each benchmark module defines its own configuration grid but reuses the shared timing utilities and output formatting in `common.py`.

## Benchmarks

- `rtu` – Low-rank RTUCell (segmented scan and standard paths)
- `axons` – Streaming RTU diagonal cell
- `slstm` – Structured LSTM kernels
- `mlstm` – Multi-head LSTM kernels
- `lstm` – Vanilla LSTM (compares Triton vs cuDNN baseline)
- `conv1d` – Causal Conv1D kernel with per-timestep resets

Use `uv run ./run.py --list` for the authoritative list and descriptions.

## Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- Triton library

## Results Summary (NVIDIA L4)

| Kernel | Speedup Range | Best Config | Notes |
|--------|---------------|-------------|-------|
| RTU (low-rank) | 16.6x - 157.3x | (8, 1024, 64, 16, resets=0.1) @ 157.31x | ✅ Strong speedup; segmented scan excels on long seqs |
| sLSTM  | 23x - 68x     | (2, 4, 256, 64) @ 67.82x | ✅ Excellent speedup |
| mLSTM  | 3.5x - 12.3x  | (8, 8, 1024, 64) @ 12.28x | ✅ Strong speedup |
| LSTM   | 0.04x - 0.17x | PyTorch faster (uses cuDNN) | ⚠️ PyTorch optimized |
| Conv1D | 0.2x - 0.97x  | PyTorch faster (uses cuDNN) | ⚠️ PyTorch optimized |

**Note:** LSTM and Conv1D Triton kernels are designed for custom features (per-timestep resets) not available in cuDNN. For standard operations without resets, PyTorch's cuDNN backend is faster.
