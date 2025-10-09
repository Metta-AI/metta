# Cortex Kernel Benchmarks

Performance benchmarks comparing Triton-accelerated kernels against pure PyTorch implementations.

## Available Benchmarks

### RTU (low‑rank)
```bash
uv run python benchmarks/bench_rtu_triton_vs_pytorch.py
```
Performance varies with sequence length and hidden size. Triton benefits most for longer sequences and when per‑timestep resets are used (segmented scan path). The script reports per‑config speed and max output difference.

### sLSTM
```bash
uv run python benchmarks/bench_slstm_triton_vs_pytorch.py
```
**Performance:** 23x - 68x speedup (excellent performance across all configs)

### mLSTM
```bash
uv run python benchmarks/bench_mlstm_triton_vs_pytorch.py
```
**Performance:** 3.5x - 12.3x speedup (best at longer sequences)

### LSTM
```bash
uv run python benchmarks/bench_lstm_triton_vs_pytorch.py
```
**Performance:** PyTorch (cuDNN) is 3x - 25x faster. Triton kernel useful for custom reset patterns.

**Why Triton is still slower:**
- Each timestep recomputes the full recurrent GEMM from global memory; cuDNN keeps weights in on-chip caches and fuses the sequence.
- The kernel tiles only along the hidden dimension (`MATMUL_K_TILE=16`), which limits tensor-core usage and overall occupancy.
- Python dispatch launches the forward kernel once per sequence iteration, whereas cuDNN handles the whole sequence inside one launch.
- Mixed-precision support is conservative (fp32 accumulation), so cuDNN’s tensor-core kernels retain a throughput edge.

Closing these gaps would mean redesigning the Triton kernel (persistent tiles, better tensor-core tiling, fewer launches), which is feasible but a substantial engineering effort.

### Conv1D
```bash
uv run python benchmarks/bench_conv1d_triton_vs_pytorch.py
```
**Performance:** PyTorch (cuDNN) is typically faster. Triton kernel optimized for per-timestep reset case.

## Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- Triton library

## Results Summary (NVIDIA L4)

| Kernel | Speedup Range | Best Config | Notes |
|--------|---------------|-------------|-------|
| sLSTM  | 23x - 68x     | (2, 4, 256, 64) @ 67.82x | ✅ Excellent speedup |
| mLSTM  | 3.5x - 12.3x  | (8, 8, 1024, 64) @ 12.28x | ✅ Strong speedup |
| LSTM   | 0.04x - 0.17x | PyTorch faster (uses cuDNN) | ⚠️ PyTorch optimized |
| Conv1D | 0.2x - 0.97x  | PyTorch faster (uses cuDNN) | ⚠️ PyTorch optimized |

**Note:** LSTM and Conv1D Triton kernels are designed for custom features (per-timestep resets) not available in cuDNN. For standard operations without resets, PyTorch's cuDNN backend is faster.
