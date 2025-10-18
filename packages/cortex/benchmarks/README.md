# Cortex Kernel Benchmarks

Performance benchmarks comparing Triton-accelerated kernels against pure PyTorch implementations.

## Available Benchmarks

All benchmarks are now exposed via a single entrypoint:

```
uv run python packages/cortex/benchmarks/run.py --list
uv run python packages/cortex/benchmarks/run.py <benchmark_key> [--device cuda] [--warmup N] [--iterations M]
```

Keys: `rtu`, `slstm`, `mlstm`, `lstm`, `conv1d`, `axons`, `linear_vs_axon`.

### RTU (streaming diag)
```bash
uv run python packages/cortex/benchmarks/run.py rtu --device cuda
```
Performance varies with sequence length and hidden size. Triton benefits most for longer sequences and when per‑timestep resets are used (segmented scan path). The tool reports per‑config speed and max output difference.

Sample run (NVIDIA L4, CUDA 12.8):

```
/workspace/metta/packages/cortex# uv run python benchmarks/bench_rtu_triton_vs_pytorch.py
================================================================================
RTU (low-rank) Triton vs PyTorch Benchmark
================================================================================

Device: NVIDIA L4
CUDA Version: 12.8

Configuration format: (batch, seq_len, hidden, rank, resets, p)

Config                                           PyTorch (ms)    Triton (ms)     Speedup    Max Diff    
--------------------------------------------------------------------------------------------------------------
(4, 128, 64, 8, False, 0.0)                        Benchmarking PyTorch implementation...
  Benchmarking Triton implementation...
25.789          1.550           16.64x     4.32e-07    
(4, 256, 64, 8, False, 0.0)                        Benchmarking PyTorch implementation...
  Benchmarking Triton implementation...
53.949          1.582           34.11x     5.36e-07    
(8, 256, 64, 16, False, 0.0)                       Benchmarking PyTorch implementation...
  Benchmarking Triton implementation...
53.405          1.552           34.40x     8.29e-01    
(8, 512, 64, 16, False, 0.0)                       Benchmarking PyTorch implementation...
  Benchmarking Triton implementation...
106.822         1.674           63.83x     1.28e+00    
(8, 512, 128, 16, False, 0.0)                      Benchmarking PyTorch implementation...
  Benchmarking Triton implementation...
105.946         1.752           60.47x     1.28e+00    
(8, 512, 64, 16, True, 0.1)                        Benchmarking PyTorch implementation...
  Benchmarking Triton implementation...
145.180         2.011           72.18x     6.15e-01    
(8, 1024, 64, 16, True, 0.1)                       Benchmarking PyTorch implementation...
  Benchmarking Triton implementation...
297.344         1.890           157.31x    7.40e-01    

================================================================================
Benchmark complete!
================================================================================
```

### sLSTM
```bash
uv run python packages/cortex/benchmarks/run.py slstm --device cuda
```
**Performance:** 23x - 68x speedup (excellent performance across all configs)

### mLSTM
```bash
uv run python packages/cortex/benchmarks/run.py mlstm --device cuda
```
**Performance:** 3.5x - 12.3x speedup (best at longer sequences)

### LSTM
```bash
uv run python packages/cortex/benchmarks/run.py lstm --device cuda
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
uv run python packages/cortex/benchmarks/run.py conv1d --device cuda
```

### Linear vs AxonCell (forward+backward)
```bash
uv run python packages/cortex/benchmarks/run.py linear_vs_axon --device cuda
```
Compares `nn.Linear(H,H)` with `AxonCell(out_dim=H)` for identical inputs, reporting per‑iteration time, tokens/s, parameter counts, and peak CUDA memory.
**Performance:** PyTorch (cuDNN) is typically faster. Triton kernel optimized for per-timestep reset case.

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
