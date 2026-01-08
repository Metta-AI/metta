# Torch Profiler Summary

Run: `relh.machina_1.perf.1223.2` Trace: `torch_profiles/trace_relh.machina_1.perf.1223.2_epoch_1.json.gz` Source
machine: `relh-sandbox-4` (AWS `g6.12xlarge`, L4 x4) Captured epoch: 1 Parsed on: 2025-12-23

Notes:

- The profiler trace is large (~1GB compressed, ~20GB raw) and sums durations across many threads, so totals can exceed
  wall-clock time. Use rankings to identify repeated or sync-heavy ops rather than interpreting absolute totals.
- This analysis uses only `ph="X"` events (duration events) in the trace.

## Key signals (high confidence)

1. **Frequent CPU↔GPU sync points**
   - `aten::item` + `aten::_local_scalar_dense` dominate CPU op time (31k+ calls).
   - CUDA runtime shows substantial `cudaStreamSynchronize` and `cudaDeviceSynchronize` time.

   These are strong indicators that the training loop (or loss/metrics) is frequently forcing synchronous GPU reads or
   blocking on GPU work.

2. **Heavy scatter/index and mutation ops**
   - `aten::index_put_`, `aten::_index_put_impl_`, `aten::index_copy_`, `aten::index_select` are high in the CPU op
     list.
   - This points to non-trivial gather/scatter logic on the hot path (likely in losses, experience assembly, or stats
     updates).

3. **Many allocations / fills**
   - `aten::empty`, `aten::fill_`, `aten::empty_like`, `aten::resize_`, `aten::empty_strided` show up frequently.
   - Suggests repeated allocation + fill cycles during training, which may indicate missing buffer reuse or avoidable
     tensor reconstruction.

4. **Kernel mix looks typical for ViT**
   - Attention kernels (fmha cutlass), layer norm, and GEMMs dominate GPU kernel time.
   - No obviously pathological kernel stands out on GPU; the bottleneck appears more
     CPU/synchronization/overhead-driven.

## Top categories by total duration (microseconds)

```
python_function     171,713,987,849 us  (60,817,676 events)
cpu_op                  142,540,656 us  (3,625,626 events)
Trace                    104,757,748 us  (1 event)
user_annotation           63,176,361 us  (3,071 events)
gpu_user_annotation       63,209,543 us  (2,943 events)
kernel                    54,398,835 us  (1,244,266 events)
cuda_runtime              52,834,753 us  (1,726,375 events)
gpu_memcpy                 1,704,105 us  (58,244 events)
cuda_driver                  257,650 us  (28,022 events)
overhead                     245,612 us  (296 events)
```

## Top CPU ops by total duration (microseconds)

```
aten::item                 26,981,357 us  (31,119 calls)
aten::_local_scalar_dense  26,930,593 us  (31,119 calls)
aten::index_put_           11,928,776 us  (32,944 calls)
aten::_index_put_impl_     11,838,839 us  (32,944 calls)
aten::index_copy_           6,899,314 us  (20,540 calls)
aten::empty                 4,874,618 us  (368,214 calls)
aten::fill_                 4,123,388 us  (508,003 calls)
aten::linear                3,376,601 us  (26,473 calls)
aten::arange                2,906,227 us  (66,400 calls)
aten::addmm                 2,280,911 us  (19,567 calls)
aten::to                    2,052,367 us  (191,156 calls)
aten::_to_copy              1,940,702 us  (42,253 calls)
aten::resize_               1,933,641 us  (60,550 calls)
aten::index_select          1,628,850 us  (21,691 calls)
aten::empty_like            1,591,069 us  (89,701 calls)
aten::copy_                 1,546,398 us  (101,831 calls)
aten::max                   1,338,736 us  (38,576 calls)
aten::mul                   1,203,786 us  (86,106 calls)
aten::empty_strided         1,135,762 us  (60,065 calls)
aten::mm                    1,015,818 us  (12,794 calls)
aten::min                     981,976 us  (34,096 calls)
aten::layer_norm              930,193 us  (11,510 calls)
aten::remainder               896,221 us  (66,152 calls)
aten::native_layer_norm       873,117 us  (11,510 calls)
```

## Top CUDA runtime ops by total duration (microseconds)

```
cudaStreamSynchronize      26,580,708 us  (37,770 calls)
cudaDeviceSynchronize      19,202,844 us  (129 calls)
cudaLaunchKernel            5,940,338 us  (1,216,244 calls)
cudaMemcpyAsync               668,042 us  (58,244 calls)
cudaMemsetAsync               378,014 us  (152,764 calls)
```

## Top CUDA kernels by total duration (microseconds)

```
RowwiseMomentsCUDAKernel<float>     10,063,760 us  (1,151 calls)
fmha_cutlassF_f32_aligned_64x64      9,667,170 us  (2,302 calls)
fmha_cutlassB_f32_aligned_64x64      7,558,314 us  (256 calls)
ampere_sgemm_64x64_tn                2,352,338 us  (4,860 calls)
vectorized_layer_norm_kernel         1,378,673 us  (10,359 calls)
GeluCUDAKernelImpl                   1,287,868 us  (2,302 calls)
...
```

## Likely bottleneck candidates (hypotheses)

These are _probable_ issues suggested by the trace; they should be confirmed by code inspection and/or targeted
profiling.

1. **Scalar extraction in hot path**
   - The `aten::item` / `_local_scalar_dense` pattern typically comes from code like `loss.item()`, `float(tensor)`,
     `tensor.cpu().numpy()` inside loops or per‑minibatch logging.
   - This forces a GPU→CPU sync each time and can dominate training time.

2. **Scatter/gather-heavy updates**
   - Frequent `index_put_` / `index_copy_` / `index_select` can indicate costly tensor assembly in losses or experience
     storage (e.g., per‑step per‑agent writes).
   - This could explain why “train” time is dominating instead of rollout.

3. **Repeated allocations / buffer churn**
   - Very high counts of `empty` / `fill_` / `empty_like` / `resize_` suggests many tensors are allocated each step or
     minibatch.
   - Potential wins from caching and reusing buffers (or reworking loops to avoid small allocations).

4. **Synchronization overhead**
   - Significant `cudaStreamSynchronize` and `cudaDeviceSynchronize` time indicates blocking synchronization likely
     caused by scalar reads, debug checks, or synchronous logging.

## Suggested next steps (for follow‑up profiling)

1. **Audit for scalar syncs**
   - Find and batch/disable `.item()`, `float(tensor)`, `tensor.cpu().numpy()` usage in the training/optimizer/loss
     logging path.

2. **Inspect index/scatter paths in losses**
   - Look for loops and `index_put_` in loss code and in experience/advantage computation. Consider vectorized
     alternatives.

3. **Reduce allocations in training loop**
   - Reuse buffers where possible. Pay attention to per‑minibatch TensorDict construction.

4. **Add lightweight profiler schedule**
   - Use a short torch.profiler schedule to avoid multi‑GB traces and reduce capture overhead.

## Additional notes

- A second trace exists for epoch 3 (`trace_relh.machina_1.perf.1223.2_epoch_3.json.gz`), but it has not been analyzed
  yet. If needed, we can repeat the summary on that file to verify consistency.
