# Torch Profiler Summary


Run: `relh.machina_1.perf.1223.2`
Trace: `torch_profiles/trace_relh.machina_1.perf.1223.2_epoch_3.json.gz`
Source machine: `relh-sandbox-4` (AWS `g6.12xlarge`, L4 x4)
Captured epoch: 3
Parsed on: 2025-12-23

Notes:
- The profiler trace is large (~1GB compressed, ~20GB raw) and sums durations across many threads, so totals can exceed wall-clock time. Use rankings to identify repeated or sync-heavy ops rather than interpreting absolute totals.
- This analysis uses only `ph="X"` events (duration events) in the trace.

## Key signals (high confidence)

1) **Frequent CPU↔GPU sync points**
   - `aten::item` + `aten::_local_scalar_dense` dominate CPU op time.
   - CUDA runtime shows substantial `cudaStreamSynchronize` and `cudaDeviceSynchronize` time.

   These are strong indicators that the training loop (or loss/metrics) is frequently forcing synchronous GPU reads or blocking on GPU work.

2) **Heavy scatter/index and mutation ops**
   - `aten::index_put_`, `aten::_index_put_impl_`, `aten::index_copy_`, `aten::index_select` are high in the CPU op list.
   - This points to non-trivial gather/scatter logic on the hot path (likely in losses, experience assembly, or stats updates).

3) **Many allocations / fills**
   - `aten::empty`, `aten::fill_`, `aten::empty_like`, `aten::resize_`, `aten::empty_strided` show up frequently.
   - Suggests repeated allocation + fill cycles during training, which may indicate missing buffer reuse or avoidable tensor reconstruction.

4) **Kernel mix looks typical for ViT**
   - Attention kernels (fmha cutlass), layer norm, and GEMMs dominate GPU kernel time.
   - No obviously pathological kernel stands out on GPU; the bottleneck appears more CPU/synchronization/overhead-driven.

## Top categories by total duration (microseconds)

```
python_function          177206065380 us  (60841999 events)
cpu_op                      149937417 us  (3626286 events)
Trace                       109690762 us  (1 events)
user_annotation              67477322 us  (3071 events)
gpu_user_annotation          66945255 us  (2943 events)
kernel                       58315862 us  (1244446 events)
cuda_runtime                 56980980 us  (1726639 events)
gpu_memcpy                    1911629 us  (58244 events)
cuda_driver                    257728 us  (28014 events)
overhead                        55564 us  (293 events)
gpu_memset                       7548 us  (137673 events)
```

## Top CPU ops by total duration (microseconds)

```
aten::item                       29864755 us  (31119 calls)
aten::_local_scalar_dense        29811790 us  (31119 calls)
aten::index_put_                 12135324 us  (32944 calls)
aten::_index_put_impl_           12046407 us  (32944 calls)
aten::index_copy_                 7081324 us  (20540 calls)
aten::empty                       4976991 us  (368274 calls)
aten::fill_                       4129234 us  (508123 calls)
aten::linear                      3604317 us  (26473 calls)
aten::arange                      3003455 us  (66400 calls)
aten::addmm                       2437665 us  (19567 calls)
aten::to                          2080491 us  (191156 calls)
aten::resize_                     2014518 us  (60550 calls)
aten::_to_copy                    1978889 us  (42253 calls)
aten::index_select                1785048 us  (21691 calls)
aten::empty_like                  1617020 us  (89701 calls)
aten::copy_                       1576304 us  (101891 calls)
aten::max                         1392160 us  (38576 calls)
aten::mul                         1194562 us  (86106 calls)
aten::empty_strided               1125037 us  (60065 calls)
_RTUStreamDiagCUDASeqAllIn        1124225 us  (2302 calls)
aten::mm                          1047803 us  (12794 calls)
aten::min                          997033 us  (34096 calls)
aten::layer_norm                   972542 us  (11510 calls)
aten::remainder                    940577 us  (66152 calls)
aten::native_layer_norm            912084 us  (11510 calls)
Torch-Compiled Region: 0/0         879321 us  (1023 calls)
aten::div_                         821791 us  (65890 calls)
aten::matmul                       739197 us  (6906 calls)
aten::clone                        737821 us  (18546 calls)
cortex::rtu_seq_allin_forward       668371 us  (2302 calls)
```

## Top CUDA runtime ops by total duration (microseconds)

```
cudaStreamSynchronize          29444736 us  (37770 calls)
cudaDeviceSynchronize          20356467 us  (129 calls)
cudaLaunchKernel                6045460 us  (1216432 calls)
cudaMemcpyAsync                  682838 us  (58244 calls)
cudaMemsetAsync                  390794 us  (152764 calls)
cudaMemGetInfo                    32561 us  (1648 calls)
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags        13554 us  (33202 calls)
cudaOccupancyMaxActiveBlocksPerMultiprocessor         6523 us  (5516 calls)
cudaStreamIsCapturing              2511 us  (5640 calls)
cudaEventQuery                     1652 us  (1066 calls)
cudaDeviceGetAttribute              931 us  (60580 calls)
cudaStreamWaitEvent                 669 us  (1282 calls)
cudaFuncGetAttributes               555 us  (256 calls)
cudaEventRecord                     518 us  (1155 calls)
cudaEventRecordWithFlags            417 us  (651 calls)
cudaThreadExchangeStreamCaptureMode          321 us  (1330 calls)
cudaPeekAtLastError                 221 us  (147924 calls)
cudaFuncSetAttribute                163 us  (256 calls)
cudaStreamGetCaptureInfo_v2           47 us  (385 calls)
cudaEventSynchronize                 22 us  (16 calls)
cudaEventElapsedTime                 17 us  (8 calls)
cudaGetFuncBySymbol                   3 us  (385 calls)
```

## Top CUDA kernels by total duration (microseconds)

```
void at::native::(anonymous namespace)::RowwiseMomentsCUDAKernel<float, float, false>(long, float, float const*, float*, float*)     11511091 us  (1151 calls)
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params)     10135733 us  (2302 calls)
fmha_cutlassB_f32_aligned_64x64_k32_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 32, false>::Params)      7958451 us  (256 calls)
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})      2982790 us  (28038 calls)
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)      2839441 us  (179120 calls)
ampere_sgemm_64x64_tn                                                                 2600165 us  (4860 calls)
void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)      2461673 us  (14330 calls)
void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float const*, float*, float*, float*)      1396981 us  (10359 calls)
void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)      1287474 us  (2302 calls)
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)      1074190 us  (6907 calls)
void at::native::(anonymous namespace)::LayerNormForwardCUDAKernel<float, float, false>(long, float const*, float const*, float const*, float const*, float const*, float*)       874151 us  (1151 calls)
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align4::Params)       841498 us  (2304 calls)
void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)       698860 us  (26427 calls)
void at_cuda_detail::cub::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::DeviceRadixSortPolicy<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy900, false, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at::cuda::cub::detail::OpaqueType<8>*, at::cuda::cub::detail::OpaqueType<8> const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t)       645212 us  (67040 calls)
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4::Params)       635827 us  (12789 calls)
void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type const*, long const*, long const*, long, long)       633619 us  (128 calls)
void at::native::vectorized_elementwise_kernel<4, at::native::GeluBackwardCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float, float)#1}, std::array<char*, 3ul> >(int, at::native::GeluBackwardCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float, float)#1}, std::array<char*, 3ul>)       621948 us  (256 calls)
ampere_sgemm_64x64_nn                                                                  614650 us  (512 calls)
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align1::Params)       605213 us  (256 calls)
void at::native::(anonymous namespace)::layer_norm_grad_input_kernel<float, float, false>(float const*, float const*, float const*, float const*, float const*, float*, int)       583806 us  (128 calls)
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1})       556853 us  (1279 calls)
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4::Params)       519782 us  (2046 calls)
ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)              475946 us  (385 calls)
void at::native::(anonymous namespace)::layer_norm_grad_input_kernel_vectorized<float, float, false>(float const*, float const*, float const*, float const*, float const*, float*, int)       453347 us  (1152 calls)
void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>)       357990 us  (3328 calls)
void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 1u, 32u, true, true, false>(long, long, float const*, float const*, float const*, float const*, float*, float*)       315129 us  (640 calls)
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4::Params)       287636 us  (256 calls)
void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::array<char*, 1ul> >(int, at::native::FillFunctor<long>, std::array<char*, 1ul>)       280644 us  (240695 calls)
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4::Params)       267049 us  (384 calls)
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1})       234508 us  (1919 calls)
```

## Likely bottleneck candidates (hypotheses)

These are *probable* issues suggested by the trace; they should be confirmed by code inspection and/or targeted profiling.

1) **Scalar extraction in hot path**
   - The `aten::item` / `_local_scalar_dense` pattern typically comes from code like `loss.item()`, `float(tensor)`, `tensor.cpu().numpy()` inside loops or per‑minibatch logging.
   - This forces a GPU→CPU sync each time and can dominate training time.

2) **Scatter/gather-heavy updates**
   - Frequent `index_put_` / `index_copy_` / `index_select` can indicate costly tensor assembly in losses or experience storage (e.g., per‑step per‑agent writes).
   - This could explain why “train” time is dominating instead of rollout.

3) **Repeated allocations / buffer churn**
   - Very high counts of `empty` / `fill_` / `empty_like` / `resize_` suggests many tensors are allocated each step or minibatch.
   - Potential wins from caching and reusing buffers (or reworking loops to avoid small allocations).

4) **Synchronization overhead**
   - Significant `cudaStreamSynchronize` and `cudaDeviceSynchronize` time indicates blocking synchronization likely caused by scalar reads, debug checks, or synchronous logging.

## Suggested next steps (for follow‑up profiling)

1) **Audit for scalar syncs**
   - Find and batch/disable `.item()`, `float(tensor)`, `tensor.cpu().numpy()` usage in the training/optimizer/loss logging path.

2) **Inspect index/scatter paths in losses**
   - Look for loops and `index_put_` in loss code and in experience/advantage computation. Consider vectorized alternatives.

3) **Reduce allocations in training loop**
   - Reuse buffers where possible. Pay attention to per‑minibatch TensorDict construction.

4) **Add lightweight profiler schedule**
   - Use a short torch.profiler schedule to avoid multi‑GB traces and reduce capture overhead.
