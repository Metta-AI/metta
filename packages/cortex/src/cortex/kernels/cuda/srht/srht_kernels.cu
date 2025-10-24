// srht_kernels.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel: per row FWHT with sign and permutation
// x: [N,H], signs: [H] (+/-1), perm: [H] (int64 indices mapping k -> src index)
// y: [N,H]

template <typename scalar_t>
__global__ void srht_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ signs,
    const int64_t* __restrict__ perm,
    scalar_t* __restrict__ y,
    int N, int H, bool normalize)
{
  extern __shared__ unsigned char smem[];
  scalar_t* buf = reinterpret_cast<scalar_t*>(smem);

  int row = blockIdx.x;
  if (row >= N) return;
  const scalar_t* xrow = x + row * H;
  scalar_t* yrow = y + row * H;

  // load with permutation and signs into shared buffer
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    int64_t src = perm ? perm[k] : k;
    scalar_t v = xrow[src] * signs[k];
    buf[k] = v;
  }
  __syncthreads();

  // iterative FWHT in shared memory
  for (int h = 1; h < H; h <<= 1) {
    int step = h << 1;
    for (int k = threadIdx.x; k < H; k += blockDim.x) {
      int block = (k / step) * step;
      int j = k % step;
      if (j < h) {
        int i1 = block + j;
        int i2 = i1 + h;
        scalar_t a = buf[i1];
        scalar_t b = buf[i2];
        buf[i1] = a + b;
        buf[i2] = a - b;
      }
    }
    __syncthreads();
  }

  scalar_t scale = normalize ? static_cast<scalar_t>(1.0 / sqrt((double)H)) : static_cast<scalar_t>(1.0);
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    yrow[k] = buf[k] * scale;
  }
}

template <typename scalar_t>
__global__ void srht_backward_kernel(
    const scalar_t* __restrict__ gy, // [N,H]
    const scalar_t* __restrict__ signs,
    const int64_t* __restrict__ perm,
    scalar_t* __restrict__ gx,
    int N, int H, bool normalize)
{
  extern __shared__ unsigned char smem[];
  scalar_t* buf = reinterpret_cast<scalar_t*>(smem);

  int row = blockIdx.x;
  if (row >= N) return;
  const scalar_t* gyrow = gy + row * H;
  scalar_t* gxrow = gx + row * H;

  // load grad into buffer
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    buf[k] = gyrow[k];
  }
  __syncthreads();

  // FWHT (self-inverse up to scale)
  for (int h = 1; h < H; h <<= 1) {
    int step = h << 1;
    for (int k = threadIdx.x; k < H; k += blockDim.x) {
      int block = (k / step) * step;
      int j = k % step;
      if (j < h) {
        int i1 = block + j;
        int i2 = i1 + h;
        scalar_t a = buf[i1];
        scalar_t b = buf[i2];
        buf[i1] = a + b;
        buf[i2] = a - b;
      }
    }
    __syncthreads();
  }

  scalar_t scale = normalize ? static_cast<scalar_t>(1.0 / sqrt((double)H)) : static_cast<scalar_t>(1.0);
  // scatter back with signs and inverse permutation: gx[perm[k]] += signs[k]*scale*buf[k]
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    int64_t dst = perm ? perm[k] : k;
    scalar_t v = buf[k] * scale * signs[k];
    gxrow[dst] = v; // no accumulation needed since mapping is one-to-one
  }
}

std::vector<at::Tensor> srht_forward_cuda(at::Tensor x, at::Tensor signs, c10::optional<at::Tensor> perm_opt, bool normalize) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be CUDA contiguous [B,T,H]");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(signs.is_cuda() && signs.is_contiguous(), "signs must be CUDA contiguous");
  at::Tensor perm;
  const int64_t* perm_ptr = nullptr;
  if (perm_opt.has_value()) {
    perm = *perm_opt;
    TORCH_CHECK(perm.is_cuda() && perm.is_contiguous() && perm.scalar_type() == at::kLong, "perm must be CUDA int64");
    perm_ptr = perm.data_ptr<int64_t>();
  }

  int B = x.size(0), T = x.size(1), H = x.size(2);
  auto opts = x.options();
  auto y = at::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  int N = B * T;
  dim3 grid(N);
  int threads = 1;
  while (threads * 2 <= H && threads < 1024) threads <<= 1;
  dim3 block(threads);
  size_t shmem = sizeof(float) * H;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "srht_forward_cuda", [&] {
    srht_forward_kernel<scalar_t><<<grid, block, shmem, stream>>>(
        x.data_ptr<scalar_t>(), signs.data_ptr<scalar_t>(), perm_ptr, y.data_ptr<scalar_t>(), N, H, normalize);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y};
}

std::vector<at::Tensor> srht_backward_cuda(at::Tensor gy, at::Tensor signs, c10::optional<at::Tensor> perm_opt, bool normalize) {
  TORCH_CHECK(gy.is_cuda() && gy.is_contiguous(), "gy must be CUDA contiguous [B,T,H]");
  TORCH_CHECK(gy.scalar_type() == at::kFloat, "gy must be float32");
  TORCH_CHECK(signs.is_cuda() && signs.is_contiguous(), "signs must be CUDA contiguous");
  at::Tensor perm;
  const int64_t* perm_ptr = nullptr;
  if (perm_opt.has_value()) {
    perm = *perm_opt;
    TORCH_CHECK(perm.is_cuda() && perm.is_contiguous() && perm.scalar_type() == at::kLong, "perm must be CUDA int64");
    perm_ptr = perm.data_ptr<int64_t>();
  }
  int B = gy.size(0), T = gy.size(1), H = gy.size(2);
  auto gx = at::empty_like(gy);
  auto stream = at::cuda::getCurrentCUDAStream();
  int N = B * T;
  dim3 grid(N);
  int threads = 1;
  while (threads * 2 <= H && threads < 1024) threads <<= 1;
  dim3 block(threads);
  size_t shmem = sizeof(float) * H;
  AT_DISPATCH_FLOATING_TYPES(gy.scalar_type(), "srht_backward_cuda", [&] {
    srht_backward_kernel<scalar_t><<<grid, block, shmem, stream>>>(
        gy.data_ptr<scalar_t>(), signs.data_ptr<scalar_t>(), perm_ptr, gx.data_ptr<scalar_t>(), N, H, normalize);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gx};
}

