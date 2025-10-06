#include <torch/extension.h>
#include <type_traits>
#include <vector>

namespace agalite_cuda {

namespace {

template <typename scalar_t>
__global__ void discounted_sum_forward_kernel(const scalar_t* __restrict__ start_state,
                                              const scalar_t* __restrict__ x,
                                              const scalar_t* __restrict__ discounts,
                                              scalar_t* __restrict__ output,
                                              int64_t T,
                                              int64_t N) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  using acc_t = typename std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
  acc_t prev = static_cast<acc_t>(start_state[idx]);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t offset = t * N + idx;
    const acc_t discount = static_cast<acc_t>(discounts[offset]);
    const acc_t value = static_cast<acc_t>(x[offset]);
    prev = discount * prev + value;
    output[offset] = static_cast<scalar_t>(prev);
  }
}

template <typename scalar_t>
__global__ void discounted_sum_backward_kernel(const scalar_t* __restrict__ grad_out,
                                               const scalar_t* __restrict__ discounts,
                                               const scalar_t* __restrict__ output,
                                               const scalar_t* __restrict__ start_state,
                                               scalar_t* __restrict__ grad_start,
                                               scalar_t* __restrict__ grad_x,
                                               scalar_t* __restrict__ grad_discounts,
                                               int64_t T,
                                               int64_t N) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  using acc_t = typename std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
  acc_t accum = static_cast<acc_t>(0);
  for (int64_t t = T - 1; t >= 0; --t) {
    const int64_t offset = t * N + idx;
    const acc_t grad_total = static_cast<acc_t>(grad_out[offset]) + accum;
    grad_x[offset] = static_cast<scalar_t>(grad_total);

    acc_t prev_output;
    if (t == 0) {
      prev_output = static_cast<acc_t>(start_state[idx]);
    } else {
      prev_output = static_cast<acc_t>(output[(t - 1) * N + idx]);
    }
    grad_discounts[offset] = static_cast<scalar_t>(grad_total * prev_output);
    const acc_t discount = static_cast<acc_t>(discounts[offset]);
    accum = grad_total * discount;
  }
  grad_start[idx] = static_cast<scalar_t>(accum);
}

}  // namespace

torch::Tensor discounted_sum_forward(torch::Tensor start_state,
                                      torch::Tensor x,
                                      torch::Tensor discounts) {
  auto output = torch::empty_like(x);
  const auto T = x.size(0);
  const auto N = x.size(1);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "discounted_sum_forward_cuda", [&] {
        discounted_sum_forward_kernel<scalar_t><<<blocks, threads>>>(
            start_state.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            discounts.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            T,
            N);
      });

  return output;
}

std::vector<torch::Tensor> discounted_sum_backward(const torch::Tensor& grad_out,
                                                   const torch::Tensor& discounts,
                                                   const torch::Tensor& output,
                                                   const torch::Tensor& start_state) {
  auto grad_start = torch::zeros_like(start_state);
  auto grad_x = torch::empty_like(grad_out);
  auto grad_discounts = torch::empty_like(discounts);

  const auto T = grad_out.size(0);
  const auto N = grad_out.size(1);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "discounted_sum_backward_cuda", [&] {
        discounted_sum_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            discounts.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            start_state.data_ptr<scalar_t>(),
            grad_start.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_discounts.data_ptr<scalar_t>(),
            T,
            N);
      });

  std::vector<torch::Tensor> grads;
  grads.reserve(3);
  grads.push_back(grad_start);
  grads.push_back(grad_x);
  grads.push_back(grad_discounts);
  return grads;
}

}  // namespace agalite_cuda
