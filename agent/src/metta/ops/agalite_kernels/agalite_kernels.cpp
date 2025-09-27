#include <ATen/Parallel.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <torch/extension.h>
#include <type_traits>

namespace {

inline void check_inputs(const torch::Tensor& start_state,
                         const torch::Tensor& x,
                         const torch::Tensor& discounts) {
  TORCH_CHECK(x.dim() == 2, "x must be 2-D (T, N)");
  TORCH_CHECK(discounts.sizes() == x.sizes(),
              "discounts must have the same shape as x");
  TORCH_CHECK(start_state.dim() == 1, "start_state must be 1-D after flattening");
  TORCH_CHECK(start_state.size(0) == x.size(1),
              "start_state must match inner dimension of x");
}

}  // namespace

namespace cpu {

torch::Tensor discounted_sum_forward(torch::Tensor start_state,
                                      torch::Tensor x,
                                      torch::Tensor discounts) {
  check_inputs(start_state, x, discounts);
  auto output = torch::empty_like(x);

  const auto T = x.size(0);
  const auto N = x.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "discounted_sum_forward_cpu", [&] {
        const auto* start_ptr = start_state.data_ptr<scalar_t>();
        const auto* x_ptr = x.data_ptr<scalar_t>();
        const auto* discounts_ptr = discounts.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();

        at::parallel_for(0, N, 1024, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            using acc_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
            acc_t prev = static_cast<acc_t>(start_ptr[idx]);
            for (int64_t t = 0; t < T; ++t) {
              const int64_t offset = t * N + idx;
              const acc_t discount = static_cast<acc_t>(discounts_ptr[offset]);
              const acc_t value = static_cast<acc_t>(x_ptr[offset]);
              prev = discount * prev + value;
              out_ptr[offset] = static_cast<scalar_t>(prev);
            }
          }
        });
      });

  return output;
}

std::vector<torch::Tensor> discounted_sum_backward(const torch::Tensor& grad_out,
                                                   const torch::Tensor& discounts,
                                                   const torch::Tensor& output,
                                                   const torch::Tensor& start_state) {
  check_inputs(start_state, grad_out, discounts);

  auto grad_start = torch::zeros_like(start_state);
  auto grad_x = torch::empty_like(grad_out);
  auto grad_discounts = torch::empty_like(discounts);

  const auto T = grad_out.size(0);
  const auto N = grad_out.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "discounted_sum_backward_cpu", [&] {
        const auto* grad_ptr = grad_out.data_ptr<scalar_t>();
        const auto* discounts_ptr = discounts.data_ptr<scalar_t>();
        const auto* output_ptr = output.data_ptr<scalar_t>();
        const auto* start_ptr = start_state.data_ptr<scalar_t>();
        auto* grad_start_ptr = grad_start.data_ptr<scalar_t>();
        auto* grad_x_ptr = grad_x.data_ptr<scalar_t>();
        auto* grad_discount_ptr = grad_discounts.data_ptr<scalar_t>();

        at::parallel_for(0, N, 1024, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            using acc_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
            acc_t accum = static_cast<acc_t>(0);
            for (int64_t t = T - 1; t >= 0; --t) {
              const int64_t offset = t * N + idx;
              const acc_t grad_total = static_cast<acc_t>(grad_ptr[offset]) + accum;
              grad_x_ptr[offset] = static_cast<scalar_t>(grad_total);

              acc_t prev_output;
              if (t == 0) {
                prev_output = static_cast<acc_t>(start_ptr[idx]);
              } else {
                prev_output = static_cast<acc_t>(output_ptr[(t - 1) * N + idx]);
              }
              grad_discount_ptr[offset] = static_cast<scalar_t>(grad_total * prev_output);
              const acc_t discount = static_cast<acc_t>(discounts_ptr[offset]);
              accum = grad_total * discount;
            }
            grad_start_ptr[idx] = static_cast<scalar_t>(accum);
          }
        });
      });

  return {grad_start, grad_x, grad_discounts};
}

}  // namespace cpu

#ifdef WITH_CUDA
namespace cuda {
  torch::Tensor discounted_sum_forward(torch::Tensor start_state,
                                       torch::Tensor x,
                                       torch::Tensor discounts);
  std::vector<torch::Tensor> discounted_sum_backward(const torch::Tensor& grad_out,
                                                     const torch::Tensor& discounts,
                                                     const torch::Tensor& output,
                                                     const torch::Tensor& start_state);
}  // namespace cuda
#endif

torch::Tensor discounted_sum_forward(torch::Tensor start_state,
                                     torch::Tensor x,
                                     torch::Tensor discounts) {
  check_inputs(start_state, x, discounts);
  if (start_state.is_cuda()) {
#ifdef WITH_CUDA
    return cuda::discounted_sum_forward(start_state, x, discounts);
#else
    TORCH_CHECK(false, "AGaLiTe kernels built without CUDA support");
#endif
  }
  return cpu::discounted_sum_forward(start_state, x, discounts);
}

std::vector<torch::Tensor> discounted_sum_backward(const torch::Tensor& grad_out,
                                                   const torch::Tensor& discounts,
                                                   const torch::Tensor& output,
                                                   const torch::Tensor& start_state) {
  check_inputs(start_state, grad_out, discounts);
  if (grad_out.is_cuda()) {
#ifdef WITH_CUDA
    return cuda::discounted_sum_backward(grad_out, discounts, output, start_state);
#else
    TORCH_CHECK(false, "AGaLiTe kernels built without CUDA support");
#endif
  }
  return cpu::discounted_sum_backward(grad_out, discounts, output, start_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("discounted_sum_forward", &discounted_sum_forward, "AGaLiTe discounted sum forward");
  m.def("discounted_sum_backward", &discounted_sum_backward, "AGaLiTe discounted sum backward");
}
