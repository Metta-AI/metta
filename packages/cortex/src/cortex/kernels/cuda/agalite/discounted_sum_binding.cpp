// Minimal CUDA extension bindings for AGaLiTe discounted sum

#include <torch/extension.h>
#include <vector>

// Forward/backward declarations implemented in the .cu file
namespace agalite_cuda_ds {
at::Tensor discounted_sum_forward_cuda(at::Tensor start_state, at::Tensor x, at::Tensor discounts);
std::vector<at::Tensor> discounted_sum_backward_cuda(const at::Tensor& grad_out,
                                                     const at::Tensor& discounts,
                                                     const at::Tensor& output,
                                                     const at::Tensor& start_state);
}  // namespace agalite_cuda_ds

// Python-visible wrappers
at::Tensor discounted_sum_forward(at::Tensor start_state, at::Tensor x, at::Tensor discounts) {
    TORCH_CHECK(x.dim() >= 2, "x must be at least [T, N]");
    return agalite_cuda_ds::discounted_sum_forward_cuda(std::move(start_state), std::move(x), std::move(discounts));
}

std::vector<at::Tensor> discounted_sum_backward(const at::Tensor& grad_out,
                                                const at::Tensor& discounts,
                                                const at::Tensor& output,
                                                const at::Tensor& start_state) {
    return agalite_cuda_ds::discounted_sum_backward_cuda(grad_out, discounts, output, start_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("discounted_sum_forward", &discounted_sum_forward, "AGaLiTe discounted sum forward (CUDA)");
    m.def("discounted_sum_backward", &discounted_sum_backward, "AGaLiTe discounted sum backward (CUDA)");
}
