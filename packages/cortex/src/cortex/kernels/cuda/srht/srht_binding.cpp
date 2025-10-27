// srht_binding.cpp
#include <torch/extension.h>

std::vector<at::Tensor> srht_forward_cuda(at::Tensor x, at::Tensor signs, c10::optional<at::Tensor> perm_opt, bool normalize);
std::vector<at::Tensor> srht_backward_cuda(at::Tensor gy, at::Tensor signs, c10::optional<at::Tensor> perm_opt, bool normalize);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &srht_forward_cuda, "SRHT forward (CUDA)");
  m.def("backward", &srht_backward_cuda, "SRHT backward (CUDA)");
}

