// rtu_seq_allin_binding.cpp
#include <torch/extension.h>

std::vector<at::Tensor> rtu_seq_allin_forward_cuda(
    at::Tensor x, at::Tensor nu_log, at::Tensor th_log,
    at::Tensor w1, at::Tensor w2,
    at::Tensor hc1_init, at::Tensor hc2_init,
    at::Tensor E_nu_c1_in, at::Tensor E_nu_c2_in,
    at::Tensor E_th_c1_in, at::Tensor E_th_c2_in,
    at::Tensor E_w1_c1_in, at::Tensor E_w1_c2_in,
    at::Tensor E_w2_c1_in, at::Tensor E_w2_c2_in,
    at::Tensor resets_u8,
    int act_id);
std::vector<at::Tensor> rtu_seq_allin_backward_cuda(
    at::Tensor grad_y, at::Tensor x,
    at::Tensor nu_log, at::Tensor th_log,
    at::Tensor w1, at::Tensor w2,
    at::Tensor pre1, at::Tensor pre2,
    at::Tensor hc1_init, at::Tensor hc2_init,
    at::Tensor resets_u8,
    at::Tensor E_nu_c1_in, at::Tensor E_nu_c2_in,
    at::Tensor E_th_c1_in, at::Tensor E_th_c2_in,
    at::Tensor E_w1_c1_in, at::Tensor E_w1_c2_in,
    at::Tensor E_w2_c1_in, at::Tensor E_w2_c2_in,
    int act_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_allin",  &rtu_seq_allin_forward_cuda,  "RTU diag forward (sequential fused, all-in)");
  m.def("backward_allin", &rtu_seq_allin_backward_cuda, "RTU diag backward (sequential fused, all-in)");
}

