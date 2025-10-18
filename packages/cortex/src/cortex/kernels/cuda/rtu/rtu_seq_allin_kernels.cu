// rtu_seq_allin_kernels.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef RTU_SEQ_THREADS
#define RTU_SEQ_THREADS 256
#endif

enum ActId : int {
  kSiLU = 0,
  kReLU = 1,
  kTanh = 2,
  kLinear = 3
};

template <typename T>
__device__ __forceinline__ T fma_(T a, T b, T c) { return fmaf(a, b, c); }

__device__ __forceinline__ float sigmoidf_(float z) {
  return 1.f / (1.f + expf(-z));
}

__device__ __forceinline__ float act_forward(int act_id, float z) {
  switch (act_id) {
    case kSiLU: {
      float s = sigmoidf_(z);
      return z * s;
    }
    case kReLU:  return z > 0.f ? z : 0.f;
    case kTanh:  return tanhf(z);
    case kLinear:
    default:     return z;
  }
}

__device__ __forceinline__ float act_deriv(int act_id, float z) {
  switch (act_id) {
    case kSiLU: {
      float s = sigmoidf_(z);
      return s * (1.f + z * (1.f - s));
    }
    case kReLU:  return z > 0.f ? 1.f : 0.f;
    case kTanh: {
      float y = tanhf(z);
      return 1.f - y * y;
    }
    case kLinear:
    default:     return 1.f;
  }
}

// -------------------------------------
// Forward: sequential, everything inside
// -------------------------------------
__global__ void rtu_seq_allin_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ nu_log,
    const float* __restrict__ th_log,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ hc1_init,
    const float* __restrict__ hc2_init,
    const float* __restrict__ E_nu_c1_in,
    const float* __restrict__ E_nu_c2_in,
    const float* __restrict__ E_th_c1_in,
    const float* __restrict__ E_th_c2_in,
    const float* __restrict__ E_w1_c1_in,
    const float* __restrict__ E_w1_c2_in,
    const float* __restrict__ E_w2_c1_in,
    const float* __restrict__ E_w2_c2_in,
    const uint8_t* __restrict__ resets,
    int B, int Ttot, int Htot,
    int act_id,
    float* __restrict__ y,
    float* __restrict__ pre1,
    float* __restrict__ pre2,
    float* __restrict__ final_hc1,
    float* __restrict__ final_hc2,
    float* __restrict__ E_nu_c1_out,
    float* __restrict__ E_nu_c2_out,
    float* __restrict__ E_th_c1_out,
    float* __restrict__ E_th_c2_out,
    float* __restrict__ E_w1_c1_out,
    float* __restrict__ E_w1_c2_out,
    float* __restrict__ E_w2_c1_out,
    float* __restrict__ E_w2_c2_out)
{
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int lanes = B * Htot;
  if (lane >= lanes) return;
  const int b = lane / Htot;
  const int h = lane % Htot;

  // decode dynamics
  float nu = nu_log[h];
  float thl = th_log[h];
  float exp_nu = expf(nu);
  float r = expf(-exp_nu);
  float theta = expf(thl);
  float s, c;
  sincosf(theta, &s, &c);
  float g = r * c;
  float phi = r * s;
  float r2 = r * r;
  float one_minus_r2 = fmaxf(1.f - r2, 0.f);
  float gamma = sqrtf(one_minus_r2);

  // derivatives wrt logs
  float d_g_d_nu     = -exp_nu * g;
  float d_phi_d_nu   = -exp_nu * phi;
  float denom = fmaxf(sqrtf(one_minus_r2), 1e-20f);
  float d_gamma_d_nu = exp_nu * r2 / denom;
  float exp_th = expf(thl);
  float d_g_d_th   = -phi * exp_th;
  float d_phi_d_th =  g   * exp_th;

  const int i_bh = b * Htot + h;
  float hc1 = hc1_init[i_bh];
  float hc2 = hc2_init[i_bh];

  float E_nu_c1 = E_nu_c1_in[i_bh];
  float E_nu_c2 = E_nu_c2_in[i_bh];
  float E_th_c1 = E_th_c1_in[i_bh];
  float E_th_c2 = E_th_c2_in[i_bh];
  float E_w1_c1 = E_w1_c1_in[i_bh];
  float E_w1_c2 = E_w1_c2_in[i_bh];
  float E_w2_c1 = E_w2_c1_in[i_bh];
  float E_w2_c2 = E_w2_c2_in[i_bh];

  float last_c1 = hc1;
  float last_c2 = hc2;

  for (int t = 0; t < Ttot; ++t) {
    const int i_bt  = b * Ttot + t;
    const int i_bth = ((b * Ttot) + t) * Htot + h;

    const float one_minus = (resets[i_bt] ? 0.f : 1.f);
    E_w1_c1 *= one_minus; E_w1_c2 *= one_minus;
    E_w2_c1 *= one_minus; E_w2_c2 *= one_minus;
    E_nu_c1 *= one_minus; E_nu_c2 *= one_minus;
    E_th_c1 *= one_minus; E_th_c2 *= one_minus;
    hc1     *= one_minus; hc2     *= one_minus;

    float xval = x[i_bth];
    float u1 = w1[h] * xval;
    float u2 = w2[h] * xval;

    float c1 = fma_(gamma, u1, g * hc1 - phi * hc2);
    float c2 = fma_(gamma, u2, g * hc2 + phi * hc1);

    pre1[i_bth] = c1;
    pre2[i_bth] = c2;

    int i_bth2 = ((b * Ttot) + t) * (2 * Htot) + h;
    float y1 = act_forward(act_id, c1);
    float y2 = act_forward(act_id, c2);
    y[i_bth2] = y1;
    y[i_bth2 + Htot] = y2;

    // input traces (w1)
    {
      float Ew11 = E_w1_c1, Ew12 = E_w1_c2;
      E_w1_c1 = fma_(gamma, xval,  g * Ew11 - phi * Ew12);
      E_w1_c2 =                       g * Ew12 + phi * Ew11;
    }
    // input traces (w2)
    {
      float Ew21 = E_w2_c1, Ew22 = E_w2_c2;
      E_w2_c2 = fma_(gamma, xval,  g * Ew22 + phi * Ew21);
      E_w2_c1 =                       g * Ew21 - phi * Ew22;
    }

    // diag traces (nu/theta) using previous c
    float cprev1 = last_c1;
    float cprev2 = last_c2;

    float Enu1 = E_nu_c1, Enu2 = E_nu_c2;
    E_nu_c1 =  d_g_d_nu * cprev1 + g * Enu1 - d_phi_d_nu * cprev2 - phi * Enu2 + d_gamma_d_nu * u1;
    E_nu_c2 =  d_g_d_nu * cprev2 + g * Enu2 + d_phi_d_nu * cprev1 + phi * Enu1 + d_gamma_d_nu * u2;

    float Eth1 = E_th_c1, Eth2 = E_th_c2;
    E_th_c1 =  d_g_d_th * cprev1 + g * Eth1 - d_phi_d_th * cprev2 - phi * Eth2;
    E_th_c2 =  d_g_d_th * cprev2 + g * Eth2 + d_phi_d_th * cprev1 + phi * Eth1;

    last_c1 = c1; last_c2 = c2;
    hc1 = c1;     hc2 = c2;
  }

  final_hc1[i_bh] = hc1;
  final_hc2[i_bh] = hc2;
  E_nu_c1_out[i_bh] = E_nu_c1;
  E_nu_c2_out[i_bh] = E_nu_c2;
  E_th_c1_out[i_bh] = E_th_c1;
  E_th_c2_out[i_bh] = E_th_c2;
  E_w1_c1_out[i_bh] = E_w1_c1;
  E_w1_c2_out[i_bh] = E_w1_c2;
  E_w2_c1_out[i_bh] = E_w2_c1;
  E_w2_c2_out[i_bh] = E_w2_c2;
}

// -------------------------------------
// Backward: sequential, everything inside
// -------------------------------------
__global__ void rtu_seq_allin_backward_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ nu_log,
    const float* __restrict__ th_log,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ pre1,
    const float* __restrict__ pre2,
    const float* __restrict__ hc1_init,
    const float* __restrict__ hc2_init,
    const uint8_t* __restrict__ resets,
    const float* __restrict__ E_nu_c1_in,
    const float* __restrict__ E_nu_c2_in,
    const float* __restrict__ E_th_c1_in,
    const float* __restrict__ E_th_c2_in,
    const float* __restrict__ E_w1_c1_in,
    const float* __restrict__ E_w1_c2_in,
    const float* __restrict__ E_w2_c1_in,
    const float* __restrict__ E_w2_c2_in,
    int B, int Ttot, int Htot,
    int act_id,
    float* __restrict__ grad_x,
    float* __restrict__ grad_nu_log,
    float* __restrict__ grad_th_log,
    float* __restrict__ grad_w1,
    float* __restrict__ grad_w2,
    float* __restrict__ grad_hc1_init,
    float* __restrict__ grad_hc2_init)
{
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int lanes = B * Htot;
  if (lane >= lanes) return;
  const int b = lane / Htot;
  const int h = lane % Htot;

  float nu = nu_log[h];
  float thl = th_log[h];
  float exp_nu = expf(nu);
  float r = expf(-exp_nu);
  float theta = expf(thl);
  float s, c;
  sincosf(theta, &s, &c);
  float g = r * c;
  float phi = r * s;
  float r2 = r * r;
  float one_minus_r2 = fmaxf(1.f - r2, 0.f);
  float gamma = sqrtf(one_minus_r2);

  float exp_th = expf(thl);
  float denom = fmaxf(sqrtf(one_minus_r2), 1e-20f);

  float s1_next = 0.f;
  float s2_next = 0.f;

  float gw1_local = 0.f;
  float gw2_local = 0.f;
  float dg_sum = 0.f;
  float dphi_sum = 0.f;
  float dgamma_sum = 0.f;

  for (int t = Ttot - 1; t >= 0; --t) {
    const int i_bth = ((b * Ttot) + t) * Htot + h;
    const int i_bt  = b * Ttot + t;

    float c1 = pre1[i_bth];
    float c2 = pre2[i_bth];

    int i_bth2 = ((b * Ttot) + t) * (2 * Htot) + h;
    float gy1 = grad_y[i_bth2];
    float gy2 = grad_y[i_bth2 + Htot];
    float d1 = gy1 * act_deriv(act_id, c1);
    float d2 = gy2 * act_deriv(act_id, c2);

    float one_minus = (resets[i_bt] ? 0.f : 1.f);
    float s1 = d1 + one_minus * (g * s1_next +  phi * s2_next);
    float s2 = d2 + one_minus * (g * s2_next + (-phi) * s1_next);

    float xval = x[i_bth];
    float s1g = gamma * s1;
    float s2g = gamma * s2;
    grad_x[i_bth] = s1g * w1[h] + s2g * w2[h];

    gw1_local += s1g * xval;
    gw2_local += s2g * xval;

    float cprev1, cprev2;
    if (t == 0) {
      int i_bh = b * Htot + h;
      cprev1 = hc1_init[i_bh];
      cprev2 = hc2_init[i_bh];
    } else {
      int i_prev = ((b * Ttot) + (t - 1)) * Htot + h;
      cprev1 = pre1[i_prev];
      cprev2 = pre2[i_prev];
    }
    cprev1 *= one_minus; cprev2 *= one_minus;

    float u1_t = w1[h] * xval;
    float u2_t = w2[h] * xval;

    dg_sum     += s1 * cprev1 + s2 * cprev2;
    dphi_sum   += -s1 * cprev2 + s2 * cprev1;
    dgamma_sum += s1 * u1_t + s2 * u2_t;

    s1_next = s1;
    s2_next = s2;
  }

  const int i_bh = b * Htot + h;
  float lambda0_c1 = s1_next;
  float lambda0_c2 = s2_next;

  float lam_prev_c1 = g * lambda0_c1 + phi * lambda0_c2;
  float lam_prev_c2 = -phi * lambda0_c1 + g * lambda0_c2;

  float head_mask = (Ttot > 0 && resets[b * Ttot + 0]) ? 0.f : 1.f;
  lam_prev_c1 *= head_mask;
  lam_prev_c2 *= head_mask;

  grad_hc1_init[i_bh] = lam_prev_c1;
  grad_hc2_init[i_bh] = lam_prev_c2;

  float w1_bc = lam_prev_c1 * E_w1_c1_in[i_bh] + lam_prev_c2 * E_w1_c2_in[i_bh];
  float w2_bc = lam_prev_c1 * E_w2_c1_in[i_bh] + lam_prev_c2 * E_w2_c2_in[i_bh];

  float nu_bc = lam_prev_c1 * E_nu_c1_in[i_bh] + lam_prev_c2 * E_nu_c2_in[i_bh];
  float th_bc = lam_prev_c1 * E_th_c1_in[i_bh] + lam_prev_c2 * E_th_c2_in[i_bh];

  float grad_nu_local =
      -exp_nu * (dg_sum * g + dphi_sum * phi)
      + exp_nu * (r2 / fmaxf(sqrtf(one_minus_r2), 1e-20f)) * dgamma_sum
      + nu_bc;

  float grad_th_local =
      exp_th * (-dg_sum * phi + dphi_sum * g)
      + th_bc;

  atomicAdd(&grad_w1[h],      gw1_local + w1_bc);
  atomicAdd(&grad_w2[h],      gw2_local + w2_bc);
  atomicAdd(&grad_nu_log[h],  grad_nu_local);
  atomicAdd(&grad_th_log[h],  grad_th_local);
}

static void expect_f32_cuda_contig(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
static void expect_u8_cuda_contig(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == at::kByte, name, " must be uint8");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::vector<at::Tensor> rtu_seq_allin_forward_cuda(
    at::Tensor x, at::Tensor nu_log, at::Tensor th_log,
    at::Tensor w1, at::Tensor w2,
    at::Tensor hc1_init, at::Tensor hc2_init,
    at::Tensor E_nu_c1_in, at::Tensor E_nu_c2_in,
    at::Tensor E_th_c1_in, at::Tensor E_th_c2_in,
    at::Tensor E_w1_c1_in, at::Tensor E_w1_c2_in,
    at::Tensor E_w2_c1_in, at::Tensor E_w2_c2_in,
    at::Tensor resets_u8,
    int act_id)
{
  expect_f32_cuda_contig(x, "x");
  expect_f32_cuda_contig(nu_log, "nu_log");
  expect_f32_cuda_contig(th_log, "theta_log");
  expect_f32_cuda_contig(w1, "w1");
  expect_f32_cuda_contig(w2, "w2");
  expect_f32_cuda_contig(hc1_init, "hc1_init");
  expect_f32_cuda_contig(hc2_init, "hc2_init");
  expect_f32_cuda_contig(E_nu_c1_in, "E_nu_c1_in");
  expect_f32_cuda_contig(E_nu_c2_in, "E_nu_c2_in");
  expect_f32_cuda_contig(E_th_c1_in, "E_th_c1_in");
  expect_f32_cuda_contig(E_th_c2_in, "E_th_c2_in");
  expect_f32_cuda_contig(E_w1_c1_in, "E_w1_c1_in");
  expect_f32_cuda_contig(E_w1_c2_in, "E_w1_c2_in");
  expect_f32_cuda_contig(E_w2_c1_in, "E_w2_c1_in");
  expect_f32_cuda_contig(E_w2_c2_in, "E_w2_c2_in");
  expect_u8_cuda_contig(resets_u8, "resets");

  const int B = x.size(0);
  const int T = x.size(1);
  const int H = x.size(2);
  auto opts = x.options();

  auto y   = at::empty({B, T, 2*H}, opts);
  auto pre1= at::empty({B, T, H}, opts);
  auto pre2= at::empty({B, T, H}, opts);
  auto fh1 = at::empty({B, H}, opts);
  auto fh2 = at::empty({B, H}, opts);
  auto Enu1= at::empty({B, H}, opts);
  auto Enu2= at::empty({B, H}, opts);
  auto Eth1= at::empty({B, H}, opts);
  auto Eth2= at::empty({B, H}, opts);
  auto Ew1c1= at::empty({B, H}, opts);
  auto Ew1c2= at::empty({B, H}, opts);
  auto Ew2c1= at::empty({B, H}, opts);
  auto Ew2c2= at::empty({B, H}, opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int lanes = B * H;
  const dim3 block(RTU_SEQ_THREADS);
  const dim3 grid((lanes + RTU_SEQ_THREADS - 1) / RTU_SEQ_THREADS);

  rtu_seq_allin_forward_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(), nu_log.data_ptr<float>(), th_log.data_ptr<float>(),
      w1.data_ptr<float>(), w2.data_ptr<float>(),
      hc1_init.data_ptr<float>(), hc2_init.data_ptr<float>(),
      E_nu_c1_in.data_ptr<float>(), E_nu_c2_in.data_ptr<float>(),
      E_th_c1_in.data_ptr<float>(), E_th_c2_in.data_ptr<float>(),
      E_w1_c1_in.data_ptr<float>(), E_w1_c2_in.data_ptr<float>(),
      E_w2_c1_in.data_ptr<float>(), E_w2_c2_in.data_ptr<float>(),
      resets_u8.data_ptr<uint8_t>(),
      B, T, H, act_id,
      y.data_ptr<float>(), pre1.data_ptr<float>(), pre2.data_ptr<float>(),
      fh1.data_ptr<float>(), fh2.data_ptr<float>(),
      Enu1.data_ptr<float>(), Enu2.data_ptr<float>(),
      Eth1.data_ptr<float>(), Eth2.data_ptr<float>(),
      Ew1c1.data_ptr<float>(), Ew1c2.data_ptr<float>(),
      Ew2c1.data_ptr<float>(), Ew2c2.data_ptr<float>()
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {y, pre1, pre2, fh1, fh2,
          Enu1, Enu2, Eth1, Eth2, Ew1c1, Ew1c2, Ew2c1, Ew2c2};
}

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
    int act_id)
{
  expect_f32_cuda_contig(grad_y, "grad_y");
  expect_f32_cuda_contig(x, "x");
  expect_f32_cuda_contig(nu_log, "nu_log");
  expect_f32_cuda_contig(th_log, "theta_log");
  expect_f32_cuda_contig(w1, "w1");
  expect_f32_cuda_contig(w2, "w2");
  expect_f32_cuda_contig(pre1, "pre1");
  expect_f32_cuda_contig(pre2, "pre2");
  expect_f32_cuda_contig(hc1_init, "hc1_init");
  expect_f32_cuda_contig(hc2_init, "hc2_init");
  expect_u8_cuda_contig(resets_u8, "resets");
  expect_f32_cuda_contig(E_nu_c1_in, "E_nu_c1_in");
  expect_f32_cuda_contig(E_nu_c2_in, "E_nu_c2_in");
  expect_f32_cuda_contig(E_th_c1_in, "E_th_c1_in");
  expect_f32_cuda_contig(E_th_c2_in, "E_th_c2_in");
  expect_f32_cuda_contig(E_w1_c1_in, "E_w1_c1_in");
  expect_f32_cuda_contig(E_w1_c2_in, "E_w1_c2_in");
  expect_f32_cuda_contig(E_w2_c1_in, "E_w2_c1_in");
  expect_f32_cuda_contig(E_w2_c2_in, "E_w2_c2_in");

  const int B = x.size(0);
  const int T = x.size(1);
  const int H = x.size(2);
  auto opts = x.options();

  auto grad_x  = at::empty_like(x);
  auto gnu     = at::zeros({H}, opts);
  auto gth     = at::zeros({H}, opts);
  auto gw1     = at::zeros({H}, opts);
  auto gw2     = at::zeros({H}, opts);
  auto ghc1    = at::empty({B, H}, opts);
  auto ghc2    = at::empty({B, H}, opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int lanes = B * H;
  const dim3 block(RTU_SEQ_THREADS);
  const dim3 grid((lanes + RTU_SEQ_THREADS - 1) / RTU_SEQ_THREADS);

  rtu_seq_allin_backward_kernel<<<grid, block, 0, stream>>>(
      grad_y.data_ptr<float>(),
      x.data_ptr<float>(),
      nu_log.data_ptr<float>(),
      th_log.data_ptr<float>(),
      w1.data_ptr<float>(),
      w2.data_ptr<float>(),
      pre1.data_ptr<float>(),
      pre2.data_ptr<float>(),
      hc1_init.data_ptr<float>(),
      hc2_init.data_ptr<float>(),
      resets_u8.data_ptr<uint8_t>(),
      E_nu_c1_in.data_ptr<float>(), E_nu_c2_in.data_ptr<float>(),
      E_th_c1_in.data_ptr<float>(), E_th_c2_in.data_ptr<float>(),
      E_w1_c1_in.data_ptr<float>(), E_w1_c2_in.data_ptr<float>(),
      E_w2_c1_in.data_ptr<float>(), E_w2_c2_in.data_ptr<float>(),
      B, T, H, act_id,
      grad_x.data_ptr<float>(),
      gnu.data_ptr<float>(),
      gth.data_ptr<float>(),
      gw1.data_ptr<float>(),
      gw2.data_ptr<float>(),
      ghc1.data_ptr<float>(),
      ghc2.data_ptr<float>()
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, gnu, gth, gw1, gw2, ghc1, ghc2};
}
