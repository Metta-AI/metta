// rtu_seq_full_kernels.cu
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

// ----------------------------------------------------
// Forward kernel: Sequential, FULL-RANK input maps
// lane = (b,h); loops over t and d
// ----------------------------------------------------
__global__ void rtu_seq_full_forward_kernel(
    const float* __restrict__ x,         // [B,T,D]
    const float* __restrict__ nu_log,    // [H]
    const float* __restrict__ th_log,    // [H]
    const float* __restrict__ Wc1,       // [D,H]
    const float* __restrict__ Wc2,       // [D,H]
    const float* __restrict__ hc1_init,  // [B,H]
    const float* __restrict__ hc2_init,  // [B,H]
    const float* __restrict__ E_nu_c1_in,// [B,H]
    const float* __restrict__ E_nu_c2_in,// [B,H]
    const float* __restrict__ E_th_c1_in,// [B,H]
    const float* __restrict__ E_th_c2_in,// [B,H]
    const float* __restrict__ E_W1_c1_in,// [B,D,H]
    const float* __restrict__ E_W1_c2_in,// [B,D,H]
    const float* __restrict__ E_W2_c1_in,// [B,D,H]
    const float* __restrict__ E_W2_c2_in,// [B,D,H]
    const uint8_t* __restrict__ resets,  // [B,T]
    int B, int Ttot, int Dtot, int Htot,
    int act_id,
    float* __restrict__ y,               // [B,T,2H]
    float* __restrict__ pre1,            // [B,T,H]
    float* __restrict__ pre2,            // [B,T,H]
    float* __restrict__ final_hc1,       // [B,H]
    float* __restrict__ final_hc2,       // [B,H]
    float* __restrict__ E_nu_c1_out,     // [B,H]
    float* __restrict__ E_nu_c2_out,     // [B,H]
    float* __restrict__ E_th_c1_out,     // [B,H]
    float* __restrict__ E_th_c2_out,     // [B,H]
    float* __restrict__ E_W1_c1_out,     // [B,D,H]
    float* __restrict__ E_W1_c2_out,     // [B,D,H]
    float* __restrict__ E_W2_c1_out,     // [B,D,H]
    float* __restrict__ E_W2_c2_out      // [B,D,H]
){
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int lanes = B * Htot;
  if (lane >= lanes) return;
  const int b = lane / Htot;
  const int h = lane % Htot;

  // decode dynamics for this h
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

  // pointers to (B,D,H) traces at fixed (b,h)
  auto idx_bdh = [&](int d){ return (b * Dtot + d) * Htot + h; };

  float last_c1 = hc1;
  float last_c2 = hc2;

  for (int t = 0; t < Ttot; ++t) {
    const int i_bt  = b * Ttot + t;
    const int i_btd = (b * Ttot + t) * Dtot;
    const int i_bth = (b * Ttot + t) * Htot + h;
    const int i_b2h = (b * Ttot + t) * (2 * Htot) + h;

    const float one_minus = (resets[i_bt] ? 0.f : 1.f);

    // reset row: zero state and ALL (B,H) and (B,D,H) traces
    if (one_minus == 0.f) {
      hc1 = 0.f; hc2 = 0.f;
      E_nu_c1 = 0.f; E_nu_c2 = 0.f;
      E_th_c1 = 0.f; E_th_c2 = 0.f;
      for (int d = 0; d < Dtot; ++d) {
        int j = idx_bdh(d);
        E_W1_c1_out[j] = 0.f; E_W1_c2_out[j] = 0.f;
        E_W2_c1_out[j] = 0.f; E_W2_c2_out[j] = 0.f;
      }
    }

    // u1 = x_t @ Wc1[:,h], u2 = x_t @ Wc2[:,h]
    float u1 = 0.f, u2 = 0.f;
    for (int d = 0; d < Dtot; ++d) {
      float xval = x[i_btd + d];
      u1 = fma_(xval, Wc1[d * Htot + h], u1);
      u2 = fma_(xval, Wc2[d * Htot + h], u2);
    }

    float c1 = fma_(gamma, u1, g * hc1 - phi * hc2);
    float c2 = fma_(gamma, u2, g * hc2 + phi * hc1);

    pre1[i_bth] = c1;
    pre2[i_bth] = c2;

    y[i_b2h]          = act_forward(act_id, c1);
    y[i_b2h + Htot]   = act_forward(act_id, c2);

    // ---- Update FULL-RANK input traces over D
    for (int d = 0; d < Dtot; ++d) {
      float xval = x[i_btd + d];
      int j = idx_bdh(d);

      float Ew11 = (t == 0 ? E_W1_c1_in[j] : E_W1_c1_out[j]);
      float Ew12 = (t == 0 ? E_W1_c2_in[j] : E_W1_c2_out[j]);
      float Ew21 = (t == 0 ? E_W2_c1_in[j] : E_W2_c1_out[j]);
      float Ew22 = (t == 0 ? E_W2_c2_in[j] : E_W2_c2_out[j]);

      // gate resets
      Ew11 *= one_minus; Ew12 *= one_minus; Ew21 *= one_minus; Ew22 *= one_minus;

      float new_Ew11 = fma_(gamma, xval,  g * Ew11 - phi * Ew12);
      float new_Ew12 =                      g * Ew12 + phi * Ew11;
      float new_Ew22 = fma_(gamma, xval,  g * Ew22 + phi * Ew21);
      float new_Ew21 =                      g * Ew21 - phi * Ew22;

      E_W1_c1_out[j] = new_Ew11;
      E_W1_c2_out[j] = new_Ew12;
      E_W2_c1_out[j] = new_Ew21;
      E_W2_c2_out[j] = new_Ew22;
    }

    // ---- Update dynamics traces (nu/theta) using previous c
    float cprev1 = last_c1;
    float cprev2 = last_c2;

    float Enu1 = E_nu_c1, Enu2 = E_nu_c2;
    float Eth1 = E_th_c1, Eth2 = E_th_c2;

    E_nu_c1 = d_g_d_nu * cprev1 + g * Enu1 - d_phi_d_nu * cprev2 - phi * Enu2 + d_gamma_d_nu * u1;
    E_nu_c2 = d_g_d_nu * cprev2 + g * Enu2 + d_phi_d_nu * cprev1 + phi * Enu1 + d_gamma_d_nu * u2;

    E_th_c1 = d_g_d_th * cprev1 + g * Eth1 - d_phi_d_th * cprev2 - phi * Eth2;
    E_th_c2 = d_g_d_th * cprev2 + g * Eth2 + d_phi_d_th * cprev1 + phi * Eth1;

    last_c1 = c1; last_c2 = c2;
    hc1 = c1;     hc2 = c2;
  }

  // write finals and carried traces
  final_hc1[i_bh] = hc1;
  final_hc2[i_bh] = hc2;
  E_nu_c1_out[i_bh] = E_nu_c1;
  E_nu_c2_out[i_bh] = E_nu_c2;
  E_th_c1_out[i_bh] = E_th_c1;
  E_th_c2_out[i_bh] = E_th_c2;
}


// ----------------------------------------------------
// Backward kernel: Sequential, FULL-RANK
// lane = (b,h); loops over t and d, atomics for grad_x and grad_W
// ----------------------------------------------------
__global__ void rtu_seq_full_backward_kernel(
    const float* __restrict__ grad_y,    // [B,T,2H]
    const float* __restrict__ x,         // [B,T,D]
    const float* __restrict__ nu_log,    // [H]
    const float* __restrict__ th_log,    // [H]
    const float* __restrict__ Wc1,       // [D,H]
    const float* __restrict__ Wc2,       // [D,H]
    const float* __restrict__ pre1,      // [B,T,H]
    const float* __restrict__ pre2,      // [B,T,H]
    const float* __restrict__ hc1_init,  // [B,H]
    const float* __restrict__ hc2_init,  // [B,H]
    const uint8_t* __restrict__ resets,  // [B,T]
    const float* __restrict__ E_nu_c1_in,// [B,H]
    const float* __restrict__ E_nu_c2_in,// [B,H]
    const float* __restrict__ E_th_c1_in,// [B,H]
    const float* __restrict__ E_th_c2_in,// [B,H]
    const float* __restrict__ E_W1_c1_in,// [B,D,H]
    const float* __restrict__ E_W1_c2_in,// [B,D,H]
    const float* __restrict__ E_W2_c1_in,// [B,D,H]
    const float* __restrict__ E_W2_c2_in,// [B,D,H]
    int B, int Ttot, int Dtot, int Htot,
    int act_id,
    float* __restrict__ grad_x,          // [B,T,D]
    float* __restrict__ grad_nu_log,     // [H]
    float* __restrict__ grad_th_log,     // [H]
    float* __restrict__ grad_Wc1,        // [D,H]
    float* __restrict__ grad_Wc2,        // [D,H]
    float* __restrict__ grad_hc1_init,   // [B,H]
    float* __restrict__ grad_hc2_init    // [B,H]
){
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

  float s1_next = 0.f;
  float s2_next = 0.f;

  float dg_sum = 0.f;
  float dphi_sum = 0.f;
  float dgamma_sum = 0.f;

  for (int t = Ttot - 1; t >= 0; --t) {
    const int i_bth = (b * Ttot + t) * Htot + h;
    const int i_bt  = b * Ttot + t;
    const int i_btd = (b * Ttot + t) * Dtot;
    const int i_b2h = (b * Ttot + t) * (2 * Htot) + h;

    float c1 = pre1[i_bth];
    float c2 = pre2[i_bth];

    float gy1 = grad_y[i_b2h];
    float gy2 = grad_y[i_b2h + Htot];
    float d1 = gy1 * act_deriv(act_id, c1);
    float d2 = gy2 * act_deriv(act_id, c2);

    // Suffix recurrence should be gated by reset at t+1 (forward zeroes state
    // before computing c_{t+1}). If at the last timestep, allow recurrence.
    float one_minus_next = 1.f;
    if (t + 1 < Ttot) {
      one_minus_next = (resets[b * Ttot + (t + 1)] ? 0.f : 1.f);
    }
    float s1 = d1 + one_minus_next * (g * s1_next +  phi * s2_next);
    float s2 = d2 + one_minus_next * (g * s2_next + (-phi) * s1_next);

    float s1g = gamma * s1;
    float s2g = gamma * s2;

    // grad_x += s1g * Wc1^T + s2g * Wc2^T  (atomic across h)
    for (int d = 0; d < Dtot; ++d) {
      float contrib = s1g * Wc1[d * Htot + h] + s2g * Wc2[d * Htot + h];
      atomicAdd(&grad_x[i_btd + d], contrib);
    }

    // dynamics accumulators need c_{t-1} and u_t
    float cprev1, cprev2;
    if (t == 0) {
      int i_bh = b * Htot + h;
      cprev1 = hc1_init[i_bh];
      cprev2 = hc2_init[i_bh];
    } else {
      int i_prev = (b * Ttot + (t - 1)) * Htot + h;
      cprev1 = pre1[i_prev];
      cprev2 = pre2[i_prev];
    }
    float one_minus_cprev = (resets[i_bt] ? 0.f : 1.f);
    cprev1 *= one_minus_cprev; cprev2 *= one_minus_cprev;

    // u1 = x_t @ Wc1[:,h], u2 = x_t @ Wc2[:,h]
    float u1_t = 0.f, u2_t = 0.f;
    for (int d = 0; d < Dtot; ++d) {
      float xval = x[i_btd + d];
      u1_t = fma_(xval, Wc1[d * Htot + h], u1_t);
      u2_t = fma_(xval, Wc2[d * Htot + h], u2_t);
    }

    dg_sum     += s1 * cprev1 + s2 * cprev2;
    dphi_sum   += -s1 * cprev2 + s2 * cprev1;
    dgamma_sum += s1 * u1_t + s2 * u2_t;

    // parameter grads for Wc1/Wc2 (reverse-mode exact)
    // grad_Wc* += X_t^T @ (gamma ⊙ s*)
    for (int d = 0; d < Dtot; ++d) {
      float xval = x[i_btd + d];
      atomicAdd(&grad_Wc1[d * Htot + h], xval * s1g);
      atomicAdd(&grad_Wc2[d * Htot + h], xval * s2g);
    }

    s1_next = s1;
    s2_next = s2;
  }

  // boundary adjoint at chunk head
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

  // boundary corrections for FULL-RANK W: add sum_b E[b,d,h]*lam_prev[b,h]
  auto idx_bdh = [&](int d){ return (b * Dtot + d) * Htot + h; };
  for (int d = 0; d < Dtot; ++d) {
    int j = idx_bdh(d);
    float w1_bc = lam_prev_c1 * E_W1_c1_in[j] + lam_prev_c2 * E_W1_c2_in[j];
    float w2_bc = lam_prev_c1 * E_W2_c1_in[j] + lam_prev_c2 * E_W2_c2_in[j];
    atomicAdd(&grad_Wc1[d * Htot + h], w1_bc);
    atomicAdd(&grad_Wc2[d * Htot + h], w2_bc);
  }

  // dynamics param grads (per h)
  float gamma_ = gamma;  // reuse computed sqrt(1 - r^2)
  float grad_nu_local =
      -exp_nu * (dg_sum * g + dphi_sum * phi)
      + exp_nu * (r2 / fmaxf(gamma_, 1e-20f)) * dgamma_sum
      + (lam_prev_c1 * E_nu_c1_in[i_bh] + lam_prev_c2 * E_nu_c2_in[i_bh]);

  float grad_th_local =
      exp_th * (-dg_sum * phi + dphi_sum * g)
      + (lam_prev_c1 * E_th_c1_in[i_bh] + lam_prev_c2 * E_th_c2_in[i_bh]);

  atomicAdd(&grad_nu_log[h], grad_nu_local);
  atomicAdd(&grad_th_log[h], grad_th_local);
}


// ----------------------------------------------------
// Small helpers
// ----------------------------------------------------
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

// ----------------------------------------------------
// Host wrappers
// ----------------------------------------------------
std::vector<at::Tensor> rtu_seq_full_forward_cuda(
    at::Tensor x, at::Tensor nu_log, at::Tensor th_log,
    at::Tensor Wc1, at::Tensor Wc2,
    at::Tensor hc1_init, at::Tensor hc2_init,
    at::Tensor E_nu_c1_in, at::Tensor E_nu_c2_in,
    at::Tensor E_th_c1_in, at::Tensor E_th_c2_in,
    at::Tensor E_W1_c1_in, at::Tensor E_W1_c2_in,
    at::Tensor E_W2_c1_in, at::Tensor E_W2_c2_in,
    at::Tensor resets_u8,
    int act_id)
{
  expect_f32_cuda_contig(x, "x");            // [B,T,D]
  expect_f32_cuda_contig(nu_log, "nu_log");  // [H]
  expect_f32_cuda_contig(th_log, "theta_log");//[H]
  expect_f32_cuda_contig(Wc1, "Wc1");        // [D,H]
  expect_f32_cuda_contig(Wc2, "Wc2");        // [D,H]
  expect_f32_cuda_contig(hc1_init, "hc1_init"); // [B,H]
  expect_f32_cuda_contig(hc2_init, "hc2_init"); // [B,H]
  expect_f32_cuda_contig(E_nu_c1_in, "E_nu_c1_in");
  expect_f32_cuda_contig(E_nu_c2_in, "E_nu_c2_in");
  expect_f32_cuda_contig(E_th_c1_in, "E_th_c1_in");
  expect_f32_cuda_contig(E_th_c2_in, "E_th_c2_in");
  expect_f32_cuda_contig(E_W1_c1_in, "E_W1_c1_in"); // [B,D,H]
  expect_f32_cuda_contig(E_W1_c2_in, "E_W1_c2_in");
  expect_f32_cuda_contig(E_W2_c1_in, "E_W2_c1_in");
  expect_f32_cuda_contig(E_W2_c2_in, "E_W2_c2_in");
  expect_u8_cuda_contig(resets_u8, "resets");

  const int B = x.size(0);
  const int T = x.size(1);
  const int D = x.size(2);
  const int H = nu_log.size(0);
  TORCH_CHECK(Wc1.size(0)==D && Wc1.size(1)==H, "Wc1 must be [D,H]");
  TORCH_CHECK(Wc2.size(0)==D && Wc2.size(1)==H, "Wc2 must be [D,H]");

  auto opts = x.options();
  auto y    = at::empty({B, T, 2*H}, opts);
  auto pre1 = at::empty({B, T, H}, opts);
  auto pre2 = at::empty({B, T, H}, opts);
  auto fh1  = at::empty({B, H}, opts);
  auto fh2  = at::empty({B, H}, opts);

  auto Enu1  = at::empty({B, H}, opts);
  auto Enu2  = at::empty({B, H}, opts);
  auto Eth1  = at::empty({B, H}, opts);
  auto Eth2  = at::empty({B, H}, opts);

  auto EW1c1 = at::empty({B, D, H}, opts);
  auto EW1c2 = at::empty({B, D, H}, opts);
  auto EW2c1 = at::empty({B, D, H}, opts);
  auto EW2c2 = at::empty({B, D, H}, opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int lanes = B * H;
  const dim3 block(RTU_SEQ_THREADS);
  const dim3 grid((lanes + RTU_SEQ_THREADS - 1) / RTU_SEQ_THREADS);

  rtu_seq_full_forward_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(), nu_log.data_ptr<float>(), th_log.data_ptr<float>(),
      Wc1.data_ptr<float>(), Wc2.data_ptr<float>(),
      hc1_init.data_ptr<float>(), hc2_init.data_ptr<float>(),
      E_nu_c1_in.data_ptr<float>(), E_nu_c2_in.data_ptr<float>(),
      E_th_c1_in.data_ptr<float>(), E_th_c2_in.data_ptr<float>(),
      E_W1_c1_in.data_ptr<float>(), E_W1_c2_in.data_ptr<float>(),
      E_W2_c1_in.data_ptr<float>(), E_W2_c2_in.data_ptr<float>(),
      resets_u8.data_ptr<uint8_t>(),
      B, T, D, H, act_id,
      y.data_ptr<float>(), pre1.data_ptr<float>(), pre2.data_ptr<float>(),
      fh1.data_ptr<float>(), fh2.data_ptr<float>(),
      Enu1.data_ptr<float>(), Enu2.data_ptr<float>(),
      Eth1.data_ptr<float>(), Eth2.data_ptr<float>(),
      EW1c1.data_ptr<float>(), EW1c2.data_ptr<float>(),
      EW2c1.data_ptr<float>(), EW2c2.data_ptr<float>()
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {y, pre1, pre2, fh1, fh2,
          Enu1, Enu2, Eth1, Eth2,
          EW1c1, EW1c2, EW2c1, EW2c2};
}

std::vector<at::Tensor> rtu_seq_full_backward_cuda(
    at::Tensor grad_y, at::Tensor x,
    at::Tensor nu_log, at::Tensor th_log,
    at::Tensor Wc1, at::Tensor Wc2,
    at::Tensor pre1, at::Tensor pre2,
    at::Tensor hc1_init, at::Tensor hc2_init,
    at::Tensor resets_u8,
    at::Tensor E_nu_c1_in, at::Tensor E_nu_c2_in,
    at::Tensor E_th_c1_in, at::Tensor E_th_c2_in,
    at::Tensor E_W1_c1_in, at::Tensor E_W1_c2_in,
    at::Tensor E_W2_c1_in, at::Tensor E_W2_c2_in,
    int act_id)
{
  expect_f32_cuda_contig(grad_y, "grad_y");
  expect_f32_cuda_contig(x, "x");
  expect_f32_cuda_contig(nu_log, "nu_log");
  expect_f32_cuda_contig(th_log, "theta_log");
  expect_f32_cuda_contig(Wc1, "Wc1");
  expect_f32_cuda_contig(Wc2, "Wc2");
  expect_f32_cuda_contig(pre1, "pre1");
  expect_f32_cuda_contig(pre2, "pre2");
  expect_f32_cuda_contig(hc1_init, "hc1_init");
  expect_f32_cuda_contig(hc2_init, "hc2_init");
  expect_u8_cuda_contig(resets_u8, "resets");
  expect_f32_cuda_contig(E_nu_c1_in, "E_nu_c1_in");
  expect_f32_cuda_contig(E_nu_c2_in, "E_nu_c2_in");
  expect_f32_cuda_contig(E_th_c1_in, "E_th_c1_in");
  expect_f32_cuda_contig(E_th_c2_in, "E_th_c2_in");
  expect_f32_cuda_contig(E_W1_c1_in, "E_W1_c1_in");
  expect_f32_cuda_contig(E_W1_c2_in, "E_W1_c2_in");
  expect_f32_cuda_contig(E_W2_c1_in, "E_W2_c1_in");
  expect_f32_cuda_contig(E_W2_c2_in, "E_W2_c2_in");

  const int B = x.size(0);
  const int T = x.size(1);
  const int D = x.size(2);
  const int H = nu_log.size(0);
  TORCH_CHECK(Wc1.size(0)==D && Wc1.size(1)==H, "Wc1 must be [D,H]");
  TORCH_CHECK(Wc2.size(0)==D && Wc2.size(1)==H, "Wc2 must be [D,H]");

  auto opts = x.options();
  auto grad_x  = at::zeros_like(x);
  auto gnu     = at::zeros({H}, opts);
  auto gth     = at::zeros({H}, opts);
  auto gW1     = at::zeros({D, H}, opts);
  auto gW2     = at::zeros({D, H}, opts);
  auto ghc1    = at::empty({B, H}, opts);
  auto ghc2    = at::empty({B, H}, opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int lanes = B * H;
  const dim3 block(RTU_SEQ_THREADS);
  const dim3 grid((lanes + RTU_SEQ_THREADS - 1) / RTU_SEQ_THREADS);

  rtu_seq_full_backward_kernel<<<grid, block, 0, stream>>>(
      grad_y.data_ptr<float>(),
      x.data_ptr<float>(),
      nu_log.data_ptr<float>(),
      th_log.data_ptr<float>(),
      Wc1.data_ptr<float>(),
      Wc2.data_ptr<float>(),
      pre1.data_ptr<float>(),
      pre2.data_ptr<float>(),
      hc1_init.data_ptr<float>(),
      hc2_init.data_ptr<float>(),
      resets_u8.data_ptr<uint8_t>(),
      E_nu_c1_in.data_ptr<float>(), E_nu_c2_in.data_ptr<float>(),
      E_th_c1_in.data_ptr<float>(), E_th_c2_in.data_ptr<float>(),
      E_W1_c1_in.data_ptr<float>(), E_W1_c2_in.data_ptr<float>(),
      E_W2_c1_in.data_ptr<float>(), E_W2_c2_in.data_ptr<float>(),
      B, T, D, H, act_id,
      grad_x.data_ptr<float>(),
      gnu.data_ptr<float>(),
      gth.data_ptr<float>(),
      gW1.data_ptr<float>(),
      gW2.data_ptr<float>(),
      ghc1.data_ptr<float>(),
      ghc2.data_ptr<float>()
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, gnu, gth, gW1, gW2, ghc1, ghc2};
}
