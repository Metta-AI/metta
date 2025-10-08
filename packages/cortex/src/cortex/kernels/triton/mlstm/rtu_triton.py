"""Triton-based, time-parallel RTU (low-rank input maps).

This module provides a Triton implementation that matches the baseline
PyTorch kernel (cortex.kernels.pytorch.rtu.LinearRTU) in forward outputs
and gradients for the common case without resets.

Notes
-----
- Resets (segmented scan) are not implemented in this path; pass `resets=None`.
- Requires CUDA + Triton. Tests will skip if Triton/CUDA are unavailable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _TRITON_AVAILABLE = False


# -----------------------------------------------------------------------------
# Activations (value + derivative)
# -----------------------------------------------------------------------------
def _act_and_deriv(z: torch.Tensor, activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    name = activation.lower()
    if name in ["silu", "swish"]:
        y = torch.nn.functional.silu(z)
        s = torch.sigmoid(z)
        dy = s * (1.0 + z * (1.0 - s))  # d/dz [z*sigmoid(z)]
        return y, dy
    elif name == "relu":
        y = torch.relu(z)
        dy = (z > 0).to(z.dtype)
        return y, dy
    elif name == "tanh":
        y = torch.tanh(z)
        dy = 1.0 - y * y
        return y, dy
    elif name in ["linear", "identity"]:
        return z, torch.ones_like(z)
    else:
        raise ValueError(f"Unsupported activation for RTU: {activation}")


# -----------------------------------------------------------------------------
# Triton kernel: one Hillis–Steele scan step (inclusive) for affine monoid
# -----------------------------------------------------------------------------
@triton.jit
def _scan_step_kernel(
    GPhi_g_ptr,  # *fp32 [B,H,T]
    GPhi_phi_ptr,  # *fp32 [B,H,T]
    Bx_ptr,  # *fp32 [B,H,T]
    By_ptr,  # *fp32 [B,H,T]
    T: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_t: tl.constexpr,
    offset: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One inclusive Hillis–Steele step along time.

    Layout is contiguous in last dim: [B,H,T] with strides (H*T, T, 1).
    We update positions t in [offset .. T-1] as:
      A[t] = A[t] ⊕ A[t-offset]
    where the operator ⊕ composes rotations and adds rotated biases.
    """
    pid_lane = tl.program_id(0)  # over B*H lanes
    pid_blk = tl.program_id(1)  # over time tiles

    H = stride_b // stride_h
    b = pid_lane // H
    h = pid_lane % H

    t_idxs = offset + pid_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t_idxs < T

    base = b * stride_b + h * stride_h

    r_idx = base + t_idxs * stride_t
    g_r = tl.load(GPhi_g_ptr + r_idx, mask=mask, other=0.0)
    p_r = tl.load(GPhi_phi_ptr + r_idx, mask=mask, other=0.0)
    bx_r = tl.load(Bx_ptr + r_idx, mask=mask, other=0.0)
    by_r = tl.load(By_ptr + r_idx, mask=mask, other=0.0)

    l_t = t_idxs - offset
    l_idx = base + l_t * stride_t
    g_l = tl.load(GPhi_g_ptr + l_idx, mask=mask, other=0.0)
    p_l = tl.load(GPhi_phi_ptr + l_idx, mask=mask, other=0.0)
    bx_l = tl.load(Bx_ptr + l_idx, mask=mask, other=0.0)
    by_l = tl.load(By_ptr + l_idx, mask=mask, other=0.0)

    # Compose complex numbers z_r * z_l
    g_new = g_r * g_l - p_r * p_l
    p_new = p_r * g_l + g_r * p_l

    # Rotate-add bias: Rot(z_r) @ B_l + B_r
    rot_lx = g_r * bx_l - p_r * by_l
    rot_ly = p_r * bx_l + g_r * by_l
    bx_new = rot_lx + bx_r
    by_new = rot_ly + by_r

    tl.store(GPhi_g_ptr + r_idx, g_new, mask=mask)
    tl.store(GPhi_phi_ptr + r_idx, p_new, mask=mask)
    tl.store(Bx_ptr + r_idx, bx_new, mask=mask)
    tl.store(By_ptr + r_idx, by_new, mask=mask)


def _hillis_steele_scan_inplace(
    gphi_g: torch.Tensor, gphi_phi: torch.Tensor, Bx: torch.Tensor, By: torch.Tensor
) -> None:
    """Inclusive scan over time for [B,H,T] tensors in-place."""
    assert gphi_g.is_contiguous() and gphi_phi.is_contiguous()
    assert Bx.is_contiguous() and By.is_contiguous()
    assert gphi_g.shape == gphi_phi.shape == Bx.shape == By.shape
    B, H, T = gphi_g.shape
    if T <= 1:
        return

    stride_t = 1
    stride_h = T
    stride_b = H * T

    BLOCK_T = 128
    offset = 1
    while offset < T:
        max_tiles = max(T - offset, 0)
        if max_tiles > 0:
            grid = (B * H, (max_tiles + BLOCK_T - 1) // BLOCK_T)
            _scan_step_kernel[grid](
                gphi_g,
                gphi_phi,
                Bx,
                By,
                T,
                stride_b,
                stride_h,
                stride_t,
                offset,
                BLOCK_T,
                num_warps=4,
            )
        offset <<= 1


# -----------------------------------------------------------------------------
# Autograd Function (forward/backward)
# -----------------------------------------------------------------------------
class _LinearRTUFunctionLR_Triton(Function):
    @staticmethod
    def forward(
        ctx,
        x_btd: torch.Tensor,  # (B,T,D)
        nu_log: torch.Tensor,  # (H,)
        theta_log: torch.Tensor,  # (H,)
        U1: torch.Tensor,  # (D,R)
        U2: torch.Tensor,  # (D,R)
        V1: torch.Tensor,  # (R,H)
        V2: torch.Tensor,  # (R,H)
        activation_name: str,
        hc1_init_bh: torch.Tensor,  # (B,H)
        hc2_init_bh: torch.Tensor,  # (B,H)
        resets_bt: Optional[torch.Tensor] = None,  # not supported in this path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Please install `triton` and ensure CUDA is available.")
        if resets_bt is not None:
            # Keep semantics explicit: this path doesn't implement segmented scans yet.
            if resets_bt.numel() > 0 and (resets_bt.any().item()):
                raise NotImplementedError("RTU Triton path does not yet support `resets`. Use PyTorch kernel instead.")

        B, T, D = x_btd.shape
        H = nu_log.shape[0]

        # Decode dynamics
        r = torch.exp(-torch.exp(nu_log))  # (H,)
        theta = torch.exp(theta_log)  # (H,)
        g = r * torch.cos(theta)
        phi = r * torch.sin(theta)
        gamma = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

        # Input projections
        a1_btr = torch.einsum("btd,dr->btr", x_btd, U1)
        a2_btr = torch.einsum("btd,dr->btr", x_btd, U2)
        u1_bth = torch.einsum("btr,rh->bth", a1_btr, V1)
        u2_bth = torch.einsum("btr,rh->bth", a2_btr, V2)

        # Bias leaves per time (scaled by gamma)
        b_bth2 = torch.stack([u1_bth, u2_bth], dim=-1) * gamma.view(1, 1, H, 1)

        # Build [B,H,T] arrays for scan
        g_bht = g.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        p_bht = phi.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        Bx_bht = b_bth2[..., 0].permute(0, 2, 1).contiguous()  # (B,H,T)
        By_bht = b_bth2[..., 1].permute(0, 2, 1).contiguous()

        # Inclusive scan over time
        _hillis_steele_scan_inplace(g_bht, p_bht, Bx_bht, By_bht)

        # Apply to initial condition: c_t = Rot(z^t) c0 + B_t
        c0_bh2 = torch.stack([hc1_init_bh, hc2_init_bh], dim=-1)
        c0x = c0_bh2[..., 0].unsqueeze(-1).expand(B, H, T)
        c0y = c0_bh2[..., 1].unsqueeze(-1).expand(B, H, T)

        rotx = g_bht * c0x - p_bht * c0y
        roty = p_bht * c0x + g_bht * c0y

        cx_bht = rotx + Bx_bht
        cy_bht = roty + By_bht

        c1_bth = cx_bht.permute(0, 2, 1).contiguous()
        c2_bth = cy_bht.permute(0, 2, 1).contiguous()

        y1_bth, _ = _act_and_deriv(c1_bth, activation_name)
        y2_bth, _ = _act_and_deriv(c2_bth, activation_name)
        y_btd_2h = torch.cat([y1_bth, y2_bth], dim=-1)

        # Save for backward
        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            U1,
            U2,
            V1,
            V2,
            c1_bth,
            c2_bth,
            g,
            phi,
            gamma,
            a1_btr,
            a2_btr,
            u1_bth,
            u2_bth,
            hc1_init_bh,
            hc2_init_bh,
        )
        ctx.activation_name = activation_name

        final_hc1_bh = c1_bth[:, -1, :] if T > 0 else hc1_init_bh
        final_hc2_bh = c2_bth[:, -1, :] if T > 0 else hc2_init_bh
        return y_btd_2h, final_hc1_bh, final_hc2_bh

    @staticmethod
    def backward(
        ctx,
        grad_y_btd_2h: torch.Tensor,  # (B,T,2H)
        grad_final_hc1: torch.Tensor,  # unused
        grad_final_hc2: torch.Tensor,  # unused
    ) -> tuple[Optional[torch.Tensor], ...]:
        (
            x_btd,
            nu_log,
            theta_log,
            U1,
            U2,
            V1,
            V2,
            c1_bth,
            c2_bth,
            g,
            phi,
            gamma,
            a1_btr,
            a2_btr,
            u1_bth,
            u2_bth,
            hc1_init_bh,
            hc2_init_bh,
        ) = ctx.saved_tensors
        activation_name = ctx.activation_name

        B, T, D = x_btd.shape
        H = nu_log.shape[0]

        # Local adjoints from activation
        _, d1 = _act_and_deriv(c1_bth, activation_name)
        _, d2 = _act_and_deriv(c2_bth, activation_name)
        gy1 = grad_y_btd_2h[:, :, :H]
        gy2 = grad_y_btd_2h[:, :, H:]
        eta1 = d1 * gy1
        eta2 = d2 * gy2

        # Reverse-time suffix scan for lambda_t = eta_t + M^T lambda_{t+1}
        eta1_rev = torch.flip(eta1, dims=[1])
        eta2_rev = torch.flip(eta2, dims=[1])

        vBx = eta1_rev.permute(0, 2, 1).contiguous()  # (B,H,T)
        vBy = eta2_rev.permute(0, 2, 1).contiguous()
        g_bar = g.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        p_bar = (-phi).view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()

        _hillis_steele_scan_inplace(g_bar, p_bar, vBx, vBy)

        lam1_bth = torch.flip(vBx.permute(0, 2, 1), dims=[1])
        lam2_bth = torch.flip(vBy.permute(0, 2, 1), dims=[1])

        # Input-path grads
        s1g = lam1_bth * gamma.view(1, 1, H)
        s2g = lam2_bth * gamma.view(1, 1, H)

        X_btD = x_btd
        GW1_DH = torch.einsum("btd,bth->dh", X_btD, s1g)
        GW2_DH = torch.einsum("btd,bth->dh", X_btD, s2g)

        tmp1_BR = torch.einsum("bth,rh->btr", s1g, V1)
        tmp2_BR = torch.einsum("bth,rh->btr", s2g, V2)
        dx_btd = torch.einsum("btr,dr->btd", tmp1_BR, U1) + torch.einsum("btr,dr->btd", tmp2_BR, U2)

        dU1_DR = torch.einsum("dh,hr->dr", GW1_DH, V1.t())
        dU2_DR = torch.einsum("dh,hr->dr", GW2_DH, V2.t())
        dV1_RH = torch.einsum("rd,dh->rh", U1.t(), GW1_DH)
        dV2_RH = torch.einsum("rd,dh->rh", U2.t(), GW2_DH)

        # Recurrent-path grads (closed form sums)
        c0_bh2 = torch.stack([hc1_init_bh, hc2_init_bh], dim=-1)
        c_bth2 = torch.stack([c1_bth, c2_bth], dim=-1)
        cprev_bth2 = torch.empty_like(c_bth2)
        cprev_bth2[:, 0, :, :] = c0_bh2
        cprev_bth2[:, 1:, :, :] = c_bth2[:, :-1, :, :]

        lam_bth2 = torch.stack([lam1_bth, lam2_bth], dim=-1)  # (B,T,H,2)

        # Dot products per (b,t,h):
        lamx, lamy = lam_bth2[..., 0], lam_bth2[..., 1]
        cpx, cpy = cprev_bth2[..., 0], cprev_bth2[..., 1]

        dg = (lamx * cpx + lamy * cpy).sum(dim=(0, 1))  # (H,)
        dphi = (lamx * (-cpy) + lamy * cpx).sum(dim=(0, 1))  # (H,)
        dgamma = (lamx * u1_bth + lamy * u2_bth).sum(dim=(0, 1))  # (H,)

        exp_nu = torch.exp(nu_log)
        exp_th = torch.exp(theta_log)
        r = torch.exp(-torch.exp(nu_log))
        gamma_ = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

        dnu_log = -exp_nu * (dg * g + dphi * phi) + exp_nu * (r * r / gamma_) * dgamma
        dth_log = exp_th * (-dg * phi + dphi * g)

        dhc1_init = torch.zeros_like(hc1_init_bh)
        dhc2_init = torch.zeros_like(hc2_init_bh)

        return (
            dx_btd,  # x
            dnu_log,  # nu_log
            dth_log,  # theta_log
            dU1_DR,  # U1
            dU2_DR,  # U2
            dV1_RH,  # V1
            dV2_RH,  # V2
            None,  # activation_name
            dhc1_init,  # hc1_init
            dhc2_init,  # hc2_init
            None,  # resets
        )


class LinearRTU_Triton(nn.Module):
    """Low-rank RTU with Triton time-parallel scans.

    Matches the PyTorch baseline for resets=None.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        rank: int,
        batch_first: bool = True,
        activation: Optional[nn.Module] = None,
        r_max: float = 1.0,
        r_min: float = 0.0,
        max_phase: float = 6.28,
    ) -> None:
        super().__init__()
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Please install `triton` and ensure CUDA is available.")
        if rank < 1 or rank > min(input_size, hidden_size):
            raise ValueError(f"rank must be in [1, min(D,H)] but got {rank}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.batch_first = batch_first
        self.activation = activation if activation is not None else nn.Identity()

        # exp-exp parameterization
        u1 = torch.rand(hidden_size)
        inner = u1 * (r_max**2 - r_min**2) + r_min**2
        nu_log_init = torch.log(-0.5 * torch.log(inner.clamp(min=1e-12)))
        u2 = torch.rand(hidden_size)
        theta_log_init = torch.log((max_phase * u2).clamp(min=1e-12))
        self.nu_log = nn.Parameter(nu_log_init)
        self.theta_log = nn.Parameter(theta_log_init)

        # Low-rank factors
        self.U1 = nn.Parameter(torch.empty(input_size, rank))
        self.U2 = nn.Parameter(torch.empty(input_size, rank))
        self.V1 = nn.Parameter(torch.empty(rank, hidden_size))
        self.V2 = nn.Parameter(torch.empty(rank, hidden_size))

        bound_in = 1.0 / math.sqrt(input_size)
        bound_r = 1.0 / math.sqrt(rank)
        with torch.no_grad():
            self.U1.uniform_(-bound_in, bound_in)
            self.U2.uniform_(-bound_in, bound_in)
            self.V1.uniform_(-bound_r, bound_r)
            self.V2.uniform_(-bound_r, bound_r)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        resets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not self.batch_first:
            x = x.transpose(0, 1)  # (B,T,D)
        B, T, D = x.shape

        if hx is None:
            hc1_init = x.new_zeros(B, self.hidden_size)
            hc2_init = x.new_zeros(B, self.hidden_size)
        else:
            hc1_init, hc2_init = hx

        act_name = self.activation.__class__.__name__
        y_btd_2h, hc1, hc2 = _LinearRTUFunctionLR_Triton.apply(
            x,
            self.nu_log,
            self.theta_log,
            self.U1,
            self.U2,
            self.V1,
            self.V2,
            act_name,
            hc1_init,
            hc2_init,
            resets,
        )

        if self.batch_first:
            return y_btd_2h, (hc1, hc2)
        else:
            return y_btd_2h.transpose(0, 1), (hc1, hc2)


__all__ = ["LinearRTU_Triton"]
