"""Finite-difference gradient check for the low-rank LinearRTU kernel (PyTorch).

This test validates that the custom autograd implementation for the RTU matches
finite-difference gradients across all parameters on a small problem.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Import the kernel directly from cortex package
from cortex.kernels.pytorch.rtu import LinearRTU

# Triton parity tests (skip when Triton/CUDA are unavailable)
from cortex.utils import TRITON_AVAILABLE


@pytest.mark.skipif(not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="CUDA/Triton not available")
def test_triton_rtu_matches_pytorch_forward_and_gradients() -> None:
    from cortex.kernels.triton import LinearRTU_Triton

    torch.manual_seed(2024)
    device = torch.device("cuda")
    dtype = torch.float32

    # Problem sizes kept modest to limit JIT compile time
    B, T, D, H, R = 2, 64, 32, 24, 8

    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

    # Construct models and align parameters
    pt_model = LinearRTU(
        input_size=D,
        hidden_size=H,
        rank=R,
        batch_first=True,
        activation=torch.nn.SiLU(),
    ).to(device=device, dtype=dtype)

    tr_model = LinearRTU_Triton(
        input_size=D,
        hidden_size=H,
        rank=R,
        batch_first=True,
        activation=torch.nn.SiLU(),
    ).to(device=device, dtype=dtype)

    # Copy parameters from PyTorch baseline to Triton model for 1:1 parity
    with torch.no_grad():
        tr_model.nu_log.copy_(pt_model.nu_log)
        tr_model.theta_log.copy_(pt_model.theta_log)
        tr_model.U1.copy_(pt_model.U1)
        tr_model.U2.copy_(pt_model.U2)
        tr_model.V1.copy_(pt_model.V1)
        tr_model.V2.copy_(pt_model.V2)

    # Forward
    y_pt, _ = pt_model(x)
    y_tr, _ = tr_model(x)

    assert torch.allclose(y_pt, y_tr, rtol=1e-5, atol=1e-6), (
        f"forward mismatch: max abs err={(y_pt - y_tr).abs().max().item():.3e}"
    )

    # Backward: compare gradients for all parameters and inputs
    loss_pt = (y_pt**2).mean()
    loss_tr = (y_tr**2).mean()

    params_pt = [pt_model.nu_log, pt_model.theta_log, pt_model.U1, pt_model.U2, pt_model.V1, pt_model.V2]
    params_tr = [tr_model.nu_log, tr_model.theta_log, tr_model.U1, tr_model.U2, tr_model.V1, tr_model.V2]

    # Compare input gradients first using autograd.grad before freeing graphs
    dx_pt = torch.autograd.grad(loss_pt, x, retain_graph=True, allow_unused=False)[0]
    dx_tr = torch.autograd.grad(loss_tr, x, retain_graph=True, allow_unused=False)[0]

    # Now accumulate parameter grads via .backward on each loss
    loss_pt.backward(retain_graph=True)
    loss_tr.backward()
    assert torch.allclose(dx_pt, dx_tr, rtol=1e-4, atol=1e-5), (
        f"dx mismatch: max abs err={(dx_pt - dx_tr).abs().max().item():.3e}"
    )

    # Compare parameter gradients
    for p_pt, p_tr, name in zip(params_pt, params_tr, ["nu_log", "theta_log", "U1", "U2", "V1", "V2"], strict=False):
        assert p_pt.grad is not None and p_tr.grad is not None
        gt, gb = p_tr.grad.detach(), p_pt.grad.detach()
        assert torch.allclose(gt, gb, rtol=1e-4, atol=1e-5), (
            f"{name} grad mismatch: max abs err={(gt - gb).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="CUDA/Triton not available")
@pytest.mark.parametrize("T", [1, 2, 3, 7, 16, 31, 64, 65, 128])
def test_triton_rtu_with_resets_various_lengths(T: int) -> None:
    from cortex.kernels.triton import LinearRTU_Triton

    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.float32

    B, D, H, R = 3, 16, 12, 6
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

    # Random reset pattern; allow reset at t=0 too
    resets = torch.rand(B, T, device=device) < 0.25

    pt_model = LinearRTU(
        input_size=D,
        hidden_size=H,
        rank=R,
        batch_first=True,
        activation=torch.nn.SiLU(),
    ).to(device=device, dtype=dtype)

    tr_model = LinearRTU_Triton(
        input_size=D,
        hidden_size=H,
        rank=R,
        batch_first=True,
        activation=torch.nn.SiLU(),
    ).to(device=device, dtype=dtype)

    # Align parameters
    with torch.no_grad():
        tr_model.nu_log.copy_(pt_model.nu_log)
        tr_model.theta_log.copy_(pt_model.theta_log)
        tr_model.U1.copy_(pt_model.U1)
        tr_model.U2.copy_(pt_model.U2)
        tr_model.V1.copy_(pt_model.V1)
        tr_model.V2.copy_(pt_model.V2)

    # Forward with resets
    y_pt, _ = pt_model(x, resets=resets)
    y_tr, _ = tr_model(x, resets=resets)

    assert torch.allclose(y_pt, y_tr, rtol=1e-5, atol=1e-6), (
        f"forward mismatch (T={T}): max abs err={(y_pt - y_tr).abs().max().item():.3e}"
    )

    # Gradients (compute dx via autograd.grad first)
    loss_pt = (y_pt**2).mean()
    loss_tr = (y_tr**2).mean()

    dx_pt = torch.autograd.grad(loss_pt, x, retain_graph=True, allow_unused=False)[0]
    dx_tr = torch.autograd.grad(loss_tr, x, retain_graph=True, allow_unused=False)[0]
    assert torch.allclose(dx_pt, dx_tr, rtol=1e-4, atol=1e-5), (
        f"dx mismatch (T={T}): max abs err={(dx_pt - dx_tr).abs().max().item():.3e}"
    )

    # Param grads
    loss_pt.backward(retain_graph=True)
    loss_tr.backward()
    for p_pt, p_tr, name in zip(
        [pt_model.nu_log, pt_model.theta_log, pt_model.U1, pt_model.U2, pt_model.V1, pt_model.V2],
        [tr_model.nu_log, tr_model.theta_log, tr_model.U1, tr_model.U2, tr_model.V1, tr_model.V2],
        ["nu_log", "theta_log", "U1", "U2", "V1", "V2"],
        strict=False,
    ):
        assert p_pt.grad is not None and p_tr.grad is not None
        assert torch.allclose(p_pt.grad, p_tr.grad, rtol=1e-4, atol=1e-5), (
            f"{name} grad mismatch (T={T}): max abs err={(p_pt.grad - p_tr.grad).abs().max().item():.3e}"
        )


torch.manual_seed(7)


def finite_diff_param(
    model: LinearRTU, x: torch.Tensor, loss_fn, param: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Full dense central-difference gradient for a single Parameter tensor."""
    grad = torch.zeros_like(param, dtype=torch.float64)
    with torch.no_grad():
        param_np = param.detach().cpu().numpy()
        it = np.nditer(param_np, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx].item()

            param[idx] = orig + eps
            y_pos, _ = model(x)
            loss_pos = loss_fn(y_pos).item()

            param[idx] = orig - eps
            y_neg, _ = model(x)
            loss_neg = loss_fn(y_neg).item()

            param[idx] = orig  # restore
            grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
            it.iternext()
    return grad


def rtrl_sanity_case(dtype: torch.dtype = torch.float64, device: str = "cpu", verbose: bool = False):
    # Tiny but nontrivial sizes
    B, T, D, H = 2, 5, 3, 4

    model = LinearRTU(
        input_size=D,
        hidden_size=H,
        rank=min(D, H),
        batch_first=True,
        activation=torch.nn.SiLU(),
    ).to(device=device, dtype=dtype)
    model.eval()  # deterministic

    # Small random input
    x = torch.randn(B, T, D, dtype=dtype, device=device)

    # Define a smooth scalar loss of all outputs
    def loss_fn(y: torch.Tensor) -> torch.Tensor:
        # Mean squared activation (smooth, stable in FP64)
        return (y**2).mean()

    # Forward + backward to get analytical grads (standard autograd)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    y, _ = model(x)
    loss = loss_fn(y)
    loss.backward()

    grads_true = {
        "nu_log": model.nu_log.grad.detach().clone(),
        "theta_log": model.theta_log.grad.detach().clone(),
        "U1": model.U1.grad.detach().clone(),
        "U2": model.U2.grad.detach().clone(),
        "V1": model.V1.grad.detach().clone(),
        "V2": model.V2.grad.detach().clone(),
    }

    # Numerical grads (central difference) per parameter
    with torch.no_grad():
        pass

    num_nu = finite_diff_param(model, x, loss_fn, model.nu_log, eps=1e-4)
    num_th = finite_diff_param(model, x, loss_fn, model.theta_log, eps=1e-4)
    num_U1 = finite_diff_param(model, x, loss_fn, model.U1, eps=1e-4)
    num_U2 = finite_diff_param(model, x, loss_fn, model.U2, eps=1e-4)
    num_V1 = finite_diff_param(model, x, loss_fn, model.V1, eps=1e-4)
    num_V2 = finite_diff_param(model, x, loss_fn, model.V2, eps=1e-4)

    grads_num = {
        "nu_log": num_nu,
        "theta_log": num_th,
        "U1": num_U1,
        "U2": num_U2,
        "V1": num_V1,
        "V2": num_V2,
    }

    return grads_true, grads_num


@pytest.mark.parametrize("device", ["cpu"])  # Extend to cuda as needed
def test_linear_rtu_rtrl_grad_matches_finite_difference(device: str, verbose: bool = False) -> None:
    # Use float64 for crisp finite differences
    grads_true, grads_num = rtrl_sanity_case(dtype=torch.float64, device=device, verbose=verbose)

    # Tolerances: FP64 + small eps
    atol = 2e-6
    rtol = 3e-5

    for k in grads_true.keys():
        gt = grads_true[k]
        gn = grads_num[k]

        # Quick sanity norms
        rel_err = (gt - gn).norm() / (gn.norm() + 1e-12)

        # Elementwise closeness
        assert torch.allclose(gt, gn, rtol=rtol, atol=atol), (
            f"{k} grads mismatch\nmax abs err: {(gt - gn).abs().max().item():.3e}\n"
            f"true norm: {gt.norm().item():.3e}  num norm: {gn.norm().item():.3e}"
        )
        assert rel_err.item() < 1e-4, f"{k} relative error too high: {rel_err.item():.3e}"


if __name__ == "__main__":
    # Allow running as a script for quick smoke tests
    torch.set_default_dtype(torch.float64)
    test_linear_rtu_rtrl_grad_matches_finite_difference("cpu", verbose=True)
    print("✓ Linear RTU (low-rank) gradients match finite differences (CPU, float64).")
