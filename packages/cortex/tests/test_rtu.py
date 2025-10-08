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
            f"{k} grads mismatch\nmax abs err: {(gt-gn).abs().max().item():.3e}\n"
            f"true norm: {gt.norm().item():.3e}  num norm: {gn.norm().item():.3e}"
        )
        assert rel_err.item() < 1e-4, f"{k} relative error too high: {rel_err.item():.3e}"


if __name__ == "__main__":
    # Allow running as a script for quick smoke tests
    torch.set_default_dtype(torch.float64)
    test_linear_rtu_rtrl_grad_matches_finite_difference("cpu", verbose=True)
    print("✓ Linear RTU (low-rank) gradients match finite differences (CPU, float64).")
