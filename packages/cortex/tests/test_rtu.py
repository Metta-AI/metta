"""RTU kernel tests (functional API via cell) for PyTorch and Triton backends.

Includes:
- Triton vs PyTorch parity (forward + gradients), with and without resets
- Finite-difference check of PyTorch autograd on a tiny problem
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cortex.cells.rtu import RTUCell
from cortex.config import RTUCellConfig
from cortex.utils import TRITON_AVAILABLE


@pytest.mark.skipif(not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="CUDA/Triton not available")
def test_triton_rtu_matches_pytorch_forward_and_gradients() -> None:
    torch.manual_seed(2024)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, H, R = 2, 64, 24, 8
    x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)

    # Build cell (parameters live in the cell)
    cell = RTUCell(RTUCellConfig(hidden_size=H, rank=R, activation="SiLU")).to(device=device, dtype=dtype)

    # Utility to force backend via monkeypatch of select_backend
    import cortex.cells.rtu as cell_mod

    def run_with_backend(which: str):
        orig = cell_mod.select_backend

        def chooser(*, triton_fn, pytorch_fn, tensor, allow_triton=True):  # type: ignore[override]
            return triton_fn if which == "triton" else pytorch_fn

        try:
            cell_mod.select_backend = chooser  # type: ignore[assignment]
            y, _ = cell(x, state=None, resets=None)
            return y
        finally:
            cell_mod.select_backend = orig  # type: ignore[assignment]

    y_pt = run_with_backend("pytorch")
    y_tr = run_with_backend("triton")

    assert torch.allclose(y_pt, y_tr, rtol=1e-5, atol=1e-6), (
        f"forward mismatch: max abs err={(y_pt - y_tr).abs().max().item():.3e}"
    )

    loss_pt = (y_pt**2).mean()
    loss_tr = (y_tr**2).mean()

    params = [cell.nu_log, cell.theta_log, cell.U1, cell.U2, cell.V1, cell.V2, x]
    g_pt = torch.autograd.grad(loss_pt, params, retain_graph=True, allow_unused=False)
    g_tr = torch.autograd.grad(loss_tr, params, retain_graph=True, allow_unused=False)

    dx_pt, dx_tr = g_pt[-1], g_tr[-1]
    assert torch.allclose(dx_pt, dx_tr, rtol=1e-4, atol=1e-5), (
        f"dx mismatch: max abs err={(dx_pt - dx_tr).abs().max().item():.3e}"
    )

    for gt, gb, name in zip(g_tr[:-1], g_pt[:-1], ["nu_log", "theta_log", "U1", "U2", "V1", "V2"], strict=False):
        rtol = 1e-3 if name in {"nu_log", "theta_log"} else 1e-4
        atol = 3e-4 if name in {"nu_log", "theta_log"} else 1e-5
        assert torch.allclose(gt, gb, rtol=rtol, atol=atol), (
            f"{name} grad mismatch: max abs err={(gt - gb).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="CUDA/Triton not available")
@pytest.mark.parametrize("T", [1, 2, 3, 7, 16, 31, 64, 65, 128])
def test_triton_rtu_with_resets_various_lengths(T: int) -> None:
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.float32

    B, H, R = 3, 12, 6
    x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)

    # Random reset pattern; allow reset at t=0 too
    resets = torch.rand(B, T, device=device) < 0.25

    cell = RTUCell(RTUCellConfig(hidden_size=H, rank=R, activation="SiLU")).to(device=device, dtype=dtype)

    import cortex.cells.rtu as cell_mod

    def run_with_backend(which: str):
        orig = cell_mod.select_backend

        def chooser(*, triton_fn, pytorch_fn, tensor, allow_triton=True):  # type: ignore[override]
            return triton_fn if which == "triton" else pytorch_fn

        try:
            cell_mod.select_backend = chooser  # type: ignore[assignment]
            y, _ = cell(x, state=None, resets=resets)
            return y
        finally:
            cell_mod.select_backend = orig  # type: ignore[assignment]

    y_pt = run_with_backend("pytorch")
    y_tr = run_with_backend("triton")

    assert torch.allclose(y_pt, y_tr, rtol=1e-5, atol=1e-6), (
        f"forward mismatch (T={T}): max abs err={(y_pt - y_tr).abs().max().item():.3e}"
    )

    loss_pt = (y_pt**2).mean()
    loss_tr = (y_tr**2).mean()
    dx_pt = torch.autograd.grad(loss_pt, x, retain_graph=True, allow_unused=False)[0]
    dx_tr = torch.autograd.grad(loss_tr, x, retain_graph=True, allow_unused=False)[0]
    assert torch.allclose(dx_pt, dx_tr, rtol=1e-4, atol=1e-5), (
        f"dx mismatch (T={T}): max abs err={(dx_pt - dx_tr).abs().max().item():.3e}"
    )

    grads_pt = torch.autograd.grad(
        loss_pt, [cell.nu_log, cell.theta_log, cell.U1, cell.U2, cell.V1, cell.V2], retain_graph=True
    )
    grads_tr = torch.autograd.grad(
        loss_tr, [cell.nu_log, cell.theta_log, cell.U1, cell.U2, cell.V1, cell.V2], retain_graph=True
    )
    for gp, gt, name in zip(grads_pt, grads_tr, ["nu_log", "theta_log", "U1", "U2", "V1", "V2"], strict=False):
        rtol = 1e-3 if name in {"nu_log", "theta_log"} else 1e-4
        atol = 3e-4 if name in {"nu_log", "theta_log"} else 1e-5
        assert torch.allclose(gp, gt, rtol=rtol, atol=atol), (
            f"{name} grad mismatch (T={T}): max abs err={(gp - gt).abs().max().item():.3e}"
        )


torch.manual_seed(7)


def finite_diff_param_fn(
    x: torch.Tensor, loss_fn, param: torch.Tensor, *, cell: RTUCell, eps: float = 1e-4
) -> torch.Tensor:
    """Full dense central-difference gradient for a single tensor parameter using the cell."""
    grad = torch.zeros_like(param, dtype=torch.float64)
    with torch.no_grad():
        param_np = param.detach().cpu().numpy()
        it = np.nditer(param_np, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx].item()

            param[idx] = orig + eps
            y_pos, _ = cell(x, state=None)
            loss_pos = loss_fn(y_pos).item()

            param[idx] = orig - eps
            y_neg, _ = cell(x, state=None)
            loss_neg = loss_fn(y_neg).item()

            param[idx] = orig  # restore
            grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
            it.iternext()
    return grad


def rtrl_sanity_case(dtype: torch.dtype = torch.float64, device: str = "cpu", verbose: bool = False):
    # Tiny but nontrivial sizes
    B, T, H = 2, 5, 4
    rank = H

    x = torch.randn(B, T, H, dtype=dtype, device=device)

    # Build cell on CPU/dtype
    cell = RTUCell(RTUCellConfig(hidden_size=H, rank=rank, activation="SiLU")).to(device=device, dtype=dtype)

    # Force PyTorch backend in this test
    import cortex.cells.rtu as cell_mod

    def chooser(*, triton_fn, pytorch_fn, tensor, allow_triton=True):  # type: ignore[override]
        return pytorch_fn

    orig = cell_mod.select_backend
    cell_mod.select_backend = chooser  # type: ignore[assignment]

    # Define a smooth scalar loss of all outputs
    def loss_fn(y: torch.Tensor) -> torch.Tensor:
        return (y**2).mean()

    # Forward + backward to get analytical grads (standard autograd)
    y, _ = cell(x, state=None)
    loss = loss_fn(y)
    grads = torch.autograd.grad(loss, [cell.nu_log, cell.theta_log, cell.U1, cell.U2, cell.V1, cell.V2])

    grads_true = {
        "nu_log": grads[0].detach().clone(),
        "theta_log": grads[1].detach().clone(),
        "U1": grads[2].detach().clone(),
        "U2": grads[3].detach().clone(),
        "V1": grads[4].detach().clone(),
        "V2": grads[5].detach().clone(),
    }

    # Numerical grads (central difference)
    num_nu = finite_diff_param_fn(x, loss_fn, cell.nu_log, cell=cell)
    num_th = finite_diff_param_fn(x, loss_fn, cell.theta_log, cell=cell)
    num_U1 = finite_diff_param_fn(x, loss_fn, cell.U1, cell=cell)
    num_U2 = finite_diff_param_fn(x, loss_fn, cell.U2, cell=cell)
    num_V1 = finite_diff_param_fn(x, loss_fn, cell.V1, cell=cell)
    num_V2 = finite_diff_param_fn(x, loss_fn, cell.V2, cell=cell)

    grads_num = {
        "nu_log": num_nu,
        "theta_log": num_th,
        "U1": num_U1,
        "U2": num_U2,
        "V1": num_V1,
        "V2": num_V2,
    }

    cell_mod.select_backend = orig  # restore
    return grads_true, grads_num


@pytest.mark.parametrize("device", ["cpu"])  # Extend to cuda as needed
def test_linear_rtu_rtrl_grad_matches_finite_difference(device: str, verbose: bool = False) -> None:
    grads_true, grads_num = rtrl_sanity_case(dtype=torch.float64, device=device, verbose=verbose)

    atol = 2e-6
    rtol = 3e-5

    for k in grads_true.keys():
        gt = grads_true[k]
        gn = grads_num[k]
        rel_err = (gt - gn).norm() / (gn.norm() + 1e-12)
        assert torch.allclose(gt, gn, rtol=rtol, atol=atol), (
            f"{k} grads mismatch\nmax abs err: {(gt - gn).abs().max().item():.3e}\n"
            f"true norm: {gt.norm().item():.3e}  num norm: {gn.norm().item():.3e}"
        )
        assert rel_err.item() < 1e-4, f"{k} relative error too high: {rel_err.item():.3e}"


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    test_linear_rtu_rtrl_grad_matches_finite_difference("cpu", verbose=True)
    print("✓ Linear RTU (low-rank) gradients match finite differences (CPU, float64).")
