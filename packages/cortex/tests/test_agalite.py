"""Combined AGaLiTe tests: kernels and cell parity/semantics."""

from __future__ import annotations

import os

import pytest
import torch
from cortex.cells.agalite import AGaLiTeCell
from cortex.config import AGaLiTeCellConfig
from cortex.kernels.pytorch.agalite import discounted_sum_pytorch
from tensordict import TensorDict


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------- Kernel-level tests ----------------------------
# Skip this module entirely by default (slow).
# Set RUN_SLOW_CORTEX_TESTS=1 to enable; otherwise keep skipped.
_RUN_SLOW = os.getenv("RUN_SLOW_CORTEX_TESTS", "0").lower() in {"1", "true", "yes", "y", "on"}
pytestmark = (
    pytest.mark.slow
    if _RUN_SLOW
    else pytest.mark.skip(reason="slow full-rank RTU parity suite (set RUN_SLOW_CORTEX_TESTS=1 to run)")
)


def _randn(shape, device, dtype):
    torch.manual_seed(0)
    return torch.randn(*shape, device=device, dtype=dtype)


@pytest.mark.parametrize("T,B,Nextra", [(8, 3, 1), (4, 2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_discounted_sum_pytorch_shapes_and_broadcast(T: int, B: int, Nextra: int, dtype) -> None:
    device = torch.device("cpu")
    x = _randn((T, B, Nextra), device, dtype)
    discounts = torch.sigmoid(_randn((T, B, Nextra), device, dtype))
    start = _randn((B, Nextra), device, dtype)  # broadcast to x[0]

    out = discounted_sum_pytorch(start, x, discounts)
    assert out.shape == x.shape

    # spot-check recurrence in float32
    out32 = out.to(torch.float32)
    x32 = x.to(torch.float32)
    d32 = discounts.to(torch.float32)
    s32 = start.to(torch.float32)
    cur = d32[0] * s32 + x32[0]
    tol = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out32[0], cur, **tol)
    cur = d32[1] * cur + x32[1]
    torch.testing.assert_close(out32[1], cur, **tol)


@pytest.mark.cuda
@pytest.mark.parametrize("shape_case", [(6, 5, 2, 3), (3, 2, 1, 7)])
def test_discounted_sum_cuda_parity(shape_case) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    T, B, R, D = shape_case
    device = torch.device("cuda")
    dtype = torch.float32

    x = torch.randn(T, B, R, D, device=device, dtype=dtype)
    discounts = torch.sigmoid(torch.randn(T, B, R, D, device=device, dtype=dtype))
    start = torch.randn(B, R, D, device=device, dtype=dtype)

    out_ref = discounted_sum_pytorch(start, x, discounts)

    from cortex.kernels.cuda.agalite.discounted_sum_cuda import discounted_sum_cuda

    out_cuda = discounted_sum_cuda(start, x, discounts)

    torch.testing.assert_close(out_ref, out_cuda, rtol=1e-4, atol=1e-4)

    # Gradient parity
    x.requires_grad_(True)
    discounts.requires_grad_(True)
    start_req = start.detach().clone().requires_grad_(True)
    out_ref = discounted_sum_pytorch(start_req, x, discounts)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    gx_ref = x.grad.detach().clone()
    gd_ref = discounts.grad.detach().clone()
    gs_ref = start_req.grad.detach().clone()

    x.grad = None
    discounts.grad = None
    start_req.grad = None
    out_cuda = discounted_sum_cuda(start_req, x, discounts)
    loss_cuda = out_cuda.sum()
    loss_cuda.backward()

    torch.testing.assert_close(x.grad, gx_ref, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(discounts.grad, gd_ref, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(start_req.grad, gs_ref, rtol=1e-3, atol=1e-2)


# ----------------------------- Cell-level tests -----------------------------


def test_agalite_sequence_shapes_and_state() -> None:
    torch.manual_seed(42)
    device = _device()
    dtype = torch.float32

    B, T, H, NH, Dh = 2, 5, 32, 4, 8
    assert H == NH * Dh
    cfg = AGaLiTeCellConfig(
        hidden_size=H,
        n_heads=NH,
        head_dim=Dh,
        eta=3,
        r=2,
        eps=1e-5,
        dropout=0.0,
    )
    cell = AGaLiTeCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    y, state = cell(x, state=None)

    assert y.shape == (B, T, H)
    assert state is not None and isinstance(state, TensorDict)
    assert set(["tilde_k", "tilde_v", "s", "tick"]) <= set(state.keys())


def test_agalite_step_vs_sequence_equivalence() -> None:
    torch.manual_seed(7)
    device = _device()
    dtype = torch.float32

    B, T, H, NH, Dh = 2, 6, 32, 4, 8
    cfg = AGaLiTeCellConfig(
        hidden_size=H,
        n_heads=NH,
        head_dim=Dh,
        eta=3,
        r=2,
        eps=1e-5,
        dropout=0.0,
    )
    cell = AGaLiTeCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    with torch.no_grad():
        y_seq, _ = cell(x, state=None)

    y_steps = []
    st = None
    with torch.no_grad():
        for t in range(T):
            y_t, st = cell(x[:, t, :], st)
            y_steps.append(y_t)
    y_step = torch.stack(y_steps, dim=1)

    torch.testing.assert_close(y_seq, y_step, rtol=5e-4, atol=5e-4)


#! CUDA-only backend parity for discounted sum is covered by kernel test above.
