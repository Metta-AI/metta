"""Combined AGaLiTe tests: kernels and cell parity/semantics."""

from __future__ import annotations

import os

import pytest
import torch
from cortex.cells.agalite import AGaLiTeCell
from cortex.config import AGaLiTeCellConfig
from cortex.cuda_utils import is_cuda_supported
from cortex.kernels.pytorch.agalite import discounted_sum_pytorch
from tensordict import TensorDict


def _device():
    return torch.device("cuda" if is_cuda_supported() else "cpu")


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
    if not is_cuda_supported():
        pytest.skip("CUDA not available")

    T, B, R, D = shape_case
    device = torch.device("cuda")
    dtype = torch.float32

    from cortex.kernels.cuda.agalite.discounted_sum_cuda import discounted_sum_cuda

    # Run on a non-default stream with a delay to detect stream-unsafe kernel launches.
    x = torch.empty((T, B, R, D), device=device, dtype=dtype)
    discounts = torch.empty_like(x)
    start = torch.empty((B, R, D), device=device, dtype=dtype)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)
        x.normal_()
        discounts.normal_()
        discounts.sigmoid_()
        start.normal_()
        out_cuda = discounted_sum_cuda(start, x, discounts)

    stream.synchronize()

    out_ref = discounted_sum_pytorch(start, x, discounts)

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 3, 7, 64, 256])
def test_agalite_chunked_vs_step_equivalence(chunk: int, dtype) -> None:
    torch.manual_seed(123)
    device = _device()

    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 not supported on this device")

    B, T, H, NH, Dh = 2, 257, 32, 4, 8
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

    # Step-wise processing
    y_steps = []
    st_step = None
    with torch.no_grad():
        for t in range(T):
            y_t, st_step = cell(x[:, t, :], st_step)
            y_steps.append(y_t)
    y_step = torch.stack(y_steps, dim=1)

    # Chunked processing
    y_chunks = []
    st_chunk = None
    with torch.no_grad():
        t = 0
        while t < T:
            x_blk = x[:, t : t + chunk, :]
            y_blk, st_chunk = cell(x_blk, st_chunk)
            y_chunks.append(y_blk)
            t += chunk
    y_chunk = torch.cat(y_chunks, dim=1)

    assert y_chunk.shape == y_step.shape
    y_chunk32 = y_chunk.to(torch.float32)
    y_step32 = y_step.to(torch.float32)
    if dtype is torch.float32:
        torch.testing.assert_close(y_chunk32, y_step32, rtol=1e-3, atol=1e-4)
    elif dtype is torch.float16:
        torch.testing.assert_close(y_chunk32, y_step32, rtol=5e-3, atol=2e-3)
    else:
        max_diff = (y_chunk32 - y_step32).abs().max().item()
        assert max_diff < 0.215, f"bfloat16 chunk vs step max diff {max_diff}"


#! CUDA-only backend parity for discounted sum is covered by kernel test above.
