"""Finite-differences parity tests for full‑rank streaming RTU (PyTorch).

This validates that autograd gradients from the reference full‑rank streaming
kernel match central finite differences for a simple quadratic loss.

The full‑rank kernel uses [D, H] input maps (W1, W2) rather than diagonal
weights. We test both with and without resets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cortex.kernels.pytorch.rtu.rtu_stream_fullrank import rtu_stream_full_pytorch

# Skip this module entirely (slow)
pytestmark = pytest.mark.skip(reason="slow full-rank RTU parity suite")

try:
    from cortex.kernels.cuda import rtu_stream_full_cuda_seq_allin as _rtu_full_cuda

    _HAS_FULL_CUDA = True
except Exception:  # pragma: no cover
    _HAS_FULL_CUDA = False


def _build_params_full(D: int, H: int, *, device, dtype):
    torch.manual_seed(123)
    nu_log = torch.randn(H, device=device, dtype=dtype, requires_grad=True) * 0.05
    theta_log = torch.randn(H, device=device, dtype=dtype, requires_grad=True) * 0.05
    W1 = torch.randn(D, H, device=device, dtype=dtype, requires_grad=True) * (1.0 / max(1, D) ** 0.5)
    W2 = torch.randn(D, H, device=device, dtype=dtype, requires_grad=True) * (1.0 / max(1, D) ** 0.5)
    return nu_log, theta_log, W1, W2


def _central_diff_grad(forward_fn, tensor: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    grad = torch.zeros_like(tensor, dtype=torch.float64)
    with torch.no_grad():
        it = np.nditer(tensor.detach().cpu().numpy(), flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = tensor[idx].item()
            tensor[idx] = orig + eps
            loss_pos = forward_fn().item()
            tensor[idx] = orig - eps
            loss_neg = forward_fn().item()
            tensor[idx] = orig
            grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
            it.iternext()
    return grad


def _forward_whole_fullrank(x, params, activation: str, resets_bt=None):
    nu_log, theta_log, W1, W2 = params
    B, _, _ = x.shape
    H = nu_log.shape[0]
    hc1_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    hc2_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    y, (_h1, _h2), _ = rtu_stream_full_pytorch(
        x_btd=x,
        nu_log=nu_log,
        theta_log=theta_log,
        Wc1=W1,
        Wc2=W2,
        activation_name=activation,
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        resets_bt=resets_bt,
    )
    return y


@pytest.mark.parametrize("with_resets", [False, True])
def test_fullrank_grads_match_finite_differences(with_resets: bool) -> None:
    torch.manual_seed(11)
    device = torch.device("cpu")
    dtype = torch.float64

    B, T, D, H = 1, 7, 5, 3
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=False)
    resets = None
    if with_resets:
        # Deterministic pattern to exercise boundary handling
        resets = torch.zeros(B, T, dtype=torch.bool, device=device)
        resets[:, 0] = False
        if T > 2:
            resets[:, 2] = True
        if T > 5:
            resets[:, 5] = True

    params = _build_params_full(D, H, device=device, dtype=dtype)
    nu_log, theta_log, W1, W2 = params

    def loss_whole():
        y = _forward_whole_fullrank(x, params, activation="SiLU", resets_bt=resets)
        return (y**2).mean()

    # Autograd gradients
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    loss = loss_whole()
    g_auto = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)

    # Finite-difference gradients (subset: nu_log, W1, W2)
    eps = 1e-5
    num_nu = _central_diff_grad(lambda: loss_whole(), nu_log, eps=eps)
    num_W1 = _central_diff_grad(lambda: loss_whole(), W1, eps=eps)
    num_W2 = _central_diff_grad(lambda: loss_whole(), W2, eps=eps)

    # Tolerances (64-bit improves stability; allow slightly looser with resets)
    if with_resets:
        tol = {
            "nu_log": (2e-3, 5e-3),
            "W": (2e-4, 5e-5),
        }
    else:
        tol = {
            "nu_log": (1e-3, 3e-3),
            "W": (1e-4, 3e-5),
        }

    atol, rtol = tol["nu_log"]
    assert torch.allclose(g_auto[0].to(torch.float64), num_nu, atol=atol, rtol=rtol)

    atolW, rtolW = tol["W"]
    assert torch.allclose(g_auto[2].to(torch.float64), num_W1, atol=atolW, rtol=rtolW)
    assert torch.allclose(g_auto[3].to(torch.float64), num_W2, atol=atolW, rtol=rtolW)


# ------------------------------
# CUDA parity (skipped if no GPU)
# ------------------------------


def _run_full_pytorch(x, params, activation: str, resets_bt=None):
    nu_log, theta_log, W1, W2 = params
    B, _, _ = x.shape
    H = nu_log.shape[0]
    hc1_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    hc2_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    return rtu_stream_full_pytorch(
        x_btd=x,
        nu_log=nu_log,
        theta_log=theta_log,
        Wc1=W1,
        Wc2=W2,
        activation_name=activation,
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets_bt,
    )


def _run_full_cuda(x, params, activation: str, resets_bt=None):
    nu_log, theta_log, W1, W2 = params
    B, _, _ = x.shape
    H = nu_log.shape[0]
    hc1_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    hc2_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    return _rtu_full_cuda(
        x_btd=x,
        nu_log=nu_log,
        theta_log=theta_log,
        Wc1=W1,
        Wc2=W2,
        activation_name=activation,
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets_bt,
    )


def _run_chunks_pytorch_full(x, params, activation: str, resets_bt=None, chunks=(5, 1000)):
    nu_log, theta_log, W1, W2 = params
    B, T, _ = x.shape
    H = nu_log.shape[0]
    hc1 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    hc2 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    trace = None
    ys = []
    t0 = 0
    for sz in chunks:
        t1 = min(T, t0 + sz)
        if t1 <= t0:
            break
        y_blk, (hc1, hc2), trace = rtu_stream_full_pytorch(
            x_btd=x[:, t0:t1, :],
            nu_log=nu_log,
            theta_log=theta_log,
            Wc1=W1,
            Wc2=W2,
            activation_name=activation,
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace,
            resets_bt=None if resets_bt is None else resets_bt[:, t0:t1],
        )
        ys.append(y_blk)
        hc1, hc2 = hc1.detach(), hc2.detach()
        trace = tuple(t.detach() for t in trace) if trace is not None else None
        t0 = t1
        if t0 >= T:
            break
    return torch.cat(ys, dim=1)


def _run_chunks_cuda_full(x, params, activation: str, resets_bt=None, chunks=(5, 1000)):
    nu_log, theta_log, W1, W2 = params
    B, T, _ = x.shape
    H = nu_log.shape[0]
    hc1 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    hc2 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    trace = None
    ys = []
    t0 = 0
    for sz in chunks:
        t1 = min(T, t0 + sz)
        if t1 <= t0:
            break
        y_blk, (hc1, hc2), trace = _rtu_full_cuda(
            x_btd=x[:, t0:t1, :],
            nu_log=nu_log,
            theta_log=theta_log,
            Wc1=W1,
            Wc2=W2,
            activation_name=activation,
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace,
            resets_bt=None if resets_bt is None else resets_bt[:, t0:t1],
        )
        ys.append(y_blk)
        hc1, hc2 = hc1.detach(), hc2.detach()
        trace = tuple(t.detach() for t in trace) if trace is not None else None
        t0 = t1
        if t0 >= T:
            break
    return torch.cat(ys, dim=1)


@pytest.mark.skipif(not _HAS_FULL_CUDA or not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("with_resets", [False, True])
def test_cuda_fullrank_full_forward_and_grad_parity(with_resets: bool) -> None:
    torch.manual_seed(101)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, D, H = 2, 17, 6, 5
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)
    resets = None
    if with_resets:
        resets = torch.rand(B, T, device=device) < 0.2

    nu0, th0, W10, W20 = _build_params_full(D, H, device=device, dtype=dtype)

    # PyTorch reference (full)
    nu_pt = nu0.clone().detach().requires_grad_(True)
    th_pt = th0.clone().detach().requires_grad_(True)
    W1_pt = W10.clone().detach().requires_grad_(True)
    W2_pt = W20.clone().detach().requires_grad_(True)
    y_pt, (h1_pt, h2_pt), _ = _run_full_pytorch(x, (nu_pt, th_pt, W1_pt, W2_pt), "SiLU", resets)
    loss_pt = (y_pt**2).mean()
    g_pt = torch.autograd.grad(loss_pt, (nu_pt, th_pt, W1_pt, W2_pt, x), retain_graph=True)

    # CUDA (full)
    nu_cu = nu0.clone().detach().requires_grad_(True)
    th_cu = th0.clone().detach().requires_grad_(True)
    W1_cu = W10.clone().detach().requires_grad_(True)
    W2_cu = W20.clone().detach().requires_grad_(True)
    y_cu, (h1_cu, h2_cu), _ = _run_full_cuda(x, (nu_cu, th_cu, W1_cu, W2_cu), "SiLU", resets)
    loss_cu = (y_cu**2).mean()
    g_cu = torch.autograd.grad(loss_cu, (nu_cu, th_cu, W1_cu, W2_cu, x), retain_graph=True)

    # Forward parity
    assert torch.allclose(y_pt, y_cu, rtol=2e-5, atol=1e-6)
    assert torch.allclose(h1_pt, h1_cu, rtol=2e-5, atol=1e-6)
    assert torch.allclose(h2_pt, h2_cu, rtol=2e-5, atol=1e-6)

    # Gradient parity
    names = ["nu_log", "theta_log", "W1", "W2", "x"]
    tolerances = {
        "nu_log": (5e-4, 2e-5),
        "theta_log": (5e-4, 2e-5),
        "W1": (5e-5, 1e-6),
        "W2": (5e-5, 1e-6),
        "x": (6e-5, 2e-6),
    }
    for gp, gc, nm in zip(g_pt, g_cu, names, strict=False):
        rtol, atol = tolerances[nm]
        assert torch.allclose(gp, gc, rtol=rtol, atol=atol), (
            f"grad {nm} mismatch: max diff={(gp - gc).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_FULL_CUDA or not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("with_resets", [False, True])
def test_cuda_fullrank_chunked_forward_and_grad_parity(with_resets: bool) -> None:
    torch.manual_seed(202)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, D, H = 2, 23, 6, 5
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)
    if with_resets:
        # Two masks: random and deterministic (to hit chunk boundaries)
        resets_rand = torch.rand(B, T, device=device) < 0.25
        resets_det = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            resets_det[b, 0] = False
            for t_idx in (4, 7, 15):
                if t_idx < T:
                    resets_det[b, t_idx] = True
        resets_list = [resets_rand, resets_det]
    else:
        resets_list = [None]

    nu0, th0, W10, W20 = _build_params_full(D, H, device=device, dtype=dtype)

    def run_py(chunks, resets):
        nu = nu0.clone().detach().requires_grad_(True)
        th = th0.clone().detach().requires_grad_(True)
        W1 = W10.clone().detach().requires_grad_(True)
        W2 = W20.clone().detach().requires_grad_(True)
        y = _run_chunks_pytorch_full(x, (nu, th, W1, W2), activation="SiLU", resets_bt=resets, chunks=chunks)
        loss = (y**2).mean()
        grads = torch.autograd.grad(loss, (nu, th, W1, W2, x), retain_graph=True)
        return y, grads

    def run_cu(chunks, resets):
        nu = nu0.clone().detach().requires_grad_(True)
        th = th0.clone().detach().requires_grad_(True)
        W1 = W10.clone().detach().requires_grad_(True)
        W2 = W20.clone().detach().requires_grad_(True)
        y = _run_chunks_cuda_full(x, (nu, th, W1, W2), activation="SiLU", resets_bt=resets, chunks=chunks)
        loss = (y**2).mean()
        grads = torch.autograd.grad(loss, (nu, th, W1, W2, x), retain_graph=True)
        return y, grads

    for resets in resets_list:
        for chunks in [(T,), (7, 16), (5, 9, 9)]:
            y_py, g_py = run_py(chunks, resets)
            y_cu, g_cu = run_cu(chunks, resets)

            assert y_py.shape == y_cu.shape
            assert torch.allclose(y_py, y_cu, rtol=2e-5, atol=1e-6)

            names = ["nu_log", "theta_log", "W1", "W2", "x"]
            tolerances = {
                "nu_log": (6e-4, 3e-5),
                "theta_log": (6e-4, 3e-5),
                "W1": (8e-5, 2e-6),
                "W2": (8e-5, 2e-6),
                "x": (1e-4, 4e-6),
            }
            for gp, gc, nm in zip(g_py, g_cu, names, strict=False):
                rtol, atol = tolerances[nm]
                assert torch.allclose(gp, gc, rtol=rtol, atol=atol), (
                    f"grad {nm} mismatch chunks={chunks}: max diff={(gp - gc).abs().max().item():.3e}"
                )


def _forward_stream_chunks_fullrank(x, params, activation: str, resets_bt=None, chunks=(3, 1000)):
    nu_log, theta_log, W1, W2 = params
    B, T, _ = x.shape
    H = nu_log.shape[0]
    hc1 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    hc2 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    trace = None
    ys = []
    t0 = 0
    for size in chunks:
        t1 = min(T, t0 + size)
        if t1 <= t0:
            break
        x_blk = x[:, t0:t1, :]
        res_blk = None if resets_bt is None else resets_bt[:, t0:t1]
        y_blk, (hc1, hc2), trace = rtu_stream_full_pytorch(
            x_btd=x_blk,
            nu_log=nu_log,
            theta_log=theta_log,
            Wc1=W1,
            Wc2=W2,
            activation_name=activation,
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace,
            resets_bt=res_blk,
        )
        ys.append(y_blk)
        # detach carry for real streaming
        hc1 = hc1.detach()
        hc2 = hc2.detach()
        if trace is not None:
            trace = tuple(t.detach() for t in trace)
        t0 = t1
        if t0 >= T:
            break
    y = torch.cat(ys, dim=1)
    return y


@pytest.mark.parametrize("with_resets", [False, True])
def test_fullrank_streaming_forward_matches_whole(with_resets: bool) -> None:
    torch.manual_seed(21)
    device = torch.device("cpu")
    dtype = torch.float32

    B, T, D, H = 2, 13, 6, 5
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)
    resets = None
    if with_resets:
        resets = torch.rand(B, T, device=device) < 0.3

    params = _build_params_full(D, H, device=device, dtype=dtype)
    activation = "SiLU"

    y_whole = _forward_whole_fullrank(x, params, activation, resets_bt=resets)
    for chunks in [(T,), (3, T), (1, 2, 3, 4, 100), (6, 7)]:
        y_stream = _forward_stream_chunks_fullrank(x, params, activation, resets_bt=resets, chunks=chunks)
        assert y_whole.shape == y_stream.shape
        max_diff = (y_whole - y_stream).abs().max().item()
        assert torch.allclose(y_whole, y_stream, rtol=1e-5, atol=1e-6), (
            f"forward mismatch (resets={with_resets}, chunks={chunks}), max diff {max_diff:.3e}"
        )


@pytest.mark.parametrize("with_resets", [False, True])
def test_fullrank_streaming_grads_match_whole(with_resets: bool) -> None:
    torch.manual_seed(22)
    device = torch.device("cpu")
    dtype = torch.float64

    B, T, D, H = 1, 10, 5, 4
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    resets = None
    if with_resets:
        resets = torch.zeros(B, T, dtype=torch.bool, device=device)
        resets[:, 0] = False
        if T > 3:
            resets[:, 3] = True
        if T > 7:
            resets[:, 7] = True

    nu0, th0, W10, W20 = _build_params_full(D, H, device=device, dtype=dtype)

    # Whole sequence autograd grads
    def grads_whole():
        nu = nu0.clone().detach().requires_grad_(True)
        th = th0.clone().detach().requires_grad_(True)
        W1 = W10.clone().detach().requires_grad_(True)
        W2 = W20.clone().detach().requires_grad_(True)
        y = _forward_whole_fullrank(x, (nu, th, W1, W2), activation="SiLU", resets_bt=resets)
        loss = (y**2).mean()
        return torch.autograd.grad(loss, (nu, th, W1, W2, x), retain_graph=True)

    # Streaming in chunks with carries detached
    def grads_stream(chunks=(4, 6)):
        nu = nu0.clone().detach().requires_grad_(True)
        th = th0.clone().detach().requires_grad_(True)
        W1 = W10.clone().detach().requires_grad_(True)
        W2 = W20.clone().detach().requires_grad_(True)
        Hloc = nu.shape[0]
        hc1 = torch.zeros(B, Hloc, device=device, dtype=dtype)
        hc2 = torch.zeros(B, Hloc, device=device, dtype=dtype)
        trace = None
        ys = []
        t0 = 0
        for sz in chunks:
            t1 = min(T, t0 + sz)
            if t1 <= t0:
                break
            y_blk, (hc1, hc2), trace = rtu_stream_full_pytorch(
                x_btd=x[:, t0:t1, :],
                nu_log=nu,
                theta_log=th,
                Wc1=W1,
                Wc2=W2,
                activation_name="SiLU",
                hc1_init_bh=hc1,
                hc2_init_bh=hc2,
                trace_in=trace,
                resets_bt=None if resets is None else resets[:, t0:t1],
            )
            ys.append(y_blk)
            hc1, hc2 = hc1.detach(), hc2.detach()
            trace = tuple(t.detach() for t in trace) if trace is not None else None
            t0 = t1
            if t0 >= T:
                break
        y_s = torch.cat(ys, dim=1)
        loss = (y_s**2).mean()
        return torch.autograd.grad(loss, (nu, th, W1, W2, x), retain_graph=True)

    g_w = grads_whole()
    g_s = grads_stream()

    names = ["nu_log", "theta_log", "W1", "W2", "x"]
    # Slightly relaxed tolerances when resets are present
    for gw, gs, nm in zip(g_w, g_s, names, strict=False):
        if nm in {"nu_log", "theta_log"} and with_resets:
            rtol, atol = 2e-2, 4e-3
        elif nm in {"nu_log", "theta_log"}:
            rtol, atol = 5e-3, 1e-3
        elif nm in {"W1", "W2"} and with_resets:
            rtol, atol = 3e-3, 3e-4
        elif nm in {"W1", "W2"}:
            rtol, atol = 2e-3, 2e-4
        elif nm == "x":
            # x-gradients can differ slightly at chunk boundaries due to carry detach;
            # boundary corrections restore parameter grads exactly but input grads may
            # show tiny numerical discrepancies. Use modest tolerances.
            rtol, atol = 8e-3, 3e-3
        else:
            rtol, atol = 2e-3, 2e-4
        assert torch.allclose(gw, gs, rtol=rtol, atol=atol), (
            f"grad parity failed for {nm} (resets={with_resets}); max diff={(gw - gs).abs().max().item():.3e}"
        )
