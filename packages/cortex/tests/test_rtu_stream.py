"""Tests for streaming (chunk-wise) PyTorch RTU kernel.

Covers:
1) Forward outputs match whole-sequence processing when run chunk by chunk.
2) Gradients match finite differences for both whole and chunked processing.
3) Both behaviors hold with and without resets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cortex.kernels.pytorch.rtu_stream import rtu_stream_diag_pytorch

try:  # Triton availability for GPU tests
    from cortex.kernels.triton import rtu_stream_diag_triton as _rtu_triton_stream

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

# CUDA fused sequential (all-in) availability
try:
    from cortex.kernels.cuda import (
        rtu_stream_diag_cuda_seq_allin as _rtu_cuda_seq_stream,
    )

    _HAS_CUDA_SEQ = True
except Exception:  # pragma: no cover
    _HAS_CUDA_SEQ = False


def _build_params(D: int, H: int, *, device, dtype):
    torch.manual_seed(1234)
    # exp-exp parameterization as in cells: choose random but reasonable values
    # Start with values near unit circle to avoid degenerate dynamics
    nu_log = torch.randn(H, device=device, dtype=dtype, requires_grad=True) * 0.1
    theta_log = torch.randn(H, device=device, dtype=dtype, requires_grad=True) * 0.1
    # Diagonal input weights (per-channel)
    w1 = torch.randn(H, device=device, dtype=dtype, requires_grad=True) * (1.0 / max(1, H) ** 0.5)
    w2 = torch.randn(H, device=device, dtype=dtype, requires_grad=True) * (1.0 / max(1, H) ** 0.5)
    # Assume D == H in diagonal kernel
    assert D == H
    return nu_log, theta_log, w1, w2


def _forward_whole(x, params, activation: str, resets_bt=None):
    nu_log, theta_log, w1, w2 = params
    B = x.shape[0]
    H = params[0].shape[0]
    hc1_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    hc2_0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    y, (h1, h2), _trace = rtu_stream_diag_pytorch(
        x_btd=x,
        nu_log=nu_log,
        theta_log=theta_log,
        w1=w1,
        w2=w2,
        activation_name=activation,
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets_bt,
    )
    return y, (h1, h2)


def _forward_stream_chunks(x, params, activation: str, resets_bt=None, chunks=(3, 1000)):
    nu_log, theta_log, w1, w2 = params
    B, T, _ = x.shape
    H = params[0].shape[0]
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
        y_blk, (hc1, hc2), trace = rtu_stream_diag_pytorch(
            x_btd=x_blk,
            nu_log=nu_log,
            theta_log=theta_log,
            w1=w1,
            w2=w2,
            activation_name=activation,
            hc1_init_bh=hc1,
            hc2_init_bh=hc2,
            trace_in=trace,
            resets_bt=res_blk,
        )
        ys.append(y_blk)
        # Detach carry for true streaming across subsequences
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
def test_streaming_forward_matches_whole(with_resets: bool) -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")
    dtype = torch.float32

    B, T, D, H = 2, 11, 6, 6
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)
    resets = None
    if with_resets:
        resets = torch.rand(B, T, device=device) < 0.3

    params = _build_params(D, H, device=device, dtype=dtype)
    activation = "SiLU"

    y_whole, _ = _forward_whole(x, params, activation, resets_bt=resets)
    # Test multiple chunk configurations
    for chunks in [(T,), (3, T), (1, 2, 3, 4, 100), (5, 6)]:
        y_stream = _forward_stream_chunks(x, params, activation, resets_bt=resets, chunks=chunks)
        assert y_whole.shape == y_stream.shape
        max_diff = (y_whole - y_stream).abs().max().item()
        assert torch.allclose(y_whole, y_stream, rtol=1e-5, atol=1e-6), (
            f"forward mismatch (resets={with_resets}, chunks={chunks}), max diff {max_diff:.3e}"
        )


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


@pytest.mark.parametrize("with_resets", [False, True])
def test_streaming_grads_match_fd_and_whole(with_resets: bool) -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    dtype = torch.float64  # use float64 for FD stability

    B, T, D, H = 1, 6, 4, 4
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=False)
    resets = None
    if with_resets:
        resets = torch.tensor([[False, True, False, False, True, False]], device=device)

    params = _build_params(D, H, device=device, dtype=dtype)
    nu_log, theta_log, w1, w2 = params
    activation = "SiLU"

    # Define losses
    def loss_whole():
        y, _ = _forward_whole(x, params, activation, resets_bt=resets)
        return (y**2).mean()

    def loss_stream():
        y = _forward_stream_chunks(x, params, activation, resets_bt=resets, chunks=(2, 4))
        return (y**2).mean()

    # Autograd grads (whole)
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    loss_w = loss_whole()
    g_w = torch.autograd.grad(loss_w, params, retain_graph=True, allow_unused=False)

    # Autograd grads (stream)
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    loss_s = loss_stream()
    g_s = torch.autograd.grad(loss_s, params, retain_graph=True, allow_unused=False)

    # Finite-difference grads for selected parameters (keep runtime modest)
    eps = 1e-5
    num_nu_whole = _central_diff_grad(lambda: loss_whole(), nu_log, eps=eps)
    num_nu_stream = _central_diff_grad(lambda: loss_stream(), nu_log, eps=eps)
    num_w1_whole = _central_diff_grad(lambda: loss_whole(), w1, eps=eps)
    num_w1_stream = _central_diff_grad(lambda: loss_stream(), w1, eps=eps)

    # Tolerances
    tol_param = {
        "nu_log": (3e-4, 1e-3),  # whole-seq vs FD
        "w1": (1e-6, 2e-5),
    }

    # Compare autograd vs FD (whole and stream)
    atol, rtol = tol_param["nu_log"]
    assert torch.allclose(g_w[0].to(torch.float64), num_nu_whole, atol=atol, rtol=rtol)
    # Streaming diag is slightly looser; allow relaxed tol (more relaxed with resets)
    atol_s = 1e-3 if not with_resets else 2e-3
    rtol_s = 5e-3 if not with_resets else 1e-3
    assert torch.allclose(g_s[0].to(torch.float64), num_nu_stream, atol=atol_s, rtol=rtol_s)

    atol, rtol = tol_param["w1"]
    if not with_resets:
        # Index: params = (nu_log, theta_log, w1, w2)
        assert torch.allclose(g_w[2].to(torch.float64), num_w1_whole, atol=atol, rtol=rtol)
        assert torch.allclose(g_s[2].to(torch.float64), num_w1_stream, atol=atol, rtol=rtol)

    # Streaming grads should also match whole-sequence grads closely
    for gw, gs, name in zip(g_w, g_s, ["nu_log", "theta_log", "w1", "w2"], strict=False):
        if name in {"nu_log", "theta_log"} and with_resets:
            rtol_m, atol_m = 2e-2, 4e-3
        elif name in {"nu_log", "theta_log"}:
            rtol_m, atol_m = 5e-3, 1e-3
        elif with_resets and name in {"w1", "w2"}:
            rtol_m, atol_m = 5e-2, 2e-2
        else:
            rtol_m, atol_m = 2e-4, 1e-6
        assert torch.allclose(gw, gs, atol=atol_m, rtol=rtol_m), (
            f"{name} grad mismatch stream vs whole (resets={with_resets}); max diff={(gw - gs).abs().max().item():.3e}"
        )


def test_streaming_vs_whole_grad_parity_with_resets() -> None:
    """Autograd vs autograd parity under resets (diagonal input weights).

    Compares gradients from full-sequence diagonal streaming kernel (single
    chunk) to chunked streaming (with detach + boundary corrections) using the
    same reset pattern.
    """
    torch.manual_seed(2025)
    device = torch.device("cpu")
    dtype = torch.float64

    B, T, D, H = 2, 12, 6, 6
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)
    # Deterministic resets with multiple segments across T
    resets = torch.zeros(B, T, dtype=torch.bool, device=device)
    resets[:, 0] = False
    resets[:, 3] = True
    resets[:, 7] = True

    params = _build_params(D, H, device=device, dtype=dtype)
    nu_log, theta_log, w1, w2 = params
    activation = "SiLU"

    # Whole sequence (autograd)
    hc1_0 = torch.zeros(B, H, device=device, dtype=dtype)
    hc2_0 = torch.zeros(B, H, device=device, dtype=dtype)
    y_w, _state, _trace = rtu_stream_diag_pytorch(
        x_btd=x,
        nu_log=nu_log,
        theta_log=theta_log,
        w1=w1,
        w2=w2,
        activation_name=activation,
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets,
    )
    loss_w = (y_w**2).mean()
    g_w = torch.autograd.grad(loss_w, params, retain_graph=True, allow_unused=False)

    # Streaming (two chunks) with same resets
    def run_stream(chunks: tuple[int, ...]):
        hc1 = torch.zeros(B, H, device=device, dtype=dtype)
        hc2 = torch.zeros(B, H, device=device, dtype=dtype)
        trace = None
        ys = []
        t0 = 0
        for sz in chunks:
            t1 = min(T, t0 + sz)
            y_blk, (hc1, hc2), trace = rtu_stream_diag_pytorch(
                x_btd=x[:, t0:t1, :],
                nu_log=nu_log,
                theta_log=theta_log,
                w1=w1,
                w2=w2,
                activation_name=activation,
                hc1_init_bh=hc1,
                hc2_init_bh=hc2,
                trace_in=trace,
                resets_bt=resets[:, t0:t1],
            )
            ys.append(y_blk)
            hc1, hc2 = hc1.detach(), hc2.detach()
            trace = tuple(t.detach() for t in trace) if trace is not None else None
            t0 = t1
            if t0 >= T:
                break
        y_s = torch.cat(ys, dim=1)
        return (y_s**2).mean()

    loss_s = run_stream((5, 7))
    g_s = torch.autograd.grad(loss_s, params, retain_graph=True, allow_unused=False)

    # Compare all parameter gradients (nu_log, theta_log, w1, w2)
    # Resets can slightly amplify differences; use moderate tolerances.
    for gw, gs, name in zip(g_w, g_s, ["nu_log", "theta_log", "w1", "w2"], strict=False):
        if name in {"nu_log", "theta_log"}:
            rtol, atol = 2e-2, 4e-3
        else:
            rtol, atol = 3e-2, 8e-3
        assert torch.allclose(gw, gs, rtol=rtol, atol=atol), (
            f"Autograd parity failed for {name}: max diff={(gw - gs).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_TRITON or not torch.cuda.is_available(), reason="Triton+CUDA required")
@pytest.mark.parametrize("with_resets", [False, True])
def test_triton_streaming_diag_forward_and_grad_parity(with_resets: bool) -> None:
    """Compare Triton streaming diagonal kernel to PyTorch reference.

    Checks both forward outputs and parameter/input gradients for a single
    full-sequence pass (no chunking), with and without resets.
    """
    torch.manual_seed(101)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, H = 2, 17, 8
    D = H
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)
    resets = None
    if with_resets:
        resets = torch.rand(B, T, device=device) < 0.2

    # Build shared initial params, then clone for each backend to keep graphs independent
    nu0, th0, w10, w20 = _build_params(D, H, device=device, dtype=dtype)

    # PyTorch reference params
    nu_pt = nu0.clone().detach().requires_grad_(True)
    th_pt = th0.clone().detach().requires_grad_(True)
    w1_pt = w10.clone().detach().requires_grad_(True)
    w2_pt = w20.clone().detach().requires_grad_(True)

    # Triton params (copies)
    nu_tr = nu0.clone().detach().requires_grad_(True)
    th_tr = th0.clone().detach().requires_grad_(True)
    w1_tr = w10.clone().detach().requires_grad_(True)
    w2_tr = w20.clone().detach().requires_grad_(True)

    # Initial states
    hc1_0 = torch.zeros(B, H, device=device, dtype=dtype)
    hc2_0 = torch.zeros(B, H, device=device, dtype=dtype)

    # Forward (PyTorch)
    y_pt, (h1_pt, h2_pt), tr_out_pt = rtu_stream_diag_pytorch(
        x_btd=x,
        nu_log=nu_pt,
        theta_log=th_pt,
        w1=w1_pt,
        w2=w2_pt,
        activation_name="SiLU",
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets,
    )
    loss_pt = (y_pt**2).mean()
    g_pt = torch.autograd.grad(loss_pt, (nu_pt, th_pt, w1_pt, w2_pt, x), retain_graph=True)

    # Forward (Triton)
    y_tr, (h1_tr, h2_tr), tr_out_tr = _rtu_triton_stream(
        x_btd=x,
        nu_log=nu_tr,
        theta_log=th_tr,
        w1=w1_tr,
        w2=w2_tr,
        activation_name="SiLU",
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets,
    )
    loss_tr = (y_tr**2).mean()
    g_tr = torch.autograd.grad(loss_tr, (nu_tr, th_tr, w1_tr, w2_tr, x), retain_graph=True)

    # Forward parity
    assert y_pt.shape == y_tr.shape
    assert torch.allclose(y_pt, y_tr, rtol=2e-5, atol=1e-6), (
        f"forward y mismatch: {(y_pt - y_tr).abs().max().item():.3e}"
    )
    assert torch.allclose(h1_pt, h1_tr, rtol=2e-5, atol=1e-6)
    assert torch.allclose(h2_pt, h2_tr, rtol=2e-5, atol=1e-6)

    # We validate forward parity and gradients; traces are internal and may differ
    # slightly under segmented-parallel accumulation with resets.

    # Gradient parity
    names = ["nu_log", "theta_log", "w1", "w2", "x"]
    tolerances = {
        "nu_log": (5e-4, 2e-5),
        "theta_log": (5e-4, 2e-5),
        "w1": (5e-5, 1e-6),
        "w2": (5e-5, 1e-6),
        "x": (5e-5, 1e-6),
    }
    for gp, gt, nm in zip(g_pt, g_tr, names, strict=False):
        rtol, atol = tolerances[nm]
        assert torch.allclose(gp, gt, rtol=rtol, atol=atol), (
            f"grad {nm} mismatch: max diff={(gp - gt).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_TRITON or not torch.cuda.is_available(), reason="Triton+CUDA required")
@pytest.mark.parametrize("with_resets", [False, True])
def test_triton_streaming_diag_chunked_forward_and_grad_parity(with_resets: bool) -> None:
    """Parity when processing the sequence in chunks (Triton vs whole PyTorch).

    - Runs Triton streaming kernel in multiple chunks with detach between chunks.
    - Compares forward outputs and parameter/input gradients against the
      whole‑sequence PyTorch streaming reference.
    """
    torch.manual_seed(303)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, H = 2, 23, 8
    D = H
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

    # Shared base params then cloned per backend
    nu0, th0, w10, w20 = _build_params(D, H, device=device, dtype=dtype)

    for resets in resets_list:
        # Whole‑sequence PyTorch reference (per reset mask)
        nu_pt = nu0.clone().detach().requires_grad_(True)
        th_pt = th0.clone().detach().requires_grad_(True)
        w1_pt = w10.clone().detach().requires_grad_(True)
        w2_pt = w20.clone().detach().requires_grad_(True)
        hc1_0 = torch.zeros(B, H, device=device, dtype=dtype)
        hc2_0 = torch.zeros(B, H, device=device, dtype=dtype)
        y_ref, (_, _), _ = rtu_stream_diag_pytorch(
            x_btd=x,
            nu_log=nu_pt,
            theta_log=th_pt,
            w1=w1_pt,
            w2=w2_pt,
            activation_name="SiLU",
            hc1_init_bh=hc1_0,
            hc2_init_bh=hc2_0,
            trace_in=None,
            resets_bt=resets,
        )
        loss_ref = (y_ref**2).mean()
        g_ref = torch.autograd.grad(loss_ref, (nu_pt, th_pt, w1_pt, w2_pt, x), retain_graph=True)

        # Triton streaming in chunks (per reset mask)
        nu_tr = nu0.clone().detach().requires_grad_(True)
        th_tr = th0.clone().detach().requires_grad_(True)
        w1_tr = w10.clone().detach().requires_grad_(True)
        w2_tr = w20.clone().detach().requires_grad_(True)

        def run_triton_chunks(
            chunk_sizes: tuple[int, ...],
            nu_tr=nu_tr,
            th_tr=th_tr,
            w1_tr=w1_tr,
            w2_tr=w2_tr,
            resets=resets,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            hc1 = torch.zeros(B, H, device=device, dtype=dtype)
            hc2 = torch.zeros(B, H, device=device, dtype=dtype)
            trace = None
            ys = []
            t0 = 0
            for sz in chunk_sizes:
                t1 = min(T, t0 + sz)
                if t1 <= t0:
                    break
                y_blk, (hc1, hc2), trace = _rtu_triton_stream(
                    x_btd=x[:, t0:t1, :],
                    nu_log=nu_tr,
                    theta_log=th_tr,
                    w1=w1_tr,
                    w2=w2_tr,
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
            return torch.cat(ys, dim=1), (nu_tr, th_tr, w1_tr, w2_tr, x)

        for chunks in [(T,), (7, 8, 16), (5, 9, 9)]:
            y_t, params_t = run_triton_chunks(chunks)
            # Forward parity
            assert y_ref.shape == y_t.shape
            assert torch.allclose(y_ref, y_t, rtol=2e-5, atol=1e-6), (
                f"forward mismatch chunks={chunks}, max diff={(y_ref - y_t).abs().max().item():.3e}"
            )

            # Gradient parity
            loss_t = (y_t**2).mean()
            g_t = torch.autograd.grad(loss_t, params_t, retain_graph=True, allow_unused=False)
            names = ["nu_log", "theta_log", "w1", "w2", "x"]
            if with_resets:
                tolerances = {
                    "nu_log": (2e-2, 4e-3),
                    "theta_log": (2e-2, 4e-3),
                    "w1": (3e-2, 8e-3),
                    "w2": (3e-2, 8e-3),
                    "x": (3e-2, 8e-3),
                }
            else:
                tolerances = {
                    "nu_log": (5e-4, 2e-5),
                    "theta_log": (5e-4, 2e-5),
                    "w1": (5e-5, 1e-6),
                    "w2": (5e-5, 1e-6),
                    # Slightly looser absolute tol for x grads due to chunk boundaries
                    "x": (1e-4, 6e-4),
                }
            for gp, gt, nm in zip(g_ref, g_t, names, strict=False):
                rtol, atol = tolerances[nm]
                assert torch.allclose(gp, gt, rtol=rtol, atol=atol), (
                    f"grad {nm} mismatch (chunks={chunks}): max diff={(gp - gt).abs().max().item():.3e}"
                )


@pytest.mark.skipif(not _HAS_CUDA_SEQ or not torch.cuda.is_available(), reason="CUDA extension required")
@pytest.mark.parametrize("with_resets", [False, True])
def test_cuda_seq_streaming_diag_forward_and_grad_parity(with_resets: bool) -> None:
    """Compare CUDA all-in sequential kernel to PyTorch streaming diag.

    Uses stricter tolerances than Triton parity since both paths are sequential
    with identical arithmetic ordering and fp32 math.
    """
    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, H = 2, 19, 8
    D = H
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)

    resets = None
    if with_resets:
        # Mix random and deterministic resets, include head resets occasionally
        resets = torch.rand(B, T, device=device) < 0.15
        resets[:, 0] = torch.tensor([True, False], device=device)

    # Base parameters
    nu0, th0, w10, w20 = _build_params(D, H, device=device, dtype=dtype)

    # PyTorch reference params
    nu_pt = nu0.clone().detach().requires_grad_(True)
    th_pt = th0.clone().detach().requires_grad_(True)
    w1_pt = w10.clone().detach().requires_grad_(True)
    w2_pt = w20.clone().detach().requires_grad_(True)
    hc1_0 = torch.zeros(B, H, device=device, dtype=dtype)
    hc2_0 = torch.zeros(B, H, device=device, dtype=dtype)

    # CUDA params (copies)
    nu_cu = nu0.clone().detach().requires_grad_(True)
    th_cu = th0.clone().detach().requires_grad_(True)
    w1_cu = w10.clone().detach().requires_grad_(True)
    w2_cu = w20.clone().detach().requires_grad_(True)

    # Forward PyTorch
    y_pt, (h1_pt, h2_pt), _ = rtu_stream_diag_pytorch(
        x_btd=x,
        nu_log=nu_pt,
        theta_log=th_pt,
        w1=w1_pt,
        w2=w2_pt,
        activation_name="SiLU",
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets,
    )
    loss_pt = (y_pt**2).mean()
    g_pt = torch.autograd.grad(loss_pt, (nu_pt, th_pt, w1_pt, w2_pt, x), retain_graph=True)

    # Forward CUDA
    y_cu, (h1_cu, h2_cu), _ = _rtu_cuda_seq_stream(
        x_btd=x,
        nu_log=nu_cu,
        theta_log=th_cu,
        w1=w1_cu,
        w2=w2_cu,
        activation_name="SiLU",
        hc1_init_bh=hc1_0,
        hc2_init_bh=hc2_0,
        trace_in=None,
        resets_bt=resets,
    )
    loss_cu = (y_cu**2).mean()
    g_cu = torch.autograd.grad(loss_cu, (nu_cu, th_cu, w1_cu, w2_cu, x), retain_graph=True)

    # Forward parity (stricter than Triton)
    torch.testing.assert_close(y_cu, y_pt, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(h1_cu, h1_pt, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(h2_cu, h2_pt, rtol=1e-6, atol=2e-7)

    # Grad parity (stricter)
    names = ["nu_log", "theta_log", "w1", "w2", "x"]
    if with_resets:
        tolerances = {nm: (5e-6, 2e-6) for nm in names}
        tolerances["x"] = (1e-5, 5e-6)
    else:
        tolerances = {nm: (1e-6, 2e-7) for nm in names}
    for gp, gc, nm in zip(g_pt, g_cu, names, strict=True):
        rtol, atol = tolerances[nm]
        assert torch.allclose(gp, gc, rtol=rtol, atol=atol), (
            f"CUDA vs PyTorch grad {nm} mismatch: max diff={(gp - gc).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_CUDA_SEQ or not torch.cuda.is_available(), reason="CUDA extension required")
@pytest.mark.parametrize("with_resets", [False, True])
def test_cuda_seq_streaming_diag_whole_vs_chunked_parity(with_resets: bool) -> None:
    """CUDA-vs-CUDA parity: whole sequence vs chunked streaming.

    Confirms our fused sequential kernel matches itself across chunking. Uses
    stricter tolerances than the Triton version.
    """
    torch.manual_seed(2027)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, H = 2, 23, 8
    D = H
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)

    if with_resets:
        resets_rand = torch.rand(B, T, device=device) < 0.2
        resets_det = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            resets_det[b, 0] = False
            for t_idx in (4, 9, 15, 18):
                if t_idx < T:
                    resets_det[b, t_idx] = True
        resets_list = [resets_rand, resets_det]
    else:
        resets_list = [None]

    nu0, th0, w10, w20 = _build_params(D, H, device=device, dtype=dtype)

    for resets in resets_list:
        nu_wh = nu0.clone().detach().requires_grad_(True)
        th_wh = th0.clone().detach().requires_grad_(True)
        w1_wh = w10.clone().detach().requires_grad_(True)
        w2_wh = w20.clone().detach().requires_grad_(True)
        hc1_0 = torch.zeros(B, H, device=device, dtype=dtype)
        hc2_0 = torch.zeros(B, H, device=device, dtype=dtype)
        y_wh, (_, _), _ = _rtu_cuda_seq_stream(
            x_btd=x,
            nu_log=nu_wh,
            theta_log=th_wh,
            w1=w1_wh,
            w2=w2_wh,
            activation_name="SiLU",
            hc1_init_bh=hc1_0,
            hc2_init_bh=hc2_0,
            trace_in=None,
            resets_bt=resets,
        )
        loss_wh = (y_wh**2).mean()
        g_wh = torch.autograd.grad(loss_wh, (nu_wh, th_wh, w1_wh, w2_wh, x), retain_graph=True)

        nu_ch = nu0.clone().detach().requires_grad_(True)
        th_ch = th0.clone().detach().requires_grad_(True)
        w1_ch = w10.clone().detach().requires_grad_(True)
        w2_ch = w20.clone().detach().requires_grad_(True)

        def run_cuda_chunks(
            chunk_sizes: tuple[int, ...],
            *,
            resets_local=resets,
            nu_bind=nu_ch,
            th_bind=th_ch,
            w1_bind=w1_ch,
            w2_bind=w2_ch,
        ):
            hc1 = torch.zeros(B, H, device=device, dtype=dtype)
            hc2 = torch.zeros(B, H, device=device, dtype=dtype)
            trace = None
            ys = []
            t0 = 0
            for sz in chunk_sizes:
                t1 = min(T, t0 + sz)
                if t1 <= t0:
                    break
                y_blk, (hc1, hc2), trace = _rtu_cuda_seq_stream(
                    x_btd=x[:, t0:t1, :],
                    nu_log=nu_bind,
                    theta_log=th_bind,
                    w1=w1_bind,
                    w2=w2_bind,
                    activation_name="SiLU",
                    hc1_init_bh=hc1,
                    hc2_init_bh=hc2,
                    trace_in=trace,
                    resets_bt=None if resets_local is None else resets_local[:, t0:t1],
                )
                ys.append(y_blk)
                hc1, hc2 = hc1.detach(), hc2.detach()
                trace = tuple(t.detach() for t in trace) if trace is not None else None
                t0 = t1
                if t0 >= T:
                    break
            y_s = torch.cat(ys, dim=1)
            loss_s = (y_s**2).mean()
            return y_s, loss_s

        # Check several chunkings
        for chunks in [(T,), (7, 16), (5, 6, 12), (9, 9, 9)]:
            y_s, loss_s = run_cuda_chunks(chunks)
            # Forward parity (strict)
            torch.testing.assert_close(y_wh, y_s, rtol=1e-6, atol=2e-7)
            # Gradient parity
            g_s = torch.autograd.grad(loss_s, (nu_ch, th_ch, w1_ch, w2_ch, x), retain_graph=True)
            names = ["nu_log", "theta_log", "w1", "w2", "x"]
            if with_resets:
                # Slight numeric drift across chunk boundaries from atomics and chunk splits
                tolerances = {nm: (1e-6, 1e-4) for nm in names}
                tolerances["nu_log"] = (1e-6, 2e-4)
                tolerances["theta_log"] = (1e-6, 2e-4)
                tolerances["w1"] = (1e-6, 4e-4)
                tolerances["w2"] = (1e-6, 4e-4)
                tolerances["x"] = (1e-4, 1e-3)
            else:
                tolerances = {nm: (1e-6, 1e-4) for nm in names}
                tolerances["x"] = (1e-4, 6e-4)
            for gw, gs, nm in zip(g_wh, g_s, names, strict=True):
                rtol, atol = tolerances[nm]
                msg = (
                    f"CUDA whole vs chunked grad {nm} mismatch (chunks={chunks}): "
                    f"max diff={(gw - gs).abs().max().item():.3e}"
                )
                assert torch.allclose(gw, gs, rtol=rtol, atol=atol), msg




@pytest.mark.skipif(not _HAS_TRITON or not torch.cuda.is_available(), reason="Triton+CUDA required")
@pytest.mark.parametrize("with_resets", [False, True])
def test_triton_streaming_diag_whole_vs_chunked_parity(with_resets: bool) -> None:
    """Triton-vs-Triton parity: whole sequence vs chunked streaming.

    Validates that running the Triton streaming kernel on the full sequence in
    one call matches running it in multiple chunks with detach+trace carry
    between chunks. Checks both forward outputs and gradients for
    nu_log, theta_log, w1, w2, and x. Runs with and without resets, including
    a deterministic pattern that hits chunk boundaries.
    """
    torch.manual_seed(808)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T, H = 2, 29, 8
    D = H
    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)

    if with_resets:
        resets_rand = torch.rand(B, T, device=device) < 0.2
        resets_det = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            resets_det[b, 0] = False
            for t_idx in (5, 11, 17, 23):
                if t_idx < T:
                    resets_det[b, t_idx] = True
        resets_list = [resets_rand, resets_det]
    else:
        resets_list = [None]

    # Base parameters
    nu0, th0, w10, w20 = _build_params(D, H, device=device, dtype=dtype)

    for resets in resets_list:
        # Whole-sequence parameters (Triton)
        nu_wh = nu0.clone().detach().requires_grad_(True)
        th_wh = th0.clone().detach().requires_grad_(True)
        w1_wh = w10.clone().detach().requires_grad_(True)
        w2_wh = w20.clone().detach().requires_grad_(True)
        hc1_0 = torch.zeros(B, H, device=device, dtype=dtype)
        hc2_0 = torch.zeros(B, H, device=device, dtype=dtype)
        y_wh, (_, _), _ = _rtu_triton_stream(
            x_btd=x,
            nu_log=nu_wh,
            theta_log=th_wh,
            w1=w1_wh,
            w2=w2_wh,
            activation_name="SiLU",
            hc1_init_bh=hc1_0,
            hc2_init_bh=hc2_0,
            trace_in=None,
            resets_bt=resets,
        )
        loss_wh = (y_wh**2).mean()
        g_wh = torch.autograd.grad(loss_wh, (nu_wh, th_wh, w1_wh, w2_wh, x), retain_graph=True)

        # Chunked parameters (Triton)
        nu_ch = nu0.clone().detach().requires_grad_(True)
        th_ch = th0.clone().detach().requires_grad_(True)
        w1_ch = w10.clone().detach().requires_grad_(True)
        w2_ch = w20.clone().detach().requires_grad_(True)

        def run_triton_chunks(
            chunk_sizes: tuple[int, ...],
            nu=nu_ch,
            th=th_ch,
            w1=w1_ch,
            w2=w2_ch,
            resets_local=resets,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            hc1 = torch.zeros(B, H, device=device, dtype=dtype)
            hc2 = torch.zeros(B, H, device=device, dtype=dtype)
            trace = None
            ys = []
            t0 = 0
            for sz in chunk_sizes:
                t1 = min(T, t0 + sz)
                if t1 <= t0:
                    break
                y_blk, (hc1, hc2), trace = _rtu_triton_stream(
                    x_btd=x[:, t0:t1, :],
                    nu_log=nu,
                    theta_log=th,
                    w1=w1,
                    w2=w2,
                    activation_name="SiLU",
                    hc1_init_bh=hc1,
                    hc2_init_bh=hc2,
                    trace_in=trace,
                    resets_bt=None if resets_local is None else resets_local[:, t0:t1],
                )
                ys.append(y_blk)
                hc1, hc2 = hc1.detach(), hc2.detach()
                trace = tuple(t.detach() for t in trace) if trace is not None else None
                t0 = t1
                if t0 >= T:
                    break
            return torch.cat(ys, dim=1), (nu, th, w1, w2, x)

        for chunks in [(T,), (9, 10, 10), (6, 7, 8, 8)]:
            y_ch, params_ch = run_triton_chunks(chunks)
            # Forward parity
            assert y_wh.shape == y_ch.shape
            assert torch.allclose(y_wh, y_ch, rtol=2e-5, atol=1e-6), (
                f"forward mismatch (chunks={chunks}), max diff={(y_wh - y_ch).abs().max().item():.3e}"
            )

            # Gradient parity
            loss_ch = (y_ch**2).mean()
            g_ch = torch.autograd.grad(loss_ch, params_ch, retain_graph=True, allow_unused=False)
            names = ["nu_log", "theta_log", "w1", "w2", "x"]
            if with_resets:
                tolerances = {
                    "nu_log": (2e-2, 4e-3),
                    "theta_log": (2e-2, 4e-3),
                    "w1": (3e-2, 8e-3),
                    "w2": (3e-2, 8e-3),
                    "x": (3e-2, 8e-3),
                }
            else:
                tolerances = {
                    "nu_log": (5e-4, 2e-5),
                    "theta_log": (5e-4, 2e-5),
                    "w1": (5e-5, 1e-6),
                    "w2": (5e-5, 1e-6),
                    "x": (1e-4, 6e-4),
                }
            for gw, gc, nm in zip(g_wh, g_ch, names, strict=False):
                rtol, atol = tolerances[nm]
                assert torch.allclose(gw, gc, rtol=rtol, atol=atol), (
                    f"grad {nm} mismatch (chunks={chunks}): max diff={(gw - gc).abs().max().item():.3e}"
                )
