# ruff: noqa: I001
# mypy: disable-error-code=import-untyped
"""Torch compile tests for cells and auto stack."""

from __future__ import annotations

import os
from typing import Tuple

import pytest
import torch


from tensordict import TensorDict  # noqa: E402  # type: ignore

os.environ.setdefault("CORTEX_DISABLE_TRITON", "1")

from cortex.cells import CausalConv1d, LSTMCell, mLSTMCell, sLSTMCell, XLCell  # noqa: E402
from cortex.cells.core import AxonCell, AxonLayer  # noqa: E402
from cortex.stacks.auto import build_cortex_auto_stack  # noqa: E402
from cortex.config import (  # noqa: E402
    AxonConfig,
    CausalConv1dConfig,
    LSTMCellConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
    XLCellConfig,
)


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    assert a.shape == b.shape
    if a.numel() == 0:
        return
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def _assert_tensordict_close(td_a: TensorDict, td_b: TensorDict) -> None:
    # Compare intersection of keys; shapes should match exactly
    keys = set(td_a.keys()) | set(td_b.keys())
    for k in keys:
        a = td_a.get(k)
        b = td_b.get(k)
        if a is None or b is None:
            # Some aux keys may be absent; skip strict enforcement here
            continue
        _assert_close(a, b)


def _compile_and_compare(
    module: torch.nn.Module,
    x: torch.Tensor,
    state: TensorDict | None,
    resets: torch.Tensor | None,
) -> Tuple[torch.Tensor, TensorDict]:
    y_e, s_e = module(x, state, resets=resets)
    compiled = torch.compile(module)
    y_c, s_c = compiled(x, state, resets=resets)
    _assert_close(y_e, y_c)
    assert isinstance(s_e, TensorDict) and isinstance(s_c, TensorDict)
    _assert_tensordict_close(s_e, s_c)
    return y_c, s_c


@pytest.mark.parametrize("hidden_size", [32])
def test_causal_conv1d_compile(hidden_size: int) -> None:
    cfg = CausalConv1dConfig(hidden_size=hidden_size, kernel_size=4, causal_conv_bias=True, channel_mixing=False)
    cell = CausalConv1d(cfg)

    B, T, H = 2, 4, hidden_size
    # Step
    x_step = torch.randn(B, H)
    _compile_and_compare(cell, x_step, state=None, resets=torch.zeros(B, dtype=torch.long))
    # Sequence
    x_seq = torch.randn(B, T, H)
    resets_bt = torch.zeros(B, T, dtype=torch.long)
    resets_bt[:, 0] = 1
    _compile_and_compare(cell, x_seq, state=None, resets=resets_bt)


@pytest.mark.parametrize("hidden_size", [32])
def test_lstm_cell_compile(hidden_size: int) -> None:
    cfg = LSTMCellConfig(hidden_size=hidden_size, num_layers=1, bias=True, proj_size=0)
    cell = LSTMCell(cfg)
    B, T, H = 2, 3, hidden_size
    x_step = torch.randn(B, H)
    _compile_and_compare(cell, x_step, state=None, resets=torch.zeros(B, dtype=torch.long))
    x_seq = torch.randn(B, T, H)
    resets_bt = torch.zeros(B, T, dtype=torch.long)
    _compile_and_compare(cell, x_seq, state=None, resets=resets_bt)


@pytest.mark.parametrize("hidden_size", [32])
def test_axon_cell_compile(hidden_size: int) -> None:
    cfg = AxonConfig(hidden_size=hidden_size)
    cell = AxonCell(cfg)
    B, T, H = 2, 3, hidden_size
    x_step = torch.randn(B, H)
    _compile_and_compare(cell, x_step, state=None, resets=torch.zeros(B, dtype=torch.long))
    x_seq = torch.randn(B, T, H)
    resets_bt = torch.zeros(B, T, dtype=torch.long)
    _compile_and_compare(cell, x_seq, state=None, resets=resets_bt)


@pytest.mark.parametrize("hidden_size,num_heads", [(32, 4)])
def test_mlstm_cell_compile(hidden_size: int, num_heads: int) -> None:
    cfg = mLSTMCellConfig(hidden_size=hidden_size, num_heads=num_heads, chunk_size=4, conv1d_kernel_size=4)
    cell = mLSTMCell(cfg)
    B, T, H = 2, 5, hidden_size
    x_step = torch.randn(B, H)
    _compile_and_compare(cell, x_step, state=None, resets=torch.zeros(B, dtype=torch.long))
    x_seq = torch.randn(B, T, H)
    resets_bt = torch.zeros(B, T, dtype=torch.long)
    resets_bt[:, 0] = 1
    _compile_and_compare(cell, x_seq, state=None, resets=resets_bt)


@pytest.mark.parametrize("hidden_size,num_heads", [(32, 4)])
def test_slstm_cell_compile(hidden_size: int, num_heads: int) -> None:
    cfg = sLSTMCellConfig(hidden_size=hidden_size, num_heads=num_heads, conv1d_kernel_size=4, dropout=0.0)
    cell = sLSTMCell(cfg)
    B, T, H = 2, 6, hidden_size
    x_step = torch.randn(B, H)
    _compile_and_compare(cell, x_step, state=None, resets=torch.zeros(B, dtype=torch.long))
    x_seq = torch.randn(B, T, H)
    resets_bt = torch.zeros(B, T, dtype=torch.long)
    _compile_and_compare(cell, x_seq, state=None, resets=resets_bt)


@pytest.mark.parametrize("hidden_size,n_heads,mem_len", [(32, 4, 8)])
def test_xl_cell_compile(hidden_size: int, n_heads: int, mem_len: int) -> None:
    cfg = XLCellConfig(hidden_size=hidden_size, n_heads=n_heads, head_dim=None, mem_len=mem_len)
    cell = XLCell(cfg)
    B, T, H = 2, 4, hidden_size
    # Step
    x_step = torch.randn(B, H)
    _compile_and_compare(cell, x_step, state=None, resets=torch.zeros(B, dtype=torch.long))
    # Sequence
    x_seq = torch.randn(B, T, H)
    resets_bt = torch.zeros(B, T, dtype=torch.long)
    resets_bt[:, 0] = 1
    _compile_and_compare(cell, x_seq, state=None, resets=resets_bt)


def test_axon_layer_compile_linear_like() -> None:
    # Also verify AxonLayer wrapper compiles and manages its nested state
    B, T, H_in, H_out = 2, 5, 32, 48
    layer = AxonLayer(H_in, H_out)
    # Parent state lives outside and is mutated in-place
    parent: TensorDict | None = TensorDict({}, batch_size=[B])
    # Sequence path
    x_seq = torch.randn(B, T, H_in)
    # Eager
    y_e = layer(x_seq, state=parent)
    # Compiled
    compiled = torch.compile(layer)
    parent_c = TensorDict({}, batch_size=[B])
    y_c = compiled(x_seq, state=parent_c)
    _assert_close(y_e, y_c)
    # Step path
    x_step = torch.randn(B, H_in)
    parent_s = TensorDict({}, batch_size=[B])
    y_e2 = layer(x_step, state=parent_s)
    compiled2 = torch.compile(layer)
    parent_s_c = TensorDict({}, batch_size=[B])
    y_c2 = compiled2(x_step, state=parent_s_c)
    _assert_close(y_e2, y_c2)


# ---- Cortex auto stack: per-block compile ----


@pytest.mark.parametrize("pattern", ["AX", "MS"])  # small, fast patterns
def test_autostack_compile_cpu(pattern: str) -> None:
    d_hidden = 32
    B, T, H = 2, 3, d_hidden

    eager = build_cortex_auto_stack(
        d_hidden=d_hidden,
        num_layers=len(pattern),
        block_pattern=pattern,
        compile_blocks=False,
        post_norm=True,
    )
    compiled = build_cortex_auto_stack(
        d_hidden=d_hidden,
        num_layers=len(pattern),
        block_pattern=pattern,
        compile_blocks=True,
        post_norm=True,
    )
    compiled.load_state_dict(eager.state_dict())

    # Sequence path
    x_seq = torch.randn(B, T, H)
    st_e = eager.init_state(B, device=x_seq.device, dtype=x_seq.dtype)
    st_c = compiled.init_state(B, device=x_seq.device, dtype=x_seq.dtype)
    y_e, _ = eager(x_seq, st_e, resets=torch.zeros(B, T, dtype=torch.long))
    y_c, _ = compiled(x_seq, st_c, resets=torch.zeros(B, T, dtype=torch.long))
    _assert_close(y_e, y_c)

    # Step path
    x_step = torch.randn(B, H)
    st_e = eager.init_state(B, device=x_step.device, dtype=x_step.dtype)
    st_c = compiled.init_state(B, device=x_step.device, dtype=x_step.dtype)
    y_e, _ = eager(x_step, st_e, resets=torch.zeros(B, dtype=torch.long))
    y_c, _ = compiled(x_step, st_c, resets=torch.zeros(B, dtype=torch.long))
    _assert_close(y_e, y_c)


@pytest.mark.cuda
def test_autostack_compile_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in this environment")

    device = torch.device("cuda")
    d_hidden = 32
    B, T, H = 2, 3, d_hidden
    # Use a pattern that avoids custom CUDA extensions (pure PyTorch kernels)
    pattern = "XS"

    eager = build_cortex_auto_stack(
        d_hidden=d_hidden,
        num_layers=len(pattern),
        block_pattern=pattern,
        compile_blocks=False,
        post_norm=True,
    ).to(device)
    compiled = build_cortex_auto_stack(
        d_hidden=d_hidden,
        num_layers=len(pattern),
        block_pattern=pattern,
        compile_blocks=True,
        post_norm=True,
    ).to(device)

    compiled.load_state_dict(eager.state_dict())

    # Sequence
    x_seq = torch.randn(B, T, H, device=device)
    st_e = eager.init_state(B, device=device, dtype=x_seq.dtype)
    st_c = compiled.init_state(B, device=device, dtype=x_seq.dtype)
    y_e, _ = eager(x_seq, st_e, resets=torch.zeros(B, T, dtype=torch.long, device=device))
    y_c, _ = compiled(x_seq, st_c, resets=torch.zeros(B, T, dtype=torch.long, device=device))
    _assert_close(y_e, y_c)

    # Step
    x_step = torch.randn(B, H, device=device)
    st_e = eager.init_state(B, device=device, dtype=x_step.dtype)
    st_c = compiled.init_state(B, device=device, dtype=x_step.dtype)
    y_e, _ = eager(x_step, st_e, resets=torch.zeros(B, dtype=torch.long, device=device))
    y_c, _ = compiled(x_step, st_c, resets=torch.zeros(B, dtype=torch.long, device=device))
    _assert_close(y_e, y_c)
