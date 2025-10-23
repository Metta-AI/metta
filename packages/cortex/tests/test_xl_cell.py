"""Tests for Transformer-XL style attention cell (XLCell)."""

from __future__ import annotations

import torch
import pytest
from cortex.cells.xl import XLCell
from cortex.config import AxonConfig, XLCellConfig
from tensordict import TensorDict

# Skip this module entirely (slow)
pytestmark = pytest.mark.skip(reason="slow test skipped in prod")

def get_test_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def test_xl_sequence_shapes_and_state() -> None:
    torch.manual_seed(0)

    device = get_test_device()
    dtype = torch.float32

    B, T, H, NH = 2, 8, 64, 8
    cfg = XLCellConfig(
        hidden_size=H,
        n_heads=NH,
        head_dim=None,
        mem_len=16,
        attn_dropout=0.0,
        out_dropout=0.0,
        use_bias=True,
    )
    cell = XLCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    y, state = cell(x, state=None)

    assert y.shape == (B, T, H)
    assert state is not None and isinstance(state, TensorDict)
    assert "mem" in state and "mem_seg" in state
    assert state["mem"].shape == (B, min(T, cfg.mem_len), H)
    assert state["mem_seg"].shape == (B, min(T, cfg.mem_len))


def test_xl_step_vs_sequence_equivalence() -> None:
    torch.manual_seed(123)

    device = get_test_device()
    dtype = torch.float32

    B, T, H, NH = 2, 10, 64, 8
    cfg = XLCellConfig(
        hidden_size=H,
        n_heads=NH,
        head_dim=None,
        mem_len=64,
        attn_dropout=0.0,
        out_dropout=0.0,
        use_bias=True,
    )
    cell = XLCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel sequence
    with torch.no_grad():
        y_seq, state_seq = cell(x, state=None)

    # Step-by-step
    y_steps = []
    state = None
    with torch.no_grad():
        for t in range(T):
            y_t, state = cell(x[:, t, :], state)
            y_steps.append(y_t)
    y_step = torch.stack(y_steps, dim=1)

    assert y_seq.shape == y_step.shape
    torch.testing.assert_close(
        y_seq,
        y_step,
        rtol=5e-4,
        atol=5e-4,
        msg="XLCell step vs sequence outputs differ beyond tolerance",
    )
    assert state is not None and state_seq is not None
    assert state["mem"].shape == state_seq["mem"].shape
    assert state["mem_seg"].shape == state_seq["mem_seg"].shape


def test_xl_memory_trim_across_calls() -> None:
    torch.manual_seed(7)

    device = get_test_device()
    dtype = torch.float32

    B, T, H, NH = 2, 6, 32, 4
    mem_len = 10
    cfg = XLCellConfig(
        hidden_size=H,
        n_heads=NH,
        head_dim=None,
        mem_len=mem_len,
        attn_dropout=0.0,
        out_dropout=0.0,
        use_bias=True,
    )
    cell = XLCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x1 = torch.randn(B, T, H, device=device, dtype=dtype)
    x2 = torch.randn(B, T, H, device=device, dtype=dtype)

    with torch.no_grad():
        _, st1 = cell(x1, state=None)
        assert st1["mem"].shape[1] == min(T, mem_len)
        _, st2 = cell(x2, state=st1)
        assert st2["mem"].shape[1] == mem_len  # trimmed to mem_len after two calls


def test_xl_with_axon_qkv_state_and_reset() -> None:
    torch.manual_seed(21)

    device = get_test_device()
    dtype = torch.float32

    B, T, H, NH = 2, 5, 64, 8
    cfg = XLCellConfig(
        hidden_size=H,
        n_heads=NH,
        head_dim=None,
        mem_len=16,
        attn_dropout=0.0,
        out_dropout=0.0,
        use_bias=True,
        use_axon_qkv=True,
        axon_qkv_config=AxonConfig(hidden_size=H, out_dim=NH * (H // NH), activation="identity"),
    )
    cell = XLCell(cfg).to(device=device, dtype=dtype)
    cell.train(False)

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    y, st = cell(x, state=None)
    assert y.shape == (B, T, H)
    assert st is not None and isinstance(st, TensorDict)
    # AxonLayer substates should be present under 'xl_qkv'
    assert "xl_qkv" in st
    group = st.get("xl_qkv")
    assert group is not None
    for name in ("q", "k", "v"):
        assert name in group.keys(), f"missing substate for {name}"
        sub = group.get(name)
        assert "hc1" in sub.keys() and "hc2" in sub.keys()
        assert sub["hc1"].shape[0] == B

    # Reset first batch element and verify Axon substates are zeroed there
    mask = torch.zeros(B, dtype=torch.float32, device=device)
    mask[0] = 1.0
    st_after = cell.reset_state(st, mask)
    grp_after = st_after.get("xl_qkv")
    for name in ("q", "k", "v"):
        sub = grp_after.get(name)
        assert torch.allclose(sub["hc1"][0], torch.zeros_like(sub["hc1"][0]))
        assert torch.allclose(sub["hc2"][0], torch.zeros_like(sub["hc2"][0]))
