"""Tests for XLCell attention memory cell."""

from __future__ import annotations

import torch

from cortex.cells.xl import XLCell
from cortex.config import XLCellConfig


def test_xl_cell_sequence_shapes() -> None:
    """XLCell produces expected output shape and memory update."""
    torch.manual_seed(0)

    cfg = XLCellConfig(hidden_size=32, n_heads=4, mem_len=6)
    cell = XLCell(cfg)
    cell.eval()

    batch_size = 2
    seq_len = 5
    hidden = 32

    x = torch.randn(batch_size, seq_len, hidden)
    state = cell.init_state(batch=batch_size, device=x.device, dtype=x.dtype)

    resets = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    y, new_state = cell(x, state, resets=resets)

    assert y.shape == (batch_size, seq_len, hidden)
    assert new_state is not None

    mem = new_state.get("mem")
    mem_seg = new_state.get("mem_seg")
    assert mem is not None
    assert mem_seg is not None
    assert mem.shape == (batch_size, seq_len, hidden)
    assert mem_seg.shape == (batch_size, seq_len)
    assert mem.dtype == x.dtype
    assert mem_seg.dtype == torch.long


def test_xl_cell_reset_state_clears_selected_batches() -> None:
    """reset_state zeroes memory for masked batches while keeping others."""
    torch.manual_seed(1)

    cfg = XLCellConfig(hidden_size=16, n_heads=4, mem_len=8)
    cell = XLCell(cfg)
    cell.eval()

    batch_size = 3
    seq_len = 4
    hidden = 16

    x1 = torch.randn(batch_size, seq_len, hidden)
    state = cell.init_state(batch=batch_size, device=x1.device, dtype=x1.dtype)
    zeros_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    _, state = cell(x1, state, resets=zeros_mask)

    x2 = torch.randn(batch_size, seq_len, hidden)
    _, state = cell(x2, state, resets=zeros_mask)
    assert state is not None

    mem_before = state.get("mem")
    mem_seg_before = state.get("mem_seg")
    assert mem_before is not None
    assert mem_seg_before is not None
    assert torch.any(mem_before.abs() > 0)

    reset_mask = torch.tensor([True, False, True])
    state_reset = cell.reset_state(state, reset_mask)
    assert state_reset is not None

    mem_after = state_reset.get("mem")
    mem_seg_after = state_reset.get("mem_seg")
    assert mem_after is not None
    assert mem_seg_after is not None
    assert torch.allclose(mem_after[0], torch.zeros_like(mem_after[0]))
    assert torch.allclose(mem_after[2], torch.zeros_like(mem_after[2]))
    assert torch.allclose(mem_after[1], mem_before[1])
    assert torch.all(mem_seg_after[0] == 0)
    assert torch.all(mem_seg_after[2] == 0)
    assert torch.all(mem_seg_after[1] == mem_seg_before[1])


def test_xl_cell_chunked_matches_full_sequence() -> None:
    """Chunked attention path matches full-sequence computation."""
    torch.manual_seed(2)

    hidden = 48
    batch_size = 2
    seq_len = 15

    cfg_full = XLCellConfig(hidden_size=hidden, n_heads=6, mem_len=10, chunk_size=None)
    cfg_chunked = XLCellConfig(hidden_size=hidden, n_heads=6, mem_len=10, chunk_size=4)

    cell_full = XLCell(cfg_full)
    cell_chunked = XLCell(cfg_chunked)
    cell_chunked.load_state_dict(cell_full.state_dict())
    cell_full.eval()
    cell_chunked.eval()

    x = torch.randn(batch_size, seq_len, hidden)
    zeros_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    state_full = cell_full.init_state(batch=batch_size, device=x.device, dtype=x.dtype)
    state_chunked = cell_chunked.init_state(
        batch=batch_size, device=x.device, dtype=x.dtype
    )

    y_full, _ = cell_full(x, state_full, resets=zeros_mask)
    y_chunked, _ = cell_chunked(x, state_chunked, resets=zeros_mask)

    torch.testing.assert_close(y_full, y_chunked, rtol=1e-5, atol=1e-5)
