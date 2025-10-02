from __future__ import annotations

from typing import Tuple

import torch


def _slstm_pointwise(
    Wx: torch.Tensor,
    Ry: torch.Tensor,
    b_flat: torch.Tensor,
    states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla sLSTM pointwise update (pure PyTorch).

    Args:
        Wx: [B, 4*H] feed-forward preactivations
        Ry: [B, 4*H] recurrent preactivations
        b_flat: [4*H] bias
        states: [4, B, H] stacked (y, c, n, m)

    Returns:
        new_states: [4, B, H] stacked (y, c, n, m)
        gates_dbg: [4, B, H] stacked (i, f, zraw, o)
    """
    raw = Wx + Ry + b_flat
    y, c, n, m = torch.unbind(states, dim=0)

    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)

    logfplusm = m + torch.nn.functional.logsigmoid(fraw)
    if torch.all(n == 0.0):
        mnew = iraw
    else:
        mnew = torch.maximum(iraw, logfplusm)

    ogate = torch.sigmoid(oraw)
    igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))
    fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))
    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew

    return (
        torch.stack((ynew, cnew, nnew, mnew), dim=0),
        torch.stack((igate, fgate, zraw, ogate), dim=0),
    )


def _recurrent_mix(y: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Compute Ry from previous y with per-head recurrent kernel.

    Args:
        y: [B, NH, DH] or [B, NH*DH]
        R: [4, NH, DH, DH] recurrent weights per gate

    Returns:
        Ry: [B, 4*NH*DH]
    """
    B = y.shape[0]
    if y.dim() == 2:
        # Reshape [B, NH*DH] -> [B, NH, DH]
        NH = R.shape[1]
        DH = R.shape[2]
        y = y.view(B, NH, DH)

    NH, DH = y.shape[1], y.shape[2]
    # y: [B, NH, DH]
    # R: [4, NH, DH, DH]
    # We want: [B, NH, DH] @ [NH, DH, 4*DH] -> [B, NH, 4*DH]
    # Reshape R: [4, NH, DH, DH] -> [NH, DH, 4*DH]
    R_reshaped = R.permute(1, 2, 0, 3).reshape(NH, DH, 4 * DH)

    # Batched matmul: [B, NH, DH] @ [NH, DH, 4*DH] -> [B, NH, 4*DH]
    y_expanded = y.unsqueeze(2)  # [B, NH, 1, DH]
    R_expanded = R_reshaped.unsqueeze(0)  # [1, NH, DH, 4*DH]
    Ry = torch.matmul(y_expanded, R_expanded).squeeze(2)  # [B, NH, 4*DH]

    return Ry.reshape(B, NH * 4 * DH)


def slstm_sequence_pytorch(
    *,
    Wx: torch.Tensor,  # (B, T, 4, NH, DH) feed-forward preactivations (i, f, z, o)
    R: torch.Tensor,  # (4, NH, DH, DH) recurrent weights per gate
    b: torch.Tensor,  # (4, NH, DH) bias per gate
    initial_states: torch.Tensor,  # (4, B, NH, DH) states (h, c, n, m)
    resets: torch.Tensor | None = None,  # (B, T) reset mask
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run sLSTM sequence using pure PyTorch (ground truth implementation).

    This is the reference implementation that Triton kernels should match exactly.

    Args:
        Wx: (B, T, 4, NH, DH) feed-forward preactivations in order (i, f, z, o)
        R: (4, NH, DH, DH) recurrent weights per gate in order (i, f, z, o)
        b: (4, NH, DH) bias per gate in order (i, f, z, o)
        initial_states: (4, B, NH, DH) states (h, c, n, m)
        resets: (B, T) reset mask, optional. If provided, states are zeroed for
                batch elements where resets[:, t] is True at each timestep t.

    Returns:
        all_states: (T, 4, B, NH, DH) states at each timestep
        last_state: (4, B, NH, DH) final states
    """
    B, T, _, NH, DH = Wx.shape
    assert Wx.shape[2] == 4, f"Wx must have 4 gates, got {Wx.shape[2]}"
    assert R.shape == (4, NH, DH, DH), f"R must be (4,NH,DH,DH), got {R.shape}"
    assert b.shape == (4, NH, DH), f"b must be (4,NH,DH), got {b.shape}"
    assert initial_states.shape == (4, B, NH, DH), f"initial_states must be (4,B,NH,DH), got {initial_states.shape}"
    if resets is not None:
        assert resets.shape == (B, T), f"resets must be (B,T), got {resets.shape}"

    # Flatten bias: [4, NH, DH] -> [4*NH*DH]
    b_flat = b.reshape(4 * NH * DH)

    # Extract initial states
    y_t = initial_states[0]  # [B, NH, DH]
    c_t = initial_states[1]
    n_t = initial_states[2]
    m_t = initial_states[3]

    all_states_list = []

    for t in range(T):
        # Apply per-timestep resets before processing this timestep
        if resets is not None:
            # resets[:, t] is shape [B], need to broadcast to [B, NH, DH]
            reset_mask = resets[:, t].view(B, 1, 1).to(dtype=y_t.dtype)  # [B, 1, 1]
            # Zero out states where reset is True (mask value of 1 means reset)
            y_t = y_t * (1.0 - reset_mask)
            c_t = c_t * (1.0 - reset_mask)
            n_t = n_t * (1.0 - reset_mask)
            m_t = m_t * (1.0 - reset_mask)

        # Get preactivations for this timestep: [B, 4, NH, DH]
        Wx_t = Wx[:, t]  # [B, 4, NH, DH]

        # Flatten to [B, 4*NH*DH]
        Wx_t_flat = Wx_t.reshape(B, 4 * NH * DH)

        # Compute recurrent mixing
        Ry_t = _recurrent_mix(y_t, R)  # [B, 4*NH*DH]

        # Stack states for pointwise update: [4, B, NH*DH]
        states_flat = torch.stack(
            (
                y_t.reshape(B, NH * DH),
                c_t.reshape(B, NH * DH),
                n_t.reshape(B, NH * DH),
                m_t.reshape(B, NH * DH),
            ),
            dim=0,
        )

        # Pointwise update
        new_states_flat, _ = _slstm_pointwise(Wx_t_flat, Ry_t, b_flat, states_flat)

        # Unpack new states
        y_t = new_states_flat[0].reshape(B, NH, DH)
        c_t = new_states_flat[1].reshape(B, NH, DH)
        n_t = new_states_flat[2].reshape(B, NH, DH)
        m_t = new_states_flat[3].reshape(B, NH, DH)

        # Store states: [4, B, NH, DH]
        states_t = torch.stack((y_t, c_t, n_t, m_t), dim=0)
        all_states_list.append(states_t)

    # Stack all states: [T, 4, B, NH, DH]
    all_states = torch.stack(all_states_list, dim=0)

    # Last state: [4, B, NH, DH]
    last_state = all_states[-1]

    return all_states, last_state


__all__ = ["slstm_sequence_pytorch"]
