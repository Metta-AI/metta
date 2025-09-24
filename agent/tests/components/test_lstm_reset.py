from __future__ import annotations

import torch
from tensordict import TensorDict

from metta.agent.components.lstm_reset import LSTMReset, LSTMResetConfig


def _build_td(batch_size: int, latent_size: int) -> TensorDict:
    latent = torch.ones(batch_size, latent_size)
    td = TensorDict(
        {
            "latent": latent,
            "bptt": torch.ones(batch_size, dtype=torch.long),
            "batch": torch.full((batch_size,), batch_size, dtype=torch.long),
        },
        batch_size=[batch_size],
    )
    return td


def test_lstm_reset_preserves_state_without_dones() -> None:
    config = LSTMResetConfig(in_key="latent", out_key="core", latent_size=8, hidden_size=4, num_layers=1)
    lstm_reset = LSTMReset(config)

    td = _build_td(batch_size=3, latent_size=8)
    lstm_reset(td.clone())

    first_hidden = lstm_reset.lstm_h.clone()
    assert torch.any(first_hidden != 0)

    lstm_reset(td.clone())

    second_hidden = lstm_reset.lstm_h.clone()
    assert torch.any(second_hidden != 0)
