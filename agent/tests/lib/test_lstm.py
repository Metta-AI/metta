import pytest
import torch
from tensordict import TensorDict

from metta.agent.components.lstm import LSTM, LSTMConfig


@pytest.fixture
def lstm_environment():
    batch_size = 4
    time_steps = 3
    latent_size = 10
    hidden_size = 16
    num_layers = 2

    config = LSTMConfig(
        in_key="latent",
        out_key="core",
        latent_size=latent_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    lstm = LSTM(config)

    def build_td():
        latent = torch.randn(batch_size * time_steps, latent_size)
        td = TensorDict(
            {
                "latent": latent,
                "bptt": torch.full((batch_size * time_steps,), time_steps, dtype=torch.long),
                "batch": torch.full((batch_size * time_steps,), batch_size, dtype=torch.long),
            },
            batch_size=[batch_size * time_steps],
        )
        return td

    return lstm, build_td, batch_size, time_steps, hidden_size, num_layers


def test_lstm_forward_creates_output_and_state(lstm_environment):
    lstm, build_td, batch_size, time_steps, hidden_size, num_layers = lstm_environment

    td = build_td()
    output = lstm(td)

    assert "core" in output
    assert output["core"].shape == (batch_size * time_steps, hidden_size)
    assert 0 in lstm.lstm_h
    assert lstm.lstm_h[0].shape == (num_layers, batch_size, hidden_size)


def test_lstm_state_persists_across_calls(lstm_environment):
    lstm, build_td, *_ = lstm_environment

    td = build_td()
    out1 = lstm(td.clone())["core"]
    out2 = lstm(td.clone())["core"]

    # With state continuation, repeated pass produces different output
    assert not torch.allclose(out1, out2)


def test_lstm_reset_memory_clears_state(lstm_environment):
    lstm, build_td, *_ = lstm_environment

    lstm(build_td())
    assert lstm.lstm_h

    lstm.reset_memory()
    assert not lstm.lstm_h
    assert not lstm.lstm_c

    # After reset, initial pass behaves like a fresh LSTM
    td = build_td()
    out1 = lstm(td.clone())["core"]
    out2 = lstm(td.clone())["core"]
    assert not torch.allclose(out1, out2)
