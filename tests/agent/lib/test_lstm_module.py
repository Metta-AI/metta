import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_modules import LSTMModule


@pytest.fixture
def lstm_environment():
    """Create a minimal environment for testing the LSTM module."""
    # Define the dimensions
    batch_size = 4
    seq_length = 3
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # Create input data
    x = torch.rand(batch_size * seq_length, input_size)
    hidden = torch.rand(batch_size * seq_length, input_size)

    # Create LSTM module
    lstm = LSTMModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    # Return all components needed for testing
    return {
        "lstm": lstm,
        "x": x,
        "hidden": hidden,
        "params": {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
    }


class TestLSTMModule:
    """Tests for the backwards-compatible LSTM module."""

    def test_initialization(self, lstm_environment):
        """Test LSTM module initialization."""
        lstm = lstm_environment["lstm"]
        params = lstm_environment["params"]

        # Check LSTM parameters
        assert lstm.input_size == params["input_size"]
        assert lstm.hidden_size == params["hidden_size"]
        assert lstm.num_layers == params["num_layers"]

        # Check key names
        assert lstm.in_keys == ["x", "hidden"]
        assert lstm.out_keys == ["_lstm_", "state"]

    def test_initial_state(self, lstm_environment):
        """Test initial state creation."""
        lstm = lstm_environment["lstm"]
        params = lstm_environment["params"]
        B = params["batch_size"] * params["seq_length"]
        state = lstm.get_initial_state(B, torch.device("cpu"))
        assert state.shape == (B, 2 * params["num_layers"], params["hidden_size"])
        assert torch.allclose(state, torch.zeros_like(state))

    def test_forward_with_none_state(self, lstm_environment):
        """Test forward pass with None state."""
        lstm = lstm_environment["lstm"]
        x = lstm_environment["x"]
        hidden = lstm_environment["hidden"]
        params = lstm_environment["params"]
        B = params["batch_size"] * params["seq_length"]
        td = TensorDict(
            {
                "x": x,
                "hidden": hidden,
            },
            batch_size=B,
        )
        result = lstm(td)
        assert result["_lstm_"].shape == (B, params["hidden_size"])
        assert "state" in result
        assert result["state"].shape == (B, 2 * params["num_layers"], params["hidden_size"])

    def test_forward_with_zero_state(self, lstm_environment):
        """Test forward pass with zero state."""
        lstm = lstm_environment["lstm"]
        x = lstm_environment["x"]
        hidden = lstm_environment["hidden"]
        params = lstm_environment["params"]
        B = params["batch_size"] * params["seq_length"]
        state = lstm.get_initial_state(B, torch.device("cpu"))
        td = TensorDict(
            {
                "x": x,
                "hidden": hidden,
                "state": state,
            },
            batch_size=B,
        )
        result = lstm(td)
        assert result["_lstm_"].shape == (B, params["hidden_size"])
        assert "state" in result
        assert result["state"].shape == (B, 2 * params["num_layers"], params["hidden_size"])
        assert not torch.allclose(result["state"], state)

    def test_state_continuity(self, lstm_environment):
        """Test state continuity across multiple forward passes."""
        lstm = lstm_environment["lstm"]
        x = lstm_environment["x"]
        hidden = lstm_environment["hidden"]
        params = lstm_environment["params"]

        # First pass with None state
        td1 = TensorDict(
            {
                "x": x,
                "hidden": hidden,
            },
            batch_size=params["batch_size"] * params["seq_length"],
        )
        result1 = lstm(td1)
        state1 = result1["state"]
        output1 = result1["_lstm_"]

        # Second pass with state from first pass
        td2 = TensorDict(
            {
                "x": x,
                "hidden": hidden,
                "state": state1,
            },
            batch_size=params["batch_size"] * params["seq_length"],
        )
        result2 = lstm(td2)
        output2 = result2["_lstm_"]

        # Outputs should be different when using state continuation
        assert not torch.allclose(output1, output2)

    def test_partial_state_reset(self, lstm_environment):
        """Test partial state reset behavior."""
        lstm = lstm_environment["lstm"]
        x = lstm_environment["x"]
        hidden = lstm_environment["hidden"]
        params = lstm_environment["params"]

        # First pass to establish non-zero state
        td1 = TensorDict(
            {
                "x": x,
                "hidden": hidden,
            },
            batch_size=params["batch_size"] * params["seq_length"],
        )
        result1 = lstm(td1)
        state1 = result1["state"].clone()

        # Create a state with some values reset to zero
        reset_state = state1.clone()
        split_point = params["num_layers"]

        # Zero out state for half the batch
        half_batch = params["batch_size"] // 2
        reset_state[:split_point, :half_batch, :] = 0.0  # Reset h state
        reset_state[split_point:, :half_batch, :] = 0.0  # Reset c state

        # Second pass with partially reset state
        td2 = TensorDict(
            {
                "x": x,
                "hidden": hidden,
                "state": reset_state,
            },
            batch_size=params["batch_size"] * params["seq_length"],
        )
        result2 = lstm(td2)

        # Run another pass with original state for comparison
        td3 = TensorDict(
            {
                "x": x,
                "hidden": hidden,
                "state": state1,
            },
            batch_size=params["batch_size"] * params["seq_length"],
        )
        result3 = lstm(td3)

        # Compare outputs
        assert not torch.allclose(result2["_lstm_"], result3["_lstm_"])

    def test_shape_validation(self, lstm_environment):
        """Test shape validation in forward pass."""
        lstm = lstm_environment["lstm"]
        params = lstm_environment["params"]

        # Create tensordict with wrong hidden shape
        td = TensorDict(
            {
                "x": torch.rand(params["batch_size"] * params["seq_length"], params["input_size"]),
                "hidden": torch.rand(
                    params["batch_size"] * params["seq_length"], params["input_size"] + 1
                ),  # Wrong size
            },
            batch_size=params["batch_size"] * params["seq_length"],
        )

        # Forward pass should raise assertion error
        with pytest.raises(AssertionError):
            lstm(td)
