import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.lstm import LSTM


@pytest.fixture
def simple_lstm_environment():
    """Create a minimal environment for testing the LSTM layer."""
    # Define the dimensions
    batch_size = 4
    seq_length = 3
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # Create input data
    sample_input = {
        "x": torch.rand(batch_size * seq_length, input_size),
        "hidden": torch.rand(batch_size * seq_length, input_size),
    }

    obs_shape = [input_size]
    cfg = {
        "name": "_lstm_test_",
        "_nn_params": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        },
        "sources": [{"name": "hidden"}],
        "obs_shape": obs_shape,
    }
    # Create LSTM layer
    lstm_layer = LSTM(**cfg)

    # Set up in_tensor_shapes manually
    lstm_layer._in_tensor_shapes = [[input_size]]

    # Initialize the network
    lstm_layer.setup(source_components=None)

    # Return all components needed for testing
    return {
        "lstm_layer": lstm_layer,
        "sample_input": sample_input,
        "params": {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
    }


class TestLSTMLayer:
    """Tests for the LSTM layer with focus on state handling behavior."""

    def test_lstm_initial_forward(self, simple_lstm_environment):
        """Test LSTM layer with an initial forward pass (no prior state)."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        td = TensorDict(sample_input, batch_size=[params["batch_size"] * params["seq_length"]])
        td.bptt = params["seq_length"]
        td.batch = params["batch_size"]

        # The LSTM state is initially empty
        assert not lstm_layer.lstm_h
        assert not lstm_layer.lstm_c

        result_td = lstm_layer._forward(td)

        # Verify output shape
        expected_shape = (params["batch_size"] * params["seq_length"], params["hidden_size"])
        assert result_td[lstm_layer._name].shape == expected_shape

        # Verify state is created and stored in the layer for the default env_id 0
        assert 0 in lstm_layer.lstm_h
        assert 0 in lstm_layer.lstm_c
        assert lstm_layer.lstm_h[0].shape == (params["num_layers"], params["batch_size"], params["hidden_size"])
        assert lstm_layer.lstm_c[0].shape == (params["num_layers"], params["batch_size"], params["hidden_size"])

    def test_lstm_state_continuity(self, simple_lstm_environment):
        """Test that the LSTM state is carried over between forward passes."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        td = TensorDict(sample_input, batch_size=[params["batch_size"] * params["seq_length"]])
        td.bptt = params["seq_length"]
        td.batch = params["batch_size"]

        # First forward pass will use initial zero state
        result1_td = lstm_layer._forward(td.clone())
        output1 = result1_td[lstm_layer._name]

        # Second forward pass will use the state stored from the first pass
        result2_td = lstm_layer._forward(td.clone())
        output2 = result2_td[lstm_layer._name]

        # Outputs should be different when using state continuation
        diff = torch.abs(output1 - output2).mean().item()
        assert diff > 1e-6, "Continued state should produce different outputs"

    def test_lstm_reset_memory(self, simple_lstm_environment):
        """Test the `reset_memory` method."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        td = TensorDict(sample_input, batch_size=[params["batch_size"] * params["seq_length"]])
        td.bptt = params["seq_length"]
        td.batch = params["batch_size"]

        # Run a forward pass to populate the state
        lstm_layer._forward(td.clone())
        assert 0 in lstm_layer.lstm_h

        # Reset the memory
        lstm_layer.reset_memory()
        assert not lstm_layer.lstm_h
        assert not lstm_layer.lstm_c

        # Running forward pass again should produce same output as the very first pass
        # First, get the output with fresh state again
        output1 = lstm_layer._forward(td.clone())[lstm_layer._name]

        # Then, run it one more time to see if state continues
        output2 = lstm_layer._forward(td.clone())[lstm_layer._name]

        # Now reset and check if we get back to output1
        lstm_layer.reset_memory()
        output3 = lstm_layer._forward(td.clone())[lstm_layer._name]

        assert torch.allclose(output1, output3, atol=1e-6)
        # and output3 should be different from output2
        diff = torch.abs(output2 - output3).mean().item()
        assert diff > 1e-6, "Resetting state should change the output of a subsequent pass"
