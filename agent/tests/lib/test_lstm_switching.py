import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.lstm_switching import LSTMswitching


@pytest.fixture
def simple_lstm_switching_environment():
    """Create a minimal environment for testing the LSTMswitching layer."""
    # Define the dimensions
    batch_size = 4
    seq_length = 3
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # Create a mock source component that LSTM expects
    class MockSourceComponent:
        def __init__(self, output_shape, name):
            self._out_tensor_shape = output_shape
            self.name = name

    # Create mock source components with proper output shapes
    mock_source = MockSourceComponent([input_size], "hidden")
    source_components = {"hidden": mock_source}

    # Create input data
    sample_input = {
        "env_obs": torch.rand(batch_size * seq_length, input_size),
        "hidden": torch.rand(batch_size * seq_length, input_size),
    }

    obs_shape = [input_size]
    cfg = {
        "name": "_lstm_switching_test_",
        "_nn_params": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        },
        "sources": [{"name": "hidden"}],
        "obs_shape": obs_shape,
    }
    # Create LSTMswitching layer
    lstm_layer = LSTMswitching(**cfg)

    # Initialize the network with proper source components
    lstm_layer.setup(source_components=source_components)

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


class TestLSTMSwitchingLayer:
    """Tests for the LSTMswitching layer focusing only on switching-specific features."""

    def test_lstm_switching_has_scalar_buffer(self, simple_lstm_switching_environment):
        """Test that LSTMswitching has the hidden_state_scalar buffer with correct value."""
        lstm_layer = simple_lstm_switching_environment["lstm_layer"]

        # Check that the buffer exists and has the correct value
        assert hasattr(lstm_layer, "hidden_state_scalar")
        assert lstm_layer.hidden_state_scalar.item() == 100.0
        assert lstm_layer.hidden_state_scalar.dtype == torch.float32

    def test_lstm_switching_scalar_multiplication(self, simple_lstm_switching_environment):
        """Test that the scalar multiplication is applied to the input."""
        lstm_layer = simple_lstm_switching_environment["lstm_layer"]
        sample_input = simple_lstm_switching_environment["sample_input"]
        params = simple_lstm_switching_environment["params"]

        # Create tensor dict
        td = TensorDict(sample_input, batch_size=[params["batch_size"] * params["seq_length"]])
        td["bptt"] = torch.tensor([params["seq_length"]] * (params["batch_size"] * params["seq_length"]))
        td["batch"] = torch.tensor([params["batch_size"]] * (params["batch_size"] * params["seq_length"]))

        # Forward pass with default scalar (100.0)
        lstm_layer.hidden_state_scalar.fill_(100.0)
        result_td_with_scale = lstm_layer._forward(td.clone())

        # Forward pass with scalar = 1.0
        lstm_layer.hidden_state_scalar.fill_(1.0)
        result_td_no_scale = lstm_layer._forward(td.clone())

        # Results should be different when scalar changes
        diff = torch.abs(result_td_with_scale[lstm_layer._name] - result_td_no_scale[lstm_layer._name]).mean().item()
        assert diff > 1e-6, "Scalar multiplication should affect output"

    def test_lstm_switching_weight_initialization(self, simple_lstm_switching_environment):
        """Test that square weight matrices are initialized with zero diagonal."""
        lstm_layer = simple_lstm_switching_environment["lstm_layer"]

        # Check that square weight matrices have zero diagonal
        for name, param in lstm_layer._net.named_parameters():
            if "weight" in name and param.ndim >= 2 and param.shape[0] == param.shape[1]:
                diagonal = torch.diag(param)
                assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6), (
                    f"Weight matrix {name} should have zero diagonal"
                )
