import pytest
import torch

from metta.agent.lib.lstm import LSTM
from metta.agent.lstm_hidden_state import LstmHiddenState


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
    hidden_size = hidden_size
    cfg = {
        "name": "_lstm_test_",
        "_nn_params": {"num_layers": num_layers},
        "sources": [{"name": "hidden"}],
    }
    # Create LSTM layer
    lstm_layer = LSTM(obs_shape, hidden_size, **cfg)

    # Set up in_tensor_shapes manually
    lstm_layer._in_tensor_shapes = [[input_size]]

    # Initialize the network
    if not hasattr(lstm_layer, "_out_tensor_shape"):
        lstm_layer._out_tensor_shape = [hidden_size]

    # Set up the network manually
    lstm_layer._initialize()

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

    def test_lstm_with_none_state(self, simple_lstm_environment):
        """Test LSTM layer behavior with None state."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        # Create dict with None state
        td = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": None}

        # Forward pass with None state
        result = lstm_layer._forward(td)

        # Verify output shape
        assert result[lstm_layer._name].shape == (params["batch_size"] * params["seq_length"], params["hidden_size"])

        # Verify state is created
        assert result["state"] is not None
        assert result["state"].shape[0] == 2 * params["num_layers"]

    def test_lstm_with_zero_state(self, simple_lstm_environment):
        """Test LSTM layer behavior with zero-initialized state."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        # The key issue: we need to reshape the state to match what the LSTM expects
        # PyTorch LSTM expects hidden/cell states with shape: (num_layers, batch_size, hidden_size)
        # But our input x is (batch_size * seq_length, input_size)

        # First, determine B (what the LSTM extracts as batch size from the input)
        x_shape = sample_input["x"].shape
        B = x_shape[0]  # batch_size * seq_length

        # Create zero state with the proper dimensions
        h_zeros = torch.zeros(params["num_layers"], B, params["hidden_size"])
        c_zeros = torch.zeros(params["num_layers"], B, params["hidden_size"])
        zero_state = torch.cat([h_zeros, c_zeros], dim=0)

        # Create dict with zero state
        td = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": zero_state}

        # Forward pass with zero state
        result = lstm_layer._forward(td)

        # Verify output shape
        assert result[lstm_layer._name].shape == (params["batch_size"] * params["seq_length"], params["hidden_size"])

        # Verify state is updated
        assert result["state"] is not None
        assert result["state"].shape[0] == 2 * params["num_layers"]

    def test_training_impact_simulation(self, simple_lstm_environment):
        """Simulate how the PR change affects outputs over multiple training steps."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        num_steps = 10  # Number of simulated training steps
        differences = []

        # Get the actual batch size from input x shape - this is critical
        B = sample_input["x"].shape[0]  # batch_size * seq_length

        # Initialize states with the correct shapes
        h_orig = torch.zeros(params["num_layers"], B, params["hidden_size"])
        c_orig = torch.zeros(params["num_layers"], B, params["hidden_size"])
        h_new = h_orig.clone()
        c_new = c_orig.clone()

        # Environment IDs that will be affected by the PR change
        # Note: We need to adjust how we access env_ids if B != params["batch_size"]
        # For simplicity, we'll use half of B
        env_ids = torch.arange(B // 2)

        # Simulate multiple training steps
        for step in range(num_steps):
            # Create new random input for each step
            x = torch.rand_like(sample_input["x"])
            hidden = torch.rand_like(sample_input["hidden"])

            # Original behavior
            state_orig = torch.cat([h_orig, c_orig], dim=0)
            td_orig = {"x": x, "hidden": hidden, "state": state_orig}

            result_orig = lstm_layer._forward(td_orig)

            # Update states for next step
            state_out = result_orig["state"]
            split_point = params["num_layers"]
            h_orig = state_out[:split_point].clone()
            c_orig = state_out[split_point:].clone()

            # New behavior (PR)
            # First, use the same state update
            h_new = h_orig.clone()
            c_new = c_orig.clone()

            # Then simulate the PR fix: reset states for some env IDs
            for i in env_ids:
                h_new[:, i, :] = 0
                c_new[:, i, :] = 0

            state_new = torch.cat([h_new, c_new], dim=0)
            td_new = {"x": x, "hidden": hidden, "state": state_new}

            result_new = lstm_layer._forward(td_new)

            # Calculate output difference for this step
            output_diff = torch.abs(result_orig[lstm_layer._name] - result_new[lstm_layer._name]).mean().item()

            differences.append(output_diff)

            print(f"Step {step + 1} output difference: {output_diff}")

        # Calculate increasing/decreasing trend
        if differences[-1] > differences[0]:
            print("Impact of PR change increases over time")
        else:
            print("Impact of PR change stabilizes or decreases over time")

        # The average difference over all steps should be significant
        avg_diff = sum(differences) / len(differences)
        assert avg_diff > 1e-6, "PR change should have measurable impact over training"

    def test_lstm_continual_state(self, simple_lstm_environment):
        """Test LSTM layer with state continuity across calls."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]

        # First pass with None state
        td1 = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": None}

        result1 = lstm_layer._forward(td1)
        state1 = result1["state"]
        output1 = result1[lstm_layer._name]

        # Second pass with state from first pass
        td2 = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": state1}

        result2 = lstm_layer._forward(td2)
        output2 = result2[lstm_layer._name]

        # Outputs should be different when using state continuation
        diff = torch.abs(output1 - output2).mean().item()
        print(f"Output difference with continued state: {diff}")

        # The difference should be significant when continuing state vs starting fresh
        assert diff > 1e-6, "Continued state should produce different outputs"

    def test_lstm_reset_behavior(self, simple_lstm_environment):
        """Test how resetting state affects LSTM output."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        # First pass to establish non-zero state
        td1 = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": None}

        result1 = lstm_layer._forward(td1)
        state1 = result1["state"].clone()

        # Create a state with some values reset to zero (simulating PR behavior)
        reset_state = state1.clone()
        split_point = params["num_layers"]

        # Zero out state for half the batch
        half_batch = params["batch_size"] // 2
        for i in range(half_batch):
            reset_state[:split_point, i, :] = 0.0  # Reset h state
            reset_state[split_point:, i, :] = 0.0  # Reset c state

        # Second pass with partially reset state
        td2 = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": reset_state}

        result2 = lstm_layer._forward(td2)

        # Run another pass with original state for comparison
        td3 = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": state1}

        result3 = lstm_layer._forward(td3)

        # Compare outputs
        output_diff = torch.abs(result2[lstm_layer._name] - result3[lstm_layer._name]).mean().item()
        print(f"Output difference due to partial state reset: {output_diff}")

        # The difference should be significant when some states are reset
        assert output_diff > 1e-6, "Partial state reset should affect outputs"

    def test_pr_change_simulation(self, simple_lstm_environment):
        """Simulate the specific change in your PR: replacing None states with zeros."""
        lstm_layer = simple_lstm_environment["lstm_layer"]
        sample_input = simple_lstm_environment["sample_input"]
        params = simple_lstm_environment["params"]

        # Build up some non-zero state first
        td_init = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": None}

        result_init = lstm_layer._forward(td_init)
        state = result_init["state"].clone()

        # Split into h and c states
        split_point = params["num_layers"]
        h_state = state[:split_point].clone()
        c_state = state[split_point:].clone()

        # Create policy states to simulate the different behaviors
        # Original behavior: Use the state directly (which would crash if None)
        lstm_state_orig = LstmHiddenState(lstm_h=h_state, lstm_c=c_state)

        # Simulate state being None for some environment IDs
        env_ids = torch.arange(params["batch_size"] // 2)

        # New behavior (PR fix): Replace None with zeros
        h_new = h_state.clone()
        c_new = c_state.clone()

        # Zero out for selected environment IDs
        for i in env_ids:
            h_new[:, i, :] = 0
            c_new[:, i, :] = 0

        lstm_state_new = LstmHiddenState(lstm_h=h_new, lstm_c=c_new)

        # Create states in the format expected by the LSTM layer
        assert lstm_state_orig.lstm_h is not None
        assert lstm_state_orig.lstm_c is not None
        assert lstm_state_new.lstm_h is not None
        assert lstm_state_new.lstm_c is not None
        state_orig = torch.cat([lstm_state_orig.lstm_h, lstm_state_orig.lstm_c], dim=0)
        state_new = torch.cat([lstm_state_new.lstm_h, lstm_state_new.lstm_c], dim=0)

        # Forward pass with original state
        td_orig = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": state_orig}

        result_orig = lstm_layer._forward(td_orig)

        # Forward pass with new state (PR behavior)
        td_new = {"x": sample_input["x"], "hidden": sample_input["hidden"], "state": state_new}

        result_new = lstm_layer._forward(td_new)

        # Compare outputs
        output_diff = torch.abs(result_orig[lstm_layer._name] - result_new[lstm_layer._name]).mean().item()

        print(f"Output difference due to PR change: {output_diff}")

        # Differences should be significant for the affected environments
        # but we're calculating average over all outputs
        assert output_diff > 1e-6, "PR change should affect LSTM outputs"
