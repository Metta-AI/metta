import pytest
import torch
from tensordict import TensorDict

# from metta.agent.lib.lstm import LSTM, MettaLSTM
# from metta.agent.policy_state import PolicyState


# @pytest.fixture
def simple_lstm_environment():
    pass  # commented out


# class TestLSTMLayer:
#     ...  # commented out


@pytest.fixture
def lstm_pair():
    from metta.agent.lib.lstm import LSTM, MettaLSTM

    batch_size = 4
    seq_length = 3
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create original LSTM
    obs_shape = [input_size]
    cfg = {
        "name": "_lstm_test_",
        "_nn_params": {"num_layers": num_layers},
        "sources": [{"name": "hidden"}],
    }
    lstm = LSTM(obs_shape, hidden_size, **cfg)
    lstm._in_tensor_shapes = [[input_size]]
    lstm._out_tensor_shape = [hidden_size]
    lstm._initialize()

    # Create MettaLSTM with same parameters
    metta_lstm = MettaLSTM(
        obs_shape=obs_shape,
        hidden_size=hidden_size,
        num_layers=num_layers,
        input_key="x",
        hidden_key="hidden",
        state_key="state",
        output_key="_lstm_test_",
    )

    # Copy parameters from original LSTM to MettaLSTM for parity
    if hasattr(lstm, "_net") and lstm._net is not None and hasattr(metta_lstm, "_net") and metta_lstm._net is not None:
        orig_params = dict(lstm._net.named_parameters())
        metta_params = dict(metta_lstm._net.named_parameters())
        for name in orig_params:
            if name in metta_params:
                metta_params[name].data.copy_(orig_params[name].data)

    # Print parameter summaries for both modules
    if hasattr(lstm, "_net") and lstm._net is not None:
        print("\nOriginal LSTM parameters:")
        for name, param in lstm._net.named_parameters():  # type: ignore
            print(
                f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, min={param.data.min():.6f}, max={param.data.max():.6f}"
            )
    else:
        print("\nOriginal LSTM _net is not initialized.")
    print("\nMettaLSTM parameters:")
    for name, param in metta_lstm._net.named_parameters():
        print(
            f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, min={param.data.min():.6f}, max={param.data.max():.6f}"
        )

    return {
        "lstm": lstm,
        "metta_lstm": metta_lstm,
        "params": {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
    }


class TestLSTMComparison:
    """Tests to verify that MettaLSTM produces identical outputs to LSTM."""

    def test_identical_outputs_no_state(self, lstm_pair):
        """Test that both LSTMs produce identical outputs with no initial state."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]

        # Create identical input tensors
        x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        state = None

        print("\nInput shapes:")
        print(f"x shape: {x.shape}")
        print(f"hidden shape: {hidden.shape}")

        # Only include non-None values in TensorDict
        td_orig = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
        td_orig["state"] = state
        result_orig = lstm._forward(td_orig)

        td_metta = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
        td_metta["state"] = state
        result_metta = metta_lstm(td_metta)

        print("\nOutput shapes:")
        print(f"Original output shape: {result_orig['_lstm_test_'].shape}")
        print(f"Metta output shape: {result_metta['_lstm_test_'].shape}")

        print("\nOutput samples (first 5 elements):")
        print(f"Original output: {result_orig['_lstm_test_'].flatten()[:5]}")
        print(f"Metta output: {result_metta['_lstm_test_'].flatten()[:5]}")

        print("\nState shapes:")
        print(f"Original state shape: {result_orig['state'].shape if result_orig['state'] is not None else None}")
        print(f"Metta state shape: {result_metta['state'].shape if result_metta['state'] is not None else None}")

        # Compare outputs
        torch.testing.assert_close(
            result_orig["_lstm_test_"], result_metta["_lstm_test_"], msg="Output tensors should be identical"
        )
        torch.testing.assert_close(result_orig["state"], result_metta["state"], msg="State tensors should be identical")

    def test_identical_outputs_with_state(self, lstm_pair):
        """Test that both LSTMs produce identical outputs with initial state."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]

        # Create identical input tensors
        x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])

        # Create initial state
        h_0 = torch.randn(params["num_layers"], params["batch_size"] * params["seq_length"], params["hidden_size"])
        c_0 = torch.randn(params["num_layers"], params["batch_size"] * params["seq_length"], params["hidden_size"])
        state = torch.cat([h_0, c_0], dim=0)

        td_orig = TensorDict({"x": x, "hidden": hidden, "state": state}, batch_size=[])
        result_orig = lstm._forward(td_orig)

        td_metta = TensorDict({"x": x, "hidden": hidden, "state": state}, batch_size=[])
        result_metta = metta_lstm(td_metta)

        # Compare outputs
        torch.testing.assert_close(
            result_orig["_lstm_test_"], result_metta["_lstm_test_"], msg="Output tensors should be identical"
        )
        torch.testing.assert_close(result_orig["state"], result_metta["state"], msg="State tensors should be identical")

    def test_identical_outputs_sequence(self, lstm_pair):
        """Test that both LSTMs produce identical outputs over a sequence of steps."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]

        # Create initial state
        state_orig = None
        state_metta = None

        # Run multiple steps
        for _ in range(3):
            # Create new input tensors for each step
            x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
            hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])

            td_orig = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            td_orig["state"] = state_orig
            result_orig = lstm._forward(td_orig)
            state_orig = result_orig["state"]

            td_metta = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            td_metta["state"] = state_metta
            result_metta = metta_lstm(td_metta)
            state_metta = result_metta["state"]

            # Compare outputs
            torch.testing.assert_close(
                result_orig["_lstm_test_"], result_metta["_lstm_test_"], msg="Output tensors should be identical"
            )
            torch.testing.assert_close(
                result_orig["state"], result_metta["state"], msg="State tensors should be identical"
            )

    def test_identical_outputs_batch_variations(self, lstm_pair):
        """Test that both LSTMs handle different batch sizes identically."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            # Create input tensors
            x = torch.randn(batch_size * params["seq_length"], params["input_size"])
            hidden = torch.randn(batch_size * params["seq_length"], params["input_size"])
            state = None

            td_orig = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            td_orig["state"] = state
            result_orig = lstm._forward(td_orig)

            td_metta = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            td_metta["state"] = state
            result_metta = metta_lstm(td_metta)

            # Compare outputs
            torch.testing.assert_close(
                result_orig["_lstm_test_"],
                result_metta["_lstm_test_"],
                msg=f"Output tensors should be identical for batch size {batch_size}",
            )
            torch.testing.assert_close(
                result_orig["state"],
                result_metta["state"],
                msg=f"State tensors should be identical for batch size {batch_size}",
            )


# class TestLSTM:
#     ...  # commented out
