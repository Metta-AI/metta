import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.lstm import LSTM, MettaLSTM
from metta.agent.lib.metta_module import MettaData

# from metta.agent.lib.lstm import LSTM, MettaLSTM
# from metta.agent.policy_state import PolicyState


# @pytest.fixture
def simple_lstm_environment():
    pass  # commented out


# class TestLSTMLayer:
#     ...  # commented out


@pytest.fixture
def lstm_pair():
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
        output_key = "_lstm_test_"

        # Create identical input tensors
        x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        state = None

        td_orig = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
        td_orig["state"] = state
        result_orig = lstm._forward(td_orig)

        td_metta = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
        md = MettaData(td_metta, {})
        result_metta = metta_lstm(md)

        # Compare outputs
        torch.testing.assert_close(
            result_orig[output_key], result_metta[output_key], msg="Output tensors should be identical"
        )
        torch.testing.assert_close(
            result_orig["state"], result_metta.data[output_key]["state"], msg="State tensors should be identical"
        )

    def test_identical_outputs_with_state(self, lstm_pair):
        """Test that both LSTMs produce identical outputs with initial state."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]
        output_key = "_lstm_test_"

        x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        h_0 = torch.randn(params["num_layers"], params["batch_size"] * params["seq_length"], params["hidden_size"])
        c_0 = torch.randn(params["num_layers"], params["batch_size"] * params["seq_length"], params["hidden_size"])
        state = torch.cat([h_0, c_0], dim=0)

        td_orig = TensorDict({"x": x, "hidden": hidden, "state": state}, batch_size=[])
        result_orig = lstm._forward(td_orig)

        td_metta = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
        md = MettaData(td_metta, {})
        md.data[output_key] = {"state": state}
        result_metta = metta_lstm(md)

        torch.testing.assert_close(
            result_orig[output_key], result_metta[output_key], msg="Output tensors should be identical"
        )
        torch.testing.assert_close(
            result_orig["state"], result_metta.data[output_key]["state"], msg="State tensors should be identical"
        )

    def test_identical_outputs_sequence(self, lstm_pair):
        """Test that both LSTMs produce identical outputs over a sequence of steps."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]
        output_key = "_lstm_test_"

        state_orig = None
        state_metta = None

        for _ in range(3):
            x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
            hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])

            td_orig = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            td_orig["state"] = state_orig
            result_orig = lstm._forward(td_orig)
            state_orig = result_orig["state"]

            td_metta = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            md = MettaData(td_metta, {})
            if state_metta is not None:
                md.data[output_key] = {"state": state_metta}
            result_metta = metta_lstm(md)
            state_metta = result_metta.data[output_key]["state"]

            torch.testing.assert_close(
                result_orig[output_key], result_metta[output_key], msg="Output tensors should be identical"
            )
            torch.testing.assert_close(
                result_orig["state"], result_metta.data[output_key]["state"], msg="State tensors should be identical"
            )

    def test_identical_outputs_batch_variations(self, lstm_pair):
        """Test that both LSTMs handle different batch sizes identically."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]
        output_key = "_lstm_test_"

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size * params["seq_length"], params["input_size"])
            hidden = torch.randn(batch_size * params["seq_length"], params["input_size"])
            state = None

            td_orig = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            td_orig["state"] = state
            result_orig = lstm._forward(td_orig)

            td_metta = TensorDict({"x": x, "hidden": hidden}, batch_size=[])
            md = MettaData(td_metta, {})
            result_metta = metta_lstm(md)

            torch.testing.assert_close(
                result_orig[output_key],
                result_metta[output_key],
                msg=f"Output tensors should be identical for batch size {batch_size}",
            )
            torch.testing.assert_close(
                result_orig["state"],
                result_metta.data[output_key]["state"],
                msg=f"State tensors should be identical for batch size {batch_size}",
            )

    def test_metadata_propagation(self, lstm_pair):
        """Test that unrelated metadata in md.data is preserved after MettaLSTM forward."""
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]
        output_key = "_lstm_test_"

        x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {"foo": "bar"})
        result = metta_lstm(md)
        assert result.data["foo"] == "bar"
        assert output_key in result.data
        assert "state" in result.data[output_key]


# class TestLSTM:
#     ...  # commented out


class TestMettaLSTMRobustness:
    def setup_metta_lstm(self, device="cpu", dtype=torch.float32, output_key="lstm1"):
        obs_shape = [10]
        hidden_size = 20
        num_layers = 2
        metta_lstm = MettaLSTM(
            obs_shape=obs_shape,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_key="x",
            hidden_key="hidden",
            state_key="state",
            output_key=output_key,
        )
        metta_lstm.to(device=device, dtype=dtype)
        return metta_lstm, obs_shape, hidden_size, num_layers

    def test_error_handling(self):
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        # Missing input key
        md = MettaData(TensorDict({}, batch_size=[]), {})
        with pytest.raises(KeyError):
            metta_lstm(md)
        # Bad input shape
        x = torch.randn(5, 5)  # Should be [batch, 10]
        hidden = torch.randn(5, 10)
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        with pytest.raises(ValueError):
            metta_lstm(md)
        # Malformed state
        x = torch.randn(5, obs_shape[0])
        hidden = torch.randn(5, obs_shape[0])
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        md.data["lstm1"] = {"state": "not_a_tensor"}
        with pytest.raises(Exception):
            metta_lstm(md)

    def test_multiple_lstm_namespacing(self):
        # Two LSTMs with different output_keys
        metta_lstm1, obs_shape, _, _ = self.setup_metta_lstm(output_key="lstm1")
        metta_lstm2, _, _, _ = self.setup_metta_lstm(output_key="lstm2")
        x = torch.randn(5, obs_shape[0])
        hidden = torch.randn(5, obs_shape[0])
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        metta_lstm1(md)
        metta_lstm2(md)
        assert "lstm1" in md.data
        assert "lstm2" in md.data
        assert "state" in md.data["lstm1"]
        assert "state" in md.data["lstm2"]

    def test_device_dtype(self):
        # CPU float32
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm(device="cpu", dtype=torch.float32)
        x = torch.randn(5, obs_shape[0], dtype=torch.float32)
        hidden = torch.randn(5, obs_shape[0], dtype=torch.float32)
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        result = metta_lstm(md)
        assert result[metta_lstm.output_key].dtype == torch.float32
        # CPU float64
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm(device="cpu", dtype=torch.float64)
        x = torch.randn(5, obs_shape[0], dtype=torch.float64)
        hidden = torch.randn(5, obs_shape[0], dtype=torch.float64)
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        result = metta_lstm(md)
        assert result[metta_lstm.output_key].dtype == torch.float64
        # CUDA (if available)
        if torch.cuda.is_available():
            metta_lstm, obs_shape, _, _ = self.setup_metta_lstm(device="cuda")
            x = torch.randn(5, obs_shape[0], device="cuda")
            hidden = torch.randn(5, obs_shape[0], device="cuda")
            md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
            result = metta_lstm(md)
            assert result[metta_lstm.output_key].device.type == "cuda"

    def test_gradients_autograd(self):
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        x = torch.randn(5, obs_shape[0], requires_grad=True)
        hidden = torch.randn(5, obs_shape[0], requires_grad=True)
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        result = metta_lstm(md)
        output = result[metta_lstm.output_key]
        loss = output.sum()
        loss.backward()
        # Only hidden.grad is checked because x is not used in the computation graph
        assert hidden.grad is not None

    def test_serialization(self, tmp_path):
        import torch

        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        x = torch.randn(5, obs_shape[0])
        hidden = torch.randn(5, obs_shape[0])
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        # Save
        torch.save(metta_lstm.state_dict(), tmp_path / "lstm.pt")
        # Load
        metta_lstm2, _, _, _ = self.setup_metta_lstm()
        metta_lstm2.load_state_dict(torch.load(tmp_path / "lstm.pt"))
        # Check output parity
        result1 = metta_lstm(md)
        result2 = metta_lstm2(md)
        torch.testing.assert_close(result1[metta_lstm.output_key], result2[metta_lstm2.output_key])

    def test_functional_stateless(self):
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        x = torch.randn(5, obs_shape[0])
        hidden = torch.randn(5, obs_shape[0])
        md = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        # Stateless: pass in state, get state out, no side effects
        state_in = None
        md.data[metta_lstm.output_key] = {"state": state_in}
        result = metta_lstm(md)
        state_out = result.data[metta_lstm.output_key]["state"]
        # Run again with state_out as input
        md2 = MettaData(TensorDict({"x": x, "hidden": hidden}, batch_size=[]), {})
        md2.data[metta_lstm.output_key] = {"state": state_out}
        result2 = metta_lstm(md2)
        # State should be different after a step
        assert not torch.equal(state_out, result2.data[metta_lstm.output_key]["state"])
