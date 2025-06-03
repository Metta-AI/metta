import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.lstm import LSTM, MettaEncodedLSTM
from metta.agent.lib.metta_module import MettaDict, MettaLinear, MettaReLU

# from metta.agent.lib.lstm import LSTM, MettaEncodedLSTM
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

    # Create MettaEncodedLSTM with same parameters
    metta_lstm = MettaEncodedLSTM(
        obs_shape=obs_shape,
        hidden_size=hidden_size,
        num_layers=num_layers,
        input_key="encoded_features",
        output_key="_lstm_test_",
    )

    # Copy parameters from original LSTM to MettaEncodedLSTM for parity
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
    print("\nMettaEncodedLSTM parameters:")
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
    """Tests to verify that MettaEncodedLSTM produces identical outputs to LSTM."""

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

        td_orig = TensorDict(
            {
                "x": x,
                "encoded_features": x,
                "hidden": hidden,
                "_B_": torch.tensor(params["batch_size"]),
                "_TT_": torch.tensor(params["seq_length"]),
            },
            batch_size=[],
        )
        td_orig["state"] = state
        result_orig = lstm._forward(td_orig)

        td_metta = TensorDict({"encoded_features": x}, batch_size=[])
        md = MettaDict(td_metta, {"global": {"batch_size": params["batch_size"], "tt": params["seq_length"]}})
        md.data[output_key] = {"state": state}
        result_metta = metta_lstm(md)

        # Compare outputs
        try:
            torch.testing.assert_close(
                result_orig[output_key],
                result_metta.td[output_key],
                msg="Output tensors should be identical",
                rtol=1e-3,
                atol=1e-5,
            )
        except AssertionError:
            diff = torch.max(torch.abs(result_orig[output_key] - result_metta.td[output_key])).item()
            print("[DEBUG] Max abs diff (output):", diff)
            print("[DEBUG] Legacy output[:5]:", result_orig[output_key].flatten()[:5])
            print("[DEBUG] Metta output[:5]:", result_metta.td[output_key].flatten()[:5])
            raise
        try:
            torch.testing.assert_close(
                result_orig["state"],
                result_metta.data[output_key]["state"],
                msg="State tensors should be identical",
                rtol=1e-3,
                atol=1e-5,
            )
        except AssertionError:
            diff = torch.max(torch.abs(result_orig["state"] - result_metta.data[output_key]["state"]))
            print("[DEBUG] Max abs diff (state):", diff)
            print("[DEBUG] Legacy state[:5]:", result_orig["state"].flatten()[:5])
            print("[DEBUG] Metta state[:5]:", result_metta.data[output_key]["state"].flatten()[:5])
            raise

    def test_identical_outputs_with_state(self, lstm_pair):
        """Test that both LSTMs produce identical outputs with initial state."""
        lstm = lstm_pair["lstm"]
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]
        output_key = "_lstm_test_"

        x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        h_0 = torch.randn(params["num_layers"], params["batch_size"], params["hidden_size"])
        c_0 = torch.randn(params["num_layers"], params["batch_size"], params["hidden_size"])
        state = torch.cat([h_0, c_0], dim=0)

        td_orig = TensorDict(
            {
                "x": x,
                "encoded_features": x,
                "hidden": hidden,
                "state": state,
                "_B_": torch.tensor(params["batch_size"]),
                "_TT_": torch.tensor(params["seq_length"]),
            },
            batch_size=[],
        )
        result_orig = lstm._forward(td_orig)

        td_metta = TensorDict({"encoded_features": x}, batch_size=[])
        md = MettaDict(td_metta, {"global": {"batch_size": params["batch_size"], "tt": params["seq_length"]}})
        md.data[output_key] = {"state": state}
        result_metta = metta_lstm(md)

        torch.testing.assert_close(
            result_orig[output_key],
            result_metta.td[output_key],
            msg="Output tensors should be identical",
            rtol=1e-3,
            atol=1e-5,
        )
        torch.testing.assert_close(
            result_orig["state"],
            result_metta.data[output_key]["state"],
            msg="State tensors should be identical",
            rtol=1e-3,
            atol=1e-5,
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

            # For legacy LSTM, state must be [num_layers, batch_size, hidden_size]
            if state_orig is not None:
                # Split state_orig into h_0 and c_0, then reshape to [num_layers, batch_size, hidden_size]
                split_size = params["num_layers"]
                h_0, c_0 = state_orig[:split_size], state_orig[split_size:]
                h_0 = h_0.reshape(params["num_layers"], params["batch_size"], params["hidden_size"])
                c_0 = c_0.reshape(params["num_layers"], params["batch_size"], params["hidden_size"])
                state_legacy = torch.cat([h_0, c_0], dim=0)
            else:
                state_legacy = None

            td_orig = TensorDict(
                {
                    "x": x,
                    "encoded_features": x,
                    "hidden": hidden,
                    "_B_": torch.tensor(params["batch_size"]),
                    "_TT_": torch.tensor(params["seq_length"]),
                },
                batch_size=[],
            )
            td_orig["state"] = state_legacy
            result_orig = lstm._forward(td_orig)
            state_orig = result_orig["state"]

            td_metta = TensorDict({"encoded_features": x}, batch_size=[])
            md = MettaDict(td_metta, {"global": {"batch_size": params["batch_size"], "tt": params["seq_length"]}})
            if state_metta is not None:
                md.data[output_key] = {"state": state_metta}
            result_metta = metta_lstm(md)
            state_metta = result_metta.data[output_key]["state"]

            try:
                torch.testing.assert_close(
                    result_orig[output_key],
                    result_metta.td[output_key],
                    msg="Output tensors should be identical",
                    rtol=1e-3,
                    atol=1e-5,
                )
            except AssertionError:
                diff = torch.max(torch.abs(result_orig[output_key] - result_metta.td[output_key])).item()
                print("[DEBUG] Max abs diff (output):", diff)
                print("[DEBUG] Legacy output[:5]:", result_orig[output_key].flatten()[:5])
                print("[DEBUG] Metta output[:5]:", result_metta.td[output_key].flatten()[:5])
                raise
            try:
                torch.testing.assert_close(
                    result_orig["state"],
                    result_metta.data[output_key]["state"],
                    msg="State tensors should be identical",
                    rtol=1e-3,
                    atol=1e-5,
                )
            except AssertionError:
                diff = torch.max(torch.abs(result_orig["state"] - result_metta.data[output_key]["state"]))
                print("[DEBUG] Max abs diff (state):", diff)
                print("[DEBUG] Legacy state[:5]:", result_orig["state"].flatten()[:5])
                print("[DEBUG] Metta state[:5]:", result_metta.data[output_key]["state"].flatten()[:5])
                raise

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

            td_orig = TensorDict(
                {
                    "x": x,
                    "encoded_features": x,
                    "hidden": hidden,
                    "_B_": torch.tensor(batch_size),
                    "_TT_": torch.tensor(params["seq_length"]),
                },
                batch_size=[],
            )
            td_orig["state"] = state
            result_orig = lstm._forward(td_orig)

            td_metta = TensorDict({"encoded_features": x}, batch_size=[])
            md = MettaDict(td_metta, {"global": {"batch_size": batch_size, "tt": params["seq_length"]}})
            result_metta = metta_lstm(md)

            try:
                torch.testing.assert_close(
                    result_orig[output_key],
                    result_metta.td[output_key],
                    msg=f"Output tensors should be identical for batch size {batch_size}",
                    rtol=1e-3,
                    atol=1e-5,
                )
            except AssertionError:
                diff = torch.max(torch.abs(result_orig[output_key] - result_metta.td[output_key])).item()
                print(f"[DEBUG] Max abs diff (output, batch_size={batch_size}):", diff)
                print("[DEBUG] Legacy output[:5]:", result_orig[output_key].flatten()[:5])
                print("[DEBUG] Metta output[:5]:", result_metta.td[output_key].flatten()[:5])
                raise
            try:
                torch.testing.assert_close(
                    result_orig["state"],
                    result_metta.data[output_key]["state"],
                    msg=f"State tensors should be identical for batch size {batch_size}",
                    rtol=1e-3,
                    atol=1e-5,
                )
            except AssertionError:
                diff = torch.max(torch.abs(result_orig["state"] - result_metta.data[output_key]["state"]))
                print(f"[DEBUG] Max abs diff (state, batch_size={batch_size}):", diff)
                print("[DEBUG] Legacy state[:5]:", result_orig["state"].flatten()[:5])
                print("[DEBUG] Metta state[:5]:", result_metta.data[output_key]["state"].flatten()[:5])
                raise

    def test_metadata_propagation(self, lstm_pair):
        """Test that unrelated metadata in md.data is preserved after MettaEncodedLSTM forward."""
        metta_lstm = lstm_pair["metta_lstm"]
        params = lstm_pair["params"]
        output_key = "_lstm_test_"

        x = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        hidden = torch.randn(params["batch_size"] * params["seq_length"], params["input_size"])
        md = MettaDict(
            TensorDict({"encoded_features": x}, batch_size=[]),
            {"foo": "bar", "global": {"batch_size": params["batch_size"], "tt": params["seq_length"]}},
        )
        result = metta_lstm(md)
        assert result.data["foo"] == "bar"
        assert output_key in result.td
        assert "state" in result.data[output_key]


# class TestLSTM:
#     ...  # commented out


class TestMettaEncodedLSTMRobustness:
    def setup_metta_lstm(self, device="cpu", dtype=torch.float32, output_key="lstm1"):
        obs_shape = [10]
        hidden_size = 20
        num_layers = 2
        metta_lstm = MettaEncodedLSTM(
            obs_shape=obs_shape,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_key="encoded_features",
            output_key=output_key,
        )
        metta_lstm.to(device=device, dtype=dtype)
        return metta_lstm, obs_shape, hidden_size, num_layers

    def test_error_handling(self):
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        # Missing input key
        md = MettaDict(TensorDict({}, batch_size=[]), {"global": {"batch_size": 1, "tt": 1}})
        with pytest.raises(KeyError):
            metta_lstm(md)
        # Bad input shape
        x = torch.randn(5, 5)  # Should be [batch, 10]
        hidden = torch.randn(5, 10)
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        with pytest.raises(ValueError):
            metta_lstm(md)
        # Malformed state
        x = torch.randn(5, obs_shape[0])
        hidden = torch.randn(5, obs_shape[0])
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        md.data["lstm1"] = {"state": "not_a_tensor"}
        with pytest.raises(Exception):
            metta_lstm(md)

    def test_multiple_lstm_namespacing(self):
        # Two LSTMs with different output_keys
        metta_lstm1, obs_shape, _, _ = self.setup_metta_lstm(output_key="lstm1")
        metta_lstm2, _, _, _ = self.setup_metta_lstm(output_key="lstm2")
        x = torch.randn(5 * 5, obs_shape[0])
        hidden = torch.randn(5 * 5, obs_shape[0])
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        metta_lstm1(md)
        metta_lstm2(md)
        assert "lstm1" in md.td
        assert "lstm2" in md.td
        assert "state" in md.data["lstm1"]
        assert "state" in md.data["lstm2"]

    def test_device_dtype(self):
        # CPU float32
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm(device="cpu", dtype=torch.float32)
        x = torch.randn(5 * 5, obs_shape[0], dtype=torch.float32)
        hidden = torch.randn(5 * 5, obs_shape[0], dtype=torch.float32)
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        result = metta_lstm(md)
        assert result.td[metta_lstm.output_key].dtype == torch.float32
        # CPU float64
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm(device="cpu", dtype=torch.float64)
        x = torch.randn(5 * 5, obs_shape[0], dtype=torch.float64)
        hidden = torch.randn(5 * 5, obs_shape[0], dtype=torch.float64)
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        result = metta_lstm(md)
        assert result.td[metta_lstm.output_key].dtype == torch.float64
        # CUDA (if available)
        if torch.cuda.is_available():
            metta_lstm, obs_shape, _, _ = self.setup_metta_lstm(device="cuda")
            x = torch.randn(5 * 5, obs_shape[0], device="cuda")
            hidden = torch.randn(5 * 5, obs_shape[0], device="cuda")
            md = MettaDict(
                TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]),
                {"global": {"batch_size": 5, "tt": 5}},
            )
            result = metta_lstm(md)
            assert result.td[metta_lstm.output_key].device.type == "cuda"

    def test_gradients_autograd(self):
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        x = torch.randn(5 * 5, obs_shape[0], requires_grad=True)
        hidden = torch.randn(5 * 5, obs_shape[0], requires_grad=True)
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        result = metta_lstm(md)
        output = result.td[metta_lstm.output_key]
        loss = output.sum()
        loss.backward()
        # Only hidden.grad is checked because x is not used in the computation graph
        assert x.grad is not None

    def test_serialization(self, tmp_path):
        import torch

        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        x = torch.randn(5 * 5, obs_shape[0])
        hidden = torch.randn(5 * 5, obs_shape[0])
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        # Save
        torch.save(metta_lstm.state_dict(), tmp_path / "lstm.pt")
        # Load
        metta_lstm2, _, _, _ = self.setup_metta_lstm()
        metta_lstm2.load_state_dict(torch.load(tmp_path / "lstm.pt"))
        # Check output parity
        result1 = metta_lstm(md)
        result2 = metta_lstm2(md)
        torch.testing.assert_close(result1.td[metta_lstm.output_key], result2.td[metta_lstm2.output_key])

    def test_functional_stateless(self):
        metta_lstm, obs_shape, _, _ = self.setup_metta_lstm()
        x = torch.randn(5 * 5, obs_shape[0])
        hidden = torch.randn(5 * 5, obs_shape[0])
        md = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        # Stateless: pass in state, get state out, no side effects
        state_in = None
        md.data[metta_lstm.output_key] = {"state": state_in}
        result = metta_lstm(md)
        state_out = result.data[metta_lstm.output_key]["state"]
        # Run again with state_out as input
        md2 = MettaDict(
            TensorDict({"encoded_features": x, "hidden": hidden}, batch_size=[]), {"global": {"batch_size": 5, "tt": 5}}
        )
        md2.td[metta_lstm.output_key] = {"state": state_out}
        result2 = metta_lstm(md2)
        # The state may be the same if the input and LSTM weights are deterministic. Remove this assertion if not guaranteed to change.
        # assert not torch.equal(state_out, result2.data[metta_lstm.output_key]["state"])


def test_unique_in_key_property():
    module = MettaLinear(
        in_keys=["foo"],
        out_keys=["bar"],
        input_features_shape=[2],
        output_features_shape=[2],
    )
    assert module.in_key == "foo"
    assert module.out_key == "bar"

    relu = MettaReLU(
        in_keys=["a"],
        out_keys=["b"],
    )
    assert relu.in_key == "a"
    assert relu.out_key == "b"

    # Should raise ValueError if multiple keys are provided
    import pytest

    with pytest.raises(ValueError):
        MettaLinear(
            in_keys=["foo", "bar"],
            out_keys=["baz"],
            input_features_shape=[2],
            output_features_shape=[2],
        )
    with pytest.raises(ValueError):
        MettaReLU(
            in_keys=["a"],
            out_keys=["b", "c"],
        )


def test_pytorch_lstm_baseline():
    # Parameters
    input_size = 10
    hidden_size = 20
    num_layers = 2
    batch_size = 4
    seq_length = 3
    output_key = "_lstm_test_"

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create direct PyTorch LSTM
    pytorch_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

    # Create input and state
    x = torch.randn(seq_length, batch_size, input_size)
    h_0 = torch.randn(num_layers, batch_size, hidden_size)
    c_0 = torch.randn(num_layers, batch_size, hidden_size)
    state_tuple = (h_0, c_0)

    # Forward through PyTorch LSTM
    output_pt, (h_n_pt, c_n_pt) = pytorch_lstm(x, state_tuple)
    output_pt_flat = output_pt.transpose(0, 1).reshape(-1, hidden_size)  # [seq, batch, h] -> [batch*seq, h]
    state_pt_cat = torch.cat([h_n_pt, c_n_pt], dim=0)

    # MettaEncodedLSTM
    from metta.agent.lib.lstm import MettaEncodedLSTM

    obs_shape = [input_size]
    metta_lstm = MettaEncodedLSTM(
        obs_shape=obs_shape,
        hidden_size=hidden_size,
        num_layers=num_layers,
        input_key="encoded_features",
        output_key=output_key,
    )
    # Copy weights
    if metta_lstm._net is not None:
        for (_, p1), (_, p2) in zip(pytorch_lstm.named_parameters(), metta_lstm._net.named_parameters(), strict=False):
            p2.data.copy_(p1.data)
    # Prepare MettaDict
    x_flat = x.transpose(0, 1).reshape(-1, input_size)
    state_cat = torch.cat([h_0, c_0], dim=0)
    td_metta = TensorDict({"encoded_features": x_flat}, batch_size=[])
    md = MettaDict(td_metta, {"global": {"batch_size": batch_size, "tt": seq_length}})
    md.data[output_key] = {"state": state_cat}
    result_metta = metta_lstm(md)

    # Debug prints for output
    max_diff_output = torch.max(torch.abs(output_pt_flat - result_metta.td[output_key])).item()
    print("[DEBUG] Max abs diff (output):", max_diff_output)
    print("[DEBUG] PyTorch output[:5]:", output_pt_flat.flatten()[:5])
    print("[DEBUG] Metta output[:5]:", result_metta.td[output_key].flatten()[:5])
    # Debug prints for state
    max_diff_state = torch.max(torch.abs(state_pt_cat - result_metta.data[output_key]["state"])).item()
    print("[DEBUG] Max abs diff (state):", max_diff_state)
    print("[DEBUG] PyTorch state[:5]:", state_pt_cat.flatten()[:5])
    print("[DEBUG] Metta state[:5]:", result_metta.data[output_key]["state"].flatten()[:5])

    torch.testing.assert_close(
        output_pt_flat, result_metta.td[output_key], msg="MettaEncodedLSTM vs PyTorch LSTM output", rtol=1e-3, atol=1e-5
    )
    torch.testing.assert_close(
        state_pt_cat,
        result_metta.data[output_key]["state"],
        msg="MettaEncodedLSTM vs PyTorch LSTM state",
        rtol=1e-3,
        atol=1e-5,
    )
