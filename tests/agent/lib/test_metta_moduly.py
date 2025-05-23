import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_modules import (
    Conv2dModule,
    DropoutModule,
    FlattenModule,
    LayerNormModule,
    LinearModule,
    MettaModule,
    ReLUModule,
)


def test_metta_module_init():
    m = MettaModule(in_keys=["a"], out_keys=["b"], input_shapes={"a": (3,)})
    assert m.in_keys == ["a"]
    assert m.out_keys == ["b"]
    assert m.input_shapes["a"] == (3,)
    assert m.output_shapes == {}


def test_metta_module_validate_shapes_pass():
    m = MettaModule(in_keys=["a"], input_shapes={"a": (2,)})
    td = TensorDict({"a": torch.randn(4, 2)}, batch_size=4)
    m.validate_shapes(td)  # Should not raise


def test_metta_module_validate_shapes_fail():
    m = MettaModule(in_keys=["a"], input_shapes={"a": (2,)})
    td = TensorDict({"a": torch.randn(4, 3)}, batch_size=4)
    with pytest.raises(ValueError, match="Input shape mismatch for 'a'"):
        m.validate_shapes(td)


def test_linear_module_forward_success():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(5, 3)}, batch_size=5)
    out_td = module(td)
    assert "y" in out_td
    assert out_td["y"].shape == (5, 2)


def test_linear_module_forward_shape_mismatch():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(5, 4)}, batch_size=5)
    with pytest.raises(ValueError, match="Input shape mismatch for 'x'"):
        module(td)


def test_linear_module_forward_missing_key():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({}, batch_size=5)
    with pytest.raises(KeyError):
        module(td)


# New tests for edge cases and additional scenarios


def test_metta_module_empty_keys():
    m = MettaModule(in_keys=[], out_keys=[])
    assert m.in_keys == []
    assert m.out_keys == []


def test_metta_module_none_keys():
    m = MettaModule(in_keys=None, out_keys=None)
    assert m.in_keys == []
    assert m.out_keys == []


def test_linear_module_batch_size_one():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(1, 3)}, batch_size=1)
    out_td = module(td)
    assert "y" in out_td
    assert out_td["y"].shape == (1, 2)


def test_linear_module_large_batch_size():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(100, 3)}, batch_size=100)
    out_td = module(td)
    assert "y" in out_td
    assert out_td["y"].shape == (100, 2)


def test_linear_module_custom_keys():
    module = LinearModule(in_features=3, out_features=2, in_key="custom_in", out_key="custom_out")
    td = TensorDict({"custom_in": torch.randn(5, 3)}, batch_size=5)
    out_td = module(td)
    assert "custom_out" in out_td
    assert out_td["custom_out"].shape == (5, 2)


def test_linear_module_integration():
    module1 = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    module2 = LinearModule(in_features=2, out_features=1, in_key="y", out_key="z")
    td = TensorDict({"x": torch.randn(5, 3)}, batch_size=5)
    out_td = module1(td)
    out_td = module2(out_td)
    assert "z" in out_td
    assert out_td["z"].shape == (5, 1)


# ReLU Module Tests


def test_relu_module_forward_success():
    module = ReLUModule(in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(5, 3)}, batch_size=5)
    out_td = module(td)
    assert "y" in out_td
    assert out_td["y"].shape == (5, 3)
    # Check that ReLU was applied (all values >= 0)
    assert torch.all(out_td["y"] >= 0)


def test_relu_module_negative_inputs():
    module = ReLUModule(in_key="x", out_key="y")
    # Create tensor with negative values
    td = TensorDict({"x": torch.tensor([[-1.0, 2.0, -3.0]], dtype=torch.float32)}, batch_size=1)
    out_td = module(td)
    assert "y" in out_td
    # Check that negative values become 0
    expected = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(out_td["y"], expected)


def test_relu_module_custom_keys():
    module = ReLUModule(in_key="input_features", out_key="activated_features")
    td = TensorDict({"input_features": torch.randn(3, 4)}, batch_size=3)
    out_td = module(td)
    assert "activated_features" in out_td
    assert out_td["activated_features"].shape == (3, 4)


def test_relu_module_missing_key():
    module = ReLUModule(in_key="x", out_key="y")
    td = TensorDict({}, batch_size=5)
    with pytest.raises(KeyError):
        module(td)


def test_linear_relu_integration():
    linear = LinearModule(in_features=3, out_features=2, in_key="x", out_key="hidden")
    relu = ReLUModule(in_key="hidden", out_key="activated")

    td = TensorDict({"x": torch.randn(5, 3)}, batch_size=5)
    td = linear(td)
    td = relu(td)

    assert "activated" in td
    assert td["activated"].shape == (5, 2)
    assert torch.all(td["activated"] >= 0)


# Conv2d Module Tests


def test_conv2d_module_forward_success():
    module = Conv2dModule(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, in_key="image", out_key="features"
    )
    # Create a 3-channel 32x32 image
    td = TensorDict({"image": torch.randn(2, 3, 32, 32)}, batch_size=2)
    out_td = module(td)
    assert "features" in out_td
    assert out_td["features"].shape == (2, 16, 32, 32)  # Same size due to padding=1


def test_conv2d_module_stride_padding():
    module = Conv2dModule(in_channels=1, out_channels=8, kernel_size=5, stride=3, padding=0, in_key="x", out_key="y")
    # Create a 1-channel 64x64 image
    td = TensorDict({"x": torch.randn(1, 1, 64, 64)}, batch_size=1)
    out_td = module(td)
    assert "y" in out_td
    # Output size calculation: (64 - 5) / 3 + 1 = 20
    assert out_td["y"].shape == (1, 8, 20, 20)


def test_conv2d_module_custom_keys():
    module = Conv2dModule(in_channels=3, out_channels=64, kernel_size=3, in_key="input_tensor", out_key="conv_output")
    td = TensorDict({"input_tensor": torch.randn(4, 3, 16, 16)}, batch_size=4)
    out_td = module(td)
    assert "conv_output" in out_td
    assert out_td["conv_output"].shape == (4, 64, 14, 14)  # 16-3+1=14


def test_conv2d_module_missing_key():
    module = Conv2dModule(in_channels=3, out_channels=16, kernel_size=3, in_key="image", out_key="features")
    td = TensorDict({}, batch_size=2)
    with pytest.raises(KeyError):
        module(td)


def test_conv2d_integration():
    conv1 = Conv2dModule(in_channels=3, out_channels=16, kernel_size=3, padding=1, in_key="image", out_key="conv1_out")
    relu = ReLUModule(in_key="conv1_out", out_key="relu_out")
    conv2 = Conv2dModule(
        in_channels=16, out_channels=32, kernel_size=3, padding=1, in_key="relu_out", out_key="conv2_out"
    )

    td = TensorDict({"image": torch.randn(2, 3, 28, 28)}, batch_size=2)
    td = conv1(td)
    td = relu(td)
    td = conv2(td)

    assert "conv2_out" in td
    assert td["conv2_out"].shape == (2, 32, 28, 28)
    # Verify that ReLU was applied in the middle
    assert torch.all(td["relu_out"] >= 0)


# Flatten Module Tests


def test_flatten_module_forward_success():
    module = FlattenModule(start_dim=1, in_key="features", out_key="flattened")
    # Create a 4D tensor (batch, channels, height, width)
    td = TensorDict({"features": torch.randn(2, 16, 8, 8)}, batch_size=2)
    out_td = module(td)
    assert "flattened" in out_td
    # Should flatten from dim 1: (2, 16*8*8) = (2, 1024)
    assert out_td["flattened"].shape == (2, 1024)


def test_flatten_module_different_start_dim():
    module = FlattenModule(start_dim=2, in_key="x", out_key="y")
    # Create a 4D tensor
    td = TensorDict({"x": torch.randn(3, 4, 5, 6)}, batch_size=3)
    out_td = module(td)
    assert "y" in out_td
    # Should flatten from dim 2: (3, 4, 5*6) = (3, 4, 30)
    assert out_td["y"].shape == (3, 4, 30)


def test_flatten_module_custom_keys():
    module = FlattenModule(start_dim=1, in_key="conv_output", out_key="flat_features")
    td = TensorDict({"conv_output": torch.randn(1, 64, 7, 7)}, batch_size=1)
    out_td = module(td)
    assert "flat_features" in out_td
    assert out_td["flat_features"].shape == (1, 64 * 7 * 7)


def test_flatten_module_missing_key():
    module = FlattenModule(in_key="features", out_key="flattened")
    td = TensorDict({}, batch_size=2)
    with pytest.raises(KeyError):
        module(td)


def test_cnn_flatten_linear_integration():
    """Test a typical CNN -> Flatten -> Linear pipeline."""
    conv = Conv2dModule(in_channels=3, out_channels=16, kernel_size=3, padding=1, in_key="image", out_key="conv_out")
    flatten = FlattenModule(start_dim=1, in_key="conv_out", out_key="flat_features")
    linear = LinearModule(in_features=16 * 28 * 28, out_features=128, in_key="flat_features", out_key="output")

    td = TensorDict({"image": torch.randn(2, 3, 28, 28)}, batch_size=2)
    td = conv(td)
    td = flatten(td)
    td = linear(td)

    assert "output" in td
    assert td["output"].shape == (2, 128)


# LayerNorm Module Tests


def test_layernorm_module_forward_success():
    module = LayerNormModule(normalized_shape=128, in_key="features", out_key="normalized")
    td = TensorDict({"features": torch.randn(4, 128)}, batch_size=4)
    out_td = module(td)
    assert "normalized" in out_td
    assert out_td["normalized"].shape == (4, 128)
    # Check that normalization was applied (mean ~0, std ~1)
    assert abs(out_td["normalized"].mean().item()) < 0.1
    assert abs(out_td["normalized"].std().item() - 1.0) < 0.1


def test_layernorm_module_shape_validation():
    module = LayerNormModule(normalized_shape=64, in_key="x", out_key="y")
    # Correct shape
    td = TensorDict({"x": torch.randn(2, 64)}, batch_size=2)
    out_td = module(td)
    assert out_td["y"].shape == (2, 64)

    # Wrong shape should raise error
    td_wrong = TensorDict({"x": torch.randn(2, 32)}, batch_size=2)
    with pytest.raises(ValueError, match="Input shape mismatch"):
        module(td_wrong)


def test_layernorm_module_custom_keys():
    module = LayerNormModule(normalized_shape=256, in_key="core_output", out_key="norm_core")
    td = TensorDict({"core_output": torch.randn(8, 256)}, batch_size=8)
    out_td = module(td)
    assert "norm_core" in out_td
    assert out_td["norm_core"].shape == (8, 256)


def test_layernorm_module_missing_key():
    module = LayerNormModule(normalized_shape=128, in_key="features", out_key="normalized")
    td = TensorDict({}, batch_size=4)
    with pytest.raises(KeyError):
        module(td)


# Dropout Module Tests


def test_dropout_module_forward_success():
    module = DropoutModule(p=0.5, in_key="features", out_key="dropped")
    td = TensorDict({"features": torch.randn(4, 64)}, batch_size=4)
    out_td = module(td)
    assert "dropped" in out_td
    assert out_td["dropped"].shape == (4, 64)


def test_dropout_module_training_vs_eval():
    module = DropoutModule(p=0.9, in_key="x", out_key="y")  # High dropout rate
    input_tensor = torch.ones(2, 100)  # All ones
    td = TensorDict({"x": input_tensor}, batch_size=2)

    # Training mode: should drop many values
    module.train()
    out_td_train = module(td.clone())
    num_zeros_train = (out_td_train["y"] == 0).sum().item()

    # Eval mode: should keep all values
    module.eval()
    out_td_eval = module(td.clone())
    num_zeros_eval = (out_td_eval["y"] == 0).sum().item()

    # In training, many should be zero; in eval, none should be zero
    assert num_zeros_train > num_zeros_eval


def test_dropout_module_different_probabilities():
    # Test with p=0.0 (no dropout)
    module_none = DropoutModule(p=0.0, in_key="x", out_key="y")
    module_none.train()
    td = TensorDict({"x": torch.ones(2, 100)}, batch_size=2)
    out_td = module_none(td)
    assert torch.all(out_td["y"] == 1.0)  # No values should be dropped


def test_dropout_module_custom_keys():
    module = DropoutModule(p=0.3, in_key="encoded_obs", out_key="regularized_obs")
    td = TensorDict({"encoded_obs": torch.randn(6, 32)}, batch_size=6)
    out_td = module(td)
    assert "regularized_obs" in out_td
    assert out_td["regularized_obs"].shape == (6, 32)


def test_dropout_module_missing_key():
    module = DropoutModule(p=0.5, in_key="features", out_key="dropped")
    td = TensorDict({}, batch_size=4)
    with pytest.raises(KeyError):
        module(td)


def test_linear_norm_dropout_integration():
    """Test a typical Linear -> LayerNorm -> Dropout pipeline."""
    linear = LinearModule(in_features=64, out_features=128, in_key="input", out_key="linear_out")
    norm = LayerNormModule(normalized_shape=128, in_key="linear_out", out_key="norm_out")
    dropout = DropoutModule(p=0.1, in_key="norm_out", out_key="final_out")

    td = TensorDict({"input": torch.randn(3, 64)}, batch_size=3)
    td = linear(td)
    td = norm(td)
    td = dropout(td)

    assert "final_out" in td
    assert td["final_out"].shape == (3, 128)
