import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_moduly import Conv2dModule, LinearModule, ReLUModule
from metta.agent.lib.wrapper_modules import RegularizedModule, SafeModule, WeightMonitoringModule


def test_safe_module_basic_functionality():
    """Test SafeModule wraps computation correctly."""
    base_module = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
    safe_module = SafeModule(base_module, nan_check=True)

    # Test normal operation
    td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
    result = safe_module(td)
    assert "output" in result
    assert result["output"].shape == (2, 5)


def test_safe_module_nan_detection():
    """Test SafeModule detects NaN in inputs and outputs."""
    base_module = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
    safe_module = SafeModule(base_module, nan_check=True)

    # Test NaN detection in input
    td_nan = TensorDict({"input": torch.tensor([[float("nan")] * 10, [1.0] * 10])}, batch_size=2)
    with pytest.raises(ValueError, match="NaN detected in input"):
        safe_module(td_nan)


def test_safe_module_action_bounds():
    """Test SafeModule applies action bounds correctly."""
    base_module = LinearModule(in_features=5, out_features=3, in_key="features", out_key="test_action")
    safe_module = SafeModule(base_module, action_bounds=(-1.0, 1.0))

    # Create input that will likely produce outputs outside bounds
    td = TensorDict({"features": torch.ones(1, 5) * 10}, batch_size=1)
    result = safe_module(td)

    # Check that action outputs are clipped to bounds
    assert torch.all(result["test_action"] >= -1.0)
    assert torch.all(result["test_action"] <= 1.0)


def test_safe_module_inheritance():
    """Test SafeModule inherits in_keys and out_keys from wrapped module."""
    base_module = Conv2dModule(in_channels=3, out_channels=16, kernel_size=3, in_key="image", out_key="features")
    safe_module = SafeModule(base_module)

    assert safe_module.in_keys == ["image"]
    assert safe_module.out_keys == ["features"]


def test_regularized_module_basic_functionality():
    """Test RegularizedModule wraps computation correctly."""
    base_module = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
    reg_module = RegularizedModule(base_module, l2_scale=0.01)

    # Test normal operation
    td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
    result = reg_module(td)
    assert "output" in result
    assert result["output"].shape == (2, 5)


def test_regularized_module_adds_regularization_loss():
    """Test RegularizedModule adds regularization loss during training."""
    base_module = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
    reg_module = RegularizedModule(base_module, l2_scale=0.01, l1_scale=0.001)

    # Set to training mode
    reg_module.train()

    td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
    result = reg_module(td)

    # Should have regularization loss added
    assert "regularization_loss" in result
    # Check that all batch elements have positive loss
    assert torch.all(result["regularization_loss"] > 0)
    assert result["regularization_loss"].shape == (2,)  # Should match batch size


def test_regularized_module_no_loss_in_eval():
    """Test RegularizedModule doesn't add loss in eval mode."""
    base_module = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
    reg_module = RegularizedModule(base_module, l2_scale=0.01)

    # Set to eval mode
    reg_module.eval()

    td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
    result = reg_module(td)

    # Should not have regularization loss in eval mode
    assert "regularization_loss" not in result


def test_weight_monitoring_module_basic_functionality():
    """Test WeightMonitoringModule wraps computation correctly."""
    base_module = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
    monitor_module = WeightMonitoringModule(base_module, monitor_health=True)

    # Test normal operation
    td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
    result = monitor_module(td)
    assert "output" in result
    assert result["output"].shape == (2, 5)


def test_weight_monitoring_module_clipping():
    """Test WeightMonitoringModule clips weights correctly."""
    base_module = LinearModule(in_features=5, out_features=3, in_key="input", out_key="output")

    # Initialize with large weights
    with torch.no_grad():
        base_module.linear.weight.fill_(10.0)

    monitor_module = WeightMonitoringModule(base_module, clip_value=1.0)
    monitor_module.train()

    # Execute forward pass (clipping happens as side effect)
    td = TensorDict({"input": torch.randn(2, 5)}, batch_size=2)
    monitor_module(td)

    # Check that weights were clipped
    assert torch.all(base_module.linear.weight.abs() <= 1.0)


def test_wrapper_composability():
    """Test that wrappers can be arbitrarily nested."""
    # Build up layers of functionality
    base_module = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")

    # Wrap with safety
    safe_module = SafeModule(base_module, nan_check=True)

    # Wrap safe module with regularization
    reg_safe_module = RegularizedModule(safe_module, l2_scale=0.01)

    # Wrap everything with monitoring
    final_module = WeightMonitoringModule(reg_safe_module, monitor_health=True)

    # Test that it works
    final_module.train()
    td = TensorDict({"input": torch.randn(3, 10)}, batch_size=3)
    result = final_module(td)

    assert "output" in result
    assert result["output"].shape == (3, 5)
    assert "regularization_loss" in result  # From RegularizedModule

    # Test that keys are inherited correctly through all wrappers
    assert final_module.in_keys == ["input"]
    assert final_module.out_keys == ["output"]


def test_wrapper_different_base_modules():
    """Test wrappers work with different types of base modules."""
    # Test with Conv2d
    conv_module = Conv2dModule(in_channels=3, out_channels=16, kernel_size=3, in_key="image", out_key="features")
    safe_conv = SafeModule(conv_module)

    td = TensorDict({"image": torch.randn(2, 3, 28, 28)}, batch_size=2)
    result = safe_conv(td)
    assert result["features"].shape == (2, 16, 26, 26)

    # Test with ReLU (module with no parameters)
    relu_module = ReLUModule(in_key="x", out_key="y")
    reg_relu = RegularizedModule(relu_module, l2_scale=0.001)
    reg_relu.train()

    td = TensorDict({"x": torch.randn(4, 10)}, batch_size=4)
    result = reg_relu(td)
    assert result["y"].shape == (4, 10)
    # ReLU has no parameters, so no regularization loss should be added
    assert "regularization_loss" not in result
