import warnings

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.component_container import ComponentContainer
from metta.agent.lib.metta_moduly import (
    Conv2dModule,
    DropoutModule,
    FlattenModule,
    LayerNormModule,
    LinearModule,
    MettaModule,
    ReLUModule,
)
from metta.agent.lib.wrapper_modules import RegularizedModule, SafeModule


class TestEdgeCases:
    """Comprehensive edge case testing for boundary conditions and error scenarios."""

    def test_zero_batch_size(self):
        """Test modules with zero batch size."""
        module = LinearModule(10, 5)

        # Empty batch
        td = TensorDict({"input": torch.empty(0, 10)}, batch_size=0)
        result = module(td)

        assert result["output"].shape == (0, 5)

    def test_single_element_batch(self):
        """Test modules with batch size 1."""
        module = LinearModule(10, 5)

        td = TensorDict({"input": torch.randn(1, 10)}, batch_size=1)
        result = module(td)

        assert result["output"].shape == (1, 5)

    def test_extremely_large_dimensions(self):
        """Test modules with very large feature dimensions."""
        # Large but manageable for testing
        large_dim = 10000
        module = LinearModule(large_dim, 100)

        td = TensorDict({"input": torch.randn(2, large_dim)}, batch_size=2)
        result = module(td)

        assert result["output"].shape == (2, 100)

    def test_minimal_dimensions(self):
        """Test modules with minimal valid dimensions."""
        # 1x1 features
        module = LinearModule(1, 1)

        td = TensorDict({"input": torch.randn(5, 1)}, batch_size=5)
        result = module(td)

        assert result["output"].shape == (5, 1)

    def test_edge_case_conv_sizes(self):
        """Test convolution with edge case input sizes."""
        # 1x1 convolution
        conv = Conv2dModule(1, 1, 1, in_key="input", out_key="output")

        # Minimal 1x1 image
        td = TensorDict({"input": torch.randn(1, 1, 1, 1)}, batch_size=1)
        result = conv(td)

        assert result["output"].shape == (1, 1, 1, 1)

    def test_inf_values_in_safe_module(self):
        """Test SafeModule with infinity values."""
        safe_module = SafeModule(LinearModule(5, 3), nan_check=True)

        # Input with infinity
        td_inf = TensorDict({"input": torch.tensor([[float("inf"), 1, 2, 3, 4]])}, batch_size=1)

        with pytest.raises(ValueError, match="Inf detected"):
            safe_module(td_inf)

    def test_very_small_values(self):
        """Test modules with very small floating point values."""
        module = LinearModule(3, 2)

        # Very small values near floating point precision
        tiny_values = torch.tensor([[1e-38, 1e-38, 1e-38]], dtype=torch.float32)
        td = TensorDict({"input": tiny_values}, batch_size=1)

        result = module(td)
        assert result["output"].shape == (1, 2)
        assert torch.isfinite(result["output"]).all()

    def test_very_large_values(self):
        """Test modules with very large values."""
        module = LinearModule(3, 2)

        # Large but finite values
        large_values = torch.tensor([[1e10, 1e10, 1e10]], dtype=torch.float32)
        td = TensorDict({"input": large_values}, batch_size=1)

        result = module(td)
        assert result["output"].shape == (1, 2)

    def test_mixed_precision_types(self):
        """Test modules with different tensor dtypes."""
        module = LinearModule(3, 2)

        # Test with different dtypes
        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            # Convert module to matching dtype
            module = module.to(dtype)

            td = TensorDict({"input": torch.randn(2, 3, dtype=dtype)}, batch_size=2)
            result = module(td)

            assert result["output"].shape == (2, 2)
            # Output dtype should match module weight dtype
            assert result["output"].dtype == module.linear.weight.dtype

    def test_zero_dropout_probability(self):
        """Test dropout with p=0.0 (no dropout)."""
        dropout = DropoutModule(p=0.0)

        td = TensorDict({"input": torch.randn(10, 5)}, batch_size=10)
        original_input = td["input"].clone()

        result = dropout(td)

        # With p=0, input should be unchanged
        assert torch.equal(result["output"], original_input)

    def test_maximum_dropout_probability(self):
        """Test dropout with p=1.0 (complete dropout)."""
        dropout = DropoutModule(p=1.0)
        dropout.train()  # Ensure training mode

        td = TensorDict({"input": torch.randn(10, 5)}, batch_size=10)
        result = dropout(td)

        # With p=1.0 in training mode, output should be all zeros
        assert torch.all(result["output"] == 0)

    def test_negative_values_with_relu(self):
        """Test ReLU with all negative inputs."""
        relu = ReLUModule()

        # All negative values
        td = TensorDict({"input": torch.tensor([[-1.0, -2.0, -3.0]])}, batch_size=1)
        result = relu(td)

        # Should be all zeros
        assert torch.all(result["output"] == 0)

    def test_component_container_empty(self):
        """Test ComponentContainer with no components."""
        container = ComponentContainer()

        # Should handle empty container gracefully
        assert len(container) == 0
        assert repr(container) == "ComponentContainer with 0 components:"

    def test_component_container_nonexistent_component(self):
        """Test accessing non-existent component."""
        container = ComponentContainer()

        with pytest.raises(KeyError):
            container.forward("nonexistent", TensorDict({}, batch_size=1))

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        container = ComponentContainer()

        # Create circular dependency: A -> B -> A
        a = LinearModule(5, 5, "input", "output")
        b = LinearModule(5, 5, "input", "output")

        container.register_component("A", a, dependencies=["B"])
        container.register_component("B", b, dependencies=["A"])

        with pytest.raises(ValueError, match="Circular dependency"):
            container.validate_dependencies()

    def test_self_dependency(self):
        """Test component depending on itself."""
        container = ComponentContainer()

        module = LinearModule(5, 5, "input", "output")
        container.register_component("self_dep", module, dependencies=["self_dep"])

        with pytest.raises(ValueError, match="Circular dependency"):
            container.validate_dependencies()

    def test_empty_keys_in_module(self):
        """Test modules with empty key lists."""
        with pytest.raises((ValueError, TypeError)):
            # Should fail during construction or validation
            module = MettaModule(in_keys=[], out_keys=[])

    def test_duplicate_output_keys(self):
        """Test modules writing to same output keys."""
        module1 = LinearModule(5, 3, "input1", "shared_output")
        module2 = LinearModule(5, 3, "input2", "shared_output")

        td = TensorDict({"input1": torch.randn(2, 5), "input2": torch.randn(2, 5)}, batch_size=2)

        # First module writes to shared_output
        result = module1(td)
        original_output = result["shared_output"].clone()

        # Second module overwrites shared_output
        result = module2(result)

        # Output should be different (overwritten)
        assert not torch.equal(result["shared_output"], original_output)

    def test_extremely_deep_pipeline(self):
        """Test very deep pipeline (many sequential modules)."""
        num_modules = 100
        feature_size = 10

        # Create deep pipeline
        modules = []
        for i in range(num_modules):
            in_key = "input" if i == 0 else f"hidden_{i - 1}"
            out_key = "output" if i == num_modules - 1 else f"hidden_{i}"
            modules.append(LinearModule(feature_size, feature_size, in_key, out_key))

        # Test data
        td = TensorDict({"input": torch.randn(2, feature_size)}, batch_size=2)

        # Execute deep pipeline
        result = td
        for module in modules:
            result = module(result)

        assert result["output"].shape == (2, feature_size)

    def test_weight_initialization_edge_cases(self):
        """Test modules with extreme weight initializations."""
        module = LinearModule(3, 2)

        # Initialize with very small weights
        nn.init.constant_(module.linear.weight, 1e-10)
        nn.init.constant_(module.linear.bias, 0)

        td = TensorDict({"input": torch.randn(5, 3)}, batch_size=5)
        result = module(td)

        # Should still work with tiny weights
        assert result["output"].shape == (5, 2)
        assert torch.isfinite(result["output"]).all()

    def test_regularization_with_zero_weights(self):
        """Test regularization when all weights are zero."""
        module = LinearModule(3, 2)

        # Zero out all weights
        nn.init.zeros_(module.linear.weight)
        nn.init.zeros_(module.linear.bias)

        reg_module = RegularizedModule(module, l2_scale=0.1, l1_scale=0.1)
        reg_module.train()

        td = TensorDict({"input": torch.randn(2, 3)}, batch_size=2)
        result = reg_module(td)

        # Regularization loss should be zero (or very small)
        assert "regularization_loss" in result
        assert torch.allclose(result["regularization_loss"], torch.zeros_like(result["regularization_loss"]))

    def test_action_bounds_edge_cases(self):
        """Test SafeModule action bounds with edge cases."""
        module = LinearModule(3, 2, "input", "action")

        # Test with very tight bounds
        safe_module = SafeModule(module, action_bounds=(-0.001, 0.001))

        td = TensorDict({"input": torch.randn(2, 3)}, batch_size=2)
        result = safe_module(td)

        # All actions should be within tight bounds
        assert torch.all(result["action"] >= -0.001)
        assert torch.all(result["action"] <= 0.001)

    def test_layer_norm_single_feature(self):
        """Test LayerNorm with single feature dimension."""
        norm = LayerNormModule(1)

        td = TensorDict({"input": torch.randn(10, 1)}, batch_size=10)
        result = norm(td)

        assert result["output"].shape == (10, 1)

    def test_conv_with_padding_larger_than_kernel(self):
        """Test convolution with padding larger than kernel size."""
        # This is valid in PyTorch
        conv = Conv2dModule(1, 1, kernel_size=3, padding=5, in_key="input", out_key="output")

        td = TensorDict({"input": torch.randn(1, 1, 5, 5)}, batch_size=1)
        result = conv(td)

        # Should work without error
        assert "output" in result
        assert result["output"].shape[0] == 1  # Batch dimension preserved

    def test_flatten_different_start_dims(self):
        """Test flatten with different start dimensions."""
        # Test flattening from dimension 0 (including batch)
        flatten_all = FlattenModule(start_dim=0, in_key="input", out_key="output")

        td = TensorDict({"input": torch.randn(2, 3, 4)}, batch_size=2)
        result = flatten_all(td)

        # Should flatten everything into 1D
        assert result["output"].shape == (24,)  # 2*3*4

    def test_module_with_no_parameters(self):
        """Test modules that have no learnable parameters."""
        relu = ReLUModule()
        flatten = FlattenModule()

        # These modules should have no parameters
        assert len(list(relu.parameters())) == 0
        assert len(list(flatten.parameters())) == 0

        # But should still work normally
        td = TensorDict({"input": torch.randn(2, 5)}, batch_size=2)

        result = relu(td)
        assert result["output"].shape == (2, 5)

        result = flatten(td.clone())
        assert result["output"].shape == (2, 5)  # Already 2D, no change

    def test_warning_suppression(self):
        """Test that expected warnings are handled appropriately."""
        # Some operations might generate warnings that we should handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Operations that might generate warnings
            module = LinearModule(1000, 1000)
            large_batch = TensorDict({"input": torch.randn(1000, 1000)}, batch_size=1000)
            result = module(large_batch)

            assert result["output"].shape == (1000, 1000)

    def test_device_consistency(self):
        """Test that modules maintain device consistency."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        module = LinearModule(5, 3).to(device)

        # Input on same device
        td = TensorDict({"input": torch.randn(2, 5, device=device)}, batch_size=2)
        result = module(td)

        assert result["output"].device == device

    def test_gradient_edge_cases(self):
        """Test gradient computation in edge cases."""
        module = LinearModule(3, 2)

        # Input requiring gradients
        td = TensorDict({"input": torch.randn(2, 3, requires_grad=True)}, batch_size=2)

        result = module(td)
        loss = result["output"].sum()

        # Should be able to compute gradients
        loss.backward()

        assert td["input"].grad is not None
        assert torch.isfinite(td["input"].grad).all()

    def test_memory_pressure_scenarios(self):
        """Test modules under memory pressure."""
        # Large tensors that push memory limits
        try:
            large_size = 10000
            module = LinearModule(large_size, 100)

            # Process in smaller chunks to avoid OOM
            for _ in range(3):
                td = TensorDict({"input": torch.randn(10, large_size)}, batch_size=10)
                result = module(td)

                # Clear intermediate results
                del td, result

            assert True  # Test passes if no OOM

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for large tensor test")
            else:
                raise
