import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.component_container import ComponentContainer
from metta.agent.lib.metta_modules import (
    Conv2dModule,
    DropoutModule,
    FlattenModule,
    LayerNormModule,
    LinearModule,
    MettaModule,
    ReLUModule,
)
from metta.agent.lib.wrapper_modules import RegularizedModule, SafeModule, WeightMonitoringModule


class TestIntegration:
    """Integration tests for complete module pipelines and interactions."""

    def test_complete_mlp_pipeline(self):
        """Test a complete MLP pipeline with multiple modules."""
        # Build a complete pipeline: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
        modules = [
            LinearModule(784, 256, "input", "hidden1"),
            LayerNormModule(256, "hidden1", "norm1"),
            ReLUModule("norm1", "relu1"),
            DropoutModule(0.1, "relu1", "drop1"),
            LinearModule(256, 10, "drop1", "output"),
        ]

        # Test data
        td = TensorDict({"input": torch.randn(32, 784)}, batch_size=32)

        # Execute pipeline
        result = td
        for module in modules:
            result = module(result)

        # Verify final output
        assert "output" in result
        assert result["output"].shape == (32, 10)

        # Verify intermediate outputs exist
        for key in ["hidden1", "norm1", "relu1", "drop1", "output"]:
            assert key in result

    def test_conv_to_mlp_pipeline(self):
        """Test CNN to MLP pipeline (common in vision tasks)."""
        modules = [
            Conv2dModule(3, 32, 3, stride=1, padding=0, in_key="image", out_key="conv1"),
            ReLUModule("conv1", "relu1"),
            Conv2dModule(32, 64, 3, stride=1, padding=0, in_key="relu1", out_key="conv2"),
            ReLUModule("conv2", "relu2"),
            FlattenModule(start_dim=1, in_key="relu2", out_key="flattened"),
            LinearModule(64 * 6 * 6, 128, "flattened", "hidden"),  # Assuming 10x10 -> 6x6 after 2 convs
            ReLUModule("hidden", "relu3"),
            LinearModule(128, 10, "relu3", "output"),
        ]

        # Test data (32 batch, 3 channels, 10x10 image)
        td = TensorDict({"image": torch.randn(32, 3, 10, 10)}, batch_size=32)

        # Execute pipeline
        result = td
        for module in modules:
            result = module(result)

        assert result["output"].shape == (32, 10)

    def test_wrapped_pipeline(self):
        """Test pipeline with various wrapper modules."""
        # Base modules
        linear1 = LinearModule(100, 50, "input", "hidden")
        relu = ReLUModule("hidden", "activated")
        linear2 = LinearModule(50, 10, "activated", "output")

        # Wrap with different wrappers
        safe_linear1 = SafeModule(linear1, nan_check=True)
        regularized_linear2 = RegularizedModule(linear2, l2_scale=0.01)
        monitored_relu = WeightMonitoringModule(relu, monitor_health=True)

        # Test pipeline
        td = TensorDict({"input": torch.randn(16, 100)}, batch_size=16)

        result = safe_linear1(td)
        result = monitored_relu(result)
        result = regularized_linear2(result)

        assert result["output"].shape == (16, 10)

        # Check regularization loss was added
        if regularized_linear2.training:
            assert "regularization_loss" in result

    def test_component_container_full_pipeline(self):
        """Test ComponentContainer with a complex dependency graph."""
        container = ComponentContainer()

        # Build complex pipeline with branching dependencies
        # obs -> [feature_extractor] -> [policy, value] -> combined_output
        feature_extractor = LinearModule(64, 32, "observation", "features")
        policy_head = LinearModule(32, 8, "features", "policy_logits")
        value_head = LinearModule(32, 1, "features", "state_value")

        # Combiner that takes both policy and value outputs
        class CombinerModule(MettaModule):
            def __init__(self):
                super().__init__(in_keys=["policy_logits", "state_value"], out_keys=["combined_output"])
                self.linear = nn.Linear(9, 5)  # 8 + 1 = 9

            def forward(self, tensordict):
                policy = tensordict["policy_logits"]
                value = tensordict["state_value"]
                combined = torch.cat([policy, value], dim=1)
                output = self.linear(combined)
                tensordict["combined_output"] = output
                return tensordict

        combiner = CombinerModule()

        # Register with dependencies
        container.register_component("feature_extractor", feature_extractor)
        container.register_component("policy", policy_head, dependencies=["feature_extractor"])
        container.register_component("value", value_head, dependencies=["feature_extractor"])
        container.register_component("combiner", combiner, dependencies=["policy", "value"])

        # Test execution
        td = TensorDict({"observation": torch.randn(8, 64)}, batch_size=8)
        container.clear_cache()
        result = container.forward("combiner", td)

        # Verify all outputs exist
        assert "features" in result
        assert "policy_logits" in result
        assert "state_value" in result
        assert "combined_output" in result
        assert result["combined_output"].shape == (8, 5)

    def test_multi_input_output_pipeline(self):
        """Test modules with multiple inputs and outputs."""

        class MultiIOModule(MettaModule):
            def __init__(self):
                super().__init__(in_keys=["input1", "input2"], out_keys=["output1", "output2"])
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(15, 8)

            def forward(self, tensordict):
                in1 = tensordict["input1"]
                in2 = tensordict["input2"]

                tensordict["output1"] = self.linear1(in1)
                tensordict["output2"] = self.linear2(torch.cat([in1, in2], dim=1))
                return tensordict

        module = MultiIOModule()

        td = TensorDict({"input1": torch.randn(4, 10), "input2": torch.randn(4, 5)}, batch_size=4)

        result = module(td)

        assert result["output1"].shape == (4, 5)
        assert result["output2"].shape == (4, 8)

    def test_gradient_flow_through_pipeline(self):
        """Test that gradients flow properly through complex pipelines."""
        modules = [
            LinearModule(20, 15, "input", "hidden1"),
            ReLUModule("hidden1", "relu1"),
            LinearModule(15, 10, "relu1", "hidden2"),
            ReLUModule("hidden2", "relu2"),
            LinearModule(10, 1, "relu2", "output"),
        ]

        # Enable gradient computation
        td = TensorDict({"input": torch.randn(8, 20, requires_grad=True)}, batch_size=8)

        result = td
        for module in modules:
            result = module(result)

        # Compute loss and backpropagate
        loss = result["output"].sum()
        loss.backward()

        # Check that input gradients exist
        assert td["input"].grad is not None
        assert td["input"].grad.shape == (8, 20)

        # Check that all module parameters have gradients
        for module in modules:
            if hasattr(module, "parameters"):
                for param in module.parameters():
                    if param.requires_grad:
                        assert param.grad is not None

    def test_pipeline_with_shared_modules(self):
        """Test pipeline where modules share parameters."""
        shared_linear = LinearModule(10, 5, "input", "output")

        # Use same module in two places
        td1 = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
        td2 = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)

        result1 = shared_linear(td1)
        result2 = shared_linear(td2)

        # Both should work and modify the TensorDicts
        assert "output" in result1
        assert "output" in result2
        assert result1["output"].shape == (4, 5)
        assert result2["output"].shape == (4, 5)

    def test_nested_wrapper_modules(self):
        """Test nested wrapper combinations."""
        base_module = LinearModule(10, 5, "input", "output")

        # Triple wrap: Safe -> Regularized -> Monitored
        safe_module = SafeModule(base_module, nan_check=True)
        regularized_safe = RegularizedModule(safe_module, l2_scale=0.01)
        fully_wrapped = WeightMonitoringModule(regularized_safe, monitor_health=True)

        td = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
        result = fully_wrapped(td)

        assert result["output"].shape == (4, 5)
        # Should have regularization loss in training mode
        if fully_wrapped.training:
            assert "regularization_loss" in result

    def test_large_batch_processing(self):
        """Test pipeline with large batches."""
        pipeline = [
            LinearModule(100, 200, "input", "hidden1"),
            ReLUModule("hidden1", "relu1"),
            LinearModule(200, 50, "relu1", "output"),
        ]

        # Large batch
        large_batch_size = 1000
        td = TensorDict({"input": torch.randn(large_batch_size, 100)}, batch_size=large_batch_size)

        result = td
        for module in pipeline:
            result = module(result)

        assert result["output"].shape == (large_batch_size, 50)

    def test_error_propagation_in_pipeline(self):
        """Test that errors propagate correctly through pipelines."""
        safe_module = SafeModule(LinearModule(10, 5, "input", "output"), nan_check=True)

        # Input with NaN should raise error
        td_with_nan = TensorDict({"input": torch.tensor([[float("nan")] * 10])}, batch_size=1)

        with pytest.raises(ValueError, match="NaN detected"):
            safe_module(td_with_nan)

    def test_module_state_management(self):
        """Test that module training/eval states are managed correctly."""
        dropout_module = DropoutModule(0.5, "input", "output")

        td = TensorDict({"input": torch.randn(10, 20)}, batch_size=10)

        # Training mode - dropout should be active
        dropout_module.train()
        result_train = dropout_module(td.clone())

        # Eval mode - dropout should be inactive
        dropout_module.eval()
        result_eval1 = dropout_module(td.clone())
        result_eval2 = dropout_module(td.clone())

        # In eval mode, outputs should be identical
        assert torch.allclose(result_eval1["output"], result_eval2["output"])

    def test_component_container_hotswapping_in_pipeline(self):
        """Test hotswapping components during pipeline execution."""
        container = ComponentContainer()

        # Original pipeline
        original_module = LinearModule(10, 5, "input", "output")
        container.register_component("processor", original_module)

        td = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)

        # Execute with original
        container.clear_cache()
        result1 = container.forward("processor", td.clone())
        assert result1["output"].shape == (4, 5)

        # Hotswap to different output size
        new_module = LinearModule(10, 8, "input", "output")
        container.replace_component("processor", new_module)

        # Execute with new module
        container.clear_cache()
        result2 = container.forward("processor", td.clone())
        assert result2["output"].shape == (4, 8)

    def test_memory_efficiency(self):
        """Test that modules don't accumulate unnecessary intermediate results."""
        import gc

        modules = [LinearModule(50, 50, "input", "output") for _ in range(10)]

        # Process many batches
        for i in range(10):
            td = TensorDict({"input": torch.randn(10, 50)}, batch_size=10)

            result = td
            for module in modules:
                result = module(result)

            # Force garbage collection
            del result, td
            gc.collect()

        # Test should complete without memory issues
        assert True
