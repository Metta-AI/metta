import json
import os
import pickle
import tempfile

import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.component_container import ComponentContainer
from metta.agent.lib.metta_modules import (
    Conv2dModule,
    LinearModule,
    MettaModule,
    ReLUModule,
)
from metta.agent.lib.wrapper_modules import RegularizedModule, SafeModule, WeightMonitoringModule


class TestSerialization:
    """Comprehensive tests for model serialization, saving, and loading."""

    def test_linear_module_state_dict(self):
        """Test saving and loading LinearModule state dict."""
        module = LinearModule(10, 5)

        # Generate some test data and run forward pass
        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
        original_output = module(td)["output"]

        # Save state dict
        state_dict = module.state_dict()

        # Create new module and load state dict
        new_module = LinearModule(10, 5)
        new_module.load_state_dict(state_dict)

        # Test that outputs are identical
        new_output = new_module(td)["output"]
        assert torch.allclose(original_output, new_output)

    def test_complex_module_state_dict(self):
        """Test state dict for modules with multiple components."""
        module = Conv2dModule(3, 16, 3, padding=1)

        td = TensorDict({"input": torch.randn(1, 3, 32, 32)}, batch_size=1)
        original_output = module(td)["output"]

        # Save and reload
        state_dict = module.state_dict()
        new_module = Conv2dModule(3, 16, 3, padding=1)
        new_module.load_state_dict(state_dict)

        new_output = new_module(td)["output"]
        assert torch.allclose(original_output, new_output)

    def test_wrapper_module_serialization(self):
        """Test serialization of wrapped modules."""
        base_module = LinearModule(8, 4)
        safe_module = SafeModule(base_module, nan_check=True)

        td = TensorDict({"input": torch.randn(3, 8)}, batch_size=3)
        original_output = safe_module(td)["output"]

        # Test state dict
        state_dict = safe_module.state_dict()

        # Recreate and load
        new_base = LinearModule(8, 4)
        new_safe = SafeModule(new_base, nan_check=True)
        new_safe.load_state_dict(state_dict)

        new_output = new_safe(td)["output"]
        assert torch.allclose(original_output, new_output)

    def test_component_container_serialization(self):
        """Test serialization of ComponentContainer with dependencies."""
        container = ComponentContainer()

        # Build a simple pipeline
        module1 = LinearModule(5, 3, "input", "hidden")
        module2 = ReLUModule("hidden", "activated")
        module3 = LinearModule(3, 2, "activated", "output")

        container.register_component("linear1", module1)
        container.register_component("relu", module2, dependencies=["linear1"])
        container.register_component("linear2", module3, dependencies=["relu"])

        td = TensorDict({"input": torch.randn(2, 5)}, batch_size=2)
        container.clear_cache()
        original_result = container.forward("linear2", td.clone())

        # Save state dict
        state_dict = container.state_dict()

        # Recreate container with same structure
        new_container = ComponentContainer()
        new_module1 = LinearModule(5, 3, "input", "hidden")
        new_module2 = ReLUModule("hidden", "activated")
        new_module3 = LinearModule(3, 2, "activated", "output")

        new_container.register_component("linear1", new_module1)
        new_container.register_component("relu", new_module2, dependencies=["linear1"])
        new_container.register_component("linear2", new_module3, dependencies=["relu"])

        # Load state dict
        new_container.load_state_dict(state_dict)

        # Test that outputs are identical
        new_container.clear_cache()
        new_result = new_container.forward("linear2", td.clone())

        assert torch.allclose(original_result["output"], new_result["output"])

    def test_torch_save_load(self):
        """Test saving and loading entire modules with torch.save/load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "module.pt")

            # Create and test module
            module = LinearModule(6, 3)
            td = TensorDict({"input": torch.randn(2, 6)}, batch_size=2)
            original_output = module(td)["output"]

            # Save entire module
            torch.save(module, save_path)

            # Load module
            loaded_module = torch.load(save_path)

            # Test functionality
            loaded_output = loaded_module(td)["output"]
            assert torch.allclose(original_output, loaded_output)

    def test_partial_state_dict_loading(self):
        """Test loading partial state dicts (for transfer learning scenarios)."""
        # Create larger module
        large_module = LinearModule(20, 10)

        # Create smaller module with compatible first layer
        small_module = LinearModule(20, 5)

        # Get state dict from large module
        large_state = large_module.state_dict()

        # Load compatible parts into small module
        small_state = small_module.state_dict()

        # Only load weight and bias that are compatible
        # (Note: This test shows the concept - real transfer learning would be more sophisticated)
        if large_state["linear.weight"].shape[1] == small_state["linear.weight"].shape[1]:
            # Can transfer input weights (but not all output weights)
            small_state["linear.weight"][:5, :] = large_state["linear.weight"][:5, :]
            small_state["linear.bias"][:5] = large_state["linear.bias"][:5]

            small_module.load_state_dict(small_state)

        # Test that module still works
        td = TensorDict({"input": torch.randn(1, 20)}, batch_size=1)
        result = small_module(td)
        assert result["output"].shape == (1, 5)

    def test_module_pickling(self):
        """Test pickling and unpickling modules."""
        module = LinearModule(4, 2)

        td = TensorDict({"input": torch.randn(3, 4)}, batch_size=3)
        original_output = module(td)["output"]

        # Pickle and unpickle
        pickled_data = pickle.dumps(module)
        unpickled_module = pickle.loads(pickled_data)

        # Test functionality
        unpickled_output = unpickled_module(td)["output"]
        assert torch.allclose(original_output, unpickled_output)

    def test_component_container_pickling(self):
        """Test pickling ComponentContainer with complex structure."""
        container = ComponentContainer()

        # Create components
        linear = LinearModule(3, 2, "input", "output")
        safe_linear = SafeModule(linear, nan_check=True)

        container.register_component("safe_processor", safe_linear)

        td = TensorDict({"input": torch.randn(2, 3)}, batch_size=2)
        container.clear_cache()
        original_result = container.forward("safe_processor", td.clone())

        # Pickle and unpickle
        pickled_container = pickle.dumps(container)
        unpickled_container = pickle.loads(pickled_container)

        # Test functionality
        unpickled_container.clear_cache()
        unpickled_result = unpickled_container.forward("safe_processor", td.clone())

        assert torch.allclose(original_result["output"], unpickled_result["output"])

    def test_cross_device_serialization(self):
        """Test serialization across different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for cross-device test")

        # Create module on CPU
        cpu_module = LinearModule(5, 3)
        td = TensorDict({"input": torch.randn(2, 5)}, batch_size=2)
        cpu_output = cpu_module(td)["output"]

        # Move to GPU
        gpu_module = cpu_module.cuda()
        gpu_td = td.cuda()
        gpu_output = gpu_module(gpu_td)["output"]

        # Save GPU module state
        gpu_state = gpu_module.state_dict()

        # Load into new CPU module
        new_cpu_module = LinearModule(5, 3)
        # Move state dict back to CPU
        cpu_state = {k: v.cpu() for k, v in gpu_state.items()}
        new_cpu_module.load_state_dict(cpu_state)

        # Test outputs match (accounting for device transfer)
        new_cpu_output = new_cpu_module(td)["output"]
        assert torch.allclose(cpu_output, new_cpu_output)

    def test_version_compatibility_simulation(self):
        """Simulate version compatibility by testing flexible loading."""
        # Create module with current structure
        module = LinearModule(4, 3)
        state_dict = module.state_dict()

        # Simulate loading into slightly different module structure
        # (This is a simplified version compatibility test)

        # Add extra metadata that should be ignored
        modified_state = state_dict.copy()
        modified_state["extra_metadata"] = torch.tensor([1, 2, 3])

        # Load only compatible parts
        try:
            new_module = LinearModule(4, 3)
            # Filter out non-compatible keys
            compatible_state = {k: v for k, v in modified_state.items() if k in new_module.state_dict()}
            new_module.load_state_dict(compatible_state)

            # Test functionality
            td = TensorDict({"input": torch.randn(1, 4)}, batch_size=1)
            result = new_module(td)
            assert result["output"].shape == (1, 3)

        except Exception as e:
            pytest.fail(f"Version compatibility simulation failed: {e}")

    def test_checkpoint_saving_loading(self):
        """Test comprehensive checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")

            # Create training state
            module = LinearModule(10, 5)
            optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
            epoch = 42
            loss = 0.123

            # Simulate training step
            td = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
            output = module(td)
            target = torch.randn_like(output["output"])
            loss_tensor = torch.nn.functional.mse_loss(output["output"], target)

            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()

            # Save comprehensive checkpoint
            checkpoint = {
                "model_state_dict": module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "model_config": {"in_features": 10, "out_features": 5, "in_key": "input", "out_key": "output"},
            }

            torch.save(checkpoint, checkpoint_path)

            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path)

            # Reconstruct training state
            new_module = LinearModule(
                loaded_checkpoint["model_config"]["in_features"],
                loaded_checkpoint["model_config"]["out_features"],
                loaded_checkpoint["model_config"]["in_key"],
                loaded_checkpoint["model_config"]["out_key"],
            )

            new_module.load_state_dict(loaded_checkpoint["model_state_dict"])

            new_optimizer = torch.optim.Adam(new_module.parameters(), lr=0.001)
            new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

            # Verify reconstruction
            assert loaded_checkpoint["epoch"] == epoch
            assert loaded_checkpoint["loss"] == loss

            # Test that continued training works
            new_output = new_module(td)
            new_loss = torch.nn.functional.mse_loss(new_output["output"], target)
            new_optimizer.zero_grad()
            new_loss.backward()
            new_optimizer.step()

    def test_metadata_preservation(self):
        """Test that module metadata is preserved during serialization."""
        module = LinearModule(7, 4, "custom_input", "custom_output")

        # Verify initial metadata
        assert module.in_keys == ["custom_input"]
        assert module.out_keys == ["custom_output"]

        # Save and load
        state_dict = module.state_dict()
        new_module = LinearModule(7, 4, "custom_input", "custom_output")
        new_module.load_state_dict(state_dict)

        # Verify metadata preservation
        assert new_module.in_keys == ["custom_input"]
        assert new_module.out_keys == ["custom_output"]

        # Test functionality with custom keys
        td = TensorDict({"custom_input": torch.randn(2, 7)}, batch_size=2)
        result = new_module(td)
        assert "custom_output" in result
        assert result["custom_output"].shape == (2, 4)

    def test_nested_module_serialization(self):
        """Test serialization of deeply nested module structures."""
        # Create nested wrapper structure
        base = LinearModule(6, 4)
        safe = SafeModule(base, nan_check=True)
        regularized = RegularizedModule(safe, l2_scale=0.01)
        monitored = WeightMonitoringModule(regularized, monitor_health=True)

        td = TensorDict({"input": torch.randn(2, 6)}, batch_size=2)
        original_output = monitored(td)["output"]

        # Save state dict
        state_dict = monitored.state_dict()

        # Reconstruct nested structure
        new_base = LinearModule(6, 4)
        new_safe = SafeModule(new_base, nan_check=True)
        new_regularized = RegularizedModule(new_safe, l2_scale=0.01)
        new_monitored = WeightMonitoringModule(new_regularized, monitor_health=True)

        # Load state dict
        new_monitored.load_state_dict(state_dict)

        # Test functionality
        new_output = new_monitored(td)["output"]
        assert torch.allclose(original_output, new_output)

    def test_serialization_with_custom_tensors(self):
        """Test serialization when modules contain custom tensor attributes."""

        class CustomModule(MettaModule):
            def __init__(self):
                super().__init__(in_keys=["input"], out_keys=["output"])
                self.linear = torch.nn.Linear(5, 3)
                # Custom tensor that should be serialized
                self.register_buffer("custom_tensor", torch.randn(3))

            def forward(self, tensordict):
                out = self.linear(tensordict["input"])
                # Use custom tensor in computation - access as getattr to fix type checking
                custom_val = self.custom_tensor
                out = out + custom_val.sum()
                tensordict["output"] = out
                return tensordict

        module = CustomModule()
        td = TensorDict({"input": torch.randn(2, 5)}, batch_size=2)
        original_output = module(td)["output"]

        # Save and load
        state_dict = module.state_dict()
        new_module = CustomModule()
        new_module.load_state_dict(state_dict)

        # Test that custom tensor was preserved - use state dict comparison
        original_tensor = module.custom_tensor
        new_tensor = new_module.custom_tensor
        assert torch.allclose(original_tensor, new_tensor)

        # Test functionality
        new_output = new_module(td)["output"]
        assert torch.allclose(original_output, new_output)

    def test_serialization_error_handling(self):
        """Test graceful handling of serialization errors."""
        module = LinearModule(3, 2)

        # Test loading incompatible state dict
        incompatible_state = {
            "linear.weight": torch.randn(5, 3),  # Wrong shape
            "linear.bias": torch.randn(5),
        }

        with pytest.raises(RuntimeError):
            module.load_state_dict(incompatible_state)

        # Test partial loading with strict=False
        try:
            module.load_state_dict(incompatible_state, strict=False)
            # Should not raise error, but weights won't be updated
        except Exception as e:
            pytest.fail(f"Partial loading should not raise error: {e}")

    def test_large_module_serialization(self):
        """Test serialization of large modules."""
        # Create a reasonably large module
        large_module = LinearModule(5000, 1000)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "large_module.pt")

            # Test that large module can be saved and loaded
            torch.save(large_module, save_path)
            loaded_module = torch.load(save_path)

            # Verify functionality
            td = TensorDict({"input": torch.randn(2, 5000)}, batch_size=2)
            original_output = large_module(td)["output"]
            loaded_output = loaded_module(td)["output"]

            assert torch.allclose(original_output, loaded_output)

    def test_config_based_reconstruction(self):
        """Test reconstruction of modules from configuration."""
        # Configuration for module creation
        config = {
            "module_type": "LinearModule",
            "in_features": 8,
            "out_features": 4,
            "in_key": "data",
            "out_key": "result",
        }

        # Create module from config
        original_module = LinearModule(
            config["in_features"], config["out_features"], config["in_key"], config["out_key"]
        )

        # Save state and config
        state_dict = original_module.state_dict()

        # Simulate saving config as JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            state_path = os.path.join(temp_dir, "state.pt")

            with open(config_path, "w") as f:
                json.dump(config, f)

            torch.save(state_dict, state_path)

            # Load config and reconstruct module
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            loaded_state = torch.load(state_path)

            # Reconstruct module
            reconstructed_module = LinearModule(
                loaded_config["in_features"],
                loaded_config["out_features"],
                loaded_config["in_key"],
                loaded_config["out_key"],
            )

            reconstructed_module.load_state_dict(loaded_state)

            # Test functionality
            td = TensorDict({config["in_key"]: torch.randn(2, config["in_features"])}, batch_size=2)
            original_output = original_module(td)[config["out_key"]]
            reconstructed_output = reconstructed_module(td)[config["out_key"]]

            assert torch.allclose(original_output, reconstructed_output)
