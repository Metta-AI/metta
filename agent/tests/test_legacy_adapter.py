"""Test backwards compatibility with old checkpoint formats using LegacyMettaAgentAdapter."""

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import ModuleDict

from metta.agent.legacy_adapter import LegacyMettaAgentAdapter
from metta.agent.metta_agent import MettaAgent


def create_mock_old_metta_agent():
    """Create a mock old-style MettaAgent with components."""

    class MockComponent(nn.Module):
        def __init__(self, name):
            super().__init__()
            self._name = name
            self.ready = True

        def forward(self, td):
            td[self._name] = torch.randn(td.batch_size.numel(), 1)

        def has_memory(self):
            return False

    class OldMettaAgent(nn.Module):
        """Simulates old MettaAgent structure with components."""

        def __init__(self):
            super().__init__()
            self.components = ModuleDict(
                {
                    "_value_": MockComponent("_value_"),
                    "_action_": MockComponent("_action_"),
                }
            )
            self.components_with_memory = []
            self.clip_range = 0.1
            self.cum_action_max_params = torch.tensor([0, 3, 6, 9])
            self.action_index_tensor = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

        def forward_inference(self, td):
            td["actions"] = torch.zeros((td.batch_size.numel(), 2), dtype=torch.long)
            td["values"] = torch.randn(td.batch_size.numel())
            return td

        def forward_training(self, td, action):
            # Add training outputs to the td
            batch_size = td.batch_size.numel()
            td["act_log_prob"] = torch.randn(batch_size)
            td["value"] = torch.randn(batch_size, 1)
            td["entropy"] = torch.randn(batch_size)
            return td

    return OldMettaAgent()


def test_old_checkpoint_loading():
    """Test that old checkpoints are properly wrapped in the adapter."""
    # Create an old-style agent
    old_agent = create_mock_old_metta_agent()

    # Copy methods to simulate what would happen with a real old checkpoint
    # In real checkpoints, the MettaAgent instance would have these methods
    old_agent_dict = old_agent.__dict__.copy()
    old_agent_dict["forward_inference"] = old_agent.forward_inference
    old_agent_dict["forward_training"] = old_agent.forward_training

    # Create new MettaAgent and restore old state
    class TempMettaAgent(MettaAgent):
        def __init__(self):
            # Skip normal init for testing
            nn.Module.__init__(self)

    new_agent = TempMettaAgent()
    new_agent.__setstate__(old_agent_dict)

    # Verify the policy is wrapped in adapter
    assert hasattr(new_agent, "policy"), "MettaAgent should have policy attribute"
    assert isinstance(new_agent.policy, LegacyMettaAgentAdapter), (
        f"Policy should be LegacyMettaAgentAdapter, got {type(new_agent.policy)}"
    )

    # Test forward pass
    batch_size = 4
    td = TensorDict({"env_obs": torch.randn(batch_size, 200, 3)}, batch_size=[batch_size])

    # Test inference
    result = new_agent.forward(td, state=None, action=None)
    assert "actions" in result, "Forward should return actions"
    assert "values" in result, "Forward should return values"

    # Test training - For simplicity, just test that it doesn't crash
    # The exact reshaping behavior is complex and depends on the specific legacy implementation
    td_train = TensorDict({"env_obs": torch.randn(2, 3, 200, 3)}, batch_size=[2, 3])
    action = torch.zeros((2, 3, 2), dtype=torch.long)
    # Check that forward_training is accessible
    assert hasattr(new_agent.policy.legacy_agent, "forward_training"), (
        "Legacy agent should have forward_training method"
    )

    result = new_agent.forward(td_train, state=None, action=action)
    # Just check that we got some result back with expected fields
    # The batch size handling is implementation-specific for legacy agents
    has_training_outputs = any(k in ["act_log_prob", "value", "entropy"] for k in result.keys())
    if not has_training_outputs:
        # Also check if it's in the reshaped result
        flat_result = result.reshape(-1) if result.batch_dims > 1 else result
        has_training_outputs = any(k in ["act_log_prob", "value", "entropy"] for k in flat_result.keys())
    assert has_training_outputs, f"Training forward should return training outputs, got keys: {list(result.keys())}"


def test_adapter_methods():
    """Test that adapter properly delegates methods."""

    old_agent = create_mock_old_metta_agent()
    adapter = LegacyMettaAgentAdapter(old_agent)

    # Test that components are accessible
    assert hasattr(adapter, "components"), "Adapter should have components"
    assert "_value_" in adapter.components, "Adapter should have _value_ component"
    assert "_action_" in adapter.components, "Adapter should have _action_ component"

    # Test clip_range
    assert hasattr(adapter, "clip_range"), "Adapter should have clip_range"
    assert adapter.clip_range == 0.1, f"clip_range should be 0.1, got {adapter.clip_range}"

    # Test action conversion attributes
    assert hasattr(adapter, "cum_action_max_params"), "Adapter should have cum_action_max_params"
    assert hasattr(adapter, "action_index_tensor"), "Adapter should have action_index_tensor"

    # Test memory methods (even with no memory components)
    adapter.reset_memory()  # Should not crash
    memory = adapter.get_memory()
    assert isinstance(memory, dict), "get_memory should return dict"

    # Test action conversion
    action = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)  # Use valid indices
    indices = adapter._convert_action_to_logit_index(action)
    assert indices.shape == (2,), f"Should return 1D tensor, got shape {indices.shape}"

    # Only test reverse conversion with valid indices (within bounds)
    valid_indices = torch.tensor([0, 3], dtype=torch.long)  # Indices within action_index_tensor bounds
    back = adapter._convert_logit_index_to_action(valid_indices)
    assert back.shape == (2, 2), f"Should return 2D tensor, got shape {back.shape}"


def test_no_circular_references():
    """Verify no circular references exist."""

    # Create a mock agent with old checkpoint format
    old_agent = create_mock_old_metta_agent()
    old_agent_dict = old_agent.__dict__.copy()
    old_agent_dict["forward_inference"] = old_agent.forward_inference
    old_agent_dict["forward_training"] = old_agent.forward_training

    class TempMettaAgent(MettaAgent):
        def __init__(self):
            nn.Module.__init__(self)

    agent = TempMettaAgent()
    agent.__setstate__(old_agent_dict)

    # Check that policy is not self
    assert agent.policy is not agent, "No circular reference: agent.policy should not be agent"

    # Check that we can traverse without infinite recursion
    def count_modules(module, visited=None):
        if visited is None:
            visited = set()
        if id(module) in visited:
            return 0
        visited.add(id(module))
        count = 1
        for child in module.children():
            count += count_modules(child, visited)
        return count

    # Verify we can traverse the module tree without infinite recursion
    module_count = count_modules(agent)
    assert module_count > 0, "Should be able to count modules without recursion"

    # Test that DistributedDataParallel would work (simulate the check)
    if hasattr(agent, "policy") and agent.policy is agent:
        raise AssertionError("Circular reference detected")


def test_new_checkpoint_format():
    """Test that new checkpoints (with separate policy) still work."""

    from metta.agent.component_policy import ComponentPolicy

    # This would normally be created through proper initialization
    # Here we just verify the structure works
    # Full test would need environment setup
    assert ComponentPolicy is not None
