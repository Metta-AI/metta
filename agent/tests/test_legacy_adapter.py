"""Test backwards compatibility with old checkpoint formats."""

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import ModuleDict

from metta.agent.component_policy import ComponentPolicy
from metta.agent.metta_agent import MettaAgent


def create_mock_old_metta_agent():
    """Create a mock old-style MettaAgent with components."""

    class MockComponent(nn.Module):
        def __init__(self, name):
            super().__init__()
            self._name = name
            self.ready = True

        def forward(self, td):
            # Simulate component output
            batch_size = td.batch_size.numel()
            if self._name == "_value_":
                td[self._name] = torch.randn(batch_size, 1)
            elif self._name == "_action_":
                # Simulate action logits for 6 actions
                td[self._name] = torch.randn(batch_size, 6)

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
            # For 2 action types with 3 params each (param values 0, 1, 2)
            # The formula is: logit_index = action_type + cum_action_max_params[action_type] + action_param
            # We want actions [0,0],[0,1],[0,2] to map to indices 0,1,2
            # and actions [1,0],[1,1],[1,2] to map to indices 3,4,5
            # So cum_action_max_params[0] = 0 and cum_action_max_params[1] = 2
            # This gives us the correct mapping
            self.cum_action_max_params = torch.tensor([0, 2])
            self.action_index_tensor = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
            self.cfg = {"clip_range": 0.1}
            self.agent_attributes = {}

    return OldMettaAgent()


def test_old_checkpoint_loading():
    """Test that old checkpoints are properly converted to new structure."""
    # Create an old-style agent
    old_agent = create_mock_old_metta_agent()

    # Get the state dict to simulate what would be in a checkpoint
    old_agent_dict = old_agent.__dict__.copy()

    # Create new MettaAgent and restore old state
    class TempMettaAgent(MettaAgent):
        def __init__(self):
            # Skip normal init for testing
            nn.Module.__init__(self)

    new_agent = TempMettaAgent()
    new_agent.__setstate__(old_agent_dict)

    # Verify the policy was created and is a ComponentPolicy
    assert hasattr(new_agent, "policy"), "MettaAgent should have policy attribute"
    assert isinstance(new_agent.policy, ComponentPolicy), (
        f"Policy should be ComponentPolicy, got {type(new_agent.policy)}"
    )

    # Verify components were transferred to policy
    assert hasattr(new_agent.policy, "components"), "Policy should have components"
    assert "_value_" in new_agent.policy.components, "Policy should have _value_ component"
    assert "_action_" in new_agent.policy.components, "Policy should have _action_ component"

    # Verify other attributes were transferred correctly
    assert new_agent.policy.clip_range == 0.1, f"clip_range should be 0.1, got {new_agent.policy.clip_range}"
    assert hasattr(new_agent.policy, "cum_action_max_params"), "Policy should have cum_action_max_params"
    assert hasattr(new_agent.policy, "action_index_tensor"), "Policy should have action_index_tensor"

    # Test forward pass with inference
    batch_size = 4
    td = TensorDict({"env_obs": torch.randn(batch_size, 200, 3)}, batch_size=[batch_size])

    # Run components to generate _value_ and _action_
    new_agent.policy.components["_value_"](td)
    new_agent.policy.components["_action_"](td)

    # Now run forward_inference
    result = new_agent.policy.forward_inference(td)
    assert "actions" in result, "Forward should return actions"
    assert "values" in result, "Forward should return values"

    # Test training forward with action evaluation
    td_train = TensorDict({"env_obs": torch.randn(2, 3, 200, 3)}, batch_size=[2, 3])

    # Flatten for components
    flat_td = td_train.reshape(6)
    new_agent.policy.components["_value_"](flat_td)
    new_agent.policy.components["_action_"](flat_td)

    # Test with valid actions - ensure they map to valid logit indices (0-5)
    # With cum_action_max_params = [0, 3], action [0,i] maps to 0+0+i = i, action [1,i] maps to 1+3+i = 4+i
    # But wait, that doesn't work. Let me think about this...
    # Actually, the formula is: action_type + cum_action_max_params[action_type] + action_param
    # So action [0,i] maps to 0 + 0 + i = i (indices 0,1,2)
    # And action [1,i] maps to 1 + 3 + i = 4+i? No that's wrong too.
    # The issue is cum_action_max_params needs to be the right cumulative sum
    action = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=torch.long).reshape(2, 3, 2)
    result = new_agent.policy.forward_training(flat_td, action)

    # Check for training outputs
    assert "act_log_prob" in result, "Training should return act_log_prob"
    assert "value" in result, "Training should return value"
    assert "entropy" in result, "Training should return entropy"


def test_no_circular_references():
    """Verify no circular references exist after conversion."""

    # Create a mock agent with old checkpoint format
    old_agent = create_mock_old_metta_agent()
    old_agent_dict = old_agent.__dict__.copy()

    # Simulate circular reference that might exist in old checkpoints
    old_agent_dict["policy"] = old_agent_dict  # Circular reference

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
    assert agent.policy is not agent, "No circular reference for DDP"


def test_new_checkpoint_format():
    """Test that new checkpoints (with separate policy) still work."""

    # Create a new checkpoint format with policy already separated
    class TempMettaAgent(MettaAgent):
        def __init__(self):
            nn.Module.__init__(self)

    agent = TempMettaAgent()

    # Create a mock ComponentPolicy
    policy = ComponentPolicy.__new__(ComponentPolicy)
    nn.Module.__init__(policy)
    policy.components = ModuleDict()
    policy.components_with_memory = []

    # Create state with new format (has policy, not components)
    new_state = {
        "policy": policy,
        "device": "cpu",
        "_total_params": 1000,
    }

    # Restore state
    agent.__setstate__(new_state)

    # Verify it was restored correctly
    assert agent.policy is policy, "Policy should be the same object"
    assert agent.device == "cpu", "Device should be restored"
    assert agent._total_params == 1000, "Total params should be restored"
