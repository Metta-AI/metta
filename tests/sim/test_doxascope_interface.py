"""Interface tests for doxascope integration.

These tests ensure the interface between doxascope and the simulation/evaluation
infrastructure remains stable. They protect against breaking changes that would
silently break doxascope data collection.

`test_doxascope_interface` does not require imports from the Doxascope
library; it just tests the rollout/sim/activation extraction methods
that Doxascope uses for data logging.

"""

import uuid
from unittest.mock import MagicMock, patch
import pytest
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulate_and_record import simulate_and_record
from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator, SimulatorEventHandler
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout
from mettagrid.simulator.rollout import Rollout
from metta.agent.components.cortex import CortexTD, CortexTDConfig
from cortex import CortexStackConfig

###############################################
# =========== EVENT HANDLER TESTS =========== #
###############################################

class Test01EventHandlerLifecycleCalled:
    """Test that SimulatorEventHandler on_step is called as expected by Doxascope."""

    def test_event_handler_lifecycle(self):
        """Verify on_step is called the correct number of times."""

        # Create a mock event handler that tracks on_step calls
        class MockEventHandler(SimulatorEventHandler):
            def __init__(self):
                super().__init__()
                self.step_count = 0

            def on_step(self):
                self.step_count += 1

        # Setup environment and rollout
        env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
        env_cfg.game.max_steps = 3  # Run for 3 steps

        # Create mock policy
        mock_policy = MagicMock()
        mock_policy.reset = MagicMock()
        mock_policy.step = MagicMock(return_value=env_cfg.game.actions.noop.Noop())

        handler = MockEventHandler()

        # Use a real Rollout
        rollout = Rollout(env_cfg, [mock_policy], event_handlers=[handler])

        # Run the rollout
        rollout.run_until_done()

        # Assert: on_step was called correct number of times
        assert handler.step_count == 3, f"on_step should be called 3 times, got {handler.step_count}"



class Test02SimulationContextPoliciesInjection:
    """
    Test that Rollout correctly injects policies into simulation._context.
    This is required for the event handler to be able to access the policies
    in order to extract activations.
    """

    def test_policies_injected_into_context(self):
        """Verify that policies are accessible via simulation._context['policies']."""
        # Create a simple environment
        env_cfg = MettaGridConfig.EmptyRoom(num_agents=2)
        env_cfg.game.max_steps = 5

        # Create mock policies
        mock_policy1 = MagicMock()
        mock_policy1.reset = MagicMock()
        mock_policy1.step = MagicMock(return_value=env_cfg.game.actions.noop.Noop())

        mock_policy2 = MagicMock()
        mock_policy2.reset = MagicMock()
        mock_policy2.step = MagicMock(return_value=env_cfg.game.actions.noop.Noop())

        mock_policies = [mock_policy1, mock_policy2]

        # Create a Rollout
        rollout = Rollout(env_cfg, mock_policies)

        # Assert: Check that policies were injected into simulation context
        assert "policies" in rollout._sim._context, "policies key missing from simulation context"
        assert rollout._sim._context["policies"] == mock_policies, "policies list doesn't match"
        assert len(rollout._sim._context["policies"]) == 2, "wrong number of policies in context"

#####################################################
# =========== GRID OBJ EXTRACTION TESTS =========== #
#####################################################


class Test03SimulationGridObjectsInterface:
    """Test that Simulation.grid_objects() returns the expected structure."""
    # Good to go and necessary
    def test_grid_objects_structure(self):
        """Verify grid_objects() returns dict with correct structure."""
        # Setup simulation with agents
        env_cfg = MettaGridConfig.EmptyRoom(num_agents=2)
        simulator = Simulator()
        sim = simulator.new_simulation(env_cfg, seed=42)

        # Call grid_objects
        objects = sim.grid_objects()

        # Assert: Return type is correct
        assert isinstance(objects, dict), "grid_objects() should return dict"
        assert all(isinstance(k, int) for k in objects.keys()), "object IDs should be ints"
        assert all(isinstance(v, dict) for v in objects.values()), "object data should be dicts"

        # Assert: Objects have required keys
        for obj_id, obj_data in objects.items():
            assert "type_name" in obj_data, f"object {obj_id} missing type_name"
            assert "r" in obj_data, f"object {obj_id} missing row coordinate"
            assert "c" in obj_data, f"object {obj_id} missing column coordinate"

    def test_agent_identification(self):
        """Verify agents can be identified by type_name starting with 'agent'."""
        # Setup simulation with agents
        env_cfg = MettaGridConfig.EmptyRoom(num_agents=3)
        simulator = Simulator()
        sim = simulator.new_simulation(env_cfg, seed=42)

        # Get grid objects
        objects = sim.grid_objects()

        # Assert: Can identify agents by type_name
        agent_objects = [obj for obj in objects.values() if obj["type_name"].startswith("agent")]
        assert len(agent_objects) == 3, f"Expected 3 agents, found {len(agent_objects)}"

class Test04SimulationObjectTypeNamesInterface:
    """Test that Simulation.object_type_names property exists and works as expected."""

    def test_object_type_names_exists(self):
        """Verify object_type_names property exists and returns list of strings."""
        # Setup simulation
        env_cfg = MettaGridConfig.EmptyRoom(num_agents=2)
        simulator = Simulator()
        sim = simulator.new_simulation(env_cfg, seed=42)

        # Assert: Property exists and has correct type
        assert hasattr(sim, "object_type_names"), "Simulation missing object_type_names property"
        type_names = sim.object_type_names

        assert isinstance(type_names, list), "object_type_names should be a list"
        assert all(isinstance(name, str) for name in type_names), "all type names should be strings"

    def test_contains_agent_type(self):
        """Verify object_type_names contains agent type."""
        # Setup simulation
        env_cfg = MettaGridConfig.EmptyRoom(num_agents=3)
        simulator = Simulator()
        sim = simulator.new_simulation(env_cfg, seed=42)

        # Get type names
        type_names = sim.object_type_names

        # Assert: Contains agent type name
        agent_types = [name for name in type_names if name.startswith("agent")]
        assert len(agent_types) > 0, "object_type_names should contain at least one agent type"

##########################################################
# =========== CORTEX MEMORY EXTRACTION TESTS =========== #
##########################################################


class Test05CortexActivationExtractionInterface:
    """Test that CortexTD components expose the interface needed for activation extraction."""

    def test_can_find_cortex_in_policy_components(self):
        """Verify we can find CortexTD in a policy's components using Doxascope's pattern."""
        import torch.nn as nn
        from cortex import PassThroughBlockConfig, LSTMCellConfig

        # Create a real CortexTD component
        stack_config = CortexStackConfig(
            d_hidden=64,
            blocks=[
                PassThroughBlockConfig(
                    cell=LSTMCellConfig(hidden_size=64, num_layers=1),
                ),
            ],
        )
        cortex_config = CortexTDConfig(
            in_key="obs",
            out_key="cortex_out",
            d_hidden=64,  # Must match stack_config.d_hidden
            stack_cfg=stack_config,
        )
        cortex_component = CortexTD(cortex_config)

        # Create a mock policy with components dict (like real policies have)
        class MockPolicyWithCortex(nn.Module):
            def __init__(self):
                super().__init__()
                self.components = nn.ModuleDict({
                    "encoder": nn.Linear(10, 64),
                    "cortex": cortex_component,
                    "decoder": nn.Linear(64, 10),
                })

        policy = MockPolicyWithCortex()

        # Test: Can access components dict
        assert hasattr(policy, "components"), "Policy should have components dict"

        # Test: Can find CortexTD by searching through components (Doxascope pattern from doxascope_data.py:461-464)
        found_cortex = None
        for comp in policy.components.values():
            if comp.__class__.__name__ in ("CortexTD", "CortexStack"):
                found_cortex = comp
                break

        # Assert: Found the CortexTD component
        assert found_cortex is not None, "Should find CortexTD in policy components"
        assert found_cortex is cortex_component, "Should find the exact CortexTD instance"
        assert hasattr(found_cortex, "_rollout_current_state"), "Found CortexTD component, but it doesn't have have the required attribute _rollout_current_state."
        assert hasattr(found_cortex, "_rollout_current_env_ids"), "Found CortexTD component, but it doesn't have have the required attribute _rollout_current_env_ids"

    def test_cortex_state_structure_for_extraction(self):
        """Verify that _rollout_current_state has the structure Doxascope expects for extraction."""
        import torch
        from tensordict import TensorDict
        from cortex import PassThroughBlockConfig, LSTMCellConfig

        # Create a CortexTD component
        stack_config = CortexStackConfig(
            d_hidden=64,
            blocks=[
                PassThroughBlockConfig(
                    cell=LSTMCellConfig(hidden_size=64, num_layers=1),
                ),
            ],
        )
        cortex_config = CortexTDConfig(
            in_key="obs",
            out_key="cortex_out",
            d_hidden=64,
            stack_cfg=stack_config,
        )
        cortex = CortexTD(cortex_config)

        # Initially, _rollout_current_state should be None or a valid TensorDict
        assert cortex._rollout_current_state is None or isinstance(
            cortex._rollout_current_state, TensorDict
        ), "_rollout_current_state should be None or TensorDict"

        # Simulate what happens during a forward pass - create a mock state
        # This mimics what Doxascope will encounter after the policy runs
        batch_size = 4
        mock_state = TensorDict(
            {
                "h": torch.randn(batch_size, 64),  # Hidden state
                "c": torch.randn(batch_size, 64),  # Cell state
            },
            batch_size=[batch_size],
        )

        # Simulate setting the state (what would happen during forward pass)
        cortex._rollout_current_state = mock_state

        # Test: Verify the state structure matches Doxascope expectations
        assert isinstance(cortex._rollout_current_state, TensorDict), "rollout state should be TensorDict"

        # Test: Verify we can extract tensors from it (pattern from doxascope_data.py:487-494)
        import optree
        leaves, _ = optree.tree_flatten(cortex._rollout_current_state, namespace="torch")

        # Assert: Should have tensor leaves
        assert len(leaves) > 0, "State should contain tensor leaves for extraction"
        for leaf in leaves:
            assert isinstance(leaf, torch.Tensor), "All leaves should be torch.Tensors"
            assert leaf.dim() >= 1, "Tensors should have at least a batch dimension"
            assert leaf.shape[0] == batch_size, f"Batch dimension should be {batch_size}"

