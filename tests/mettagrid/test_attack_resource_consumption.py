"""Test that attack actions properly consume configured resources."""

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from metta.mettagrid.mettagrid_env import MettaGridEnv


def test_attack_consumes_laser_resource():
    """Test that attacks consume laser resources as configured in mettagrid.yaml."""
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize with the arena/basic config
    with initialize_config_dir(config_dir="../../configs", version_base="1.1"):
        cfg = compose(config_name="env/mettagrid/arena/basic.yaml")

        # Override to ensure laser is in inventory and we have 2 agents
        cfg.game.num_agents = 2
        cfg.game.inventory_item_names = list(cfg.game.inventory_item_names) + ["laser"]

        # Create environment
        env = MettaGridEnv(cfg)
        obs, info = env.reset()

        # Get initial laser count for agent 0
        # The inventory is stored in the agent's state
        initial_laser_count = 5

        # Give agent 0 some lasers using the gridworld interface
        grid = env.grid
        agent = grid._agents[0]
        agent.update_inventory(env.grid._inventory_item_names.index("laser"), initial_laser_count)

        # Verify agent has lasers
        laser_id = env.grid._inventory_item_names.index("laser")
        assert agent.inventory[laser_id] == initial_laser_count

        # Position agent 1 in front of agent 0 for attack
        agent1 = grid._agents[1]
        # Agent 0 is facing up (orientation 0), so put agent 1 one cell up
        agent1.location.x = agent.location.x
        agent1.location.y = agent.location.y - 1

        # Perform attack action (action_id for attack needs to be determined)
        attack_action_id = None
        for i, handler in enumerate(grid._action_handlers):
            if handler.action_name() == "attack":
                attack_action_id = i
                break

        assert attack_action_id is not None, "Attack action not found"

        # Execute attack with arg 0 (target directly in front)
        actions = np.zeros((2, 2), dtype=np.int32)
        actions[0, 0] = attack_action_id  # Attack action
        actions[0, 1] = 0  # Target directly in front

        obs, reward, done, truncated, info = env.step(actions)

        # Check that laser was consumed
        final_laser_count = agent.inventory[laser_id]
        assert final_laser_count == initial_laser_count - 1, \
            f"Attack should consume 1 laser, but laser count went from {initial_laser_count} to {final_laser_count}"

        # Verify the attack was successful (agent 1 should be frozen)
        assert agent1.frozen > 0, "Target agent should be frozen after successful attack"


def test_attack_fails_without_laser():
    """Test that attacks fail when agent doesn't have required laser resource."""
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir="../../configs", version_base="1.1"):
        cfg = compose(config_name="env/mettagrid/arena/basic.yaml")

        # Override to ensure laser is in inventory and we have 2 agents
        cfg.game.num_agents = 2
        cfg.game.inventory_item_names = list(cfg.game.inventory_item_names) + ["laser"]

        # Create environment
        env = MettaGridEnv(cfg)
        obs, info = env.reset()

        # Don't give agent any lasers
        grid = env.grid
        agent = grid._agents[0]
        agent1 = grid._agents[1]

        # Position agent 1 in front of agent 0 for attack
        agent1.location.x = agent.location.x
        agent1.location.y = agent.location.y - 1

        # Get attack action id
        attack_action_id = None
        for i, handler in enumerate(grid._action_handlers):
            if handler.action_name() == "attack":
                attack_action_id = i
                break

        assert attack_action_id is not None, "Attack action not found"

        # Try to execute attack without laser
        actions = np.zeros((2, 2), dtype=np.int32)
        actions[0, 0] = attack_action_id
        actions[0, 1] = 0

        obs, reward, done, truncated, info = env.step(actions)

        # Check that attack failed (agent 1 should not be frozen)
        assert agent1.frozen == 0, "Attack should fail without required laser resource"

        # Check that agent received failure penalty
        assert reward[0] < 0, "Agent should receive penalty for failed action"


if __name__ == "__main__":
    test_attack_consumes_laser_resource()
    test_attack_fails_without_laser()
    print("All tests passed!")
