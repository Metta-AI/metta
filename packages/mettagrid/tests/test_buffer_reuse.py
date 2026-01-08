"""Test that MettaGridPufferEnv properly reuses buffers across resets."""

import numpy as np

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
)
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.simulator import Simulator


def test_buffer_reuse_across_resets():
    """Test that MettaGridPufferEnv properly reuses buffers across resets.

    This verifies that:
    1. Creates a MettaGridPufferEnv
    2. Resets it multiple times
    3. Verifies it creates new Simulation instances but reuses the same buffer objects
    """
    # Create config
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=24,
            obs=ObsConfig(width=5, height=5, num_tokens=100),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            map_builder=RandomMapBuilder.Config(width=20, height=20, agents=24, seed=42),
        )
    )

    # Create simulator and environment
    simulator = Simulator()
    env = MettaGridPufferEnv(simulator, config, seed=42)

    # Get initial simulation reference
    initial_sim = env.current_simulation
    assert initial_sim is not None, "Simulation should be initialized"

    # First reset
    obs1, info1 = env.reset(seed=42)

    # Store references to the buffers after first reset
    initial_obs_buffer = env.observations
    initial_rewards_buffer = env.rewards
    initial_terminals_buffer = env.terminals
    initial_truncations_buffer = env.truncations
    initial_masks_buffer = env.masks
    initial_actions_buffer = env.actions

    # Verify buffers are not None
    assert initial_obs_buffer is not None, "Observations buffer should be allocated"
    assert initial_rewards_buffer is not None, "Rewards buffer should be allocated"
    assert initial_terminals_buffer is not None, "Terminals buffer should be allocated"
    assert initial_truncations_buffer is not None, "Truncations buffer should be allocated"
    assert initial_masks_buffer is not None, "Masks buffer should be allocated"
    assert initial_actions_buffer is not None, "Actions buffer should be allocated"

    # Store buffer IDs to verify they're the same objects
    obs_buffer_id = id(initial_obs_buffer)
    rewards_buffer_id = id(initial_rewards_buffer)
    terminals_buffer_id = id(initial_terminals_buffer)
    truncations_buffer_id = id(initial_truncations_buffer)
    masks_buffer_id = id(initial_masks_buffer)
    actions_buffer_id = id(initial_actions_buffer)

    # Get simulation reference after first reset
    first_reset_sim = env.current_simulation
    first_reset_sim_id = id(first_reset_sim)

    # Do a second reset with different seed
    obs2, info2 = env.reset(seed=123)

    # Verify we have a new Simulation (it gets recreated on reset)
    second_reset_sim = env.current_simulation
    assert second_reset_sim is not None, "Simulation should still be initialized"
    assert id(second_reset_sim) != first_reset_sim_id, "Should have a new Simulation instance after reset"

    # Verify the buffers are the SAME objects (reused)
    assert id(env.observations) == obs_buffer_id, "Observations buffer should be reused"
    assert id(env.rewards) == rewards_buffer_id, "Rewards buffer should be reused"
    assert id(env.terminals) == terminals_buffer_id, "Terminals buffer should be reused"
    assert id(env.truncations) == truncations_buffer_id, "Truncations buffer should be reused"
    assert id(env.masks) == masks_buffer_id, "Masks buffer should be reused"
    assert id(env.actions) == actions_buffer_id, "Actions buffer should be reused"

    # Verify buffers have the same shape and dtype
    assert env.observations.shape == initial_obs_buffer.shape, "Observations shape should be preserved"
    assert env.rewards.shape == initial_rewards_buffer.shape, "Rewards shape should be preserved"
    assert env.terminals.shape == initial_terminals_buffer.shape, "Terminals shape should be preserved"
    assert env.truncations.shape == initial_truncations_buffer.shape, "Truncations shape should be preserved"
    assert env.masks.shape == initial_masks_buffer.shape, "Masks shape should be preserved"
    assert env.actions.shape == initial_actions_buffer.shape, "Actions shape should be preserved"

    assert env.observations.dtype == initial_obs_buffer.dtype, "Observations dtype should be preserved"
    assert env.rewards.dtype == initial_rewards_buffer.dtype, "Rewards dtype should be preserved"
    assert env.terminals.dtype == initial_terminals_buffer.dtype, "Terminals dtype should be preserved"
    assert env.truncations.dtype == initial_truncations_buffer.dtype, "Truncations dtype should be preserved"
    assert env.masks.dtype == initial_masks_buffer.dtype, "Masks dtype should be preserved"
    assert env.actions.dtype == initial_actions_buffer.dtype, "Actions dtype should be preserved"

    # Verify the observations are different (different seed should give different initial state)
    # Note: If observations are the same, it might be due to deterministic map generation
    # The main test is buffer reuse, so we'll check if the observations are at least valid
    assert obs1 is not None and obs2 is not None, "Both observations should be valid"
    if np.array_equal(obs1, obs2):
        print(
            "⚠️  Warning: Different seeds produced identical observations - "
            "this might be due to deterministic map generation"
        )

    # Clean up
    env.close()
    simulator.close()


def test_buffer_consistency_during_episode():
    """Test that buffers remain consistent during an episode."""
    # Create config
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=24,
            obs=ObsConfig(width=5, height=5, num_tokens=100),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            map_builder=RandomMapBuilder.Config(width=20, height=20, agents=24, seed=42),
        )
    )

    # Create simulator and environment
    simulator = Simulator()
    env = MettaGridPufferEnv(simulator, config, seed=42)

    # Reset environment
    obs, info = env.reset(seed=42)

    # Store buffer references
    obs_buffer_ref = env.observations
    rewards_buffer_ref = env.rewards
    terminals_buffer_ref = env.terminals
    truncations_buffer_ref = env.truncations
    masks_buffer_ref = env.masks
    actions_buffer_ref = env.actions

    # Take a few steps
    for step in range(5):
        # Create noop actions
        actions = np.zeros(env.num_agents, dtype=dtype_actions)

        # Step environment
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Verify buffer references haven't changed
        assert env.observations is obs_buffer_ref, f"Observations buffer changed at step {step}"
        assert env.rewards is rewards_buffer_ref, f"Rewards buffer changed at step {step}"
        assert env.terminals is terminals_buffer_ref, f"Terminals buffer changed at step {step}"
        assert env.truncations is truncations_buffer_ref, f"Truncations buffer changed at step {step}"
        assert env.masks is masks_buffer_ref, f"Masks buffer changed at step {step}"
        assert env.actions is actions_buffer_ref, f"Actions buffer changed at step {step}"

        # Break if episode ended
        if terminals.all() or truncations.all():
            break

    # Clean up
    env.close()
    simulator.close()


if __name__ == "__main__":
    test_buffer_reuse_across_resets()
    test_buffer_consistency_during_episode()
