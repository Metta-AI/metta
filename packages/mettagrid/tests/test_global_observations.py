"""Test global observation features in MettaGrid (non-reward related)."""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    GameConfig,
    GlobalObsConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.simulator import Action, Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


class TestGlobalObservations:
    """Test global observation features in MettaGrid (non-reward related).

    This file tests global observation features such as:
    - episode_completion_pct
    - last_action
    - last_reward

    Reward-related global observations are tested in test_global_reward_observations.py
    """

    def test_last_action_is_noop_when_action_fails(self):
        """Test that agents see noop (index 0) as last_action when their action fails."""
        # Create config with last_action observation enabled and move action requiring resources
        game_config = GameConfig(
            num_agents=1,
            obs=ObsConfig(width=7, height=7, num_tokens=100),
            max_steps=100,
            resource_names=["ore"],
            global_obs=GlobalObsConfig(last_action=True),
            agent=AgentConfig(inventory=InventoryConfig(default_limit=10), freeze_duration=0, rewards=AgentRewards()),
            actions=ActionsConfig(
                noop=NoopActionConfig(enabled=True),
                move=MoveActionConfig(enabled=True, required_resources={"ore": 1}),
            ),
            objects={"wall": WallConfig()},
        )

        game_map = [
            ["wall", "wall", "wall"],
            ["wall", "agent.agent", "wall"],
            ["wall", "wall", "wall"],
        ]

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        sim = Simulation(cfg, seed=42)

        # First step with noop to initialize observations
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        # Verify noop action index is 0
        assert sim.action_names[0] == "noop", "noop should be at index 0"

        # Now try a move action that will fail (no resources)
        move_action_name = next((name for name in sim.action_names if name.startswith("move")), None)
        assert move_action_name is not None, "Expected move action in action names"

        sim.agent(0).set_action(Action(name=move_action_name))
        sim.step()

        # Verify the action failed
        assert not sim.agent(0).last_action_success, "Move should fail without required resources"

        # Check that last_action observation is noop (index 0)
        agent = sim.agent(0)
        global_obs_data = agent.global_observations
        assert "last_action" in global_obs_data, "Should have last_action observation"
        assert global_obs_data["last_action"] == 0, (
            f"Expected last_action to be 0 (noop) when action fails, got {global_obs_data['last_action']}"
        )

    def test_last_action_is_actual_action_when_action_succeeds(self):
        """Test that agents see the actual action index as last_action when their action succeeds."""
        # Create config with last_action observation enabled
        game_config = GameConfig(
            num_agents=1,
            obs=ObsConfig(width=7, height=7, num_tokens=100),
            max_steps=100,
            resource_names=["ore"],
            global_obs=GlobalObsConfig(last_action=True),
            agent=AgentConfig(inventory=InventoryConfig(default_limit=10), freeze_duration=0, rewards=AgentRewards()),
            actions=ActionsConfig(
                noop=NoopActionConfig(enabled=True),
                move=MoveActionConfig(enabled=True),
            ),
            objects={"wall": WallConfig()},
        )

        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", "empty", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        sim = Simulation(cfg, seed=42)

        # First step with noop to initialize observations
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        # Find move_east action (should succeed since there's empty space to the east)
        move_east_idx = None
        for i, name in enumerate(sim.action_names):
            if name == "move_east":
                move_east_idx = i
                break

        assert move_east_idx is not None, "Expected move_east action"

        sim.agent(0).set_action(Action(name="move_east"))
        sim.step()

        # Verify the action succeeded
        assert sim.agent(0).last_action_success, "Move should succeed"

        # Check that last_action observation is the move_east action index
        agent = sim.agent(0)
        global_obs_data = agent.global_observations
        assert "last_action" in global_obs_data, "Should have last_action observation"
        assert global_obs_data["last_action"] == move_east_idx, (
            f"Expected last_action to be {move_east_idx} (move_east) when action succeeds, "
            f"got {global_obs_data['last_action']}"
        )
