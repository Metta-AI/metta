"""
Test suite for programmatic environment creation without config files.
Demonstrates the new pattern for creating environments as shown in experiments/arena.py
"""

import pytest

import mettagrid.builder.envs as eb
from mettagrid import MettaGridEnv
from mettagrid.builder import building
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder


class TestProgrammaticEnvironments:
    """Test creating environments programmatically without config files."""

    def test_create_simple_environment(self):
        """Test creating a simple environment with basic components."""
        config = MettaGridConfig(
            label="test_simple",
            game=GameConfig(
                num_agents=4,
                max_steps=100,
                objects={
                    "wall": building.wall,
                },
                actions=ActionsConfig(
                    move=ActionConfig(),
                    rotate=ActionConfig(),
                    noop=ActionConfig(),
                ),
                agent=AgentConfig(
                    rewards=AgentRewards(
                        inventory={
                            "heart": 1,
                        }
                    ),
                ),
                map_builder=RandomMapBuilder.Config(
                    agents=4,
                    width=15,
                    height=15,
                    border_object="wall",
                    border_width=1,
                ),
            ),
        )

        assert config.label == "test_simple"
        assert config.game.num_agents == 4
        assert config.game.max_steps == 100
        assert "wall" in config.game.objects
        assert config.game.actions.move is not None

    def test_create_arena_like_environment(self):
        """Test creating an arena-style environment similar to experiments/arena.py."""

        # Use the make_arena function from the envs module
        arena_env = eb.make_arena(num_agents=8, combat=False)

        assert arena_env.game.num_agents == 8
        assert arena_env.label == "arena"
        # Check that combat is disabled (laser cost should be high)
        assert arena_env.game.actions.attack.consumed_resources["laser"] == 100

        # Test with combat enabled
        combat_env = eb.make_arena(num_agents=8, combat=True)
        assert combat_env.label == "arena.combat"
        assert combat_env.game.actions.attack.consumed_resources["laser"] == 1

    def test_create_navigation_environment(self):
        """Test creating a navigation environment."""

        nav_env = eb.make_navigation(num_agents=4)

        assert nav_env.game.num_agents == 4
        assert "altar" in nav_env.game.objects
        assert "wall" in nav_env.game.objects
        assert nav_env.game.actions.move is not None
        assert nav_env.game.actions.rotate is not None
        assert nav_env.game.actions.get_items is not None

    def test_environment_with_custom_rewards(self):
        """Test creating an environment with custom reward configuration."""
        config = MettaGridConfig(
            label="custom_rewards",
            game=GameConfig(
                num_agents=2,
                objects={
                    "wall": building.wall,
                    "altar": building.assembler_altar,
                },
                actions=ActionsConfig(
                    move=ActionConfig(),
                    get_items=ActionConfig(),
                ),
                agent=AgentConfig(
                    rewards=AgentRewards(
                        inventory={
                            "heart": 1.0,
                            "ore_red": 0.5,
                            "battery_red": 0.8,
                        },
                    ),
                    resource_limits={
                        "heart": 255,
                        "ore_red": 10,
                        "battery_red": 5,
                    },
                ),
                map_builder=RandomMapBuilder.Config(
                    agents=2,
                    width=10,
                    height=10,
                    border_object="wall",
                    border_width=1,
                ),
            ),
        )

        # Verify custom rewards are set
        rewards = config.game.agent.rewards.inventory
        assert rewards["heart"] == 1.0
        assert rewards["ore_red"] == 0.5
        assert rewards["battery_red"] == 0.8

        # Verify resource limits
        limits = config.game.agent.resource_limits
        assert limits["heart"] == 255
        assert limits["ore_red"] == 10
        assert limits["battery_red"] == 5

    def test_environment_with_teams(self):
        """Test creating an environment with agent teams."""
        # Create agents with different team configurations
        agents = []
        # Team 0: 3 agents with higher heart reward
        for _ in range(3):
            agents.append(
                AgentConfig(
                    team_id=0,
                    rewards=AgentRewards(
                        inventory={
                            "heart": 2,
                        },
                    ),
                )
            )
        # Team 1: 3 agents with lower heart reward
        for _ in range(3):
            agents.append(
                AgentConfig(
                    team_id=1,
                    rewards=AgentRewards(
                        inventory={
                            "heart": 1,
                        },
                    ),
                )
            )

        config = MettaGridConfig(
            label="teams_test",
            game=GameConfig(
                num_agents=6,
                objects={
                    "wall": building.wall,
                },
                actions=ActionsConfig(
                    move=ActionConfig(),
                ),
                agent=AgentConfig(),
                agents=agents,
                map_builder=RandomMapBuilder.Config(
                    agents=6,
                    width=20,
                    height=20,
                    border_object="wall",
                    border_width=1,
                ),
            ),
        )

        assert len(config.game.agents) == 6
        # Check that we have 3 agents in each team
        team_0_count = sum(1 for a in config.game.agents if a.team_id == 0)
        team_1_count = sum(1 for a in config.game.agents if a.team_id == 1)
        assert team_0_count == 3
        assert team_1_count == 3
        # Check rewards are set correctly
        for agent in config.game.agents[:3]:
            assert agent.rewards.inventory["heart"] == 2
        for agent in config.game.agents[3:]:
            assert agent.rewards.inventory["heart"] == 1

        # This would be a good addition to the test, but we don't currently expose cpp_config.objects.
        # cpp_config = convert_to_cpp_game_config(config.game)
        # # The keys will be ints, but the values should be easy to check. Did we merge correctly?
        # assert set(cpp_config.objects["agent.team_a"]["resource_rewards"].values()) == {0.5, 0.8, 2}
        # assert set(cpp_config.objects["agent.team_b"]["resource_rewards"].values()) == {0.5, 0.8, 1}

    @pytest.mark.slow
    def test_environment_with_mettagrid_integration(self):
        """Test that programmatic environments work with MettaGridEnv."""

        # Create environment config
        config = eb.make_navigation(num_agents=2)

        # Initialize the actual environment
        env = MettaGridEnv(config)

        try:
            # Reset and verify
            obs, info = env.reset()
            assert obs is not None
            assert obs.shape[0] == 2  # 2 agents

            # Verify action space
            assert env.action_space is not None

            # Take a random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            assert obs is not None
            assert reward is not None
            assert done is not None
            assert truncated is not None

        finally:
            env.close()


class TestTypeIdAllocation:
    """Unit tests for automatic type_id resolution."""

    def test_auto_assign_type_ids_with_mixed_explicit_and_implicit_values(self):
        objects = {
            "apple": WallConfig(type_id=2),
            "banana": WallConfig(),
            "carrot": WallConfig(type_id=4),
            "date": WallConfig(),
            "elderberry": WallConfig(),
        }

        config = GameConfig(objects=objects)

        assert config.objects["apple"].type_id == 2
        assert config.objects["banana"].type_id == 1
        assert config.objects["carrot"].type_id == 4
        assert config.objects["date"].type_id == 3
        assert config.objects["elderberry"].type_id == 5

        assert config.resolved_type_ids["apple"] == 2
        assert config.resolved_type_ids["banana"] == 1
        assert config.resolved_type_ids["carrot"] == 4
        assert config.resolved_type_ids["date"] == 3
        assert config.resolved_type_ids["elderberry"] == 5

    def test_auto_assign_type_ids_raises_when_pool_exhausted(self):
        objects = {f"object_{index:03d}": WallConfig() for index in range(256)}

        with pytest.raises(ValueError) as err:
            GameConfig(objects=objects)

        assert "auto-generated type_id exceeds uint8 range" in str(err.value)
