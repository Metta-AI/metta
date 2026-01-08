"""
Test suite for programmatic environment creation without config files.
Demonstrates the new pattern for creating environments as shown in experiments/arena.py
"""

import pytest
from pydantic import ValidationError

import mettagrid.builder.envs as eb
from mettagrid.builder import building
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.map_builder.random_map import RandomMapBuilder


class TestProgrammaticEnvironments:
    """Test creating environments programmatically without config files."""

    def test_create_simple_config(self):
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
                    move=MoveActionConfig(),
                    noop=NoopActionConfig(),
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

    def test_create_arena_like_config(self):
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

    def test_create_navigation_config(self):
        """Test creating a navigation environment."""

        nav_env = eb.make_navigation(num_agents=4)

        assert nav_env.game.num_agents == 4
        assert "assembler" in nav_env.game.objects
        assert "wall" in nav_env.game.objects
        assert nav_env.game.actions.move is not None

    def test_config_with_custom_rewards(self):
        """Test creating an environment with custom reward configuration."""
        config = MettaGridConfig(
            label="custom_rewards",
            game=GameConfig(
                num_agents=2,
                objects={
                    "wall": building.wall,
                    "assembler": building.assembler_assembler,
                },
                actions=ActionsConfig(
                    move=MoveActionConfig(),
                    noop=NoopActionConfig(),
                ),
                agent=AgentConfig(
                    rewards=AgentRewards(
                        inventory={
                            "heart": 1.0,
                            "ore_red": 0.5,
                            "battery_red": 0.8,
                        },
                    ),
                    inventory=InventoryConfig(
                        limits={
                            "heart": ResourceLimitsConfig(limit=255, resources=["heart"]),
                            "ore_red": ResourceLimitsConfig(limit=10, resources=["ore_red"]),
                            "battery_red": ResourceLimitsConfig(limit=5, resources=["battery_red"]),
                        },
                    ),
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
        assert config.game.agent.inventory.get_limit("heart") == 255
        assert config.game.agent.inventory.get_limit("ore_red") == 10
        assert config.game.agent.inventory.get_limit("battery_red") == 5

    def test_config_with_teams(self):
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
                    move=MoveActionConfig(),
                    noop=NoopActionConfig(),
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


class TestTypeIdentityBreakingChange:
    """Tests reflecting the breaking removal of type_id from Python configs."""

    def test_passing_type_id_raises_validation_error(self):
        with pytest.raises(ValidationError):
            # Pydantic should reject unknown field 'type_id'
            _ = WallConfig(type_id=123)  # type: ignore[arg-type]

    def test_cpp_config_builds_without_type_ids(self):
        # Building C++ config should succeed with name-only objects
        from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config

        objects = {
            "apple": WallConfig(name="apple"),
            "banana": WallConfig(name="banana"),
            "carrot": WallConfig(name="carrot"),
        }
        cfg = GameConfig(objects=objects)
        cpp_cfg = convert_to_cpp_game_config(cfg)
        assert cpp_cfg is not None
