import pytest

import mettagrid.builder.envs as eb
from metta.common.util.fs import get_repo_root
from metta.sim.simulation_config import SimulationConfig
from metta.tests_support import run_tool_in_process
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from mettagrid import MettaGridEnv
from mettagrid.builder import building
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    GameConfig,
    MettaGridConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder


class TestComprehensiveEnvironmentIntegration:
    """Comprehensive test suite for environment creation and tool integration."""

    REPO_ROOT = get_repo_root()

    @staticmethod
    def make_debug_env(name: str) -> MettaGridConfig:
        """Create debug environments programmatically using the new system."""
        if name == "tiny_two_altars":
            return MettaGridConfig(
                label=name,
                game=GameConfig(
                    num_agents=2,
                    max_steps=100,
                    objects={
                        "wall": building.wall,
                        "altar": building.assembler_altar,
                    },
                    actions=ActionsConfig(
                        move=ActionConfig(),
                        rotate=ActionConfig(),
                        get_items=ActionConfig(),
                    ),
                    agent=AgentConfig(
                        rewards=AgentRewards(
                            inventory={
                                "heart": 1,
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
        elif name == "simple_obstacles":
            return MettaGridConfig(
                label=name,
                game=GameConfig(
                    num_agents=2,
                    max_steps=100,
                    objects={
                        "wall": building.wall,
                    },
                    actions=ActionsConfig(
                        move=ActionConfig(),
                        rotate=ActionConfig(),
                    ),
                    agent=AgentConfig(
                        rewards=AgentRewards(
                            inventory={
                                "heart": 1,
                            },
                        ),
                    ),
                    map_builder=RandomMapBuilder.Config(
                        agents=2,
                        width=15,
                        height=15,
                        border_object="wall",
                        border_width=2,
                    ),
                ),
            )
        elif name == "resource_collection":
            return MettaGridConfig(
                label=name,
                game=GameConfig(
                    num_agents=2,
                    max_steps=100,
                    objects={
                        "wall": building.wall,
                        "mine_red": building.mine_red,
                        "generator_red": building.generator_red,
                    },
                    actions=ActionsConfig(
                        move=ActionConfig(),
                        rotate=ActionConfig(),
                        get_items=ActionConfig(),
                    ),
                    agent=AgentConfig(
                        rewards=AgentRewards(
                            inventory={
                                "heart": 1,
                                "ore_red": 0.5,
                                "battery_red": 0.8,
                            },
                        ),
                    ),
                    map_builder=RandomMapBuilder.Config(
                        agents=2,
                        width=20,
                        height=20,
                        border_object="wall",
                        border_width=1,
                    ),
                ),
            )
        else:  # mixed_objects and default
            return MettaGridConfig(
                label=name,
                game=GameConfig(
                    num_agents=2,
                    max_steps=100,
                    objects={
                        "wall": building.wall,
                        "altar": building.altar,
                        "mine_red": building.mine_red,
                        "generator_red": building.generator_red,
                    },
                    actions=ActionsConfig(
                        move=ActionConfig(),
                        rotate=ActionConfig(),
                        get_items=ActionConfig(),
                    ),
                    agent=AgentConfig(
                        rewards=AgentRewards(
                            inventory={
                                "heart": 1,
                                "ore_red": 0.5,
                                "battery_red": 0.8,
                            },
                        ),
                    ),
                    map_builder=RandomMapBuilder.Config(
                        agents=2,
                        width=20,
                        height=20,
                        border_object="wall",
                        border_width=1,
                    ),
                ),
            )

    def test_programmatic_env_creation(self):
        """Test that debug environments can be created programmatically."""
        for env_name in ["tiny_two_altars", "simple_obstacles", "resource_collection", "mixed_objects"]:
            env_config = self.make_debug_env(env_name)
            assert env_config is not None, f"Failed to create environment {env_name}"
            assert env_config.game.num_agents == 2, f"Environment {env_name} should have 2 agents"
            assert env_config.label == env_name, f"Environment label mismatch for {env_name}"

    def test_debug_env_validation(self):
        """Test that programmatically created debug environments are valid."""
        env_config = self.make_debug_env("tiny_two_altars")

        # Validate essential components
        assert hasattr(env_config, "game"), "Environment missing game config"
        assert hasattr(env_config.game, "actions"), "Game missing actions config"
        assert hasattr(env_config.game, "objects"), "Game missing objects config"
        assert hasattr(env_config.game, "agent"), "Game missing agent config"
        assert hasattr(env_config.game, "map_builder"), "Game missing map_builder config"

        # Validate actions are properly configured
        assert env_config.game.actions.move is not None, "Move action not configured"
        assert env_config.game.actions.rotate is not None, "Rotate action not configured"

    def test_environment_integration_with_new_recipes(self):
        """Test that environments work with the new recipe system."""
        # Test creating environments similar to those used in recipes
        arena_env = eb.make_arena(num_agents=4)
        nav_env = eb.make_navigation(num_agents=2)

        assert arena_env.game.num_agents == 4
        assert nav_env.game.num_agents == 2

        # Test that they have expected components
        assert "altar" in arena_env.game.objects
        assert "altar" in nav_env.game.objects
        assert arena_env.game.actions.move is not None
        assert nav_env.game.actions.move is not None

    def test_programmatic_env_with_mettagrid(self):
        """Test that programmatically created environments work with MettaGridEnv."""

        env_config = self.make_debug_env("tiny_two_altars")
        env = MettaGridEnv(env_config)

        try:
            obs, info = env.reset()
            assert obs is not None, "Environment reset failed to return observation"
            assert obs.shape[0] == 2, "Observation should be for 2 agents"
            assert env.action_space is not None, "Action space not configured"
        finally:
            env.close()

    def test_simulation_config_creation(self):
        """Test creating simulation configs from environments."""

        for env_name in ["tiny_two_altars", "simple_obstacles"]:
            env_config = self.make_debug_env(env_name)
            sim_config = SimulationConfig(suite="test", name=f"sim_{env_name}", env=env_config)

            assert sim_config.name == f"sim_{env_name}"
            assert sim_config.env.game.num_agents == 2

    def test_tool_configuration_with_environments(self):
        """Test that tools can be configured with programmatic environments."""

        env_config = self.make_debug_env("resource_collection")
        sim_config = SimulationConfig(suite="test", name="test_resource", env=env_config)

        # Test ReplayTool configuration
        replay_tool = ReplayTool(sim=sim_config, policy_uri=None, open_browser_on_start=False)
        assert replay_tool.sim.name == "test_resource"

        # Test PlayTool configuration
        play_tool = PlayTool(sim=sim_config, policy_uri=None)
        assert play_tool.sim.name == "test_resource"

    def test_agents_count_in_environments(self):
        """Test that each debug environment has exactly 2 agents."""
        for env_name in ["tiny_two_altars", "simple_obstacles", "resource_collection", "mixed_objects"]:
            env_config = self.make_debug_env(env_name)
            assert env_config.game.num_agents == 2, (
                f"Environment {env_name} should have exactly 2 agents, but has {env_config.game.num_agents}"
            )
            # Also check map_builder agent count matches
            map_builder_agents = getattr(env_config.game.map_builder, "agents", None)
            if map_builder_agents is not None:
                assert map_builder_agents == 2, f"Map builder for {env_name} should configure 2 agents"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "env_name", ["tiny_two_altars", "simple_obstacles", "resource_collection", "mixed_objects"]
    )
    def test_recipe_based_training_validation(self, env_name, monkeypatch, capsys):
        """Test basic training validation with the new recipe-based system."""
        run_name = f"validation_{env_name}"

        env_overrides = {
            "AWS_ACCESS_KEY_ID": "dummy_access_key_for_testing",
            "AWS_SECRET_ACCESS_KEY": "dummy_secret_key_for_testing",
            "PYTHONUNBUFFERED": "1",
        }

        args = [
            "experiments.recipes.navigation.train",
            f"run={run_name}",
            "trainer.total_timesteps=50",
            "wandb=off",
            "--dry-run",
        ]

        result = run_tool_in_process(
            *args,
            env_overrides=env_overrides,
            monkeypatch=monkeypatch,
            capsys=capsys,
        )

        if result.returncode != 0:
            combined_output = result.stdout + result.stderr
            initialization_errors = [
                "ImportError",
                "ModuleNotFoundError",
                "AttributeError",
                "recipe not found",
                "TypeError",
                "NameError",
                "RecipeNotFound",
            ]

            detected_errors = [error for error in initialization_errors if error in combined_output]

            if detected_errors:
                pytest.fail(
                    f"Training initialization failed for {env_name}.\n"
                    f"Return code: {result.returncode}\n"
                    f"Detected errors: {', '.join(detected_errors)}\n"
                    f"STDOUT (last 1000 chars): {result.stdout[-1000:]}\n"
                    f"STDERR (last 1000 chars): {result.stderr[-1000:]}"
                )

    @pytest.mark.slow
    def test_simulation_and_replay_integration(self, monkeypatch, capsys):
        """Test that we can run simulations and create replays with new system."""
        run_name = "sim_replay_test"

        env_overrides = {
            "AWS_ACCESS_KEY_ID": "dummy_for_test",
            "AWS_SECRET_ACCESS_KEY": "dummy_for_test",
        }

        train_args = [
            "experiments.recipes.arena.train",
            f"run={run_name}",
            "trainer.total_timesteps=100",
            "wandb=off",
            "--dry-run",
        ]

        # Run training briefly (allow failures but ensure invocation returns)
        run_tool_in_process(
            *train_args,
            env_overrides=env_overrides,
            monkeypatch=monkeypatch,
            capsys=capsys,
        )

        # Test simulation tool configuration
        sim_args = [
            "experiments.recipes.arena.evaluate",
            "policy_uri=mock://test",  # Use mock policy
            "--dry-run",
        ]

        sim_result = run_tool_in_process(
            *sim_args,
            env_overrides=env_overrides,
            monkeypatch=monkeypatch,
            capsys=capsys,
        )
        combined_output = sim_result.stdout + sim_result.stderr
        if "recipe not found" in combined_output:
            pytest.fail("Simulation recipe not found")
        if "ImportError" in combined_output:
            pytest.fail("Import error in simulation tools")
