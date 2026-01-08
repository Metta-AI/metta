"""Tests for the cogsguard recipe.

These tests verify that the cogsguard environment can be created and played.
"""

from __future__ import annotations

import random

import pytest

# Import after cogsguard to avoid circular import issues
from mettagrid.simulator import Simulation
from recipes.experiment import cogsguard


class TestCogsguardEnvironment:
    """Test cogsguard environment creation and basic operation."""

    def test_make_env_creates_valid_config(self) -> None:
        """Test that make_env creates a valid MettaGridConfig."""
        env_config = cogsguard.make_env(num_agents=4, max_steps=100)

        assert env_config is not None
        assert env_config.game.num_agents == 4
        assert env_config.game.max_steps == 100

        # Check resources are configured
        assert "energy" in env_config.game.resource_names
        assert "heart" in env_config.game.resource_names
        assert "hp" in env_config.game.resource_names

        # Check gear resources
        for gear_type in cogsguard.gear:
            assert gear_type in env_config.game.resource_names

        # Check element resources
        for element in cogsguard.elements:
            assert element in env_config.game.resource_names

    def test_environment_simulation_runs(self) -> None:
        """Test that the environment can be simulated for multiple steps."""
        env_config = cogsguard.make_env(num_agents=4, max_steps=100)
        sim = Simulation(env_config)

        # Verify simulation initialized correctly
        assert sim.num_agents == 4
        assert len(sim.action_names) > 0
        assert "noop" in sim.action_names

        # Run simulation for a few steps with random actions
        num_steps = 10
        for _step in range(num_steps):
            # Set random actions for each agent
            for agent_id in range(sim.num_agents):
                action = random.choice(sim.action_names)
                sim.agent(agent_id).set_action(action)

            sim.step()

            # Verify observations and rewards are valid
            obs = sim._c_sim.observations()
            rewards = sim._c_sim.rewards()
            terminals = sim._c_sim.terminals()

            assert obs.shape[0] == sim.num_agents
            assert rewards.shape[0] == sim.num_agents
            assert terminals.shape[0] == sim.num_agents

    def test_environment_with_noop_actions(self) -> None:
        """Test that the environment works with noop actions."""
        env_config = cogsguard.make_env(num_agents=2, max_steps=50)
        sim = Simulation(env_config)

        # Run with noop actions only
        for _ in range(5):
            for agent_id in range(sim.num_agents):
                sim.agent(agent_id).set_action("noop")
            sim.step()

        # Should complete without errors
        assert sim.current_step == 5

    def test_objects_configured_correctly(self) -> None:
        """Test that game objects are configured properly."""
        env_config = cogsguard.make_env(num_agents=4, max_steps=100)

        objects = env_config.game.objects
        assert len(objects) > 0

        # Check that key object types exist
        assert "wall" in objects
        assert "assembler" in objects  # nexus
        assert "charger" in objects  # supply depot
        assert "chest" in objects

        # Check extractors for all elements
        for element in cogsguard.elements:
            assert f"{element}_extractor" in objects

        # Check gear stations
        for gear_type in cogsguard.gear:
            assert f"{gear_type}_station" in objects

    def test_collectives_configured(self) -> None:
        """Test that collectives are properly configured."""
        env_config = cogsguard.make_env(num_agents=4, max_steps=100)

        collective_names = [c.name for c in env_config.game.collectives]
        assert "cogs" in collective_names
        assert "clips" in collective_names

        # Check cogs collective has initial resources
        cogs = next(c for c in env_config.game.collectives if c.name == "cogs")
        assert cogs.inventory.initial.get("carbon", 0) > 0
        assert cogs.inventory.initial.get("heart", 0) > 0


class TestCogsguardCurriculum:
    """Test cogsguard curriculum configuration."""

    def test_make_curriculum_returns_valid_config(self) -> None:
        """Test that make_curriculum creates a valid curriculum config."""
        curriculum = cogsguard.make_curriculum()

        assert curriculum is not None
        # Check that the curriculum has tasks configured
        assert curriculum.task_generator is not None

    def test_simulations_returns_valid_configs(self) -> None:
        """Test that simulations() returns valid simulation configs."""
        sims = cogsguard.simulations()

        assert len(sims) > 0
        for sim in sims:
            assert sim.suite == "cogsguard"
            assert sim.env is not None


@pytest.mark.parametrize("num_agents", [2, 4, 8])
def test_environment_scales_with_agents(num_agents: int) -> None:
    """Test that the environment works with different agent counts."""
    env_config = cogsguard.make_env(num_agents=num_agents, max_steps=50)
    sim = Simulation(env_config)

    assert sim.num_agents == num_agents

    # Run a few steps
    for _ in range(3):
        for agent_id in range(sim.num_agents):
            sim.agent(agent_id).set_action("noop")
        sim.step()

    assert sim.current_step == 3
