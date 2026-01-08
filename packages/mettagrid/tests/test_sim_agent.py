"""Tests for SimulationAgent functionality."""

import numpy as np

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.simulator import Action, Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


def create_test_sim(initial_inventory: dict[str, int] | None = None) -> Simulation:
    """Create a test simulation with optional initial inventory."""
    if initial_inventory is None:
        initial_inventory = {"wood": 5, "stone": 3, "iron": 1}

    game_config = GameConfig(
        max_steps=50,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=50),
        resource_names=["wood", "stone", "iron"],
        actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
        objects={"wall": WallConfig()},
        agent=AgentConfig(
            inventory=InventoryConfig(
                limits={
                    "wood": ResourceLimitsConfig(limit=10, resources=["wood"]),
                    "stone": ResourceLimitsConfig(limit=10, resources=["stone"]),
                    "iron": ResourceLimitsConfig(limit=10, resources=["iron"]),
                },
                initial=initial_inventory,
            ),
        ),
    )

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(
        map_data=[
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.default", "empty", "agent.default", "wall"],
            ["wall", "empty", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]
    )

    return Simulation(cfg, seed=42)


class TestSimulationAgentProperties:
    """Test SimulationAgent property accessors."""

    def test_agent_id(self):
        """Test that agent ID is correctly returned."""
        sim = create_test_sim()

        agent0 = sim.agent(0)
        agent1 = sim.agent(1)

        assert agent0.id == 0
        assert agent1.id == 1

    def test_inventory_property(self):
        """Test that inventory property returns correct values."""
        sim = create_test_sim(initial_inventory={"wood": 7, "stone": 2, "iron": 4})

        # Step once to populate observations
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        agent = sim.agent(0)
        inv = agent.inventory

        # Check inventory values
        assert "wood" in inv, "Wood should be in inventory"
        assert "stone" in inv, "Stone should be in inventory"
        assert "iron" in inv, "Iron should be in inventory"

        assert inv["wood"] == 7, f"Expected wood=7, got {inv['wood']}"
        assert inv["stone"] == 2, f"Expected stone=2, got {inv['stone']}"
        assert inv["iron"] == 4, f"Expected iron=4, got {inv['iron']}"

    def test_inventory_empty(self):
        """Test inventory property with zero initial inventory."""
        sim = create_test_sim(initial_inventory={})

        # Step once to populate observations
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        agent = sim.agent(0)
        inv = agent.inventory

        # All resources should be 0 or not present
        assert inv.get("wood", 0) == 0
        assert inv.get("stone", 0) == 0
        assert inv.get("iron", 0) == 0

    def test_observation_property(self):
        """Test that observation property returns AgentObservation."""
        sim = create_test_sim()

        # Step once
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        agent = sim.agent(0)
        obs = agent.observation

        assert obs.agent_id == 0
        assert hasattr(obs, "tokens")
        assert len(obs.tokens) > 0, "Should have observation tokens"

    def test_step_reward(self):
        """Test step reward property."""
        sim = create_test_sim()

        agent = sim.agent(0)

        # Initial reward should be 0
        assert agent.step_reward == 0.0

        # Step with noop
        agent.set_action(Action(name="noop"))
        for i in range(1, sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        # Reward should still be accessible (may be 0 or some value)
        reward = agent.step_reward
        assert isinstance(reward, (int, float, np.floating)), "Reward should be numeric"

    def test_episode_reward(self):
        """Test episode reward accumulation."""
        sim = create_test_sim()

        agent = sim.agent(0)

        # Initial episode reward should be 0
        initial_reward = agent.episode_reward
        assert initial_reward == 0.0

        # Take a few steps
        for _ in range(3):
            for i in range(sim.num_agents):
                sim.agent(i).set_action(Action(name="noop"))
            sim.step()

        # Episode reward should be accessible
        episode_reward = agent.episode_reward
        assert isinstance(episode_reward, (int, float, np.floating)), "Episode reward should be numeric"


class TestSimulationAgentActions:
    """Test SimulationAgent action-related methods."""

    def test_set_action(self):
        """Test setting an action for an agent."""
        sim = create_test_sim()

        agent = sim.agent(0)

        # Set action should not raise
        agent.set_action(Action(name="noop"))

        # Complete the step for all agents
        for i in range(1, sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))

        sim.step()

        # Should complete without error
        assert sim.current_step == 1

    def test_set_different_actions(self):
        """Test that different agents can have different actions."""
        sim = create_test_sim()

        # Agent 0: noop, Agent 1: move
        sim.agent(0).set_action(Action(name="noop"))
        sim.agent(1).set_action(Action(name="move_east"))

        sim.step()

        # Both should succeed (or fail based on game logic, but no exceptions)
        assert sim.current_step == 1


class TestSimulationAgentInventoryModification:
    """Test SimulationAgent inventory modification methods."""

    def test_set_inventory(self):
        """Test setting agent inventory."""
        sim = create_test_sim(initial_inventory={})

        agent = sim.agent(0)

        # Set new inventory
        new_inventory = {"wood": 8, "stone": 5, "iron": 2}
        agent.set_inventory(new_inventory)

        # Step to update observations
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        # Check inventory was updated
        inv = agent.inventory
        assert inv.get("wood", 0) == 8, f"Expected wood=8, got {inv.get('wood', 0)}"
        assert inv.get("stone", 0) == 5, f"Expected stone=5, got {inv.get('stone', 0)}"
        assert inv.get("iron", 0) == 2, f"Expected iron=2, got {inv.get('iron', 0)}"

    def test_set_inventory_partial(self):
        """Test setting partial inventory (only some resources)."""
        sim = create_test_sim(initial_inventory={"wood": 5, "stone": 3, "iron": 1})

        agent = sim.agent(0)

        # Set only wood and stone (iron should be cleared)
        agent.set_inventory({"wood": 10, "stone": 7})

        # Step to update observations
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        # Check inventory
        inv = agent.inventory
        assert inv.get("wood", 0) == 10
        assert inv.get("stone", 0) == 7
        # Iron should be 0 since it wasn't mentioned in set_inventory
        assert inv.get("iron", 0) == 0

    def test_set_inventory_empty(self):
        """Test clearing inventory by setting empty dict."""
        sim = create_test_sim(initial_inventory={"wood": 5, "stone": 3, "iron": 1})

        agent = sim.agent(0)

        # Clear inventory
        agent.set_inventory({})

        # Step to update observations
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        # All resources should be 0
        inv = agent.inventory
        assert inv.get("wood", 0) == 0
        assert inv.get("stone", 0) == 0
        assert inv.get("iron", 0) == 0


class TestMultipleAgents:
    """Test interactions between multiple agents."""

    def test_independent_agents(self):
        """Test that agents have independent inventory."""
        sim = create_test_sim()

        # Set different inventory for each agent
        sim.agent(0).set_inventory({"wood": 10, "stone": 0, "iron": 0})
        sim.agent(1).set_inventory({"wood": 0, "stone": 10, "iron": 0})

        # Step
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        # Check they have different inventory
        inv0 = sim.agent(0).inventory
        inv1 = sim.agent(1).inventory

        assert inv0.get("wood", 0) == 10
        assert inv0.get("stone", 0) == 0

        assert inv1.get("wood", 0) == 0
        assert inv1.get("stone", 0) == 10

    def test_agent_iterator(self):
        """Test iterating over all agents."""
        sim = create_test_sim()

        # Get all agents
        agents = sim.agents()

        assert len(agents) == 2
        assert all(hasattr(agent, "id") for agent in agents)
        assert agents[0].id == 0
        assert agents[1].id == 1


class TestAgentInitialVibe:
    """Test agent initial_vibe configuration."""

    def test_initial_vibe_default(self):
        """Test that agents start with vibe 0 by default."""
        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=50),
            resource_names=["wood"],
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects={"wall": WallConfig()},
            agent=AgentConfig(inventory=InventoryConfig(default_limit=10)),
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(
            map_data=[
                ["wall", "wall", "wall"],
                ["wall", "agent.default", "wall"],
                ["wall", "wall", "wall"],
            ]
        )

        sim = Simulation(cfg, seed=42)
        grid_objects = sim._c_sim.grid_objects(0)
        agent_obj = next(o for o in grid_objects.values() if o["type_name"] == "agent")

        assert agent_obj["vibe"] == 0, f"Expected default vibe=0, got {agent_obj['vibe']}"

    def test_initial_vibe_custom(self):
        """Test that agents start with configured initial_vibe value."""
        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=50),
            resource_names=["wood"],
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects={"wall": WallConfig()},
            agent=AgentConfig(
                initial_vibe=2,
                inventory=InventoryConfig(default_limit=10),
            ),
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(
            map_data=[
                ["wall", "wall", "wall"],
                ["wall", "agent.default", "wall"],
                ["wall", "wall", "wall"],
            ]
        )

        sim = Simulation(cfg, seed=42)
        grid_objects = sim._c_sim.grid_objects(0)
        agent_obj = next(o for o in grid_objects.values() if o["type_name"] == "agent")

        assert agent_obj["vibe"] == 2, f"Expected initial_vibe=2, got {agent_obj['vibe']}"
