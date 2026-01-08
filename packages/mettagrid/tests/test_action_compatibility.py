"""Test cases for action system compatibility and behavior."""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AttackActionConfig,
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
from mettagrid.test_support.actions import get_agent_position
from mettagrid.test_support.map_builders import ObjectNameMapBuilder

# Rebuild GameConfig after MapBuilderConfig is imported
GameConfig.model_rebuild()


def create_sim(game_config: GameConfig, game_map: list[list[str]], seed: int = 42) -> Simulation:
    """Helper to create a Simulation from config and map."""
    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
    return Simulation(cfg, seed=seed)


def create_basic_config() -> GameConfig:
    """Create a minimal valid game configuration."""
    return GameConfig(
        resource_names=["ore", "wood"],
        num_agents=1,
        obs=ObsConfig(width=7, height=7, num_tokens=50),
        max_steps=100,
        agent=AgentConfig(
            freeze_duration=0,
            inventory=InventoryConfig(
                limits={
                    "ore": ResourceLimitsConfig(limit=10, resources=["ore"]),
                    "wood": ResourceLimitsConfig(limit=10, resources=["wood"]),
                },
            ),
        ),
        actions=ActionsConfig(move=MoveActionConfig(), noop=NoopActionConfig()),
        objects={"wall": WallConfig()},
    )


def create_simple_map():
    """Create a simple 5x5 map with walls around edges."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.default", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


def create_multi_agent_map():
    """Create a simple 7x7 map with multiple agents."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.default", "empty", "agent.default", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.default", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def basic_config():
    """Fixture for basic configuration."""
    return create_basic_config()


@pytest.fixture
def simple_map():
    """Fixture for simple map."""
    return create_simple_map()


@pytest.fixture
def multi_agent_map():
    """Fixture for multi-agent map."""
    return create_multi_agent_map()


class TestActionOrdering:
    """Tests related to action ordering and indexing."""

    def test_action_order_is_fixed(self, basic_config, simple_map):
        """Test that action order is deterministic regardless of config order."""
        # Create environment with original config
        sim1 = create_sim(basic_config, simple_map, 42)
        action_names1 = sim1.action_names

        # Create config with different action order
        reordered_config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs=basic_config.obs,
            agent=basic_config.agent,
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects=basic_config.objects,
        )

        sim2 = create_sim(reordered_config, simple_map, 42)
        action_names2 = sim2.action_names

        # Action order should remain the same despite different config order
        assert action_names1 == action_names2, "Action order should be deterministic"

        # Verify that action names are consistent
        # Note: We don't check exact action list as basic_config may include various actions
        # Just verify noop is first and move actions follow
        assert action_names1[0] == "noop", "Noop should always be first"
        assert any(name.startswith("move_") for name in action_names1), "Should have move actions"

    def test_action_indices_consistency(self, basic_config, simple_map):
        """Test that action indices remain consistent."""
        sim = create_sim(basic_config, simple_map, 42)
        action_names = sim.action_names

        # Verify ordering (noop first, followed by move variants)
        assert action_names[0] == "noop"
        assert action_names[1].startswith("move")


class TestActionValidation:
    """Tests for action validation and error handling."""

    def test_invalid_action_name(self, basic_config, simple_map):
        """Test that invalid action names are handled properly."""
        sim = create_sim(basic_config, simple_map, 42)

        # Try to use an action that doesn't exist
        try:
            sim.agent(0).set_action(Action(name="invalid_action_that_does_not_exist"))
            # This should raise a KeyError when trying to convert action name to index
            raise AssertionError("Should have raised KeyError for invalid action name")
        except KeyError:
            # Expected behavior
            pass

    def test_valid_action_succeeds(self, basic_config, simple_map):
        """Test that valid actions are accepted."""
        sim = create_sim(basic_config, simple_map, 42)

        # Use a valid action
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        # Should complete without error
        assert sim.current_step == 1


class TestResourceRequirements:
    """Tests for action resource requirements."""

    def test_action_with_resource_requirement(self, basic_config, simple_map):
        """Test that actions fail when resource requirements aren't met."""
        # Create new config with resource requirement
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs=basic_config.obs,
            agent=basic_config.agent,
            actions=ActionsConfig(
                move=MoveActionConfig(enabled=True, required_resources={"ore": 1}), noop=NoopActionConfig()
            ),
            objects=basic_config.objects,
        )

        sim = create_sim(config, simple_map, 42)

        move_action_name = next((name for name in sim.action_names if name.startswith("move")), None)
        assert move_action_name is not None, "Expected move action in action names"

        # Agent starts with no resources, so move should fail
        sim.agent(0).set_action(Action(name=move_action_name))
        sim.step()
        assert not sim.agent(0).last_action_success, "Move should fail without required resources"

    def test_action_consumes_resources(self, basic_config, simple_map):
        """Test that actions consume resources when configured."""
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs=basic_config.obs,
            agent=AgentConfig(
                freeze_duration=0,
                inventory=InventoryConfig(
                    limits={
                        "ore": ResourceLimitsConfig(limit=10, resources=["ore"]),
                        "wood": ResourceLimitsConfig(limit=10, resources=["wood"]),
                    },
                    initial={"ore": 5, "wood": 3},
                ),
            ),
            actions=ActionsConfig(
                move=MoveActionConfig(enabled=True, consumed_resources={"ore": 1}), noop=NoopActionConfig()
            ),
            objects=basic_config.objects,
        )

        sim = create_sim(config, simple_map, 42)
        agent = sim.agent(0)

        # Step once to populate observations
        agent.set_action(Action(name="noop"))
        sim.step()

        # Get initial inventory using the inventory property
        initial_inventory = agent.inventory
        initial_ore = initial_inventory.get("ore", 0)
        initial_wood = initial_inventory.get("wood", 0)

        # Verify initial inventory
        assert initial_ore == 5, f"Expected initial ore to be 5, got {initial_ore}"
        assert initial_wood == 3, f"Expected initial wood to be 3, got {initial_wood}"

        # Get agent position before move
        agent_pos = get_agent_position(sim, 0)

        # Move east (which consumes ore)
        agent.set_action(Action(name="move_east"))
        sim.step()

        action_success = sim.agent(0).last_action_success
        new_pos = get_agent_position(sim, 0)
        position_changed = new_pos != agent_pos

        # Get final inventory
        final_inventory = agent.inventory
        final_ore = final_inventory.get("ore", 0)
        final_wood = final_inventory.get("wood", 0)

        # Verify resource consumption
        if action_success and position_changed:
            # Move succeeded, ore should be consumed
            assert final_ore == initial_ore - 1, f"Expected ore to decrease by 1, got {final_ore}"
            assert final_wood == initial_wood, f"Wood should remain unchanged, got {final_wood}"
        else:
            # Move failed, resources should not be consumed
            assert final_ore == initial_ore, "Ore should not be consumed on failed move"
            assert final_wood == initial_wood, "Wood should remain unchanged"


class TestActionSpace:
    """Tests for action space properties."""

    def test_action_space_shape(self, basic_config, simple_map):
        """Test action space dimensions."""
        sim = create_sim(basic_config, simple_map, 42)

        action_names = sim.action_names

        # Verify all action names are unique
        assert len(set(action_names)) == len(action_names), "All action names should be unique"

    def test_single_action_space(self, basic_config, multi_agent_map):
        """Test action space for multi-agent environment."""
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=3,
            max_steps=basic_config.max_steps,
            obs=basic_config.obs,
            agent=basic_config.agent,
            actions=basic_config.actions,
            objects=basic_config.objects,
        )

        sim = create_sim(config, multi_agent_map, 42)

        action_names = sim.action_names
        assert len(set(action_names)) == len(action_names), "All action names should be unique"

        # When stepping, we need to provide actions for all agents
        # Create actions for all 3 agents using the noop action
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))

        # This should work without error
        sim.step()

        # Verify we get results for all agents
        action_success = [sim.agent(i).last_action_success for i in range(sim.num_agents)]
        assert len(action_success) == 3, "Should get action success for all agents"


class TestSpecialActions:
    """Tests for special action types."""

    def test_attack_action_registration(self, basic_config, simple_map):
        """Test that attack does NOT create standalone actions (attack triggers via move only)."""
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs=basic_config.obs,
            agent=basic_config.agent,
            actions=ActionsConfig(
                attack=AttackActionConfig(
                    enabled=True, required_resources={}, consumed_resources={}, defense_resources={}
                ),
                move=MoveActionConfig(),
                noop=NoopActionConfig(),
            ),
            objects=basic_config.objects,
        )

        sim = create_sim(config, simple_map, 42)
        action_names = sim.action_names

        # Attack only triggers via move, no standalone attack actions
        attack_actions = [name for name in action_names if name.startswith("attack_")]
        assert len(attack_actions) == 0, f"Expected no attack variants (attack via move only), found {attack_actions}"

        # Verify noop is first, followed by move actions
        assert action_names[0] == "noop", "Noop should always be first"
        assert any(name.startswith("move_") for name in action_names), "Should have move actions"


class TestResourceOrdering:
    """Tests for inventory item ordering effects."""

    def test_resource_order(self, basic_config, simple_map):
        """Test that resources maintain their order."""
        # Config with ore first
        config1 = GameConfig(
            resource_names=["ore", "wood"],
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs=basic_config.obs,
            agent=basic_config.agent,
            actions=basic_config.actions,
            objects=basic_config.objects,
        )

        # Config with wood first
        config2 = GameConfig(
            resource_names=["wood", "ore"],
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs=basic_config.obs,
            agent=basic_config.agent,
            actions=basic_config.actions,
            objects=basic_config.objects,
        )

        sim1 = create_sim(config1, simple_map, 42)
        sim2 = create_sim(config2, simple_map, 42)

        assert sim1.resource_names == ["ore", "wood"]
        assert sim2.resource_names == ["wood", "ore"]

        # This affects resource indices in the implementation
        # ore is index 0 in sim1, but index 1 in sim2
