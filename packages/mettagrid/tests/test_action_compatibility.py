"""Test cases for action system compatibility and behavior."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AttackActionConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.mettagrid_c import MettaGrid, dtype_actions
from mettagrid.test_support.actions import action_index, get_agent_position, get_current_observation
from mettagrid.test_support.orientation import Orientation


def create_basic_config() -> GameConfig:
    """Create a minimal valid game configuration."""
    return GameConfig(
        resource_names=["ore", "wood"],
        num_agents=1,
        max_steps=100,
        obs_width=7,
        obs_height=7,
        num_observation_tokens=50,
        agent=AgentConfig(
            freeze_duration=0,
            resource_limits={"ore": 10, "wood": 10},
        ),
        actions=ActionsConfig(move=ActionConfig(), noop=ActionConfig(), rotate=ActionConfig()),
        objects={"wall": WallConfig(type_id=1, swappable=False)},
        allow_diagonals=True,
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
        env1 = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        action_names1 = env1.action_names()

        # Create config with different action order
        reordered_config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=basic_config.agent,
            actions=ActionsConfig(rotate=ActionConfig(), noop=ActionConfig(), move=ActionConfig()),
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        env2 = MettaGrid(from_mettagrid_config(reordered_config), simple_map, 42)
        action_names2 = env2.action_names()

        # Action order should remain the same despite different config order
        assert action_names1 == action_names2, "Action order should be deterministic"

        # Verify the expected order (noop is always first when enabled)
        orientation_labels = ["north", "south", "west", "east"]
        if basic_config.allow_diagonals:
            orientation_labels.extend(["northwest", "northeast", "southwest", "southeast"])

        expected = ["noop"]
        expected.extend([f"move_{label}" for label in orientation_labels])
        expected.extend([f"rotate_{label}" for label in orientation_labels])
        assert action_names1 == expected

    def test_action_indices_consistency(self, basic_config, simple_map):
        """Test that action indices remain consistent."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        action_names = env.action_names()

        # Verify ordering (noop first, followed by move and rotate variants)
        assert action_names[0] == "noop"
        assert action_names[1].startswith("move")
        assert any(name.startswith("rotate") for name in action_names)


class TestActionValidation:
    """Tests for action validation and error handling."""

    def test_invalid_action_type(self, basic_config, simple_map):
        """Test that invalid action types are handled properly."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        env.reset()

        # Try invalid flattened index
        invalid_action = np.array([env.action_space.n + 99], dtype=dtype_actions)
        env.step(invalid_action)

        # Action should fail
        assert not env.action_success()[0], "Invalid action type should fail"

    def test_invalid_action_index(self, basic_config, simple_map):
        """Test that invalid action indices are handled properly."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        env.reset()

        # Use an out-of-range flattened index to simulate invalid input
        invalid_action = np.array([env.action_space.n + 1], dtype=dtype_actions)
        env.step(invalid_action)

        # Action should fail
        assert not env.action_success()[0], "Invalid action argument should fail"


class TestResourceRequirements:
    """Tests for action resource requirements."""

    def test_action_with_resource_requirement(self, basic_config, simple_map):
        """Test that actions fail when resource requirements aren't met."""
        # Create new config with resource requirement
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=basic_config.agent,
            actions=ActionsConfig(
                move=ActionConfig(enabled=True, required_resources={"ore": 1}),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        env.reset()

        move_action_idx = next(
            (idx for idx, name in enumerate(env.action_names()) if name.startswith("move")),
            None,
        )
        assert move_action_idx is not None, "Expected move action in action names"
        move_action = np.array([move_action_idx], dtype=dtype_actions)

        # Agent starts with no resources, so move should fail
        env.step(move_action)
        assert not env.action_success()[0], "Move should fail without required resources"

    def test_action_consumes_resources(self, basic_config, simple_map):
        """Test that actions consume resources when configured."""
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=AgentConfig(
                freeze_duration=0,
                resource_limits={"ore": 10, "wood": 10},
                initial_inventory={"ore": 5, "wood": 3},
            ),
            actions=ActionsConfig(
                move=ActionConfig(enabled=True, consumed_resources={"ore": 1}),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        env.reset()

        # Get initial observation
        initial_obs = get_current_observation(env, agent_idx=0)

        ore_feature_id = env.feature_spec()["inv:ore"]["id"]
        wood_feature_id = env.feature_spec()["inv:wood"]["id"]

        # Find inventory tokens by feature type
        initial_ore_count = None
        initial_wood_count = None

        for i in range(initial_obs.shape[1]):
            token = initial_obs[0, i, :]
            if token[1] == ore_feature_id:
                initial_ore_count = token[2]
            elif token[1] == wood_feature_id:
                initial_wood_count = token[2]

        # Verify initial inventory
        assert initial_ore_count == 5, f"Expected initial ore to be 5, got {initial_ore_count}"
        assert initial_wood_count == 3, f"Expected initial wood to be 3, got {initial_wood_count}"

        # Get agent position
        agent_pos = get_agent_position(env, 0)

        # Move east
        move_action_idx = action_index(env, "move", Orientation.EAST)
        move_action = np.array([move_action_idx], dtype=dtype_actions)

        obs_after, _rewards, _dones, _truncs, _infos = env.step(move_action)
        action_success = env.action_success()[0]

        # Check new position
        new_pos = get_agent_position(env, 0)
        position_changed = new_pos != agent_pos

        # Get final inventory counts
        final_ore_count = None
        final_wood_count = None

        for i in range(obs_after.shape[1]):
            token = obs_after[0, i, :]
            if token[1] == ore_feature_id:
                final_ore_count = token[2]
            elif token[1] == wood_feature_id:
                final_wood_count = token[2]

        # Verify resource consumption
        if action_success and position_changed:
            assert final_ore_count == initial_ore_count - 1
            assert final_wood_count == initial_wood_count
        else:
            assert final_ore_count == initial_ore_count
            assert final_wood_count == initial_wood_count


class TestActionSpace:
    """Tests for action space properties."""

    def test_action_space_shape(self, basic_config, simple_map):
        """Test action space dimensions."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)

        action_space = env.action_space

        # Should be Discrete with one dimension
        from gymnasium import spaces

        assert isinstance(action_space, spaces.Discrete), "Action space should be Discrete"

        action_names = env.action_names()
        assert action_space.n == len(action_names)
        assert len(set(action_names)) == len(action_names)

    def test_single_action_space(self, basic_config, multi_agent_map):
        """Test action space for multi-agent environment."""
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=3,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=basic_config.agent,
            actions=basic_config.actions,
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        env = MettaGrid(from_mettagrid_config(config), multi_agent_map, 42)

        # Check that action_space exists
        assert hasattr(env, "action_space")

        action_space = env.action_space
        assert action_space is not None

        from gymnasium import spaces

        # The action space should be Discrete for each agent's action
        assert isinstance(action_space, spaces.Discrete)
        action_names = env.action_names()
        assert action_space.n == len(action_names)
        assert len(set(action_names)) == len(action_names)

        # When stepping, we need to provide actions for all agents
        env.reset()

        # Create actions for all 3 agents using the noop label
        noop_idx = action_names.index("noop")
        actions = np.full(env.num_agents, noop_idx, dtype=dtype_actions)

        # This should work without error
        env.step(actions)

        # Verify we get results for all agents
        assert len(env.action_success()) == 3, "Should get action success for all agents"


class TestSpecialActions:
    """Tests for special action types."""

    def test_attack_action_registration(self, basic_config, simple_map):
        """Test that attack action is properly registered when enabled."""
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=basic_config.agent,
            actions=ActionsConfig(
                attack=AttackActionConfig(
                    enabled=True, required_resources={}, consumed_resources={}, defense_resources={}
                ),
                move=ActionConfig(),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        action_names = env.action_names()

        # Attack variants should be present
        attack_actions = [name for name in action_names if name.startswith("attack_")]
        assert len(attack_actions) == 9, f"Expected 9 attack variants, found {attack_actions}"

        orientations = ["north", "south", "west", "east"]
        if basic_config.allow_diagonals:
            orientations.extend(["northwest", "northeast", "southwest", "southeast"])

        expected = ["noop"]
        expected.extend([f"move_{name}" for name in orientations])
        expected.extend([f"rotate_{name}" for name in orientations])
        expected.extend([f"attack_{i}" for i in range(9)])

        assert action_names == expected

    def test_swap_action_registration(self, basic_config, simple_map):
        """Test that swap action is properly registered when enabled."""
        config = GameConfig(
            resource_names=basic_config.resource_names,
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=basic_config.agent,
            actions=ActionsConfig(
                swap=ActionConfig(),
                move=ActionConfig(),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        action_names = env.action_names()

        assert "swap" in action_names


class TestResourceOrdering:
    """Tests for inventory item ordering effects."""

    def test_resource_order(self, basic_config, simple_map):
        """Test that resources maintain their order."""
        # Config with ore first
        config1 = GameConfig(
            resource_names=["ore", "wood"],
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=basic_config.agent,
            actions=basic_config.actions,
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        # Config with wood first
        config2 = GameConfig(
            resource_names=["wood", "ore"],
            num_agents=basic_config.num_agents,
            max_steps=basic_config.max_steps,
            obs_width=basic_config.obs_width,
            obs_height=basic_config.obs_height,
            num_observation_tokens=basic_config.num_observation_tokens,
            agent=basic_config.agent,
            actions=basic_config.actions,
            objects=basic_config.objects,
            allow_diagonals=basic_config.allow_diagonals,
        )

        env1 = MettaGrid(from_mettagrid_config(config1), simple_map, 42)
        env2 = MettaGrid(from_mettagrid_config(config2), simple_map, 42)

        assert env1.resource_names() == ["ore", "wood"]
        assert env2.resource_names() == ["wood", "ore"]

        # This affects resource indices in the implementation
        # ore is index 0 in env1, but index 1 in env2
