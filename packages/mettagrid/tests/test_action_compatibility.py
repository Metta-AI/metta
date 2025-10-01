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
    MettaGridConfig,
    WallConfig,
)
from mettagrid.envs.mettagrid_env import MettaGridEnv
from mettagrid.mettagrid_c import MettaGrid, dtype_actions
from mettagrid.test_support.actions import get_agent_position, get_current_observation


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
        assert action_names1 == ["noop", "move", "rotate"]

    def test_action_indices_consistency(self, basic_config, simple_map):
        """Test that action indices remain consistent."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        action_names = env.action_names()

        # Verify indices (noop is first when enabled)
        assert action_names.index("noop") == 0
        assert action_names.index("move") == 1
        assert action_names.index("rotate") == 2


class TestActionValidation:
    """Tests for action validation and error handling."""

    def test_invalid_action_type(self, basic_config, simple_map):
        """Test that invalid action types are handled properly."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        env.reset()

        # Try invalid action type
        invalid_action = np.array([[99, 0]], dtype=dtype_actions)
        env.step(invalid_action)

        # Action should fail
        assert not env.action_success()[0], "Invalid action type should fail"

    def test_invalid_action_argument(self, basic_config, simple_map):
        """Test that invalid action arguments are handled properly."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        env.reset()

        move_idx = env.action_names().index("move")
        max_args = env.max_action_args()

        # Try argument exceeding max_arg
        invalid_arg = max_args[move_idx] + 10
        invalid_action = np.array([[move_idx, invalid_arg]], dtype=dtype_actions)
        env.step(invalid_action)

        # Action should fail
        assert not env.action_success()[0], "Invalid action argument should fail"

    def test_sanitize_actions_clamps_values(self, basic_config, simple_map):
        """Sanitization clamps action type and argument into valid ranges."""

        env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
        env = MettaGridEnv(env_cfg=env_cfg, is_training=True)
        env.reset()

        try:
            invalid = np.array([[999, 999]], dtype=np.int32)
            sanitized = env._sanitize_actions(invalid)

            assert sanitized.dtype == np.int32
            assert sanitized.shape == invalid.shape

            action_type = int(sanitized[0, 0])
            assert 0 <= action_type < len(env.action_names)

            max_args = env.max_action_args
            assert 0 <= sanitized[0, 1] <= max_args[action_type]
        finally:
            env.close()

    def test_max_action_args(self, basic_config, simple_map):
        """Test max_action_args for different actions."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)

        action_names = env.action_names()
        max_args = env.max_action_args()

        # Check expected max args
        move_idx = action_names.index("move")
        noop_idx = action_names.index("noop")
        rotate_idx = action_names.index("rotate")

        assert basic_config.allow_diagonals, "tests assume diagonals are allowed"
        assert max_args[move_idx] == 7, "Move should have max_arg=7 (8 orientations)"
        assert max_args[noop_idx] == 0, "Noop should have max_arg=0"
        assert max_args[rotate_idx] == 7, "Rotate should have max_arg=7 (8 orientations)"


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

        move_idx = env.action_names().index("move")
        move_action = np.array([[move_idx, 0]], dtype=dtype_actions)

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

        # Move east (direction 2)
        move_idx = env.action_names().index("move")
        move_action = np.array([[move_idx, 2]], dtype=dtype_actions)

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

        # Should be MultiDiscrete with 2 dimensions
        assert hasattr(action_space, "nvec"), "Action space should be MultiDiscrete"
        assert len(action_space.nvec) == 2, "Action space should have 2 dimensions"

        num_actions = action_space.nvec[0]
        max_arg_plus_one = action_space.nvec[1]

        # Should match our configuration
        assert num_actions == len(env.action_names())

        # Max arg is the maximum across all actions
        max_args = env.max_action_args()
        assert max_arg_plus_one == max(max_args) + 1

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

        # The action space should be MultiDiscrete for each agent's action
        assert hasattr(action_space, "nvec"), "Action space should be MultiDiscrete"
        assert len(action_space.nvec) == 2, "Action space should have 2 dimensions (action_type, action_arg)"

        # First dimension is number of action types
        num_actions = action_space.nvec[0]
        assert num_actions == len(env.action_names())

        # Second dimension is max action argument + 1
        max_arg_plus_one = action_space.nvec[1]
        max_args = env.max_action_args()
        assert max_arg_plus_one == max(max_args) + 1

        # When stepping, we need to provide actions for all agents
        env.reset()

        # Create actions for all 3 agents (noop for each)
        noop_idx = env.action_names().index("noop")
        actions = np.array([[noop_idx, 0]] * 3, dtype=dtype_actions)

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

        # Attack should be present
        assert "attack" in action_names

        # Check the expected order (noop is first when enabled, attack comes last)
        expected_actions = ["noop", "move", "rotate", "attack"]
        assert action_names == expected_actions

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
