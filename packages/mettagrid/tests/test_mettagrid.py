import pytest

import mettagrid.config.mettagrid_config
import mettagrid.map_builder.random
import mettagrid.mettagrid_c
import mettagrid.simulator
import mettagrid.test_support

NUM_OBS_TOKENS = 50


@pytest.fixture
def basic_env() -> mettagrid.simulator.Simulation:
    """Create a basic test environment with 8x4 grid and 2 agents."""
    cfg = mettagrid.config.mettagrid_config.MettaGridConfig(
        game=mettagrid.config.mettagrid_config.GameConfig(
            num_agents=2,
            obs=mettagrid.config.mettagrid_config.ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
            actions=mettagrid.config.mettagrid_config.ActionsConfig(
                noop=mettagrid.config.mettagrid_config.NoopActionConfig(),
                move=mettagrid.config.mettagrid_config.MoveActionConfig(),
            ),
            map_builder=mettagrid.map_builder.random.RandomMapBuilder.Config(
                width=8,
                height=4,
                agents=2,
                seed=42,  # For deterministic test results
            ),
        )
    )
    return mettagrid.simulator.Simulation(cfg)


class TestBasicFunctionality:
    """Test basic environment functionality."""

    def test_environment_initialization(self, basic_env: mettagrid.simulator.Simulation):
        """Test basic environment properties."""
        assert basic_env.map_width == 8
        assert basic_env.map_height == 4
        assert len(basic_env.action_names) > 0
        assert "noop" in basic_env.action_names

        obs = basic_env._c_sim.observations()

        assert obs.shape == (basic_env.num_agents, NUM_OBS_TOKENS, mettagrid.test_support.TokenTypes.OBS_TOKEN_SIZE)

    def test_grid_hash(self, basic_env: mettagrid.simulator.Simulation):
        """Test grid hash consistency."""
        assert basic_env.initial_grid_hash == 14602406112020495965  # Updated for RandomMapBuilder with seed=42

    def test_action_interface(self, basic_env: mettagrid.simulator.Simulation):
        """Test action interface and basic action execution."""

        action_names = basic_env.action_names
        assert "noop" in action_names
        assert any(name.startswith("move") for name in action_names)

        for agent_id in range(basic_env.num_agents):
            basic_env.agent(agent_id).set_action("noop")

        basic_env.step()
        obs = basic_env._c_sim.observations()
        rewards = basic_env._c_sim.rewards()
        terminals = basic_env._c_sim.terminals()
        truncations = basic_env._c_sim.truncations()
        info = {}

        # Check shapes and types
        assert obs.shape == (basic_env.num_agents, NUM_OBS_TOKENS, mettagrid.test_support.TokenTypes.OBS_TOKEN_SIZE)
        assert rewards.shape == (basic_env.num_agents,)
        assert terminals.shape == (basic_env.num_agents,)
        assert truncations.shape == (basic_env.num_agents,)
        assert isinstance(info, dict)

        # Action success should be boolean and per-agent
        action_success = [basic_env.agent(i).last_action_success for i in range(basic_env.num_agents)]
        assert len(action_success) == basic_env.num_agents
        assert all(isinstance(x, bool) for x in action_success)

    def test_environment_state_consistency(self, basic_env: mettagrid.simulator.Simulation):
        """Test that environment state remains consistent across operations."""
        initial_objects = basic_env.grid_objects()

        for agent_id in range(basic_env.num_agents):
            basic_env.agent(agent_id).set_action("noop")

        basic_env.step()
        post_step_objects = basic_env.grid_objects()

        # Object count should remain unchanged
        assert len(initial_objects) == len(post_step_objects)

        # Dimensions should remain unchanged
        assert basic_env.map_width == 8
        assert basic_env.map_height == 4

        # Action set should remain unchanged after stepping
        actions1 = basic_env.action_names
        for agent_id in range(basic_env.num_agents):
            basic_env.agent(agent_id).set_action("noop")
        basic_env.step()
        actions2 = basic_env.action_names
        assert actions1 == actions2


class TestPackedCoordinate:
    """Test PackedCoordinate functionality."""

    def test_packed_coordinate(self):
        """Test the PackedCoordinate functionality."""
        # Test constants
        assert mettagrid.mettagrid_c.PackedCoordinate.MAX_PACKABLE_COORD == 14

        # Test all valid coordinates
        successfully_packed = 0
        for row in range(15):  # 0-14
            for col in range(15):  # 0-14
                packed = mettagrid.mettagrid_c.PackedCoordinate.pack(row, col)
                unpacked = mettagrid.mettagrid_c.PackedCoordinate.unpack(packed)
                assert unpacked == (row, col), f"Roundtrip failed for ({row}, {col})"
                successfully_packed += 1

        # Verify we can pack 225 positions (15x15 grid)
        assert successfully_packed == 225, f"Expected 225 packable positions, got {successfully_packed}"

        # Test empty/0xFF handling
        assert mettagrid.mettagrid_c.PackedCoordinate.is_empty(0xFF)
        assert mettagrid.mettagrid_c.PackedCoordinate.unpack(0xFF) is None
        assert not mettagrid.mettagrid_c.PackedCoordinate.is_empty(0x00)
        assert not mettagrid.mettagrid_c.PackedCoordinate.is_empty(0xE0)
        assert not mettagrid.mettagrid_c.PackedCoordinate.is_empty(0xEE)

        # Test invalid coordinates
        invalid_coords = [(15, 0), (0, 15), (15, 15), (16, 0), (0, 16), (255, 255)]
        for row, col in invalid_coords:
            try:
                mettagrid.mettagrid_c.PackedCoordinate.pack(row, col)
                raise AssertionError(f"Should have raised exception for ({row}, {col})")
            except ValueError:
                pass  # Expected
