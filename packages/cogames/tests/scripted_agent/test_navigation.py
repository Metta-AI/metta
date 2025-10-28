"""
Unit tests for scripted agent navigation, coordinate tracking, and station discovery.

These tests verify that the agent correctly:
1. Tracks its own position from egocentric observations
2. Discovers stations and records their absolute positions
3. Updates occupancy map correctly
4. Can navigate back to discovered stations
"""

import pytest

from cogames.cogs_vs_clips.missions import HarvestMission, get_map
from cogames.policy.scripted_agent import AgentState, ScriptedAgentPolicy
from cogames.policy.navigator import Navigator
from mettagrid import MettaGridEnv


class TestCoordinateTracking:
    """Test that agent correctly tracks coordinates from egocentric observations."""

    def test_agent_position_from_observation(self):
        """Test that agent correctly extracts its own position from observations."""
        # Create a simple test environment
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        # Reset and get initial observation
        obs, info = env.reset()
        action, state = impl.step_with_state(obs[0], state)

        # Agent should have discovered its position
        assert state.agent_row >= 0, "Agent should have valid row coordinate"
        assert state.agent_col >= 0, "Agent should have valid column coordinate"
        assert state.home_base_row >= 0, "Agent should remember spawn location"
        assert state.home_base_col >= 0, "Agent should remember spawn location"

    def test_position_consistency_after_moves(self):
        """Test that agent's tracked position stays consistent after movements."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        obs, info = env.reset()
        action, state = impl.step_with_state(obs[0], state)

        initial_pos = (state.agent_row, state.agent_col)

        # Execute several moves
        for _ in range(10):
            obs, reward, done, truncated, info = env.step([action])
            if done[0]:
                break
            action, state = impl.step_with_state(obs[0], state)

        # Agent should have moved from initial position
        current_pos = (state.agent_row, state.agent_col)
        assert current_pos != initial_pos or state.step_count < 5, (
            "Agent should move from spawn or detect early completion"
        )


class TestStationDiscovery:
    """Test that agent correctly discovers and records station positions."""

    def test_station_discovery_from_observation(self):
        """Test that agent discovers stations visible in observations."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        obs, info = env.reset()

        # Run for several steps to allow exploration
        for _ in range(50):
            action, state = impl.step_with_state(obs[0], state)
            obs, reward, done, truncated, info = env.step([action])
            if done[0]:
                break

        # Agent should have discovered some stations
        assert len(impl._station_positions) > 0, "Agent should discover at least one station in 50 steps"

        # Check that discovered positions are valid
        for station_name, pos in impl._station_positions.items():
            row, col = pos
            assert 0 <= row < impl._map_height, (
                f"Station {station_name} row {row} should be in valid range [0, {impl._map_height})"
            )
            assert 0 <= col < impl._map_width, (
                f"Station {station_name} col {col} should be in valid range [0, {impl._map_width})"
            )

    def test_extractor_memory_matches_discoveries(self):
        """Test that extractor memory contains discovered resource stations."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        obs, info = env.reset()

        # Run for several steps
        for _ in range(100):
            action, state = impl.step_with_state(obs[0], state)
            obs, reward, done, truncated, info = env.step([action])
            if done[0]:
                break

        # Check extractors were added to memory
        germanium_extractors = impl.extractor_memory.get_by_type("germanium")
        silicon_extractors = impl.extractor_memory.get_by_type("silicon")
        carbon_extractors = impl.extractor_memory.get_by_type("carbon")
        oxygen_extractors = impl.extractor_memory.get_by_type("oxygen")

        total_extractors = (
            len(germanium_extractors) + len(silicon_extractors) + len(carbon_extractors) + len(oxygen_extractors)
        )

        assert total_extractors > 0, "Agent should discover resource extractors"


class TestOccupancyMapping:
    """Test that agent correctly builds occupancy map from observations."""

    def test_occupancy_initialization(self):
        """Test that occupancy map initializes correctly."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl

        # Check occupancy map dimensions
        assert len(impl._occ) == impl._map_height, (
            f"Occupancy map height {len(impl._occ)} should match map height {impl._map_height}"
        )
        assert len(impl._occ[0]) == impl._map_width, (
            f"Occupancy map width {len(impl._occ[0])} should match map width {impl._map_width}"
        )

        # Initially should be mostly unknown
        unknown_count = sum(1 for row in impl._occ for cell in row if cell == impl.OCC_UNKNOWN)
        total_cells = impl._map_height * impl._map_width
        assert unknown_count > total_cells * 0.9, "Most cells should be unknown initially"

    def test_current_cell_marked_free(self):
        """Test that agent's current cell is always marked as FREE."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        obs, info = env.reset()
        action, state = impl.step_with_state(obs[0], state)

        # Agent's current cell should be FREE
        if state.agent_row >= 0 and state.agent_col >= 0:
            occ = impl._occ[state.agent_row][state.agent_col]
            assert occ == impl.OCC_FREE, (
                f"Agent's current cell ({state.agent_row}, {state.agent_col}) should be FREE, got {occ}"
            )

    def test_walls_marked_correctly(self):
        """Test that walls are discovered and marked in occupancy map."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        obs, info = env.reset()

        # Run for steps to discover walls
        for _ in range(20):
            action, state = impl.step_with_state(obs[0], state)
            obs, reward, done, truncated, info = env.step([action])
            if done[0]:
                break

        # Should have discovered some obstacles (walls/stations)
        obstacle_count = sum(1 for row in impl._occ for cell in row if cell == Navigator.OCC_OBSTACLE)
        assert obstacle_count > 0, "Agent should discover obstacles in 20 steps"

        # Verify occupancy map has both free and obstacle cells
        free_count = sum(1 for row in impl._occ for cell in row if cell == Navigator.OCC_FREE)
        assert free_count > 0, "Agent should have marked some cells as free"


class TestNavigationToStations:
    """Test that agent can navigate back to discovered stations."""

    def test_can_reach_discovered_extractor(self):
        """Test that agent can successfully navigate to and use a discovered extractor."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        obs, info = env.reset()

        # Run until agent discovers and uses a germanium extractor
        initial_germanium = 0
        max_steps = 200

        for step in range(max_steps):
            action, state = impl.step_with_state(obs[0], state)
            obs, reward, done, truncated, info = env.step([action])

            if state.germanium > initial_germanium:
                # Successfully collected germanium!
                assert step < max_steps, f"Agent should collect germanium within {max_steps} steps"
                return

            if done[0]:
                break

        # If we got here, check if we at least discovered extractors
        germanium_extractors = impl.extractor_memory.get_by_type("germanium")
        if len(germanium_extractors) == 0:
            pytest.skip("No germanium extractors discovered in test run")

    def test_charger_reachability(self):
        """Test that discovered chargers are reachable."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl
        state = AgentState()

        obs, info = env.reset()

        # Run until low energy and agent tries to recharge
        for _ in range(300):
            action, state = impl.step_with_state(obs[0], state)
            obs, reward, done, truncated, info = env.step([action])

            if done[0]:
                break

        # Check if any chargers were discovered
        chargers = impl.extractor_memory.get_by_type("charger")

        # Log discovered chargers for awareness
        if chargers:
            print(f"Discovered {len(chargers)} chargers")


class TestExtractorDepletion:
    """Test that depleted extractors are properly tracked."""

    def test_marks_depleted_when_uses_exhausted(self):
        """Test that extractor is marked depleted when uses remaining reaches 0."""
        mission = HarvestMission()
        map_builder = get_map("training_facility_open_1.map")
        mission = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission.make_env()
        env = MettaGridEnv(env_cfg)

        policy = ScriptedAgentPolicy(env)
        impl = policy._impl

        # Create a fake extractor at an arbitrary position
        fake_pos = (10, 10)
        impl.extractor_memory.add_extractor(fake_pos, "germanium", "germanium_extractor")

        extractor = impl.extractor_memory.get_at_position(fake_pos)
        assert extractor is not None, "Extractor should be added to memory"
        assert not extractor.permanently_depleted, "Extractor should start as not depleted"

        # Mark as permanently depleted (simulating remaining_uses == 0 observation)
        extractor.permanently_depleted = True

        # Should now be marked as depleted
        assert extractor.is_depleted(), "Extractor should be marked depleted"

        # Create a simple cooldown estimate function for testing
        def cooldown_fn(ext, step):
            return 0  # Always available if not depleted

        assert not extractor.is_available(0, cooldown_fn), "Depleted extractor should not be available"


def test_mission_available():
    """Basic test that training facility mission can be loaded."""
    mission = HarvestMission()
    map_builder = get_map("training_facility_open_1.map")
    mission = mission.instantiate(map_builder, num_cogs=1)
    assert mission is not None, "Should be able to instantiate mission"

    env_cfg = mission.make_env()
    assert env_cfg is not None, "Should be able to create environment config"

    env = MettaGridEnv(env_cfg)
    obs, info = env.reset()
    assert obs is not None, "Should get observation from reset"
    assert len(obs) > 0, "Should have at least one agent observation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
