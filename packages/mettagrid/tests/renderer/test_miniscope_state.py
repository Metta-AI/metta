"""Tests for MiniscopeState class."""

import numpy as np
import pytest

from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, PlaybackState, RenderMode


class TestMiniscopeState:
    """Test suite for MiniscopeState functionality."""

    def test_state_initialization(self):
        """Test MiniscopeState default initialization."""
        state = MiniscopeState()

        # Check default playback state
        assert state.playback == PlaybackState.STOPPED
        assert state.fps == 4.0
        assert state.step_count == 0
        assert state.max_steps is None

        # Check default camera and viewport
        assert state.camera_row == 0
        assert state.camera_col == 0
        assert state.viewport_height == 20
        assert state.viewport_width == 40

        # Check default mode and selection
        assert state.mode == RenderMode.FOLLOW
        assert state.selected_agent == 0
        assert state.cursor_row == 0
        assert state.cursor_col == 0

        # Check agent control
        assert len(state.manual_agents) == 0
        assert state.user_action is None
        assert state.should_step is False

        # Check user input
        assert state.user_input is None

        # Check rewards
        assert state.total_rewards is None

        # Check map bounds
        assert state.min_row == 0
        assert state.min_col == 0
        assert state.map_height == 0
        assert state.map_width == 0

        # Check shared data
        assert state.resource_names is None
        assert state.symbol_map is None
        assert state.vibes is None

    def test_is_running(self):
        """Test is_running method for different playback states."""
        state = MiniscopeState()

        state.playback = PlaybackState.STOPPED
        assert not state.is_running()

        state.playback = PlaybackState.RUNNING
        assert state.is_running()

        state.playback = PlaybackState.PAUSED
        assert state.is_running()

        state.playback = PlaybackState.STEPPING
        assert state.is_running()

    def test_should_render_frame(self):
        """Test should_render_frame method."""
        state = MiniscopeState()

        # Should not render when stopped
        state.playback = PlaybackState.STOPPED
        state.should_step = False
        assert not state.should_render_frame()

        # Should render when running
        state.playback = PlaybackState.RUNNING
        assert state.should_render_frame()

        # Should not render when paused (unless should_step)
        state.playback = PlaybackState.PAUSED
        assert not state.should_render_frame()

        # Should render when should_step is True
        state.should_step = True
        assert state.should_render_frame()

    def test_toggle_pause(self):
        """Test toggle_pause method."""
        state = MiniscopeState()

        # Start paused
        state.playback = PlaybackState.PAUSED
        state.toggle_pause()
        assert state.playback == PlaybackState.RUNNING

        # Toggle back to paused
        state.toggle_pause()
        assert state.playback == PlaybackState.PAUSED

        # From other states, should not toggle
        state.playback = PlaybackState.STOPPED
        state.toggle_pause()
        assert state.playback == PlaybackState.STOPPED  # No change

    def test_speed_controls(self):
        """Test increase_speed and decrease_speed methods."""
        state = MiniscopeState()
        state.fps = 4.0

        # Increase speed
        state.increase_speed()
        assert state.fps == pytest.approx(6.0)  # 4.0 * 1.5

        # Increase again
        state.increase_speed()
        assert state.fps == pytest.approx(9.0)  # 6.0 * 1.5

        # Test max limit
        state.fps = 400.0
        state.increase_speed()
        assert state.fps == 600.0  # Max is 600

        # Decrease speed
        state.fps = 6.0
        state.decrease_speed()
        assert state.fps == pytest.approx(4.0)  # 6.0 / 1.5

        # Test min limit
        state.fps = 0.015
        state.decrease_speed()
        assert state.fps == 0.01  # Min is 0.01

    def test_get_frame_delay(self):
        """Test get_frame_delay method."""
        state = MiniscopeState()

        state.fps = 4.0
        assert state.get_frame_delay() == 0.25  # 1/4

        state.fps = 10.0
        assert state.get_frame_delay() == 0.1  # 1/10

        state.fps = 60.0
        assert state.get_frame_delay() == pytest.approx(1 / 60)

        # Edge case: 0 fps
        state.fps = 0
        assert state.get_frame_delay() == 0.25  # Default

    def test_set_mode(self):
        """Set render mode directly without cycling."""
        state = MiniscopeState()

        # Follow is default; switching to PAN and SELECT works directly
        state.set_mode(RenderMode.PAN)
        assert state.mode == RenderMode.PAN

        state.set_mode(RenderMode.SELECT)
        assert state.mode == RenderMode.SELECT

        # Trying to set helper modes should be ignored
        state.set_mode(RenderMode.VIBE_PICKER)
        assert state.mode == RenderMode.SELECT

        state.set_mode(RenderMode.HELP)
        assert state.mode == RenderMode.SELECT

    def test_enter_vibe_picker(self):
        """Test enter_vibe_picker method."""
        state = MiniscopeState()

        state.mode = RenderMode.FOLLOW
        state.enter_vibe_picker()
        assert state.mode == RenderMode.VIBE_PICKER

    def test_toggle_manual_control(self):
        """Test toggle_manual_control method."""
        state = MiniscopeState()

        # Initially no manual agents
        assert 0 not in state.manual_agents

        # Add agent 0 to manual control
        state.toggle_manual_control(0)
        assert 0 in state.manual_agents

        # Toggle off
        state.toggle_manual_control(0)
        assert 0 not in state.manual_agents

        # Add multiple agents
        state.toggle_manual_control(0)
        state.toggle_manual_control(1)
        state.toggle_manual_control(2)
        assert state.manual_agents == {0, 1, 2}

    def test_select_agents(self):
        """Test select_next_agent and select_previous_agent methods."""
        state = MiniscopeState()
        num_agents = 4

        # Start with agent 0
        assert state.selected_agent == 0

        # Select next
        state.select_next_agent(num_agents)
        assert state.selected_agent == 1

        state.select_next_agent(num_agents)
        assert state.selected_agent == 2

        # Wrap around
        state.selected_agent = 3
        state.select_next_agent(num_agents)
        assert state.selected_agent == 0

        # Select previous
        state.select_previous_agent(num_agents)
        assert state.selected_agent == 3  # Wrap around

        state.selected_agent = 2
        state.select_previous_agent(num_agents)
        assert state.selected_agent == 1

        # Handle None case
        state.selected_agent = None
        state.select_next_agent(num_agents)
        assert state.selected_agent == 0

        state.selected_agent = None
        state.select_previous_agent(num_agents)
        assert state.selected_agent == 0

    def test_move_camera(self):
        """Test move_camera method with bounds checking."""
        state = MiniscopeState()
        state.set_bounds(0, 0, 10, 10)  # 10x10 map

        # Start at center
        state.camera_row = 5
        state.camera_col = 5

        # Move within bounds
        state.move_camera(2, 3)
        assert state.camera_row == 7
        assert state.camera_col == 8

        # Try to move beyond bounds
        state.move_camera(10, 10)
        assert state.camera_row == 9  # Clamped to max
        assert state.camera_col == 9  # Clamped to max

        # Try to move below bounds
        state.camera_row = 2
        state.camera_col = 2
        state.move_camera(-5, -5)
        assert state.camera_row == 0  # Clamped to min
        assert state.camera_col == 0  # Clamped to min

    def test_move_cursor(self):
        """Test move_cursor method with bounds checking."""
        state = MiniscopeState()
        state.set_bounds(0, 0, 10, 10)

        # Start at center
        state.cursor_row = 5
        state.cursor_col = 5

        # Move within bounds
        state.move_cursor(1, -2)
        assert state.cursor_row == 6
        assert state.cursor_col == 3

        # Try to move beyond bounds
        state.cursor_row = 8
        state.cursor_col = 8
        state.move_cursor(5, 5)
        assert state.cursor_row == 9  # Clamped
        assert state.cursor_col == 9  # Clamped

    def test_set_bounds(self):
        """Test set_bounds method."""
        state = MiniscopeState()

        # Set bounds
        state.set_bounds(5, 10, 20, 30)
        assert state.min_row == 5
        assert state.min_col == 10
        assert state.map_height == 20
        assert state.map_width == 30

        # Camera and cursor should be clamped to bounds
        state.camera_row = 50
        state.camera_col = 50
        state.cursor_row = 50
        state.cursor_col = 50

        state.set_bounds(0, 0, 10, 10)
        assert state.camera_row == 9  # Clamped to height-1
        assert state.camera_col == 9  # Clamped to width-1
        assert state.cursor_row == 9
        assert state.cursor_col == 9

    def test_reset_for_episode(self):
        """Test reset_for_episode method."""
        state = MiniscopeState()

        # Set some non-default values
        state.step_count = 100
        state.playback = PlaybackState.RUNNING
        state.mode = RenderMode.SELECT
        state.selected_agent = 5
        state.manual_agents.add(1)
        state.manual_agents.add(2)
        state.user_action = 3
        state.should_step = True

        # Reset for new episode
        state.reset_for_episode(num_agents=3, map_height=15, map_width=20)

        # Check reset values
        assert state.step_count == 0
        assert state.playback == PlaybackState.PAUSED
        assert state.mode == RenderMode.FOLLOW
        assert state.selected_agent == 0  # First agent
        assert len(state.manual_agents) == 0
        assert state.user_action is None
        assert state.should_step is False

        # Check rewards initialized
        assert state.total_rewards is not None
        assert len(state.total_rewards) == 3
        assert np.all(state.total_rewards == 0)

        # Check camera centered
        assert state.camera_row == 7  # 15 // 2
        assert state.camera_col == 10  # 20 // 2
        assert state.cursor_row == 7
        assert state.cursor_col == 10

        # Test with 0 agents
        state.reset_for_episode(num_agents=0, map_height=10, map_width=10)
        assert state.selected_agent is None
        assert state.total_rewards is None

    def test_set_bounds_with_offset(self):
        """Test set_bounds with non-zero min values."""
        state = MiniscopeState()

        # Set bounds with offset
        state.set_bounds(10, 20, 5, 5)  # Starting at (10, 20), 5x5 size

        # Camera should be clamped within these bounds
        state.camera_row = 0
        state.camera_col = 0
        state.set_bounds(10, 20, 5, 5)

        # Camera should be clamped to minimum bounds
        assert state.camera_row >= 10
        assert state.camera_col >= 20

        # And not exceed maximum
        assert state.camera_row < 10 + 5
        assert state.camera_col < 20 + 5
