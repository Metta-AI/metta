"""Main miniscope renderer class."""

import io
import select
import sys
import termios
import time
import tty
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console

from cogames.cogs_vs_clips.glyphs import GLYPH_DATA
from mettagrid import MettaGridEnv
from mettagrid.renderer.renderer import Renderer

from .components import (
    AgentControlComponent,
    AgentInfoComponent,
    GlyphPickerComponent,
    HelpPanelComponent,
    MapComponent,
    MiniscopeComponent,
    ObjectInfoComponent,
    SimControlComponent,
    SymbolsTableComponent,
)
from .miniscope_panel import PanelLayout
from .miniscope_state import MiniscopeState, PlaybackState
from .symbol import DEFAULT_SYMBOL_MAP


class MiniscopeRenderer(Renderer):
    """Emoji-based renderer for MettaGridEnv using component architecture."""

    def __init__(self, interactive: bool = True):
        """Initialize the renderer.

        Args:
            interactive: Ignored, always runs in interactive mode
        """
        # Environment reference
        self._env: Optional[MettaGridEnv] = None

        # Renderer state
        self._state = MiniscopeState()

        # Rich console for rendering
        self._console = Console()

        # Panel layout
        self._panels = PanelLayout(self._console)

        # Components list
        self._components: List[MiniscopeComponent] = []

        # Terminal settings
        self._old_terminal_settings = None
        self._terminal_fd = None

        # Timing
        self._last_frame_time = 0.0

    def on_episode_start(self, env: MettaGridEnv) -> None:
        """Initialize the renderer for a new episode."""
        self._env = env

        # Reset state for new episode
        self._state.reset_for_episode(num_agents=env.num_agents, map_height=env.map_height, map_width=env.map_width)

        # Initialize configuration in state
        self._state.object_type_names = env.object_type_names
        self._state.resource_names = env.resource_names
        self._state.symbol_map = DEFAULT_SYMBOL_MAP.copy()

        # Add custom symbols from game config
        for obj in env.mg_config.game.objects.values():
            self._state.symbol_map[obj.name] = obj.render_symbol

        self._state.glyphs = [g.symbol for g in GLYPH_DATA] if GLYPH_DATA else None

        # Update viewport size first to ensure panels have correct dimensions
        self._update_viewport_size()

        # Create all components with panel layout
        self._components = []

        # Create components - all get the same PanelLayout
        self._components.append(MapComponent(env=env, state=self._state, panels=self._panels))
        self._components.append(SimControlComponent(env=env, state=self._state, panels=self._panels))
        self._components.append(AgentControlComponent(env=env, state=self._state, panels=self._panels))
        self._components.append(AgentInfoComponent(env=env, state=self._state, panels=self._panels))
        self._components.append(ObjectInfoComponent(env=env, state=self._state, panels=self._panels))
        self._components.append(SymbolsTableComponent(env=env, state=self._state, panels=self._panels))
        self._components.append(GlyphPickerComponent(env=env, state=self._state, panels=self._panels))
        self._components.append(HelpPanelComponent(env=env, state=self._state, panels=self._panels))

        # Set up terminal (hide cursor and set up input handling)
        self._setup_terminal()

        # Start live display for flicker-free rendering
        self._panels.start_live()

        # Start paused
        self._state.playback = PlaybackState.PAUSED
        self._last_frame_time = time.time()

    def on_step(
        self,
        current_step: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        """Handle step event."""
        self._state.step_count = current_step
        if self._state.total_rewards is not None:
            self._state.total_rewards += rewards

    def should_continue(self) -> bool:
        """Check if rendering should continue."""
        return self._state.is_running()

    def on_episode_end(self, infos: Dict[str, Any]) -> None:
        """Clean up renderer resources."""
        self._state.playback = PlaybackState.STOPPED
        self._panels.stop_live()
        self._cleanup_terminal()
        self._env = None

    def render(self) -> None:
        """Run the rendering loop until an action is ready or simulation should advance.

        When paused, this loops indefinitely until user takes an action.
        When running, this returns after the frame delay has elapsed.
        """
        assert self._env is not None

        start_time = time.time()
        frame_delay = self._state.get_frame_delay()
        was_paused_last_frame = False

        while True:
            # Clear previous user action before reading new input
            self._state.user_action = None

            # Read user input and store in state
            self._state.user_input = self._get_input()

            # Handle user input for agent controls
            self._handle_user_input()

            # Update viewport size in state
            self._update_viewport_size()

            # Clear panels for new frame
            self._panels.clear_all()

            # Let each component update its panel
            for component in self._components:
                component.update()

            # Display the composed frame
            self._render_display()

            # Clear input after processing
            self._state.user_input = None

            # If user requested to quit, break
            if not self._state.is_running():
                break

            # If we have a manual action ready, return to advance simulation
            if self._state.user_action is not None:
                # Clear should_step after taking the action
                self._state.should_step = False
                break

            # If paused, keep looping until we get a user action or unpause
            if self._state.playback == PlaybackState.PAUSED:
                was_paused_last_frame = True
                time.sleep(1.0 / 60.0)  # Sleep at 60 FPS for smooth interaction
                continue

            # If we just unpaused, reset the start time
            if was_paused_last_frame:
                start_time = time.time()
                frame_delay = self._state.get_frame_delay()
                was_paused_last_frame = False

            # If running, check if enough time has elapsed to advance simulation
            elapsed = time.time() - start_time
            if elapsed >= frame_delay:
                break

            # Sleep to maintain target display FPS
            time.sleep(1.0 / 60.0)

    def _handle_user_input(self) -> None:
        """Handle user input by delegating to components."""
        assert self._env is not None
        if not self._state.user_input:
            return

        ch = self._state.user_input

        # Handle help separately (shown in special screen)
        if ch == "?":
            return

        # Let each component handle the input
        for component in self._components:
            if component.handle_input(ch):
                break  # Stop after first component handles it

    def _render_display(self) -> None:
        """Render the panel layout to the terminal."""
        if not self._env:
            return

        # Render using Rich Console API
        self._panels.render_to_console()

    def _show_help_screen(self, help_lines: List[str]) -> None:
        """Show the help screen and wait for input."""
        self._console.clear()
        for line in help_lines:
            self._console.print(line)

        # Wait for key press
        if self._terminal_fd:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                sys.stdin.read(1)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _get_input(self) -> Optional[str]:
        """Get keyboard input if available."""
        if self._terminal_fd is None:
            return None

        # Check for input
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def _should_step_simulation(self) -> bool:
        """Check if simulation should step."""
        if self._state.should_step:
            return True

        if self._state.playback == PlaybackState.RUNNING:
            current_time = time.time()
            frame_delay = self._state.get_frame_delay()
            time_elapsed = current_time - self._last_frame_time
            return time_elapsed >= frame_delay

        return False

    def _sleep_for_fps(self) -> None:
        """Sleep to maintain target FPS."""
        if self._state.playback == PlaybackState.RUNNING:
            current_time = time.time()
            frame_delay = self._state.get_frame_delay()
            time_elapsed = current_time - self._last_frame_time

            if time_elapsed < frame_delay:
                time.sleep(frame_delay - time_elapsed)
        elif self._state.playback == PlaybackState.PAUSED:
            # Sleep a bit when paused to avoid busy waiting
            time.sleep(0.05)

    def _update_viewport_size(self) -> None:
        """Update viewport size in state based on terminal size."""
        try:
            import shutil

            terminal_size = shutil.get_terminal_size()
            viewport_height = max(5, terminal_size.lines - 6)
            side_panel_width = 46
            spacing = 2
            available_width = max(10, terminal_size.columns - side_panel_width - spacing)
            viewport_width = available_width // 2
        except Exception:
            viewport_height = 20
            viewport_width = 40

        self._state.viewport_height = viewport_height
        self._state.viewport_width = viewport_width

        # Update panel dimensions
        self._panels.map_view.width = viewport_width * 2  # Each cell takes 2 chars
        self._panels.map_view.height = viewport_height

    def get_user_actions(self) -> dict[int, tuple[int, int]]:
        """Get the current user actions for manually controlled agents.

        Returns:
            Dictionary mapping agent_id to (action_id, action_param).
            Empty dict if no manual actions are set.
        """
        actions = {}

        # If there's a manual action for the selected agent, return it
        if self._state.user_action is not None and self._state.selected_agent is not None:
            actions[self._state.selected_agent] = self._state.user_action
            # Clear the action after returning it
            self._state.user_action = None

        return actions

    def _setup_terminal(self) -> None:
        """Set up terminal for interactive mode."""
        try:
            self._terminal_fd = sys.stdin.fileno()
            self._old_terminal_settings = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            self._console.show_cursor(False)
        except (OSError, io.UnsupportedOperation, termios.error):
            # stdin is not available (e.g., in tests or when redirected)
            self._terminal_fd = None
            self._old_terminal_settings = None

    def _cleanup_terminal(self) -> None:
        """Restore terminal settings."""
        if self._terminal_fd and self._old_terminal_settings:
            termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._old_terminal_settings)
        self._console.show_cursor(True)

    def __del__(self):
        """Clean up on destruction."""
        self._cleanup_terminal()
