"""Main miniscope renderer class."""

import io
import select
import shutil
import sys
import termios
import time
import tty
from typing import List, Optional

from rich.console import Console

from mettagrid.config.vibes import VIBES as VIBE_DATA
from mettagrid.renderer.miniscope.components import (
    AgentControlComponent,
    AgentInfoComponent,
    HelpPanelComponent,
    MapComponent,
    MiniscopeComponent,
    ObjectInfoComponent,
    SimControlComponent,
    SymbolsTableComponent,
    VibePickerComponent,
)
from mettagrid.renderer.renderer import Renderer

from .miniscope_panel import LAYOUT_PADDING, RESERVED_VERTICAL_LINES, SIDEBAR_WIDTH, PanelLayout
from .miniscope_state import MiniscopeState, PlaybackState, RenderMode
from .symbol import DEFAULT_SYMBOL_MAP


class MiniscopeRenderer(Renderer):
    """Emoji-based renderer for MettaGridEnv using component architecture."""

    def __init__(self, interactive: bool = True):
        """Initialize the renderer.

        Args:
            interactive: Ignored, always runs in interactive mode
        """
        super().__init__()

        # Renderer state
        self._state = MiniscopeState()

        # Rich console for rendering - reduce size by 1 to prevent wrapping
        try:
            term_size = shutil.get_terminal_size()
            if term_size.columns > 0 and term_size.lines > 0:
                self._initial_terminal_columns = term_size.columns
                self._initial_terminal_lines = term_size.lines
            else:
                raise ValueError("Invalid terminal size")
            console_width = max(80, term_size.columns - 1)
            console_height = max(24, term_size.lines - 1)
        except Exception:
            console_width = 119
            console_height = 39
            self._initial_terminal_columns = console_width + 1
            self._initial_terminal_lines = console_height + 1
        self._console = Console(width=console_width, height=console_height)

        # Panel layout
        self._panels = PanelLayout(self._console)

        # Components list
        self._components: List[MiniscopeComponent] = []

        # Terminal settings
        self._old_terminal_settings = None
        self._terminal_fd = None

        # Timing
        self._last_frame_time = 0.0
        self._ema_frame_time: float = 0.0  # Exponential moving average of frame times
        self._ema_alpha: float = 0.2  # Smoothing factor for EMA (higher = more responsive)

        # Sidebar hotkey mapping
        self._sidebar_hotkeys: dict[str, str] = {}

    def on_episode_start(self) -> None:
        """Initialize the renderer for a new episode."""
        # Reset state for new episode
        self._state.reset_for_episode(
            num_agents=self._sim.num_agents,
            map_height=self._sim.map_height,
            map_width=self._sim.map_width,
        )

        # Initialize configuration in state
        self._state.resource_names = self._sim.resource_names
        self._state.symbol_map = DEFAULT_SYMBOL_MAP.copy()

        # Add custom symbols from game config
        for obj in self._sim.config.game.objects.values():
            # Key by render_name (preferred) and also alias by name for convenience
            self._state.symbol_map[obj.render_name or obj.name] = obj.render_symbol
            if obj.render_name and obj.render_name != obj.name:
                self._state.symbol_map[obj.name] = obj.render_symbol

        self._state.vibes = [g.symbol for g in VIBE_DATA] if VIBE_DATA else None

        # Configure viewport once using the initial terminal size
        self._apply_initial_viewport_size()

        # Rebuild sidebar panel stack for this episode
        sidebar_defs = [
            ("1", "agent_info", AgentInfoComponent),
            ("2", "object_info", ObjectInfoComponent),
            ("3", "symbols", SymbolsTableComponent),
        ]
        self._sidebar_hotkeys = {hotkey: name for hotkey, name, _ in sidebar_defs}

        self._panels.reset_sidebar_panels()
        # Register all panels including modal ones
        for _, name, _ in sidebar_defs:
            self._panels.register_sidebar_panel(name)
        self._panels.register_sidebar_panel("vibe_picker")
        self._panels.register_sidebar_panel("help")

        # Initialize sidebar visibility state
        self._state.initialize_sidebar_visibility([name for _, name, _ in sidebar_defs] + ["vibe_picker", "help"])

        # Create all components with panel layout
        self._components = []

        # Create components - all get the same PanelLayout
        self._components.append(MapComponent(sim=self._sim, state=self._state, panels=self._panels))
        self._components.append(SimControlComponent(sim=self._sim, state=self._state, panels=self._panels))
        self._components.append(AgentControlComponent(sim=self._sim, state=self._state, panels=self._panels))
        self._components.append(AgentInfoComponent(sim=self._sim, state=self._state, panels=self._panels))
        self._components.append(ObjectInfoComponent(sim=self._sim, state=self._state, panels=self._panels))
        self._components.append(SymbolsTableComponent(sim=self._sim, state=self._state, panels=self._panels))
        self._components.append(VibePickerComponent(sim=self._sim, state=self._state, panels=self._panels))
        self._components.append(HelpPanelComponent(sim=self._sim, state=self._state, panels=self._panels))

        # Set up terminal (hide cursor and set up input handling)
        self._setup_terminal()

        # Start live display for flicker-free rendering
        self._panels.start_live()

        # Start paused
        self._state.playback = PlaybackState.PAUSED
        self._last_frame_time = time.time()
        self._ema_frame_time = 0.0  # Reset EMA for new episode

    def _update_fps(self, current_time: float) -> None:
        """Update FPS calculation using exponential moving average.

        Args:
            current_time: Current timestamp in seconds
        """
        if self._last_frame_time > 0:
            frame_time = current_time - self._last_frame_time
            # Update exponential moving average
            if self._ema_frame_time == 0:
                # Initialize EMA with first measurement
                self._ema_frame_time = frame_time
            else:
                # Apply EMA formula: EMA = alpha * new_value + (1 - alpha) * previous_EMA
                self._ema_frame_time = self._ema_alpha * frame_time + (1 - self._ema_alpha) * self._ema_frame_time

            # Calculate FPS from average frame time
            self._state.true_fps = 1.0 / self._ema_frame_time if self._ema_frame_time > 0 else 0.0

        self._last_frame_time = current_time

    def on_step(self) -> None:
        """Handle step event."""
        self._state.step_count = self._sim.current_step
        if self._state.total_rewards is not None:
            self._state.total_rewards = self._sim.episode_rewards

    def on_episode_end(self) -> None:
        """Clean up renderer resources."""
        self._state.playback = PlaybackState.STOPPED
        self._panels.stop_live()
        self._cleanup_terminal()

    def render(self) -> None:
        """Run the rendering loop until an action is ready or simulation should advance.

        When paused, this loops indefinitely until user takes an action.
        When running, this returns after the frame delay has elapsed.
        """
        start_time = time.time()
        frame_delay = self._state.get_frame_delay()
        was_paused_last_frame = False

        while True:
            # Check if we should exit (episode done or stopped)
            if self._sim.is_done() or self._state.playback == PlaybackState.STOPPED:
                break

            # Clear previous user action before reading new input
            self._state.user_action = None

            # Read user input and store in state
            self._state.user_input = self._get_input()

            # Handle user input for agent controls
            self._handle_user_input()

            # Update viewport size based on sidebar visibility
            self._update_viewport_size()

            # Update FPS calculation
            self._update_fps(time.time())

            # Clear panels for new frame
            self._panels.clear_all()

            # Let each component update its panel
            for component in self._components:
                component.update()

            # Display the composed frame
            self._render_display()

            # Clear input after processing
            self._state.user_input = None

            # If we have a manual action ready, set it and return
            if self._state.user_action is not None and self._state.selected_agent is not None:
                # Set the action for the manually controlled agent
                self._sim.agent(self._state.selected_agent).set_action(self._state.user_action)
                # Clear should_step and action after setting it
                self._state.should_step = False
                self._state.user_action = None
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
        if not self._state.user_input:
            return

        ch = self._state.user_input

        # Modal input handling: if in VIBE_PICKER mode, route directly to vibe picker
        # This ensures the picker gets ALL input and blocks everything else
        if self._state.mode == RenderMode.VIBE_PICKER:
            for component in self._components:
                if isinstance(component, VibePickerComponent):
                    component.handle_input(ch)
                    return

        # Modal input handling: if in HELP mode, any key exits
        if self._state.mode == RenderMode.HELP:
            self._state.exit_help()
            return

        # Handle help activation
        if ch == "?":
            self._state.enter_help()
            return

        # Handle sidebar toggles
        if ch.isdigit() and ch in self._sidebar_hotkeys:
            panel_name = self._sidebar_hotkeys[ch]
            self._state.toggle_sidebar_panel(panel_name)
            return

        # Let each component handle the input
        for component in self._components:
            if component.handle_input(ch):
                break  # Stop after first component handles it

    def _render_display(self) -> None:
        """Render the panel layout to the terminal."""
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

    def _apply_initial_viewport_size(self) -> None:
        """Configure viewport size using the initial terminal dimensions."""
        self._update_viewport_size()

    def _update_viewport_size(self) -> None:
        """Update viewport size based on current sidebar visibility."""
        columns = max(2, self._initial_terminal_columns)
        lines = max(2, self._initial_terminal_lines)

        viewport_height = max(1, lines - RESERVED_VERTICAL_LINES)
        if self._state.map_height:
            viewport_height = min(viewport_height, self._state.map_height)

        # Check if any sidebar panels are visible
        sidebar_visible = self._is_sidebar_visible()

        if sidebar_visible:
            # Reserve space for sidebar
            available_width = max(2, columns - SIDEBAR_WIDTH - LAYOUT_PADDING)
        else:
            # Use full width when sidebar is hidden
            available_width = max(2, columns - LAYOUT_PADDING)

        viewport_width = max(1, available_width // 2)
        if self._state.map_width:
            viewport_width = min(viewport_width, self._state.map_width)

        self._state.viewport_height = viewport_height
        self._state.viewport_width = viewport_width

        map_panel_width = max(2, min(available_width, viewport_width * 2))
        self._panels.map_view.width = map_panel_width
        self._panels.map_view.height = max(1, viewport_height)

    def _is_sidebar_visible(self) -> bool:
        """Check if any sidebar panels are currently visible."""
        for _name, visible in self._state.sidebar_visibility.items():
            if visible:
                return True
        return False

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
        if self._terminal_fd is not None and self._old_terminal_settings is not None:
            try:
                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._old_terminal_settings)
            except termios.error:
                pass
        self._terminal_fd = None
        self._old_terminal_settings = None
        self._console.show_cursor(True)

    def __del__(self):
        """Clean up on destruction."""
        self._cleanup_terminal()
