"""Miniscope renderer for replay playback.

Provides unicode-based rendering of replay files in the terminal.
"""

import io
import select
import shutil
import sys
import termios
import time
import tty
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from mettagrid.config.vibes import VIBES as VIBE_DATA
from mettagrid.renderer.miniscope.components import (
    AgentInfoComponent,
    HelpPanelComponent,
    MapComponent,
    MiniscopeComponent,
    ObjectInfoComponent,
    SymbolsTableComponent,
)
from mettagrid.renderer.miniscope.miniscope_panel import (
    LAYOUT_PADDING,
    RESERVED_VERTICAL_LINES,
    SIDEBAR_WIDTH,
    PanelLayout,
)
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, PlaybackState, RenderMode
from mettagrid.renderer.miniscope.symbol import DEFAULT_SYMBOL_MAP
from mettagrid.simulator.replay_loader import ReplaySimulation, load_replay


class ReplaySimControlComponent(MiniscopeComponent):
    """Playback control for replay mode (play/pause, speed, stepping)."""

    def __init__(
        self,
        sim: ReplaySimulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        super().__init__(sim=sim, state=state, panels=panels)
        self._set_panel(panels.header)

    def handle_input(self, ch: str) -> bool:
        """Handle playback controls."""
        # Play/pause
        if ch == " ":
            self._state.toggle_pause()
            return True
        # Single step forward/backward
        elif ch == ".":
            self._state.playback = PlaybackState.PAUSED
            self._sim.step_forward(1)  # type: ignore[union-attr]
            self._state.step_count = self._sim.current_step  # type: ignore[union-attr]
            return True
        elif ch == ",":
            self._state.playback = PlaybackState.PAUSED
            self._sim.step_backward(1)  # type: ignore[union-attr]
            self._state.step_count = self._sim.current_step  # type: ignore[union-attr]
            return True
        # 10-step seeking
        elif ch == ">":
            self._state.playback = PlaybackState.PAUSED
            self._sim.step_forward(10)  # type: ignore[union-attr]
            self._state.step_count = self._sim.current_step  # type: ignore[union-attr]
            return True
        elif ch == "<":
            self._state.playback = PlaybackState.PAUSED
            self._sim.step_backward(10)  # type: ignore[union-attr]
            self._state.step_count = self._sim.current_step  # type: ignore[union-attr]
            return True
        # Speed controls (same as live miniscope: < and > with shift)
        elif ch == "+":
            self._state.increase_speed()
            return True
        elif ch == "-":
            self._state.decrease_speed()
            return True
        elif ch == "q":
            self._state.playback = PlaybackState.STOPPED
            return True
        # Camera panning (IJKL to match live miniscope)
        elif ch == "i":
            self._state.move_camera(-1, 0)
            return True
        elif ch == "I":
            self._state.move_camera(-10, 0)
            return True
        elif ch == "k":
            self._state.move_camera(1, 0)
            return True
        elif ch == "K":
            self._state.move_camera(10, 0)
            return True
        elif ch == "j":
            self._state.move_camera(0, -1)
            return True
        elif ch == "J":
            self._state.move_camera(0, -10)
            return True
        elif ch == "l":
            self._state.move_camera(0, 1)
            return True
        elif ch == "L":
            self._state.move_camera(0, 10)
            return True
        # Mode switching (same as live miniscope)
        elif ch == "f":
            self._state.set_mode(RenderMode.FOLLOW)
            return True
        elif ch == "p":
            self._state.set_mode(RenderMode.PAN)
            return True
        elif ch == "t":
            self._state.set_mode(RenderMode.SELECT)
            return True
        # Agent selection
        elif ch == "]":
            self._state.select_next_agent(self._sim.num_agents)
            return True
        elif ch == "[":
            self._state.select_previous_agent(self._sim.num_agents)
            return True
        return False

    def update(self) -> None:
        """Show playback status."""
        assert self._panel is not None
        max_step = getattr(self._sim, "max_steps", self._state.step_count + 1)
        status = "▶" if self._state.playback == PlaybackState.RUNNING else "⏸"
        mode_str = self._state.mode.value.upper()
        lines = [
            f"Step: {self._state.step_count}/{max_step} {status} FPS: {self._state.fps:.1f} Mode: {mode_str}",
            "SPACE: play/pause  ,/.: step  </> 10-step  +/-: speed  q: quit  ?: help",
        ]
        self._panel.set_content(lines)


class MiniscopeReplayRenderer:
    """Replay renderer using miniscope components."""

    def __init__(self, replay_path: Path | str):
        """Initialize the replay renderer.

        Args:
            replay_path: Path to the replay file (.json.z)
        """
        self._replay_path = Path(replay_path)
        self._replay_sim: Optional[ReplaySimulation] = None
        self._state = MiniscopeState()

        # Terminal setup
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

        self._panels = PanelLayout(self._console)
        self._components: List[MiniscopeComponent] = []
        self._old_terminal_settings = None
        self._terminal_fd = None
        self._last_frame_time = 0.0
        self._ema_frame_time: float = 0.0
        self._ema_alpha: float = 0.2
        self._sidebar_hotkeys: dict[str, str] = {}

    def run(self) -> None:
        """Main entry point - load replay and run playback loop."""
        self._replay_sim = load_replay(self._replay_path)
        self._initialize()
        try:
            self._run_loop()
        finally:
            self._cleanup()

    def _initialize(self) -> None:
        """Initialize renderer for replay playback."""
        assert self._replay_sim is not None

        # Reset state
        self._state.reset_for_episode(
            num_agents=self._replay_sim.num_agents,
            map_height=self._replay_sim.map_height,
            map_width=self._replay_sim.map_width,
        )

        # Initialize configuration in state
        self._state.resource_names = self._replay_sim.resource_names
        self._state.symbol_map = DEFAULT_SYMBOL_MAP.copy()

        # Add custom symbols from game config
        for obj in self._replay_sim.config.game.objects.values():
            self._state.symbol_map[obj.render_name or obj.name] = obj.render_symbol
            if obj.render_name and obj.render_name != obj.name:
                self._state.symbol_map[obj.name] = obj.render_symbol

        self._state.vibes = [g.symbol for g in VIBE_DATA] if VIBE_DATA else None
        self._state.max_steps = self._replay_sim.max_steps

        # Configure viewport
        self._apply_initial_viewport_size()

        # Setup sidebar panels
        sidebar_defs = [
            ("1", "agent_info", AgentInfoComponent),
            ("2", "object_info", ObjectInfoComponent),
            ("3", "symbols", SymbolsTableComponent),
        ]
        self._sidebar_hotkeys = {hotkey: name for hotkey, name, _ in sidebar_defs}

        self._panels.reset_sidebar_panels()
        for _, name, _ in sidebar_defs:
            self._panels.register_sidebar_panel(name)
        self._panels.register_sidebar_panel("help")

        self._state.initialize_sidebar_visibility([name for _, name, _ in sidebar_defs] + ["help"])

        # Create components - use replay-specific sim control
        self._components = [
            MapComponent(sim=self._replay_sim, state=self._state, panels=self._panels),  # type: ignore[arg-type]
            ReplaySimControlComponent(sim=self._replay_sim, state=self._state, panels=self._panels),  # type: ignore[arg-type]
            AgentInfoComponent(sim=self._replay_sim, state=self._state, panels=self._panels),  # type: ignore[arg-type]
            ObjectInfoComponent(sim=self._replay_sim, state=self._state, panels=self._panels),  # type: ignore[arg-type]
            SymbolsTableComponent(sim=self._replay_sim, state=self._state, panels=self._panels),  # type: ignore[arg-type]
            HelpPanelComponent(sim=self._replay_sim, state=self._state, panels=self._panels),  # type: ignore[arg-type]
        ]

        # Setup terminal
        self._setup_terminal()
        self._panels.start_live()
        self._state.playback = PlaybackState.PAUSED
        self._last_frame_time = time.time()

    def _run_loop(self) -> None:
        """Main playback loop."""
        assert self._replay_sim is not None

        start_time = time.time()
        frame_delay = self._state.get_frame_delay()
        was_paused_last_frame = False

        while self._state.playback != PlaybackState.STOPPED:
            # Check if replay is done
            if self._replay_sim.is_done() and self._state.playback == PlaybackState.RUNNING:
                self._state.playback = PlaybackState.PAUSED

            # Read and handle input
            self._state.user_input = self._get_input()
            self._handle_user_input()

            # Sync rewards from replay state
            self._sync_rewards()

            # Update viewport
            self._update_viewport_size()

            # Update FPS
            self._update_fps(time.time())

            # Update and render components
            self._panels.clear_all()
            for component in self._components:
                component.update()
            self._panels.render_to_console()

            # Clear input
            self._state.user_input = None

            # Handle playback timing
            if self._state.playback == PlaybackState.PAUSED:
                was_paused_last_frame = True
                time.sleep(1.0 / 60.0)
                continue

            if was_paused_last_frame:
                start_time = time.time()
                frame_delay = self._state.get_frame_delay()
                was_paused_last_frame = False

            elapsed = time.time() - start_time
            if elapsed >= frame_delay:
                # Advance replay
                self._replay_sim.step_forward(1)
                self._state.step_count = self._replay_sim.current_step
                start_time = time.time()
                frame_delay = self._state.get_frame_delay()

            time.sleep(1.0 / 60.0)

    def _sync_rewards(self) -> None:
        """Sync total_rewards state from replay episode_rewards."""
        if self._replay_sim is None:
            return
        episode_rewards = self._replay_sim.episode_rewards
        if episode_rewards and self._state.total_rewards is not None:
            for i, reward in enumerate(episode_rewards):
                if i < len(self._state.total_rewards):
                    self._state.total_rewards[i] = reward

    def _handle_user_input(self) -> None:
        if not self._state.user_input:
            return

        ch = self._state.user_input

        # Modal input handling for HELP mode
        if self._state.mode == RenderMode.HELP:
            self._state.exit_help()
            return

        # Help activation
        if ch == "?":
            self._state.enter_help()
            return

        # Sidebar toggles
        if ch.isdigit() and ch in self._sidebar_hotkeys:
            panel_name = self._sidebar_hotkeys[ch]
            self._state.toggle_sidebar_panel(panel_name)
            return

        # Delegate to components
        for component in self._components:
            if component.handle_input(ch):
                break

    def _get_input(self) -> Optional[str]:
        if self._terminal_fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def _update_fps(self, current_time: float) -> None:
        if self._last_frame_time > 0:
            frame_time = current_time - self._last_frame_time
            if self._ema_frame_time == 0:
                self._ema_frame_time = frame_time
            else:
                self._ema_frame_time = self._ema_alpha * frame_time + (1 - self._ema_alpha) * self._ema_frame_time
            self._state.true_fps = 1.0 / self._ema_frame_time if self._ema_frame_time > 0 else 0.0
        self._last_frame_time = current_time

    def _apply_initial_viewport_size(self) -> None:
        self._update_viewport_size()

    def _update_viewport_size(self) -> None:
        columns = max(2, self._initial_terminal_columns)
        lines = max(2, self._initial_terminal_lines)

        viewport_height = max(1, lines - RESERVED_VERTICAL_LINES)
        if self._state.map_height:
            viewport_height = min(viewport_height, self._state.map_height)

        sidebar_visible = self._is_sidebar_visible()

        if sidebar_visible:
            available_width = max(2, columns - SIDEBAR_WIDTH - LAYOUT_PADDING)
        else:
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
        for _name, visible in self._state.sidebar_visibility.items():
            if visible:
                return True
        return False

    def _setup_terminal(self) -> None:
        try:
            self._terminal_fd = sys.stdin.fileno()
            self._old_terminal_settings = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            self._console.show_cursor(False)
        except (OSError, io.UnsupportedOperation, termios.error):
            self._terminal_fd = None
            self._old_terminal_settings = None

    def _cleanup(self) -> None:
        self._panels.stop_live()
        if self._terminal_fd is not None and self._old_terminal_settings is not None:
            try:
                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._old_terminal_settings)
            except termios.error:
                pass
        self._terminal_fd = None
        self._old_terminal_settings = None
        self._console.show_cursor(True)


def replay_unicode(replay_path: Path | str) -> None:
    # Entry point for unicode replay rendering
    renderer = MiniscopeReplayRenderer(replay_path)
    renderer.run()
