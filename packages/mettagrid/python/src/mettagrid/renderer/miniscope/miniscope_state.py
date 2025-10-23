"""State management for miniscope renderer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np

FOLLOW_MODE_KEY = "f"
PAN_MODE_KEY = "p"
SELECT_MODE_KEY = "t"


class RenderMode(str, Enum):
    """Render mode for the miniscope."""

    FOLLOW = "follow"
    PAN = "pan"
    SELECT = "select"
    GLYPH_PICKER = "glyph_picker"
    HELP = "help"


class PlaybackState(Enum):
    """Playback state for the renderer."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"  # Single step mode


@dataclass
class MiniscopeState:
    """State container for miniscope renderer."""

    # Playback state
    playback: PlaybackState = PlaybackState.STOPPED
    fps: float = 4.0
    step_count: int = 0
    max_steps: Optional[int] = None

    # Camera and viewport
    camera_row: int = 0
    camera_col: int = 0
    viewport_height: int = 20
    viewport_width: int = 40

    # Mode and selection
    mode: RenderMode = RenderMode.FOLLOW
    selected_agent: Optional[int] = 0
    cursor_row: int = 0
    cursor_col: int = 0

    # Agent control
    manual_agents: Set[int] = field(default_factory=set)
    user_action: Optional[tuple[int, int]] = None
    should_step: bool = False

    # User input
    user_input: Optional[str] = None

    # Rewards tracking
    total_rewards: Optional[np.ndarray] = None

    # Map bounds (computed from grid)
    min_row: int = 0
    min_col: int = 0
    map_height: int = 0
    map_width: int = 0

    # Shared data for components
    object_type_names: Optional[List[str]] = None
    resource_names: Optional[List[str]] = None
    symbol_map: Optional[Dict[str, str]] = None
    glyphs: Optional[List[str]] = None

    # Sidebar panel visibility
    sidebar_visibility: Dict[str, bool] = field(default_factory=dict)
    _saved_sidebar_visibility: Optional[Dict[str, bool]] = field(default=None)

    def is_running(self) -> bool:
        """Check if the renderer should continue running."""
        return self.playback in (PlaybackState.RUNNING, PlaybackState.PAUSED, PlaybackState.STEPPING)

    def should_render_frame(self) -> bool:
        """Check if a new frame should be rendered."""
        return self.playback == PlaybackState.RUNNING or self.should_step

    def toggle_pause(self) -> None:
        """Toggle between paused and running states."""
        if self.playback == PlaybackState.PAUSED:
            self.playback = PlaybackState.RUNNING
        elif self.playback == PlaybackState.RUNNING:
            self.playback = PlaybackState.PAUSED

    def increase_speed(self) -> None:
        """Increase playback speed."""
        self.fps = min(60.0, self.fps * 1.5)

    def decrease_speed(self) -> None:
        """Decrease playback speed."""
        self.fps = max(0.5, self.fps / 1.5)

    def get_frame_delay(self) -> float:
        """Get the delay between frames in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0.25

    def set_mode(self, mode: RenderMode) -> None:
        """Set the render mode when manually selected."""
        if mode in (RenderMode.GLYPH_PICKER, RenderMode.HELP):
            return
        self.mode = mode

        # Auto-show object_info when entering SELECT mode
        if mode == RenderMode.SELECT and "object_info" in self.sidebar_visibility:
            self.sidebar_visibility["object_info"] = True

    def enter_glyph_picker(self) -> None:
        """Enter glyph picker mode and configure sidebar."""
        # Save current sidebar state before modifying
        self._saved_sidebar_visibility = self.sidebar_visibility.copy()

        self.mode = RenderMode.GLYPH_PICKER
        # Hide all sidebar panels except agent_info and glyph_picker
        for name in self.sidebar_visibility.keys():
            self.sidebar_visibility[name] = name in ("agent_info", "glyph_picker")

    def exit_glyph_picker(self) -> None:
        """Exit glyph picker mode and restore previous state."""
        self.mode = RenderMode.FOLLOW

        # Restore saved sidebar visibility if available
        if self._saved_sidebar_visibility is not None:
            self.sidebar_visibility = self._saved_sidebar_visibility.copy()
            self._saved_sidebar_visibility = None
        else:
            # Fallback to default if no saved state
            for name in self.sidebar_visibility.keys():
                if name in ("agent_info", "object_info", "symbols"):
                    self.sidebar_visibility[name] = True
                else:
                    self.sidebar_visibility[name] = False

    def enter_help(self) -> None:
        """Enter help mode and configure sidebar."""
        # Save current sidebar state before modifying
        self._saved_sidebar_visibility = self.sidebar_visibility.copy()

        self.mode = RenderMode.HELP
        # Hide all sidebar panels except help
        for name in self.sidebar_visibility.keys():
            self.sidebar_visibility[name] = name == "help"

    def exit_help(self) -> None:
        """Exit help mode and restore previous state."""
        self.mode = RenderMode.FOLLOW

        # Restore saved sidebar visibility if available
        if self._saved_sidebar_visibility is not None:
            self.sidebar_visibility = self._saved_sidebar_visibility.copy()
            self._saved_sidebar_visibility = None
        else:
            # Fallback to default if no saved state
            for name in self.sidebar_visibility.keys():
                if name in ("agent_info", "object_info", "symbols"):
                    self.sidebar_visibility[name] = True
                else:
                    self.sidebar_visibility[name] = False

    def toggle_manual_control(self, agent_id: int) -> None:
        """Toggle manual control for an agent."""
        if agent_id in self.manual_agents:
            self.manual_agents.remove(agent_id)
        else:
            self.manual_agents.add(agent_id)

    def select_next_agent(self, num_agents: int) -> None:
        """Select the next agent."""
        if self.selected_agent is None:
            self.selected_agent = 0
        else:
            self.selected_agent = (self.selected_agent + 1) % num_agents

    def select_previous_agent(self, num_agents: int) -> None:
        """Select the previous agent."""
        if self.selected_agent is None:
            self.selected_agent = 0
        else:
            self.selected_agent = (self.selected_agent - 1) % num_agents

    def move_camera(self, delta_row: int, delta_col: int) -> None:
        """Move the camera by the given deltas."""
        self.camera_row = max(self.min_row, min(self.min_row + self.map_height - 1, self.camera_row + delta_row))
        self.camera_col = max(self.min_col, min(self.min_col + self.map_width - 1, self.camera_col + delta_col))

    def move_cursor(self, delta_row: int, delta_col: int) -> None:
        """Move the cursor by the given deltas."""
        self.cursor_row = max(self.min_row, min(self.min_row + self.map_height - 1, self.cursor_row + delta_row))
        self.cursor_col = max(self.min_col, min(self.min_col + self.map_width - 1, self.cursor_col + delta_col))

    def set_bounds(self, min_row: int, min_col: int, height: int, width: int) -> None:
        """Set the map bounds."""
        self.min_row = min_row
        self.min_col = min_col
        self.map_height = height
        self.map_width = width

        # Clamp camera and cursor to bounds
        self.camera_row = max(min_row, min(min_row + height - 1, self.camera_row))
        self.camera_col = max(min_col, min(min_col + width - 1, self.camera_col))
        self.cursor_row = max(min_row, min(min_row + height - 1, self.cursor_row))
        self.cursor_col = max(min_col, min(min_col + width - 1, self.cursor_col))

    def reset_for_episode(self, num_agents: int, map_height: int, map_width: int) -> None:
        """Reset state for a new episode."""
        self.step_count = 0
        self.playback = PlaybackState.PAUSED  # Start paused
        self.mode = RenderMode.FOLLOW
        self.selected_agent = 0 if num_agents > 0 else None
        self.total_rewards = np.zeros(num_agents) if num_agents > 0 else None
        self.manual_agents.clear()
        self.user_action = None
        self.should_step = False
        self.sidebar_visibility.clear()

        # Store map dimensions
        self.map_height = map_height
        self.map_width = map_width

        # Set initial camera to center of map
        self.camera_row = map_height // 2
        self.camera_col = map_width // 2
        self.cursor_row = map_height // 2
        self.cursor_col = map_width // 2

    def initialize_sidebar_visibility(self, panels: list[str]) -> None:
        """Initialize visibility of sidebar panels, defaulting to visible for regular panels."""
        for name in panels:
            # Modal panels (glyph_picker, help) start hidden
            if name in ("glyph_picker", "help"):
                self.sidebar_visibility[name] = False
            else:
                self.sidebar_visibility[name] = True

    def toggle_sidebar_panel(self, name: str) -> None:
        """Toggle visibility for a specific sidebar panel."""
        current = self.sidebar_visibility.get(name, True)
        self.sidebar_visibility[name] = not current

    def is_sidebar_visible(self, name: str) -> bool:
        """Check if a sidebar panel should be visible."""
        return self.sidebar_visibility.get(name, True)

    def set_sidebar_visibility(self, name: str, visible: bool) -> None:
        """Set explicit visibility for a sidebar panel."""
        self.sidebar_visibility[name] = visible
