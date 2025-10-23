"""Simulation control component for miniscope renderer."""

import numpy as np
from rich.text import Text

from mettagrid import MettaGridEnv
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import (
    FOLLOW_MODE_KEY,
    PAN_MODE_KEY,
    SELECT_MODE_KEY,
    MiniscopeState,
    PlaybackState,
    RenderMode,
)

from .base import MiniscopeComponent


class SimControlComponent(MiniscopeComponent):
    """Component for displaying simulation status and handling playback controls."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the simulation control component."""
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.header)

    def handle_input(self, ch: str) -> bool:
        """Handle simulation control inputs."""
        # Handle play/pause
        if ch == " ":
            self._state.toggle_pause()
            return True

        # Handle speed changes
        elif ch in ["<", ","]:
            self._state.decrease_speed()
            return True
        elif ch in [">", "."]:
            self._state.increase_speed()
            return True

        # Handle mode selection
        elif ch in [FOLLOW_MODE_KEY, FOLLOW_MODE_KEY.upper()]:
            self._state.set_mode(RenderMode.FOLLOW)
            return True
        elif ch in [PAN_MODE_KEY, PAN_MODE_KEY.upper()]:
            self._state.set_mode(RenderMode.PAN)
            return True
        elif ch in [SELECT_MODE_KEY, SELECT_MODE_KEY.upper()]:
            self._state.set_mode(RenderMode.SELECT)
            return True

        # Handle quit
        elif ch in ["q", "Q"]:
            self._state.playback = PlaybackState.STOPPED
            return True

        # Handle help
        elif ch == "?":
            # Help is displayed by HelpPanelComponent
            return True

        # Handle camera panning
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

        return False

    def update(self) -> None:
        """Update the simulation control header display."""
        # Calculate total reward
        total_reward = 0.0
        if self.state.total_rewards is not None:
            total_reward = float(np.sum(self.state.total_rewards))

        # Format values
        mode_text = self.state.mode.value.upper()
        status = "PAUSED" if self.state.playback == PlaybackState.PAUSED else "PLAYING"
        sps = f"{self.state.fps:.1f}" if self.state.fps < 10 else f"{int(self.state.fps)}"
        camera_pos = f"({self.state.camera_row},{self.state.camera_col})"

        # Build sidebar status indicators (only toggleable panels)
        sidebar_panels = ["agent_info", "object_info", "symbols"]
        squares = " ".join(
            [f"{i + 1}[{'x' if self.state.is_sidebar_visible(name) else ' '}]" for i, name in enumerate(sidebar_panels)]
        )

        # Build first line with controls and sidebar indicators; use cell width for alignment
        controls = (
            "?=Help  SPACE=Play/Pause  <>=Speed  "
            f"{FOLLOW_MODE_KEY.upper()}=Follow {PAN_MODE_KEY.upper()}=Pan {SELECT_MODE_KEY.upper()}=Select  "
            "IJKL=Pan  Q=Quit"
        )

        terminal_width = self._panels.console.width if self._panels and self._panels.console else 120
        controls_text = Text(controls)
        squares_text = Text(squares)
        padding_available = terminal_width - controls_text.cell_len - squares_text.cell_len
        padding_length = max(1, padding_available)
        first_line = f"{controls_text.plain}{' ' * padding_length}{squares_text.plain}\n"

        # Build status text
        text = Text()
        text.append(first_line)
        text.append(
            f"Step {self.state.step_count} | "
            + f"Reward: {total_reward:.2f} | "
            + f"SPS: {sps} | Status: {status} | "
            + f"Mode: {mode_text} | Camera: {camera_pos}"
        )

        # Set panel content
        self._panel.set_content(text)
