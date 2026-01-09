"""Simulation control component for miniscope renderer."""

import numpy as np
from rich.text import Text

from mettagrid.renderer.miniscope.components.base import MiniscopeComponent
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, PlaybackState, RenderMode
from mettagrid.simulator.simulator import Simulation


class SimControlComponent(MiniscopeComponent):
    """Component for displaying simulation status and handling playback controls."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the simulation control component."""
        super().__init__(sim=sim, state=state, panels=panels)
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
        elif ch in ["f", "F"]:
            self._state.set_mode(RenderMode.FOLLOW)
            return True
        elif ch in ["p", "P"]:
            self._state.set_mode(RenderMode.PAN)
            return True
        elif ch in ["t", "T"]:
            self._state.set_mode(RenderMode.SELECT)
            return True

        # Handle quit
        elif ch in ["q", "Q"]:
            self._sim.end_episode()
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
        panel = self._panel
        assert panel is not None
        # Calculate total reward
        total_reward = 0.0
        if self.state.total_rewards is not None:
            total_reward = float(np.sum(self.state.total_rewards))

        # Format values
        mode_text = self.state.mode.value.upper()
        status = "PAUSED" if self.state.playback == PlaybackState.PAUSED else "PLAYING"
        fps = f"{self.state.fps:.1f}" if self.state.fps < 10 else f"{int(self.state.fps)}"
        true_fps = f"{self.state.true_fps:.1f}" if self.state.true_fps < 10 else f"{int(self.state.true_fps)}"
        camera_pos = f"({self.state.camera_row},{self.state.camera_col})"

        # Build sidebar status indicators (only toggleable panels)
        sidebar_panels = ["agent_info", "object_info", "symbols"]
        squares = " ".join(
            [f"{i + 1}[{'x' if self.state.is_sidebar_visible(name) else ' '}]" for i, name in enumerate(sidebar_panels)]
        )

        # Build first line with controls and sidebar indicators; use cell width for alignment
        controls = "?=Help  SPACE=Play/Pause  <>=Speed  F=Follow P=Pan T=Select  IJKL=Pan  Q=Quit"

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
            + f"SPS: {fps} ({true_fps}) | Status: {status} | "
            + f"Mode: {mode_text} | Camera: {camera_pos}"
        )

        # Set panel content
        panel.set_content(text)
