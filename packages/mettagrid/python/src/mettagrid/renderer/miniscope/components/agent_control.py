"""Agent control component for miniscope renderer."""

from typing import Dict

from rich.table import Table
from rich.text import Text

from mettagrid import MettaGridEnv
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState

from .base import MiniscopeComponent


class AgentControlComponent(MiniscopeComponent):
    """Component for displaying keyboard controls and handling agent actions."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the agent control component."""
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.footer)

        action_lookup: Dict[str, int] = {name: idx for idx, name in enumerate(self._env.action_names)}

        # Assumes move_{cardinal_direction} named actions exist
        # If not found, fall back to 0: north, 1: south, 2: west, 3: east, 4: rest/noop
        self._move_action_lookup: Dict[str, int] = {
            "W": action_lookup.get("move_north", 0),
            "S": action_lookup.get("move_south", 1),
            "A": action_lookup.get("move_west", 2),
            "D": action_lookup.get("move_east", 3),
            "R": action_lookup.get("noop", 4),
        }

    def handle_input(self, ch: str) -> bool:
        """Handle agent control inputs.

        Args:
            ch: The character input from the user

        Returns:
            True if the input was handled
        """
        ch = ch.upper()
        # Handle agent selection
        if ch == "[":
            self._state.select_previous_agent(self._env.num_agents)
            return True
        elif ch == "]":
            self._state.select_next_agent(self._env.num_agents)
            return True

        # Handle manual mode toggle
        elif self._state.selected_agent is not None:
            # Handle movement actions
            if (action_id := self._move_action_lookup.get(ch)) is not None:
                self._state.user_action = (action_id, 0)
                self._state.should_step = True
                return True
            # Handle glyph picker
            elif ch == "E":
                self._state.enter_glyph_picker()
                return True
            # Handle manual mode toggle
            elif ch == "M":
                self._state.toggle_manual_control(self._state.selected_agent)
                return True

        return False

    def update(self) -> None:
        """Update the agent control panel display."""
        # Get agent selection info
        if self._state.selected_agent is not None:
            agent_text = f"[Agent {self._state.selected_agent}]"
            manual_text = " (Manual)" if self._state.selected_agent in self._state.manual_agents else ""
        else:
            agent_text = "[AI Control]"
            manual_text = ""

        # Use compact format if height is limited (< 3 lines)
        if self._height and self._height < 3:
            content = Text(f"{agent_text}{manual_text} | []=Agent | M=Manual | WASD=Move | E=Emote | R=Rest")
        else:
            # Create table with controls
            table = Table(show_header=False, show_edge=True, box=None, padding=(0, 1))
            table.add_column("Controls", justify="left", no_wrap=True)

            # Agent controls only
            table.add_row(f"{agent_text}{manual_text}")
            table.add_row("[]=Agent  M=Manual  WASD=Move  E=Emote  R=Rest")
            content = table

        # Set panel content
        self._panel.set_content(content)
