"""Agent control component for miniscope renderer."""

from typing import Dict, Optional

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
        self._action_lookup: Dict[str, int] = {name: idx for idx, name in enumerate(self._env.action_names)}
        self._noop_action_id: Optional[int] = self._action_lookup.get("noop")
        self._move_action_lookup: Dict[int, Optional[int]] = {
            0: self._action_lookup.get("move_north"),
            1: self._action_lookup.get("move_south"),
            2: self._action_lookup.get("move_west"),
            3: self._action_lookup.get("move_east"),
        }

    def _set_move_action(self, orientation_idx: int) -> None:
        """Set a movement action for the selected agent."""
        move_action_id = self._move_action_lookup.get(orientation_idx)
        if move_action_id is not None:
            self._state.user_action = (move_action_id, 0)
        else:
            # Fall back to legacy verb/argument pairs if the flattened action is unavailable.
            self._state.user_action = (-1, orientation_idx)
        self._state.should_step = True

    def _set_rest_action(self) -> None:
        """Set a rest/no-op action for the selected agent."""
        if self._noop_action_id is not None:
            self._state.user_action = (self._noop_action_id, 0)
        else:
            # Default to noop when explicit action name is missing.
            self._state.user_action = (-1, -1)
        self._state.should_step = True

    def handle_input(self, ch: str) -> bool:
        """Handle agent control inputs.

        Args:
            ch: The character input from the user

        Returns:
            True if the input was handled
        """
        ch_lower = ch.lower()
        # Handle agent selection
        if ch == "[":
            self._state.select_previous_agent(self._env.num_agents)
            return True
        elif ch == "]":
            self._state.select_next_agent(self._env.num_agents)
            return True

        # Handle manual mode toggle
        elif ch in ["m", "M"] and self._state.selected_agent is not None:
            self._state.toggle_manual_control(self._state.selected_agent)
            return True

        # Handle agent movement commands (only when an agent is selected)
        elif ch_lower == "w" and self._state.selected_agent is not None:
            self._set_move_action(0)  # NORTH
            return True
        elif ch_lower == "s" and self._state.selected_agent is not None:
            self._set_move_action(1)  # SOUTH
            return True
        elif ch_lower == "a" and self._state.selected_agent is not None:
            self._set_move_action(2)  # WEST
            return True
        elif ch_lower == "d" and self._state.selected_agent is not None:
            self._set_move_action(3)  # EAST
            return True
        elif ch_lower == "r" and self._state.selected_agent is not None:
            self._set_rest_action()
            return True

        # Handle glyph picker
        elif ch in ["e", "E"] and self._state.selected_agent is not None:
            self._state.enter_glyph_picker()
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
