"""Agent control component for miniscope renderer."""

from typing import Dict

from rich.table import Table
from rich.text import Text

from mettagrid.renderer.miniscope.components.base import MiniscopeComponent
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState
from mettagrid.simulator.simulator import Simulation
from mettagrid.simulator.types import Action


class AgentControlComponent(MiniscopeComponent):
    """Component for displaying keyboard controls and handling agent actions."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the agent control component."""
        super().__init__(sim=sim, state=state, panels=panels)
        self._set_panel(panels.footer)

        # Setup movement action mapping (maps keys to action names)
        self._move_action_lookup: Dict[str, str] = {}
        if hasattr(sim, "action_ids"):
            action_ids = sim.action_ids
            # Map WASD keys to directional movement actions
            if "move_north" in action_ids:
                self._move_action_lookup["W"] = "move_north"
            if "move_west" in action_ids:
                self._move_action_lookup["A"] = "move_west"
            if "move_south" in action_ids:
                self._move_action_lookup["S"] = "move_south"
            if "move_east" in action_ids:
                self._move_action_lookup["D"] = "move_east"
            # Map R to rest/noop
            if "noop" in action_ids:
                self._move_action_lookup["R"] = "noop"

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
            self._state.select_previous_agent(self._sim.num_agents)
            return True
        elif ch == "]":
            self._state.select_next_agent(self._sim.num_agents)
            return True

        # Handle manual mode toggle
        elif self._state.selected_agent is not None:
            # Handle movement actions
            if (action_name := self._move_action_lookup.get(ch)) is not None:
                self._state.user_action = Action(name=action_name)
                self._state.should_step = True
                return True
            elif ch == "E":
                self._state.enter_vibe_picker()
                return True
            elif ch == "M":
                self._state.toggle_manual_control(self._state.selected_agent)
                return True

        return False

    def update(self) -> None:
        """Update the agent control panel display."""
        panel = self._panel
        assert panel is not None
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
        panel.set_content(content)
