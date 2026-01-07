"""Vibe picker component for miniscope renderer."""

from mettagrid.renderer.miniscope.components.base import MiniscopeComponent
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, RenderMode
from mettagrid.simulator.simulator import Simulation
from mettagrid.simulator.types import Action

try:
    from mettagrid.config.vibes import VIBES as VIBE_DATA
    from mettagrid.config.vibes import search_vibes
except ImportError:
    VIBE_DATA = None
    search_vibes = None


class VibePickerComponent(MiniscopeComponent):
    """Component for vibe selection interface."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the vibe picker component.

        Args:
            sim: MettaGrid simulator reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(sim=sim, state=state, panels=panels)
        sidebar_panel = panels.get_sidebar_panel("vibe_picker")
        assert sidebar_panel is not None
        self._set_panel(sidebar_panel)
        self._vibe_query: str = ""

    def handle_input(self, ch: str) -> bool:
        """Handle user input when the vibe picker is active."""
        if self._state.mode != RenderMode.VIBE_PICKER:
            return False

        # In vibe picker mode - handle all input here
        if ch == "\n" or ch == "\r":
            # Enter - confirm selection (always select first result)
            vibe_id = None
            if search_vibes:
                results = search_vibes(self._vibe_query) if self._vibe_query else []
                if not results and VIBE_DATA:
                    results = [(i, VIBE_DATA[i]) for i in range(min(10, len(VIBE_DATA)))]
                if results:
                    vibe_id = results[0][0]

            if vibe_id is not None and VIBE_DATA and 0 <= vibe_id < len(VIBE_DATA):
                # Get the vibe name and construct the proper action name
                vibe = VIBE_DATA[vibe_id]
                action_name = f"change_vibe_{vibe.name}"
                # Check if the action exists before setting it
                if action_name in self._sim.action_ids:
                    self.state.user_action = Action(name=action_name)
                    self.state.should_step = True
                    self._exit_vibe_picker()
                else:
                    # Action doesn't exist - exit picker without setting action
                    self._exit_vibe_picker()
        elif ch == "\x1b":  # Escape
            self._exit_vibe_picker()
        elif ch == "\x7f" or ch == "\x08":  # Backspace
            self._vibe_query = self._vibe_query[:-1] if self._vibe_query else ""
        elif ch == "[":
            # Cycle to previous agent
            self._state.select_previous_agent(self._sim.num_agents)
        elif ch == "]":
            # Cycle to next agent
            self._state.select_next_agent(self._sim.num_agents)
        elif ch and ch.isprintable():
            self._vibe_query = self._vibe_query + ch

        return True  # Always return True to block other components

    def update(self) -> None:
        """Render the vibe picker panel using current state."""
        # Always render when in VIBE_PICKER mode, regardless of sidebar visibility
        in_picker_mode = self._state.mode == RenderMode.VIBE_PICKER

        panel = self._panel
        assert panel is not None
        if not in_picker_mode and not self.state.is_sidebar_visible("vibe_picker"):
            panel.clear()
            return

        lines = self._build_lines(self._vibe_query)
        panel.set_content(lines)

    def _build_lines(self, query: str) -> list[str]:
        """Build vibe picker display as plain text lines.

        Args:
            query: Current search query

        Returns:
            List of strings for display
        """
        width = self._width if self._width else 40
        lines = []

        if not VIBE_DATA:
            lines.append("Vibe Picker".ljust(width))
            lines.append("-" * width)
            lines.append("VIBE_DATA not available".ljust(width))
            lines.append("Install cogames.cogs_vs_clips".ljust(width))
            return lines

        # Header
        agent_info = f" [Agent {self._state.selected_agent}]" if self._state.selected_agent is not None else ""
        header = f"Vibe Picker{agent_info}: {query}"
        lines.append(header[:width].ljust(width))
        lines.append("-" * min(width, 40))

        # Get matches using fuzzy name search only
        if query and search_vibes:
            results = search_vibes(query)[:10]
        else:
            # Show first 5 vibes when no query
            if VIBE_DATA:
                results = [(i, VIBE_DATA[i]) for i in range(min(5, len(VIBE_DATA)))]
            else:
                results = []

        if results:
            for idx, (_vibe_id, vibe) in enumerate(results):
                # Highlight the first row (what will be selected on Enter)
                if idx == 0:
                    line = f"> {vibe.name:<{width - 10}} {vibe.symbol:>5}"
                else:
                    line = f"  {vibe.name:<{width - 10}} {vibe.symbol:>5}"
                lines.append(line[:width].ljust(width))
        else:
            lines.append("(no matches)".ljust(width))

        # Add help text
        lines.append("")
        lines.append("[]=Agent  Enter=OK  Esc=Cancel".ljust(width))

        return lines

    def _exit_vibe_picker(self) -> None:
        """Exit vibe picker mode and reset query."""
        self._vibe_query = ""
        self.state.exit_vibe_picker()
