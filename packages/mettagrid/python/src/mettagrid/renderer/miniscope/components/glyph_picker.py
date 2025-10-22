"""Glyph picker component for miniscope renderer."""

from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, RenderMode
from mettagrid.simulator import Action, Simulation

from .base import MiniscopeComponent

try:
    from cogames.cogs_vs_clips.vibes import VIBES as GLYPH_DATA
    from cogames.cogs_vs_clips.vibes import search_vibes as search_glyphs
except ImportError:
    GLYPH_DATA = None
    search_glyphs = None


class GlyphPickerComponent(MiniscopeComponent):
    """Component for glyph selection interface."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the glyph picker component.

        Args:
            sim: MettaGrid simulator reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(sim=sim, state=state, panels=panels)
        self._panel = panels.sidebar
        self._glyph_query: str = ""

    def handle_input(self, ch: str) -> bool:
        """Handle user input when the glyph picker is active."""
        if self._state.mode != RenderMode.GLYPH_PICKER:
            return False

        # In glyph picker mode - handle all input here
        if ch == "\n" or ch == "\r":
            # Enter - confirm selection (always select first result)
            glyph_id = None
            if search_glyphs:
                results = search_glyphs(self._glyph_query) if self._glyph_query else []
                if not results and GLYPH_DATA:
                    results = [(i, GLYPH_DATA[i]) for i in range(min(10, len(GLYPH_DATA)))]
                if results:
                    glyph_id = results[0][0]

            if glyph_id is not None and 0 <= glyph_id < len(GLYPH_DATA):
                # Set glyph action
                # Note: For now, we create a basic Action with the change_glyph name
                # The glyph_id parameter will need to be handled separately
                # TODO: Enhance Action class to support parameters
                self.state.user_action = Action(name="change_glyph")
                self.state.should_step = True
                self._exit_glyph_picker()
        elif ch == "\x1b":  # Escape
            self._exit_glyph_picker()
        elif ch == "\x7f" or ch == "\x08":  # Backspace
            self._glyph_query = self._glyph_query[:-1] if self._glyph_query else ""
        elif ch == "[":
            # Cycle to previous agent
            self._state.select_previous_agent(self._sim.state.num_agents)
        elif ch == "]":
            # Cycle to next agent
            self._state.select_next_agent(self._sim.state.num_agents)
        elif ch and ch.isprintable():
            self._glyph_query = self._glyph_query + ch

        return True  # Always return True to block other components

    def update(self) -> None:
        """Render the glyph picker panel using current state."""
        # Always render when in GLYPH_PICKER mode, regardless of sidebar visibility
        in_picker_mode = self._state.mode == RenderMode.GLYPH_PICKER

        if not in_picker_mode and not self.state.is_sidebar_visible("glyph_picker"):
            self._panel.clear()
            return

        lines = self._build_lines(self._glyph_query)
        self._panel.set_content(lines)

    def _build_lines(self, query: str) -> list[str]:
        """Build glyph picker display as plain text lines.

        Args:
            query: Current search query

        Returns:
            List of strings for display
        """
        width = self._width if self._width else 40
        lines = []

        if not GLYPH_DATA:
            lines.append("Glyph Picker".ljust(width))
            lines.append("-" * width)
            lines.append("GLYPH_DATA not available".ljust(width))
            lines.append("Install cogames.cogs_vs_clips".ljust(width))
            return lines

        # Header
        agent_info = f" [Agent {self._state.selected_agent}]" if self._state.selected_agent is not None else ""
        header = f"Glyph Picker{agent_info}: {query}"
        lines.append(header[:width].ljust(width))
        lines.append("-" * min(width, 40))

        # Get matches using fuzzy name search only
        if query and search_glyphs:
            results = search_glyphs(query)[:10]
        else:
            # Show first 5 glyphs when no query
            if GLYPH_DATA:
                results = [(i, GLYPH_DATA[i]) for i in range(min(5, len(GLYPH_DATA)))]
            else:
                results = []

        if results:
            for idx, (_glyph_id, glyph) in enumerate(results):
                # Highlight the first row (what will be selected on Enter)
                if idx == 0:
                    line = f"> {glyph.name:<{width - 10}} {glyph.symbol:>5}"
                else:
                    line = f"  {glyph.name:<{width - 10}} {glyph.symbol:>5}"
                lines.append(line[:width].ljust(width))
        else:
            lines.append("(no matches)".ljust(width))

        # Add help text
        lines.append("")
        lines.append("[]=Agent  Enter=OK  Esc=Cancel".ljust(width))

        return lines

    def _exit_glyph_picker(self) -> None:
        """Exit glyph picker mode and reset query."""
        self._glyph_query = ""
        self.state.exit_glyph_picker()
