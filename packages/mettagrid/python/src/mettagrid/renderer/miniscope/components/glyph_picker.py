"""Glyph picker component for miniscope renderer."""

from typing import Optional, Tuple

from rich import box
from rich.table import Table

from mettagrid import MettaGridEnv
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, RenderMode

from .base import MiniscopeComponent

try:
    from cogames.cogs_vs_clips.glyphs import GLYPH_DATA, search_glyphs
except ImportError:
    GLYPH_DATA = None
    search_glyphs = None


class GlyphPickerComponent(MiniscopeComponent):
    """Component for glyph selection interface."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the glyph picker component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(env=env, state=state, panels=panels)
        self._panel = panels.sidebar
        self._glyph_query: str = ""

    def update(self) -> Table:
        """Render the glyph picker panel using current state.

        Returns:
            Rich Table with glyph picker interface
        """

        # Handle input if in glyph picker mode
        if self._state.mode == RenderMode.GLYPH_PICKER:
            self._handle_input()

        return self._build_table(self._glyph_query)

    def _handle_input(self) -> None:
        """Handle user input for glyph picker."""
        if not self.state or not self.state.user_input:
            return

        ch = self.state.user_input
        query = self._glyph_query

        if ch == "\n" or ch == "\r":
            # Enter - confirm selection
            glyph_id = None
            if query.isdigit():
                glyph_id = int(query)
            elif query and search_glyphs:
                results = search_glyphs(query)
                if results:
                    glyph_id = results[0][0]

            if glyph_id is not None and 0 <= glyph_id < len(GLYPH_DATA):
                # Set glyph action
                if self.env and "change_glyph" in self.env.action_names:
                    change_glyph_idx = self.env.action_names.index("change_glyph")
                    self.state.user_action = (change_glyph_idx, glyph_id)
                    self.state.should_step = True
                self._exit_glyph_picker()
        elif ch == "\x1b":  # Escape
            self._exit_glyph_picker()
        elif ch == "\x7f" or ch == "\x08":  # Backspace
            self._glyph_query = query[:-1] if query else ""
        elif ch and ch.isprintable():
            self._glyph_query = query + ch

    def _build_table(self, query: str) -> Table:
        """Build the glyph picker table.

        Args:
            query: Current search query

        Returns:
            Rich Table object
        """
        # Create table with border
        table = Table(
            title=f"Glyph: {query}",
            show_header=False,
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._width,
        )
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Symbol", no_wrap=True)
        table.add_column("Name", style="white")

        # Get matches
        if query:
            # Check if query is numeric - support partial matches
            if query.isdigit():
                # Find all glyphs whose ID starts with the query
                results = []
                for i, glyph in enumerate(GLYPH_DATA):
                    if str(i).startswith(query):
                        results.append((i, glyph))
                        if len(results) >= 5:
                            break
            else:
                results = search_glyphs(query)[:5]
        else:
            # Show first 5 glyphs when no query
            results = [(i, GLYPH_DATA[i]) for i in range(min(5, len(GLYPH_DATA)))]

        if results:
            for glyph_id, glyph in results:
                table.add_row(str(glyph_id), glyph.symbol, glyph.name)
        else:
            table.add_row("", "", "(no matches)")

        # Add help text row
        table.add_row("", "", "")
        table.add_row("Enter=OK", "", "Esc=Cancel")

        return table

    def _exit_glyph_picker(self) -> None:
        """Exit glyph picker mode and reset query."""
        self.state.mode = RenderMode.FOLLOW
        self._glyph_query = ""

    def process_input(self, query: str, char: str) -> Tuple[str, Optional[int]]:
        """Process keyboard input for the glyph picker (legacy method).

        Args:
            query: Current query string
            char: Character input

        Returns:
            Tuple of (new_query, selected_glyph_id or None)
        """
        if char == "\n" or char == "\r":
            # Enter - confirm selection
            glyph_id = None
            if query.isdigit():
                glyph_id = int(query)
            elif query and search_glyphs:
                results = search_glyphs(query)
                if results:
                    glyph_id = results[0][0]

            if glyph_id is not None and 0 <= glyph_id < len(GLYPH_DATA):
                return "", glyph_id
            return query, None
        elif char == "\x1b":  # Escape
            return "", None  # Cancel
        elif char == "\x7f" or char == "\x08":  # Backspace
            return query[:-1] if query else "", None
        elif char.isprintable():
            return query + char, None
        return query, None
