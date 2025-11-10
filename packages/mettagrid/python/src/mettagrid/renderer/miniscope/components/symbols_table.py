"""Symbols table component for miniscope renderer."""

from typing import TYPE_CHECKING

from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState

if TYPE_CHECKING:
    from mettagrid.simulator import Simulation

from .base import MiniscopeComponent


class SymbolsTableComponent(MiniscopeComponent):
    """Component for displaying game symbols and their meanings."""

    def __init__(
        self,
        sim: "Simulation",
        state: MiniscopeState,
        panels: PanelLayout,
        max_rows: int = 1000,
    ):
        """Initialize the symbols table component.

        Args:
            sim: MettaGrid simulator reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
            max_rows: Maximum number of rows to display (default 1000 for unlimited)
        """
        super().__init__(sim=sim, state=state, panels=panels)
        sidebar_panel = panels.get_sidebar_panel("symbols")
        if sidebar_panel is None:
            sidebar_panel = panels.register_sidebar_panel("symbols")
        self._set_panel(sidebar_panel)
        self._max_rows = max_rows

    def _get_symbol_map(self) -> dict[str, str]:
        """Get symbol map from state."""
        return self.state.symbol_map if self.state else {}

    def update(self) -> None:
        """Render the symbols table."""
        if not self.state.is_sidebar_visible("symbols"):
            self._panel.clear()
            return

        symbol_map = self._get_symbol_map()
        if not symbol_map:
            self._panel.set_content(["No symbol map available"])
            return

        entries = self._build_entries(symbol_map)
        lines = self._build_lines(entries)
        self._panel.set_content(lines)

    def _build_entries(self, symbol_map: dict[str, str]) -> list[tuple[str, str]]:
        """Create the list of displayable symbol entries."""
        entries: list[tuple[str, str]] = []
        seen_names = set()

        for name, symbol in sorted(symbol_map.items()):
            if name in ["empty", "cursor", "?"] or not symbol:
                continue
            base_name = name.split(".")[0]
            if base_name in seen_names:
                continue
            seen_names.add(base_name)

            display_name = base_name.replace("_", " ").title()
            entries.append((symbol, display_name))

        return entries

    def _build_lines(self, entries: list[tuple[str, str]]) -> list[str]:
        """Format entries into fixed-width lines for the sidebar."""
        if not entries:
            return ["Symbols", "(none)"]

        width = self._width if self._width else 40
        width = max(20, width)

        header = "Symbols"
        separator = "-" * min(width, 40)

        lines: list[str] = [header, separator]

        # Single column layout
        max_visible = self._max_rows
        visible_entries = entries[:max_visible]
        hidden_count = len(entries) - len(visible_entries)

        for symbol, name in visible_entries:
            entry_text = f"{symbol} {name}"
            lines.append(entry_text[:width].ljust(width))

        if hidden_count > 0:
            lines.append(f"(+{hidden_count} more)")

        return lines
