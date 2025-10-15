"""Symbols table component for miniscope renderer."""

from typing import List

from rich import box
from rich.table import Table

from mettagrid import MettaGridEnv
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState

from .base import MiniscopeComponent


class SymbolsTableComponent(MiniscopeComponent):
    """Component for displaying game symbols and their meanings."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
        max_rows: int = 10,
    ):
        """Initialize the symbols table component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
            max_rows: Maximum number of rows to display
        """
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.sidebar)
        self._max_rows = max_rows

    def _get_symbol_map(self) -> dict[str, str]:
        """Get symbol map from state."""
        return self.state.symbol_map if self.state else {}

    def update(self) -> List[str]:
        """Render the symbols table.

        Returns:
            List of strings representing the symbols table
        """
        symbol_map = self._get_symbol_map()
        if not symbol_map:
            return ["[Symbols: No symbol map]"]

        table = self._build_table()
        return table

    def _build_table(self) -> Table:
        """Build the symbols table.

        Returns:
            Rich Table object
        """
        table = Table(
            title="Symbols",
            show_header=False,
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._width,
        )
        table.add_column("Symbol", no_wrap=True, style="white", width=3)
        table.add_column("Name", style="cyan", overflow="ellipsis", width=15)
        table.add_column("Symbol", no_wrap=True, style="white", width=3)
        table.add_column("Name", style="cyan", overflow="ellipsis", width=15)

        # Build list of (symbol, name) tuples from symbol_map, sorted by name
        symbols_list = []
        seen_names = set()

        symbol_map = self._get_symbol_map()
        for name, symbol in sorted(symbol_map.items()):
            # Skip special entries
            if name in ["empty", "cursor", "?"] or not symbol:
                continue
            # Skip if we've seen this base name already
            base_name = name.split(".")[0]
            if base_name in seen_names:
                continue
            seen_names.add(base_name)

            # Format display name
            display_name = base_name.replace("_", " ").title()
            symbols_list.append((symbol, display_name))

        # Add rows in two columns
        for i in range(min(self._max_rows, (len(symbols_list) + 1) // 2)):
            left_idx = i
            right_idx = i + self._max_rows

            left_symbol, left_name = symbols_list[left_idx] if left_idx < len(symbols_list) else ("", "")
            right_symbol, right_name = symbols_list[right_idx] if right_idx < len(symbols_list) else ("", "")

            table.add_row(left_symbol, left_name, right_symbol, right_name)

        # If there are more symbols, add ellipsis
        if len(symbols_list) > self._max_rows * 2:
            remaining = len(symbols_list) - self._max_rows * 2
            table.add_row("⋯", f"({remaining} more)", "", "")

        return table
