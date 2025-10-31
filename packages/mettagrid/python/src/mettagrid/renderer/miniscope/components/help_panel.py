"""Help panel component for miniscope renderer."""

from typing import List

from rich import box
from rich.table import Table

from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState
from mettagrid.simulator import Simulation

from .base import MiniscopeComponent


class HelpPanelComponent(MiniscopeComponent):
    """Component for displaying help information."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the help panel component.

        Args:
            sim: MettaGrid simulator reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(sim=sim, state=state, panels=panels)
        sidebar_panel = panels.get_sidebar_panel("help")
        if sidebar_panel is None:
            sidebar_panel = panels.register_sidebar_panel("help")
        self._set_panel(sidebar_panel)

    def update(self) -> None:
        """Render the help panel."""
        if not self.state.is_sidebar_visible("help"):
            self._panel.clear()
            return

        lines = []
        width = self._width if self._width else 70
        lines.append("=" * width)
        lines.append(" " * ((width - 22) // 2) + "🎮 MINISCOPE HELP 🎮")
        lines.append("=" * width)
        lines.append("")

        # Navigation section
        lines.append("📍 NAVIGATION & VIEWING")
        lines.append("-" * 30)
        lines.append("  i/k     - Move camera/cursor up/down (1 space)")
        lines.append("  j/l     - Move camera/cursor left/right (1 space)")
        lines.append("  I/K     - Move camera/cursor up/down (10 spaces)")
        lines.append("  J/L     - Move camera/cursor left/right (10 spaces)")
        lines.append("  f       - Follow mode (camera tracks selected agent)")
        lines.append("  p       - Pan mode (free camera movement)")
        lines.append("  t       - Select mode (move cursor to inspect objects)")
        lines.append("")

        # Agent control section
        lines.append("🤖 AGENT CONTROL")
        lines.append("-" * 30)
        lines.append("  [/]     - Select previous/next agent")
        lines.append("  m       - Toggle manual mode for selected agent")
        lines.append("  w/a/s/d - Move selected agent (North/West/South/East)")
        lines.append("  r       - Rest (no action)")
        lines.append("  e       - Change glyph/emote")
        lines.append("")

        # Simulation section
        lines.append("⚙️ SIMULATION")
        lines.append("-" * 30)
        lines.append("  SPACE   - Play/Pause simulation")
        lines.append("  </>     - Decrease/Increase speed")
        lines.append("")

        # System section
        lines.append("💻 SYSTEM")
        lines.append("-" * 30)
        lines.append("  ?       - Show this help")
        lines.append("  q       - Quit")
        lines.append("")

        # Sidebar toggle section
        lines.append("📚 SIDEBAR PANELS")
        lines.append("-" * 30)
        lines.append("  1       - Toggle Agent info")
        lines.append("  2       - Toggle Object info")
        lines.append("  3       - Toggle Symbols list")
        lines.append("")

        lines.append("=" * width)
        lines.append(" " * ((width - 24) // 2) + "Press any key to continue")
        lines.append("=" * width)

        self._panel.set_content(lines)

    def render_as_table(self) -> List[str]:
        """Render the help panel as a Rich table.

        Returns:
            List of strings representing the help table
        """
        table = Table(
            title="🎮 MINISCOPE HELP 🎮",
            show_header=True,
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._width,
        )
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Key", style="yellow", no_wrap=True)
        table.add_column("Action", style="white")

        # Navigation
        table.add_row("Navigation", "i/k", "Move camera/cursor up/down (1 space)")
        table.add_row("", "j/l", "Move camera/cursor left/right (1 space)")
        table.add_row("", "I/K", "Move camera/cursor up/down (10 spaces)")
        table.add_row("", "J/L", "Move camera/cursor left/right (10 spaces)")
        table.add_row("", "f", "Switch to Follow mode")
        table.add_row("", "p", "Switch to Pan mode")
        table.add_row("", "t", "Switch to Select mode")
        table.add_row("", "", "")

        # Agent control
        table.add_row("Agent", "[/]", "Select previous/next agent")
        table.add_row("", "m", "Toggle manual mode")
        table.add_row("", "w/a/s/d", "Move agent (North/West/South/East)")
        table.add_row("", "r", "Rest (no action)")
        table.add_row("", "e", "Change glyph/emote")
        table.add_row("", "", "")

        # Simulation
        table.add_row("Simulation", "SPACE", "Play/Pause")
        table.add_row("", "</>", "Decrease/Increase speed")
        table.add_row("", "", "")

        # System
        table.add_row("System", "?", "Show help")
        table.add_row("", "q", "Quit")
        table.add_row("", "", "")

        # Sidebar toggles
        table.add_row("Sidebar", "1", "Toggle Agent info")
        table.add_row("", "2", "Toggle Object info")
        table.add_row("", "3", "Toggle Symbols list")

        return self._table_to_lines(table)
