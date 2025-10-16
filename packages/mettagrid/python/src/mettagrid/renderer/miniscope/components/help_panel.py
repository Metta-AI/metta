"""Help panel component for miniscope renderer."""

from typing import List

from rich import box
from rich.table import Table
from rich.text import Text

from mettagrid import MettaGridEnv
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState
from mettagrid.renderer.miniscope.styles import chip_markup, gradient_title, surface_panel

from .base import MiniscopeComponent


class HelpPanelComponent(MiniscopeComponent):
    """Component for displaying help information."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the help panel component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.sidebar)

    def update(self) -> None:
        """Render the help panel."""
        table = self._build_help_table()
        panel = surface_panel(
            table,
            title=gradient_title("Quick Reference"),
            border_variant="alt",
            variant="dark",
        )
        self._panel.append_block(panel)

    def _build_help_table(self) -> Table:
        """Create a compact quick-reference table."""
        table = Table.grid(padding=(0, 0))
        table.add_column(justify="right", no_wrap=True)
        table.add_column()

        def add_section(header: str, items: List[tuple[str, str]]) -> None:
            table.add_row(Text(header, style="accent.dim"), Text("", style="muted"))
            for key, desc in items:
                table.add_row(Text.from_markup(chip_markup(key)), Text(desc, style="muted"))
            table.add_row(Text("", style="muted"), Text("", style="muted"))

        add_section(
            "Navigation",
            [
                ("I / K", "Camera north/south"),
                ("J / L", "Camera west/east"),
                ("Shift+IJKL", "Jump 10 tiles"),
                ("O", "Cycle follow → pan → inspect"),
            ],
        )

        add_section(
            "Agents",
            [
                ("[ / ]", "Cycle agent"),
                ("M", "Toggle manual"),
                ("WASD", "Manual move"),
                ("E", "Glyph picker"),
            ],
        )

        add_section(
            "Simulation",
            [
                ("Space", "Play / pause"),
                ("< / >", "Speed down / up"),
                ("Q", "Quit"),
            ],
        )

        return table

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
        table.add_row("", "o", "Cycle mode: Follow → Pan → Select")
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

        return self._table_to_lines(table)
