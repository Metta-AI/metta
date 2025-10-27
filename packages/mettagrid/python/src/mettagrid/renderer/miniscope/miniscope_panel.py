"""Panel system for miniscope renderer."""

from typing import List, Optional, Union

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SIDEBAR_WIDTH = 46
"""Default character width allocated to the sidebar stack."""

LAYOUT_PADDING = 4
"""Horizontal padding between the map column and sidebar."""

RESERVED_VERTICAL_LINES = 6
"""Terminal rows reserved for static chrome (header/footer, spacing)."""


class MiniscopePanel:
    """Panel container for miniscope display areas."""

    def __init__(
        self,
        name: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: Optional[str] = None,
        border: bool = False,
    ):
        """Initialize a miniscope panel.

        Args:
            name: Panel identifier
            width: Fixed width in characters
            height: Fixed height in lines
            title: Optional title for the panel
            border: Whether to show a border
        """
        self.name = name
        self.width = width
        self.height = height
        self.title = title
        self.border = border
        self._content: List[str] = []
        self._rich_content: Optional[Union[Table, Panel, Text]] = None

    def set_content(self, content: Union[List[str], Table, Panel, Text]) -> None:
        """Set panel contents from plain text or a Rich renderable."""
        if isinstance(content, list):
            self._content = content
            self._rich_content = None
        else:
            self._content = []
            self._rich_content = content

    def get_content(self) -> List[str]:
        """Return panel content as a list of strings."""
        if self._rich_content:
            # Convert Rich object to strings
            console = Console(width=self.width or 80, legacy_windows=False)
            with console.capture() as capture:
                console.print(self._rich_content)
            return capture.get().split("\n")
        return self._content

    def get_rich_content(self) -> Optional[Union[Table, Panel, Text]]:
        """Return the current Rich renderable content, if any."""
        return self._rich_content

    def clear(self) -> None:
        """Reset stored content."""
        self._content = []
        self._rich_content = None

    def is_empty(self) -> bool:
        """Return True when the panel stores no content."""
        return not self._content and not self._rich_content

    def size(self) -> tuple[Optional[int], Optional[int]]:
        """Return the configured (width, height)."""
        return (self.width, self.height)

    def render(self) -> List[str]:
        """Render panel content and apply width/height constraints."""
        lines = self.get_content()

        # Apply height constraints
        if self.height:
            if len(lines) < self.height:
                # Pad with empty lines
                padding = self.height - len(lines)
                lines = lines + [""] * padding
            elif len(lines) > self.height:
                # Truncate
                lines = lines[: self.height]

        # Apply width constraints
        if self.width:
            formatted_lines = []
            for line in lines:
                if len(line) < self.width:
                    # Pad with spaces
                    formatted_lines.append(line + " " * (self.width - len(line)))
                elif len(line) > self.width:
                    # Truncate
                    formatted_lines.append(line[: self.width])
                else:
                    formatted_lines.append(line)
            lines = formatted_lines

        return lines


class PanelLayout:
    """Layout manager for miniscope panels."""

    def __init__(self, console: Console):
        """Initialize the panel layout.

        Args:
            console: Rich Console instance for rendering
        """
        self.console = console
        self.panels: dict[str, MiniscopePanel] = {}

        # Create standard panels
        self.header = MiniscopePanel("header", height=2)
        self.footer = MiniscopePanel("footer", height=2)
        self.map_view = MiniscopePanel("map_view")

        # Sidebar configuration
        self._sidebar_width = SIDEBAR_WIDTH
        self._sidebar_panels: dict[str, MiniscopePanel] = {}
        self._sidebar_order: list[str] = []

        # Register core panels
        self.panels["header"] = self.header
        self.panels["footer"] = self.footer
        self.panels["map_view"] = self.map_view

        # Live display for smooth updates without flicker
        self._live: Optional[Live] = None

    def get_panel(self, name: str) -> Optional[MiniscopePanel]:
        """Look up a panel by name."""
        return self.panels.get(name)

    def add_panel(self, panel: MiniscopePanel) -> None:
        """Register a custom panel."""
        self.panels[panel.name] = panel

    def register_sidebar_panel(self, name: str, title: Optional[str] = None) -> MiniscopePanel:
        """Create or return a named sidebar panel."""
        if name in self._sidebar_panels:
            panel = self._sidebar_panels[name]
            if title is not None:
                panel.title = title
            return panel

        panel = MiniscopePanel(name=f"sidebar.{name}", width=self._sidebar_width, title=title)
        self._sidebar_panels[name] = panel
        self._sidebar_order.append(name)
        self.panels[panel.name] = panel
        return panel

    def get_sidebar_panel(self, name: str) -> Optional[MiniscopePanel]:
        """Fetch a sidebar panel by logical name."""
        return self._sidebar_panels.get(name)

    def reset_sidebar_panels(self) -> None:
        """Remove all registered sidebar panels."""
        for panel in self._sidebar_panels.values():
            if panel.name in self.panels:
                del self.panels[panel.name]
        self._sidebar_panels.clear()
        self._sidebar_order.clear()

    def clear_all(self) -> None:
        """Clear all panel contents."""
        for panel in self.panels.values():
            panel.clear()

    def start_live(self) -> None:
        """Start live display mode for flicker-free updates."""
        if self._live is None:
            self._live = Live(console=self.console, refresh_per_second=60, screen=True)
            self._live.start()

    def stop_live(self) -> None:
        """Stop live display mode."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _compose_sidebar_content(self) -> Union[str, Text]:
        """Build the renderable content for the sidebar stack."""
        combined_lines: list[str] = []

        for name in self._sidebar_order:
            panel = self._sidebar_panels.get(name)
            if not panel:
                continue

            lines = panel.render()
            if not lines:
                continue

            if combined_lines:
                combined_lines.append("")

            combined_lines.extend(lines)

        if not combined_lines:
            return ""

        return Text("\n".join(combined_lines))

    def render_to_console(self) -> None:
        """Render the complete layout to console using Rich Table."""
        # Create main layout table with header, map/info panes, and footer
        layout = Table.grid(padding=0, expand=True)
        layout.add_column(ratio=1)  # Single column for full width content

        # Add header
        header_content = self.header.get_rich_content() or "\n".join(self.header.get_content())
        layout.add_row(header_content)

        # Get sidebar content to determine if we need to show it
        sidebar_content = self._compose_sidebar_content()

        # Create horizontal layout - conditionally include sidebar
        main_row = Table.grid(padding=0, expand=True)

        map_content = self.map_view.get_rich_content() or "\n".join(self.map_view.render())

        if sidebar_content:
            # Show map + border + sidebar
            main_row.add_column(ratio=1, overflow="ignore")  # Map column
            main_row.add_column(width=1, no_wrap=True, overflow="ignore")  # Border column
            main_row.add_column(width=self._sidebar_width, no_wrap=True, overflow="ignore")  # Sidebar column

            # Count lines in map to match height
            map_lines = str(map_content).count("\n") + 1 if map_content else 1
            border_content = "\n".join(["|"] * map_lines)

            main_row.add_row(map_content, border_content, sidebar_content)
        else:
            # Show map only (full width)
            main_row.add_column(ratio=1, overflow="ignore")  # Map column only
            main_row.add_row(map_content)

        layout.add_row(main_row)

        # Add footer
        footer_content = self.footer.get_rich_content() or "\n".join(self.footer.get_content())
        layout.add_row(footer_content)

        # Update live display if active, otherwise clear and print
        if self._live is not None:
            self._live.update(layout)
        else:
            self.console.clear()
            self.console.print(layout)
