"""Panel system for miniscope renderer."""

from typing import List, Optional, Union

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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
        """Set the panel content.

        Args:
            content: Either list of strings or Rich renderable object
        """
        if isinstance(content, list):
            self._content = content
            self._rich_content = None
        else:
            self._content = []
            self._rich_content = content

    def get_content(self) -> List[str]:
        """Get the panel content as list of strings.

        Returns:
            List of strings representing the panel content
        """
        if self._rich_content:
            # Convert Rich object to strings
            console = Console(width=self.width or 80, legacy_windows=False)
            with console.capture() as capture:
                console.print(self._rich_content)
            return capture.get().split("\n")
        return self._content

    def get_rich_content(self) -> Optional[Union[Table, Panel, Text]]:
        """Get the Rich renderable content if available.

        Returns:
            Rich renderable object or None
        """
        return self._rich_content

    def clear(self) -> None:
        """Clear the panel content."""
        self._content = []
        self._rich_content = None

    def is_empty(self) -> bool:
        """Check if panel has content.

        Returns:
            True if panel is empty
        """
        return not self._content and not self._rich_content

    def size(self) -> tuple[Optional[int], Optional[int]]:
        """Get the panel size as (width, height).

        Returns:
            Tuple of (width, height) where None indicates dynamic sizing
        """
        return (self.width, self.height)

    def render(self) -> List[str]:
        """Render the panel to list of strings.

        Returns:
            List of strings with proper padding/truncation
        """
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
        self.sidebar = MiniscopePanel("sidebar", width=46)

        # Register panels
        self.panels["header"] = self.header
        self.panels["footer"] = self.footer
        self.panels["map_view"] = self.map_view
        self.panels["sidebar"] = self.sidebar

        # Live display for smooth updates without flicker
        self._live: Optional[Live] = None

    def get_panel(self, name: str) -> Optional[MiniscopePanel]:
        """Get a panel by name.

        Args:
            name: Panel name

        Returns:
            Panel instance or None
        """
        return self.panels.get(name)

    def add_panel(self, panel: MiniscopePanel) -> None:
        """Add a custom panel.

        Args:
            panel: Panel to add
        """
        self.panels[panel.name] = panel

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

    def render_to_console(self) -> None:
        """Render the complete layout to console using Rich Table."""
        # Create main layout table with header, map/info panes, and footer
        layout = Table.grid(padding=0, expand=True)
        layout.add_column(ratio=1)  # Single column for full width content

        # Add header
        header_content = self.header.get_rich_content() or "\n".join(self.header.get_content())
        layout.add_row(header_content)

        # Create horizontal layout for map (left) and info pane (right)
        main_row = Table.grid(padding=0, expand=True)
        main_row.add_column(ratio=1)  # Map pane (left, flexible width)
        main_row.add_column(width=46)  # Info pane (right, fixed width)

        # Get content for map and sidebar
        map_content = self.map_view.get_rich_content() or "\n".join(self.map_view.render())
        sidebar_content = self.sidebar.get_rich_content() or "\n".join(self.sidebar.render())

        main_row.add_row(map_content, sidebar_content)
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
