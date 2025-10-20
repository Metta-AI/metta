"""Tests for MiniscopePanel and PanelLayout classes."""

from rich.console import Console
from rich.table import Table
from rich.text import Text

from mettagrid.renderer.miniscope.miniscope_panel import MiniscopePanel, PanelLayout


class TestMiniscopePanel:
    """Test suite for MiniscopePanel functionality."""

    def test_panel_initialization(self):
        """Test panel initialization with different configurations."""
        # Test basic panel
        panel = MiniscopePanel("test", width=50, height=10)
        assert panel.name == "test"
        assert panel.width == 50
        assert panel.height == 10
        assert panel.title is None
        assert panel.border is False

        # Test panel with title and border
        panel2 = MiniscopePanel("header", width=80, height=2, title="Status", border=True)
        assert panel2.title == "Status"
        assert panel2.border is True

    def test_panel_size_method(self):
        """Test the size() method returns correct dimensions."""
        panel = MiniscopePanel("test", width=40, height=20)
        width, height = panel.size()
        assert width == 40
        assert height == 20

        # Test with None dimensions
        panel2 = MiniscopePanel("dynamic")
        width, height = panel2.size()
        assert width is None
        assert height is None

    def test_panel_set_content_strings(self):
        """Test setting panel content with list of strings."""
        panel = MiniscopePanel("test")
        content = ["Line 1", "Line 2", "Line 3"]
        panel.set_content(content)
        assert panel.get_content() == content
        assert panel.get_rich_content() is None

    def test_panel_set_content_rich_table(self):
        """Test setting panel content with Rich Table."""
        panel = MiniscopePanel("test", width=50)
        table = Table(title="Test Table")
        table.add_column("Column 1")
        table.add_column("Column 2")
        table.add_row("Value 1", "Value 2")

        panel.set_content(table)
        assert panel.get_rich_content() == table

        # Get as strings
        lines = panel.get_content()
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_panel_set_content_rich_text(self):
        """Test setting panel content with Rich Text."""
        panel = MiniscopePanel("test")
        text = Text("Test text with style", style="bold blue")
        panel.set_content(text)
        assert panel.get_rich_content() == text

        # Get as strings
        lines = panel.get_content()
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_panel_clear(self):
        """Test clearing panel content."""
        panel = MiniscopePanel("test")
        panel.set_content(["Line 1", "Line 2"])
        assert not panel.is_empty()

        panel.clear()
        assert panel.is_empty()
        assert panel.get_content() == []
        assert panel.get_rich_content() is None

    def test_panel_render_with_padding(self):
        """Test panel rendering with height padding."""
        panel = MiniscopePanel("test", width=20, height=5)
        panel.set_content(["Line 1", "Line 2"])

        rendered = panel.render()
        assert len(rendered) == 5  # Padded to height
        assert rendered[0] == "Line 1" + " " * 14  # Width padded to 20
        assert rendered[1] == "Line 2" + " " * 14  # Width padded to 20
        assert rendered[2] == " " * 20  # Padding
        assert rendered[3] == " " * 20  # Padding
        assert rendered[4] == " " * 20  # Padding

    def test_panel_render_with_truncation(self):
        """Test panel rendering with height truncation."""
        panel = MiniscopePanel("test", width=20, height=2)
        panel.set_content(["Line 1", "Line 2", "Line 3", "Line 4"])

        rendered = panel.render()
        assert len(rendered) == 2  # Truncated to height
        assert rendered[0] == "Line 1" + " " * 14  # Width padded to 20
        assert rendered[1] == "Line 2" + " " * 14  # Width padded to 20

    def test_panel_render_with_width_padding(self):
        """Test panel rendering with width padding."""
        panel = MiniscopePanel("test", width=10, height=1)
        panel.set_content(["Hi"])

        rendered = panel.render()
        assert len(rendered) == 1
        assert rendered[0] == "Hi        "  # Padded to 10 chars
        assert len(rendered[0]) == 10

    def test_panel_render_with_width_truncation(self):
        """Test panel rendering with width truncation."""
        panel = MiniscopePanel("test", width=5, height=1)
        panel.set_content(["This is a long line"])

        rendered = panel.render()
        assert len(rendered) == 1
        assert rendered[0] == "This "  # Truncated to 5 chars
        assert len(rendered[0]) == 5


class TestPanelLayout:
    """Test suite for PanelLayout functionality."""

    def test_layout_initialization(self):
        """Test that PanelLayout creates standard panels."""
        layout = PanelLayout(Console())

        # Check standard panels exist
        assert layout.header is not None
        assert layout.footer is not None
        assert layout.map_view is not None
        assert layout.sidebar is not None

        # Check they're registered
        assert layout.get_panel("header") == layout.header
        assert layout.get_panel("footer") == layout.footer
        assert layout.get_panel("map_view") == layout.map_view
        assert layout.get_panel("sidebar") == layout.sidebar

        # Check dimensions
        assert layout.header.height == 2
        assert layout.footer.height == 2
        assert layout.sidebar.width == 46

    def test_layout_add_custom_panel(self):
        """Test adding custom panels to layout."""
        layout = PanelLayout(Console())
        custom_panel = MiniscopePanel("custom", width=30, height=10)

        layout.add_panel(custom_panel)
        assert layout.get_panel("custom") == custom_panel
        assert len(layout.panels) == 5  # 4 standard + 1 custom

    def test_layout_clear_all(self):
        """Test clearing all panel contents."""
        layout = PanelLayout(Console())

        # Add content to panels
        layout.header.set_content(["Header line"])
        layout.footer.set_content(["Footer line"])
        layout.map_view.set_content(["Map content"])
        layout.sidebar.set_content(["Sidebar content"])

        # Clear all
        layout.clear_all()

        # Check all are empty
        assert layout.header.is_empty()
        assert layout.footer.is_empty()
        assert layout.map_view.is_empty()
        assert layout.sidebar.is_empty()

    def test_layout_render_basic(self):
        """Test basic rendering of panels."""
        console = Console()
        layout = PanelLayout(console)

        # Set some content
        layout.header.set_content(["Header Line 1", "Header Line 2"])
        layout.footer.set_content(["Footer Line 1", "Footer Line 2"])
        layout.map_view.set_content(["Map Line 1", "Map Line 2", "Map Line 3"])
        layout.sidebar.set_content(["Sidebar Line 1", "Sidebar Line 2"])

        # Test that rendering works without errors
        layout.render_to_console()

        # Verify panels have content
        assert not layout.header.is_empty()
        assert not layout.footer.is_empty()
        assert not layout.map_view.is_empty()
        assert not layout.sidebar.is_empty()

    def test_layout_render_empty_panels(self):
        """Test rendering with some empty panels."""
        console = Console()
        layout = PanelLayout(console)

        # Only set map content
        layout.map_view.set_content(["Map content"])

        # Should render without errors even with empty panels
        layout.render_to_console()

        assert not layout.map_view.is_empty()
        assert layout.header.is_empty()
        assert layout.footer.is_empty()

    def test_layout_render_to_console(self):
        """Test rendering to console using Rich API."""
        console = Console()
        layout = PanelLayout(console)
        layout.header.set_content(["Test Header"])

        # Test that it renders without errors
        layout.render_to_console()

        # Verify header has content
        assert not layout.header.is_empty()

    def test_layout_render_with_different_content(self):
        """Test rendering when panels have different amounts of content."""
        console = Console()
        layout = PanelLayout(console)

        # Map has more lines than sidebar
        layout.map_view.set_content([f"Map Line {i}" for i in range(10)])
        layout.sidebar.set_content(["Sidebar Line 1", "Sidebar Line 2"])

        # Should render without errors
        layout.render_to_console()

        # Verify content is present
        assert len(layout.map_view.get_content()) == 10
        assert len(layout.sidebar.get_content()) == 2
