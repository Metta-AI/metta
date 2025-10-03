"""Help screen for miniscope interactive viewer."""

from rich import box
from rich.console import Console
from rich.table import Table


def show_help_screen() -> list[str]:
    """Generate full-screen help display with all commands.

    Returns:
        List of strings representing the help screen lines
    """
    console = Console()

    # Create help table
    table = Table(
        title="Miniscope Interactive Viewer - Help",
        show_header=True,
        box=box.ROUNDED,
        padding=(0, 1),
        title_style="bold cyan",
    )
    table.add_column("Key", style="yellow", no_wrap=True, width=15)
    table.add_column("Command", style="white", width=50)

    # General controls
    table.add_row("[bold cyan]General[/bold cyan]", "")
    table.add_row("?", "Show this help screen")
    table.add_row("Q", "Quit miniscope")
    table.add_row("Space", "Toggle pause/play")
    table.add_row("< or ,", "Decrease speed (lower SPS)")
    table.add_row("> or .", "Increase speed (higher SPS)")

    # Agent controls
    table.add_row("", "")
    table.add_row("[bold cyan]Agent Selection[/bold cyan]", "")
    table.add_row("[ or ]", "Select previous/next agent")
    table.add_row("M", "Toggle manual mode for selected agent")

    # Camera/View controls
    table.add_row("", "")
    table.add_row("[bold cyan]Camera & View[/bold cyan]", "")
    table.add_row("I/K", "Pan camera up/down (1 space)")
    table.add_row("J/L", "Pan camera left/right (1 space)")
    table.add_row("Shift+I/K/J/L", "Pan camera 10 spaces")
    table.add_row("S", "Toggle select mode (cursor navigation)")

    # Agent actions (manual mode)
    table.add_row("", "")
    table.add_row("[bold cyan]Agent Actions[/bold cyan]", "[dim](requires agent selection)[/dim]")
    table.add_row("W/A/S/D", "Move selected agent (N/W/S/E)")
    table.add_row("R", "Rest (noop action)")
    table.add_row("E", "Toggle emoji/glyph picker")

    # Select mode
    table.add_row("", "")
    table.add_row("[bold cyan]Select Mode[/bold cyan]", "[dim](press S to enter)[/dim]")
    table.add_row("I/K/J/L", "Move cursor to inspect objects")
    table.add_row("Shift+I/K/J/L", "Move cursor 10 spaces")
    table.add_row("S", "Exit select mode")

    # Glyph picker mode
    table.add_row("", "")
    table.add_row("[bold cyan]Glyph Picker[/bold cyan]", "[dim](press E to enter)[/dim]")
    table.add_row("Type text", "Filter glyphs by name")
    table.add_row("Up/Down", "Navigate glyph list")
    table.add_row("Enter", "Select glyph and change agent appearance")
    table.add_row("E or Esc", "Exit glyph picker")

    # Modes
    table.add_row("", "")
    table.add_row("[bold cyan]Modes[/bold cyan]", "")
    table.add_row("Follow", "Camera follows selected agent")
    table.add_row("Pan", "Camera free to pan (activated on I/J/K/L)")
    table.add_row("Select", "Cursor mode to inspect objects (press S)")
    table.add_row("Glyph Picker", "Choose agent emoji/glyph (press E)")

    # Manual mode explanation
    table.add_row("", "")
    table.add_row("[bold cyan]Manual Mode[/bold cyan]", "")
    table.add_row("", "When enabled (press M), agent ignores policy")
    table.add_row("", "and only responds to manual WASD/R commands.")
    table.add_row("", "Shown as 'Mode: MANUAL' in Agent Info.")

    # Capture the table as strings
    with console.capture() as capture:
        console.print(table)
        console.print("")
        console.print("[dim cyan]Press any key to return to miniscope[/dim cyan]", justify="center")

    return capture.get().split("\n")
