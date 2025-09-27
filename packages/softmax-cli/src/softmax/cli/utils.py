import functools
import importlib
import itertools
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from rich.console import Console

T = TypeVar("T")


@functools.cache
def get_console() -> Console:
    return Console()


def _format_message(message: str, indent: int = 0) -> str:
    return textwrap.indent(textwrap.dedent(message).strip(), " " * indent)


def colorize(message: str, color: str) -> str:
    return f"[{color}]{message}[/{color}]"


def _output(color: str, message: str, indent: int = 0, **kwargs) -> None:
    console = get_console()
    formatted = _format_message(message, indent)
    console.print(colorize(formatted, color), **kwargs)


success = functools.partial(_output, "green")
info = functools.partial(_output, "blue")
warning = functools.partial(_output, "yellow")
error = functools.partial(_output, "red")
header = functools.partial(_output, "bold cyan")
step = functools.partial(_output, "white")
debug = functools.partial(_output, "magenta")


@contextmanager
def spinner(message: str = "Loading..."):
    """Context manager that shows a spinner while executing code.

    Usage:
        with spinner("Checking status..."):
            # Do some work
            time.sleep(2)
    """
    spinner_chars = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    stop_spinner = threading.Event()

    def show_spinner():
        while not stop_spinner.is_set():
            print(f"\r{next(spinner_chars)} {message}", end="", flush=True)
            time.sleep(0.1)
        print("\r" + " " * (len(message) + 4) + "\r", end="", flush=True)  # Clear the line

    spinner_thread = threading.Thread(target=show_spinner)
    spinner_thread.start()

    try:
        yield
    finally:
        stop_spinner.set()
        spinner_thread.join()


def prompt_choice(
    prompt: str,
    choices: list[tuple[T, str]],
    default: T | None = None,
    current: T | None = None,
    non_interactive: bool = False,
) -> T:
    """Prompt user to select from a list of choices with arrow key support.

    Args:
        prompt: The prompt message
        choices: List of (value, description) tuples
        default: Default choice if user presses Enter
        current: Current value to highlight
        non_interactive: If True, automatically return default/current/first choice

    Returns:
        The selected value
    """
    # Handle non-interactive mode by delegating to simple fallback
    if non_interactive:
        return _simple_prompt_choice(prompt, choices, default, current, non_interactive)

    try:
        from simple_term_menu import TerminalMenu

        # Extract descriptions for menu display
        menu_entries = [f"{i + 1}. {desc}" for i, (_, desc) in enumerate(choices)]

        # Find initial selection
        cursor_index = 0
        if current is not None:
            for i, (value, _) in enumerate(choices):
                if value == current:
                    cursor_index = i
                    break
        elif default is not None:
            for i, (value, _) in enumerate(choices):
                if value == default:
                    cursor_index = i
                    break

        # Display header
        header(prompt)

        # Create menu
        terminal_menu = TerminalMenu(
            menu_entries,
            cursor_index=cursor_index,
            menu_cursor="▶ ",
            menu_cursor_style=("fg_cyan",),
            menu_highlight_style=("fg_cyan",),
            cycle_cursor=True,
            clear_screen=False,
        )

        menu_entry_index = terminal_menu.show()

        if menu_entry_index is None:
            # User cancelled
            raise KeyboardInterrupt()

        return choices[menu_entry_index][0]  # type: ignore

    except ImportError:
        # Fallback to simple prompt
        return _simple_prompt_choice(prompt, choices, default, current, non_interactive)


def _simple_prompt_choice(
    prompt: str,
    choices: list[tuple[T, str]],
    default: T | None = None,
    current: T | None = None,
    non_interactive: bool = False,
) -> T:
    """Simple numbered choice prompt without arrow keys."""
    # Handle non-interactive mode
    if non_interactive:
        if default is not None:
            info(f"{prompt} -> Using default: {default}")
            return default
        elif current is not None:
            info(f"{prompt} -> Using current: {current}")
            return current
        elif choices:
            choice_value = choices[0][0]
            info(f"{prompt} -> Using first option: {choice_value}")
            return choice_value
        else:
            raise ValueError("No valid choice available for non-interactive mode")

    header(prompt)
    for i, (value, desc) in enumerate(choices):
        markers = []
        if current is not None and value == current:
            markers.append("current")
        if default is not None and value == default:
            markers.append("default")

        marker = f" ({', '.join(markers)})" if markers else ""
        if current is not None and value == current:
            debug(f"{i + 1}. {desc}{marker}", indent=2)
        else:
            debug(f"{i + 1}. {desc}{marker}", indent=2)

    while True:
        try:
            choice = input("\nEnter your choice (1-{}): ".format(len(choices))).strip()
            if not choice and default is not None:
                return default
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
            else:
                warning(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            warning("Please enter a valid number")


def import_all_modules_from_subpackage(package_name: str, subpackage: str) -> None:
    """Import all Python modules from a subpackage directory.

    This is useful for auto-registering modules that use decorators.
    Works with PEP 420 namespace packages.

    Args:
        package_name: The parent package name (e.g., 'softmax.cli')
        subpackage: The subpackage name (e.g., 'components')
    """
    # Since we're in softmax/cli/utils.py, we can use relative path
    current_file = Path(__file__)
    setup_dir = current_file.parent
    subpackage_path = setup_dir / subpackage

    if not subpackage_path.exists():
        return

    # Import all Python files in the subpackage
    for module_file in subpackage_path.glob("*.py"):
        if module_file.stem != "__init__" and not module_file.stem.startswith("_"):
            module_name = f"{package_name}.{subpackage}.{module_file.stem}"
            try:
                _ = importlib.import_module(module_name)
            except ImportError:
                pass
