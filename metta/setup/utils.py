import importlib
import itertools
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from metta.common.util.text_styles import Fore, blue, bold, colorize, cyan, green, red, yellow

T = TypeVar("T")


def _format_message(message: str, indent: int = 0) -> str:
    """Apply dedent and strip to message."""
    return textwrap.indent(textwrap.dedent(message).strip(), " " * indent)


def success(message: str, indent: int = 0, **kwargs) -> None:
    print(green(_format_message(message, indent)), **kwargs)


def info(message: str, indent: int = 0, **kwargs) -> None:
    print(blue(_format_message(message, indent)), **kwargs)


def warning(message: str, indent: int = 0, **kwargs) -> None:
    print(yellow(_format_message(message, indent)), **kwargs)


def error(message: str, indent: int = 0, **kwargs) -> None:
    print(red(_format_message(message)), **kwargs)


def header(message: str, indent: int = 0) -> None:
    print(f"\n{bold(cyan(_format_message(message, indent)))}")


def step(message: str, indent: int = 0) -> None:
    print(colorize(_format_message(message, indent), Fore.WHITE))


def debug(message: str, indent: int = 0) -> None:
    print(colorize(_format_message(message, indent), Fore.LIGHTMAGENTA_EX))


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


def prompt_choice(prompt: str, choices: list[tuple[T, str]], default: T | None = None, current: T | None = None) -> T:
    """Prompt user to select from a list of choices with arrow key support.

    Args:
        prompt: The prompt message
        choices: List of (value, description) tuples
        default: Default choice if user presses Enter
        current: Current value to highlight

    Returns:
        The selected value
    """
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
        return _simple_prompt_choice(prompt, choices, default, current)


def _simple_prompt_choice(
    prompt: str, choices: list[tuple[T, str]], default: T | None = None, current: T | None = None
) -> T:
    """Simple numbered choice prompt without arrow keys."""
    header(prompt)
    for i, (value, desc) in enumerate(choices):
        markers = []
        if current is not None and value == current:
            markers.append("current")
        if default is not None and value == default:
            markers.append("default")

        marker = f" ({', '.join(markers)})" if markers else ""
        if current is not None and value == current:
            print(cyan(f"  {i + 1}. {desc}{marker}"))
        else:
            print(f"  {i + 1}. {desc}{marker}")

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
        package_name: The parent package name (e.g., 'metta.setup')
        subpackage: The subpackage name (e.g., 'components')
    """
    # Since we're in metta/setup/utils.py, we can use relative path
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
