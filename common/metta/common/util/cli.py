"""
Common utilities for CLI scripts.
"""

import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from metta.common.util.text_styles import blue, bold, cyan, green, magenta, red, yellow


class Spinner:
    """A simple CLI spinner for showing progress."""

    def __init__(
        self,
        message: str = "Processing",
        spinner_chars: Optional[list[str]] = None,
        style: Optional[Callable[[str], str]] = None,
    ):
        self.message = message
        self.spinner_chars = spinner_chars or ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.style = style or yellow  # Default to yellow if no style specified
        self._thread = None
        self._stop_event = threading.Event()
        self._spinner_index = 0

    def _spin(self):
        """The spinning animation loop."""
        max_line_length = 0  # Track the longest line we've written

        while not self._stop_event.is_set():
            char = self.spinner_chars[self._spinner_index % len(self.spinner_chars)]
            # Clear line and print styled spinner with message
            styled_char = self.style(char)
            current_line = f"{styled_char} {self.message}"

            # Calculate how much we need to clear
            current_length = len(char) + 1 + len(self.message)  # char + space + message
            max_line_length = max(max_line_length, current_length)

            # Clear the entire line up to the maximum length we've written
            sys.stdout.write(f"\r{' ' * max_line_length}\r")
            sys.stdout.write(current_line)
            sys.stdout.flush()

            self._spinner_index += 1
            time.sleep(0.1)

        # Clear the spinner line when done
        sys.stdout.write(f"\r{' ' * max_line_length}\r")
        sys.stdout.flush()

    def start(self):
        """Start the spinner."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spin)
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        """Stop the spinner."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()

    def update_message(self, message: str):
        """Update the spinner message."""
        self.message = message


@contextmanager
def spinner(
    message: str = "Processing", spinner_chars: Optional[list[str]] = None, style: Optional[Callable[[str], str]] = None
) -> Iterator[Spinner]:
    """
    Context manager for showing a spinner during long operations.

    Args:
        message: The message to display next to the spinner
        spinner_chars: Custom spinner characters (optional)
        style: Style function to apply to the spinner (e.g., cyan, yellow, bold, bold_green)

    Usage:
        with spinner("Loading data"):
            # Do some long operation
            time.sleep(5)

        # With custom style:
        with spinner("Processing", style=yellow):
            time.sleep(2)

        # With bold style:
        with spinner("Important operation", style=bold):
            time.sleep(2)

        # Or update the message during operation:
        with spinner("Processing", style=green) as sp:
            sp.update_message("Processing step 1")
            time.sleep(2)
            sp.update_message("Processing step 2")
            time.sleep(2)
    """
    sp = Spinner(message, spinner_chars, style)
    sp.start()
    try:
        yield sp
    finally:
        sp.stop()


def get_user_confirmation(prompt: str = "Should we proceed?") -> bool:
    """Get user confirmation before proceeding with an action."""

    response = input(f"{prompt} (Y/n): ").strip().lower()
    if response not in ["", "y", "yes"]:
        print(yellow("Action cancelled by user."))
        return False

    return True


def sh(
    cmd: list[str],
    show_spinner: bool = False,
    spinner_message: Optional[str] = None,
    spinner_style: Optional[Callable[[str], str]] = None,
    **kwargs,
) -> str:
    """
    Run a command and return its stdout (raises if the command fails).

    Args:
        cmd: Command to run as a list of strings
        show_spinner: Whether to show a spinner while the command runs
        spinner_message: Custom spinner message (defaults to the command)
        spinner_style: Style function for the spinner (e.g., cyan, yellow, bold)
        **kwargs: Additional arguments passed to subprocess
    """
    if show_spinner:
        message = spinner_message or f"Running: {' '.join(cmd[:3])}..."
        with spinner(message, style=spinner_style):
            return subprocess.check_output(cmd, text=True, **kwargs).strip()
    else:
        return subprocess.check_output(cmd, text=True, **kwargs).strip()


def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def main():
    """Demo the spinner functionality."""
    print(cyan("CLI Spinner Demo"))
    print(cyan("=" * 40))

    # Demo 1: Basic spinner
    print(f"\n{yellow('1. Basic spinner for 3 seconds:')}")
    with spinner("Loading data"):
        time.sleep(3)
    print(green("✓ Done!"))

    # Demo 2: Spinner with message updates
    print(f"\n{yellow('2. Spinner with changing messages:')}")
    with spinner("Initializing", style=blue) as sp:
        time.sleep(1)
        sp.update_message("Connecting to database")
        time.sleep(1)
        sp.update_message("Fetching records")
        time.sleep(1)
        sp.update_message("Processing data")
        time.sleep(1)
    print(green("✓ Complete!"))

    # Demo 3: Different spinner styles with different text styles
    print(f"\n{yellow('3. Different spinner styles and text styles:')}")
    spinner_styles = [
        (["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"], f"{cyan('cyan')} Dots", cyan),
        (["|", "/", "-", "\\"], f"{yellow('yellow')} Slash", yellow),
        (["◐", "◓", "◑", "◒"], f"{green('green')} Circle", green),
        (["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"], f"{magenta('magenta')} Bar", magenta),
    ]

    for chars, name, text_style in spinner_styles:
        with spinner(f"Testing {name} spinner", spinner_chars=chars, style=text_style):
            time.sleep(2)
        print(green(f"✓ {name} complete!"))

    # Demo 4: Interactive demo with bold green spinner
    print(f"\n{yellow('4. Interactive demo:')}")
    with spinner("Press Enter to continue", style=lambda x: bold(green(x))):
        input()
    print(green("✓ Thanks for trying the spinner!"))

    # Demo 5: Command execution with spinner
    print(f"\n{yellow('5. Running command with spinner:')}")
    try:
        result = sh(
            ["echo", "Hello from subprocess"],
            show_spinner=True,
            spinner_message="Executing command",
            spinner_style=green,
        )
        print(green(f"✓ Command output: {result}"))
    except Exception as e:
        print(red(f"✗ Command failed: {e}"))

    # Demo 6: Error scenario with red spinner
    print(f"\n{yellow('6. Error handling demo:')}")
    try:
        with spinner("Simulating error", style=red):
            time.sleep(1)
            raise ValueError("Something went wrong!")
    except ValueError:
        print(red("✗ Error occurred as expected"))

    # Demo 7: Bold spinner
    print(f"\n{yellow('7. Bold spinner demo:')}")
    with spinner("Important operation in progress", style=bold):
        time.sleep(2)
    print(green("✓ Critical operation completed!"))


if __name__ == "__main__":
    main()
