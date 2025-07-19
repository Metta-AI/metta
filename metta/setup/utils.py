import importlib
import textwrap
from pathlib import Path
from typing import TypeVar

from metta.common.util.colorama import Fore, blue, bold, colorize, cyan, green, red, yellow

T = TypeVar("T")


def _format_message(message: str) -> str:
    """Apply dedent and strip to message."""
    return textwrap.dedent(message).strip()


def success(message: str, **kwargs) -> None:
    print(green(_format_message(message)), **kwargs)


def info(message: str, **kwargs) -> None:
    print(blue(_format_message(message)), **kwargs)


def warning(message: str, **kwargs) -> None:
    print(yellow(_format_message(message)), **kwargs)


def error(message: str, **kwargs) -> None:
    print(red(_format_message(message)), **kwargs)


def header(message: str) -> None:
    print(f"\n{bold(cyan(_format_message(message)))}")


def step(message: str) -> None:
    print(colorize(_format_message(message), Fore.WHITE))


def prompt_choice(prompt: str, choices: list[tuple[T, str]], default: T | None = None, current: T | None = None) -> T:
    """Prompt user to select from a list of choices.

    Args:
        prompt: The prompt message
        choices: List of (value, description) tuples
        default: Default choice if user presses Enter
        current: Current value to highlight

    Returns:
        The selected value
    """
    print(f"\n{prompt}")
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
