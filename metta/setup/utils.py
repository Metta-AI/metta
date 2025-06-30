import importlib
import textwrap
from pathlib import Path

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def _format_message(message: str) -> str:
    """Apply dedent and strip to message."""
    return textwrap.dedent(message).strip()


def success(message: str, **kwargs) -> None:
    print(f"{Fore.GREEN}{_format_message(message)}{Style.RESET_ALL}", **kwargs)


def info(message: str, **kwargs) -> None:
    print(f"{Fore.BLUE}{_format_message(message)}{Style.RESET_ALL}", **kwargs)


def warning(message: str, **kwargs) -> None:
    print(f"{Fore.YELLOW}{_format_message(message)}{Style.RESET_ALL}", **kwargs)


def error(message: str, **kwargs) -> None:
    print(f"{Fore.RED}{_format_message(message)}{Style.RESET_ALL}", **kwargs)


def header(message: str) -> None:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{_format_message(message)}{Style.RESET_ALL}")


def step(message: str) -> None:
    print(f"{Fore.WHITE}{_format_message(message)}{Style.RESET_ALL}")


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
                importlib.import_module(module_name)
            except ImportError:
                # Silently skip modules that can't be imported
                pass
