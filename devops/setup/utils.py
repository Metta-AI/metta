import textwrap

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
