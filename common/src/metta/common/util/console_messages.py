import textwrap

from metta.common.util.colorama import Fore, blue, bold, colorize, cyan, green, red, yellow


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
