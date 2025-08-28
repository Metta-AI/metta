"""Simple test module for demonstrating basic functionality."""


def get_greeting() -> str:
    """Return a greeting message.

    Returns:
        A simple greeting string.
    """
    return "hello world"


def main() -> None:
    """Main entry point for the script."""
    print(get_greeting())


if __name__ == "__main__":
    main()
