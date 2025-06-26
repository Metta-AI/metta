"""
Common utilities for CLI scripts.
"""

import subprocess
import sys

from metta.common.util.colorama import yellow


def get_user_confirmation(prompt: str = "Should we proceed?") -> bool:
    """Get user confirmation before proceeding with an action."""

    response = input(f"{prompt} (Y/n): ").strip().lower()
    if response not in ["", "y", "yes"]:
        print(yellow("Action cancelled by user."))
        return False

    return True


def sh(cmd: list[str], **kwargs) -> str:
    """Run a command and return its stdout (raises if the command fails)."""
    return subprocess.check_output(cmd, text=True, **kwargs).strip()


def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)
