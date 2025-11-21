#!/usr/bin/env python3

import sys

from sky.cli import cli

from devops.skypilot.utils.terminal import disable_canonical_mode


def main() -> None:
    """Wrapper for sky CLI that disables canonical mode for login commands to prevent token truncation."""
    is_login = len(sys.argv) >= 3 and sys.argv[1] == "api" and sys.argv[2] == "login"

    if is_login:
        with disable_canonical_mode():
            cli()
    else:
        cli()


if __name__ == "__main__":
    main()
