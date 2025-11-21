#!/usr/bin/env python3

import sys
import termios

from sky.cli import cli


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "api" and sys.argv[2] == "login":
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except (OSError, termios.error):
            cli()
            return

        try:
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~termios.ICANON
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
            cli()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        cli()


if __name__ == "__main__":
    main()
