#!/usr/bin/env python3


from sky.cli import cli

from devops.skypilot.utils.terminal import disable_canonical_mode


def main() -> None:
    """Run sky api login with canonical mode disabled to prevent token truncation."""
    with disable_canonical_mode():
        cli()


if __name__ == "__main__":
    main()
