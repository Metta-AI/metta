#!/usr/bin/env -S uv run python3
"""Common initialization for SkyPilot jobs.

This script outputs shell commands to be evaluated by the calling shell,
allowing environment changes to persist.
"""

import os


def main():
    """Generate shell commands for job initialization."""
    commands = []

    # Change to workspace directory
    commands.append("cd /workspace/metta")

    # Handle virtual environment activation
    # Note that the docker image may start with its own venv - switch to metta venv
    if os.environ.get("VIRTUAL_ENV"):
        commands.append("deactivate 2>/dev/null || true")
    commands.append(". .venv/bin/activate")

    # Configure environment based on job type
    sandbox_mode = os.environ.get("SANDBOX_MODE", "false").lower() == "true"
    if sandbox_mode:
        commands.append("./devops/skypilot/utils/configure_environment.py --sandbox || exit 1")
    else:
        commands.append("./devops/skypilot/utils/configure_environment.py || exit 1")

    # Get and source environment file
    commands.append('METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"')
    commands.append('source "$METTA_ENV_FILE"')

    # Output all commands
    print("\n".join(commands))


if __name__ == "__main__":
    main()
