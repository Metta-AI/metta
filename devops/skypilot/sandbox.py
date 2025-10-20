#!/usr/bin/env -S uv run
"""Wrapper script to launch a sandbox from the skypilot directory."""

from devops.skypilot.recipes.sandbox import main

if __name__ == "__main__":
    main()
