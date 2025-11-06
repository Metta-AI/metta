#!/usr/bin/env -S uv run
"""Wrapper script to launch a sandbox from the skypilot directory."""

import devops.skypilot.recipes.sandbox

if __name__ == "__main__":
    devops.skypilot.recipes.sandbox.main()
