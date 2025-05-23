#!/bin/bash
# This script ensures that tests run inside the virtual environment created by `uv`.
# It is executed automatically by Codex before running the test command.

source .venv/bin/activate
