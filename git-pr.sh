#!/bin/bash
set -e

# Ensure not on main branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" = "main" ]; then
  echo "You are on the main branch. Create a feature branch first." >&2
  exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
  git status --short
  read -rp "Uncommitted changes detected. Continue anyway? (y/N) " answer
  if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Aborting." >&2
    exit 1
  fi
fi


# Rebuild packages
echo "Rebuilding Metta..."
uv pip install -e .
echo "Rebuilding MettaGrid..."
uv pip --directory mettagrid install -e .

# Lint and format
ruff format
ruff check

# Run unit tests
uv run pytest
uv run python -m pytest mettagrid/tests

# Submit the PR
gt submit
