#!/bin/bash

# Script to unpin researcher_current tag and resume auto-updates
# Usage: ./scripts/unpin-researcher-current.sh

set -e

# Check if currently pinned
if [ ! -f ".researcher_pin" ]; then
    echo "researcher_current tag is not currently pinned."
    exit 0
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    echo "Error: Not on a branch. Please checkout a branch first."
    exit 1
fi

# Show current pin info
PINNED_COMMIT=$(cat .researcher_pin)
echo "Currently pinned to commit: $PINNED_COMMIT"

# Confirm unpinning
read -p "Do you want to unpin and resume auto-updates? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Remove pin file
rm .researcher_pin

# Update tag to current HEAD
CURRENT_COMMIT=$(git rev-parse HEAD)
git tag -f researcher_current "$CURRENT_COMMIT"

# Commit the removal
git add .researcher_pin || true  # In case file is already staged for deletion
git commit -m "Unpin researcher_current tag - resume auto-updates

This commit removes the pin file, allowing the researcher_current tag
to automatically update with new commits again.

Tag updated to current commit: $CURRENT_COMMIT"

# Push changes
echo "Pushing changes and updated tag..."
git push origin "$CURRENT_BRANCH"
git push origin researcher_current --force

echo "✅ Successfully unpinned researcher_current tag"
echo "The tag will now auto-update with new commits to the main branch"
