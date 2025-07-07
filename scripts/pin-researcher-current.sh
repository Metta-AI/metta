#!/bin/bash

# Script to pin researcher_current tag to a specific commit
# Usage: ./scripts/pin-researcher-current.sh [commit-hash]
# If no commit hash provided, pins to current HEAD

set -e

# Get the commit to pin to
if [ -z "$1" ]; then
    COMMIT_HASH=$(git rev-parse HEAD)
    echo "No commit specified, pinning to current HEAD: $COMMIT_HASH"
else
    COMMIT_HASH="$1"
    # Verify commit exists
    if ! git cat-file -e "$COMMIT_HASH^{commit}" 2>/dev/null; then
        echo "Error: Commit $COMMIT_HASH does not exist"
        exit 1
    fi
    echo "Pinning to specified commit: $COMMIT_HASH"
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    echo "Error: Not on a branch. Please checkout a branch first."
    exit 1
fi

# Check if already pinned
if [ -f ".researcher_pin" ]; then
    CURRENT_PIN=$(cat .researcher_pin)
    echo "Warning: researcher_current is already pinned to commit: $CURRENT_PIN"
    read -p "Do you want to change the pin to $COMMIT_HASH? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Create pin file
echo "$COMMIT_HASH" > .researcher_pin

# Update the tag to the pinned commit
git tag -f researcher_current "$COMMIT_HASH"

# Commit the pin file
git add .researcher_pin
git commit -m "Pin researcher_current tag to commit $COMMIT_HASH

This commit creates a pin file that prevents the researcher_current tag
from being automatically updated until unpinned."

# Push changes
echo "Pushing pin file and updated tag..."
git push origin "$CURRENT_BRANCH"
git push origin researcher_current --force

echo "✅ Successfully pinned researcher_current tag to commit $COMMIT_HASH"
echo "The tag will not auto-update until you run unpin-researcher-current.sh"
