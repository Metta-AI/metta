#!/bin/bash

# Script to show the status of researcher_current tag
# Usage: ./scripts/researcher-current-status.sh

set -e

echo "=== Researcher Current Tag Status ==="
echo

# Check if tag exists
if ! git rev-parse --verify researcher_current >/dev/null 2>&1; then
    echo "❌ researcher_current tag does not exist yet"
    echo "The tag will be created automatically on the next push to main"
    exit 0
fi

# Get tag commit info
TAG_COMMIT=$(git rev-parse researcher_current)
TAG_COMMIT_SHORT=$(git rev-parse --short researcher_current)
TAG_MESSAGE=$(git log -1 --pretty=format:"%s" researcher_current)
TAG_DATE=$(git log -1 --pretty=format:"%cr" researcher_current)

echo "📍 researcher_current points to: $TAG_COMMIT_SHORT"
echo "   Message: $TAG_MESSAGE"
echo "   Date: $TAG_DATE"
echo

# Check if pinned
if [ -f ".researcher_pin" ]; then
    PINNED_COMMIT=$(cat .researcher_pin)
    echo "🔒 Status: PINNED"
    echo "   Pin file contains: $PINNED_COMMIT"

    # Check if pin file matches tag
    if [ "$PINNED_COMMIT" = "$TAG_COMMIT" ]; then
        echo "   ✅ Pin file matches tag"
    else
        echo "   ⚠️  Pin file doesn't match tag - this shouldn't happen"
    fi

    # Show commits since pin
    COMMITS_SINCE=$(git rev-list researcher_current..HEAD --count 2>/dev/null || echo "unknown")
    if [ "$COMMITS_SINCE" = "0" ]; then
        echo "   📊 No new commits since pin"
    else
        echo "   📊 $COMMITS_SINCE commits have been made since pin"
    fi

else
    echo "🔄 Status: AUTO-UPDATING"

    # Show if tag is up to date with HEAD
    HEAD_COMMIT=$(git rev-parse HEAD)
    if [ "$TAG_COMMIT" = "$HEAD_COMMIT" ]; then
        echo "   ✅ Tag is up to date with HEAD"
    else
        COMMITS_BEHIND=$(git rev-list researcher_current..HEAD --count 2>/dev/null || echo "unknown")
        echo "   📊 Tag is $COMMITS_BEHIND commits behind HEAD"
        echo "   (Tag will update automatically on next push to main)"
    fi
fi

echo
echo "=== Quick Commands ==="
echo "Pin to current commit:    ./scripts/pin-researcher-current.sh"
echo "Pin to specific commit:   ./scripts/pin-researcher-current.sh <commit-hash>"
echo "Unpin and resume auto:    ./scripts/unpin-researcher-current.sh"
echo "Checkout stable version:  git checkout researcher_current"
