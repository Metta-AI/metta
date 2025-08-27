#!/bin/bash

# Script to lock the researcher tag system by creating researcher_current_lock
# and removing researcher_current to prevent accidental use
# Usage: ./scripts/pin-researcher-current.sh [commit-hash]
# If no commit hash provided, locks to current HEAD

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the commit to lock to
if [ -z "$1" ]; then
    # No commit specified - lock to current HEAD
    COMMIT_HASH=$(git rev-parse HEAD)
    echo -e "${YELLOW}No commit specified, locking to current HEAD: ${COMMIT_HASH:0:8}${NC}"

    # Check current branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [ -n "$CURRENT_BRANCH" ]; then
        echo -e "   Current branch: $CURRENT_BRANCH"
    fi

    # Warn if researcher_current exists and points elsewhere
    if git rev-parse --verify researcher_current >/dev/null 2>&1; then
        CURRENT_TAG=$(git rev-parse researcher_current)
        if [ "$CURRENT_TAG" != "$COMMIT_HASH" ]; then
            echo -e "${YELLOW}Note: researcher_current currently points to ${CURRENT_TAG:0:8}${NC}"
            echo -e "      but locking to your HEAD position ${COMMIT_HASH:0:8}"
        fi
    fi
else
    COMMIT_HASH="$1"
    # Verify commit exists
    if ! git cat-file -e "$COMMIT_HASH^{commit}" 2>/dev/null; then
        echo -e "${RED}Error: Commit $COMMIT_HASH does not exist${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Locking to specified commit: ${COMMIT_HASH:0:8}${NC}"
fi

# Check if already locked
if git ls-remote --tags origin | grep -q "refs/tags/researcher_current_lock"; then
    CURRENT_LOCK=$(git ls-remote --tags origin | grep "refs/tags/researcher_current_lock" | cut -f1)
    echo -e "${YELLOW}Warning: System is already locked at commit: ${CURRENT_LOCK:0:8}${NC}"
    read -p "Do you want to move the lock to ${COMMIT_HASH:0:8}? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Create lock tag
echo -e "${YELLOW}Creating lock tag at ${COMMIT_HASH:0:8}...${NC}"
git tag -f researcher_current_lock "$COMMIT_HASH" -m "Locked at $(date)"
git push origin researcher_current_lock --force

# Move researcher_current tag to the lock position
echo -e "${YELLOW}Moving researcher_current tag to lock position...${NC}"
git tag -f researcher_current "$COMMIT_HASH" -m "Pinned to lock at $(date)"
git push origin researcher_current --force

echo -e "${GREEN}✅ Successfully locked the researcher tag system${NC}"
echo -e "${GREEN}   Lock tag points to: ${COMMIT_HASH:0:8}${NC}"
echo -e "${GREEN}   researcher_current tag has been moved to match${NC}"
echo -e "${GREEN}   Researchers can use either tag to check out the locked version.${NC}"
echo -e "${GREEN}   To unlock: ./scripts/unpin-researcher-current.sh${NC}"
