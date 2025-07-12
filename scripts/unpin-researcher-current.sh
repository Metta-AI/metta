#!/bin/bash

# Script to unlock the researcher tag system by removing researcher_current_lock
# and recreating researcher_current for auto-updates
# Usage: ./scripts/unpin-researcher-current.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if currently locked
if ! git ls-remote --tags origin | grep -q "refs/tags/researcher_current_lock"; then
    echo -e "${YELLOW}System is not currently locked.${NC}"

    # Check if researcher_current exists
    if git ls-remote --tags origin | grep -q "refs/tags/researcher_current$"; then
        echo -e "${GREEN}researcher_current tag exists and is auto-updating normally.${NC}"
    else
        echo -e "${RED}Warning: Neither researcher_current nor researcher_current_lock exist!${NC}"
        echo "The auto-update workflow will create researcher_current on the next push to main."
    fi
    exit 0
fi

# Get lock info
LOCK_COMMIT=$(git ls-remote --tags origin | grep "refs/tags/researcher_current_lock" | cut -f1)
echo -e "${YELLOW}Currently locked at commit: ${LOCK_COMMIT:0:8}${NC}"

# Confirm unlocking
read -p "Do you want to unlock and restore auto-updates? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Determine where to create researcher_current
echo
echo "Where should researcher_current be created?"
echo "1) At the lock position (${LOCK_COMMIT:0:8})"
echo "2) At latest main"
echo "3) At a specific commit"
read -p "Choose [1-3] (default: 2): " -n 1 -r choice
echo

case "$choice" in
    1)
        TARGET_COMMIT="$LOCK_COMMIT"
        echo -e "${YELLOW}Will create researcher_current at lock position${NC}"
        ;;
    3)
        read -p "Enter commit hash: " custom_commit
        if ! git cat-file -e "$custom_commit^{commit}" 2>/dev/null; then
            echo -e "${RED}Error: Commit $custom_commit does not exist${NC}"
            exit 1
        fi
        TARGET_COMMIT="$custom_commit"
        echo -e "${YELLOW}Will create researcher_current at ${TARGET_COMMIT:0:8}${NC}"
        ;;
    *)
        # Default to latest main
        git fetch origin main >/dev/null 2>&1
        TARGET_COMMIT=$(git rev-parse origin/main)
        echo -e "${YELLOW}Will create researcher_current at latest main (${TARGET_COMMIT:0:8})${NC}"
        ;;
esac

# Create researcher_current tag
echo -e "${YELLOW}Creating researcher_current tag at ${TARGET_COMMIT:0:8}...${NC}"
git tag -f researcher_current "$TARGET_COMMIT"
git push origin researcher_current --force

# Remove lock tag
echo -e "${YELLOW}Removing lock tag...${NC}"
git push origin :refs/tags/researcher_current_lock

# Also remove local lock tag if it exists
if git tag -l | grep -q "^researcher_current_lock$"; then
    git tag -d researcher_current_lock
fi

echo -e "${GREEN}✅ Successfully unlocked the researcher tag system${NC}"
echo -e "${GREEN}   researcher_current now points to: ${TARGET_COMMIT:0:8}${NC}"
echo -e "${GREEN}   Auto-updates are now enabled${NC}"
echo -e "${GREEN}   The tag will update automatically with new pushes to main${NC}"
