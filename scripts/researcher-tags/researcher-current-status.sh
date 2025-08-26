#!/bin/bash

# Script to show the status of researcher tag system
# Usage: ./scripts/researcher-current-status.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== Researcher Tag System Status ===${NC}"
echo

# Fetch latest tag info from remote
git fetch origin --tags >/dev/null 2>&1

# Check system state
HAS_CURRENT=$(git ls-remote --tags origin | grep -q "refs/tags/researcher_current$" && echo "yes" || echo "no")
HAS_LOCK=$(git ls-remote --tags origin | grep -q "refs/tags/researcher_current_lock$" && echo "yes" || echo "no")

if [ "$HAS_LOCK" = "yes" ]; then
    # System is locked
    LOCK_COMMIT=$(git ls-remote --tags origin | grep "refs/tags/researcher_current_lock$" | cut -f1)
    LOCK_COMMIT_SHORT=$(echo "$LOCK_COMMIT" | cut -c1-8)

    # Get commit info
    git fetch origin "$LOCK_COMMIT" >/dev/null 2>&1
    LOCK_MESSAGE=$(git log -1 --pretty=format:"%s" "$LOCK_COMMIT")
    LOCK_DATE=$(git log -1 --pretty=format:"%cr" "$LOCK_COMMIT")

    echo -e "${YELLOW}🔒 Status: LOCKED${NC}"
    echo -e "${BLUE}📍 researcher_current_lock points to: ${LOCK_COMMIT_SHORT}${NC}"
    echo -e "   Message: ${LOCK_MESSAGE}"
    echo -e "   Date: ${LOCK_DATE}"
    echo

    if [ "$HAS_CURRENT" = "yes" ]; then
        # Both tags exist - check if they point to the same commit
        CURRENT_COMMIT=$(git ls-remote --tags origin | grep "refs/tags/researcher_current$" | cut -f1)
        if [ "$CURRENT_COMMIT" = "$LOCK_COMMIT" ]; then
            echo -e "${GREEN}✅ researcher_current is pinned to the same commit${NC}"
        else
            echo -e "${RED}⚠️  WARNING: researcher_current and researcher_current_lock point to different commits!${NC}"
            echo -e "${RED}   researcher_current: ${CURRENT_COMMIT:0:8}${NC}"
            echo -e "${RED}   This shouldn't happen - the system may be in an inconsistent state${NC}"
        fi
    else
        echo -e "${RED}⚠️  WARNING: researcher_current tag is missing!${NC}"
        echo -e "${RED}   This shouldn't happen - the tag should be pinned to the lock position${NC}"
    fi

    # Show commits since lock
    git fetch origin main >/dev/null 2>&1
    COMMITS_SINCE=$(git rev-list "$LOCK_COMMIT"..origin/main --count 2>/dev/null || echo "unknown")
    if [ "$COMMITS_SINCE" = "0" ]; then
        echo -e "${GREEN}📊 No new commits to main since lock${NC}"
    else
        echo -e "${YELLOW}📊 $COMMITS_SINCE commits have been made to main since lock${NC}"

        # Show recent commits
        if [ "$COMMITS_SINCE" != "unknown" ] && [ "$COMMITS_SINCE" -gt 0 ]; then
            echo -e "\n${CYAN}Recent commits to main:${NC}"
            git log "$LOCK_COMMIT"..origin/main --oneline --max-count=5 | sed 's/^/  /'
            if [ "$COMMITS_SINCE" -gt 5 ]; then
                echo "  ... and $((COMMITS_SINCE - 5)) more"
            fi
        fi
    fi

    echo
    echo -e "${YELLOW}ℹ️  To use the locked version: git checkout researcher_current_lock${NC}"
    echo -e "${YELLOW}   (or git checkout researcher_current - both point to the same commit)${NC}"

elif [ "$HAS_CURRENT" = "yes" ]; then
    # System is not locked, normal operation
    TAG_COMMIT=$(git rev-parse researcher_current 2>/dev/null)
    TAG_COMMIT_SHORT=$(git rev-parse --short researcher_current 2>/dev/null)
    TAG_MESSAGE=$(git log -1 --pretty=format:"%s" researcher_current)
    TAG_DATE=$(git log -1 --pretty=format:"%cr" researcher_current)

    echo -e "${GREEN}🔄 Status: AUTO-UPDATING${NC}"
    echo -e "${BLUE}📍 researcher_current points to: ${TAG_COMMIT_SHORT}${NC}"
    echo -e "   Message: ${TAG_MESSAGE}"
    echo -e "   Date: ${TAG_DATE}"
    echo

    # Check if local and remote tags match
    REMOTE_TAG=$(git ls-remote --tags origin | grep "refs/tags/researcher_current$" | cut -f1)
    if [ "$TAG_COMMIT" = "$REMOTE_TAG" ]; then
        echo -e "${GREEN}✅ Local and remote tags are in sync${NC}"
    else
        echo -e "${YELLOW}⚠️  Local and remote tags differ - run 'git fetch --tags'${NC}"
    fi

    # Show if tag is up to date with main
    git fetch origin main >/dev/null 2>&1
    HEAD_COMMIT=$(git rev-parse origin/main)
    if [ "$TAG_COMMIT" = "$HEAD_COMMIT" ]; then
        echo -e "${GREEN}✅ Tag is up to date with main branch${NC}"
    else
        COMMITS_BEHIND=$(git rev-list researcher_current..origin/main --count 2>/dev/null || echo "unknown")
        echo -e "${YELLOW}📊 Tag is $COMMITS_BEHIND commits behind main${NC}"
        echo "   (Tag will update automatically on next push to main)"
    fi

else
    # Neither tag exists
    echo -e "${RED}❌ No researcher tags exist${NC}"
    echo "The researcher_current tag will be created automatically on the next push to main"
    echo "Or you can manually create it with: git tag researcher_current <commit>"
fi

echo
echo -e "${CYAN}=== Quick Commands ===${NC}"
if [ "$HAS_LOCK" = "yes" ]; then
    echo "Unlock system:                ./scripts/unpin-researcher-current.sh"
    echo "Move lock to another commit:  ./scripts/pin-researcher-current.sh <commit-hash>"
    echo "Use locked version:           git checkout researcher_current_lock"
else
    echo "Lock to current position:     ./scripts/pin-researcher-current.sh"
    echo "Lock to specific commit:      ./scripts/pin-researcher-current.sh <commit-hash>"
    if [ "$HAS_CURRENT" = "yes" ]; then
        echo "Use stable version:           git checkout researcher_current"
    fi
fi
echo "Update local tags:            git fetch --tags"
