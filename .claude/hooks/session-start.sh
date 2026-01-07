#!/bin/bash
branch=$(git branch --show-current 2>/dev/null)
if [ -n "$branch" ]; then
    echo "Branch: $branch"
    echo "Recent commits:"
    git log --oneline -3 2>/dev/null
    uncommitted=$(git status --short 2>/dev/null | wc -l | tr -d ' ')
    if [ "$uncommitted" -gt 0 ]; then
        echo "Uncommitted changes: $uncommitted files"
    fi
fi
exit 0
