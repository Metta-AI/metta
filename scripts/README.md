# Researcher Tag System

This directory contains scripts for managing the researcher tag system, which provides stable versions of the codebase for research use.

## Overview

The system uses two tags:
- **`researcher_current`**: In normal state, auto-updates with main branch. In locked state, pinned to the same commit as `researcher_current_lock`.
- **`researcher_current_lock`**: Only exists in locked state, marks the stable commit.

### Key Design Principle
When the system is locked, both tags exist and point to the same commit. The `researcher_current` tag is moved to match `researcher_current_lock`, preventing auto-updates while maintaining clarity about which version researchers should use.

## How It Works

### Normal State (Auto-updating)
- Only `researcher_current` exists
- Automatically updates with each push to main
- Researchers use: `git checkout researcher_current`

### Locked State
- Both `researcher_current_lock` and `researcher_current` exist
- Both tags point to the same stable commit
- Auto-updates are disabled
- Researchers can use either: `git checkout researcher_current_lock` or `git checkout researcher_current`

### State Transitions
1. **Lock**: Creates `researcher_current_lock` and moves `researcher_current` to the same commit
2. **Unlock**: Removes `researcher_current_lock` and optionally moves `researcher_current` to a new position

## Scripts

### `researcher-current-status.sh`
Shows the current state of the tag system.
```bash
./scripts/researcher-current-status.sh
```

### `pin-researcher-current.sh`
Locks the system to a specific commit.
```bash
# Lock to current HEAD (wherever you are now)
./scripts/pin-researcher-current.sh

# Lock to a specific commit
./scripts/pin-researcher-current.sh <commit-hash>
```

### `unpin-researcher-current.sh`
Unlocks the system and restores auto-updates.
```bash
./scripts/unpin-researcher-current.sh
# Then choose where to create researcher_current:
# 1) At the lock position
# 2) At latest main (default)
# 3) At a specific commit
```

## Common Use Cases

### For Researchers
```bash
# Check system status
./scripts/researcher-current-status.sh

# Use the appropriate tag based on status
git checkout researcher_current      # If system is auto-updating
git checkout researcher_current_lock  # If system is locked

# Always update your local tags first
git fetch --tags
```

### For Maintainers

#### When a bug is found in main:
```bash
# 1. Lock to last known good commit
./scripts/pin-researcher-current.sh <good-commit>

# 2. Fix the bug in main...

# 3. Unlock when fixed
./scripts/unpin-researcher-current.sh
```

#### Before critical experiments:
```bash
# Lock to current version
./scripts/pin-researcher-current.sh

# Run experiments...

# Unlock when done
./scripts/unpin-researcher-current.sh
```

## GitHub Actions Integration

The `.github/workflows/auto-tag-researcher-current.yml` workflow:
- Runs on every push to main
- Checks if `researcher_current_lock` exists
- Only creates/updates `researcher_current` if system is not locked
- Ensures the two tags are mutually exclusive

