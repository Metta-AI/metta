# Researcher Tag System

This directory contains scripts for managing the researcher tag system, which provides stable versions of the codebase for research use.

## Overview

The system uses two mutually exclusive tags:
- **`researcher_current`**: Auto-updating tag that follows main branch (normal state)
- **`researcher_current_lock`**: Locked tag pointing to a stable commit (locked state)

### Key Design Principle
When the system is locked, `researcher_current` is removed entirely and replaced by `researcher_current_lock`. This prevents accidental use of an auto-updating tag when stability is required.

## How It Works

### Normal State (Auto-updating)
- Only `researcher_current` exists
- Automatically updates with each push to main
- Researchers use: `git checkout researcher_current`

### Locked State
- Only `researcher_current_lock` exists
- Points to a specific stable commit
- Auto-updates are disabled
- Researchers use: `git checkout researcher_current_lock`

### State Transitions
1. **Lock**: Creates `researcher_current_lock` and removes `researcher_current`
2. **Unlock**: Creates `researcher_current` and removes `researcher_current_lock`

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

