# Researcher Current Tag Management

This directory contains scripts for managing the `researcher_current` tag, which provides a stable baseline for research.

## How It Works

- **Auto-updating**: By default, `researcher_current` tag automatically follows new commits to `main`
- **Pinning**: When pinned, the tag stays fixed until manually unpinned. Pin to a good commit if a bug is found and unpin after it is fixed.

## Scripts

### `researcher-current-status.sh`
Shows the current status of the researcher_current tag.

```bash
./scripts/researcher-current-status.sh
```

### `pin-researcher-current.sh`
Pin the tag to a specific commit to prevent auto-updates.

```bash
# Pin to current HEAD
./scripts/pin-researcher-current.sh

# Pin to specific commit
./scripts/pin-researcher-current.sh abc123def456
```

### `unpin-researcher-current.sh`
Unpin the tag and resume auto-updates.

```bash
./scripts/unpin-researcher-current.sh
```

## Usage for Researchers

### Safe Research Baseline
```bash
# Use the stable version for research
git checkout researcher_current
```

### Help Find Bugs
```bash
# Work with latest code to find issues
git checkout main

# If you find a bug, you can always go back to stable
git checkout researcher_current
```

### Pin When You Need Stability
```bash
# Pin the current stable version before a long experiment
./scripts/pin-researcher-current.sh

# Your 2-week experiment...

# Unpin when ready for updates
./scripts/unpin-researcher-current.sh
```

## System Components

- **GitHub Action** (`.github/workflows/auto-tag-researcher-current.yml`): Automatically updates the tag on pushes to main
- **Pin file** (`.researcher_pin`): When present, prevents auto-updates
- **Scripts**: Manual pin/unpin/status operations

